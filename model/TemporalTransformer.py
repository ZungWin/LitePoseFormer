import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
class LearnableGraphConv(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(LearnableGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.M = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))

        # spatial/temporal local topology
        self.adj = adj

        # simulated spatial/temporal global topology
        self.adj2 = nn.Parameter(torch.ones_like(adj))
        nn.init.constant_(self.adj2, 1e-6)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])

        adj = self.adj.to(input.device) + self.adj2.to(input.device)

        adj = (adj.T + adj) / 2

        E = torch.eye(adj.size(0), dtype=torch.float).to(input.device)

        output = torch.matmul(adj * E, self.M * h0) + torch.matmul(adj * (1 - E), self.M * h1)
        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
class Residual(nn.Module):
    def __init__(self, fn, drop_path=0.):
        super().__init__()
        self.fn = fn
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class Residual_Pool(nn.Module):
    def __init__(self, fn, dropout=0.,i=-1, stride_num=-1):
        super().__init__()
        self.fn = fn
        self.dropout = nn.Dropout(dropout)
        self.stride_num = stride_num
        self.i = i
        self.pooling = nn.MaxPool1d(1, stride_num[i])

    def forward(self, x, **kwargs):
        if self.i != -1:
            if self.stride_num[self.i] != 1:
                res = self.pooling(x.permute(0, 2, 1))
                res = res.permute(0, 2, 1)
                x, x_dsc = self.fn(x, **kwargs)
                return res + self.dropout(x), x_dsc
            else:
                x, x_dsc = self.fn(x, **kwargs)
                return x + self.dropout(x), x_dsc
        else:
            x, x_dsc = self.fn(x, **kwargs)
            return x + self.dropout(x), x_dsc

class PreNorm(nn.Module):
    def __init__(self, dim, fn,fusion_factor=1):
        super().__init__()
        self.norm = nn.LayerNorm(dim*fusion_factor)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
class DSCFeedForward(nn.Module):
    def __init__(self, dim, dropout = 0.):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=dim, out_channels=dim,kernel_size=3, padding=1, groups=dim)
        self.pointwise_conv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.gelu = nn.GELU()
        self.drop_out = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.depthwise_conv(x)
        x = self.gelu(x)
        x = self.drop_out(x)
        x = self.pointwise_conv(x)
        x = self.drop_out(x)
        return x.transpose(1, 2)

class StrideDSCFeedForward(nn.Module):
    def __init__(self, dim, d_ff, frames, number,stride_num=None, dropout = 0.):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=dim, out_channels=dim,kernel_size=3, padding=1, groups=dim)
        self.pointwise_conv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.gelu = nn.GELU()
        self.drop_out = nn.Dropout(dropout)
        self.number = number

        self.pos_embedding_1 = nn.Parameter(torch.randn(1, frames, dim))  # 位置嵌入，长度为 [1,351,512]
        self.pos_embedding_2 = nn.Parameter(torch.randn(1, frames, dim))  # 位置嵌入，长度为 [1,351,512]
        self.pos_embedding_3 = nn.Parameter(torch.randn(1, frames, dim))  # 位置嵌入，长度为 [1,351,512]

        self.pos_embedding ={
            '0':self.pos_embedding_1,
            '1':self.pos_embedding_2,
            '2':self.pos_embedding_3

        }

        self.w_1 = nn.Conv1d(dim, d_ff, kernel_size=1, stride=1)
        self.w_2 = nn.Conv1d(d_ff, dim, kernel_size=3, stride=stride_num[number], padding=1)

    def forward(self, x):

        x += self.pos_embedding[str(self.number)][:,:x.shape[1]]
        x_dw = self.depthwise_conv(x.transpose(1, 2))
        x_dw = self.gelu(x_dw)
        x_dw = self.drop_out(x_dw)
        x_pw = self.pointwise_conv(x_dw)
        x_dsc = self.drop_out(x_pw)

        x = x.permute(0, 2, 1)
        x = self.w_2(self.drop_out(self.gelu(self.w_1(x))))
        x = x.permute(0, 2, 1)

        return x, x_dsc.transpose(1, 2)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, attn_drop=0., proj_drop=0., scale_with_head=False):
        super().__init__()
        self.heads = heads
        self.scale = (dim//heads) ** -0.5 if scale_with_head else  dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop)
        )

    def forward(self, x):

        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale # (8,12,273,273)

        attn = dots.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class CrossAttention(nn.Module):

    def __init__(self, dim_q, dim_kv, heads=8, num_keypoints=17, attn_drop=0., proj_drop=0., scale_with_head=False):

        super().__init__()
        self.heads = heads
        self.scale = (dim_q // heads) ** -0.5 if scale_with_head else dim_q ** -0.5

        self.to_q = nn.Linear(dim_q, dim_q, bias=False)
        self.to_k = nn.Linear(dim_kv, dim_q, bias=False)
        self.to_v = nn.Linear(dim_kv, dim_q, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.to_out = nn.Sequential(
            nn.Linear(dim_q, dim_q),
            nn.Dropout(proj_drop)
        )
        self.num_keypoints = num_keypoints

    def forward(self, x, kv):
        """
        参数：
        - x: 查询张量，来自多头注意力机制的输出，形状为 (batch_size, n_q, dim_q)
        - kv: 键和值张量，来自 Transformer 编码器的输出，形状为 (batch_size, num_frames, dim_kv)
        - temporal_position_embedding: 位置嵌入，形状为 (1, num_frames, dim_q)
        - mask: 可选的掩码张量
        """
        b, n_q, _ = x.shape
        b, n_kv, _ = kv.shape
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(kv)        # 形状: (b, n_kv, dim_q)
        v = self.to_v(kv)        # 形状: (b, n_kv, dim_q)

        q = rearrange(q, 'b n (h d) -> b h n d', h=h)  # 形状: (b, h, n_q, d)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)  # 形状: (b, h, n_kv, d)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)  # 形状: (b, h, n_kv, d)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale  # 形状: (b, h, n_q, n_kv)

        attn = dots.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # 形状: (b, h, n_q, d)
        out = rearrange(out, 'b h n d -> b n (h d)')    # 形状: (b, n_q, dim_q)
        out = self.to_out(out)                          # 最终输出形状: (b, n_q, dim_q)

        return out

class TemporalTransformer(nn.Module):
    def __init__(self, adj_temporal, dim, mlp_dim, heads, frames, num_keypoints=17, dropout=0., attn_drop=0., drop_path_rate=0.2, stride_num=None, scale_with_head=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.pos_embedding = []
        self.deconv = []
        self.momentum = 0.1
        self.frames = frames
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, len(stride_num))]

        for i in range(round(math.log(frames, 3))):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, attn_drop=attn_drop, proj_drop=dropout, scale_with_head=scale_with_head)), dpr[0]),
                Residual(PreNorm(dim, DSCFeedForward(dim, dropout=dropout))),
            ]))

        for i in range(round(math.log(frames, 3))):
            self.pos_embedding.append(nn.Parameter(torch.zeros(1, frames, dim)).cuda())

        for i in range(round(math.log(frames, 3)) -1):
            self.deconv.append(nn.ConvTranspose1d(in_channels=dim,
                                     out_channels=dim,
                                     kernel_size=3,
                                     stride=3,
                                     padding=1,
                                     output_padding=2).cuda())

        self.gcnov = LearnableGraphConv(dim, dim, adj_temporal)
        self.cross_attn = CrossAttention(dim, dim, heads=heads, num_keypoints=num_keypoints, attn_drop=attn_drop, proj_drop=dropout, scale_with_head=scale_with_head)

    def pooling(self, x, p, stride=None):
        if p > 1:
            if stride is None:
                x = nn.MaxPool1d(p)(x)  # B x F x V/p
            else:
                x = nn.MaxPool1d(kernel_size=p, stride=stride)(x)  # B x F x V/p
            return x
        else:
            return x

    def forward(self, x, skeleton_tokens):
        # skeleton_tokens = self.gcnov(skeleton_tokens)
        x_dscs = []
        num_frames = self.frames
        for idx, (attn, ff) in enumerate(self.layers):
            if x.shape[1] != 1:
                x = x + self.pos_embedding[idx][:, :x.shape[1]]
            x = attn(x)
            x = ff(x)
            x_dscs.append(x)
            output_data = []
            frames = num_frames // 3
            for j in range(frames):
                # 按顺序取 3 帧
                output_frames = x[:, j * 3:(j * 3) + 3, :]  # 取出第 j*3 到 (j*3+2) 帧 (bs, 3, feature_dim)
                # 进行融合操作，假设 self.pooling 对特征维度进行操作
                pooled_frame = self.pooling(output_frames.permute(0, 2, 1), 3).permute(0, 2, 1)  # (bs, 1, feature_dim)
                output_data.append(pooled_frame)

            # 拼接所有融合后的帧
            x = torch.cat(output_data, dim=1)  # 将 (bs, 1, feature_dim) 拼接成 (bs, new_frames, feature_dim)
            num_frames = x.shape[1]
        # x_dcs: [(256, 27, 128), (256, 9, 128), (256, 3, 128)]

        for i in range(len(x_dscs) - 1, -1, -1):
            # x.shape: [bs, 1, 128]'
            if i == len(x_dscs) - 1:
                x_up = x.repeat(1, x_dscs[i].size(1), 1)
                x_up_dsc = x_up  - x_dscs[i] 
            else:
                x = x_up_dsc
                x_up = self.deconv[i](x.permute(0, 2, 1)).permute(0, 2, 1)
                x_up_dsc = x_up - x_dscs[i]  

        skeleton_tokens = self.cross_attn(skeleton_tokens, x_up_dsc)

        return x_up_dsc, skeleton_tokens
