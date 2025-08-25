import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from timm.models.layers import DropPath
import math

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn, drop_path=0.):
        super().__init__()
        self.fn = fn
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, **kwargs):
        return self.drop_path(self.fn(x, **kwargs)) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn, fusion_factor=1):
        super().__init__()
        self.norm = nn.LayerNorm(dim * fusion_factor)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class DSCFeedForward(nn.Module):
    def __init__(self, dim, dropout=0.):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim)
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


class Attention(nn.Module):
    def __init__(self, dim, heads=8, attn_drop=0., proj_drop=0., num_keypoints=None, scale_with_head=False):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5 if scale_with_head else dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop)
        )
        self.num_keypoints = num_keypoints

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale  # (8,12,273,273)
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = mask.mean(dim=0).to(x.device)

            mask = mask.unsqueeze(0).unsqueeze(0).expand(b, h, n, n).bool()
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
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

    def forward(self, x, kv, dist=0):
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
        k = self.to_k(kv)  # 形状: (b, n_kv, dim_q)
        v = self.to_v(kv)  # 形状: (b, n_kv, dim_q)

        q = rearrange(q, 'b n (h d) -> b h n d', h=h)  # 形状: (b, h, n_q, d)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)  # 形状: (b, h, n_kv, d)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)  # 形状: (b, h, n_kv, d)

        dots = (torch.einsum('bhid,bhjd->bhij', q, k) * self.scale) + dist  # 形状: (b, h, n_q, n_kv)

        attn = dots.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # 形状: (b, h, n_q, d)
        out = rearrange(out, 'b h n d -> b n (h d)')  # 形状: (b, n_q, dim_q)
        out = self.to_out(out)  # 最终输出形状: (b, n_q, dim_q)

        return out


class SpatialTransformer(nn.Module):
    def __init__(self, adj, graph, dim, heads, proj_drop, attn_drop=0., num_keypoints=None, scale_with_head=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.num_keypoints = num_keypoints
        self.momentum = 0.1
        for i in range(3):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, attn_drop=attn_drop, proj_drop=proj_drop,
                                                num_keypoints=num_keypoints, scale_with_head=scale_with_head))),
                Residual(PreNorm(dim, DSCFeedForward(dim, dropout=proj_drop))),
                # Residual(PreNorm(dim, Mlp(dim, spatial_hidden_dim)))
            ]))
        # self.gconv = LearnableGraphConv(dim, dim, adj)
        self.graph = graph

        self.conv = nn.Sequential(
            nn.Conv1d(dim * 2, dim, kernel_size=1, padding=0),
            nn.BatchNorm1d(dim, momentum=self.momentum),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(dim * 2, dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim, momentum=self.momentum),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        self.cross_attn = CrossAttention(dim, dim, heads=heads, num_keypoints=num_keypoints, attn_drop=attn_drop,
                                         proj_drop=proj_drop, scale_with_head=scale_with_head)
        self.cross_attn_recover = CrossAttention(dim, dim, heads=heads, num_keypoints=num_keypoints,
                                                 attn_drop=attn_drop, proj_drop=proj_drop,
                                                 scale_with_head=scale_with_head)

        self.distance_embedding = nn.Parameter(torch.zeros((1, 5, dim)))
        self.part_embedding = nn.Parameter(torch.zeros((1, 5, dim)))
        self.pose_tokens = nn.Parameter(torch.zeros(1, num_keypoints, dim))
        self.mlp = nn.Linear(1, heads)

    def pooling(self, x, p, stride=None):
        if p > 1:
            if stride is None:
                x = nn.MaxPool1d(p)(x)  # B x F x V/p
            else:
                x = nn.MaxPool1d(kernel_size=p, stride=stride)(x)  # B x F x V/p
            return x
        else:
            return x

    def forward(self, x, skeleton_tokens, mask=None):
        bf, n, d = x.shape
        # skeleton_tokens = self.gconv(skeleton_tokens)
        x_dscs = []
        # 距离矩阵
        dist_center = np.array([int(dist) for dist in self.graph.dist_center])
        # 距离嵌入
        for i in range(self.num_keypoints):
            dist = dist_center[i]
            x[:, i, :] = x[:, i, :] + self.distance_embedding[:, dist, :]

        for idx, (attn, ff) in enumerate(self.layers):
            if idx == 0:  # 第一次进行所有关节的注意力机制
                x = attn(x, mask=mask)
                x = ff(x)
                x_dscs.append(x)
            elif idx == 1:  # 第二次进行逐部位的注意力机制
                for i in range(len(self.graph.part)):
                    num_node = len(self.graph.part[i])
                    x_i = x[:, self.graph.part[i], :]
                    x_i = self.pooling(x_i.permute(0, 2, 1), (num_node)).permute(0, 2, 1) + self.part_embedding[:, i,
                                                                                            :]  # 加上每一部位的嵌入
                    x_sub1 = torch.cat((x_sub1, x_i), 1) if i > 0 else x_i  # b * f, 5, 32
                x = attn(x_sub1)
                x = ff(x)
                x_dscs.append(x)
            else:  # 第三次对一个总的身体部位进行注意力机制
                x_sub2 = self.pooling(x.permute(0, 2, 1), len(self.graph.part)).permute(0, 2, 1)
                # x = attn(x_sub2)
                x = ff(x_sub2)

        pose_tokens = repeat(self.pose_tokens, '() n d -> bf n d', bf=bf)  # 初始化全零向量保存上采样
        part_center = torch.from_numpy(np.array(self.graph.part_center)).float().to(x.device)
        dist_tokens = self.mlp(part_center.unsqueeze(-1))
        x_up = torch.zeros((bf, n, d)).to(x.device)
        for idx in range(len(x_dscs) - 1, -1, -1):
            if idx == len(x_dscs) - 1:  # 1 —> 5
                x_up_sub = torch.cat((x.repeat(1, len(self.graph.part), 1), x_dscs[idx]), dim=2)
                x_up_sub = self.conv(x_up_sub.permute(0, 2, 1)).permute(0, 2, 1)
            if idx == 0:  # 5 -> 17
                for i in range(len(self.graph.part)):
                    part_tokens = pose_tokens[:, self.graph.part[i], :].clone()

                    part_tokens = self.cross_attn_recover(part_tokens, x_up_sub[:, i, :].unsqueeze(-2), dist_tokens[i])

                    # pose_tokens[:, self.graph.part[i], :] = part_tokens

                    x_up[:, self.graph.part[i], :] = x_up[:, self.graph.part[i], :] + part_tokens

                x_up = self.conv4(torch.cat((x_up,x_dscs[idx]), dim=2).permute(0, 2, 1)).permute(0, 2, 1)
        pose_toens = x_up
        skeleton_tokens = self.cross_attn(skeleton_tokens, pose_toens)
        return x_up, skeleton_tokens
