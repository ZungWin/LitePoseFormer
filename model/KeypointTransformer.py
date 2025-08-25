import torch
import torch.nn as nn

from einops import rearrange,repeat
from timm.models.layers import DropPath



class Residual(nn.Module):
    def __init__(self, fn, drop_path=0.):
        super().__init__()
        self.fn = fn
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

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


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, attn_drop=0., proj_drop=0., num_keypoints=None, scale_with_head=False):
        super().__init__()
        self.heads = heads
        self.scale = (dim//heads) ** -0.5 if scale_with_head else  dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop)
        )
        self.num_keypoints = num_keypoints

    def forward(self, x, mask = None):

        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale # (8,12,273,273)
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
        self.to_keypoint_token = nn.Identity()
        self.num_keypoints = num_keypoints

        # self.skeleton_tokens = nn.Parameter(torch.rand((1,17,dim_q)))

    def forward(self, x, kv, skeleton_tokens, temporal_position_embedding):
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

        q = torch.cat((self.to_q(x), skeleton_tokens),dim=1)
        # q = self.to_q(torch.cat((x, skeleton_tokens),dim=1))
        k = self.to_k(kv)        # 形状: (b, n_kv, dim_q)
        v = self.to_v(kv)        # 形状: (b, n_kv, dim_q)

        k += temporal_position_embedding  # 确保位置嵌入的形状为 (1, n_kv, dim_q)
        v += temporal_position_embedding

        q = rearrange(q, 'b n (h d) -> b h n d', h=h)  # 形状: (b, h, n_q, d)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)  # 形状: (b, h, n_kv, d)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)  # 形状: (b, h, n_kv, d)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale  # 形状: (b, h, n_q, n_kv)

        attn = dots.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # 形状: (b, h, n_q, d)
        out = rearrange(out, 'b h n d -> b n (h d)')    # 形状: (b, n_q, dim_q)
        out = self.to_out(out)                          # 最终输出形状: (b, n_q, dim_q)
        keypoint_tokens = self.to_keypoint_token(out[:,0:self.num_keypoints])

        return keypoint_tokens


class KeypointTransformer(nn.Module):
    def __init__(self, dim, dim_q, dim_kv, depth, heads, dropout, attn_drop=0., num_keypoints=None, scale_with_head=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.num_keypoints = num_keypoints
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, attn_drop=attn_drop, proj_drop = dropout, num_keypoints=num_keypoints, scale_with_head=scale_with_head))),
                Residual(PreNorm(dim, CrossAttention(dim_q, dim_kv,heads = heads, num_keypoints=num_keypoints, attn_drop=attn_drop, proj_drop = dropout, scale_with_head=scale_with_head))),
                Residual(PreNorm(dim, DSCFeedForward(dim, dropout = dropout)))
            ]))
    def forward(self, keypoint_tokens, pose_tokens, skeleton_tokens, pos=None, mask = None):
        x = keypoint_tokens

        for idx,(attn, cross_attn, ff) in enumerate(self.layers):
            x = attn(x, mask = mask)
            x = cross_attn(x, kv=pose_tokens, skeleton_tokens=skeleton_tokens, temporal_position_embedding=pos)
            x = ff(x)
        return x
