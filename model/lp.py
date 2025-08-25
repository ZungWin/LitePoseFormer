import math
import os
import sys

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from common.graph_utils import adj_mx_from_skeleton_temporal, adj_mx_from_skeleton
from common.h36m_dataset import Human36mDataset
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange,repeat
from model.block.graph_frames import Graph
from model.block.temporal_transformer_encoder import TemporalTransformerEncoder
from model.block.temporal_transformer_decoder import TemporalTransformerDecoder
from model.block.SpatialTransformer import SpatialTransformer
from easydict import EasyDict as edict

class Model(nn.Module):
    def __init__(self, adj, adj_temporal, args):
        super().__init__()

        spatial_embed_dim = args.spatial_embed_dim
        temporal_embed_dim = args.temporal_embed_dim

        self.num_keypoints = args.num_keypoints

        self.spatial_position_embdding = nn.Parameter(torch.zeros(1, args.num_keypoints, spatial_embed_dim))
        self.temporal_position_embdding = nn.Parameter(torch.zeros(1, args.frames, temporal_embed_dim))

        self.pos_drop = nn.Dropout(p=args.drop_rate)
        self.graph = Graph('hm36_gt', 'spatial')

        self.mask = adj

        depth = args.depth
        heads = args.heads

        spatial_mlp_hidden_dim = int(spatial_embed_dim * args.mlp_ratio)
        temporal_mlp_hidden_dim = int(temporal_embed_dim * args.mlp_ratio)

        self.Spatial_Transformer = SpatialTransformer(adj, self.graph, spatial_embed_dim, heads, args.drop_rate, args.attn_drop, self.num_keypoints,True)
        self.Spatial_embedding = nn.Linear(2, spatial_embed_dim)
        self.Temporal_embedding = nn.Linear(spatial_embed_dim * self.num_keypoints, temporal_embed_dim)

        self.Temporal_Transformer_Encoder_1 = TemporalTransformerEncoder(adj_temporal, temporal_embed_dim, temporal_mlp_hidden_dim, heads, args.frames, args.num_keypoints, args.drop_rate ,args.attn_drop,  args.drop_path_rate, args.stride_num, True)

        self.Temporal_Transformer_Decoder_1 = TemporalTransformerDecoder(temporal_embed_dim,temporal_embed_dim ,temporal_embed_dim, depth, heads, args.drop_rate, args.attn_drop, self.num_keypoints, True)

        self.Spatial_norm = nn.LayerNorm(spatial_embed_dim)
        self.Temporal_norm = nn.LayerNorm(temporal_embed_dim)

        self.keypoint_tokens = nn.Parameter(torch.zeros((1, self.num_keypoints, temporal_embed_dim)))
        self.skeleton_tokens = nn.Parameter(torch.zeros((1, args.frames, self.num_keypoints, spatial_embed_dim)))

        self.head = nn.Linear(temporal_embed_dim, 3)

        if args.train and args.apply_init:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        print("Initialization...")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):

        pose_2d = x

        # keypoint_tokens = self.keypoint_tokens    # 1,17,192
        b, f, p, _ = pose_2d.shape

        pose_tokens = self.Spatial_embedding(rearrange(pose_2d, 'b f p c  -> (b f) p  c').contiguous()) # (b * f), 17 , 2
        skeleton_tokens = rearrange(repeat(self.skeleton_tokens, '() f n d -> b f n d', b=b),'b f n d -> (b f) n d')

        # pose_tokens += self.spatial_position_embdding  # (b * f), 17 , spatial_embed_dim

        pose_tokens, skeleton_tokens = self.Spatial_Transformer(pose_tokens, skeleton_tokens, mask=self.mask) # (b * f), 17 , spatial_embed_dim

        pose_tokens, skeleton_tokens = self.Temporal_embedding(rearrange(self.Spatial_norm(pose_tokens),'(b f) w c -> b f (w c)',f=f)), self.Temporal_embedding(rearrange(skeleton_tokens,'(b f) w c -> b f (w c)',f=f))

        keypoint_tokens = repeat(self.keypoint_tokens, '() n d -> b n d', b = b)

        pose_tokens_encoder_1, skeleton_tokens = self.Temporal_Transformer_Encoder_1(pose_tokens, skeleton_tokens) # bs,num_frames,temporal_embed_dim

        pose_tokens_decoder1 = self.Temporal_norm(self.Temporal_Transformer_Decoder_1(keypoint_tokens, pose_tokens_encoder_1, skeleton_tokens, pos=self.temporal_position_embdding, mask=self.mask))

        pose_3d = self.head(pose_tokens_decoder1)            # B 17 3

        pose_3d = rearrange(pose_3d, 'b j c -> b 1 j c').contiguous() # B, 1, 17, 3

        return pose_3d
