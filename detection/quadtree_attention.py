import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
import sys

sys.path.append("../QuadTreeAttention")
from QuadtreeAttention.modules.quadtree_attention import QTAttA, QTAttB
from einops.einops import rearrange


class QuadtreeAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        attn_type=None,
        topks=[32, 32, 32, 32],
    ):
        self.attn_type = attn_type
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        # print('attention layer')

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # print('dim ' + str(dim) + ', dim each level ' + str(dims_each_level))

        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=qkv_bias)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=qkv_bias)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=qkv_bias)

        if attn_type == "B":
            self.py_att = QTAttB(num_heads, dim // num_heads, scale=sr_ratio, topks=topks, lepe=True)
            self.value_branchs = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.GroupNorm(1, dim),
                        nn.ReLU(),
                        nn.Conv2d(dim, dim, kernel_size=2, stride=2),
                        nn.GroupNorm(1, dim),
                        nn.ReLU(),
                        nn.Conv2d(dim, dim, kernel_size=1, stride=1),
                    )
                    for i in range(sr_ratio - 1)
                ]
            )
        else:
            self.py_att = QTAttA(num_heads, dim // num_heads, topks=topks)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            # m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            trunc_normal_(m.weight, std=0.02)
            m.init = True
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        B, N, C = x.shape
        y = x
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        target = x
        keys = []
        values = []
        queries = []

        q = self.q_proj(x)
        k = self.k_proj(target)
        v = self.v_proj(target)

        for i in range(self.sr_ratio):
            keys.append(k.float())
            values.append(v.float())
            queries.append(q.float())
            if i != self.sr_ratio - 1:
                k = F.avg_pool2d(k, kernel_size=2, stride=2)

                q = F.avg_pool2d(q, kernel_size=2, stride=2)
                if self.attn_type == "B":
                    v = self.value_branchs[i](v)
                else:
                    v = F.avg_pool2d(v, kernel_size=2, stride=2)

        msg = self.py_att(queries, keys, values).contiguous().view(B, -1, C)

        x = self.proj(msg)
        x = self.proj_drop(x)

        return x
