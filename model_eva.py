# Based on codes from EVA
# https://github.com/baaivision/EVA

import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn.parameter import Parameter
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from trans_utils import MyTestPlace,MyAt
from torch import Tensor, Size
from typing import Union, List
import numbers


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


_shape_t = Union[int, List[int], Size]



class LayerNormWithForceFP32(nn.Module):
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: _shape_t
    eps: float
    elementwise_affine: bool

    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super(LayerNormWithForceFP32, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
            self.bias = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(
            input.float(), self.normalized_shape, self.weight.float(), self.bias.float(), self.eps).type_as(input)

    def extra_repr(self) -> Tensor:
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,id=id):
#         super().__init__()
#         self.id=id
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.x1_in_test = MyTestPlace(place = 'fc1')
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.x2_in_test = MyTestPlace(place = 'fc2')
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x, norm_weight=None, norm_bias=None,T=0):
#         x = sum(self.x1_in_test(x))
#         x = self.fc1(x)
#         x = self.act(x)
#         x = sum(self.x2_in_test(x))
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,id = 0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.x1_in_test = MyTestPlace(place = 'fc1')
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.x2_in_test = MyTestPlace(place = 'fc2')
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, norm_weight=None, norm_bias=None,T=0):
        x = sum(self.x1_in_test(x))
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = sum(self.x2_in_test(x))
        x = self.fc2(x)
        x = self.drop(x)
        return x

# class Attention(nn.Module):
#     def __init__(
#             self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
#             proj_drop=0., window_size=None, attn_head_dim=None, use_decoupled_rel_pos_bias=False,id=0):
#         super().__init__()
#         self.id=id
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         if attn_head_dim is not None:
#             head_dim = attn_head_dim
#         all_head_dim = head_dim * self.num_heads
#         self.scale = qk_scale or head_dim ** -0.5

#         self.qkv = nn.Linear(dim, all_head_dim * 3)
#         if qkv_bias:
#             self.qkv.bias = nn.Parameter(torch.zeros(all_head_dim*3))
#             self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
#             self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
#         else:
#             self.q_bias = None
#             self.v_bias = None

#         self.rel_pos_bias = None
#         self.qk_float = True

#         self.window_size = None
#         self.relative_position_bias_table = None

#         if window_size:
#             if use_decoupled_rel_pos_bias:
#                 self.rel_pos_bias = DecoupledRelativePositionBias(window_size=window_size, num_heads=num_heads)
#             else:
#                 self.window_size = window_size
#                 self.num_relative_distance = (2 * window_size[0] - 1) * (
#                             2 * window_size[1] - 1) + 3  # (2*14-1) * (2*14-1) + 3
#                 self.relative_position_bias_table = nn.Parameter(
#                     torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
#                 # cls to token & token 2 cls & cls to cls

#                 # get pair-wise relative position index for each token inside the window
#                 coords_h = torch.arange(window_size[0])
#                 coords_w = torch.arange(window_size[1])
#                 coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
#                 coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
#                 relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
#                 relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
#                 relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
#                 relative_coords[:, :, 1] += window_size[1] - 1
#                 relative_coords[:, :, 0] *= 2 * window_size[1] - 1
#                 relative_position_index = \
#                     torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
#                 relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
#                 relative_position_index[0, 0:] = self.num_relative_distance - 3
#                 relative_position_index[0:, 0] = self.num_relative_distance - 2
#                 relative_position_index[0, 0] = self.num_relative_distance - 1

#                 self.register_buffer("relative_position_index", relative_position_index)
#         self.at1 = MyAt()
#         self.at2 = MyAt()
#         self.softmax = nn.Softmax(dim=-1)
#         self.x_in_test = MyTestPlace('fc_qkv')
#         self.q_out_test = MyTestPlace('q')
#         self.k_out_test = MyTestPlace('k')
#         self.v_out_test = MyTestPlace('v')

#         self.attn_drop = nn.Dropout(attn_drop)
#         self.attn_out_test = MyTestPlace('s')
#         self.proj = nn.Linear(all_head_dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.x_out_test = MyTestPlace('fc_out')

#     def forward(self, x, rel_pos_bias=None, attn_mask=None, norm_weight=None, norm_bias=None,T=0):
#         B, N, C = x.shape
#         x = sum(self.x_in_test(x))
#         qkv_bias = None
#         if self.q_bias is not None:
#             self.qkv.bias = nn.Parameter(torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias)))
#         qkv = self.qkv(x)
#         # qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
#         qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
#         #3,B,Nh,N,C/Nh
#         q = sum(self.q_out_test(q))
#         k = sum(self.k_out_test(k))
#         v = sum(self.v_out_test(v))
#         q = q * self.scale
#         if self.qk_float:
#             attn = self.at1(q.float(), k.float().transpose(-2, -1))
#         else:
#             attn = self.at1(q, k.transpose(-2, -1))
#         if self.relative_position_bias_table is not None:
#             relative_position_bias = \
#                 self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
#                     self.window_size[0] * self.window_size[1] + 1,
#                     self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
#             relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
#             attn = attn + relative_position_bias.unsqueeze(0).type_as(attn)

#         if self.rel_pos_bias is not None:
#             attn = attn + self.rel_pos_bias().type_as(attn)

#         if rel_pos_bias is not None:
#             attn = attn + rel_pos_bias.type_as(attn)
#         if attn_mask is not None:
#             attn_mask = attn_mask.bool()
#             attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
#         attn = self.softmax(attn)
#         attn = attn.type_as(x)
#         attn = sum(self.attn_out_test(attn))
#         x = self.at2(attn, v)
#         x = x.transpose(1, 2).reshape(B, N, -1)
#         x = sum(self.x_out_test(x))
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
    
class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 attn_drop=0., proj_drop=0., window_size=None, attn_head_dim=None, use_decoupled_rel_pos_bias=False,id=0
                ):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if qkv_bias:
            all_head_dim = head_dim * self.num_heads
            self.qkv.bias = nn.Parameter(torch.zeros(all_head_dim*3))
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        
        self.at1 = MyAt()
        self.at2 = MyAt()
        self.softmax = nn.Softmax(dim=-1)
        self.x_in_test = MyTestPlace(place = 'fc_qkv')
        self.q_out_test = MyTestPlace(place = 'q')
        self.k_out_test = MyTestPlace(place = 'k')
        self.v_out_test = MyTestPlace(place = 'v')
        self.attn_out_test = MyTestPlace(place = 's')
        self.x_out_test = MyTestPlace(place = 'fc_out')

    def forward(self, x, rel_pos_bias=None, attn_mask=None, norm_weight=None, norm_bias=None,T=0):
        
        if self.q_bias is not None:
            self.qkv.bias = nn.Parameter(torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias)))
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape
        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        
        x = sum(self.x_in_test(x))
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        
        q = sum(self.q_out_test(q))
        k = sum(self.k_out_test(k))
        v = sum(self.v_out_test(v))
        
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        
        attn = self.at1(q, k.transpose(-2, -1))
        attn *= self.scale
        
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        
        attn = sum(self.attn_out_test(attn))
        
        x = self.at2(attn, v)
        x = x.transpose(1, 2).reshape(B, N, -1)
        x = sum(self.x_out_test(x))
        
        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# class Block(nn.Module):

#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
#                  window_size=None, attn_head_dim=None, use_decoupled_rel_pos_bias=False,
#                  postnorm=False,id=0):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
#             attn_drop=attn_drop, proj_drop=drop, window_size=window_size,
#             use_decoupled_rel_pos_bias=use_decoupled_rel_pos_bias, attn_head_dim=attn_head_dim,id=id)
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         # print(dim,mlp_hidden_dim,mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,id=id)

#         if init_values is not None and init_values > 0:
#             self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
#             self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
#         else:
#             self.gamma_1, self.gamma_2 = None, None

#         self.postnorm = postnorm

#     def forward(self, x, rel_pos_bias=None, attn_mask=None,T=0):
#         if self.gamma_1 is None:
#             if self.postnorm:
#                 x = x + self.drop_path(
#                     self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
#                 x = x + self.drop_path(self.norm2(self.mlp(x)))
#             else:
#                 x1 = self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask,T=T)
#                 # x1 = self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)
#                 x = x + x1
#                 # x1 = self.drop_path(self.mlp(self.norm2(x), norm_weight=self.norm2.func.weight, norm_bias=self.norm2.func.bias))
#                 x1 = self.drop_path(self.mlp(self.norm2(x),T=T))
#                 x = x + x1
#         else:
#             if self.postnorm:
#                 x = x + self.drop_path(
#                     self.gamma_1 * self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
#                 x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x)))
#             else:
#                 x = x + self.drop_path(
#                     self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
#                 x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
#         return x

class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 drop=0., attn_drop=0.,
                 drop_path=0., init_values=None,
                 window_size=None, attn_head_dim=None, use_decoupled_rel_pos_bias=False,
                 postnorm=False,id=0):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        # Multi-head attention模块
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        #  MLP模块
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x, rel_pos_bias=None, attn_mask=None,T=0):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# class PatchEmbed(nn.Module):
#     """ Image to Patch Embedding
#     """

#     def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
#         self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = num_patches
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

#     def forward(self, x, **kwargs):
#         B, C, H, W = x.shape
#         # FIXME look at relaxing size constraints
#         assert H == self.img_size[0] and W == self.img_size[1], \
#             f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
#         x = self.proj(x).flatten(2).transpose(1, 2)
#         return x
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        # 图片分辨率 h w
        img_size = (img_size, img_size)
        # 卷积核大小
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        # 分别计算w、h方向上的patch的个数
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        # 一张图片的pacth个数
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        # 卷积的步长巧妙地实现图片切分操作,而后与patch大小一致的卷积核完成线性映射
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()      # nn.Identity()恒等函数 f(x)=x

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        # 一维展平
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class RelativePositionBias(nn.Module):

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

        # trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


def _maske_1d_rel_pos_index(seq_len):
    index = torch.arange(seq_len)
    return index.view(1, seq_len) - index.view(seq_len, 1) + seq_len - 1


def _add_cls_to_index_matrix(index, num_tokens, offset):
    index = index.contiguous().view(num_tokens, num_tokens)
    new_index = torch.zeros(size=(num_tokens + 1, num_tokens + 1), dtype=index.dtype)
    new_index[1:, 1:] = index
    new_index[0, 0:] = offset
    new_index[0:, 0] = offset + 1
    new_index[0, 0] = offset + 2
    return new_index


class DecoupledRelativePositionBias(nn.Module):

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] + 2, 2 * window_size[1] + 2)

        num_tokens = window_size[0] * window_size[1]

        self.relative_position_bias_for_high = nn.Parameter(torch.zeros(self.num_relative_distance[0], num_heads))
        self.relative_position_bias_for_width = nn.Parameter(torch.zeros(self.num_relative_distance[1], num_heads))
        # cls to token & token 2 cls & cls to cls

        h_index = _maske_1d_rel_pos_index(window_size[0]).view(
            window_size[0], 1, window_size[0], 1).expand(-1, window_size[1], -1, window_size[1])
        h_index = _add_cls_to_index_matrix(h_index, num_tokens, 2 * window_size[0] - 1)
        self.register_buffer("relative_position_high_index", h_index)

        w_index = _maske_1d_rel_pos_index(window_size[1]).view(
            1, window_size[1], 1, window_size[1]).expand(window_size[0], -1, window_size[0], -1)
        w_index = _add_cls_to_index_matrix(w_index, num_tokens, 2 * window_size[1] - 1)

        self.register_buffer("relative_position_width_index", w_index)

    def forward(self):
        relative_position_bias = \
            F.embedding(input=self.relative_position_high_index, weight=self.relative_position_bias_for_high) + \
            F.embedding(input=self.relative_position_width_index, weight=self.relative_position_bias_for_width)
        return relative_position_bias.permute(2, 0, 1).contiguous()


# class VisionTransformer(nn.Module):
#     """ Vision Transformer with support for patch or hybrid CNN input stage
#     """

#     def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
#                  num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
#                  drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, use_abs_pos_emb=True,
#                  use_rel_pos_bias=False, use_shared_rel_pos_bias=False, use_decoupled_rel_pos_bias=False,
#                  use_mean_pooling=True, init_scale=0.001, use_checkpoint=True, stop_grad_conv1=True):
#         super().__init__()
#         self.num_classes = num_classes
#         self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

#         self.patch_embed = PatchEmbed(
#             img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
#         num_patches = self.patch_embed.num_patches

#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         if use_abs_pos_emb:
#             self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
#         else:
#             self.pos_embed = None
#         self.pos_drop = nn.Dropout(p=drop_rate)

#         if use_shared_rel_pos_bias:
#             self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
#         else:
#             self.rel_pos_bias = None

#         self.use_decoupled_rel_pos_bias = use_decoupled_rel_pos_bias
#         self.use_checkpoint = use_checkpoint
#         self.stop_grad_conv1 = stop_grad_conv1

#         if use_decoupled_rel_pos_bias or use_rel_pos_bias:
#             window_size = self.patch_embed.patch_shape
#         else:
#             window_size = None

#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
#         self.use_rel_pos_bias = use_rel_pos_bias
#         self.blocks = nn.ModuleList([
#             Block(
#                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
#                 init_values=init_values, window_size=window_size, use_decoupled_rel_pos_bias=use_decoupled_rel_pos_bias,id=i)
#             for i in range(depth)])
#         self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
#         self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
#         self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

#         if self.pos_embed is not None:
#             trunc_normal_(self.pos_embed, std=.02)
#         trunc_normal_(self.cls_token, std=.02)
#         # trunc_normal_(self.mask_token, std=.02)
#         if isinstance(self.head, nn.Linear):
#             trunc_normal_(self.head.weight, std=.02)
#         self.apply(self._init_weights)
#         self.fix_init_weight()

#         if isinstance(self.head, nn.Linear):
#             self.head.weight.data.mul_(init_scale)
#             self.head.bias.data.mul_(init_scale)

#     def fix_init_weight(self):
#         def rescale(param, layer_id):
#             param.div_(math.sqrt(2.0 * layer_id))

#         for layer_id, layer in enumerate(self.blocks):
#             rescale(layer.attn.proj.weight.data, layer_id + 1)
#             rescale(layer.mlp.fc2.weight.data, layer_id + 1)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def get_num_layers(self):
#         return len(self.blocks)

#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'pos_embed', 'cls_token'}

#     def get_classifier(self):
#         return self.head

#     def reset_classifier(self, num_classes, global_pool=''):
#         self.num_classes = num_classes
#         self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

#     def forward_features(self, x, return_patch_tokens=False,T = 0):
#         x = self.patch_embed(x)

#         if self.stop_grad_conv1:
#             x = x.detach()

#         batch_size, seq_len, _ = x.size()
#         cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
#         x = torch.cat((cls_tokens, x), dim=1)
#         if self.pos_embed is not None:
#             x = x + self.pos_embed
#         # x = self.pos_drop(x)

#         rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
#         for blk in self.blocks:
#             if self.use_checkpoint:
#                 x = checkpoint.checkpoint(blk, x, rel_pos_bias,None,T)
#             else:
#                 x = blk(x, rel_pos_bias,T=T)
#         # print(x.reshape(-1)[0:50])
#         x = self.norm(x)
#         if self.fc_norm is not None:
#             t = x[:, 1:, :]
#             if return_patch_tokens:
#                 return self.fc_norm(t)
#             else:
#                 return self.fc_norm(t.mean(1))
#         else:
#             if return_patch_tokens:
#                 return x[:, 1:]
#             else:
#                 return x[:, 0]

#     def forward(self, x, return_patch_tokens=False, T=1):
#         x2 = None
#         x2_list = []
#         for i in range(T):
#             x1 = torch.clone(x)
#             x1 = self.forward_features(x1, return_patch_tokens=return_patch_tokens,T = i)
#             x1 = self.head(x1)
#             x2 = x1 if x2 == None else x1 + x2
#             x2_list.append(x2)
#         return x2_list
#         # x = self.forward_features(x, return_patch_tokens=return_patch_tokens)
#         # x = self.head(x)
#         # return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None,use_mean_pooling=True):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        # 类别个数
        self.num_classes = num_classes
        # embed_dim默认tansformer的base 768
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        # 源码distilled是为了其他任务,分类暂时不考虑
        self.num_tokens = 2 if distilled else 1
        # LayerNorm:对每单个batch进行的归一化
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        # act_layer默认tansformer的GELU
        act_layer = act_layer or nn.GELU
        # embed_layer默认是patch embedding,在其他应用中应该会有其他选择
        # patch embedding过程
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_c, embed_dim=embed_dim)
        # patche个数
        num_patches = self.patch_embed.num_patches
        # class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # positional embedding过程
        # self.pos_embedding = PositionEmbs(num_patches, embed_dim, drop_ratio)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
        # depth是Block的个数
        # 不同block层数 drop_ratio的概率不同,越深度越高
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        # blocks搭建
        # self.blocks = nn.ModuleList([
        #     Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #         drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
        #         init_values=init_values, window_size=window_size, use_decoupled_rel_pos_bias=use_decoupled_rel_pos_bias,id=i)
        #     for i in range(depth)])
        
        # self.blocks = nn.Sequential(*[
        #     Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #           drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
        #           norm_layer=norm_layer, act_layer=act_layer)
        #     for i in range(depth)
        # ])
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_ratio, attn_drop=attn_drop_ratio, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        # self.head_dist = None
#         if distilled:
#             self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

#         # Weight init
#         if self.dist_token is not None:
#             nn.init.trunc_normal_(self.dist_token, std=0.02)

#         nn.init.trunc_normal_(self.cls_token, std=0.02)
#         self.apply(_init_vit_weights)
        
    # def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
    #              num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
    #              drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, use_abs_pos_emb=True,
    #              use_rel_pos_bias=False, use_shared_rel_pos_bias=False, use_decoupled_rel_pos_bias=False,
    #              use_mean_pooling=True, init_scale=0.001, use_checkpoint=True, stop_grad_conv1=True):
    #     super().__init__()
    #     self.num_classes = num_classes
    #     self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
    #     self.patch_embed = PatchEmbed(
    #         img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
    #     num_patches = self.patch_embed.num_patches
    #     self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    #     self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
    #     self.pos_drop = nn.Dropout(p=drop_rate)
    #     self.rel_pos_bias = None
    #     self.use_decoupled_rel_pos_bias = use_decoupled_rel_pos_bias
    #     self.use_checkpoint = use_checkpoint
    #     self.stop_grad_conv1 = stop_grad_conv1
    #     window_size = None
    #     dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
    #     self.use_rel_pos_bias = use_rel_pos_bias
    #     self.blocks = nn.ModuleList([
    #         Block(
    #             dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
    #             drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
    #             init_values=init_values, window_size=window_size, use_decoupled_rel_pos_bias=use_decoupled_rel_pos_bias,id=i)
    #         for i in range(depth)])
    #     self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
    #     self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
    #     self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def forward_features(self, x, return_patch_tokens=False,T = 0):
        x = self.patch_embed(x)

        # if self.stop_grad_conv1:
        #     x = x.detach()

        batch_size, seq_len, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        # x = self.pos_drop(x)
        self.rel_pos_bias = None
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            # if self.use_checkpoint:
            x = checkpoint.checkpoint(blk, x, rel_pos_bias,None,T)
            # else:
            #     x = blk(x, rel_pos_bias,T=T)
        # print(x.reshape(-1)[0:50])
        x = self.norm(x)
        if self.fc_norm is not None:
            t = x[:, 1:, :]
            if return_patch_tokens:
                return self.fc_norm(t)
            else:
                return self.fc_norm(t.mean(1))
        else:
            if return_patch_tokens:
                return x[:, 1:]
            else:
                return x[:, 0]

    def forward(self, x, return_patch_tokens=False, T=1):
        x2 = None
        x2_list = []
        for i in range(T):
            x1 = torch.clone(x)
            x1 = self.forward_features(x1, return_patch_tokens=return_patch_tokens,T = i)
            x1 = self.head(x1)
            x2 = x1 if x2 == None else x1 + x2
            x2_list.append(x2)
        return x2_list
    
def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
class MAEAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None, use_decoupled_rel_pos_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=True)

        self.rel_pos_bias = None
        self.qk_float = False

        self.window_size = None
        self.relative_position_bias_table = None

        if window_size:
            if use_decoupled_rel_pos_bias:
                self.rel_pos_bias = DecoupledRelativePositionBias(window_size=window_size, num_heads=num_heads)
            else:
                self.window_size = window_size
                self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
                self.relative_position_bias_table = nn.Parameter(
                    torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
                # cls to token & token 2 cls & cls to cls

                # get pair-wise relative position index for each token inside the window
                coords_h = torch.arange(window_size[0])
                coords_w = torch.arange(window_size[1])
                coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
                coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
                relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
                relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
                relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
                relative_coords[:, :, 1] += window_size[1] - 1
                relative_coords[:, :, 0] *= 2 * window_size[1] - 1
                relative_position_index = \
                    torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
                relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
                relative_position_index[0, 0:] = self.num_relative_distance - 3
                relative_position_index[0:, 0] = self.num_relative_distance - 2
                relative_position_index[0, 0] = self.num_relative_distance - 1

                self.register_buffer("relative_position_index", relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        B, N, C = x.shape
        # qkv_bias = None
        # if self.q_bias is not None:
        #     qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        if self.qk_float:
            attn = (q.float() @ k.float().transpose(-2, -1))
        else:
            attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0).type_as(attn)

        if self.rel_pos_bias is not None:
            attn = attn + self.rel_pos_bias().type_as(attn)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias.type_as(attn)
        if attn_mask is not None:
            attn_mask = attn_mask.bool()
            attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1).type_as(x)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MAEBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None, use_decoupled_rel_pos_bias=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MAEAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size,
            use_decoupled_rel_pos_bias=use_decoupled_rel_pos_bias, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(
                self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(
                self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class MAEVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, use_abs_pos_emb=True,
                 use_rel_pos_bias=False, use_shared_rel_pos_bias=False, use_decoupled_rel_pos_bias=False,
                 use_mean_pooling=True, init_scale=0.001, use_checkpoint=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        self.use_decoupled_rel_pos_bias = use_decoupled_rel_pos_bias
        self.use_checkpoint = use_checkpoint
        if use_decoupled_rel_pos_bias or use_rel_pos_bias:
            window_size = self.patch_embed.patch_shape
        else:
            window_size = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList([
            MAEBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=window_size, use_decoupled_rel_pos_bias=use_decoupled_rel_pos_bias)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.mask_token, std=.02)
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, return_patch_tokens=False):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, rel_pos_bias)
            else:
                x = blk(x, rel_pos_bias)

        x = self.norm(x)
        if self.fc_norm is not None:
            t = x[:, 1:, :]
            if return_patch_tokens:
                return self.fc_norm(t)
            else:
                return self.fc_norm(t.mean(1))
        else:
            if return_patch_tokens:
                return x[:, 1:]
            else:
                return x[:, 0]

    def forward(self, x, return_patch_tokens=False):
        x = self.forward_features(x, return_patch_tokens=return_patch_tokens)
        x = self.head(x)
        return x


@register_model
def eva_g_patch14(pretrained=False, **kwargs):
    model = VisionTransformer(patch_size=14, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=6144 / 1408, qkv_bias=True,
        norm_layer=partial(LayerNormWithForceFP32, eps=1e-6), img_size= 336)
    model.default_cfg = _cfg()
    return model