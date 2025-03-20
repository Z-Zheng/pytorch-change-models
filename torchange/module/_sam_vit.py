# Modified by Zhuo Zheng
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Tuple, Type, List

import ever as er
import math
from timm.models.layers import DropPath


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT(nn.Module):
    def __init__(
            self,
            img_size: int = 1024,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            out_chans: int = 256,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_abs_pos: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            global_attn_indexes: Tuple[int, ...] = (),
            drop_path_rate=0.,
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                drop_path_ratio=dpr[i],
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )
        self.out_channels = out_chans

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + resample_abs_pos_embed_nhwc(self.pos_embed, x.shape[1:3])

        for blk in self.blocks:
            if self.training:
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(blk), x, use_reentrant=False)
            else:
                x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x


class MLPBlock(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            mlp_dim: int,
            act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            input_size: Optional[Tuple[int, int]] = None,
            drop_path_ratio=0.,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

        #
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.drop_path(self.attn(x))
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(0.)

        self.fused_attn = torch.__version__ >= '2.0.0'
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                    input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        if self.use_rel_pos:
            attn_bias = get_decomposed_rel_pos_bias(q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        if self.fused_attn:
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_bias,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if attn_bias is not None:
                attn = attn + attn_bias
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
        windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def get_decomposed_rel_pos_bias(
        q: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py
    Args:
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        bias (Tensor): attention bias to add to attention map
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn_bias = rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    return attn_bias.reshape(B, q_h * q_w, k_h * k_w)


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
            self,
            kernel_size: Tuple[int, int] = (16, 16),
            stride: Tuple[int, int] = (16, 16),
            padding: Tuple[int, int] = (0, 0),
            in_chans: int = 3,
            embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


def _build_sam_vit(
        encoder_embed_dim,
        encoder_depth,
        encoder_num_heads,
        encoder_global_attn_indexes,
        checkpoint=None,
        image_size=1024,
        drop_path_rate=0.,
):
    prompt_embed_dim = 256
    vit_patch_size = 16

    encoder = ImageEncoderViT(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
        drop_path_rate=drop_path_rate,
    )
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
            state_dict = extract_vit_weight(state_dict)
            encoder.load_state_dict(state_dict, True)
    return encoder


def extract_vit_weight(state_dict):
    return {
        k.replace('image_encoder.', ''): v
        for k, v in state_dict.items() if 'image_encoder' in k
    }


def resample_abs_pos_embed_nhwc(
        posemb,
        new_size: List[int],
        interpolation: str = 'bicubic',
        antialias: bool = True,
):
    if new_size[0] == posemb.shape[-3] and new_size[1] == posemb.shape[-2]:
        return posemb

    orig_dtype = posemb.dtype
    posemb = posemb.float()
    # do the interpolation
    posemb = posemb.reshape(1, posemb.shape[-3], posemb.shape[-2], posemb.shape[-1]).permute(0, 3, 1, 2)
    posemb = F.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
    posemb = posemb.permute(0, 2, 3, 1).to(orig_dtype)

    return posemb


def build_sam_vit_t(checkpoint=None, img_size=1024, drop_path_rate=0., ):
    return _build_sam_vit(
        encoder_embed_dim=192,
        encoder_depth=12,
        encoder_num_heads=3,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=None,
        image_size=img_size,
        drop_path_rate=drop_path_rate,
    )


def build_sam_vit_s(checkpoint=None, img_size=1024, drop_path_rate=0.):
    return _build_sam_vit(
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=None,
        image_size=img_size,
        drop_path_rate=drop_path_rate,
    )


def build_sam_vit_b(checkpoint=None, img_size=1024, drop_path_rate=0., ):
    return _build_sam_vit(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        image_size=img_size,
        drop_path_rate=drop_path_rate,
    )


def build_sam_vit_l(checkpoint=None, img_size=1024, drop_path_rate=0.):
    return _build_sam_vit(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        image_size=img_size,
        drop_path_rate=drop_path_rate,
    )


def build_sam_vit_h(checkpoint=None, img_size=1024, drop_path_rate=0.):
    return _build_sam_vit(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        image_size=img_size,
        drop_path_rate=drop_path_rate,
    )


class SimpleFeaturePyramid(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels=256,
            scale_factors=(4.0, 2.0, 1.0, 0.5),
            norm="LN",
    ):
        super().__init__()
        self.scale_factors = scale_factors
        use_bias = norm == ""
        strides = [int(16 / scale) for scale in scale_factors]
        self.stages = []
        dim = in_channels
        for idx, scale in enumerate(scale_factors):
            out_dim = dim
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    LayerNorm2d(dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            layers.extend(
                [
                    nn.Sequential(
                        nn.Conv2d(out_dim, out_channels, 1, bias=use_bias),
                        LayerNorm2d(out_channels),
                    ),
                    nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, 1, bias=use_bias),
                        LayerNorm2d(out_channels),
                    ),
                ]
            )
            layers = nn.Sequential(*layers)

            stage = int(math.log2(strides[idx]))
            self.add_module(f"simfp_{stage}", layers)
            self.stages.append(layers)

    def forward(self, x):
        results = []

        for stage in self.stages:
            results.append(stage(x))

        return results


@er.registry.MODEL.register()
class SAMEncoder(er.ERModule):
    def __init__(self, config):
        super().__init__(config)
        dpr = self.cfg.drop_path_rate
        if self.config.vit_type == 'vit_b':
            encoder = build_sam_vit_b(checkpoint=self.config.checkpoint, img_size=1024, drop_path_rate=dpr)
        elif self.config.vit_type == 'vit_l':
            encoder = build_sam_vit_l(checkpoint=self.config.checkpoint, img_size=1024, drop_path_rate=dpr)
        elif self.config.vit_type == 'vit_h':
            encoder = build_sam_vit_h(checkpoint=self.config.checkpoint, img_size=1024, drop_path_rate=dpr)
        elif self.config.vit_type == 'vit_s':
            encoder = build_sam_vit_s(checkpoint=None, img_size=1024, drop_path_rate=dpr)
        elif self.config.vit_type == 'vit_t':
            encoder = build_sam_vit_t(checkpoint=None, img_size=1024, drop_path_rate=dpr)
        else:
            raise ValueError(f'Unknown vit_type: {self.config.vit_type}')

        self.vit = encoder
        self.sfp = SimpleFeaturePyramid(encoder.out_channels, self.cfg.out_channels)

        if self.cfg.freeze_vit:
            self.freeze_vit()

        if self.cfg.lora:
            lora_alpha = getattr(self.cfg.lora, 'alpha', self.cfg.lora.r)
            lora_dropout = getattr(self.cfg.lora, 'dropout', 0.)
            lora_mode = getattr(self.cfg.lora, 'mode', 'all')
            lora_merge = getattr(self.cfg.lora, 'merge_weights', False)
            if lora_mode == 'all':
                self.vit = LoraLinear.convert_lora_linear(
                    self.vit,
                    r=self.cfg.lora.r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    merge_weights=lora_merge,
                )
            elif lora_mode == 'attn':
                self.vit = lora_on_attention(
                    self.vit,
                    r=self.cfg.lora.r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    merge_weights=lora_merge,
                )
            else:
                raise ValueError(f'unknown lora.mode: {lora_mode}')

    def forward(self, x):
        assert x.size(2) == x.size(3)

        x = self.vit(x)
        features = self.sfp(x)
        return features

    def set_default_config(self):
        self.config.update(dict(
            checkpoint=None,
            vit_type='vit_b',
            out_channels=256,
            freeze_vit=True,
            lora=None,
            drop_path_rate=0.
        ))

    def freeze_vit(self):
        for name, p in self.vit.named_parameters():
            if 'lora' not in name:
                p.requires_grad_(False)
        for name, m in self.vit.named_modules():
            if 'lora' not in name:
                m.training = False

        if self.cfg.lora:
            self.vit.neck.requires_grad_(True)
            self.vit.neck.train()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.cfg.freeze_vit:
            self.freeze_vit()

    @property
    def out_channels(self):
        return self.vit.out_channels

    def custom_param_groups(self):
        param_groups = [{'params': [], 'weight_decay': 0.}, {'params': []}]
        for i, p in self.named_parameters():
            if 'norm' in i:
                param_groups[0]['params'].append(p)
            else:
                param_groups[1]['params'].append(p)
        return param_groups


class LoRALayer:
    def __init__(
            self,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class LoraLinear(nn.Linear, LoRALayer):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool,
            r: int,
            lora_alpha: int,
            lora_dropout: float = 0.,
            merge_weights: bool = False,
    ):
        nn.Linear.__init__(self, in_features, out_features, bias)
        LoRALayer.__init__(self, r, lora_alpha, lora_dropout, merge_weights)

        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = F.linear(x, self.weight, bias=self.bias)
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, self.weight, bias=self.bias)

    @classmethod
    def convert_lora_linear(cls, module, r, lora_alpha, lora_dropout=0., merge_weights=False):
        module_output = module
        if isinstance(module, nn.Linear):
            module_output = LoraLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                merge_weights=merge_weights,
            )
            with torch.no_grad():
                module_output.weight = module.weight
                if module.bias is not None:
                    module_output.bias = module.bias

            if hasattr(module, "qconfig"):
                module_output.qconfig = module.qconfig
        for name, child in module.named_children():
            module_output.add_module(
                name, cls.convert_lora_linear(child, r, lora_alpha, lora_dropout, merge_weights)
            )
        del module
        return module_output


def lora_on_attention(vit: ImageEncoderViT, **kwargs):
    for block in vit.blocks:
        block.attn = LoraLinear.convert_lora_linear(block.attn, **kwargs)

    return vit
