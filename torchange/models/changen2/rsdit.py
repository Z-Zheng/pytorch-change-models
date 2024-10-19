# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# Modified from:
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
# --------------------------------------------------------
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import PatchEmbed, Attention
import torch.nn.functional as F
from einops import rearrange

__all__ = ['RSDiT_models', 'RSDiT_B_2', 'RSDiT_L_2', 'RSDiT_XL_2', 'RSDiT_S_2']


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


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


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DenseLabelEmbedder(nn.Module):
    def __init__(self, label_channels, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.convs = nn.Sequential(
            nn.Conv2d(label_channels, hidden_size // 8, 3, 1, 1, bias=True),
            LayerNorm2d(hidden_size // 8),
            nn.SiLU(True),
            nn.Conv2d(hidden_size // 8, hidden_size // 8, 3, 1, 1, bias=True),
            LayerNorm2d(hidden_size // 8),
            nn.SiLU(True),
            # down 2x
            nn.Conv2d(hidden_size // 8, hidden_size // 4, 3, 2, 1, bias=True),
            LayerNorm2d(hidden_size // 4),
            nn.SiLU(True),
            nn.Conv2d(hidden_size // 4, hidden_size // 4, 3, 1, 1, bias=True),
            LayerNorm2d(hidden_size // 4),
            nn.SiLU(True),
            # down 2x
            nn.Conv2d(hidden_size // 4, hidden_size // 2, 3, 2, 1, bias=True),
            LayerNorm2d(hidden_size // 2),
            nn.SiLU(True),
            nn.Conv2d(hidden_size // 2, hidden_size // 2, 3, 1, 1, bias=True),
            LayerNorm2d(hidden_size // 2),
            nn.SiLU(True),
            # down 2x
            nn.Conv2d(hidden_size // 2, hidden_size, 3, 2, 1, bias=True),
            LayerNorm2d(hidden_size),
            nn.SiLU(True),
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1, bias=True),
            LayerNorm2d(hidden_size),
        )

    def forward(self, labels):
        return self.convs(labels)


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(
            self,
            img_size=224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer=None,
            bias: bool = True,
    ):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        if img_size is not None:
            self.img_size = (img_size, img_size)
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        x = self.norm(x)
        return x


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class ConvMlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

        self.dwconv = nn.Conv2d(
            self.fc1.out_features,
            self.fc1.out_features, 3, 1, 1, bias=False,
            groups=self.fc1.out_features
        )

    def forward(self, x, h, w):
        x = self.fc1(x)
        x = rearrange(x, 'b (h w) c ->b c h w', h=h, w=w).contiguous()
        x = self.dwconv(x)
        x = rearrange(x, 'b c h w ->b (h w) c').contiguous()
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def window_partition(x: torch.Tensor, window_size: int):
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
        windows: torch.Tensor, window_size: int, pad_hw, hw
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


class RSDiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, window_size: int = 0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = ConvMlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        # self.linear = nn.Linear(256, hidden_size, 1)
        self.window_size = window_size

    def set_h_w(self, h, w):
        self.h = h
        self.w = w

    def forward(self, x, c):
        # x: N T D
        # c: N T D
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        _x = modulate(self.norm1(x), shift_msa, scale_msa)

        if self.window_size > 0:
            # S = int(math.sqrt(x.size(1)))
            _x = _x.reshape(_x.size(0), self.h, self.w, _x.size(2))
            _x, pad_hw = window_partition(_x, self.window_size)
            _x = _x.reshape(_x.size(0), -1, _x.size(3))

        _x = self.attn(_x)
        if self.window_size > 0:
            _x = _x.reshape(_x.size(0), self.window_size, self.window_size, _x.size(2))
            _x = window_unpartition(_x, self.window_size, pad_hw, (self.h, self.w))
            _x = _x.reshape(_x.size(0), -1, _x.size(3))

        x = x + gate_msa * _x
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp), h=self.h, w=self.w)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class RSDiT(nn.Module):
    def __init__(
            self,
            input_size=32,
            patch_size=2,
            in_channels=4,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            class_dropout_prob=0.0,
            label_channels=1,
            learn_sigma=True,
            window_size=8,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = DenseLabelEmbedder(label_channels, hidden_size)

        self.blocks = nn.ModuleList([
            RSDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, window_size=window_size if (i + 1) % 4 != 0 else 0) for i in
            range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        # h = w = int(x.shape[1] ** 0.5)
        assert self.h * self.w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], self.h, self.w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, self.h * p, self.w * p))
        return imgs

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def forward(self, x, t, y):
        """
        Forward pass of RSDiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, H, W) tensor of class labels
        """
        N = x.size(0)
        H = x.size(2)
        W = x.size(3)
        x = self.x_embedder(x)  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t).reshape(N, 1, -1)  # (N, 1, D)
        y = self.y_embedder(y)  # (N, D, H, W)
        y = F.interpolate(y.to(torch.float32), size=(H // self.patch_size, W // self.patch_size), mode='nearest').to(t.dtype)
        h, w = y.shape[-2:]
        self.h = h
        self.w = w
        y = y.flatten(2).transpose(1, 2)  # NDHW -> NLD
        c = t + y  # (N, T, D)
        for block in self.blocks:
            block.set_h_w(h, w)
            if self.training:
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c, use_reentrant=False)  # (N, T, D)
            else:
                x = block(x, c)

        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                                   RSDiT Configs                               #
#################################################################################

def RSDiT_XL_2(**kwargs):
    return RSDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def RSDiT_L_2(**kwargs):
    return RSDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def RSDiT_B_2(**kwargs):
    return RSDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def RSDiT_S_2(**kwargs):
    return RSDiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


RSDiT_models = {
    'RSDiT-B/2': RSDiT_B_2,
    'RSDiT-L/2': RSDiT_L_2,
    'RSDiT-XL/2': RSDiT_XL_2,
    'RSDiT-S/2': RSDiT_S_2,
}
