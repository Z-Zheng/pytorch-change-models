# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
FarSeg (Foreground-Aware Relation Network) module for semantic segmentation.

This module implements the FarSeg architecture which uses scene context to refine
multi-scale feature representations through Feature-Scene Relation (FSR) modules.
Multiple encoder backbones are supported including ResNet, Swin Transformer, SAM, and DINOv3.
"""

import ever as er
import ever.module as M

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torchange.module.sam_vit import SAMEncoder, SimpleFeaturePyramid
from torchange.module.tv_swin import TVSwinTransformer


@er.registry.MODEL.register(verbose=False)
class FarSegEncoder(M.ResNetEncoder):
    """
    FarSeg encoder based on ResNet backbone.

    Uses ResNet as feature extractor with FPN, Feature-Scene Relation module,
    and asymmetric decoder for semantic segmentation.
    """

    def __init__(self, config):
        super().__init__(config)
        if self.config.resnet_type in ['resnet18', 'resnet34']:
            max_channels = 512
        else:
            max_channels = 2048
        self.fpn = M.FPN([max_channels // (2 ** (3 - i)) for i in range(4)], 256)
        self.fsr = M.FSRelation(
            max_channels,
            [256 for _ in range(4)],
            256,
            True
        )
        self.dec = M.AssymetricDecoder(
            256,
            self.config.out_channels
        )

    def forward(self, inputs):
        features = super().forward(inputs)
        coarsest_features = features[-1]
        scene_embedding = F.adaptive_avg_pool2d(coarsest_features, 1)
        features = self.fpn(features)
        features = self.fsr(scene_embedding, features)
        features = self.dec(features)

        return features

    def set_default_config(self):
        super().set_default_config()
        self.config.update(dict(
            out_channels=96,
        ))


class FSRelationV3(nn.Module):
    """
    Feature-Scene Relation module (version 3).

    Refines multi-scale features by modeling the relationship between scene-level
    context and spatial features at each scale. Supports both scale-aware and
    scale-agnostic projection modes.

    Args:
        scene_embedding_dim: Dimension of the scene embedding feature
        in_channels_list: List of input channels for each feature scale
        out_channels: Number of output channels for all scales
        scale_aware_proj: If True, uses separate scene encoders for each scale
    """

    def __init__(
            self,
            scene_embedding_dim,
            in_channels_list,
            out_channels,
            scale_aware_proj=False,
    ):
        super().__init__()
        self.scale_aware_proj = scale_aware_proj

        if scale_aware_proj:
            self.scene_encoder = nn.ModuleList(
                [nn.Sequential(
                    nn.Conv2d(scene_embedding_dim, out_channels, 1),
                    M.LayerNorm2d(out_channels),
                    nn.GELU(),
                    nn.Conv2d(out_channels, out_channels, 1),
                    M.LayerNorm2d(out_channels),
                    nn.GELU(),
                ) for _ in range(len(in_channels_list))]
            )
            self.project = nn.ModuleList(
                [nn.Sequential(
                    nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
                    M.LayerNorm2d(out_channels),
                    nn.GELU(),
                    nn.Dropout2d(p=0.1)
                ) for _ in range(len(in_channels_list))]
            )
        else:
            # 2mlp
            self.scene_encoder = nn.Sequential(
                nn.Conv2d(scene_embedding_dim, out_channels, 1),
                M.LayerNorm2d(out_channels),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, 1),
                M.LayerNorm2d(out_channels),
                nn.GELU(),
            )
            self.project = nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
                M.LayerNorm2d(out_channels),
                nn.GELU(),
                nn.Dropout2d(p=0.1)
            )

        self.content_encoders = nn.ModuleList()
        self.feature_reencoders = nn.ModuleList()
        for c in in_channels_list:
            self.content_encoders.append(
                nn.Sequential(
                    nn.Conv2d(c, out_channels, 1),
                    M.LayerNorm2d(out_channels),
                    nn.GELU(),
                )
            )
            self.feature_reencoders.append(
                nn.Sequential(
                    nn.Conv2d(c, out_channels, 1),
                    M.LayerNorm2d(out_channels),
                    nn.GELU(),
                )
            )

        self.normalizer = nn.Sigmoid()

    def forward(self, scene_feature, features: list):
        """
        Forward pass of Feature-Scene Relation module.

        Args:
            scene_feature: Scene-level feature tensor of shape [N, C, 1, 1]
            features: List of multi-scale spatial features, each of shape [N, C_i, H_i, W_i]

        Returns:
            List of refined features at multiple scales with shape [N, out_channels, H_i, W_i]
        """
        # [N, C, H, W]
        content_feats = [c_en(p_feat) for c_en, p_feat in zip(self.content_encoders, features)]
        if self.scale_aware_proj:
            scene_feats = [op(scene_feature) for op in self.scene_encoder]
            relations = [self.normalizer((sf * cf).sum(dim=1, keepdim=True)) for sf, cf in
                         zip(scene_feats, content_feats)]
        else:
            # [N, C, 1, 1]
            scene_feat = self.scene_encoder(scene_feature)
            relations = [self.normalizer((scene_feat * cf).sum(dim=1, keepdim=True)) for cf in content_feats]

        p_feats = [op(p_feat) for op, p_feat in zip(self.feature_reencoders, features)]

        refined_feats = [torch.cat([r * p, o], dim=1) for r, p, o in zip(relations, p_feats, features)]

        if self.scale_aware_proj:
            ffeats = [op(x) for op, x in zip(self.project, refined_feats)]
        else:
            ffeats = [self.project(x) for x in refined_feats]

        return ffeats


class FarSegMixin(nn.Module):
    """
    Reusable FarSeg components including FPN, FSR module, and decoder.

    Can be combined with different encoder backbones to create FarSeg variants.

    Args:
        in_channels: List of input channels from backbone at different scales
        fpn_channels: Number of channels in FPN outputs
        out_channels: Number of output feature channels
    """

    def __init__(self, in_channels, fpn_channels, out_channels):
        super().__init__()
        self.fpn = M.FPN(in_channels, fpn_channels)
        self.fsr = FSRelationV3(
            in_channels[-1],
            [fpn_channels for _ in range(4)],
            fpn_channels,
            scale_aware_proj=True
        )
        self.dec = M.AssymetricDecoder(
            fpn_channels,
            out_channels,
            norm_fn=M.LayerNorm2d
        )

    def forward(self, x, scene_embedding=None):
        """
        Forward pass through FPN, FSR, and decoder.

        Args:
            x: List of multi-scale features from backbone
            scene_embedding: Optional scene-level feature. If None, computed via adaptive pooling

        Returns:
            Refined output features of shape [N, out_channels, H, W]
        """
        if scene_embedding is None:
            scene_embedding = F.adaptive_avg_pool2d(x[-1], 1)
        features = self.fpn(x)
        features = self.fsr(scene_embedding, features)
        features = self.dec(features)
        return features


@er.registry.MODEL.register(verbose=False)
class SwinFarSeg(TVSwinTransformer):
    """
    FarSeg with Swin Transformer backbone.

    Combines Swin Transformer hierarchical features with FarSeg's FPN and FSR modules
    for semantic segmentation.
    """

    # arguments below will be automatically captured
    def __init__(self, name='swin_t', weights=tv.models.Swin_T_Weights, out_channels=256):
        """
        Initialize SwinFarSeg model.

        Args:
            name: Swin Transformer variant ('swin_t', 'swin_s', 'swin_b', etc.)
            weights: Pretrained weights for Swin Transformer
            out_channels: Number of output feature channels
        """

        self.farseg = FarSegMixin(
            in_channels=self.out_channels(),
            fpn_channels=out_channels,
            out_channels=out_channels,
        )

    def forward(self, x):
        features = super().forward(x)
        features = self.farseg(features)
        return features


@er.registry.MODEL.register(verbose=False)
class SAMEncoderFarSeg(SAMEncoder):
    """
    FarSeg with SAM (Segment Anything Model) encoder backbone.

    Integrates SAM's vision transformer encoder with FarSeg components for
    semantic segmentation tasks.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        in_channels = [self.out_channels for _ in range(4)]

        self.farseg = FarSegMixin(
            in_channels=in_channels,
            fpn_channels=self.cfg.fpn_channels,
            out_channels=self.cfg.out_channels,
        )

    def forward(self, x):
        features = super().forward(x)
        features = self.farseg(features)
        return features

    def set_default_config(self):
        super().set_default_config()
        self.config.update(dict(
            fpn_channels=256,
        ))


@er.registry.MODEL.register(verbose=False)
class DINOv3ViTLFarSeg(er.ERModule):
    """
    FarSeg with DINOv3 Vision Transformer (ViT-L) backbone.

    Leverages DINOv3's self-supervised pretrained features with FarSeg architecture.
    Supports LoRA fine-tuning and flexible feature extraction modes.
    """

    # arguments below will be automatically captured
    def __init__(
            self,
            pretrained=None,
            freeze_vit=True,
            drop_path_rate=0.,
            lora=None,
            out_channels=1024,
            dinov3_forward_mode='four_levels',
    ):
        """
        Initialize DINOv3ViTLFarSeg model.

        Args:
            pretrained: Path to pretrained DINOv3 weights
            freeze_vit: If True, freezes the ViT backbone parameters
            drop_path_rate: Stochastic depth rate for transformer blocks
            lora: Optional dict with LoRA configuration (r, lora_alpha) for parameter-efficient fine-tuning
            out_channels: Number of output feature channels
            dinov3_forward_mode: Feature extraction mode - 'four_levels' or 'one_level'
        """
        from ever.module.dinov3 import vitl16_sat493m
        # assert self.cfg.pretrained is not None, "Please specify the pretrained model path."

        self.encoder = vitl16_sat493m(pretrained=pretrained, drop_path_rate=drop_path_rate)
        embed_dim = self.encoder.embed_dim
        if freeze_vit:
            self.encoder.requires_grad_(False)

        if lora:
            from torchange.module.lora import LoraLinear
            LoraLinear.convert_lora_linear(self.encoder, **lora)
            er.info(f"applying LoRA: {lora}")

        self.sfp = SimpleFeaturePyramid(embed_dim, embed_dim)
        in_channels = [embed_dim for _ in range(4)]

        self.farseg = FarSegMixin(
            in_channels=in_channels,
            fpn_channels=out_channels,
            out_channels=out_channels,
        )

    def _forward_dinov3_four_levels(self, x):
        outputs = self.encoder.get_intermediate_layers(x, n=[5, 11, 17, 23], reshape=True, return_class_token=True)
        features = [out[0] for out in outputs]
        cls_tokens = [out[1] for out in outputs]
        features = self.sfp(features)
        return features, cls_tokens[-1]

    def _forward_dinov3_one_level(self, x):
        output, cls_token = self.encoder.get_intermediate_layers(x, n=1, reshape=True, return_class_token=True)[0]
        features = self.sfp(output)
        return features, cls_token

    def forward(self, x):
        """
        Forward pass through DINOv3 encoder and FarSeg modules.

        Args:
            x: Input tensor of shape [N, 3, H, W]

        Returns:
            Refined output features of shape [N, out_channels, H, W]
        """
        if self.cfg.dinov3_forward_mode == 'four_levels':
            features, cls_token = self._forward_dinov3_four_levels(x)
        elif self.cfg.dinov3_forward_mode == 'one_level':
            features, cls_token = self._forward_dinov3_one_level(x)
        else:
            raise ValueError(f"Unknown dinov3_forward_mode: {self.cfg.dinov3_forward_mode}")
        features = self.farseg(features, scene_embedding=cls_token.reshape(cls_token.shape[0], -1, 1, 1))

        return features

    def custom_param_groups(self):
        param_groups = [{'params': [], 'weight_decay': 0.}, {'params': []}]
        for i, p in self.named_parameters():
            if 'norm' in i:
                param_groups[0]['params'].append(p)
            else:
                param_groups[1]['params'].append(p)
        return param_groups


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    m = DINOv3ViTLFarSeg(
        out_channels=256,
        dinov3_forward_mode='four_levels',
        lora=dict(r=32, lora_alpha=320),
    )
    print(m)
    er.param_util.trainable_parameters(m)
