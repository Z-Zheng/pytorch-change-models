# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ever as er
import ever.module as M

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchange.module._sam_vit import SAMEncoder


@er.registry.MODEL.register()
class FarSegEncoder(M.ResNetEncoder):
    def __init__(self, config):
        super().__init__(config)
        if self.config.resnet_type in ['resnet18', 'resnet34']:
            max_channels = 512
        else:
            max_channels = 2048
        self.fpn = M.FPN([max_channels // (2 ** (3 - i)) for i in range(4)], 256)
        self.fsr = M.FSRelation(max_channels,
                                [256 for _ in range(4)],
                                256,
                                True)
        self.dec = M.AssymetricDecoder(256,
                                       self.config.out_channels)

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

    def forward(self, x):
        scene_embedding = F.adaptive_avg_pool2d(x[-1], 1)
        features = self.fpn(x)
        features = self.fsr(scene_embedding, features)
        features = self.dec(features)
        return features


@er.registry.MODEL.register()
class SAMEncoderFarSeg(SAMEncoder):
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
