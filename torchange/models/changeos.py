# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import ever as er
import ever.module as M
import ever.module.loss as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import measure
import torchange as tc

""" tips on how to build a changeos model from a predefined config file

import ever as er
from torchange.configs.changeos import cos_r50, cos_swint
from torchange.models.changeos import ChangeOS

model = er.builder.make_model(cos_r50.config['model'])
model = er.builder.make_model(cos_swint.config['model'])
"""

class FuseConv(nn.Sequential):
    def __init__(self, inchannels, outchannels):
        super(FuseConv, self).__init__(
            nn.Conv2d(inchannels, outchannels, kernel_size=1),
            nn.BatchNorm2d(outchannels),
        )
        self.relu = nn.ReLU(True)
        self.se = M.SEBlock(outchannels, 16)

    def forward(self, x):
        out = super(FuseConv, self).forward(x)
        residual = out
        out = self.se(out)
        out += residual
        out = self.relu(out)
        return out


class FuseMLP(nn.Sequential):
    def __init__(self, inchannels, outchannels):
        super().__init__(
            nn.Conv2d(inchannels, outchannels, kernel_size=1),
            M.LayerNorm2d(outchannels),
            nn.GELU(),
            nn.Conv2d(outchannels, outchannels, kernel_size=1),
            M.LayerNorm2d(outchannels),
        )


@er.registry.MODEL.register(verbose=False)
class ChangeOSDecoder(er.ERModule):
    def __init__(self, config):
        super().__init__(config)

        self.loc_neck = nn.Sequential(
            M.FPN(self.config.in_channels_list, self.config.out_channels, M.fpn.conv_bn_block),
            M.AssymetricDecoder(self.config.out_channels, self.config.out_channels)
        )

        self.dam_neck = nn.Sequential(
            M.FPN(self.config.in_channels_list, self.config.out_channels, M.fpn.conv_bn_block),
            M.AssymetricDecoder(self.config.out_channels, self.config.out_channels)
        )
        if self.config.fusion_type == 'residual_se':
            self.fuse_conv = FuseConv(2 * self.config.out_channels, self.config.out_channels)
        elif self.config.fusion_type == '2mlps':
            self.fuse_conv = FuseMLP(2 * self.config.out_channels, self.config.out_channels)
        else:
            raise ValueError(f'unknown fusion_type: {self.config.fusion_type}')

    def forward(self, t1_features, t2_features):
        t1_features = self.loc_neck(t1_features)
        t2_features = self.dam_neck(t2_features)

        st_features = self.fuse_conv(torch.cat([t1_features, t2_features], dim=1))
        return t1_features, st_features

    def set_default_config(self):
        self.config.update(dict(
            in_channels_list=(64, 128, 256, 512),
            out_channels=256,
            fusion_type='residual_se'
        ))


class DeepHead(nn.Module):
    def __init__(self, in_channels, bottlneck_channels, num_blocks, num_classes, upsample_scale):
        super().__init__()
        assert num_blocks > 0
        self.relu = nn.ReLU(True)
        self.blocks = nn.ModuleList([nn.Sequential(
            # 1x1
            nn.Conv2d(in_channels, bottlneck_channels, 1),
            nn.BatchNorm2d(bottlneck_channels),
            nn.ReLU(True),
            # 3x3
            nn.Conv2d(bottlneck_channels, bottlneck_channels, 3, 1, 1),
            nn.BatchNorm2d(bottlneck_channels),
            # 1x1
            nn.Conv2d(bottlneck_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            M.SEBlock(in_channels, 16)
        ) for _ in range(num_blocks)])

        self.cls = nn.Conv2d(in_channels, num_classes, 1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=upsample_scale)

    def forward(self, x, upsample=True):
        indentity = x
        for m in self.blocks:
            x = m(x)
            x += indentity
            x = self.relu(x)
            indentity = x
        x = self.cls(x)
        if upsample:
            x = self.up(x)
        return x


@er.registry.MODEL.register(verbose=False)
class ChangeOSHead(er.ERModule):
    def __init__(self, config):
        super().__init__(config)
        if self.config.loc_head.deep_head:
            self.config.loc_head.pop('deep_head')
            self.loc_cls = DeepHead(**self.config.loc_head)
        else:
            self.loc_cls = M.ConvUpsampling(
                self.config.loc_head.in_channels,
                self.config.loc_head.num_classes,
                self.config.loc_head.upsample_scale,
                1
            )

        if self.config.dam_head.deep_head:
            self.config.dam_head.pop('deep_head')
            self.dam_cls = DeepHead(**self.config.dam_head)
        else:
            self.dam_cls = M.ConvUpsampling(
                self.config.dam_head.in_channels,
                self.config.dam_head.num_classes,
                self.config.dam_head.upsample_scale,
                1
            )

    def forward(self, t1_features, st_features, y=None):
        loc_logit = self.loc_cls(t1_features)
        dam_logit = self.dam_cls(st_features)

        if self.training:
            s1 = loc_logit
            c = dam_logit

            gt_t1 = (y['masks'][0] > 0).to(torch.float32)
            t1_bce_loss = L.binary_cross_entropy_with_logits(s1, gt_t1)
            t1_tver_loss = L.tversky_loss_with_logits(
                s1, gt_t1,
                alpha=0.9, beta=0.1, gamma=1.0
            )
            gt_c = y['masks'][-1].to(torch.int64)
            c_ce_loss = F.cross_entropy(c, gt_c, ignore_index=255)
            c_dice_loss = L.dice_loss_with_logits(c, gt_c, ignore_index=255)
            loss_dict = {}

            loss_dict.update(
                t1_bce_loss=t1_bce_loss,
                t1_tver_loss=t1_tver_loss,
                c_ce_loss=c_ce_loss,
                c_dice_loss=c_dice_loss
            )
            return loss_dict

        return tc.ChangeDetectionModelOutput(
            change_prediction=dam_logit.softmax(dim=1),
            t1_semantic_prediction=loc_logit.sigmoid(),
        )

    def pixel_based_infer(self, pre_pred, post_pred, logit=True):
        if logit:
            pr_loc = pre_pred > 0.
        else:
            pr_loc = pre_pred > .5
        pr_dam = post_pred.argmax(dim=1, keepdim=True)
        return pr_loc, pr_dam

    def object_based_infer(self, pre_pred, post_pred, logit=True):
        if logit:
            loc = pre_pred > 0.
        else:
            loc = pre_pred > .5
        loc = loc.cpu().squeeze(1).numpy()
        dam = post_pred.argmax(dim=1).cpu().squeeze(1).numpy()

        refined_dam = np.zeros_like(dam)
        for i, (single_loc, single_dam) in enumerate(zip(loc, dam)):
            refined_dam[i, :, :] = _object_vote(single_loc, single_dam)

        return torch.from_numpy(loc), torch.from_numpy(refined_dam)

    def set_default_config(self):
        self.config.update(dict(
            loc_head=None,
            dam_head=None,
            inference_mode='pixel-based'
        ))


def _object_vote(loc, dam, cls_weight_list=(8., 38., 25., 11.)):
    damage_cls_list = [1, 2, 3, 4]
    # 1. read localization mask
    local_mask = loc
    # 2. get connected regions
    labeled_local, nums = measure.label(local_mask, connectivity=2, background=0, return_num=True)
    region_idlist = np.unique(labeled_local)
    # 3. start vote
    if len(region_idlist) > 1:
        dam_mask = dam
        new_dam = local_mask.copy()
        for region_id in region_idlist:
            # if background, ignore it
            if all(local_mask[local_mask == region_id]) == 0:
                continue
            region_dam_count = [int(np.sum(dam_mask[labeled_local == region_id] == dam_cls_i)) * cls_weight \
                                for dam_cls_i, cls_weight in zip(damage_cls_list, cls_weight_list)]
            # vote
            dam_index = np.argmax(region_dam_count) + 1
            new_dam = np.where(labeled_local == region_id, dam_index, new_dam)
    else:
        new_dam = local_mask.copy()

    return new_dam


@er.registry.MODEL.register(verbose=False)
class ChangeOS(er.ERModule):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = er.registry.MODEL[self.config.encoder.type](self.config.encoder.params)
        self.decoder = ChangeOSDecoder(self.config.decoder)
        self.head = ChangeOSHead(self.config.head)
        self.init_from_weight_file()

    def forward(self, x, y=None):
        features = tc.bitemporal_forward(self.encoder, x)
        decoder_features = self.decoder(*features)
        t1_features, st_features = decoder_features
        return self.head(t1_features, st_features, y=y)

    def custom_param_groups(self):
        param_groups = []
        param_groups += self.encoder.custom_param_groups()
        param_groups += [{'params': self.decoder.parameters()}]
        param_groups += [{'params': self.head.parameters()}]
        return param_groups

    def set_default_config(self):
        self.cfg.update(dict(
            encoder=None,
            decoder=dict(
                in_channels_list=[96 * (2 ** i) for i in range(4)],
                out_channels=256,
                fusion_type='2mlps'
            ),
            head=dict(
                loc_head=dict(
                    in_channels=256,
                    bottlneck_channels=128,
                    num_blocks=1,
                    num_classes=1,
                    upsample_scale=4.,
                    deep_head=False,
                ),
                dam_head=dict(
                    in_channels=256,
                    bottlneck_channels=128,
                    num_blocks=1,
                    num_classes=5,
                    upsample_scale=4.,
                    deep_head=False,
                ),
            ),
        ))
