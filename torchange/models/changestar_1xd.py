# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import ever as er
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import ever.module as M
import ever.module.loss as L

CHANGE = 'change_prediction'
T1SEM = 't1_semantic_prediction'
T2SEM = 't2_semantic_prediction'


def bitemporal_forward(module, x):
    x = rearrange(x, 'b (t c) h w -> (b t) c h w', t=2)
    features = module(x)
    if isinstance(features, list) or isinstance(features, tuple):
        t1_features, t2_features = [], []
        for feat in features:
            t1_feat, t2_feat = rearrange(feat, '(b t) c h w -> t b c h w', t=2)
            t1_features.append(t1_feat)
            t2_features.append(t2_feat)
    else:
        t1_features, t2_features = rearrange(features, '(b t) c h w -> t b c h w', t=2)

    return t1_features, t2_features


@torch.amp.autocast('cuda', dtype=torch.float32)
def sc_mse_loss(s1_logit, s2_logit, gt_masks):
    c_gt = gt_masks[-1].to(torch.float32).unsqueeze(1)

    s1_p = s1_logit.log_softmax(dim=1).exp()
    s2_p = s2_logit.log_softmax(dim=1).exp()

    diff = (s1_p - s2_p) ** 2
    losses = (1 - c_gt) * diff + c_gt * (1 - diff)

    return losses.mean()


@er.registry.MODEL.register()
class ChangeStar1xd(er.ERModule):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = er.registry.MODEL[self.config.encoder.type](self.config.encoder.params)

        self.cfg.head.in_channels = 2 * self.config.encoder.params.out_channels
        self.cfg.head.out_channels = self.config.encoder.params.out_channels

        self.head = ChangeMixinBiSupN1(**self.cfg.head)
        self.init_from_weight_file()

    def forward(self, x, y=None):
        if self.cfg.encoder.bitemporal_forward:
            bitemporal_features = bitemporal_forward(self.encoder, x)
        else:
            bitemporal_features = self.encoder(x)

        preds = self.head(*bitemporal_features)

        if self.training:
            return self.loss(preds, y)

        return preds

    @torch.amp.autocast('cuda', dtype=torch.float32)
    def loss(self, preds, y):
        # masks[0] - cls, masks[1] - cls, masks[2] - change
        # masks[0] - cls, masks[1] - change
        # masks[0] - change
        gt_change = y['masks'][-1].to(torch.float32)
        change_logit = preds[CHANGE].to(torch.float32)

        loss_dict = dict()
        if hasattr(self.cfg.loss, 'change'):
            if ('bce' in self.cfg.loss.change) or ('ce' in self.cfg.loss.change):
                if change_logit.size(1) == 1:
                    ls = self.cfg.loss.change.bce.get('ls', 0.0)
                    loss = L.label_smoothing_binary_cross_entropy(change_logit, gt_change, eps=ls)
                    loss_dict.update(
                        c_bce_loss=loss
                    )
                else:
                    ls = self.cfg.loss.change.ce.get('ls', 0.0)
                    loss = F.cross_entropy(change_logit, gt_change.to(torch.int64), ignore_index=255, label_smoothing=ls)
                    loss_dict.update(
                        c_ce_loss=loss
                    )

            if 'dice' in self.cfg.loss.change:
                gamma = self.cfg.loss.change.dice.get('gamma', 1.0)
                if change_logit.size(1) == 1:
                    loss_dict.update(
                        c_dice_loss=L.tversky_loss_with_logits(change_logit, gt_change, alpha=0.5, beta=0.5, gamma=gamma),
                    )
                else:
                    loss_dict.update(
                        c_dice_loss=L.dice_loss_with_logits(change_logit, gt_change),
                    )

        if preds[T1SEM] is not None and 't1' in self.cfg.loss:
            gt_t1 = y['masks'][0]
            if preds[T1SEM].size(1) > 1:
                loss_dict.update(dict(
                    t1_ce_loss=F.cross_entropy(preds[T1SEM], gt_t1.to(torch.int64), reduction='mean', ignore_index=255),
                    t1_dice_loss=L.dice_loss_with_logits(preds[T1SEM], gt_t1.to(torch.int64))
                ))
            else:
                gt_t1 = gt_t1.to(torch.float32)
                loss_dict.update(dict(
                    t1_bce_loss=L.binary_cross_entropy_with_logits(
                        preds[T1SEM], gt_t1.reshape_as(preds[T1SEM]), reduction='mean'),
                    t1_dice_loss=L.dice_loss_with_logits(preds[T1SEM], gt_t1),
                ))

        if preds[T2SEM] is not None and 't2' in self.cfg.loss:
            gt_t2 = y['masks'][1]
            if preds[T2SEM].size(1) > 1:
                loss_dict.update(dict(
                    t2_ce_loss=F.cross_entropy(preds[T2SEM], gt_t2.to(torch.int64), reduction='mean', ignore_index=255),
                    t2_dice_loss=L.dice_loss_with_logits(preds[T2SEM], gt_t2.to(torch.int64)),
                ))
            else:
                gt_t2 = gt_t2.to(torch.float32)
                loss_dict.update(dict(
                    t2_bce_loss=F.binary_cross_entropy_with_logits(
                        preds[T2SEM], gt_t2.reshape_as(preds[T2SEM]), reduction='mean'),
                    t2_dice_loss=L.dice_loss_with_logits(preds[T2SEM], gt_t2),
                ))

        if 'sc' in self.cfg.loss:
            loss_dict.update(dict(
                sc_mse_loss=sc_mse_loss(preds[T1SEM], preds[T2SEM], y['masks'])
            ))

        return loss_dict

    def set_default_config(self):
        self.config.update(dict(
            encoder=dict(type=None, params=dict(), bitemporal_forward=False),
            head=dict(
                in_channels=-1,
                out_channels=-1,
                temporal_symmetric=True,
                num_semantic_classes=None,
                num_change_classes=None
            ),
            loss=dict(
            )
        ))

    def log_info(self):
        return dict(
            encoder=self.encoder,
            head=self.head
        )

    def custom_param_groups(self):
        param_groups = []

        if isinstance(self.encoder, er.ERModule):
            param_groups += self.encoder.custom_param_groups()
        else:
            param_groups += [{'params': self.encoder.parameters()}]

        if isinstance(self.head, er.ERModule):
            param_groups += self.head.custom_param_groups()
        else:
            param_groups += [{'params': self.head.parameters()}]

        return param_groups


class ChangeMixinBiSupN1(nn.Module):
    def __init__(self, in_channels, out_channels, temporal_symmetric=True,
                 num_semantic_classes=None, num_change_classes=None):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            M.LayerNorm2d(out_channels),
            nn.GELU()
        )
        if num_change_classes is None:
            num_change_classes = 1

        self.temporal_symmetric = temporal_symmetric
        self.change_conv = M.ConvUpsampling(out_channels, num_change_classes, scale_factor=4, kernel_size=1)
        self.num_semantic_classes = num_semantic_classes
        if isinstance(num_semantic_classes, int):
            self.semantic_conv = M.ConvUpsampling(out_channels, num_semantic_classes, scale_factor=4, kernel_size=1)
        elif isinstance(num_semantic_classes, (tuple, list)):
            self.semantic_conv = nn.ModuleList([
                M.ConvUpsampling(out_channels, nc, scale_factor=4, kernel_size=1)
                for nc in num_semantic_classes
            ])
        else:
            self.semantic_conv = nn.Identity()

    def forward(self, t1_feature, t2_feature):
        pre_logit = self.conv(torch.cat([t1_feature, t2_feature], dim=1))
        if self.temporal_symmetric:
            pre_logit = pre_logit + self.conv(torch.cat([t2_feature, t1_feature], dim=1))

        change_logit = self.change_conv(pre_logit)
        if isinstance(self.num_semantic_classes, int) or self.num_semantic_classes is None:
            t1_semantic_logit = self.semantic_conv(t1_feature)
            t2_semantic_logit = self.semantic_conv(t2_feature)
        else:
            t1_semantic_logit = self.semantic_conv[0](t1_feature)
            t2_semantic_logit = self.semantic_conv[1](t2_feature)

        if self.training:
            return {
                CHANGE: change_logit,
                T1SEM: t1_semantic_logit if self.num_semantic_classes else None,
                T2SEM: t2_semantic_logit if self.num_semantic_classes else None,
            }
        else:
            def _act(logit):
                if logit.size(1) > 1:
                    return logit.softmax(dim=1)
                else:
                    return logit.sigmoid()

            return {
                CHANGE: change_logit.sigmoid(),
                T1SEM: _act(t1_semantic_logit) if self.num_semantic_classes else None,
                T2SEM: _act(t2_semantic_logit) if self.num_semantic_classes else None,
            }
