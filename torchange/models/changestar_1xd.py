# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F

import ever as er
import ever.module as M
import ever.module.loss as L
from torchange.utils.outputs import ChangeDetectionModelOutput
from torchange.utils.mask_data import Mask

from einops import rearrange


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
def sc_mse_loss(s1_logit, s2_logit, change_mask):
    c_gt = change_mask.to(torch.float32).unsqueeze(1)

    s1_p = s1_logit.log_softmax(dim=1).exp()
    s2_p = s2_logit.log_softmax(dim=1).exp()

    diff = (s1_p - s2_p) ** 2
    losses = (1 - c_gt) * diff + c_gt * (1 - diff)

    return losses.mean()


@er.registry.MODEL.register(verbose=False)
class ChangeStar1xd(er.ERModule):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = er.builder.make_model(self.config.encoder)

        self.cfg.head.in_channels = 2 * self.encoder.config.out_channels
        self.cfg.head.out_channels = self.encoder.config.out_channels

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

    def _semantic_loss(self, pred, target, loss_cfg, ignore_index=255):
        loss_dict = {}
        if pred.size(1) > 1:
            target = target.to(torch.int64)
            loss_dict['ce_loss'] = F.cross_entropy(pred, target.to(torch.int64), reduction='mean', ignore_index=ignore_index)
        else:
            target = target.to(torch.float32)
            loss_dict['bce_loss'] = L.binary_cross_entropy_with_logits(
                pred, target.reshape_as(pred), reduction='mean',
                ignore_index=ignore_index
            )

        if 'tver' in loss_cfg:
            tver = loss_cfg.tver.to_dict()  # make torch.compile happy
            alpha = tver.get('alpha', 0.5)
            beta = 1 - alpha
            gamma = tver.get('gamma', 1.0)
            loss_dict['tver_loss'] = L.tversky_loss_with_logits(
                pred, target, alpha=alpha, beta=beta, gamma=gamma, ignore_index=ignore_index
            )
        else:
            alpha = 0.5
            beta = 1 - alpha
            gamma = 1.0
            loss_dict['dice_loss'] = L.tversky_loss_with_logits(
                pred, target, alpha=alpha, beta=beta, gamma=gamma, ignore_index=ignore_index
            )
        return loss_dict

    def _change_loss(self, change_logit, gt_change, ignore_index=255):
        loss_dict = {}
        is_binary = change_logit.size(1) == 1
        loss_cfg = self.cfg.loss.change
        if ('bce' in loss_cfg) or ('ce' in loss_cfg):
            if is_binary:
                ls = loss_cfg.bce.get('ls', 0.0)
                loss_dict['c_bce_loss'] = L.label_smoothing_binary_cross_entropy(
                    change_logit, gt_change, eps=ls,
                    ignore_index=ignore_index
                )
            else:
                ls = loss_cfg.ce.get('ls', 0.0)
                loss_dict['c_ce_loss'] = F.cross_entropy(
                    change_logit, gt_change.to(torch.int64),
                    ignore_index=ignore_index,
                    label_smoothing=ls
                )

        if 'dice' in loss_cfg:
            gamma = loss_cfg.dice.get('gamma', 1.0)
            loss_dict['c_dice_loss'] = L.tversky_loss_with_logits(change_logit, gt_change, alpha=0.5, beta=0.5, gamma=gamma,
                                                                  ignore_index=ignore_index)

        if 'tver' in loss_cfg:
            alpha = loss_cfg.tver.get('alpha', 0.5)
            beta = 1 - alpha
            gamma = loss_cfg.tver.get('gamma', 1.0)
            loss_dict['c_tver_loss'] = L.tversky_loss_with_logits(change_logit, gt_change, alpha=alpha, beta=beta, gamma=gamma,
                                                                  ignore_index=ignore_index)

        return loss_dict

    @torch.amp.autocast('cuda', dtype=torch.float32)
    def loss(self, preds: ChangeDetectionModelOutput, y):
        # masks[0] - cls, masks[1] - cls, masks[2] - change
        # masks[0] - cls, masks[1] - change
        # masks[0] - change
        assert hasattr(self.cfg.loss, 'change'), 'loss must contain change term'

        y_masks = y['masks']
        if not isinstance(y_masks, Mask):
            y_masks = Mask.from_list(y_masks)

        gt_change = y_masks.change_mask.to(torch.float32)
        change_logit = preds.change_prediction.to(torch.float32)

        loss_dict = {}
        loss_dict |= self._change_loss(change_logit, gt_change)

        for t in ['t1', 't2']:
            pred = getattr(preds, f'{t}_semantic_prediction')
            mask = getattr(y_masks, f'{t}_semantic_mask')

            if pred is not None and t in self.cfg.loss:
                loss_cfg = getattr(self.cfg.loss, t)
                losses = self._semantic_loss(pred, mask, loss_cfg)
                loss_dict |= {f"{t}_{k}": v for k, v in losses.items()}

        if 'sc' in self.cfg.loss:
            loss_dict['sc_mse_loss'] = sc_mse_loss(
                preds.t1_semantic_prediction,
                preds.t2_semantic_prediction,
                y_masks.change_mask
            )

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
            return ChangeDetectionModelOutput(
                change_prediction=change_logit,
                t1_semantic_prediction=t1_semantic_logit if self.num_semantic_classes else None,
                t2_semantic_prediction=t2_semantic_logit if self.num_semantic_classes else None,
            )
        else:
            def _act(logit):
                if logit.size(1) > 1:
                    return logit.softmax(dim=1)
                else:
                    return logit.sigmoid()

            return ChangeDetectionModelOutput(
                change_prediction=_act(change_logit),
                t1_semantic_prediction=_act(t1_semantic_logit) if self.num_semantic_classes else None,
                t2_semantic_prediction=_act(t2_semantic_logit) if self.num_semantic_classes else None,
            )
