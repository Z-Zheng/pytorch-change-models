# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import ever as er
import ever.module.loss as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    import segmentation_models_pytorch as smp
    from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
except ImportError:
    print(f"segmentation_models_pytorch not found. please `pip install segmentation_models_pytorch`")



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


@torch.cuda.amp.autocast(dtype=torch.float32)
def mse_loss(s1_logit, s2_logit, gt_masks):
    c_gt = gt_masks[-1].to(torch.float32).unsqueeze(1)

    s1_p = s1_logit.log_softmax(dim=1).exp()
    s2_p = s2_logit.log_softmax(dim=1).exp()

    diff = (s1_p - s2_p) ** 2
    losses = (1 - c_gt) * diff + c_gt * (1 - diff)

    return losses.mean()


@torch.cuda.amp.autocast(dtype=torch.float32)
def loss(
        s1_logit, s2_logit, c_logit,
        gt_masks,
):
    s1_gt = gt_masks[0].to(torch.int64)
    s2_gt = gt_masks[1].to(torch.int64)

    s1_ce = F.cross_entropy(s1_logit, s1_gt, ignore_index=255)
    s1_dice = L.dice_loss_with_logits(s1_logit, s1_gt)

    s2_ce = F.cross_entropy(s2_logit, s2_gt, ignore_index=255)
    s2_dice = L.dice_loss_with_logits(s2_logit, s2_gt)

    c_gt = gt_masks[-1].to(torch.float32)
    c_dice = L.dice_loss_with_logits(c_logit, c_gt)
    c_bce = L.binary_cross_entropy_with_logits(c_logit, c_gt)

    sim_loss = mse_loss(s1_logit, s2_logit, gt_masks)
    return {
        's1_ce_loss': s1_ce,
        's1_dice_loss': s1_dice,
        's2_ce_loss': s2_ce,
        's2_dice_loss': s2_dice,
        'c_dice_loss': c_dice,
        'c_bce_loss': c_bce,
        # to improve semantic-change consistency, this is a well-known issue in ChangeMask-like SCD methods.
        # original implementation doesn't have this objective.
        'sim_loss': sim_loss
    }


class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return x.squeeze(dim=self.dim)


class SpatioTemporalInteraction(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation=1,
                 type='conv3d'):
        if type == 'conv3d':
            padding = dilation * (kernel_size - 1) // 2
            super(SpatioTemporalInteraction, self).__init__(
                nn.Conv3d(in_channels, out_channels, [2, kernel_size, kernel_size], stride=1,
                          dilation=(1, dilation, dilation),
                          padding=(0, padding, padding),
                          bias=False),
                Squeeze(dim=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )
        elif type == 'conv1plus2d':
            super(SpatioTemporalInteraction, self).__init__(
                nn.Conv3d(in_channels, out_channels, (2, 1, 1), stride=1,
                          padding=(0, 0, 0),
                          bias=False),
                Squeeze(dim=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, kernel_size, 1,
                          kernel_size // 2) if kernel_size > 1 else nn.Identity(),
                nn.BatchNorm2d(out_channels) if kernel_size > 1 else nn.Identity(),
                nn.ReLU(True) if kernel_size > 1 else nn.Identity(),
            )


class TemporalSymmetricTransformer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation=1,
                 interaction_type='conv3d',
                 symmetric_fusion='add'):
        super(TemporalSymmetricTransformer, self).__init__()

        if isinstance(in_channels, list) or isinstance(in_channels, tuple):
            self.t = nn.ModuleList([
                SpatioTemporalInteraction(inc, outc, kernel_size, dilation=dilation, type=interaction_type)
                for inc, outc in zip(in_channels, out_channels)
            ])
        else:
            self.t = SpatioTemporalInteraction(in_channels, out_channels, kernel_size, dilation=dilation,
                                               type=interaction_type)

        if symmetric_fusion == 'add':
            self.symmetric_fusion = lambda x, y: x + y
        elif symmetric_fusion == 'mul':
            self.symmetric_fusion = lambda x, y: x * y
        elif symmetric_fusion == None:
            self.symmetric_fusion = None

    def forward(self, features1, features2):
        if isinstance(features1, list):
            d12_features = [op(torch.stack([f1, f2], dim=2)) for op, f1, f2 in
                            zip(self.t, features1, features2)]
            if self.symmetric_fusion:
                d21_features = [op(torch.stack([f2, f1], dim=2)) for op, f1, f2 in
                                zip(self.t, features1, features2)]
                change_features = [self.symmetric_fusion(d12, d21) for d12, d21 in zip(d12_features, d21_features)]
            else:
                change_features = d12_features
        else:
            if self.symmetric_fusion:
                change_features = self.symmetric_fusion(self.t(torch.stack([features1, features2], dim=2)),
                                                        self.t(torch.stack([features2, features1], dim=2)))
            else:
                change_features = self.t(torch.stack([features1, features2], dim=2))
            change_features = change_features.squeeze(dim=2)
        return change_features


@er.registry.MODEL.register()
class ChangeMask(er.ERModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.encoder = smp.encoders.get_encoder('efficientnet-b0', weights='imagenet')
        out_channels = self.encoder.out_channels
        self.semantic_decoder = UnetDecoder(
            encoder_channels=out_channels,
            decoder_channels=[256, 128, 64, 32, 16],
        )

        self.change_decoder = UnetDecoder(
            encoder_channels=out_channels,
            decoder_channels=[256, 128, 64, 32, 16],
        )

        self.temporal_transformer = TemporalSymmetricTransformer(
            out_channels, out_channels,
            3, interaction_type='conv3d', symmetric_fusion='add',
        )
        self.s = nn.Conv2d(16, self.cfg.num_semantic_classes, 1)
        self.c = nn.Conv2d(16, 1, 1)

    def forward(self, x, y=None):
        t1_features, t2_features = bitemporal_forward(self.encoder, x)

        s1_logit = self.s(self.semantic_decoder(*t1_features))
        s2_logit = self.s(self.semantic_decoder(*t2_features))

        temporal_features = self.temporal_transformer(t1_features, t2_features)
        c_logit = self.c(self.change_decoder(*temporal_features))

        if self.training:
            return loss(s1_logit, s2_logit, c_logit, y['masks'])

        return {
            T1SEM: s1_logit.softmax(dim=1),
            T2SEM: s2_logit.softmax(dim=1),
            CHANGE: c_logit.sigmoid(),
        }

    def set_default_config(self):
        self.cfg.update(dict(
            num_semantic_classes=6
        ))
