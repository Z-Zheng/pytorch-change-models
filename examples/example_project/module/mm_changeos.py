# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import ever as er
import torch
import torch.nn as nn
import torch.nn.functional as F
import ever.module.loss as L
import torchange as tc
from torchange.models.changeos import ChangeOS, ChangeOSHead, ChangeOSDecoder
import types


def patch_first_conv(model, new_in_channels, default_in_channels=3, pretrained=True):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    # get first conv
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == default_in_channels:
            break

    weight = module.weight.detach()
    module.in_channels = new_in_channels

    if not pretrained:
        module.weight = nn.parameter.Parameter(
            torch.Tensor(module.out_channels, new_in_channels // module.groups, *module.kernel_size)
        )
        module.reset_parameters()

    elif new_in_channels == 1:
        new_weight = weight.sum(1, keepdim=True)
        module.weight = nn.parameter.Parameter(new_weight)

    else:
        new_weight = torch.Tensor(module.out_channels, new_in_channels // module.groups, *module.kernel_size)

        for i in range(new_in_channels):
            new_weight[:, i] = weight[:, i % default_in_channels]

        new_weight = new_weight * (default_in_channels / new_in_channels)
        module.weight = nn.parameter.Parameter(new_weight)


def _forward(self, t1_features, st_features, y=None):
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
        c_tver_loss = L.tversky_loss_with_logits(
            c, gt_c,
            alpha=[0.5, 0.5, 0.9, 0.5], gamma=1.0,
            ignore_index=255)
        loss_dict = {}

        loss_dict.update(
            t1_bce_loss=t1_bce_loss,
            t1_tver_loss=t1_tver_loss,
            c_ce_loss=c_ce_loss,
            c_tver_loss=c_tver_loss,
        )
        return loss_dict

    return tc.ChangeDetectionModelOutput(
        change_prediction=dam_logit.softmax(dim=1),
        t1_semantic_prediction=loc_logit.sigmoid(),
    )


@er.registry.MODEL.register()
class mmChangeOS(ChangeOS):
    def __init__(self, config):
        super().__init__(config)
        enc_sar = er.registry.MODEL[self.config.encoder.type](self.config.encoder.params)
        patch_first_conv(enc_sar, 1, 3, True)

        self.encoder = nn.ModuleDict(dict(
            opt=er.registry.MODEL[self.config.encoder.type](self.config.encoder.params),
            sar=enc_sar,
        ))

        self.decoder = ChangeOSDecoder(self.config.decoder)
        self.head = ChangeOSHead(self.config.head)
        self.head.forward = types.MethodType(_forward, self.head)

        self.init_from_weight_file()

    def parse_x(self, x):
        opt = x[:, :3, :, :]
        sar = x[:, 3:, :, :]
        return opt, sar

    def forward(self, x, y=None):
        opt, sar = self.parse_x(x)
        x_opt = self.encoder['opt'](opt)
        x_sar = self.encoder['sar'](sar)

        decoder_features = self.decoder(x_opt, x_sar)
        t1_features, st_features = decoder_features
        return self.head(t1_features, st_features, y=y)

    def custom_param_groups(self):
        param_groups = []
        param_groups += self.encoder['opt'].custom_param_groups()
        param_groups += self.encoder['sar'].custom_param_groups()
        param_groups += [{'params': self.decoder.parameters()}]
        param_groups += [{'params': self.head.parameters()}]
        return param_groups
