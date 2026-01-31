# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import ever as er
import torch
import torch.nn as nn
from torchange.models.changeos import ChangeOS


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
