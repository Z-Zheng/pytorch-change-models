# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import ever as er
import copy
import torch
import torch.nn as nn
from torchange.models.changestar_1xd import ChangeStar1xd
from torchange.utils.mask_data import Mask


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

    return module


@er.registry.MODEL.register()
class mmChangeStar1xd(ChangeStar1xd):
    def __init__(self, cfg):
        super().__init__(cfg)
        dense_enc_sar = copy.deepcopy(self.encoder)
        first_conv = patch_first_conv(dense_enc_sar, 1, 3, True)
        first_conv.requires_grad_(True)

        self.image_dense_encoder = nn.ModuleDict(dict(
            opt=copy.deepcopy(self.encoder),
            sar=dense_enc_sar,
        ))
        del self.encoder

    def forward(self, x, y=None):
        opt = x[:, :3, :, :]
        sar = x[:, 3:, :, :]
        opt_embed = self.image_dense_encoder['opt'](opt)
        sar_embed = self.image_dense_encoder['sar'](sar)

        preds = self.head(opt_embed, sar_embed)

        if self.training:
            assert len(y['masks']) == 1, 'bright dataset has only one change mask'
            cmask = y['masks'][-1]
            y['masks'] = Mask(change_mask=cmask, t1_semantic_mask=(cmask > 0).to(torch.float32))
            return self.loss(preds, y, self.cfg.loss)

        return preds

    def custom_param_groups(self):
        param_groups = []

        param_groups += self.image_dense_encoder['opt'].custom_param_groups()
        param_groups += self.image_dense_encoder['sar'].custom_param_groups()
        param_groups += [{'params': self.head.parameters()}]

        return param_groups

    def log_info(self):
        return dict(
            image_dense_encoder=self.image_dense_encoder,
            changemixin=self.head
        )
