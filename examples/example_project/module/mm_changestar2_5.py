# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import ever as er
import torch
import torch.nn as nn
import torchange as tc
from torchange.models.changestar2_5 import ChangeStar2_5, up4x


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
class mmChangeStar2_5(ChangeStar2_5):
    def __init__(self, cfg):
        super().__init__(cfg)
        dense_enc_sar = er.builder.make_model(self.cfg.image_dense_encoder)
        patch_first_conv(dense_enc_sar, 1, 3, True)

        self.image_dense_encoder = nn.ModuleDict(dict(
            opt=er.builder.make_model(self.cfg.image_dense_encoder),
            sar=dense_enc_sar,
        ))

    def parse_x(self, x):
        opt = x[:, :3, :, :]
        sar = x[:, 3:, :, :]
        return opt, sar

    def forward(self, x, y=None):
        opt, sar = self.parse_x(x)
        opt_embed = self.image_dense_encoder['opt'](opt)
        sar_embed = self.image_dense_encoder['sar'](sar)

        s1_logit, _, c_logit = self.mixin(opt_embed, sar_embed)
        s1_logit = up4x(s1_logit)
        c_logit = up4x(c_logit)

        output = tc.ChangeDetectionModelOutput(
            t1_semantic_prediction=s1_logit, change_prediction=c_logit
        )
        if self.training:
            assert len(y['masks']) == 1, 'bright dataset has only one change mask'
            cmask = y['masks'][-1]
            y['masks'] = tc.Mask(change_mask=cmask, t1_semantic_mask=(cmask > 0).to(torch.float32))
            return self.train_loss(output, y)
        else:
            return self.predict(output)
