# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from ever.module import ResNetEncoder

config = dict(
    model=dict(
        type='ChangeOS',
        params=dict(
            encoder=dict(
                type='ResNetEncoder',
                params=dict(
                    resnet_type='resnet18',
                    pretrained=True,
                    output_stride=32,
                ),
            ),
            decoder=dict(
                in_channels_list=(64, 128, 256, 512),
                out_channels=256,
                fusion_type='residual_se'
            ),
            head=dict(
                loc_head=dict(
                    in_channels=256,
                    bottlneck_channels=128,
                    num_blocks=1,
                    num_classes=1,
                    upsample_scale=4.,
                    deep_head=True,
                ),
                dam_head=dict(
                    in_channels=256,
                    bottlneck_channels=128,
                    num_blocks=1,
                    num_classes=5,
                    upsample_scale=4.,
                    deep_head=True,
                ),
            ),
        )
    ),
)
