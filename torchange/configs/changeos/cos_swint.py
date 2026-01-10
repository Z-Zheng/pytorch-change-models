# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from torchvision.models import Swin_T_Weights

config = dict(
    model=dict(
        type='ChangeOS',
        params=dict(
            encoder=dict(
                type='TVSwinTransformer',
                params=dict(
                    name='swin_t',
                    weights=Swin_T_Weights.IMAGENET1K_V1
                ),
            ),
            decoder=dict(
                in_channels_list=[96 * (2 ** i) for i in range(4)],
                out_channels=256,
                fusion_type='2mlps'
            ),
        )
    ),
)
