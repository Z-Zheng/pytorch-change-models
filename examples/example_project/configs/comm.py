# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import albumentations as A
import albumentations.pytorch
from torchange.data.bright import Setup

train_data = dict(
    train=dict(
        type='HFBRIGHT',
        params=dict(
            setting=Setup.STANDARD,
            setting_splits=['train'],
            transform=A.Compose([
                A.RandomCrop(640, 640),
                A.D4(),
                A.Normalize(
                    (0.430, 0.411, 0.296, 0.225),
                    (0.213, 0.156, 0.143, 0.151),
                    max_pixel_value=255
                ),
                A.pytorch.ToTensorV2(),
            ]),
            batch_size=16,
            num_workers=4,
        ),
    ),
)