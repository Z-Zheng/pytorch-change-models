# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import albumentations as A
import albumentations.pytorch
from torchange.data.bright import Setup


class _CropNonEmptyMaskIfExists(A.CropNonEmptyMaskIfExists):
    def get_params_dependent_on_data(self, params, data):
        return super().get_params_dependent_on_data(params, data={'mask': data['change']})


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
                    mean=(0.485, 0.456, 0.406, 0.225),
                    std=(0.229, 0.224, 0.225, 0.151),
                    max_pixel_value=255
                ),
                A.pytorch.ToTensorV2(),
            ]),
            batch_size=16,
            num_workers=4,
        ),
    ),
)
