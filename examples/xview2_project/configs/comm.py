# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import albumentations as A
import albumentations.pytorch

# No RandomCrop here: unlike HFBRIGHT, HFxView2 already crops each 1024x1024 image to a
# `crop_size` tile in compute_tile_slice(). A.D4() is purely geometric, so the 255 ignore
# values in the damage mask survive it, and to_bitemporal_compose() makes it apply the same
# geometry to t2_image/t2_mask.
train_data = dict(
    train=dict(
        type='HFxView2',
        params=dict(
            hf_repo_name='EVER-Z/torchange_xView2',
            splits=['train', 'tier3'],
            training=True,
            crop_size=512,
            stride=256,
            transform=A.Compose([
                A.D4(),
                A.Normalize(),
                A.pytorch.ToTensorV2(),
            ]),
            batch_size=8,
            num_workers=4,
        ),
    ),
)
