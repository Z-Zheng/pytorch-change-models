# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.utils.data import Dataset
from skimage.io import imread
import albumentations as A
import torch
from pathlib import Path
import numpy as np


class BinarizeMask(A.DualTransform):
    def __init__(self):
        super().__init__(True, 1.0)

    def apply(self, img, *args, **params):
        return img

    def apply_to_mask(self, mask, *args, **params):
        return (mask > 0).astype(np.float32)

    def get_transform_init_args_names(self):
        return ()


def check_name(t1_image_name, other):
    assert t1_image_name == other, f'expected {t1_image_name}, but {other}'


class BitemporalDataset(Dataset):
    PRE = 'pre'
    TWISE = 'temporalwise'
    POST = 'post'

    def __init__(
            self,
            t1_image_fps,
            t2_image_fps,
            t1_mask_fps=None,
            t2_mask_fps=None,
            change_fps=None,
            transform=None,
            name_checker=check_name
    ):
        self.t1_image_fps = t1_image_fps
        self.t2_image_fps = t2_image_fps
        self.t1_mask_fps = t1_mask_fps
        self.t2_mask_fps = t2_mask_fps
        self.change_fps = change_fps

        self.name_checker = name_checker

        self.t = {
            self.PRE: A.NoOp(),
            self.TWISE: A.NoOp(),
            self.POST: A.NoOp(),
        }
        if isinstance(transform, A.Compose):
            self.t[self.POST] = self.to_bitemporal_compose(transform)
        elif isinstance(transform, dict):
            for k, v in transform.items():
                assert k in [self.PRE, self.TWISE, self.POST]
                if isinstance(v, A.Compose):
                    v = self.to_bitemporal_compose(v)
                self.t[k] = v
        else:
            self.t = transform

    def __getitem__(self, idx):
        base_name = Path(self.t1_image_fps[idx]).name
        self.name_checker(base_name, Path(self.t2_image_fps[idx]).name)

        img1 = imread(self.t1_image_fps[idx])
        img2 = imread(self.t2_image_fps[idx])

        data = {
            'image': img1,
            't2_image': img2
        }

        if self.t1_mask_fps:
            self.name_checker(base_name, Path(self.t1_mask_fps[idx]).name)
            msk1 = imread(self.t1_mask_fps[idx])
            data['mask'] = msk1

        if self.t2_mask_fps:
            self.name_checker(base_name, Path(self.t2_mask_fps[idx]).name)
            msk2 = imread(self.t2_mask_fps[idx])
            data['t2_mask'] = msk2

        if self.change_fps:
            self.name_checker(base_name, Path(self.change_fps[idx]).name)
            cmask = imread(self.change_fps[idx])
            data['change'] = cmask

        data = self.data_transform(data)

        img = torch.cat([data['image'], data['t2_image']], dim=0)

        masks = []
        if 'mask' in data:
            masks.append(data['mask'])

        if 't2_mask' in data:
            masks.append(data['t2_mask'])

        if 'change' in data:
            masks.append(data['change'])

        ann = dict(
            masks=masks,
            image_filename=str(Path(self.t1_image_fps[idx]).name)
        )

        return img, ann

    def __len__(self):
        return len(self.t1_image_fps)

    def data_transform(self, data):
        if isinstance(self.t, dict):
            data = self.t[self.PRE](**data)

            if not isinstance(self.t[self.TWISE], A.NoOp):
                if 'mask' in data:
                    t1_data = {'image': data['image'], 'mask': data['mask']}
                else:
                    t1_data = {'image': data['image']}
                t1_data = self.t[self.TWISE](**t1_data)
                data.update(t1_data)

                if 't2_mask' in data:
                    t2_data = {'image': data['t2_image'], 'mask': data['t2_mask']}
                else:
                    t2_data = {'image': data['t2_image']}
                t2_data = self.t[self.TWISE](**t2_data)
                t2_data = {f't2_{k}': v for k, v in t2_data.items()}
                data.update(t2_data)

            data = self.t[self.POST](**data)
        else:
            data = self.t(data)
        return data

    @staticmethod
    def to_bitemporal_compose(compose):
        assert isinstance(compose, A.Compose)

        return A.Compose(compose.transforms, additional_targets={
            't2_image': 'image',
            't2_mask': 'mask',
            'change': 'mask'
        })
