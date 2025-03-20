# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import albumentations as A
import ever as er
import numpy as np
import torch
from skimage.io import imread
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets
from typing import Dict

PRE = 'pre'
TWISE = 'temporalwise'
POST = 'post'


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


def to_bitemporal_compose(compose):
    assert isinstance(compose, A.Compose)

    return A.Compose(compose.transforms, additional_targets={
        't2_image': 'image',
        't2_mask': 'mask',
        'change': 'mask'
    })


def data_transform(T, data) -> Dict:
    if isinstance(T, dict):
        data = T[PRE](**data)

        if not isinstance(T[TWISE], A.NoOp):
            if 'mask' in data:
                t1_data = {'image': data['image'], 'mask': data['mask']}
            else:
                t1_data = {'image': data['image']}
            t1_data = T[TWISE](**t1_data)
            data.update(t1_data)

            if 't2_mask' in data:
                t2_data = {'image': data['t2_image'], 'mask': data['t2_mask']}
            else:
                t2_data = {'image': data['t2_image']}
            t2_data = T[TWISE](**t2_data)
            t2_data = {f't2_{k}': v for k, v in t2_data.items()}
            data.update(t2_data)

        data = T[POST](**data)
    else:
        data = T(data)
    return data


class BitemporalDataset(Dataset):
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
            PRE: A.NoOp(),
            TWISE: A.NoOp(),
            POST: A.NoOp(),
        }
        if isinstance(transform, A.Compose):
            self.t[POST] = to_bitemporal_compose(transform)
        elif isinstance(transform, dict):
            for k, v in transform.items():
                assert k in [PRE, TWISE, POST]
                if isinstance(v, A.Compose):
                    v = to_bitemporal_compose(v)
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

        data = data_transform(self.t, data)

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


@er.registry.DATASET.register()
class HFBitemporalDataset(er.ERDataset):
    def __init__(self, config):
        super().__init__(config)
        ds = []
        for s in self.cfg.splits:
            d = load_dataset(self.cfg.hf_repo_name, split=s)
            ds.append(d)
        hfd = concatenate_datasets(ds) if len(ds) > 1 else ds[0]
        self.hfd = hfd.with_format('numpy')

        transform = self.cfg.transform
        self.t = {
            PRE: A.NoOp(),
            TWISE: A.NoOp(),
            POST: A.NoOp(),
        }
        if isinstance(transform, A.Compose):
            self.t[POST] = to_bitemporal_compose(transform)
        elif isinstance(transform, dict):
            for k, v in transform.items():
                assert k in [PRE, TWISE, POST]
                if isinstance(v, A.Compose):
                    v = to_bitemporal_compose(v)
                self.t[k] = v
        else:
            self.t = transform

    def _slice_data(self, data, tile_slice):
        if tile_slice is None:
            return data

        x1, y1, x2, y2 = tile_slice
        return data[y1:y2, x1:x2]

    def compute_tile_slice(self, idx):
        return idx, None

    def __getitem__(self, idx):
        idx, tile_slice = self.compute_tile_slice(idx)

        example = self.hfd[idx]
        img1 = self._slice_data(example['t1_image'], tile_slice)
        img2 = self._slice_data(example['t2_image'], tile_slice)

        data = {
            'image': img1,
            't2_image': img2
        }

        if 't1_mask' in example:
            data['mask'] = self._slice_data(example['t1_mask'], tile_slice)

        if 't2_mask' in example:
            data['t2_mask'] = self._slice_data(example['t2_mask'], tile_slice)

        if 'change_mask' in example:
            data['change'] = self._slice_data(example['change_mask'], tile_slice)

        data = data_transform(self.t, data)

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
            image_filename=str(example['image_name'])
        )

        return img, ann

    def __len__(self):
        return len(self.hfd)

    def set_default_config(self):
        self.cfg.update(dict(
            hf_repo_name=None,
            splits=[],
            transform=None,
        ))
