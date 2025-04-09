# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np
from pathlib import Path
import ever as er

from torchange.data.bitemporal import HFBitemporalDataset, BitemporalDataset, data_transform
from skimage.io import imread
import torch
from tqdm import tqdm


@er.registry.DATASET.register()
class xView2(BitemporalDataset, er.ERDataset):
    def __init__(self, cfg):
        er.ERDataset.__init__(self, cfg)
        dataset_dir = self.cfg.dataset_dir
        pre_img_fps, post_img_fps, pre_gt_fps, post_gt_fps = [], [], [], []
        splits = []
        if isinstance(dataset_dir, str):
            splits.append(Path(dataset_dir).name)
            pre_img_fps, post_img_fps, pre_gt_fps, post_gt_fps = self.parse_dataset_dir(self.cfg.dataset_dir)
        elif isinstance(dataset_dir, (tuple, list)):
            for dd in dataset_dir:
                splits.append(Path(dd).name)
                a, b, c, d = self.parse_dataset_dir(dd)
                pre_img_fps += a
                post_img_fps += b
                pre_gt_fps += c
                post_gt_fps += d
        else:
            raise ValueError

        if self.cfg.training:
            self.tiles = er.sliding_window((1024, 1024), self.cfg.crop_size, self.cfg.stride)
        else:
            self.tiles = er.sliding_window((1024, 1024), 1024, 1024)

        if self.cfg.training:
            split_name = '_'.join(splits)
            indices_file = Path(os.curdir) / f'xView2_{split_name}_valid_indices_p{self.cfg.crop_size}_s{self.cfg.stride}.npy'
            if indices_file.exists():
                self.valid_patch_indices = np.load(str(indices_file))
            else:
                valid = np.ones([len(post_gt_fps) * self.tiles.shape[0]], dtype=np.uint8)
                for img_idx in tqdm(range(len(post_gt_fps)), disable=not er.dist.is_main_process()):
                    t2_mask = imread(post_gt_fps[img_idx]).astype(np.float32)
                    for tile_idx in range(self.tiles.shape[0]):
                        x1, y1, x2, y2 = self.tiles[tile_idx]
                        sub = t2_mask[y1:y2, x1:x2]
                        sub[sub == 255] = 0
                        if np.sum(sub) == 0:
                            valid[img_idx * self.tiles.shape[0] + tile_idx] = 0
                self.valid_patch_indices = np.nonzero(valid.astype(bool))[0]
                np.save(str(indices_file), self.valid_patch_indices)
        else:
            self.valid_patch_indices = np.arange(len(post_gt_fps))

        super().__init__(
            t1_image_fps=pre_img_fps,
            t2_image_fps=post_img_fps,
            t1_mask_fps=pre_gt_fps,
            t2_mask_fps=post_gt_fps,
            transform=self.cfg.transforms,
            name_checker=lambda x, y: True,
        )

    def parse_dataset_dir(self, dataset_dir):
        dataset_dir = Path(dataset_dir)
        img_dir = dataset_dir / 'images'
        tgt_dir = dataset_dir / 'targets'

        pre_gt_fps = list(tgt_dir.glob('*_pre_*.png'))
        post_gt_fps = [tgt_dir / fp.name.replace('pre', 'post') for fp in pre_gt_fps]

        pre_img_fps = [img_dir / fp.name.replace('_target.png', '.png') for fp in pre_gt_fps]
        post_img_fps = [img_dir / fp.name.replace('_target.png', '.png') for fp in post_gt_fps]

        return pre_img_fps, post_img_fps, pre_gt_fps, post_gt_fps

    def __getitem__(self, idx):
        idx = self.valid_patch_indices[idx]
        img_idx = idx // self.tiles.shape[0]
        tile_idx = idx % self.tiles.shape[0]
        x1, y1, x2, y2 = self.tiles[tile_idx]

        img1 = imread(self.t1_image_fps[img_idx])[y1:y2, x1:x2]
        img2 = imread(self.t2_image_fps[img_idx])[y1:y2, x1:x2]

        data = {
            'image': img1,
            't2_image': img2
        }

        if self.t1_mask_fps:
            msk1 = imread(self.t1_mask_fps[img_idx])[y1:y2, x1:x2]
            data['mask'] = msk1

        if self.t2_mask_fps:
            msk2 = imread(self.t2_mask_fps[img_idx])[y1:y2, x1:x2]
            if self.cfg.ignore_t2_bg:
                msk2[msk2 == 0] = 255
            data['t2_mask'] = msk2

        data = data_transform(self.t, data)

        img = torch.cat([data['image'], data['t2_image']], dim=0)

        masks = []
        if 'mask' in data:
            masks.append(data['mask'])

        if 't2_mask' in data:
            masks.append(data['t2_mask'])

        ann = dict(
            masks=masks,
            image_filename=str(Path(self.t1_image_fps[img_idx]).name)
        )
        return img, ann

    def __len__(self):
        return self.valid_patch_indices.shape[0]

    def set_default_config(self):
        self.config.update(dict(
            dataset_dir=None,
            crop_size=512,
            stride=256,
            training=True,
            ignore_t2_bg=False,
        ))


@er.registry.DATASET.register()
class HFxView2(HFBitemporalDataset):
    def __init__(self, cfg):
        super().__init__(cfg)
        if self.cfg.training:
            self.tiles = er.sliding_window((1024, 1024), self.cfg.crop_size, self.cfg.stride)
        else:
            self.tiles = er.sliding_window((1024, 1024), 1024, 1024)

        self.build_index()

    def build_index(self):
        if self.cfg.training:
            split_name = '_'.join(self.cfg.splits)
            basename = f'HFxView2_{split_name}_valid_indices_p{self.cfg.crop_size}_s{self.cfg.stride}.npy'
            indices_file = Path(os.curdir) / basename
            if indices_file.exists():
                self.valid_patch_indices = np.load(str(indices_file))
            else:
                valid = np.ones([len(self.hfd) * self.tiles.shape[0]], dtype=np.uint8)
                t2_masks = self.hfd['t2_mask']
                for img_idx in tqdm(range(len(self.hfd)), disable=not er.dist.is_main_process()):
                    t2_mask = np.array(t2_masks[img_idx]).astype(np.float32)
                    for tile_idx in range(self.tiles.shape[0]):
                        x1, y1, x2, y2 = self.tiles[tile_idx]
                        sub = t2_mask[y1:y2, x1:x2]
                        sub[sub == 255] = 0
                        if np.sum(sub) == 0:
                            valid[img_idx * self.tiles.shape[0] + tile_idx] = 0
                self.valid_patch_indices = np.nonzero(valid.astype(bool))[0]
                np.save(str(indices_file), self.valid_patch_indices)
        else:
            self.valid_patch_indices = np.arange(len(self.hfd))

    def compute_tile_slice(self, idx):
        idx = self.valid_patch_indices[idx]
        img_idx = idx // self.tiles.shape[0]
        tile_idx = idx % self.tiles.shape[0]
        return int(img_idx), self.tiles[tile_idx]

    def __len__(self):
        return self.valid_patch_indices.shape[0]

    def set_default_config(self):
        super().set_default_config()
        self.cfg.update(dict(
            hf_repo_name='EVER-Z/torchange_xView2',
            crop_size=512,
            stride=256,
            training=True,
        ))
