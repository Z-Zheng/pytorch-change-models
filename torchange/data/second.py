# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ever as er
from torchange.data.bitemporal import BitemporalDataset
from pathlib import Path
import torch


@er.registry.DATASET.register()
class BinarySECOND(BitemporalDataset, er.ERDataset):
    def __init__(self, cfg):
        er.ERDataset.__init__(self, cfg)
        root_dir = Path(self.cfg.dataset_dir)
        img1_fps = [str(fp) for fp in (root_dir / 'im1').glob('*.png')]
        img2_fps = [fp.replace('im1', 'im2') for fp in img1_fps]
        label1_fps = [fp.replace('im1', 'label1_wo_cm') for fp in img1_fps]

        super().__init__(
            t1_image_fps=img1_fps,
            t2_image_fps=img2_fps,
            change_fps=label1_fps,
            transform=self.cfg.transforms
        )

    def __getitem__(self, idx):
        img, ann = super().__getitem__(idx)
        # to binary change mask
        ann['masks'][-1] = (ann['masks'][-1] > 0).float()
        return img, ann

    def set_default_config(self):
        self.config.update(dict(
            dataset_dir=None,
            transforms=None,
        ))


@er.registry.DATASET.register()
class SECOND(BitemporalDataset, er.ERDataset):
    def __init__(self, cfg):
        er.ERDataset.__init__(self, cfg)
        root_dir = Path(self.cfg.dataset_dir)
        img1_fps = [str(fp) for fp in (root_dir / 'im1').glob('*.png')]
        img2_fps = [fp.replace('im1', 'im2') for fp in img1_fps]
        label1_fps = [fp.replace('im1', 'label1_wo_cm') for fp in img1_fps]
        label2_fps = [fp.replace('im1', 'label2_wo_cm') for fp in img1_fps]

        super().__init__(
            t1_image_fps=img1_fps,
            t2_image_fps=img2_fps,
            t1_mask_fps=label1_fps,
            t2_mask_fps=label2_fps,
            transform=self.cfg.transforms
        )

    def __getitem__(self, idx):
        img, ann = super().__getitem__(idx)
        # append binary change mask
        ann['masks'].append((ann['masks'][0] > 0).float())
        # convert 0-6 to 255, 0-5, where 255 will be ignored.
        ann['masks'][0] = ann['masks'][0].to(torch.uint8) - 1
        ann['masks'][1] = ann['masks'][1].to(torch.uint8) - 1

        return img, ann

    def set_default_config(self):
        self.config.update(dict(
            dataset_dir=None,
            transforms=None,
        ))
