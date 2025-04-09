# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ever as er
from torchange.data.bitemporal import BitemporalDataset
import glob
import math
import os


@er.registry.DATASET.register()
class S2Looking(BitemporalDataset, er.ERDataset):
    def __init__(self, cfg):
        er.ERDataset.__init__(self, cfg)
        A_image_fps = sorted(glob.glob(os.path.join(self.cfg.dataset_dir, 'Image1', '*.png')))
        N = len(A_image_fps)
        if self.cfg.subsample_ratio < 1.0:
            split = math.floor(len(A_image_fps) * self.cfg.subsample_ratio)
            A_image_fps = A_image_fps[:split]

            A_image_fps = math.ceil(N / len(A_image_fps)) * A_image_fps
            A_image_fps = A_image_fps[:N]
            er.info(f'use subsample ratio of {self.cfg.subsample_ratio}, {split} training samples')

        B_image_fps = [fp.replace('Image1', 'Image2') for fp in A_image_fps]
        gt_fps = [fp.replace('Image1', 'label') for fp in A_image_fps]

        super().__init__(
            t1_image_fps=A_image_fps,
            t2_image_fps=B_image_fps,
            change_fps=gt_fps,
            transform=self.cfg.transforms
        )

    def set_default_config(self):
        self.cfg.update(dict(
            dataset_dir=None,
            transforms=None,
            subsample_ratio=1.0
        ))
