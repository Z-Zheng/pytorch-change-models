# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from skimage.segmentation import find_boundaries
from .segment_anything.utils.amg import (
    area_from_rle,
    box_xyxy_to_xywh,
    rle_to_mask,
    MaskData
)
import matplotlib.pyplot as plt
import numpy as np


def show_mask_data(mask_data, ax=None):
    assert isinstance(mask_data, MaskData)
    anns = []
    for idx in range(len(mask_data["rles"])):
        ann_i = {
            "segmentation": rle_to_mask(mask_data["rles"][idx]),
            "area": area_from_rle(mask_data["rles"][idx]),
        }
        if 'boxes' in mask_data._stats:
            ann_i['bbox'] = box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist()
        anns.append(ann_i)

    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    if ax is None:
        ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        boundary = find_boundaries(m)
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        color_boundary = np.array([0., 1., 1., 0.8])
        img[m] = color_mask
        img[boundary] = color_boundary

        if 'label' in ann:
            x, y, w, h = ann['bbox']
            ax.text(
                x + w / 2,
                y + h / 2,
                ann['label'],
                bbox={
                    'facecolor': 'black',
                    'alpha': 0.8,
                    'pad': 0.7,
                    'edgecolor': 'none'
                },
                color='red',
                fontsize=4,
                verticalalignment='top',
                horizontalalignment='left'
            )
    ax.imshow(img)


def show_change_masks(img1, img2, change_masks):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    axes[0].imshow(img1)
    show_mask_data(change_masks, axes[0])

    axes[1].imshow(img2)
    show_mask_data(change_masks, axes[1])

    axes[2].imshow(255 * np.ones_like(img1))
    show_mask_data(change_masks, axes[2])
    for ax in axes:
        ax.axis('off')

    return fig, axes
