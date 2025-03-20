# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries
from skimage.morphology import binary_opening, dilation, square
import numpy as np
import random

MAXIMUM_TRY = 50


class LC:
    Bareland = 1
    Rangeland = 2
    DevelopedSpace = 3
    Road = 4
    Tree = 5
    Water = 6
    AgricultureLand = 7
    Building = 8


# OEM
OEM_Transition = [
    [i + 1 for i in range(8)],
    [LC.Rangeland, LC.DevelopedSpace, LC.Tree, LC.Water, LC.AgricultureLand],
    [LC.Bareland, LC.DevelopedSpace, LC.Tree, LC.Water, LC.AgricultureLand],
    [LC.Bareland, LC.Rangeland, LC.Tree, LC.Water, LC.AgricultureLand],
    [LC.Bareland, LC.DevelopedSpace, LC.Tree, LC.Water],
    [LC.Bareland, LC.Rangeland, LC.DevelopedSpace, LC.Water, LC.AgricultureLand],
    [LC.Bareland, LC.Rangeland, LC.DevelopedSpace, LC.Tree, LC.AgricultureLand],
    [LC.Bareland, LC.Rangeland, LC.DevelopedSpace, LC.Tree, LC.Water, LC.AgricultureLand],
    [LC.DevelopedSpace, LC.Tree, LC.Water]
]


def object_proposal(mask):
    mask = (mask > 0).astype(np.uint8, copy=False)
    props = regionprops(label(mask))
    return props


def add_object(obj_mask, max_add_num_per_frame, min_add_num_per_frame=1):
    h, w = obj_mask.shape
    props = object_proposal(obj_mask)
    props = [p for p in props if p.area > 8 * 8]
    num_objs = random.randint(min_add_num_per_frame, max_add_num_per_frame)

    random.shuffle(props)
    props = props[:num_objs]

    new_obj_mask = (obj_mask > 0).astype(np.uint8)

    for obj in props:
        rr, cc = obj.coords.T

        ymin, xmin, ymax, xmax = obj.bbox

        for _ in range(MAXIMUM_TRY):
            # [-ymin, h - ymax)
            yscale = (h - ymax) + ymin
            yshift = -ymin
            yoffset = int(np.random.rand() * yscale + yshift)
            # [-xmin, w - xmax)
            xscale = (w - xmax) + xmin
            xshift = -xmin
            xoffset = int(np.random.rand() * xscale + xshift)

            candidate = new_obj_mask[rr + yoffset, cc + xoffset]
            if np.sum(candidate) == 0:
                new_obj_mask[rr + yoffset, cc + xoffset] = 1
                break

    return new_obj_mask


def remove_object(obj_mask, max_rm_num_per_frame, min_rm_num_per_frame=1):
    props = object_proposal(obj_mask)

    props = [p for p in props if p.area > 8 * 8]

    num_objs = random.randint(min_rm_num_per_frame, max_rm_num_per_frame)
    num_objs = min(num_objs, len(props))

    random.shuffle(props)
    props = props[:num_objs]

    obj_mask = obj_mask.copy()

    for obj in props:
        rr, cc = obj.coords.T
        obj_mask[rr, cc] = 0

    return obj_mask


def remove_add_object(obj_mask, max_change_num_per_frame):
    obj_mask = remove_object(obj_mask, max_change_num_per_frame)
    obj_mask = add_object(obj_mask, max_change_num_per_frame)
    return obj_mask


def random_transition(mask, num_classes, transition_kernel=None, p=0.3):
    if transition_kernel is None:
        transition_kernel = OEM_Transition
    eye = np.eye(num_classes)
    bin_masks = eye[mask]

    canvas = np.zeros_like(mask, dtype=np.int64)
    for i in range(num_classes):
        mask = bin_masks[:, :, i]
        if (mask == 0).all():
            continue
        props = object_proposal(mask)
        props = [obj for obj in props if obj.area > 8 * 8]
        for obj in props:
            rr, cc = obj.coords.T
            if random.random() < p:
                canvas[rr, cc] = random.choice(transition_kernel[i])
            else:
                canvas[rr, cc] = i
    return canvas


# mainly for SAM masks
def remove_instance(ins_mask, p=0.1):
    ins_mask = np.copy(ins_mask)
    for i in np.unique(ins_mask):
        if i == 0:
            continue
        if random.random() < p:
            ins_mask[ins_mask == i] = 0
    return ins_mask


# Changen2, Sec 3.5, Fig.7
def next_time_contour_gen(t1_mask, t2_mask):
    # compute change mask
    cmsk = ((t1_mask > 0) != (t2_mask > 0)).astype(np.uint8)
    cmsk = binary_opening(cmsk).astype(np.uint8)
    # compute t2 boundary
    bd1 = find_boundaries(t1_mask).astype(np.uint8)
    _cmsk = dilation(cmsk, square(3))
    bd2 = bd1 * (1 - _cmsk)
    return bd2


def generate_mask_seq(mask, seq_len=6, max_change_num_per_frame=5, mode='remove', seed=None, min_change_num_per_frame=1):
    random.seed(seed)
    if mode == 'remove':
        ds = [mask]
        for _ in range(seq_len - 1):
            ds.append(remove_object(ds[-1], max_change_num_per_frame, min_rm_num_per_frame=min_change_num_per_frame))
    elif mode == 'add':
        ds = [mask]
        for _ in range(seq_len - 1):
            ds.append(add_object(ds[-1], max_change_num_per_frame, min_add_num_per_frame=min_change_num_per_frame))
    elif mode == 'mix':
        ds = [mask]
        for _ in range(seq_len - 1):
            ds.append(remove_add_object(ds[-1], max_change_num_per_frame))
    else:
        raise NotImplementedError
    return ds
