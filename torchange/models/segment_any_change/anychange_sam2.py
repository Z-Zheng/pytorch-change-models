import numpy as np
import torch

import torch.nn.functional as F
from torchange.models.segment_any_change.segment_anything.utils.amg import rle_to_mask, MaskData
import copy
from skimage.filters.thresholding import threshold_otsu
from torchange.models.segment_any_change.base_sam2 import SegmentAnyChange, build_sam2, SAM2AutomaticMaskGenerator
import math
from torchvision.ops.boxes import batched_nms


def angle2cosine(a):
    assert 0 <= a <= 180
    return math.cos(a / 180 * math.pi)


def cosine2angle(c):
    assert -1 <= c <= 1
    return math.acos(c) * 180 / math.pi


class AnyChange_sam2(SegmentAnyChange):
    def __init__(self, model_cfg="configs/sam2.1/sam2.1_hiera_t.yaml",
                 sam2_checkpoint="/slow_disk/ccl/codes/pytorch-change-models/sam2.1_hiera_tiny.pt"):
        device = device = torch.device("cuda")
        sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.sam2 = sam2.to(self.device)

        self.mask_generator = SAM2AutomaticMaskGenerator(sam2)
        # self.mask_generator = SAM2AutomaticMaskGenerator(sam2,
        #                                             points_per_side=64,
        #                                             points_per_batch=128,
        #                                             pred_iou_thresh=0.7,
        #                                             stability_score_thresh=0.92,
        #                                             stability_score_offset=0.7,
        #                                             crop_n_layers=1,
        #                                             box_nms_thresh=0.7,
        #                                             crop_n_points_downscale_factor=2,
        #                                             min_mask_region_area=25,
        #                                             use_m2m=False,
        #                                             )
        # self.mask_generator = SAM2AutomaticMaskGenerator(sam2)

        self.set_hyperparameters()

        self.embed_data1 = None
        self.embed_data2 = None

        # layernorm = self.sam.image_encoder.neck[3]
        # w = layernorm.weight.data
        # b = layernorm.bias.data
        # w = w.reshape(w.size(0), 1, 1)
        # b = b.reshape(b.size(0), 1, 1)
        #
        # self.inv_transform = lambda e: (e - b) / w
