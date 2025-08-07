# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Dict, List, Optional, Union

from transformers import PretrainedConfig


class AnyChangeConfig(PretrainedConfig):
    """
    Configuration class for AnyChange model.
    
    Args:
        model_type (str): Type of SAM model to use ('vit_b', 'vit_l', 'vit_h') (default: 'vit_b')
        sam_checkpoint (str): Path to SAM checkpoint file (default: './sam_weights/sam_vit_b_01ec64.pth')
        points_per_side (int): Number of points per side for mask generation (default: 32)
        points_per_batch (int): Number of points per batch (default: 64)
        pred_iou_thresh (float): IoU threshold for mask prediction (default: 0.5)
        stability_score_thresh (float): Stability score threshold (default: 0.95)
        stability_score_offset (float): Stability score offset (default: 1.0)
        box_nms_thresh (float): Box NMS threshold (default: 0.7)
        min_mask_region_area (int): Minimum mask region area (default: 0)
        change_confidence_threshold (float): Change confidence threshold in degrees (default: 155)
        auto_threshold (bool): Whether to use automatic thresholding (default: False)
        use_normalized_feature (bool): Whether to use normalized features (default: True)
        area_thresh (float): Area threshold for mask filtering (default: 0.8)
        match_hist (bool): Whether to match histograms (default: False)
        object_sim_thresh (float): Object similarity threshold in degrees (default: 60)
        bitemporal_match (bool): Whether to use bitemporal matching (default: True)
        num_change_classes (int): Number of change classes (default: 1)
        loss_config (dict): Loss configuration
        **kwargs: Additional arguments
    """
    
    model_type = "anychange"
    
    def __init__(
        self,
        model_type: str = "vit_b",
        sam_checkpoint: str = "./sam_weights/sam_vit_b_01ec64.pth",
        points_per_side: int = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.5,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        min_mask_region_area: int = 0,
        change_confidence_threshold: float = 155,
        auto_threshold: bool = False,
        use_normalized_feature: bool = True,
        area_thresh: float = 0.8,
        match_hist: bool = False,
        object_sim_thresh: float = 60,
        bitemporal_match: bool = True,
        num_change_classes: int = 1,
        loss_config: Dict = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.model_type = model_type
        self.sam_checkpoint = sam_checkpoint
        self.points_per_side = points_per_side
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.min_mask_region_area = min_mask_region_area
        self.change_confidence_threshold = change_confidence_threshold
        self.auto_threshold = auto_threshold
        self.use_normalized_feature = use_normalized_feature
        self.area_thresh = area_thresh
        self.match_hist = match_hist
        self.object_sim_thresh = object_sim_thresh
        self.bitemporal_match = bitemporal_match
        self.num_change_classes = num_change_classes
        self.loss_config = loss_config or {}
        
    def to_dict(self):
        """Convert config to dictionary."""
        output = super().to_dict()
        output.update({
            "model_type": self.model_type,
            "sam_checkpoint": self.sam_checkpoint,
            "points_per_side": self.points_per_side,
            "points_per_batch": self.points_per_batch,
            "pred_iou_thresh": self.pred_iou_thresh,
            "stability_score_thresh": self.stability_score_thresh,
            "stability_score_offset": self.stability_score_offset,
            "box_nms_thresh": self.box_nms_thresh,
            "min_mask_region_area": self.min_mask_region_area,
            "change_confidence_threshold": self.change_confidence_threshold,
            "auto_threshold": self.auto_threshold,
            "use_normalized_feature": self.use_normalized_feature,
            "area_thresh": self.area_thresh,
            "match_hist": self.match_hist,
            "object_sim_thresh": self.object_sim_thresh,
            "bitemporal_match": self.bitemporal_match,
            "num_change_classes": self.num_change_classes,
            "loss_config": self.loss_config,
        })
        return output 