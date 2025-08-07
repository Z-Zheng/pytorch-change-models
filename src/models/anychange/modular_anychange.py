# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

# Import SAM components from libs
from libs.segment_any.segment_anything.utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)


class ModularAnyChange:
    """
    Modular Any Change implementation for zero-shot change detection.
    
    This class provides a modular interface for using SAM-based change detection
    with various mask generation strategies and bitemporal matching.
    """
    
    def __init__(
            self,
            model,
            points_per_side=32,
            points_per_batch: int = 64,
            pred_iou_thresh: float = 0.5,
            stability_score_thresh: float = 0.95,
            stability_score_offset: float = 1.0,
            box_nms_thresh: float = 0.7,
            point_grids=None,
            min_mask_region_area: int = 0,
    ):
        """
        Initialize the modular any change model.
        
        Args:
            model: SAM model instance
            points_per_side: Number of points per side for mask generation
            points_per_batch: Number of points per batch
            pred_iou_thresh: IoU threshold for mask prediction
            stability_score_thresh: Stability score threshold
            stability_score_offset: Stability score offset
            box_nms_thresh: Box NMS threshold
            point_grids: Pre-computed point grids
            min_mask_region_area: Minimum mask region area
        """
        self.model = model
        self.predictor = model.predictor if hasattr(model, 'predictor') else None
        self.points_per_side = points_per_side
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.min_mask_region_area = min_mask_region_area
        
        if point_grids is None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                0,  # crop_n_layers
                1,  # crop_n_points_downscale_factor
            )
        else:
            self.point_grids = point_grids

    @torch.no_grad()
    def image_encoder(self, image):
        """
        Extract image embeddings using SAM.
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Dictionary containing image embedding and metadata
        """
        if self.predictor is None:
            # Create a simple predictor if not available
            from libs.segment_any.segment_anything.predictor import SamPredictor
            self.predictor = SamPredictor(self.model)
        
        self.predictor.set_image(image)
        return {
            'image_embedding': self.predictor.get_image_embedding(),
            'original_size': image.shape[:2],
            'input_size': self.predictor.input_size,
        }

    @torch.no_grad()
    def generate_with_image_embedding(self, image_embedding, original_size):
        """
        Generate masks using pre-computed image embedding.
        
        Args:
            image_embedding: Pre-computed image embedding
            original_size: Original image size (height, width)
            
        Returns:
            MaskData containing generated masks
        """
        h, w = original_size
        points_scale = np.array([w, h])
        points_for_image = self.point_grids[0] * points_scale
        
        data = MaskData()
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(
                self.predictor,
                image_embedding,
                points,
                (h, w),
                [0, 0, w, h],  # crop_box
                (h, w),  # orig_size
            )
            data.cat(batch_data)
            del batch_data
        
        # Remove duplicates
        keep_by_nms = torch.ops.torchvision.nms(
            data["boxes"].float(),
            data["iou_preds"],
            self.box_nms_thresh,
        )
        data.filter(keep_by_nms)
        
        return data

    @torch.no_grad()
    def generate_with_points(self, image, points):
        """
        Generate masks from specific points.
        
        Args:
            image: Input image (numpy array)
            points: Point coordinates (numpy array)
            
        Returns:
            Tuple of (masks, iou_predictions)
        """
        if self.predictor is None:
            from libs.segment_any.segment_anything.predictor import SamPredictor
            self.predictor = SamPredictor(self.model)
        
        self.predictor.set_image(image)
        masks, iou_preds, _ = self.predictor.predict_torch(
            torch.as_tensor(points, device=self.predictor.device),
            torch.ones(len(points), dtype=torch.int, device=self.predictor.device),
            multimask_output=True,
            return_logits=True,
        )
        
        return masks, iou_preds

    @torch.no_grad()
    def embedding_point_to_mask(self, image_embedding, original_size, points):
        """
        Generate masks from points using pre-computed embedding.
        
        Args:
            image_embedding: Pre-computed image embedding
            original_size: Original image size (height, width)
            points: Point coordinates
            
        Returns:
            MaskData containing generated masks
        """
        h, w = original_size
        points = torch.as_tensor(points, device=self.predictor.device)
        labels = torch.ones(len(points), dtype=torch.int, device=self.predictor.device)
        
        masks, iou_preds, _ = self.predictor.predict_torch(
            points,
            labels,
            multimask_output=True,
            return_logits=True,
        )
        
        # Convert to MaskData format
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=points.repeat(masks.shape[1], 1),
        )
        
        # Filter by predicted IoU
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            data.filter(keep_mask)
        
        # Calculate stability score
        data["stability_score"] = calculate_stability_score(
            data["masks"], self.model.mask_threshold, self.stability_score_offset
        )
        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            data.filter(keep_mask)
        
        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.model.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])
        
        # Compress to RLE
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]
        
        return data

    @torch.no_grad()
    def generate(self, image, mask_output_mode='rle'):
        """
        Generate masks for the entire image.
        
        Args:
            image: Input image (numpy array)
            mask_output_mode: Output mode ('rle', 'coco_rle', 'binary_mask')
            
        Returns:
            MaskData containing generated masks
        """
        if self.predictor is None:
            from libs.segment_any.segment_anything.predictor import SamPredictor
            self.predictor = SamPredictor(self.model)
        
        self.predictor.set_image(image)
        image_embedding = self.predictor.get_image_embedding()
        
        data = self.generate_with_image_embedding(image_embedding, image.shape[:2])
        
        # Encode masks
        if mask_output_mode == "coco_rle":
            data["segmentations"] = [coco_encode_rle(rle) for rle in data["rles"]]
        elif mask_output_mode == "binary_mask":
            data["segmentations"] = [rle_to_mask(rle) for rle in data["rles"]]
        else:
            data["segmentations"] = data["rles"]
        
        return data

    def _process_batch(
            self,
            predictor,
            image_embedding,
            points: np.ndarray,
            im_size,
            crop_box,
            orig_size,
    ) -> MaskData:
        """
        Process a batch of points to generate masks.
        
        Args:
            predictor: SAM predictor instance
            image_embedding: Pre-computed image embedding
            points: Point coordinates
            im_size: Image size
            crop_box: Crop box coordinates
            orig_size: Original image size
            
        Returns:
            MaskData containing processed masks
        """
        orig_h, orig_w = orig_size
        
        # Run model on this batch
        transformed_points = predictor.transform.apply_coords(points, im_size)
        in_points = torch.as_tensor(transformed_points, device=predictor.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        
        masks, iou_preds, _ = predictor.predict_torch(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=True,
            return_logits=True,
        )
        
        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
        )
        del masks
        
        # Filter by predicted IoU
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            data.filter(keep_mask)
        
        # Calculate stability score
        data["stability_score"] = calculate_stability_score(
            data["masks"], self.model.mask_threshold, self.stability_score_offset
        )
        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            data.filter(keep_mask)
        
        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.model.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])
        
        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
        if not torch.all(keep_mask):
            data.filter(keep_mask)
        
        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]
        
        return data

    def set_hyperparameters(self, **kwargs):
        """
        Set hyperparameters for the model.
        
        Args:
            **kwargs: Hyperparameters to set
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown hyperparameter '{key}'")

    def get_hyperparameters(self):
        """
        Get current hyperparameters.
        
        Returns:
            Dictionary of current hyperparameters
        """
        return {
            'points_per_side': self.points_per_side,
            'points_per_batch': self.points_per_batch,
            'pred_iou_thresh': self.pred_iou_thresh,
            'stability_score_thresh': self.stability_score_thresh,
            'stability_score_offset': self.stability_score_offset,
            'box_nms_thresh': self.box_nms_thresh,
            'min_mask_region_area': self.min_mask_region_area,
        } 