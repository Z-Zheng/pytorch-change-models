# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import math
import copy
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import PreTrainedModel
from torch.cuda.amp import autocast
from torchvision.ops.boxes import batched_nms
from skimage.filters.thresholding import threshold_otsu
from skimage.exposure import match_histograms
from safetensors.torch import load_file

from .configuration_anychange import AnyChangeConfig

# Import SAM components from libs
try:
    from libs.segment_any.segment_anything import sam_model_registry
    from libs.segment_any.segment_anything.utils.amg import MaskData, rle_to_mask
    from .modular_anychange import ModularAnyChange
except ImportError:
    print("SAM components not found. Please ensure SAM is properly installed in libs/segment_any/segment_anything.")

CHANGE = 'change_prediction'


def angle2cosine(a):
    """Convert angle to cosine value."""
    assert 0 <= a <= 180
    return math.cos(a / 180 * math.pi)


def cosine2angle(c):
    """Convert cosine value to angle."""
    assert -1 <= c <= 1
    return math.acos(c) * 180 / math.pi


class AnyChangeModel(PreTrainedModel):
    """
    AnyChange model for zero-shot change detection using SAM.
    
    This model uses the Segment Anything Model (SAM) with bitemporal latent matching
    to perform zero-shot change detection without requiring training on change data.
    """
    
    config_class = AnyChangeConfig
    base_model_prefix = "anychange"
    supports_gradient_checkpointing = True
    
    def __init__(self, config: AnyChangeConfig):
        super().__init__(config)
        
        # Initialize SAM model
        self.sam = sam_model_registry[config.model_type](checkpoint=config.sam_checkpoint)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sam = self.sam.to(self.device)
        
        # Initialize mask generator
        self.maskgen = ModularAnyChange(
            self.sam,
            points_per_side=config.points_per_side,
            points_per_batch=config.points_per_batch,
            pred_iou_thresh=config.pred_iou_thresh,
            stability_score_thresh=config.stability_score_thresh,
            stability_score_offset=config.stability_score_offset,
            box_nms_thresh=config.box_nms_thresh,
            min_mask_region_area=config.min_mask_region_area,
        )
        
        # Set hyperparameters
        self.change_confidence_threshold = config.change_confidence_threshold
        self.auto_threshold = config.auto_threshold
        self.use_normalized_feature = config.use_normalized_feature
        self.area_thresh = config.area_thresh
        self.match_hist = config.match_hist
        self.object_sim_thresh = config.object_sim_thresh
        self.use_bitemporal_match = config.bitemporal_match
        
        # Initialize layer normalization inverse transform
        layernorm = self.sam.image_encoder.neck[3]
        w = layernorm.weight.data
        b = layernorm.bias.data
        w = w.reshape(w.size(0), 1, 1)
        b = b.reshape(b.size(0), 1, 1)
        self.inv_transform = lambda e: (e - b) / w
        
        # Cache for embeddings
        self.embed_data1 = None
        self.embed_data2 = None
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights."""
        # SAM weights are loaded from checkpoint, no additional initialization needed
        pass
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[AnyChangeConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: Optional[bool] = None,
        weights_only: bool = True,
        **kwargs,
    ):
        """
        Load a pretrained AnyChange model.
        
        Args:
            pretrained_model_name_or_path: Path to pretrained model or model identifier
            config: Model configuration
            cache_dir: Directory to cache downloaded models
            ignore_mismatched_sizes: Whether to ignore mismatched sizes
            force_download: Whether to force download
            local_files_only: Whether to use only local files
            token: HuggingFace token for private models
            revision: Model revision
            use_safetensors: Whether to use safetensors
            weights_only: Whether to load only weights
            **kwargs: Additional arguments
            
        Returns:
            Loaded AnyChange model
        """
        # Load config if not provided
        if config is None:
            config = cls.config_class.from_pretrained(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                **kwargs
            )
        
        # Create model instance
        model = cls(config, *model_args, **kwargs)
        
        # Load SAM weights if they exist in the checkpoint
        if pretrained_model_name_or_path is not None:
            # Check if SAM weights are included in the checkpoint
            sam_weights_path = os.path.join(pretrained_model_name_or_path, "sam_weights")
            if os.path.exists(sam_weights_path):
                # Load SAM weights from the checkpoint
                sam_checkpoint_path = os.path.join(sam_weights_path, f"sam_{config.model_type}_01ec64.pth")
                if os.path.exists(sam_checkpoint_path):
                    # Reload SAM with the checkpoint weights
                    model.sam = sam_model_registry[config.model_type](checkpoint=sam_checkpoint_path)
                    model.sam = model.sam.to(model.device)
                    
                    # Reinitialize mask generator with new SAM
                    model.maskgen = ModularAnyChange(
                        model.sam,
                        points_per_side=config.points_per_side,
                        points_per_batch=config.points_per_batch,
                        pred_iou_thresh=config.pred_iou_thresh,
                        stability_score_thresh=config.stability_score_thresh,
                        stability_score_offset=config.stability_score_offset,
                        box_nms_thresh=config.box_nms_thresh,
                        min_mask_region_area=config.min_mask_region_area,
                    )
                    
                    # Reinitialize layer normalization inverse transform
                    layernorm = model.sam.image_encoder.neck[3]
                    w = layernorm.weight.data
                    b = layernorm.bias.data
                    w = w.reshape(w.size(0), 1, 1)
                    b = b.reshape(b.size(0), 1, 1)
                    model.inv_transform = lambda e: (e - b) / w
        
        return model
    
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
        """
        Save the AnyChange model to a directory.
        
        Args:
            save_directory: Directory to save the model
            is_main_process: Whether this is the main process
            state_dict: State dictionary to save
            save_function: Function to use for saving
            push_to_hub: Whether to push to HuggingFace Hub
            max_shard_size: Maximum shard size
            safe_serialization: Whether to use safe serialization
            variant: Model variant
            token: HuggingFace token
            save_peft_format: Whether to save in PEFT format
            **kwargs: Additional arguments
        """
        # Create save directory
        os.makedirs(save_directory, exist_ok=True)
        
        # Save configuration
        self.config.save_pretrained(save_directory)
        
        # Save SAM weights if they exist
        sam_weights_dir = os.path.join(save_directory, "sam_weights")
        os.makedirs(sam_weights_dir, exist_ok=True)
        
        # Copy SAM checkpoint if it exists
        if hasattr(self.config, 'sam_checkpoint') and os.path.exists(self.config.sam_checkpoint):
            import shutil
            sam_filename = f"sam_{self.config.model_type}_01ec64.pth"
            sam_dest_path = os.path.join(sam_weights_dir, sam_filename)
            shutil.copy2(self.config.sam_checkpoint, sam_dest_path)
            
            # Update config to point to the saved SAM weights
            self.config.sam_checkpoint = sam_dest_path
        
        # Save model state dict (if any additional weights exist)
        if state_dict is None:
            state_dict = self.state_dict()
        
        # Only save if there are actual model weights (not just SAM)
        if state_dict:
            # Use parent's save_pretrained for the actual model weights
            super().save_pretrained(
                save_directory=save_directory,
                is_main_process=is_main_process,
                state_dict=state_dict,
                save_function=save_function,
                push_to_hub=push_to_hub,
                max_shard_size=max_shard_size,
                safe_serialization=safe_serialization,
                variant=variant,
                token=token,
                save_peft_format=save_peft_format,
                **kwargs
            )
    
    def set_hyperparameters(
        self,
        change_confidence_threshold: float = None,
        auto_threshold: bool = None,
        use_normalized_feature: bool = None,
        area_thresh: float = None,
        match_hist: bool = None,
        object_sim_thresh: float = None,
        bitemporal_match: bool = None,
    ):
        """Set hyperparameters for the model."""
        if change_confidence_threshold is not None:
            self.change_confidence_threshold = change_confidence_threshold
        if auto_threshold is not None:
            self.auto_threshold = auto_threshold
        if use_normalized_feature is not None:
            self.use_normalized_feature = use_normalized_feature
        if area_thresh is not None:
            self.area_thresh = area_thresh
        if match_hist is not None:
            self.match_hist = match_hist
        if object_sim_thresh is not None:
            self.object_sim_thresh = object_sim_thresh
        if bitemporal_match is not None:
            self.use_bitemporal_match = bitemporal_match
    
    def make_mask_generator(self, **kwargs):
        """Customize the mask generator parameters."""
        self.maskgen = ModularAnyChange(self.sam, **kwargs)
    
    def extract_image_embedding(self, img1, img2):
        """Extract image embeddings for both temporal images."""
        self.embed_data1 = self.maskgen.image_encoder(img1)
        self.embed_data2 = self.maskgen.image_encoder(img2)
        return self.embed_data1, self.embed_data2
    
    def set_cached_embedding(self, embedding):
        """Set cached embeddings for faster inference."""
        data = embedding
        oh, ow = data['original_size'].numpy()
        h, w = data['input_size']
        self.embed_data1 = {
            'image_embedding': data['t1'].to(self.device),
            'original_size': (oh, ow),
        }
        self.embed_data2 = {
            'image_embedding': data['t2'].to(self.device),
            'original_size': (oh, ow),
        }
        self.maskgen.predictor.input_size = (h, w)
        self.maskgen.predictor.original_size = (oh, ow)
    
    def load_cached_embedding(self, filepath):
        """Load cached embeddings from file."""
        data = load_file(filepath, device='cpu')
        self.set_cached_embedding(data)
    
    def clear_cached_embedding(self):
        """Clear cached embeddings."""
        self.embed_data1 = None
        self.embed_data2 = None
        self.maskgen.predictor.input_size = None
        self.maskgen.predictor.original_size = None
    
    def proposal(self, img1, img2):
        """Generate mask proposals for both temporal images."""
        h, w = img1.shape[:2]
        if self.embed_data1 is None:
            self.extract_image_embedding(img1, img2)
        
        mask_data1 = self.maskgen.generate_with_image_embedding(**self.embed_data1)
        mask_data2 = self.maskgen.generate_with_image_embedding(**self.embed_data2)
        mask_data1.filter((mask_data1['areas'] / (h * w)) < self.area_thresh)
        mask_data2.filter((mask_data2['areas'] / (h * w)) < self.area_thresh)
        
        return {
            't1_mask_data': mask_data1,
            't1_image_embedding': self.embed_data1['image_embedding'],
            't2_mask_data': mask_data2,
            't2_image_embedding': self.embed_data2['image_embedding'],
        }
    
    def bitemporal_match(self, t1_mask_data, t1_image_embedding, t2_mask_data, t2_image_embedding) -> MaskData:
        """Perform bitemporal matching to find change regions."""
        t1_img_embed = t1_image_embedding
        t2_img_embed = t2_image_embedding
        h, w = self.embed_data1['original_size']
        
        seq_img_embed = [t1_img_embed, t2_img_embed]
        seq_img_embed_data = [{'image_embedding': img_embed,
                               'original_size': self.embed_data1['original_size']}
                              for img_embed in seq_img_embed]
        
        seq_mask_data = [t1_mask_data, ]
        for img_embed_data in seq_img_embed_data[1:-1]:
            mask_data = self.maskgen.generate_with_image_embedding(**img_embed_data)
            mask_data.filter((mask_data['areas'] / (h * w)) < self.area_thresh)
            seq_mask_data.append(mask_data)
        
        seq_mask_data.append(t2_mask_data)
        
        if self.use_normalized_feature:
            t1_img_embed = self.inv_transform(t1_img_embed)
            t2_img_embed = self.inv_transform(t2_img_embed)
        
        t1_img_embed = F.interpolate(t1_img_embed, size=(h, w), mode='bilinear', align_corners=True)
        t2_img_embed = F.interpolate(t2_img_embed, size=(h, w), mode='bilinear', align_corners=True)
        t1_img_embed = t1_img_embed.squeeze_(0)
        t2_img_embed = t2_img_embed.squeeze_(0)
        
        if self.auto_threshold:
            cosv = -F.cosine_similarity(t1_img_embed, t2_img_embed, dim=0)
            cosv = cosv.reshape(-1).cpu().numpy()
            threshold = threshold_otsu(cosv, cosv.shape[0])
            self.change_confidence_threshold = cosine2angle(threshold)
        
        def _latent_match(mask_data, t1_img_embed, t2_img_embed):
            change_confidence = torch.zeros(len(mask_data['rles']), dtype=torch.float32, device=self.device)
            for i, rle in enumerate(mask_data['rles']):
                bmask = torch.from_numpy(rle_to_mask(rle)).to(self.device)
                t1_mask_embed = torch.mean(t1_img_embed[:, bmask], dim=-1)
                t2_mask_embed = torch.mean(t2_img_embed[:, bmask], dim=-1)
                score = -F.cosine_similarity(t1_mask_embed, t2_mask_embed, dim=0)
                change_confidence[i] += score
            
            keep = change_confidence > angle2cosine(self.change_confidence_threshold)
            
            mask_data = copy.deepcopy(mask_data)
            mask_data['change_confidence'] = change_confidence
            mask_data.filter(keep)
            return mask_data
        
        changemasks = MaskData()
        if self.use_bitemporal_match:
            for i in range(2):
                cmasks = _latent_match(seq_mask_data[i], t1_img_embed, t2_img_embed)
                changemasks.cat(cmasks)
        else:
            cmasks = _latent_match(seq_mask_data[1], t1_img_embed, t2_img_embed)
            changemasks.cat(cmasks)
        del cmasks
        
        return changemasks
    
    def single_point_q_mask(self, xy, img):
        """Generate query mask from a single point."""
        point = np.array(xy).reshape(1, 2)
        
        embed_data = self.maskgen.image_encoder(img)
        embed_data.update(dict(points=point))
        mask_data = self.maskgen.embedding_point_to_mask(**embed_data)
        
        if len(mask_data['rles']) > 0:
            q_mask = torch.from_numpy(rle_to_mask(mask_data['rles'][0]))
        else:
            q_mask = torch.zeros(img.shape[0], img.shape[1])
        return q_mask
    
    def single_point_match(self, xy, temporal, img1, img2):
        """Match change regions based on a single point query."""
        h, w = img1.shape[:2]
        point = np.array(xy).reshape(1, 2)
        
        embed_data1 = self.maskgen.image_encoder(img1)
        embed_data2 = self.maskgen.image_encoder(img2)
        self.embed_data1 = embed_data1
        self.embed_data2 = embed_data2
        
        mask_data1 = self.maskgen.generate_with_image_embedding(**embed_data1)
        mask_data2 = self.maskgen.generate_with_image_embedding(**embed_data2)
        mask_data1.filter((mask_data1['areas'] / (h * w)) < self.area_thresh)
        mask_data2.filter((mask_data2['areas'] / (h * w)) < self.area_thresh)
        
        if temporal == 1:
            embed_data1.update(dict(points=point))
            mask_data = self.maskgen.embedding_point_to_mask(**embed_data1)
        elif temporal == 2:
            embed_data2.update(dict(points=point))
            mask_data = self.maskgen.embedding_point_to_mask(**embed_data2)
        else:
            raise ValueError("temporal must be 1 or 2")
        
        q_area = mask_data['areas'][0]
        q_mask = torch.from_numpy(rle_to_mask(mask_data['rles'][0]))
        
        image_embedding1 = F.interpolate(embed_data1['image_embedding'], (h, w), mode='bilinear',
                                         align_corners=True).squeeze_(0)
        image_embedding2 = F.interpolate(embed_data2['image_embedding'], (h, w), mode='bilinear',
                                         align_corners=True).squeeze_(0)
        
        if temporal == 1:
            q_mask_features = torch.mean(image_embedding1[:, q_mask], dim=-1)
        elif temporal == 2:
            q_mask_features = torch.mean(image_embedding2[:, q_mask], dim=-1)
        else:
            raise ValueError("temporal must be 1 or 2")
        
        cosmap1 = torch.cosine_similarity(q_mask_features.reshape(-1, 1, 1), image_embedding1, dim=0)
        cosmap2 = torch.cosine_similarity(q_mask_features.reshape(-1, 1, 1), image_embedding2, dim=0)
        
        obj_map1 = cosmap1 > angle2cosine(self.object_sim_thresh)
        obj_map2 = cosmap2 > angle2cosine(self.object_sim_thresh)
        
        def _filter_obj(obj_map, mask_data):
            mask_data = copy.deepcopy(mask_data)
            keep = (q_area * 0.25 < mask_data['areas']) & (mask_data['areas'] < q_area * 4)
            mask_data.filter(keep)
            keep = []
            for i, rle in enumerate(mask_data['rles']):
                mask = rle_to_mask(rle)
                keep.append(np.mean(obj_map[mask]) > 0.5)
            keep = torch.from_numpy(np.array(keep)).to(torch.bool)
            mask_data.filter(keep)
            return mask_data
        
        mask_data1 = _filter_obj(obj_map1.cpu().numpy(), mask_data1)
        mask_data2 = _filter_obj(obj_map2.cpu().numpy(), mask_data2)
        
        data = {
            't1_mask_data': mask_data1,
            't1_image_embedding': embed_data1['image_embedding'],
            't2_mask_data': mask_data2,
            't2_image_embedding': embed_data2['image_embedding'],
        }
        cmasks = self.bitemporal_match(**data)
        
        keep = batched_nms(
            cmasks["boxes"].float(),
            cmasks["iou_preds"],
            torch.zeros_like(cmasks["boxes"][:, 0]),
            iou_threshold=self.maskgen.box_nms_thresh,
        )
        cmasks.filter(keep)
        if len(cmasks['rles']) > 1000:
            scores = cmasks['change_confidence']
            sorted_scores, _ = torch.sort(scores, descending=True, stable=True)
            keep = scores > sorted_scores[1000]
            cmasks.filter(keep)
        
        return cmasks
    
    def multi_points_match(self, xyts, img1, img2):
        """Match change regions based on multiple point queries."""
        h, w = img1.shape[:2]
        
        embed_data1 = self.maskgen.image_encoder(img1)
        embed_data2 = self.maskgen.image_encoder(img2)
        self.embed_data1 = embed_data1
        self.embed_data2 = embed_data2
        
        mask_data1 = self.maskgen.generate_with_image_embedding(**embed_data1)
        mask_data2 = self.maskgen.generate_with_image_embedding(**embed_data2)
        mask_data1.filter((mask_data1['areas'] / (h * w)) < self.area_thresh)
        mask_data2.filter((mask_data2['areas'] / (h * w)) < self.area_thresh)
        
        image_embedding1 = F.interpolate(embed_data1['image_embedding'], (h, w), mode='bilinear',
                                         align_corners=True).squeeze_(0)
        image_embedding2 = F.interpolate(embed_data2['image_embedding'], (h, w), mode='bilinear',
                                         align_corners=True).squeeze_(0)
        
        q_areas = []
        q_features = []
        for xyt in xyts:
            t = xyt[-1]
            point = xyt[:2].reshape(1, 2)
            
            if t == 1:
                embed_data1.update(dict(points=point))
                mask_data = self.maskgen.embedding_point_to_mask(**embed_data1)
            elif t == 2:
                embed_data2.update(dict(points=point))
                mask_data = self.maskgen.embedding_point_to_mask(**embed_data2)
            else:
                raise ValueError("temporal must be 1 or 2")
            
            q_area = mask_data['areas'][0]
            q_mask = torch.from_numpy(rle_to_mask(mask_data['rles'][0]))
            
            q_areas.append(q_area)
            
            if t == 1:
                q_mask_features = torch.mean(image_embedding1[:, q_mask], dim=-1)
            elif t == 2:
                q_mask_features = torch.mean(image_embedding2[:, q_mask], dim=-1)
            else:
                raise ValueError("temporal must be 1 or 2")
            q_features.append(q_mask_features)
        
        q_area = sum(q_areas) / len(q_areas)
        q_mask_features = sum(q_features) / len(q_features)
        
        cosmap1 = torch.cosine_similarity(q_mask_features.reshape(-1, 1, 1), image_embedding1, dim=0)
        cosmap2 = torch.cosine_similarity(q_mask_features.reshape(-1, 1, 1), image_embedding2, dim=0)
        
        obj_map1 = cosmap1 > angle2cosine(self.object_sim_thresh)
        obj_map2 = cosmap2 > angle2cosine(self.object_sim_thresh)
        
        def _filter_obj(obj_map, mask_data):
            mask_data = copy.deepcopy(mask_data)
            keep = (q_area * 0.25 < mask_data['areas']) & (mask_data['areas'] < q_area * 4)
            mask_data.filter(keep)
            keep = []
            for i, rle in enumerate(mask_data['rles']):
                mask = rle_to_mask(rle)
                keep.append(np.mean(obj_map[mask]) > 0.5)
            keep = torch.from_numpy(np.array(keep)).to(torch.bool)
            mask_data.filter(keep)
            return mask_data
        
        mask_data1 = _filter_obj(obj_map1.cpu().numpy(), mask_data1)
        mask_data2 = _filter_obj(obj_map2.cpu().numpy(), mask_data2)
        
        data = {
            't1_mask_data': mask_data1,
            't1_image_embedding': embed_data1['image_embedding'],
            't2_mask_data': mask_data2,
            't2_image_embedding': embed_data2['image_embedding'],
        }
        cmasks = self.bitemporal_match(**data)
        
        keep = batched_nms(
            cmasks["boxes"].float(),
            cmasks["iou_preds"],
            torch.zeros_like(cmasks["boxes"][:, 0]),
            iou_threshold=self.maskgen.box_nms_thresh,
        )
        cmasks.filter(keep)
        if len(cmasks['rles']) > 1000:
            scores = cmasks['change_confidence']
            sorted_scores, _ = torch.sort(scores, descending=True, stable=True)
            keep = scores > sorted_scores[1000]
            cmasks.filter(keep)
        
        return cmasks
    
    def forward(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        return_dict: Optional[bool] = None,
    ) -> Union[Dict[str, torch.Tensor], Tuple]:
        """
        Forward pass of the model.
        
        Args:
            img1: First temporal image (numpy array)
            img2: Second temporal image (numpy array)
            return_dict: Whether to return a dictionary
            
        Returns:
            Change detection outputs
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        h, w = img1.shape[:2]
        
        if self.match_hist:
            img2 = match_histograms(image=img2, reference=img1, channel_axis=-1).astype(np.uint8)
        
        data = self.proposal(img1, img2)
        changemasks = self.bitemporal_match(**data)
        
        keep = batched_nms(
            changemasks["boxes"].float(),
            changemasks["iou_preds"],
            torch.zeros_like(changemasks["boxes"][:, 0]),
            iou_threshold=self.maskgen.box_nms_thresh,
        )
        changemasks.filter(keep)
        
        if len(changemasks['rles']) > 1000:
            scores = changemasks['change_confidence']
            sorted_scores, _ = torch.sort(scores, descending=True, stable=True)
            keep = scores > sorted_scores[1000]
            changemasks.filter(keep)
        
        if not return_dict:
            return changemasks, data['t1_mask_data'], data['t2_mask_data']
        
        return {
            "change_masks": changemasks,
            "t1_mask_data": data['t1_mask_data'],
            "t2_mask_data": data['t2_mask_data']
        }
    
    def to_eval_format_predictions(self, cmasks):
        """Convert change masks to evaluation format."""
        boxes = cmasks['boxes']
        rle_masks = cmasks['rles']
        labels = torch.ones(boxes.size(0), dtype=torch.int64)
        scores = cmasks['change_confidence']
        predictions = {
            'boxes': boxes.to(torch.float32).cpu(),
            'scores': scores.cpu(),
            'labels': labels.cpu(),
            'masks': rle_masks
        }
        return predictions
    
    def __call__(self, img1, img2):
        """Call method for easy inference."""
        cmasks, t1_masks, t2_masks = self.forward(img1, img2)
        predictions = self.to_eval_format_predictions(cmasks)
        self.clear_cached_embedding()
        return predictions


class AnyChangeForChangeDetection(AnyChangeModel):
    """
    AnyChange model specialized for change detection.
    """
    
    def __init__(self, config: AnyChangeConfig):
        super().__init__(config)
    
    def forward(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        return_dict: Optional[bool] = None,
    ) -> Union[Dict[str, torch.Tensor], Tuple]:
        """
        Forward pass for change detection.
        
        Args:
            img1: First temporal image (numpy array)
            img2: Second temporal image (numpy array)
            return_dict: Whether to return a dictionary
            
        Returns:
            Change detection outputs
        """
        outputs = super().forward(img1, img2, return_dict)
        
        if isinstance(outputs, dict):
            return {"change_masks": outputs["change_masks"]}
        else:
            return outputs[0]  # Return only change_masks 