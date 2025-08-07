# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Dict, List, Optional, Union

from transformers import PretrainedConfig


class ChangeStar2Config(PretrainedConfig):
    """
    Configuration class for ChangeStar2 model.
    
    Args:
        segmentation_config (dict): Configuration for the segmentation model
        semantic_classifier_config (dict): Configuration for semantic classifier
        change_detector_config (dict): Configuration for change detector
        target_generator_config (dict): Configuration for target generator
        loss_config (dict): Loss configuration
        pcm_m2m_inference (bool): Whether to use PCM M2M inference
        change_type (str): Type of change detection ('binary' or 'multi_class')
        **kwargs: Additional arguments
    """
    
    model_type = "changestar2"
    
    def __init__(
        self,
        segmentation_config: Dict = None,
        semantic_classifier_config: Dict = None,
        change_detector_config: Dict = None,
        target_generator_config: Dict = None,
        loss_config: Dict = None,
        pcm_m2m_inference: bool = False,
        change_type: str = "binary",
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Default segmentation configuration (Farseg with ResNet50)
        self.segmentation_config = segmentation_config or {
            "model_type": "farseg",
            "backbone": {
                "resnet_type": "resnet50",
                "pretrained": True,
                "freeze_at": 0,
                "output_stride": 32,
            },
            "head": {
                "fpn": {
                    "in_channels_list": (256, 512, 1024, 2048),
                    "out_channels": 256,
                },
                "fs_relation": {
                    "scene_embedding_channels": 2048,
                    "in_channels_list": (256, 256, 256, 256),
                    "out_channels": 256,
                    "scale_aware_proj": True
                },
                "fpn_decoder": {
                    "in_channels": 256,
                    "out_channels": 256,
                    "in_feat_output_strides": (4, 8, 16, 32),
                    "out_feat_output_stride": 4,
                    "classifier_config": None
                }
            },
        }
        
        # Default semantic classifier configuration
        self.semantic_classifier_config = semantic_classifier_config or {
            "in_channels": 256,
            "out_channels": 1,
            "scale": 4.0
        }
        
        # Default change detector configuration (TSMTDM)
        self.change_detector_config = change_detector_config or {
            "name": "TSMTDM",
            "in_channels": 256,
            "scale": 4.0,
            "tsm_cfg": {
                "dim": 16,
                "drop_path_prob": 0.2,
                "num_convs": 4,
            },
            "tdm_cfg": {
                "NConvNeXtBlock": 9,
                "PreNorm": "LN"
            },
        }
        
        # Default target generator configuration
        self.target_generator_config = target_generator_config or {
            "name": "sync_generate_target_v3",
            "shuffle_prob": 1.0
        }
        
        # Default loss configuration
        self.loss_config = loss_config or {
            "change": {
                "symmetry_loss": True,
                "bce": True,
                "dice": False,
                "weight": 0.5,
                "ignore_index": -1,
                "log_bce_pos_neg_stat": True,
            },
            "semantic": {
                "on": True,
                "bce": True,
                "dice": True,
                "ignore_index": -1,
            },
        }
        
        self.pcm_m2m_inference = pcm_m2m_inference
        self.change_type = change_type
        
    def to_dict(self):
        """Convert config to dictionary."""
        output = super().to_dict()
        output.update({
            "segmentation_config": self.segmentation_config,
            "semantic_classifier_config": self.semantic_classifier_config,
            "change_detector_config": self.change_detector_config,
            "target_generator_config": self.target_generator_config,
            "loss_config": self.loss_config,
            "pcm_m2m_inference": self.pcm_m2m_inference,
            "change_type": self.change_type,
        })
        return output 