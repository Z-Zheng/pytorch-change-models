# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Dict, List, Optional, Union

from transformers import PretrainedConfig


class ChangeMaskConfig(PretrainedConfig):
    """
    Configuration class for ChangeMask model.
    
    Args:
        encoder_type (str): Type of encoder to use (default: 'efficientnet-b0')
        encoder_weights (str): Pre-trained weights for encoder (default: 'imagenet')
        decoder_channels (List[int]): List of decoder channel sizes
        temporal_interaction_type (str): Type of temporal interaction ('conv3d' or 'conv1plus2d')
        temporal_kernel_size (int): Kernel size for temporal interaction
        temporal_dilation (int): Dilation for temporal interaction
        temporal_symmetric_fusion (str): Symmetric fusion type ('add', 'mul', or None)
        num_semantic_classes (int): Number of semantic classes
        num_change_classes (int): Number of change classes
        loss_config (dict): Loss configuration
        **kwargs: Additional arguments
    """
    
    model_type = "changemask"
    
    def __init__(
        self,
        encoder_type: str = "efficientnet-b0",
        encoder_weights: str = "imagenet",
        decoder_channels: List[int] = None,
        temporal_interaction_type: str = "conv3d",
        temporal_kernel_size: int = 3,
        temporal_dilation: int = 1,
        temporal_symmetric_fusion: str = "add",
        num_semantic_classes: int = 6,
        num_change_classes: int = 1,
        loss_config: Dict = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.encoder_type = encoder_type
        self.encoder_weights = encoder_weights
        self.decoder_channels = decoder_channels or [256, 128, 64, 32, 16]
        self.temporal_interaction_type = temporal_interaction_type
        self.temporal_kernel_size = temporal_kernel_size
        self.temporal_dilation = temporal_dilation
        self.temporal_symmetric_fusion = temporal_symmetric_fusion
        self.num_semantic_classes = num_semantic_classes
        self.num_change_classes = num_change_classes
        self.loss_config = loss_config or {}
        
    def to_dict(self):
        """Convert config to dictionary."""
        output = super().to_dict()
        output.update({
            "encoder_type": self.encoder_type,
            "encoder_weights": self.encoder_weights,
            "decoder_channels": self.decoder_channels,
            "temporal_interaction_type": self.temporal_interaction_type,
            "temporal_kernel_size": self.temporal_kernel_size,
            "temporal_dilation": self.temporal_dilation,
            "temporal_symmetric_fusion": self.temporal_symmetric_fusion,
            "num_semantic_classes": self.num_semantic_classes,
            "num_change_classes": self.num_change_classes,
            "loss_config": self.loss_config,
        })
        return output 