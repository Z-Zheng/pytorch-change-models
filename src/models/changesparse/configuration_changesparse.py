# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Dict, List, Optional, Union

from transformers import PretrainedConfig


class ChangeSparseConfig(PretrainedConfig):
    """
    Configuration class for ChangeSparse model.
    
    Args:
        backbone_name (str): Type of backbone to use (default: 'er.R18')
        backbone_pretrained (bool): Whether to use pre-trained backbone (default: True)
        temporal_reduction_type (str): Type of temporal reduction ('conv' or 'ADBN')
        inner_channels (int): Number of inner channels (default: 96)
        num_heads (tuple): Number of attention heads for each stage (default: (3, 3, 3, 3))
        qkv_bias (bool): Whether to use bias in QKV projection (default: True)
        drop (float): Dropout rate (default: 0.0)
        attn_drop (float): Attention dropout rate (default: 0.0)
        drop_path (float): Drop path rate (default: 0.0)
        change_threshold (float): Threshold for change region prediction (default: 0.5)
        min_keep_ratio (float): Minimum ratio of regions to keep (default: 0.01)
        max_keep_ratio (float): Maximum ratio of regions to keep (default: 0.1)
        train_max_keep (int): Maximum number of regions to keep during training (default: 2000)
        num_blocks (tuple): Number of transformer blocks for each stage (default: (2, 2, 2, 2))
        disable_attn_refine (bool): Whether to disable attention refinement (default: False)
        output_type (str): Output type ('single_scale' or 'multi_scale')
        pc_upsample (str): Upsampling method for probability maps (default: 'nearest')
        num_change_classes (int): Number of change classes (default: 1)
        num_semantic_classes (int): Number of semantic classes (default: 6)
        loss_config (dict): Loss configuration
        **kwargs: Additional arguments
    """
    
    model_type = "changesparse"
    
    def __init__(
        self,
        backbone_name: str = "er.R18",
        backbone_pretrained: bool = True,
        temporal_reduction_type: str = "conv",
        inner_channels: int = 96,
        num_heads: tuple = (3, 3, 3, 3),
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        change_threshold: float = 0.5,
        min_keep_ratio: float = 0.01,
        max_keep_ratio: float = 0.1,
        train_max_keep: int = 2000,
        num_blocks: tuple = (2, 2, 2, 2),
        disable_attn_refine: bool = False,
        output_type: str = "single_scale",
        pc_upsample: str = "nearest",
        num_change_classes: int = 1,
        num_semantic_classes: int = 6,
        loss_config: Dict = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.backbone_name = backbone_name
        self.backbone_pretrained = backbone_pretrained
        self.temporal_reduction_type = temporal_reduction_type
        self.inner_channels = inner_channels
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.drop = drop
        self.attn_drop = attn_drop
        self.drop_path = drop_path
        self.change_threshold = change_threshold
        self.min_keep_ratio = min_keep_ratio
        self.max_keep_ratio = max_keep_ratio
        self.train_max_keep = train_max_keep
        self.num_blocks = num_blocks
        self.disable_attn_refine = disable_attn_refine
        self.output_type = output_type
        self.pc_upsample = pc_upsample
        self.num_change_classes = num_change_classes
        self.num_semantic_classes = num_semantic_classes
        self.loss_config = loss_config or {}
        
    def to_dict(self):
        """Convert config to dictionary."""
        output = super().to_dict()
        output.update({
            "backbone_name": self.backbone_name,
            "backbone_pretrained": self.backbone_pretrained,
            "temporal_reduction_type": self.temporal_reduction_type,
            "inner_channels": self.inner_channels,
            "num_heads": self.num_heads,
            "qkv_bias": self.qkv_bias,
            "drop": self.drop,
            "attn_drop": self.attn_drop,
            "drop_path": self.drop_path,
            "change_threshold": self.change_threshold,
            "min_keep_ratio": self.min_keep_ratio,
            "max_keep_ratio": self.max_keep_ratio,
            "train_max_keep": self.train_max_keep,
            "num_blocks": self.num_blocks,
            "disable_attn_refine": self.disable_attn_refine,
            "output_type": self.output_type,
            "pc_upsample": self.pc_upsample,
            "num_change_classes": self.num_change_classes,
            "num_semantic_classes": self.num_semantic_classes,
            "loss_config": self.loss_config,
        })
        return output 