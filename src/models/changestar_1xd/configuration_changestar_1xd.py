# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Dict, List, Optional, Union

from transformers import PretrainedConfig


class ChangeStar1xdConfig(PretrainedConfig):
    """
    Configuration class for ChangeStar1xd model.
    
    Args:
        encoder_type (str): Type of encoder to use
        encoder_params (dict): Parameters for the encoder
        bitemporal_forward (bool): Whether to use bitemporal forward pass
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        temporal_symmetric (bool): Whether to use temporal symmetric processing
        num_semantic_classes (Optional[Union[int, List[int]]]): Number of semantic classes
        num_change_classes (Optional[int]): Number of change classes
        loss_config (dict): Loss configuration
        **kwargs: Additional arguments
    """
    
    model_type = "changestar_1xd"
    
    def __init__(
        self,
        encoder_type: str = "resnet",
        encoder_params: Dict = None,
        bitemporal_forward: bool = False,
        in_channels: int = 3,
        out_channels: int = 256,
        temporal_symmetric: bool = True,
        num_semantic_classes: Optional[Union[int, List[int]]] = None,
        num_change_classes: Optional[int] = None,
        loss_config: Dict = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.encoder_type = encoder_type
        self.encoder_params = encoder_params or {}
        self.bitemporal_forward = bitemporal_forward
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temporal_symmetric = temporal_symmetric
        self.num_semantic_classes = num_semantic_classes
        self.num_change_classes = num_change_classes or 1
        self.loss_config = loss_config or {}
        
        # Set encoder output channels if not provided
        if "out_channels" not in self.encoder_params:
            self.encoder_params["out_channels"] = self.out_channels
            
    def to_dict(self):
        """Convert config to dictionary."""
        output = super().to_dict()
        output.update({
            "encoder_type": self.encoder_type,
            "encoder_params": self.encoder_params,
            "bitemporal_forward": self.bitemporal_forward,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "temporal_symmetric": self.temporal_symmetric,
            "num_semantic_classes": self.num_semantic_classes,
            "num_change_classes": self.num_change_classes,
            "loss_config": self.loss_config,
        })
        return output 