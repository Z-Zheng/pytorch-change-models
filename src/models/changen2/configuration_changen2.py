# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Dict, List, Optional, Union

from transformers import PretrainedConfig


class Changen2Config(PretrainedConfig):
    """
    Configuration class for Changen2 model.
    
    Args:
        model_type (str): Type of RSDiT model to use ('RSDiT-B/2', 'RSDiT-L/2', 'RSDiT-XL/2', 'RSDiT-S/2')
        input_size (int): Input image size
        patch_size (int): Patch size for patch embedding
        in_channels (int): Number of input channels
        label_channels (int): Number of label channels
        hidden_size (int): Hidden size of the model
        depth (int): Number of transformer blocks
        num_heads (int): Number of attention heads
        mlp_ratio (float): MLP ratio for transformer blocks
        window_size (int): Window size for window attention
        class_dropout_prob (float): Class dropout probability
        learn_sigma (bool): Whether to learn sigma parameter
        frequency_embedding_size (int): Size of frequency embedding
        **kwargs: Additional arguments
    """
    
    model_type = "changen2"
    
    def __init__(
        self,
        model_type: str = "RSDiT-B/2",
        input_size: int = 256,
        patch_size: int = 2,
        in_channels: int = 4,
        label_channels: int = 1,
        hidden_size: Optional[int] = None,
        depth: Optional[int] = None,
        num_heads: Optional[int] = None,
        mlp_ratio: float = 4.0,
        window_size: int = 8,
        class_dropout_prob: float = 0.0,
        learn_sigma: bool = True,
        frequency_embedding_size: int = 256,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.model_type = model_type
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.label_channels = label_channels
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size
        self.class_dropout_prob = class_dropout_prob
        self.learn_sigma = learn_sigma
        self.frequency_embedding_size = frequency_embedding_size
        
        # Set model-specific parameters based on model_type
        if model_type == "RSDiT-B/2":
            self.hidden_size = hidden_size or 768
            self.depth = depth or 12
            self.num_heads = num_heads or 12
        elif model_type == "RSDiT-L/2":
            self.hidden_size = hidden_size or 1024
            self.depth = depth or 24
            self.num_heads = num_heads or 16
        elif model_type == "RSDiT-XL/2":
            self.hidden_size = hidden_size or 1152
            self.depth = depth or 28
            self.num_heads = num_heads or 16
        elif model_type == "RSDiT-S/2":
            self.hidden_size = hidden_size or 384
            self.depth = depth or 12
            self.num_heads = num_heads or 6
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    def to_dict(self):
        """Convert config to dictionary."""
        output = super().to_dict()
        output.update({
            "model_type": self.model_type,
            "input_size": self.input_size,
            "patch_size": self.patch_size,
            "in_channels": self.in_channels,
            "label_channels": self.label_channels,
            "hidden_size": self.hidden_size,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "window_size": self.window_size,
            "class_dropout_prob": self.class_dropout_prob,
            "learn_sigma": self.learn_sigma,
            "frequency_embedding_size": self.frequency_embedding_size,
        })
        return output 