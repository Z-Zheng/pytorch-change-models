# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from .configuration_changen2 import Changen2Config
from .modular_change2 import RSDiT_models, RSDiT


class Changen2Model(PreTrainedModel):
    """
    Changen2 model for multi-temporal remote sensing image generation.
    
    This model can be used for generating time series of remote sensing images
    and corresponding semantic and change labels from single-temporal images.
    """
    
    config_class = Changen2Config
    base_model_prefix = "changen2"
    supports_gradient_checkpointing = True
    
    def __init__(self, config: Changen2Config):
        super().__init__(config)
        
        # Initialize RSDiT model based on config
        if config.model_type not in RSDiT_models:
            raise ValueError(f"Model type {config.model_type} not found in RSDiT_models")
        
        self.rsdit = RSDiT_models[config.model_type](
            input_size=config.input_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            hidden_size=config.hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            class_dropout_prob=config.class_dropout_prob,
            label_channels=config.label_channels,
            learn_sigma=config.learn_sigma,
            window_size=config.window_size,
            frequency_embedding_size=config.frequency_embedding_size,
        )
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights."""
        # Weights are already initialized in RSDiT.__init__()
        pass
    
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        timesteps: torch.LongTensor,
        labels: torch.LongTensor,
        return_dict: Optional[bool] = None,
    ) -> Union[Dict[str, torch.Tensor], Tuple]:
        """
        Forward pass of the model.
        
        Args:
            pixel_values: Input tensor of shape (batch_size, channels, height, width)
            timesteps: Diffusion timesteps tensor of shape (batch_size,)
            labels: Label tensor of shape (batch_size, height, width)
            return_dict: Whether to return a dictionary
            
        Returns:
            Model outputs
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Forward pass through RSDiT
        outputs = self.rsdit(pixel_values, timesteps, labels)
        
        if not return_dict:
            return outputs
        
        return {"logits": outputs}
    
    def forward_with_cfg(
        self,
        pixel_values: torch.FloatTensor,
        timesteps: torch.LongTensor,
        labels: torch.LongTensor,
        cfg_scale: float,
        return_dict: Optional[bool] = None,
    ) -> Union[Dict[str, torch.Tensor], Tuple]:
        """
        Forward pass with classifier-free guidance.
        
        Args:
            pixel_values: Input tensor of shape (batch_size, channels, height, width)
            timesteps: Diffusion timesteps tensor of shape (batch_size,)
            labels: Label tensor of shape (batch_size, height, width)
            cfg_scale: Classifier-free guidance scale
            return_dict: Whether to return a dictionary
            
        Returns:
            Model outputs with CFG
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Forward pass through RSDiT with CFG
        outputs = self.rsdit.forward_with_cfg(pixel_values, timesteps, labels, cfg_scale)
        
        if not return_dict:
            return outputs
        
        return {"logits": outputs}


class Changen2ForImageGeneration(Changen2Model):
    """
    Changen2 model for image generation tasks.
    """
    
    def __init__(self, config: Changen2Config):
        super().__init__(config)
    
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        timesteps: torch.LongTensor,
        labels: torch.LongTensor,
        return_dict: Optional[bool] = None,
    ) -> Union[Dict[str, torch.Tensor], Tuple]:
        """
        Forward pass for image generation.
        
        Args:
            pixel_values: Input tensor of shape (batch_size, channels, height, width)
            timesteps: Diffusion timesteps tensor of shape (batch_size,)
            labels: Label tensor of shape (batch_size, height, width)
            return_dict: Whether to return a dictionary
            
        Returns:
            Image generation outputs
        """
        outputs = super().forward(pixel_values, timesteps, labels, return_dict)
        
        if not return_dict:
            return outputs
        
        return {"generated_images": outputs["logits"]}


class Changen2ForChangeDetection(Changen2Model):
    """
    Changen2 model for change detection tasks.
    """
    
    def __init__(self, config: Changen2Config):
        super().__init__(config)
    
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        timesteps: torch.LongTensor,
        labels: torch.LongTensor,
        return_dict: Optional[bool] = None,
    ) -> Union[Dict[str, torch.Tensor], Tuple]:
        """
        Forward pass for change detection.
        
        Args:
            pixel_values: Input tensor of shape (batch_size, channels, height, width)
            timesteps: Diffusion timesteps tensor of shape (batch_size,)
            labels: Label tensor of shape (batch_size, height, width)
            return_dict: Whether to return a dictionary
            
        Returns:
            Change detection outputs
        """
        outputs = super().forward(pixel_values, timesteps, labels, return_dict)
        
        if not return_dict:
            return outputs
        
        return {"change_logits": outputs["logits"]} 