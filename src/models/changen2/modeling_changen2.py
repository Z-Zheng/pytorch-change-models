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
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load a pretrained model from a directory or HuggingFace hub.
        
        Args:
            pretrained_model_name_or_path: Path to the pretrained model directory or model identifier from HuggingFace hub
            *model_args: Additional arguments for model initialization
            **kwargs: Additional keyword arguments for model initialization
            
        Returns:
            Changen2Model: Loaded model
        """
        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        mirror = kwargs.pop("mirror", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        device_map = kwargs.pop("device_map", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", False)
        
        # Load config
        if config is None:
            config = Changen2Config.from_pretrained(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                mirror=mirror,
            )
        
        # Create model
        model = cls(config, *model_args, **kwargs)
        
        # Load state dict if provided
        if state_dict is not None:
            model.load_state_dict(state_dict)
        else:
            # Try to load from the pretrained path
            try:
                state_dict = torch.load(
                    os.path.join(pretrained_model_name_or_path, "pytorch_model.bin"),
                    map_location="cpu"
                )
                model.load_state_dict(state_dict)
            except (FileNotFoundError, OSError):
                # If no state dict found, return model with initialized weights
                pass
        
        # Set dtype and device if specified
        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)
        
        if device_map is not None:
            model = model.to(device_map)
        
        return model
    
    def save_pretrained(self, save_directory, **kwargs):
        """
        Save the model to a directory.
        
        Args:
            save_directory: Directory to save the model
            **kwargs: Additional arguments for saving
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save config
        self.config.save_pretrained(save_directory)
        
        # Save model weights
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))
        
        # Save model info
        model_info = {
            "model_type": self.config.model_type,
            "version": "1.0.0",
        }
        
        with open(os.path.join(save_directory, "model_info.json"), "w") as f:
            import json
            json.dump(model_info, f, indent=2)
    
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
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load a pretrained model from a directory or HuggingFace hub.
        
        Args:
            pretrained_model_name_or_path: Path to the pretrained model directory or model identifier from HuggingFace hub
            *model_args: Additional arguments for model initialization
            **kwargs: Additional keyword arguments for model initialization
            
        Returns:
            Changen2ForImageGeneration: Loaded model
        """
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
    
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
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load a pretrained model from a directory or HuggingFace hub.
        
        Args:
            pretrained_model_name_or_path: Path to the pretrained model directory or model identifier from HuggingFace hub
            *model_args: Additional arguments for model initialization
            **kwargs: Additional keyword arguments for model initialization
            
        Returns:
            Changen2ForChangeDetection: Loaded model
        """
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
    
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