# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Unified Models Package

This package provides a unified interface for all change detection models
in this repository, including AnyChange and Changen2.
"""

from typing import Dict, Any, Union, Optional
from enum import Enum

# Import all model configurations and classes
from .anychange import (
    AnyChangeConfig,
    AnyChangeModel,
    AnyChangeForChangeDetection,
    ModularAnyChange,
)
from .changen2 import (
    Changen2Config,
    Changen2Model,
    Changen2ForChangeDetection,
    Changen2ForImageGeneration,
    RSDiT,
    RSDiT_models,
    ChangeEventSimulation,
)
from .changestar_1xd import ChangeStar1xdConfig, ChangeStar1xdModel, ChangeStar1xdForChangeDetection


class ModelType(Enum):
    """Enumeration of available model types."""
    ANYCHANGE = "anychange"
    CHANGEN2 = "changen2"


class ChangeDetectionModelFactory:
    """
    Factory class for creating change detection models.
    
    This factory provides a unified interface for creating different types
    of change detection models with consistent configuration patterns.
    """
    
    @staticmethod
    def create_model(
        model_type: Union[str, ModelType],
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Create a change detection model of the specified type.
        
        Args:
            model_type: Type of model to create ("anychange" or "changen2")
            config: Model configuration dictionary
            **kwargs: Additional configuration parameters
            
        Returns:
            Initialized model instance
            
        Example:
            ```python
            # Create AnyChange model
            anychange_model = ChangeDetectionModelFactory.create_model(
                "anychange",
                config={"model_type": "vit_b", "change_confidence_threshold": 155}
            )
            
            # Create Changen2 model
            changen2_model = ChangeDetectionModelFactory.create_model(
                "changen2",
                config={"model_type": "RSDiT-B/2", "input_size": 256}
            )
            ```
        """
        if isinstance(model_type, str):
            model_type = ModelType(model_type.lower())
        
        config = config or {}
        config.update(kwargs)
        
        if model_type == ModelType.ANYCHANGE:
            return ChangeDetectionModelFactory._create_anychange_model(config)
        elif model_type == ModelType.CHANGEN2:
            return ChangeDetectionModelFactory._create_changen2_model(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def _create_anychange_model(config: Dict[str, Any]) -> AnyChangeForChangeDetection:
        """Create an AnyChange model with the given configuration."""
        anychange_config = AnyChangeConfig(**config)
        return AnyChangeForChangeDetection(anychange_config)
    
    @staticmethod
    def _create_changen2_model(config: Dict[str, Any]) -> Changen2ForChangeDetection:
        """Create a Changen2 model with the given configuration."""
        changen2_config = Changen2Config(**config)
        return Changen2ForChangeDetection(changen2_config)
    
    @staticmethod
    def get_default_config(model_type: Union[str, ModelType]) -> Dict[str, Any]:
        """
        Get default configuration for a model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Default configuration dictionary
        """
        if isinstance(model_type, str):
            model_type = ModelType(model_type.lower())
        
        if model_type == ModelType.ANYCHANGE:
            return {
                "model_type": "vit_b",
                "sam_checkpoint": "./sam_weights/sam_vit_b_01ec64.pth",
                "points_per_side": 32,
                "pred_iou_thresh": 0.5,
                "stability_score_thresh": 0.95,
                "change_confidence_threshold": 155,
                "auto_threshold": False,
                "use_normalized_feature": True,
                "area_thresh": 0.8,
                "object_sim_thresh": 60,
                "bitemporal_match": True,
            }
        elif model_type == ModelType.CHANGEN2:
            return {
                "model_type": "RSDiT-B/2",
                "input_size": 256,
                "patch_size": 2,
                "in_channels": 4,
                "label_channels": 1,
                "mlp_ratio": 4.0,
                "window_size": 8,
                "class_dropout_prob": 0.0,
                "learn_sigma": True,
                "frequency_embedding_size": 256,
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def list_available_models() -> Dict[str, Dict[str, Any]]:
        """
        List all available models with their information.
        
        Returns:
            Dictionary mapping model types to their information
        """
        return {
            "anychange": {
                "description": "Zero-shot change detection using SAM with bitemporal latent matching",
                "strengths": ["No training required", "Works on unseen change types", "Interactive point queries"],
                "best_for": ["Zero-shot scenarios", "Interactive detection", "General change detection"],
                "config_options": [
                    "model_type (vit_b, vit_l, vit_h)",
                    "change_confidence_threshold",
                    "auto_threshold",
                    "use_normalized_feature",
                    "area_thresh",
                    "object_sim_thresh",
                    "bitemporal_match"
                ]
            },
            "changen2": {
                "description": "Diffusion-based change detection using RSDiT",
                "strengths": ["High accuracy", "Learned representations", "Structured output"],
                "best_for": ["Supervised scenarios", "High-accuracy requirements", "Structured change detection"],
                "config_options": [
                    "model_type (RSDiT-S/2, RSDiT-B/2, RSDiT-L/2, RSDiT-XL/2)",
                    "input_size",
                    "patch_size",
                    "in_channels",
                    "label_channels",
                    "hidden_size",
                    "depth",
                    "num_heads"
                ]
            }
        }


# Convenience functions for backward compatibility
def create_anychange_model(**config) -> AnyChangeForChangeDetection:
    """Create an AnyChange model with the given configuration."""
    return ChangeDetectionModelFactory.create_model("anychange", config)


def create_changen2_model(**config) -> Changen2ForChangeDetection:
    """Create a Changen2 model with the given configuration."""
    return ChangeDetectionModelFactory.create_model("changen2", config)


# Export all public classes and functions
__all__ = [
    # Model types
    "ModelType",
    
    # Factory
    "ChangeDetectionModelFactory",
    "create_anychange_model",
    "create_changen2_model",
    
    # AnyChange models
    "AnyChangeConfig",
    "AnyChangeModel",
    "AnyChangeForChangeDetection",
    "ModularAnyChange",
    
    # Changen2 models
    "Changen2Config",
    "Changen2Model",
    "Changen2ForChangeDetection",
    "Changen2ForImageGeneration",
    
    # Changen2 components
    "RSDiT",
    "RSDiT_models",
    "ChangeEventSimulation",
    "ChangeStar1xdConfig",
    "ChangeStar1xdModel", 
    "ChangeStar1xdForChangeDetection",
] 