# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Unified Change Detection Pipeline

This pipeline provides a unified interface for change detection using different models:
- AnyChange: Zero-shot change detection using SAM with bitemporal latent matching
- Changen2: Diffusion-based change detection using RSDiT

The pipeline automatically selects the appropriate model and method based on the input
configuration and provides consistent output formats.
"""

import os
import warnings
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import requests

from transformers import Pipeline, add_end_docstrings
from transformers.pipelines.base import build_pipeline_init_args
from transformers.utils import is_torch_available, logging

# Import both model types
from ..models.anychange import AnyChangeConfig, AnyChangeModel, AnyChangeForChangeDetection
from ..models.changen2 import Changen2Config, Changen2ForChangeDetection

logger = logging.get_logger(__name__)


class ChangeDetectionMethod(Enum):
    """Enumeration of available change detection methods."""
    ANYCHANGE = "anychange"
    CHANGEN2 = "changen2"
    AUTO = "auto"


class UnifiedChangeDetectionPipeline(Pipeline):
    """
    Unified change detection pipeline supporting multiple models and methods.
    
    This pipeline provides a consistent interface for change detection using:
    - AnyChange: Zero-shot SAM-based change detection
    - Changen2: Diffusion-based change detection
    
    Example:
    
    ```python
    >>> from src.pipelines import UnifiedChangeDetectionPipeline
    
    >>> # Use AnyChange for zero-shot detection
    >>> detector = UnifiedChangeDetectionPipeline(
    ...     method="anychange",
    ...     model_config={"model_type": "vit_b"},
    ...     device="cuda"
    ... )
    
    >>> # Use Changen2 for diffusion-based detection
    >>> detector = UnifiedChangeDetectionPipeline(
    ...     method="changen2",
    ...     model_config={"model_type": "RSDiT-B/2"},
    ...     device="cuda"
    ... )
    
    >>> # Auto-detect method based on model
    >>> detector = UnifiedChangeDetectionPipeline(
    ...     method="auto",
    ...     model_path="path/to/model",
    ...     device="cuda"
    ... )
    
    >>> # Detect changes
    >>> results = detector(
    ...     t1_image="path/to/t1_image.png",
    ...     t2_image="path/to/t2_image.png"
    ... )
    ```
    """
    
    _load_processor = False
    _load_image_processor = False
    _load_feature_extractor = False
    _load_tokenizer = False
    
    def __init__(
        self,
        method: Union[str, ChangeDetectionMethod] = ChangeDetectionMethod.AUTO,
        model: Optional[Union[str, AnyChangeModel, Changen2ForChangeDetection]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        device: Optional[Union[int, str, torch.device]] = None,
        torch_dtype: Optional[Union[str, torch.dtype]] = None,
        **kwargs
    ):
        """
        Initialize the unified change detection pipeline.
        
        Args:
            method: Change detection method ("anychange", "changen2", "auto")
            model: Path to pretrained model or model instance
            model_config: Model configuration dictionary
            device: Device to run inference on
            torch_dtype: Data type for model tensors
            **kwargs: Additional arguments
        """
        if not is_torch_available():
            raise ImportError("PyTorch is required for this pipeline")
        
        # Convert method string to enum
        if isinstance(method, str):
            method = ChangeDetectionMethod(method.lower())
        
        self.method = method
        self.model_config = model_config or {}
        
        # Set default parameters
        self.threshold = kwargs.get("threshold", 0.5)
        self.return_confidence = kwargs.get("return_confidence", True)
        self.post_process = kwargs.get("post_process", True)
        self.min_change_area = kwargs.get("min_change_area", 100)
        
        # Initialize model based on method
        self.model = self._initialize_model(method, model, model_config, device, torch_dtype)
        
        super().__init__(
            model=self.model,
            device=device,
            torch_dtype=torch_dtype,
            **kwargs
        )
    
    def _initialize_model(
        self,
        method: ChangeDetectionMethod,
        model: Optional[Union[str, AnyChangeModel, Changen2ForChangeDetection]],
        model_config: Dict[str, Any],
        device: Optional[Union[int, str, torch.device]],
        torch_dtype: Optional[Union[str, torch.dtype]]
    ) -> Union[AnyChangeModel, Changen2ForChangeDetection]:
        """
        Initialize the appropriate model based on the method.
        
        Args:
            method: Change detection method
            model: Model instance or path
            model_config: Model configuration
            device: Device to run on
            torch_dtype: Data type
            
        Returns:
            Initialized model
        """
        if model is not None:
            # Use provided model
            if isinstance(model, str):
                # Auto-detect method from model path or config
                if method == ChangeDetectionMethod.AUTO:
                    method = self._detect_method_from_path(model)
                
                if method == ChangeDetectionMethod.ANYCHANGE:
                    config = AnyChangeConfig.from_pretrained(model, **model_config)
                    model = AnyChangeForChangeDetection.from_pretrained(
                        model, config=config, torch_dtype=torch_dtype
                    )
                elif method == ChangeDetectionMethod.CHANGEN2:
                    config = Changen2Config.from_pretrained(model, **model_config)
                    model = Changen2ForChangeDetection.from_pretrained(
                        model, config=config, torch_dtype=torch_dtype
                    )
            else:
                # Use provided model instance
                pass
        else:
            # Create new model based on method
            if method == ChangeDetectionMethod.ANYCHANGE:
                config = AnyChangeConfig(**model_config)
                model = AnyChangeForChangeDetection(config)
            elif method == ChangeDetectionMethod.CHANGEN2:
                config = Changen2Config(**model_config)
                model = Changen2ForChangeDetection(config)
            elif method == ChangeDetectionMethod.AUTO:
                raise ValueError("Cannot auto-detect method without model path or instance")
        
        # Move model to device
        if device is not None:
            model = model.to(device)
        
        model.eval()
        return model
    
    def _detect_method_from_path(self, model_path: str) -> ChangeDetectionMethod:
        """
        Detect the method from model path or configuration.
        
        Args:
            model_path: Path to model
            
        Returns:
            Detected method
        """
        # Try to load config and detect method
        try:
            # Try AnyChange config first
            config = AnyChangeConfig.from_pretrained(model_path)
            return ChangeDetectionMethod.ANYCHANGE
        except:
            try:
                # Try Changen2 config
                config = Changen2Config.from_pretrained(model_path)
                return ChangeDetectionMethod.CHANGEN2
            except:
                # Default to AnyChange for zero-shot capability
                logger.warning("Could not detect method from model path, defaulting to AnyChange")
                return ChangeDetectionMethod.ANYCHANGE
    
    def _sanitize_parameters(
        self,
        threshold: Optional[float] = None,
        return_confidence: Optional[bool] = None,
        post_process: Optional[bool] = None,
        min_change_area: Optional[int] = None,
        **kwargs
    ):
        """
        Sanitize pipeline parameters.
        
        Args:
            threshold: Threshold for binary change detection
            return_confidence: Whether to return confidence scores
            post_process: Whether to apply post-processing
            min_change_area: Minimum change area
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (preprocess_params, forward_params, postprocess_params)
        """
        preprocess_params = {}
        forward_params = {}
        postprocess_params = {}
        
        if threshold is not None:
            postprocess_params["threshold"] = threshold
        if return_confidence is not None:
            postprocess_params["return_confidence"] = return_confidence
        if post_process is not None:
            postprocess_params["post_process"] = post_process
        if min_change_area is not None:
            postprocess_params["min_change_area"] = min_change_area
        
        return preprocess_params, forward_params, postprocess_params
    
    def preprocess(
        self,
        t1_image: Union[str, Image.Image, np.ndarray],
        t2_image: Union[str, Image.Image, np.ndarray],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Preprocess input images for change detection.
        
        Args:
            t1_image: First temporal image
            t2_image: Second temporal image
            **kwargs: Additional preprocessing parameters
            
        Returns:
            Dictionary containing preprocessed data
        """
        # Load and preprocess images
        t1_array, t2_array = self._load_and_preprocess_images(t1_image, t2_image, **kwargs)
        
        if self.method == ChangeDetectionMethod.ANYCHANGE:
            # AnyChange expects numpy arrays
            return {
                "t1_image": t1_array,
                "t2_image": t2_array,
            }
        elif self.method == ChangeDetectionMethod.CHANGEN2:
            # Changen2 expects tensors
            t1_tensor, t2_tensor = self._prepare_changen2_input(t1_array, t2_array)
            return {
                "pixel_values": t1_tensor,
                "timesteps": torch.zeros(1, dtype=torch.long),
                "labels": torch.zeros(1, t1_tensor.shape[2], t1_tensor.shape[3], dtype=torch.long),
            }
    
    def _load_and_preprocess_images(
        self,
        t1_image: Union[str, Image.Image, np.ndarray],
        t2_image: Union[str, Image.Image, np.ndarray],
        target_size: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess images for change detection.
        
        Args:
            t1_image: First temporal image
            t2_image: Second temporal image
            target_size: Target size for resizing
            normalize: Whether to normalize pixel values
            **kwargs: Additional parameters
            
        Returns:
            Tuple of preprocessed numpy arrays
        """
        # Load images if they are paths
        if isinstance(t1_image, str):
            t1_image = self._load_image(t1_image)
        if isinstance(t2_image, str):
            t2_image = self._load_image(t2_image)
        
        # Convert to PIL if numpy arrays
        if isinstance(t1_image, np.ndarray):
            t1_image = Image.fromarray(t1_image)
        if isinstance(t2_image, np.ndarray):
            t2_image = Image.fromarray(t2_image)
        
        # Resize images if target size is specified
        if target_size is not None:
            t1_image = t1_image.resize(target_size, Image.BILINEAR)
            t2_image = t2_image.resize(target_size, Image.BILINEAR)
        
        # Convert to numpy arrays
        t1_array = np.array(t1_image).astype(np.float32)
        t2_array = np.array(t2_image).astype(np.float32)
        
        # Handle different channel configurations
        if len(t1_array.shape) == 2:  # Grayscale
            t1_array = t1_array[..., None]
            t2_array = t2_array[..., None]
        elif t1_array.shape[-1] > 3:  # Multi-spectral
            # Take first 3 channels for RGB-like processing
            t1_array = t1_array[..., :3]
            t2_array = t2_array[..., :3]
        
        # Normalize to [0, 1] if requested
        if normalize:
            t1_array = t1_array / 255.0
            t2_array = t2_array / 255.0
        
        return t1_array, t2_array
    
    def _load_image(self, image: Union[str, Image.Image]) -> Image.Image:
        """
        Load an image from a path or PIL Image object.
        
        Args:
            image: Path to image file, URL, or PIL Image object
            
        Returns:
            PIL Image object
        """
        if isinstance(image, str):
            if image.startswith("http://") or image.startswith("https://"):
                image = Image.open(requests.get(image, stream=True).raw)
            else:
                image = Image.open(image)
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Image must be a string path or PIL Image, got {type(image)}")
        
        return image
    
    def _prepare_changen2_input(
        self,
        t1_array: np.ndarray,
        t2_array: np.ndarray
    ) -> torch.Tensor:
        """
        Prepare input for Changen2 model.
        
        Args:
            t1_array: First temporal image array
            t2_array: Second temporal image array
            
        Returns:
            Input tensor for Changen2
        """
        # Convert to tensors and add batch dimension
        t1_tensor = torch.from_numpy(t1_array).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        t2_tensor = torch.from_numpy(t2_array).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        
        # Concatenate T1 and T2 images for Changen2 input
        input_tensor = torch.cat([t1_tensor, t2_tensor], dim=1)  # (1, 6, H, W)
        
        return input_tensor
    
    def _forward(self, model_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass through the model.
        
        Args:
            model_inputs: Preprocessed model inputs
            
        Returns:
            Model outputs
        """
        with torch.no_grad():
            if self.method == ChangeDetectionMethod.ANYCHANGE:
                # AnyChange forward pass
                outputs = self.model(
                    model_inputs["t1_image"],
                    model_inputs["t2_image"]
                )
            elif self.method == ChangeDetectionMethod.CHANGEN2:
                # Changen2 forward pass
                outputs = self.model(**model_inputs)
        
        return outputs
    
    def postprocess(
        self,
        model_outputs: Dict[str, Any],
        threshold: float = 0.5,
        return_confidence: bool = True,
        post_process: bool = True,
        min_change_area: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Postprocess model outputs to generate change detection results.
        
        Args:
            model_outputs: Raw model outputs
            threshold: Threshold for binary change detection
            return_confidence: Whether to return confidence scores
            post_process: Whether to apply post-processing
            min_change_area: Minimum change area
            **kwargs: Additional postprocessing parameters
            
        Returns:
            Dictionary containing change detection results
        """
        if self.method == ChangeDetectionMethod.ANYCHANGE:
            return self._postprocess_anychange(
                model_outputs, threshold, return_confidence, post_process, min_change_area
            )
        elif self.method == ChangeDetectionMethod.CHANGEN2:
            return self._postprocess_changen2(
                model_outputs, threshold, return_confidence, post_process, min_change_area
            )
    
    def _postprocess_anychange(
        self,
        model_outputs: Dict[str, Any],
        threshold: float,
        return_confidence: bool,
        post_process: bool,
        min_change_area: int
    ) -> Dict[str, Any]:
        """
        Postprocess AnyChange model outputs.
        
        Args:
            model_outputs: AnyChange model outputs
            threshold: Threshold for binary change detection
            return_confidence: Whether to return confidence scores
            post_process: Whether to apply post-processing
            min_change_area: Minimum change area
            
        Returns:
            Postprocessed results
        """
        # Extract change masks from AnyChange output
        change_masks = model_outputs
        
        # Convert to binary mask
        if hasattr(change_masks, 'rles') and len(change_masks.rles) > 0:
            # Convert RLE masks to binary mask
            from libs.segment_any.segment_anything.utils.amg import rle_to_mask
            
            h, w = change_masks['original_size']
            change_mask = np.zeros((h, w), dtype=np.uint8)
            
            for rle in change_masks.rles:
                mask = rle_to_mask(rle)
                change_mask = np.logical_or(change_mask, mask).astype(np.uint8)
            
            # Calculate confidence from change confidence scores
            confidence = float(np.max(change_masks['change_confidence'])) if 'change_confidence' in change_masks else 1.0
        else:
            change_mask = np.zeros((256, 256), dtype=np.uint8)  # Default size
            confidence = 0.0
        
        # Apply post-processing
        if post_process:
            change_mask = self._post_process_mask(change_mask, min_change_area)
        
        # Prepare results
        results = {
            "change_mask": change_mask,
            "method": "anychange",
        }
        
        if return_confidence:
            results["confidence"] = confidence
            results["change_statistics"] = self._calculate_change_statistics(change_mask)
        
        return results
    
    def _postprocess_changen2(
        self,
        model_outputs: Dict[str, Any],
        threshold: float,
        return_confidence: bool,
        post_process: bool,
        min_change_area: int
    ) -> Dict[str, Any]:
        """
        Postprocess Changen2 model outputs.
        
        Args:
            model_outputs: Changen2 model outputs
            threshold: Threshold for binary change detection
            return_confidence: Whether to return confidence scores
            post_process: Whether to apply post-processing
            min_change_area: Minimum change area
            
        Returns:
            Postprocessed results
        """
        # Extract logits from model outputs
        logits = model_outputs["change_logits"]  # (batch_size, 1, H, W)
        
        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(logits).squeeze(1)  # (batch_size, H, W)
        
        # Create binary change mask
        change_mask = (probabilities > threshold).float()
        
        # Convert to numpy for post-processing
        change_mask_np = change_mask.cpu().numpy().astype(np.uint8)
        probabilities_np = probabilities.cpu().numpy()
        
        # Apply post-processing if requested
        if post_process:
            change_mask_np = self._post_process_mask(change_mask_np, min_change_area)
        
        # Prepare results
        results = {
            "change_mask": change_mask_np,
            "change_probability": probabilities_np,
            "method": "changen2",
        }
        
        if return_confidence:
            # Calculate confidence as the maximum probability in change regions
            confidence = np.max(probabilities_np) if np.any(change_mask_np) else 0.0
            results["confidence"] = float(confidence)
            results["change_statistics"] = self._calculate_change_statistics(change_mask_np)
        
        return results
    
    def _post_process_mask(
        self, 
        mask: np.ndarray, 
        min_change_area: int = 100
    ) -> np.ndarray:
        """
        Apply post-processing to clean up change mask.
        
        Args:
            mask: Binary change mask
            min_change_area: Minimum area for valid change regions
            
        Returns:
            Post-processed mask
        """
        from skimage import morphology, measure
        
        # Remove small objects
        if min_change_area > 0:
            mask = morphology.remove_small_objects(
                mask.astype(bool), 
                min_size=min_change_area
            ).astype(np.uint8)
        
        # Apply morphological operations to clean up the mask
        kernel = morphology.disk(2)
        mask = morphology.binary_closing(mask, kernel).astype(np.uint8)
        mask = morphology.binary_opening(mask, kernel).astype(np.uint8)
        
        return mask
    
    def _calculate_change_statistics(self, change_mask: np.ndarray) -> Dict[str, Any]:
        """
        Calculate statistics about detected changes.
        
        Args:
            change_mask: Binary change mask
            
        Returns:
            Dictionary containing change statistics
        """
        change_pixels = np.sum(change_mask)
        total_pixels = change_mask.size
        change_percentage = (change_pixels / total_pixels) * 100
        
        return {
            "change_pixels": int(change_pixels),
            "total_pixels": int(total_pixels),
            "change_percentage": float(change_percentage),
        }
    
    def __call__(
        self,
        t1_image: Union[str, Image.Image, np.ndarray],
        t2_image: Union[str, Image.Image, np.ndarray],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform change detection between two images.
        
        Args:
            t1_image: First temporal image (T1)
            t2_image: Second temporal image (T2)
            **kwargs: Additional parameters for preprocessing and postprocessing
            
        Returns:
            Dictionary containing change detection results:
            - change_mask: Binary change mask
            - method: Method used for detection
            - confidence: Overall confidence score (if return_confidence=True)
            - change_statistics: Statistics about detected changes (if return_confidence=True)
            - change_probability: Probability map (for Changen2 method)
        """
        return super().__call__(t1_image, t2_image, **kwargs)
    
    def batch_detect(
        self,
        image_pairs: List[Tuple[Union[str, Image.Image, np.ndarray], 
                               Union[str, Image.Image, np.ndarray]]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Perform batch change detection on multiple image pairs.
        
        Args:
            image_pairs: List of (t1_image, t2_image) pairs
            **kwargs: Additional parameters
            
        Returns:
            List of change detection results
        """
        results = []
        for t1_image, t2_image in image_pairs:
            result = self(t1_image, t2_image, **kwargs)
            results.append(result)
        
        return results
    
    def get_method_info(self) -> Dict[str, Any]:
        """
        Get information about the current change detection method.
        
        Returns:
            Dictionary containing method information
        """
        info = {
            "method": self.method.value,
            "model_type": type(self.model).__name__,
        }
        
        if self.method == ChangeDetectionMethod.ANYCHANGE:
            info.update({
                "description": "Zero-shot change detection using SAM with bitemporal latent matching",
                "strengths": ["No training required", "Works on unseen change types", "Interactive point queries"],
                "best_for": ["Zero-shot scenarios", "Interactive detection", "General change detection"]
            })
        elif self.method == ChangeDetectionMethod.CHANGEN2:
            info.update({
                "description": "Diffusion-based change detection using RSDiT",
                "strengths": ["High accuracy", "Learned representations", "Structured output"],
                "best_for": ["Supervised scenarios", "High-accuracy requirements", "Structured change detection"]
            })
        
        return info 