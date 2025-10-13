# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Zero-shot Change Detection Pipeline for Changen2

This pipeline enables zero-shot change detection in remote sensing imagery
using the Changen2 model without requiring training data.
"""

import os
import warnings
from typing import Any, Dict, List, Optional, Union, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import requests

from transformers import Pipeline, add_end_docstrings
from transformers.pipelines.base import build_pipeline_init_args
from transformers.utils import is_torch_available, logging

from ..models.changen2 import Changen2Config, Changen2ForChangeDetection

logger = logging.get_logger(__name__)


def load_image(image: Union[str, Image.Image]) -> Image.Image:
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


def preprocess_images(
    t1_image: Union[str, Image.Image, np.ndarray],
    t2_image: Union[str, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (256, 256),
    normalize: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess two temporal images for change detection.
    
    Args:
        t1_image: First temporal image (T1)
        t2_image: Second temporal image (T2)
        target_size: Target size for resizing (height, width)
        normalize: Whether to normalize pixel values to [0, 1]
        
    Returns:
        Tuple of preprocessed tensors (t1_tensor, t2_tensor)
    """
    # Load images if they are paths
    if isinstance(t1_image, str):
        t1_image = load_image(t1_image)
    if isinstance(t2_image, str):
        t2_image = load_image(t2_image)
    
    # Convert to PIL if numpy arrays
    if isinstance(t1_image, np.ndarray):
        t1_image = Image.fromarray(t1_image)
    if isinstance(t2_image, np.ndarray):
        t2_image = Image.fromarray(t2_image)
    
    # Resize images
    t1_image = t1_image.resize(target_size, Image.BILINEAR)
    t2_image = t2_image.resize(target_size, Image.BILINEAR)
    
    # Convert to tensors
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
    
    # Convert to tensors and add batch dimension
    t1_tensor = torch.from_numpy(t1_array).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    t2_tensor = torch.from_numpy(t2_array).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    
    return t1_tensor, t2_tensor


@add_end_docstrings(
    build_pipeline_init_args(has_image_processor=False),
    r"""
        threshold (`float`, *optional*, defaults to 0.5):
            Threshold for binary change detection.
        return_confidence (`bool`, *optional*, defaults to True):
            Whether to return confidence scores along with change masks.
        post_process (`bool`, *optional*, defaults to True):
            Whether to apply post-processing to clean up change masks.
        min_change_area (`int`, *optional*, defaults to 100):
            Minimum area (in pixels) for a change region to be considered valid.
    """,
)
class ChangeDetectionPipeline(Pipeline):
    """
    Zero-shot change detection pipeline using Changen2 model.
    
    This pipeline detects changes between two temporal remote sensing images
    without requiring training data or fine-tuning.
    
    Example:
    
    ```python
    >>> from src.pipelines import ChangeDetectionPipeline
    
    >>> detector = ChangeDetectionPipeline(
    ...     model="path/to/changen2/model",
    ...     device="cuda"
    ... )
    
    >>> # Detect changes between two images
    >>> results = detector(
    ...     t1_image="path/to/t1_image.png",
    ...     t2_image="path/to/t2_image.png"
    ... )
    
    >>> # Results contain change mask and confidence scores
    >>> change_mask = results["change_mask"]
    >>> confidence = results["confidence"]
    ```
    
    This pipeline can be used for various change detection tasks:
    - Urban development monitoring
    - Deforestation detection
    - Disaster damage assessment
    - Agricultural monitoring
    - Infrastructure changes
    """
    
    _load_processor = False
    _load_image_processor = False
    _load_feature_extractor = False
    _load_tokenizer = False
    
    def __init__(
        self,
        model: Union[str, Changen2ForChangeDetection],
        config: Optional[Changen2Config] = None,
        device: Optional[Union[int, str, torch.device]] = None,
        torch_dtype: Optional[Union[str, torch.dtype]] = None,
        **kwargs
    ):
        """
        Initialize the change detection pipeline.
        
        Args:
            model: Path to pretrained model or model instance
            config: Model configuration
            device: Device to run inference on
            torch_dtype: Data type for model tensors
            **kwargs: Additional arguments
        """
        if not is_torch_available():
            raise ImportError("PyTorch is required for this pipeline")
        
        super().__init__(
            model=model,
            config=config,
            device=device,
            torch_dtype=torch_dtype,
            **kwargs
        )
        
        # Set default parameters
        self.threshold = kwargs.get("threshold", 0.5)
        self.return_confidence = kwargs.get("return_confidence", True)
        self.post_process = kwargs.get("post_process", True)
        self.min_change_area = kwargs.get("min_change_area", 100)
        
        # Initialize model if needed
        if isinstance(model, str):
            if config is None:
                config = Changen2Config.from_pretrained(model)
            self.model = Changen2ForChangeDetection.from_pretrained(
                model, 
                config=config,
                torch_dtype=self.torch_dtype
            )
        else:
            self.model = model
        
        # Move model to device
        if device is not None:
            self.model = self.model.to(device)
        
        self.model.eval()
    
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
        target_size: Tuple[int, int] = (256, 256),
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess input images for change detection.
        
        Args:
            t1_image: First temporal image
            t2_image: Second temporal image
            target_size: Target size for resizing
            **kwargs: Additional preprocessing parameters
            
        Returns:
            Dictionary containing preprocessed tensors
        """
        # Preprocess images
        t1_tensor, t2_tensor = preprocess_images(
            t1_image, t2_image, target_size=target_size
        )
        
        # Create input tensor by concatenating T1 and T2 images
        # Changen2 expects input of shape (batch_size, channels, height, width)
        # where channels = in_channels (typically 4 for RGB + label channel)
        input_tensor = torch.cat([t1_tensor, t2_tensor], dim=1)  # (1, 6, H, W)
        
        # Create dummy labels (zeros) for zero-shot inference
        batch_size, _, height, width = input_tensor.shape
        labels = torch.zeros(batch_size, height, width, dtype=torch.long)
        
        # Create dummy timesteps (for diffusion model)
        timesteps = torch.zeros(batch_size, dtype=torch.long)
        
        return {
            "pixel_values": input_tensor,
            "timesteps": timesteps,
            "labels": labels,
        }
    
    def _forward(self, model_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            model_inputs: Preprocessed model inputs
            
        Returns:
            Model outputs
        """
        with torch.no_grad():
            outputs = self.model(**model_inputs)
        
        return outputs
    
    def postprocess(
        self,
        model_outputs: Dict[str, torch.Tensor],
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
            change_mask_np = self._post_process_mask(
                change_mask_np, min_change_area=min_change_area
            )
        
        # Prepare results
        results = {
            "change_mask": change_mask_np,
            "change_probability": probabilities_np,
        }
        
        if return_confidence:
            # Calculate confidence as the maximum probability in change regions
            confidence = np.max(probabilities_np) if np.any(change_mask_np) else 0.0
            results["confidence"] = float(confidence)
            
            # Calculate change statistics
            change_pixels = np.sum(change_mask_np)
            total_pixels = change_mask_np.size
            change_percentage = (change_pixels / total_pixels) * 100
            
            results["change_statistics"] = {
                "change_pixels": int(change_pixels),
                "total_pixels": int(total_pixels),
                "change_percentage": float(change_percentage),
            }
        
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
    
    def __call__(
        self,
        t1_image: Union[str, Image.Image, np.ndarray],
        t2_image: Union[str, Image.Image, np.ndarray],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform zero-shot change detection between two images.
        
        Args:
            t1_image: First temporal image (T1)
            t2_image: Second temporal image (T2)
            **kwargs: Additional parameters for preprocessing and postprocessing
            
        Returns:
            Dictionary containing change detection results:
            - change_mask: Binary change mask
            - change_probability: Probability map
            - confidence: Overall confidence score
            - change_statistics: Statistics about detected changes
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