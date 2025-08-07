# Any Change Model

This directory contains the re-implemented Any Change model following the same style and structure as the ChangeMask and ChangeSparse models.

## Overview

Any Change is a zero-shot change detection model that uses the Segment Anything Model (SAM) with bitemporal latent matching. It can perform change detection without requiring training on change data, making it suitable for zero-shot scenarios.

## Architecture

The model consists of:

1. **SAM Backbone**: Uses various SAM model types (ViT-B, ViT-L, ViT-H)
2. **ModularAnyChange**: Modular mask generation and processing
3. **Bitemporal Matching**: Latent space matching for change detection
4. **Point Query Support**: Interactive change detection with point queries

## Key Components

### ModularAnyChange

- Modular interface for SAM-based mask generation
- Configurable hyperparameters for mask generation
- Support for various output formats (RLE, COCO RLE, binary masks)
- Efficient batch processing of point queries

### Bitemporal Latent Matching

- Compares latent representations between temporal images
- Uses cosine similarity for change confidence scoring
- Supports automatic thresholding with Otsu's method
- Configurable change confidence thresholds

### Point Query Mechanism

- Single point queries for targeted change detection
- Multi-point queries for complex change scenarios
- Object similarity filtering for improved accuracy
- Temporal-aware point processing

## Usage

### Basic Usage

```python
from models.anychange import AnyChangeConfig, AnyChangeModel

# Create configuration
config = AnyChangeConfig(
    model_type="vit_b",
    sam_checkpoint="./sam_weights/sam_vit_b_01ec64.pth",
    points_per_side=32,
    pred_iou_thresh=0.5,
    stability_score_thresh=0.95,
    change_confidence_threshold=155,
    auto_threshold=False,
    use_normalized_feature=True,
    area_thresh=0.8,
    object_sim_thresh=60,
    bitemporal_match=True,
)

# Create model
model = AnyChangeModel(config)

# Forward pass
import numpy as np
img1 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
img2 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
outputs = model(img1, img2)
```

### Automatic Change Detection

```python
# Automatic mode - detect all changes
changemasks, t1_masks, t2_masks = model.forward(img1, img2)
```

### Point Query Mode

```python
# Single point query
changemasks = model.single_point_match(
    xy=[926, 44],
    temporal=2,  # 1 for T1, 2 for T2
    img1=img1,
    img2=img2
)

# Multi-point query
xyts = [[926, 44, 2], [100, 100, 1]]  # [x, y, temporal]
changemasks = model.multi_points_match(xyts, img1, img2)
```

### Hyperparameter Customization

```python
# Set hyperparameters
model.set_hyperparameters(
    change_confidence_threshold=145,
    use_normalized_feature=True,
    bitemporal_match=True,
    object_sim_thresh=60,
)

# Customize mask generator
model.make_mask_generator(
    points_per_side=32,
    stability_score_thresh=0.95,
    pred_iou_thresh=0.5,
)
```

### Change Detection Only

```python
from models.anychange import AnyChangeForChangeDetection

# Use specialized change detection model
change_model = AnyChangeForChangeDetection(config)
change_outputs = change_model(img1, img2)
change_masks = change_outputs['change_masks']
```

## Configuration Options

### AnyChangeConfig Parameters

- `model_type`: SAM model type ("vit_b", "vit_l", "vit_h") (default: "vit_b")
- `sam_checkpoint`: Path to SAM checkpoint file
- `points_per_side`: Number of points per side for mask generation (default: 32)
- `points_per_batch`: Number of points per batch (default: 64)
- `pred_iou_thresh`: IoU threshold for mask prediction (default: 0.5)
- `stability_score_thresh`: Stability score threshold (default: 0.95)
- `stability_score_offset`: Stability score offset (default: 1.0)
- `box_nms_thresh`: Box NMS threshold (default: 0.7)
- `min_mask_region_area`: Minimum mask region area (default: 0)
- `change_confidence_threshold`: Change confidence threshold in degrees (default: 155)
- `auto_threshold`: Whether to use automatic thresholding (default: False)
- `use_normalized_feature`: Whether to use normalized features (default: True)
- `area_thresh`: Area threshold for mask filtering (default: 0.8)
- `match_hist`: Whether to match histograms (default: False)
- `object_sim_thresh`: Object similarity threshold in degrees (default: 60)
- `bitemporal_match`: Whether to use bitemporal matching (default: True)

## Model Outputs

### Automatic Mode

Returns a tuple of:

- `change_masks`: MaskData containing detected change regions
- `t1_mask_data`: MaskData for T1 image masks
- `t2_mask_data`: MaskData for T2 image masks

### Point Query Mode

Returns:

- `change_masks`: MaskData containing change regions matching the query

### Evaluation Format

```python
predictions = model.to_eval_format_predictions(change_masks)
# Returns: {'boxes': tensor, 'scores': tensor, 'labels': tensor, 'masks': list}
```

## Key Features

### Zero-Shot Capability

- No training required on change detection data
- Works on unseen change types and data distributions
- Leverages SAM's pre-trained knowledge

### Interactive Detection

- Point query mechanism for targeted change detection
- Support for both single and multi-point queries
- Temporal-aware point processing

### Efficient Processing

- Modular design for easy customization
- Cached embedding support for faster inference
- Batch processing for multiple queries

### Flexible Configuration

- Extensive hyperparameter tuning options
- Support for different SAM model sizes
- Configurable mask generation strategies

## Dependencies

- torch
- transformers
- numpy
- skimage
- safetensors
- libs.segment_any.segment_anything (SAM components)

## Example

See `examples/anychange_example.py` for a complete usage example.

## Performance

The zero-shot approach provides:

- **Generality**: Works on unseen change types
- **Efficiency**: No training required
- **Accuracy**: Competitive performance with supervised methods
- **Flexibility**: Supports various input formats and query types

## Citation

If you use this implementation, please cite the original Any Change paper:

```bibtex
@inproceedings{
zheng2024anychange,
title={Any Change},
author={Zhuo Zheng and Yanfei Zhong and Liangpei Zhang and Stefano Ermon},
booktitle={Advances in Neural Information Processing Systems},
year={2024},
}
```
