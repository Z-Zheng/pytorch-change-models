# ChangeStar2 Model

This directory contains the transformers-style implementation of the ChangeStar2 model for change detection with semantic segmentation.

## Overview

ChangeStar2 is a re-implementation of the ChangeStar2 model following the transformers library pattern. It provides a clean, modular interface for change detection tasks with optional semantic segmentation capabilities.

## Architecture

The model consists of several key components:

1. **Segmentation Backbone**: Feature extraction using various segmentation models (Farseg, DeepLabV3, etc.)
2. **Semantic Classifier**: Performs semantic segmentation on individual time steps
3. **Change Detector**: TSMTDM (Temporal Symmetric Module + Temporal Difference Module) for change detection
4. **Target Generator**: Generates pseudo-labels during training for self-supervision

## Key Features

- **Bitemporal Processing**: Handles pairs of images from different time periods
- **Semantic Segmentation**: Optional semantic segmentation capabilities
- **Change Detection**: Binary or multi-class change detection
- **Self-Supervised Learning**: Target generation for training without labels
- **Modular Design**: Easy to customize and extend

## Usage

### Basic Usage

```python
from models.changestar2 import ChangeStar2Config, ChangeStar2ForChangeDetection

# Initialize configuration
config = ChangeStar2Config(
    segmentation_config={
        "model_type": "farseg",
        "backbone": {
            "resnet_type": "resnet50",
            "pretrained": True,
        }
    },
    change_type="binary"
)

# Initialize model
model = ChangeStar2ForChangeDetection(config)

# Prepare input (bitemporal images)
bitemporal_input = torch.cat([t1_image, t2_image], dim=0).unsqueeze(0)

# Run inference
with torch.no_grad():
    outputs = model(pixel_values=bitemporal_input)
    change_prediction = outputs["change_logits"]
```

### Training

```python
# Prepare labels for training
labels = {
    "masks": [t1_mask, t2_mask, change_mask]  # Semantic masks and change mask
}

# Forward pass with labels
outputs = model(pixel_values=bitemporal_input, labels=labels)
loss = outputs["loss"]
```

## Configuration Options

### Segmentation Configuration

- `model_type`: Type of segmentation model ('farseg', 'deeplabv3', 'pspnet', etc.)
- `backbone`: Backbone network configuration
- `head`: Segmentation head configuration

### Change Detector Configuration

- `name`: Detector type ('TSMTDM')
- `in_channels`: Input feature channels
- `scale`: Output scale factor
- `tsm_cfg`: Temporal Symmetric Module configuration
- `tdm_cfg`: Temporal Difference Module configuration

### Loss Configuration

- `change`: Change detection loss settings
- `semantic`: Semantic segmentation loss settings
- `symmetry_loss`: Whether to use symmetry loss

## Model Variants

- **ChangeStar2Model**: Base model with full outputs
- **ChangeStar2ForChangeDetection**: Specialized for change detection tasks

## Differences from Original Implementation

1. **Transformers-style Interface**: Follows Hugging Face transformers patterns
2. **Configuration-based**: All parameters configurable through config objects
3. **Clean API**: Simplified forward pass with clear input/output structure
4. **Better Documentation**: Comprehensive docstrings and type hints
5. **Modular Design**: Easier to extend and customize

## Dependencies

- PyTorch
- Transformers
- Ever (for segmentation modules)
- Timm (for ConvNeXt blocks)
- Einops (for tensor operations)

## Examples

See the Jupyter notebook `examples/changestar2_pretrained_changestar2_inference_demo.ipynb` for a complete usage example.
