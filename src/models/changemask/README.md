# ChangeMask Model

This directory contains the re-implemented ChangeMask model following the same style and structure as the ChangeStar1xd model.

## Overview

ChangeMask is a change detection model that performs simultaneous change detection and semantic segmentation using a U-Net style architecture with temporal interaction modules.

## Architecture

The model consists of:

1. **Encoder**: Uses segmentation_models_pytorch encoders (default: EfficientNet-B0)
2. **Semantic Decoder**: U-Net decoder for semantic segmentation
3. **Change Decoder**: U-Net decoder for change detection
4. **Temporal Symmetric Transformer**: Handles temporal interaction between T1 and T2 features

## Key Components

### TemporalSymmetricTransformer

- Processes bitemporal features using 3D convolutions or 1+2D convolutions
- Supports symmetric fusion (additive or multiplicative)
- Handles both single and multi-scale features

### SpatioTemporalInteraction

- Implements the core temporal interaction mechanism
- Supports different interaction types: `conv3d` and `conv1plus2d`
- Includes batch normalization and ReLU activation

## Usage

### Basic Usage

```python
from models.changemask import ChangeMaskConfig, ChangeMaskModel

# Create configuration
config = ChangeMaskConfig(
    encoder_type="efficientnet-b0",
    encoder_weights="imagenet",
    num_semantic_classes=6,
    num_change_classes=1,
    loss_config={
        't1': {'ce': {}, 'dice': {}},
        't2': {'ce': {}, 'dice': {}},
        'change': {'bce': {'ls': 0.1}, 'dice': {'gamma': 1.0}},
        'sc': {}
    }
)

# Create model
model = ChangeMaskModel(config)

# Forward pass
pixel_values = torch.randn(2, 6, 256, 256)  # (batch, 2*channels, height, width)
outputs = model(pixel_values)
```

### Training

```python
# Create labels
labels = {
    'masks': [
        torch.randint(0, 6, (2, 256, 256)),  # T1 semantic
        torch.randint(0, 6, (2, 256, 256)),  # T2 semantic
        torch.randint(0, 2, (2, 256, 256)).float(),  # Change mask
    ]
}

# Training forward pass
model.train()
outputs = model(pixel_values, labels=labels)
loss = outputs['loss']
```

### Change Detection Only

```python
from models.changemask import ChangeMaskForChangeDetection

# Use specialized change detection model
change_model = ChangeMaskForChangeDetection(config)
change_outputs = change_model(pixel_values)
change_logits = change_outputs['change_logits']
```

## Configuration Options

### ChangeMaskConfig Parameters

- `encoder_type`: Type of encoder (default: "efficientnet-b0")
- `encoder_weights`: Pre-trained weights (default: "imagenet")
- `decoder_channels`: List of decoder channel sizes (default: [256, 128, 64, 32, 16])
- `temporal_interaction_type`: Interaction type ("conv3d" or "conv1plus2d")
- `temporal_kernel_size`: Kernel size for temporal interaction (default: 3)
- `temporal_dilation`: Dilation for temporal interaction (default: 1)
- `temporal_symmetric_fusion`: Fusion type ("add", "mul", or None)
- `num_semantic_classes`: Number of semantic classes (default: 6)
- `num_change_classes`: Number of change classes (default: 1)
- `loss_config`: Loss configuration dictionary

### Loss Configuration

The `loss_config` supports:

- `t1`: T1 semantic segmentation losses
- `t2`: T2 semantic segmentation losses
- `change`: Change detection losses
- `sc`: Semantic consistency loss

Each loss type can be configured with specific parameters like label smoothing (`ls`) and gamma values.

## Model Outputs

### Training Mode

Returns a dictionary with loss components:

- `s1_ce_loss`: T1 cross-entropy loss
- `s1_dice_loss`: T1 dice loss
- `s2_ce_loss`: T2 cross-entropy loss
- `s2_dice_loss`: T2 dice loss
- `c_bce_loss`: Change binary cross-entropy loss
- `c_dice_loss`: Change dice loss
- `sc_mse_loss`: Semantic consistency MSE loss

### Inference Mode

Returns a dictionary with predictions:

- `t1_semantic_prediction`: T1 semantic segmentation probabilities
- `t2_semantic_prediction`: T2 semantic segmentation probabilities
- `change_prediction`: Change detection probabilities

## Dependencies

- torch
- transformers
- einops
- ever
- segmentation_models_pytorch

## Example

See `examples/changemask_example.py` for a complete usage example.
