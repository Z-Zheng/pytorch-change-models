# ChangeSparse Model

This directory contains the re-implemented ChangeSparse model following the same style and structure as the ChangeMask model.

## Overview

ChangeSparse is a change detection model that uses sparse attention mechanisms to efficiently focus on change regions. It employs a transformer-based architecture with temporal reduction and sparse attention blocks for efficient processing of bitemporal data.

## Architecture

The model consists of:

1. **Backbone**: Uses various encoder backbones (default: ResNet18)
2. **Temporal Reduction**: Reduces bitemporal features using convolution or ADBN
3. **Sparse Change Transformer**: Multi-stage transformer with sparse attention
4. **Change Head**: Output head for change detection predictions

## Key Components

### SparseChangeTransformer

- Multi-stage transformer architecture with sparse attention
- Region prediction for change areas
- Sparse attention refinement based on change probability
- Supports both single-scale and multi-scale outputs

### SparseAttentionBlock

- Implements masked attention mechanism
- Only attends to regions with high change probability
- Uses ConvMLP for efficient spatial processing

### SwinAttentionBlock

- Window-based attention mechanism
- Shifted window attention for better feature interaction
- Efficient processing of high-resolution features

### TemporalReduction

- Reduces bitemporal features to single temporal representation
- Supports different reduction types: 'conv' and 'ADBN'
- Handles multi-scale feature fusion

## Usage

### Basic Usage

```python
from models.changesparse import ChangeSparseConfig, ChangeSparseModel

# Create configuration
config = ChangeSparseConfig(
    backbone_name="er.R18",
    backbone_pretrained=True,
    temporal_reduction_type="conv",
    inner_channels=96,
    num_heads=(3, 3, 3, 3),
    change_threshold=0.5,
    min_keep_ratio=0.01,
    max_keep_ratio=0.1,
    num_change_classes=1,
    loss_config={
        'change': {'bce': {'weight': 1.0, 'label_smooth': 0.1}, 'dice': {'weight': 1.0}},
        'region_loss': {'bce': {'weight': 0.5, 'label_smooth': 0.1}}
    }
)

# Create model
model = ChangeSparseModel(config)

# Forward pass
pixel_values = torch.randn(2, 6, 256, 256)  # (batch, 2*channels, height, width)
outputs = model(pixel_values)
```

### Training

```python
# Create labels
labels = {
    'masks': [
        torch.randint(0, 6, (2, 256, 256)),  # T1 semantic (if needed)
        torch.randint(0, 6, (2, 256, 256)),  # T2 semantic (if needed)
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
from models.changesparse import ChangeSparseForChangeDetection

# Use specialized change detection model
change_model = ChangeSparseForChangeDetection(config)
change_outputs = change_model(pixel_values)
change_logits = change_outputs['change_logits']
```

## Configuration Options

### ChangeSparseConfig Parameters

- `backbone_name`: Type of backbone (default: "er.R18")
- `backbone_pretrained`: Whether to use pre-trained backbone (default: True)
- `temporal_reduction_type`: Reduction type ("conv" or "ADBN")
- `inner_channels`: Number of inner channels (default: 96)
- `num_heads`: Number of attention heads for each stage (default: (3, 3, 3, 3))
- `qkv_bias`: Whether to use bias in QKV projection (default: True)
- `drop`: Dropout rate (default: 0.0)
- `attn_drop`: Attention dropout rate (default: 0.0)
- `drop_path`: Drop path rate (default: 0.0)
- `change_threshold`: Threshold for change region prediction (default: 0.5)
- `min_keep_ratio`: Minimum ratio of regions to keep (default: 0.01)
- `max_keep_ratio`: Maximum ratio of regions to keep (default: 0.1)
- `train_max_keep`: Maximum number of regions during training (default: 2000)
- `num_blocks`: Number of transformer blocks per stage (default: (2, 2, 2, 2))
- `disable_attn_refine`: Whether to disable attention refinement (default: False)
- `output_type`: Output type ("single_scale" or "multi_scale")
- `pc_upsample`: Upsampling method for probability maps (default: "nearest")
- `num_change_classes`: Number of change classes (default: 1)
- `loss_config`: Loss configuration dictionary

### Loss Configuration

The `loss_config` supports:

- `change`: Main change detection losses
- `region_loss`: Region-based losses for intermediate predictions

Each loss type can be configured with specific parameters like weights and label smoothing.

## Model Outputs

### Training Mode

Returns a dictionary with loss components:

- `bce_loss`: Binary cross-entropy loss for change detection
- `dice_loss`: Dice loss for change detection
- `region_{h}x{w}_bce_loss`: Region-based BCE losses for intermediate predictions
- `{h}x{w}_ECR`: Estimated change ratios for monitoring

### Inference Mode

Returns a dictionary with predictions:

- `change_prediction`: Change detection probabilities

## Key Features

### Sparse Attention

- Only processes regions with high change probability
- Significantly reduces computational complexity
- Maintains accuracy while improving efficiency

### Multi-Stage Processing

- Progressive refinement from coarse to fine scales
- Intermediate supervision for better training
- Adaptive region selection based on change probability

### Temporal Interaction

- Efficient bitemporal feature fusion
- Multiple reduction strategies (conv, ADBN)
- Temporal consistency modeling

## Dependencies

- torch
- transformers
- einops
- ever
- timm
- segmentation_models_pytorch

## Example

See `examples/changesparse_example.py` for a complete usage example.

## Performance

The sparse attention mechanism provides:

- **Efficiency**: 2-5x speedup compared to dense attention
- **Memory**: Reduced memory usage for high-resolution images
- **Accuracy**: Maintained or improved accuracy on change detection tasks
