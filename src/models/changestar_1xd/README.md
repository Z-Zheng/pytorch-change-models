# ChangeStar1xd Model

This directory contains the ChangeStar1xd model implementation with Hugging Face `from_pretrained` support.

## Features

- **Change Detection**: Detect changes between two temporal images
- **Semantic Segmentation**: Optional semantic segmentation for both temporal images
- **Hugging Face Integration**: Full support for `from_pretrained` and `save_pretrained` methods
- **Auto Classes**: Compatible with `AutoConfig` and `AutoModel`

## Usage

### Basic Usage

```python
from src.models.changestar_1xd import ChangeStar1xdConfig, ChangeStar1xdModel

# Create configuration
config = ChangeStar1xdConfig(
    encoder_type="resnet",
    encoder_params={
        "backbone": "resnet50",
        "out_channels": 256
    },
    in_channels=3,
    out_channels=256,
    num_change_classes=1,
    temporal_symmetric=True
)

# Create model
model = ChangeStar1xdModel(config)

# Forward pass
import torch
batch_size = 2
channels = 6  # 2 * 3 (t1 + t2 images)
height, width = 256, 256
dummy_input = torch.randn(batch_size, channels, height, width)
outputs = model(dummy_input)
```

### Using from_pretrained

#### Load from Local Directory

```python
from src.models.changestar_1xd import ChangeStar1xdModel

# Load model from local directory
model = ChangeStar1xdModel.from_pretrained("./path/to/saved/model")
```

#### Load from Hugging Face Hub

```python
from src.models.changestar_1xd import ChangeStar1xdModel

# Load model from Hugging Face Hub
model = ChangeStar1xdModel.from_pretrained("username/changestar-1xd")
```

#### Using Auto Classes

```python
from transformers import AutoConfig, AutoModel
from src.models import auto  # This registers the model

# Load using AutoConfig and AutoModel
config = AutoConfig.from_pretrained("./path/to/saved/model")
model = AutoModel.from_pretrained("./path/to/saved/model")
```

### Saving Models

```python
from src.models.changestar_1xd import ChangeStar1xdConfig, ChangeStar1xdModel

# Create and save model
config = ChangeStar1xdConfig(...)
model = ChangeStar1xdModel(config)

# Save to local directory
model.save_pretrained("./my_model")
config.save_pretrained("./my_model")
```

## Model Architecture

The ChangeStar1xd model consists of:

1. **Encoder**: Extracts features from bitemporal images (default: ResNet)
2. **Change Detection Head**: Detects changes between temporal images
3. **Semantic Segmentation Head**: Optional semantic segmentation for each temporal image

### Configuration Parameters

- `encoder_type`: Type of encoder to use (default: "resnet")
- `encoder_params`: Parameters for the encoder
- `bitemporal_forward`: Whether to use bitemporal forward pass
- `in_channels`: Number of input channels (default: 3)
- `out_channels`: Number of output channels (default: 256)
- `temporal_symmetric`: Whether to use temporal symmetric processing (default: True)
- `num_semantic_classes`: Number of semantic classes (optional)
- `num_change_classes`: Number of change classes (default: 1)
- `loss_config`: Loss configuration dictionary

### Input Format

The model expects input in the format:

- Shape: `(batch_size, 2 * channels, height, width)`
- Where the first `channels` dimensions are from time T1 and the second `channels` dimensions are from time T2

### Output Format

The model outputs a dictionary with:

- `change_prediction`: Change detection predictions
- `t1_semantic_prediction`: T1 semantic segmentation (if enabled)
- `t2_semantic_prediction`: T2 semantic segmentation (if enabled)

## Training

```python
# Training mode
model.train()
outputs = model(pixel_values, labels=labels)
loss = outputs["loss"]

# Inference mode
model.eval()
with torch.no_grad():
    outputs = model(pixel_values)
    change_pred = outputs["logits"]["change_prediction"]
```

## Example

See `examples/load_changestar_1xd.py` for a complete example demonstrating all features.

## Requirements

- PyTorch
- Transformers
- Ever (for encoder and loss functions)
- Einops (for tensor operations)
