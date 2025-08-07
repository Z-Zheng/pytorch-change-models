# Unified Change Detection Pipeline

This repository provides a unified interface for change detection using multiple state-of-the-art models:

- **AnyChange**: Zero-shot change detection using SAM with bitemporal latent matching
- **Changen2**: Diffusion-based change detection using RSDiT

## Overview

The unified pipeline provides a consistent interface for different change detection approaches, allowing users to easily switch between methods based on their specific requirements and available resources.

## Key Features

### 🎯 **Unified Interface**

- Single pipeline class supporting multiple change detection methods
- Consistent input/output formats across different models
- Automatic method detection and model initialization

### 🔧 **Flexible Configuration**

- Easy model creation with factory pattern
- Configurable hyperparameters for each method
- Support for custom model configurations

### 🚀 **Multiple Use Cases**

- **AnyChange**: Perfect for zero-shot scenarios with no training data
- **Changen2**: Ideal for high-accuracy supervised scenarios
- **Auto-detection**: Automatically selects the best method based on available resources

### 📊 **Rich Output**

- Binary change masks
- Confidence scores
- Change statistics
- Probability maps (for Changen2)
- Method information and recommendations

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd pytorch-change-models

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

### Basic Usage

```python
from src.pipelines import UnifiedChangeDetectionPipeline
from src.models import ChangeDetectionModelFactory

# Method 1: Direct pipeline creation
pipeline = UnifiedChangeDetectionPipeline(
    method="anychange",
    model_config={"model_type": "vit_b"}
)

# Method 2: Using model factory
model = ChangeDetectionModelFactory.create_model("anychange")
pipeline = UnifiedChangeDetectionPipeline(method="anychange", model=model)

# Detect changes
results = pipeline(t1_image, t2_image)
print(f"Change percentage: {results['change_statistics']['change_percentage']:.2f}%")
```

### Different Methods

```python
# AnyChange for zero-shot detection
anychange_pipeline = UnifiedChangeDetectionPipeline(
    method="anychange",
    model_config={
        "model_type": "vit_b",
        "change_confidence_threshold": 155,
        "auto_threshold": False,
    }
)

# Changen2 for diffusion-based detection
changen2_pipeline = UnifiedChangeDetectionPipeline(
    method="changen2",
    model_config={
        "model_type": "RSDiT-B/2",
        "input_size": 256,
    }
)

# Auto-detect method
auto_pipeline = UnifiedChangeDetectionPipeline(
    method="auto",
    model_path="path/to/pretrained/model"
)
```

## Model Comparison

| Feature               | AnyChange           | Changen2            |
| --------------------- | ------------------- | ------------------- |
| **Training Required** | ❌ Zero-shot        | ✅ Pre-trained      |
| **Accuracy**          | Good                | Excellent           |
| **Speed**             | Fast                | Moderate            |
| **Memory Usage**      | Low                 | High                |
| **Best For**          | Zero-shot scenarios | High-accuracy tasks |
| **Interactive**       | ✅ Point queries    | ❌ No               |
| **Generalization**    | ✅ Unseen changes   | ⚠️ Domain-specific  |

## Detailed Usage

### 1. AnyChange Method

**Best for**: Zero-shot scenarios, interactive detection, general change detection

```python
from src.pipelines import UnifiedChangeDetectionPipeline

# Create AnyChange pipeline
pipeline = UnifiedChangeDetectionPipeline(
    method="anychange",
    model_config={
        "model_type": "vit_b",  # or "vit_l", "vit_h"
        "sam_checkpoint": "./sam_weights/sam_vit_b_01ec64.pth",
        "change_confidence_threshold": 155,  # in degrees
        "auto_threshold": False,
        "use_normalized_feature": True,
        "area_thresh": 0.8,
        "object_sim_thresh": 60,
        "bitemporal_match": True,
    }
)

# Detect changes
results = pipeline(t1_image, t2_image)

# Access results
change_mask = results["change_mask"]
confidence = results["confidence"]
statistics = results["change_statistics"]
```

**Key Parameters**:

- `change_confidence_threshold`: Threshold for change detection (degrees)
- `auto_threshold`: Use automatic thresholding with Otsu's method
- `area_thresh`: Area threshold for mask filtering
- `object_sim_thresh`: Object similarity threshold

### 2. Changen2 Method

**Best for**: High-accuracy scenarios, supervised learning, structured output

```python
# Create Changen2 pipeline
pipeline = UnifiedChangeDetectionPipeline(
    method="changen2",
    model_config={
        "model_type": "RSDiT-B/2",  # or "RSDiT-S/2", "RSDiT-L/2", "RSDiT-XL/2"
        "input_size": 256,
        "patch_size": 2,
        "in_channels": 4,
        "label_channels": 1,
        "hidden_size": 768,
        "depth": 12,
        "num_heads": 12,
    }
)

# Detect changes
results = pipeline(t1_image, t2_image)

# Access results
change_mask = results["change_mask"]
change_probability = results["change_probability"]
confidence = results["confidence"]
```

**Key Parameters**:

- `model_type`: RSDiT model variant
- `input_size`: Input image size
- `in_channels`: Number of input channels
- `hidden_size`: Hidden dimension size

### 3. Model Factory

The model factory provides a convenient way to create models with default configurations:

```python
from src.models import ChangeDetectionModelFactory, ModelType

# Get default configuration
anychange_config = ChangeDetectionModelFactory.get_default_config("anychange")
changen2_config = ChangeDetectionModelFactory.get_default_config("changen2")

# Create models
anychange_model = ChangeDetectionModelFactory.create_model("anychange", anychange_config)
changen2_model = ChangeDetectionModelFactory.create_model("changen2", changen2_config)

# List available models
models_info = ChangeDetectionModelFactory.list_available_models()
for model_type, info in models_info.items():
    print(f"{model_type}: {info['description']}")
```

## Advanced Features

### Batch Processing

```python
# Process multiple image pairs
image_pairs = [(t1_1, t2_1), (t1_2, t2_2), (t1_3, t2_3)]
results = pipeline.batch_detect(image_pairs)

for i, result in enumerate(results):
    print(f"Pair {i+1}: {result['change_statistics']['change_percentage']:.2f}% change")
```

### Custom Post-processing

```python
# Custom post-processing parameters
results = pipeline(
    t1_image, t2_image,
    threshold=0.3,  # Lower threshold for more sensitive detection
    min_change_area=50,  # Minimum area for valid changes
    post_process=True,  # Apply morphological operations
    return_confidence=True  # Include confidence scores
)
```

### Method Information

```python
# Get information about the current method
info = pipeline.get_method_info()
print(f"Method: {info['method']}")
print(f"Description: {info['description']}")
print(f"Strengths: {info['strengths']}")
```

## Input Formats

The pipeline accepts various input formats:

```python
# File paths
results = pipeline("path/to/t1.png", "path/to/t2.png")

# PIL Images
from PIL import Image
t1_pil = Image.open("t1.png")
t2_pil = Image.open("t2.png")
results = pipeline(t1_pil, t2_pil)

# NumPy arrays
import numpy as np
t1_array = np.array(t1_pil)
t2_array = np.array(t2_pil)
results = pipeline(t1_array, t2_array)

# URLs
results = pipeline("https://example.com/t1.jpg", "https://example.com/t2.jpg")
```

## Output Format

All methods return a consistent output format:

```python
{
    "change_mask": np.ndarray,  # Binary change mask
    "method": str,  # Method used ("anychange" or "changen2")
    "confidence": float,  # Overall confidence score (if return_confidence=True)
    "change_statistics": {  # Statistics (if return_confidence=True)
        "change_pixels": int,
        "total_pixels": int,
        "change_percentage": float,
    },
    "change_probability": np.ndarray,  # Probability map (Changen2 only)
}
```

## Examples

See `examples/unified_change_detection_example.py` for comprehensive examples including:

1. **Basic Usage**: Simple change detection with different methods
2. **Model Factory**: Using the factory pattern for model creation
3. **Advanced Usage**: Custom configurations and parameters
4. **Batch Processing**: Processing multiple image pairs
5. **Method Comparison**: Comparing different methods on the same data

Run the examples:

```bash
python examples/unified_change_detection_example.py
```

## Performance Tips

### AnyChange

- Use `vit_b` for faster inference, `vit_h` for higher accuracy
- Adjust `change_confidence_threshold` based on your data
- Enable `auto_threshold` for automatic threshold optimization
- Use `area_thresh` to filter out small changes

### Changen2

- Use smaller models (`RSDiT-S/2`) for faster inference
- Use larger models (`RSDiT-XL/2`) for higher accuracy
- Adjust `input_size` based on your image resolution
- Consider using gradient checkpointing for memory efficiency

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Use smaller models or reduce input size
2. **SAM checkpoint not found**: Download SAM checkpoints or use different model type
3. **Import errors**: Ensure all dependencies are installed correctly
4. **Poor detection quality**: Adjust thresholds and parameters for your specific data

### Getting Help

- Check the example scripts for usage patterns
- Review the model documentation for parameter explanations
- Ensure you have the correct dependencies installed
- Try different model configurations for your specific use case

## Citation

If you use this unified pipeline in your research, please cite the original papers:

```bibtex
@inproceedings{zheng2024anychange,
    title={Any Change},
    author={Zhuo Zheng and Yanfei Zhong and Liangpei Zhang and Stefano Ermon},
    booktitle={Advances in Neural Information Processing Systems},
    year={2024},
}

@inproceedings{zheng2024changen2,
    title={Changen2},
    author={Zhuo Zheng and Yanfei Zhong and Liangpei Zhang and Stefano Ermon},
    booktitle={Advances in Neural Information Processing Systems},
    year={2024},
}
```

## License

This project is licensed under the same license as the original models. Please refer to the LICENSE file for details.
