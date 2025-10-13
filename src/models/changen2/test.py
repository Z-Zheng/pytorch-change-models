from modeling_changen2 import (
    Changen2Model,
    Changen2ForImageGeneration,
    Changen2ForChangeDetection,
)
from configuration_changen2 import Changen2Config

# Initialize configuration
config = Changen2Config(
    model_type="RSDiT-B/2",  # Choose from: RSDiT-S/2, RSDiT-B/2, RSDiT-L/2, RSDiT-XL/2
    input_size=256,
    in_channels=4,  # e.g., RGBN
    label_channels=1,  # Change mask
)

# For image generation
model = Changen2ForImageGeneration(config)

# For change detection
model = Changen2ForChangeDetection(config)
