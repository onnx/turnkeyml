# labels: test_group::turnkey name::ssd300_vgg16 author::torchvision task::Computer_Vision license::bsd-3-clause
# Skip reason: Fails during the use of the Discover Tool in turnkey
"""
https://pytorch.org/vision/stable/models/ssd.html
"""

from turnkeyml.parser import parse
import torch
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights


torch.manual_seed(0)

# Parsing command-line arguments
pretrained, batch_size, num_channels, width, height = parse(
    ["pretrained", "batch_size", "num_channels", "width", "height"]
)


# Model and input configurations
model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT if pretrained else None)
model.eval()
inputs = {"images": torch.ones([batch_size, num_channels, width, height])}


# Call model
model(**inputs)
