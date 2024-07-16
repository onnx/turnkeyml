# labels: test_group::turnkey name::keypointrcnn_resnet50_fpn author::torchvision task::Computer_Vision license::bsd-3-clause
# Skip reason: Fails during the discover tool of turnkey
"""
https://pytorch.org/vision/stable/models/keypoint_rcnn.html
"""

from turnkeyml.parser import parse
import torch
from torchvision.models.detection import (
    keypointrcnn_resnet50_fpn,
    KeypointRCNN_ResNet50_FPN_Weights,
)


torch.manual_seed(0)

# Parsing command-line arguments
pretrained, batch_size, num_channels, width, height = parse(
    ["pretrained", "batch_size", "num_channels", "width", "height"]
)


# Model and input configurations
model = keypointrcnn_resnet50_fpn(
    weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
)
model.eval()
inputs = {"images": torch.ones([batch_size, num_channels, width, height])}


# Call model
model(**inputs)
