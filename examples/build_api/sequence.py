"""
    This example illustrates how to set the build sequence. In this
    example, we apply onnx optimization, whereas in most other examples
    we only perform a torch to onnx export.
"""

import torch
from turnkeyml import build_model
import turnkeyml.build.export as export
import turnkeyml.build.stage as stage
from turnkeyml.analyze.discover import Discover


torch.manual_seed(0)


# Define model class
class SmallModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SmallModel, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        output = self.fc(x)
        return output


# Instantiate model and generate inputs
input_size = 10
output_size = 5

pytorch_model = SmallModel(input_size, output_size)
inputs = {"x": torch.rand(input_size, dtype=torch.float32)}

onnx_sequence = stage.Sequence(
    stages={
        Discover(): [],
        export.ExportPytorchModel(): [],
        export.OptimizeOnnxModel(): [],
    }
)

# Build model
build_model(sequence=onnx_sequence, model=pytorch_model, inputs=inputs)
