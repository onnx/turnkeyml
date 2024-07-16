"""
    This example demonstrates the difference between the build_model() argument,
    monitor, when set to "True" (its default value) and then "False".
"""

import torch
from turnkeyml import build_model
from turnkeyml.sequence import Sequence
from turnkeyml.tools.export import ExportPytorchModel

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
inputs = {"x": torch.rand(input_size)}

# Build pytorch_model with `monitor` explicitly set to True
sequence = Sequence(tools={ExportPytorchModel(): []})
print("build_model() will now build the model with the monitor enabled...")
build_model(
    sequence=sequence,
    model=pytorch_model,
    inputs=inputs,
    monitor=True,
    build_name="monitor_enabled",
)

# Rebuild pytorch_model with the monitor disabled
print("build_model() will now build the model with the monitor disabled...")
build_model(
    sequence=sequence,
    model=pytorch_model,
    inputs=inputs,
    monitor=False,
    build_name="monitor_disabled",
)

print("Example no_monitor.py finished")
