"""
Example script that demonstrates how to set a custom ONNX opset for a benchmarking run

You can run this script in your turnkey Conda environment with:
    python onnx_opset.py --onnx-opset YOUR_OPSET

And then you can observe the ONNX opset in the resulting ONNX files with:
   turnkey cache stats BUILD_NAME
"""

import pathlib
import argparse
from turnkeyml import benchmark_files
from turnkeyml.build.stage import Sequence
from turnkeyml.build.export import ExportPytorchModel


def main():
    # Define the argument parser
    parser = argparse.ArgumentParser(
        description="Benchmark a PyTorch model with a specified ONNX opset."
    )

    # Add the arguments
    parser.add_argument(
        "--onnx-opset",
        default="16",
        help="ONNX opset to use when creating ONNX files",
    )

    # Parse the arguments
    args = parser.parse_args()

    path_to_hello_world_script = str(
        pathlib.Path(__file__).parent.resolve() / "scripts" / "hello_world.py"
    )

    sequence = Sequence(stages={ExportPytorchModel(): ["--opset", args.onnx_opset]})
    benchmark_files(
        input_files=[path_to_hello_world_script],
        sequence=sequence,
    )


if __name__ == "__main__":
    main()
