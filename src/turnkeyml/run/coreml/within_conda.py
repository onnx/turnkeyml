import argparse
import json
import os
import time
import numpy as np
import coremltools as ct
from pathlib import Path


def run_coreml_profile(
    coreml_file_path: str,
    iterations_file: str,
    iterations: int,
):
    # Run the provided onnx model using onnxruntime and measure average latency

    per_iteration_latency = []
    
    # Load the CoreML model
    model = ct.models.MLModel(coreml_file_path)

    # Get inputs
    inputs_path = os.path.join(Path(coreml_file_path).parents[2],"inputs.npy")
    input_data = np.load(inputs_path, allow_pickle=True)[0]

    # Change input keys to match model
    input_data = {key + '_1': value for key, value in input_data.items()}

    # Run model for a certain number of iterations
    for _ in range(iterations):
        start = time.perf_counter()
        model.predict(input_data)
        end = time.perf_counter()
        iteration_latency = end - start
        per_iteration_latency.append(iteration_latency)

    with open(iterations_file, "w", encoding="utf-8") as out_file:
        json.dump(per_iteration_latency, out_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # Parse Inputs
    parser = argparse.ArgumentParser(description="Execute models using coreml")
    parser.add_argument(
        "--coreml-file",
        required=True,
        help="Path where the coreml file is located",
    )
    parser.add_argument(
        "--iterations-file",
        required=True,
        help="File in which to place the per-iteration execution timings",
    )
    parser.add_argument(
        "--iterations",
        required=True,
        type=int,
        help="Number of times to execute the received onnx model",
    )
    args = parser.parse_args()

    run_coreml_profile(
        coreml_file_path=args.coreml_file,
        iterations_file=args.iterations_file,
        iterations=args.iterations,
    )
