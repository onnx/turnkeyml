import argparse
import re
import os
import math
import json
import time
import numpy as np
import onnxruntime as ort
import onnx
from onnx.external_data_helper import (
    _get_all_tensors,
    uses_external_data,
    ExternalDataInfo,
)
from onnx.onnx_pb import ModelProto, TensorProto


def load_fake_data_for_model(model: ModelProto) -> None:
    """Loads fake tensors into model

    Arguments:
        model: ModelProto to load fake data to
    """
    for tensor in _get_all_tensors(model):
        if uses_external_data(tensor):

            # Load fake data for tensor
            info = ExternalDataInfo(tensor)
            if info.length:
                tensor.raw_data = os.urandom(info.length)
            else:
                tensor.raw_data = os.urandom()

            # Change the state of tensors and remove external data
            tensor.data_location = TensorProto.DEFAULT
            del tensor.external_data[:]


def run_ort_profile(
    onnx_file_path: str,
    iterations_file: str,
    iterations: int,
):
    # Run the provided onnx model using onnxruntime and measure average latency

    per_iteration_latency = []
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    onnx_session = None
    if False:
        onnx_session = ort.InferenceSession(onnx_file_path, sess_options)
    else:
        onnx_model = onnx.load(onnx_file_path, load_external_data=False)
        load_fake_data_for_model(onnx_model)
        serialized_model = onnx_model.SerializeToString()
        onnx_session = ort.InferenceSession(serialized_model, sess_options)

    sess_input = onnx_session.get_inputs()
    input_feed = dummy_inputs(sess_input)
    output_name = onnx_session.get_outputs()[0].name

    for _ in range(iterations):
        start = time.perf_counter()
        onnx_session.run([output_name], input_feed)
        end = time.perf_counter()
        iteration_latency = end - start
        per_iteration_latency.append(iteration_latency)

    with open(iterations_file, "w", encoding="utf-8") as out_file:
        json.dump(per_iteration_latency, out_file, ensure_ascii=False, indent=4)


def dummy_inputs(sess_input) -> dict:
    # Generate dummy inputs of the expected shape and type for the input model
    input_stats = []
    for _idx, input_ in enumerate(range(len(sess_input))):
        input_name = sess_input[input_].name
        input_shape = sess_input[input_].shape

        # TODO: Use onnx update_inputs_outputs_dims to automatically freeze models
        for dim in input_shape:
            if isinstance(dim, str) is True or math.isnan(dim) is True:
                raise AssertionError(
                    "Error: Model has dynamic inputs. Freeze the graph and try again"
                )

        input_type = sess_input[input_].type
        input_stats.append([input_name, input_shape, input_type])

    input_feed = {}
    for stat in input_stats:
        dtype_str = re.search(r"\((.*)\)", stat[2])
        assert dtype_str is not None
        datatype = dtype_ort2str(dtype_str.group(1))
        input_feed[stat[0]] = np.random.rand(*stat[1]).astype(datatype)
    return input_feed


def dtype_ort2str(dtype_str: str):
    if dtype_str == "float16":
        datatype = "float16"
    elif dtype_str == "float":
        datatype = "float32"
    elif dtype_str == "double":
        datatype = "float64"
    elif dtype_str == "long":
        datatype = "int64"
    else:
        datatype = dtype_str
    return datatype


if __name__ == "__main__":
    # Parse Inputs
    parser = argparse.ArgumentParser(description="Execute models using onnxruntime")
    parser.add_argument(
        "--onnx-file",
        required=True,
        help="Path where the ONNX file is located",
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

    run_ort_profile(
        onnx_file_path=args.onnx_file,
        iterations_file=args.iterations_file,
        iterations=args.iterations,
    )
