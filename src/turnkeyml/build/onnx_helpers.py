"""
Helper functions for dealing with ONNX files and ONNX models
"""

import os
from typing import Tuple
import re
import math
import numpy as np
import onnx
import onnxruntime as ort
import turnkeyml.common.exceptions as exp
from onnx.external_data_helper import (
    _get_all_tensors,
    uses_external_data,
    ExternalDataInfo,
)
from onnx.onnx_pb import ModelProto, TensorProto


def parameter_count(model):
    weights = model.graph.initializer
    parameter_count = 0

    for w in weights:
        weight = onnx.numpy_helper.to_array(w)
        parameter_count += np.prod(weight.shape)
    return parameter_count


def io_bytes(onnx_path: str) -> Tuple[int, int]:
    """Return the number of bytes of each of the inputs and outputs"""
    # pylint: disable = no-member

    def elem_type_to_bytes(elem_type) -> int:
        """
        Convert ONNX's elem_type to the number of bytes used by
        hardware to send that specific datatype through PCIe
        """
        if (
            elem_type == onnx.TensorProto.DataType.UINT8
            or elem_type == onnx.TensorProto.DataType.INT8
            or elem_type == onnx.TensorProto.DataType.BOOL
        ):
            # Each bool requires an entire byte
            return 1
        elif (
            elem_type == onnx.TensorProto.DataType.UINT16
            or elem_type == onnx.TensorProto.DataType.INT16
            or elem_type == onnx.TensorProto.DataType.FLOAT16
        ):
            return 2
        if (
            elem_type == onnx.TensorProto.DataType.FLOAT
            or elem_type == onnx.TensorProto.DataType.INT32
            or elem_type == onnx.TensorProto.DataType.INT64
            or elem_type == onnx.TensorProto.DataType.DOUBLE
            or elem_type == onnx.TensorProto.DataType.UINT64
        ):
            # 64 bit ints are treated as 32 bits everywhere
            # Doubles are treated as floats
            return 4
        elif (
            elem_type == onnx.TensorProto.DataType.COMPLEX64
            or elem_type == onnx.TensorProto.DataType.COMPLEX128
            or elem_type == onnx.TensorProto.DataType.STRING
            or elem_type == onnx.TensorProto.DataType.UNDEFINED
        ):
            raise exp.Error("Unsupported data type")
        else:
            raise exp.Error("Unsupported data type (unknown to ONNX)")

    def get_nodes_bytes(nodes):
        nodes_bytes = {}
        for node in nodes:

            # Get the number of the data type
            dtype_bytes = elem_type_to_bytes(node.type.tensor_type.elem_type)

            # Calculate the total number of elements based on the shape
            shape = str(node.type.tensor_type.shape.dim)
            num_elements = np.prod([int(s) for s in shape.split() if s.isdigit()])

            # Assign a total number of bytes to each node
            nodes_bytes[node.name] = num_elements * dtype_bytes

        return nodes_bytes

    # Get the number of bytes of each of the inputs and outputs
    model = onnx.load(onnx_path)
    onnx_input_bytes = get_nodes_bytes(model.graph.input)
    onnx_output_bytes = get_nodes_bytes(model.graph.output)

    return int(sum(onnx_input_bytes.values())), int(sum(onnx_output_bytes.values()))


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


def dummy_inputs(onnx_file: str) -> dict:
    # Generate dummy inputs of the expected shape and type for the input model
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    onnx_session = None
    if False:
        onnx_session = ort.InferenceSession(onnx_file, sess_options)
    else:
        onnx_model = onnx.load(onnx_file, load_external_data=False)
        load_fake_data_for_model(onnx_model)
        serialized_model = onnx_model.SerializeToString()
        onnx_session = ort.InferenceSession(serialized_model, sess_options)
    sess_input = onnx_session.get_inputs()

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


def get_opset(model: onnx.ModelProto) -> int:
    return getattr(model.opset_import[0], "version", None)
