import sys
from dataclasses import dataclass
from typing import Callable, List, Union, Dict, Optional
import dataclasses
import numpy as np
import torch
import onnx
from turnkeyml.common import printing
import turnkeyml.common.build as build
from turnkeyml.common.performance import MeasuredPerformance
import turnkeyml.common.filesystem as fs


class AnalysisException(Exception):
    """
    Indicates a failure during analysis
    """


@dataclass
class UniqueInvocationInfo:
    """
    Refers to unique static model invocations
    (i.e. models executed with unique input shapes)
    """

    hash: Union[str, None] = None
    parent_hash: Union[str, None] = None
    performance: MeasuredPerformance = None
    traceback: List[str] = None
    inputs: Union[dict, None] = None
    input_shapes: Union[dict, None] = None
    executed: int = 0
    exec_time: float = 0.0
    status_message: str = ""
    is_target: bool = False
    status_message_color: printing.Colors = printing.Colors.ENDC
    traceback_message_color: printing.Colors = printing.Colors.FAIL
    stats_keys: Optional[List[str]] = None
    stats: fs.Stats = None


@dataclass
class ModelInfo:
    model: torch.nn.Module
    name: str
    script_name: str
    file: str = ""
    line: int = 0
    params: int = 0
    depth: int = 0
    hash: Union[str, None] = None
    parent_hash: Union[str, None] = None
    old_forward: Union[Callable, None] = None
    unique_invocations: Union[
        Dict[str, UniqueInvocationInfo], None
    ] = dataclasses.field(default_factory=dict)
    last_unique_invocation_executed: Union[str, None] = None
    build_model: bool = False
    model_type: build.ModelType = build.ModelType.PYTORCH

    def __post_init__(self):
        self.params = count_parameters(self.model, self.model_type)


def count_parameters(model: torch.nn.Module, model_type: build.ModelType) -> int:
    """
    Returns the number of parameters of a given model
    """
    if model_type == build.ModelType.PYTORCH:
        return sum([parameter.numel() for _, parameter in model.named_parameters()])
    elif model_type == build.ModelType.KERAS:
        return model.count_params()
    elif model_type == build.ModelType.ONNX_FILE:
        onnx_model = onnx.load(model)
        return int(
            sum(
                np.prod(tensor.dims)
                for tensor in onnx_model.graph.initializer
                if tensor.name not in onnx_model.graph.input
            )
        )

    # Raise exception if an unsupported model type is provided
    raise AnalysisException(f"model_type {model_type} is not supported")


def get_onnx_ops_list(onnx_model) -> Dict:
    """
    List unique ops found in the onnx model
    """
    onnx_ops_counter = {}
    try:
        model = onnx.load(onnx_model)
    except Exception as e:  # pylint: disable=broad-except
        printing.log_warning(f"Failed to get ONNX ops list from {onnx_model}: {str(e)}")
        return onnx_ops_counter
    for node in model.graph.node:  # pylint: disable=E1101
        onnx_ops_counter[node.op_type] = onnx_ops_counter.get(node.op_type, 0) + 1
    return onnx_ops_counter


def attribute_to_dict(attr):
    """
    Helper function that returns a dictionary containing node attributes
    """
    attr_dict = {}
    for field in ["f", "i", "s"]:
        if attr.HasField(field):
            attr_dict[attr.name] = getattr(attr, field)
            return attr_dict
    if attr.ints:
        attr_dict[attr.name] = list(attr.ints)
    elif attr.floats:
        attr_dict[attr.name] = list(attr.floats)
    elif attr.strings:
        attr_dict[attr.name] = list(attr.strings)
    else:
        attr_dict[attr.name] = "unknown_type"
    return attr_dict


def get_onnx_flops(onnx_model) -> Union[int, None]:
    """
    Calculate FLOPs found in the onnx model
    """
    try:
        model = onnx.shape_inference.infer_shapes(
            onnx.load(onnx_model), strict_mode=True, data_prop=True
        )
        # onnx.save(model, "mymodel.onnx")
    except Exception as e:  # pylint: disable=broad-except
        printing.log_warning(f"Failed to get ONNX FLOPs from {onnx_model}: {str(e)}")
        return None

    onnx_flops = np.int64(0)
    for node in model.graph.node:  # pylint: disable=E1101
        inp_tensors = {tensor.name: tensor for tensor in model.graph.input}
        val_tensors = {tensor.name: tensor for tensor in model.graph.value_info}
        init_tensors = {tensor.name: tensor for tensor in model.graph.initializer}

        # inp_dims is a 2-dim array where the first dim indexes inputs
        # and the second dim indexes dimensions
        inp_dims = []
        for inp in node.input:
            inp_dims.append([])
            if inp in inp_tensors or inp in val_tensors:
                tensor = inp_tensors.get(inp) or val_tensors.get(inp)
                inp_dims[-1].extend(
                    [
                        np.int64(dim.dim_value)
                        for dim in tensor.type.tensor_type.shape.dim
                    ]
                )
            elif inp in init_tensors:
                inp_dims[-1].extend(
                    [np.int64(dim) for dim in init_tensors.get(inp).dims]
                )

        attrs = {}
        for attr in node.attribute:
            attrs.update(attribute_to_dict(attr))

        if node.op_type == "MatMul":
            # MatMul is constrained to have 2 N-dimensional inputs
            onnx_flops += (
                np.int64(2)
                * np.prod(inp_dims[0][:-1], dtype=np.int64)
                * inp_dims[1][-1]
            )
        elif node.op_type == "Gemm":
            mm_dims = [
                inp_dims[0][0] if not attrs.get("transA", 0) else inp_dims[0][1],
                inp_dims[0][1] if not attrs.get("transA", 0) else inp_dims[0][0],
                inp_dims[1][1] if not attrs.get("transB", 0) else inp_dims[1][0],
            ]

            if len(mm_dims) == 3:  # if there is a bias input
                onnx_flops += np.prod(inp_dims[2], dtype=np.int64)

            onnx_flops += np.int64(2) * np.prod(mm_dims, dtype=np.int64)

        elif node.op_type == "Conv":
            x_shape = inp_dims[0]  # N, C, d1, ..., dn
            w_shape = inp_dims[1]  # M, C/group, k1, ..., kn
            num_dims = len(x_shape) - 2
            pads = attrs.get("pads", [0] * num_dims * 2)
            strides = attrs.get("strides", [1] * num_dims)
            dilation = attrs.get("dilations", [1] * num_dims)
            kernel_shape = w_shape[2:]
            bias_ops = 0 if len(inp_dims) < 3 else inp_dims[2][0]
            batch_size = x_shape[0]
            inp_channels = x_shape[1]
            out_channels = w_shape[0]
            out_dims = [batch_size, out_channels]
            for i in range(num_dims):
                dim_in = x_shape[i + 2]
                out_dim = (
                    dim_in
                    + pads[i]
                    + pads[i + num_dims]
                    - dilation[i] * (kernel_shape[i] - 1)
                    - 1
                ) // strides[i] + 1
                out_dims.append(out_dim)
            kernel_flops = np.prod(kernel_shape, dtype=np.int64) * inp_channels
            output_points = np.prod(out_dims, dtype=np.int64)
            onnx_flops += np.int64(2) * kernel_flops * output_points + bias_ops
        elif node.op_type == "Lstm":
            hidden_size = attrs.get("hidden_size")
            direction = (
                np.int64(2)
                if attrs.get("direction") == "bidirectional"
                else np.int64(1)
            )
            seq_length, batch_size, input_dim = inp_dims[0]
            gate_inp_flops = np.int64(2) * input_dim * hidden_size
            gate_hid_flops = np.int64(2) * hidden_size * hidden_size
            num_gates = np.int64(4)
            unit_flops = num_gates * gate_inp_flops * gate_hid_flops
            onnx_flops += batch_size * seq_length * direction * unit_flops

    return int(onnx_flops)


def populate_onnx_model_info(onnx_model) -> Dict:
    """
    Read the model metadata to populate IR, Opset and model size
    """
    result_dict = {
        "ir_version": None,
        "opset": None,
        "size on disk (KiB)": None,
    }
    try:
        model = onnx.load(onnx_model)
    except Exception as e:  # pylint: disable=broad-except
        printing.log_warning(f"Failed to get ONNX ops list from {onnx_model}: {str(e)}")
        result_dict.update({"error": "ONNX model analysis failed"})
        return result_dict
    # pylint: disable=E1101
    result_dict.update(
        {
            "ir_version": getattr(model, "ir_version", None),
            "opset": getattr(model.opset_import[0], "version", None),
        }
    )
    try:
        result_dict.update(
            {
                "size on disk (KiB)": round(
                    model.SerializeToString().__sizeof__() / 1024, 4
                ),
            }
        )
    except ValueError:
        # Models >2GB on disk cannot have their model size measured this
        # way and will throw a ValueError https://github.com/onnx/turnkeyml/issues/41
        pass

    return result_dict


def onnx_input_dimensions(onnx_model) -> Dict:
    """
    Read model input dimensions
    """
    input_shape = {}
    try:
        model = onnx.load(onnx_model)
    except Exception as e:  # pylint: disable=broad-except
        printing.log_warning(f"Failed to get ONNX ops list from {onnx_model}: {str(e)}")
        return input_shape
    for inp in model.graph.input:  # pylint: disable=E1101
        shape = str(inp.type.tensor_type.shape.dim)
        input_shape[inp.name] = [int(s) for s in shape.split() if s.isdigit()]
    return input_shape


def stop_logger_forward() -> None:
    """
    Stop forwarding stdout and stderr to file
    """
    if hasattr(sys.stdout, "terminal"):
        sys.stdout = sys.stdout.terminal
    if hasattr(sys.stderr, "terminal_err"):
        sys.stderr = sys.stderr.terminal_err


def analyze_onnx(build_name: str, cache_dir: str, stats: fs.Stats):
    # ONNX stats that we want to save into the build's turnkey_stats.yaml file
    # so that they can be easily accessed by the report command later
    if fs.Keys.ONNX_FILE in stats.evaluation_stats.keys():
        # Just in case the ONNX file was generated on a different machine:
        # strip the state's cache dir, then prepend the current cache dir
        final_onnx_file = fs.rebase_cache_dir(
            stats.evaluation_stats[fs.Keys.ONNX_FILE],
            build_name,
            cache_dir,
        )

        onnx_ops_counter = get_onnx_ops_list(final_onnx_file)
        onnx_flops_counter = get_onnx_flops(final_onnx_file)
        onnx_model_info = populate_onnx_model_info(final_onnx_file)
        input_dimensions = onnx_input_dimensions(final_onnx_file)

        stats.save_model_stat(
            fs.Keys.ONNX_OPS_COUNTER,
            onnx_ops_counter,
        )
        stats.save_model_stat(
            fs.Keys.ONNX_FLOPS_COUNTER,
            onnx_flops_counter,
        )
        stats.save_model_stat(
            fs.Keys.ONNX_MODEL_INFO,
            onnx_model_info,
        )
        stats.save_model_stat(
            fs.Keys.ONNX_INPUT_DIMENSIONS,
            input_dimensions,
        )
