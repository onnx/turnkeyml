import sys
from dataclasses import dataclass
from typing import Callable, List, Union, Dict, Optional
import dataclasses
import os
import math
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


def _pretty_print_key(key: str) -> str:
    result = key.split("_")
    result = [word.capitalize() for word in result]
    result = " ".join(result)
    return result


def parameters_to_size(parameters: int, byte_per_parameter: int = 4) -> str:
    size_bytes = parameters * byte_per_parameter
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


@dataclass
class BasicInfo:
    name: str
    script_name: str
    file: str = ""
    line: int = 0
    params: int = 0
    depth: int = 0
    parent_hash: Union[str, None] = None
    build_model: bool = False
    model_type: build.ModelType = build.ModelType.PYTORCH
    model_class: type = None
    hash: Union[str, None] = None


@dataclasses.dataclass
class SkipFields:
    """
    Keep track of which fields of a model's status should be skipped
    during printout. There are two use cases in mind:
    - For incremental printout: fields that have already been printed.
    - For low-verbosity: fields that should never be printed.
    """

    file_name: bool = False
    model_name: bool = False
    type: bool = False
    parameters: bool = False
    class_name: bool = False
    location: bool = False
    input_shape: bool = False
    hash: bool = False
    build_dir: bool = False
    outcome: bool = False
    unique_input_shape: bool = False


@dataclass
class UniqueInvocationInfo(BasicInfo):
    """
    Refers to unique static model invocations
    (i.e. models executed with unique input shapes)
    """

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

    # Fields specific to printing status
    skip: SkipFields = SkipFields()
    extension: str = None
    indent: str = None

    def _print_heading(
        self,
        exec_time_formatted: str,
        print_file_name: bool,
        model_visited: bool,
        multiple_unique_invocations: bool,
    ):
        if print_file_name:
            print(f"{self.script_name}{self.extension}:")

        # Print invocation about the model (only applies to scripts, not ONNX files)
        if self.model_type != build.ModelType.ONNX_FILE:
            if self.depth == 0 and multiple_unique_invocations:
                if not model_visited:
                    printing.logn(f"{self.indent}{self.name}")
            else:
                printing.log(f"{self.indent}{self.name}")
                printing.logn(
                    f" (executed {self.executed}x{exec_time_formatted})",
                    c=printing.Colors.OKGREEN,
                )

        self.skip.file_name = True
        self.skip.model_name = True

    def _print_model_type(
        self,
        model_visited: bool,
    ):

        if (self.depth == 0 and not model_visited) or (self.depth != 0):
            if self.depth == 0:
                if self.model_type == build.ModelType.PYTORCH:
                    print(f"{self.indent}\tModel Type:\tPytorch (torch.nn.Module)")
                elif self.model_type == build.ModelType.KERAS:
                    print(f"{self.indent}\tModel Type:\tKeras (tf.keras.Model)")
                elif self.model_type == build.ModelType.ONNX_FILE:
                    print(f"{self.indent}\tModel Type:\tONNX File (.onnx)")

            self.skip.type = True

    def _print_class(self):
        if self.skip.class_name:
            return

        # Display class of model and where it was found, if
        # the an input script (and not an input onnx file) was used
        if self.model_type != build.ModelType.ONNX_FILE:
            print(
                f"{self.indent}\tClass:\t\t{self.model_class.__name__} ({self.model_class})"
            )
            self.skip.class_name = True

    def _print_location(self):
        if self.skip.location:
            return

        if self.depth == 0:
            print(f"{self.indent}\tLocation:\t{self.file}, line {self.line}")
            self.skip.location = True

    def _print_parameters(self):
        if self.skip.parameters:
            return

        # Display number of parameters and size
        parameters_size = parameters_to_size(self.params)
        print(
            f"{self.indent}\tParameters:\t{'{:,}'.format(self.params)} ({parameters_size})"
        )

        self.skip.parameters = True

    def _print_unique_input_shape(
        self,
        exec_time_formatted: str,
        invocation_idx: int,
        multiple_unique_invocations: bool,
    ):
        if self.skip.unique_input_shape:
            return

        if self.depth == 0 and multiple_unique_invocations:
            printing.logn(
                f"\n{self.indent}\tWith input shape {invocation_idx+1} "
                f"(executed {self.executed}x{exec_time_formatted})",
                c=printing.Colors.OKGREEN,
            )

        self.skip.unique_input_shape = True

    def _print_input_shape(self):
        if self.skip.input_shape:
            return

        # Prepare input shape to be printed
        input_shape = dict(self.input_shapes)
        input_shape = {key: value for key, value in input_shape.items() if value != ()}
        input_shape = str(input_shape).replace("{", "").replace("}", "")

        print(f"{self.indent}\tInput Shape:\t{input_shape}")

        self.skip.input_shape = True

    def _print_hash(self):
        if self.skip.hash:
            return

        print(f"{self.indent}\tHash:\t\t" + self.hash)

        self.skip.hash = True

    def _print_build_dir(self, cache_dir: str, build_name: str):
        if self.skip.build_dir:
            return

        print(f"{self.indent}\tBuild dir:\t {build.output_dir(cache_dir, build_name)}")

        self.skip.build_dir = True

    def _print_outcome(self):
        if self.skip.outcome:
            return

        # Print turnkey results if turnkey was run
        if self.performance:
            printing.log(f"{self.indent}\tStatus:\t\t")
            printing.logn(
                f"Successfully benchmarked on {self.performance.device} "
                f"({self.performance.runtime} "
                f"v{self.performance.runtime_version}) ",
                c=self.status_message_color,
            )
            printing.logn(
                f"{self.indent}\t\t\tMean Latency:\t{self.performance.mean_latency:.3f}"
                f"\t{self.performance.latency_units}"
            )
            printing.logn(
                f"{self.indent}\t\t\tThroughput:\t{self.performance.throughput:.1f}"
                f"\t{self.performance.throughput_units}"
            )

            if self.stats_keys is not None:
                for key in self.stats_keys:
                    nice_key = _pretty_print_key(key)
                    value = self.stats.evaluation_stats[key]
                    printing.logn(f"{self.indent}\t\t\t{nice_key}:\t{value}")
            print()
        else:
            if self.is_target and self.build_model:
                printing.log(f"{self.indent}\tStatus:\t\t")
                printing.logn(
                    f"{self.status_message}",
                    c=self.status_message_color,
                )

                if self.traceback is not None:
                    if os.environ.get("TURNKEY_TRACEBACK") != "False":
                        for line in self.traceback:
                            for subline in line.split("\n")[:-1]:
                                print(f"{self.indent}\t{subline}")

                    else:
                        printing.logn(
                            f"{self.indent}\t\t\tTo see the full stack trace, "
                            "rerun with `export TURNKEY_TRACEBACK=True`.\n",
                            c=self.status_message_color,
                        )
                else:
                    print()

        self.skip.outcome = True

    def print(
        self,
        build_name: str,
        cache_dir: str,
        print_file_name: bool = False,
        invocation_idx: int = 0,
        model_visited: bool = False,
        multiple_unique_invocations: bool = False,
    ):
        """
        Print information about a given model or submodel.
        """

        if self.model_type == build.ModelType.ONNX_FILE:
            self.extension = ".onnx"
            self.indent = "\t" * (2 * self.depth)
        else:
            self.extension = ".py"
            self.indent = "\t" * (2 * self.depth + 1)

        if self.exec_time == 0 or self.build_model:
            exec_time_formatted = ""
        else:
            exec_time_formatted = f" - {self.exec_time:.2f}s"

        self._print_heading(
            exec_time_formatted,
            print_file_name,
            model_visited,
            multiple_unique_invocations,
        )
        self._print_model_type(model_visited)
        self._print_class()
        self._print_location()
        self._print_parameters()
        self._print_unique_input_shape(
            exec_time_formatted, invocation_idx, multiple_unique_invocations
        )
        self._print_input_shape()
        self._print_hash()
        self._print_build_dir(cache_dir=cache_dir, build_name=build_name)
        self._print_outcome()

        print()


@dataclass
class ModelInfo(BasicInfo):
    model: torch.nn.Module = None
    old_forward: Union[Callable, None] = None
    unique_invocations: Union[Dict[str, UniqueInvocationInfo], None] = (
        dataclasses.field(default_factory=dict)
    )
    last_unique_invocation_executed: Union[str, None] = None

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


def attribute_to_dict(attribute):
    """
    Helper function that returns a dictionary containing node attributes
    """
    attribute_dict = {}
    for field in ["f", "i", "s"]:
        if attribute.HasField(field):
            attribute_dict[attribute.name] = getattr(attribute, field)
            return attribute_dict
    if attribute.ints:
        attribute_dict[attribute.name] = list(attribute.ints)
    elif attribute.floats:
        attribute_dict[attribute.name] = list(attribute.floats)
    elif attribute.strings:
        attribute_dict[attribute.name] = list(attribute.strings)
    else:
        attribute_dict[attribute.name] = "unknown_type"
    return attribute_dict


def get_onnx_total_flops(onnx_model) -> Union[int, None]:
    """
    Calculate total number of FLOPs found in the onnx model.
    FLOP is defined as one floating-point operation. This distinguishes
    from multiply-accumulates (MACs) where FLOPs == 2 * MACs.
    """
    try:
        onnx.shape_inference.infer_shapes_path(
            model_path=onnx_model,
            output_path=onnx_model,
            strict_mode=True,
            data_prop=True,
        )
        model = onnx.load(onnx_model, load_external_data=False)
    except Exception as e:  # pylint: disable=broad-except
        printing.log_warning(f"Failed to get ONNX FLOPs from {onnx_model}: {str(e)}")
        return None

    # If the ONNX model contains one of the following unsupported ops, then this
    # function will return None since the FLOP total is expected to be incorrect
    unsupported_ops = [
        "Einsum",
        "RNN",
        "GRU",
        "DeformConv",
    ]

    total_flops = np.int64(0)
    for node in model.graph.node:  # pylint: disable=E1101
        input_tensors = {tensor.name: tensor for tensor in model.graph.input}
        output_tensors = {tensor.name: tensor for tensor in model.graph.output}
        value_tensors = {tensor.name: tensor for tensor in model.graph.value_info}
        init_tensors = {tensor.name: tensor for tensor in model.graph.initializer}

        # input_dims is a 2 dimensional array where the first dimension indexes inputs
        # and the second dimension indexes dimensions
        input_dims = []
        for input in node.input:
            input_dims.append([])
            if (
                input in input_tensors
                or input in value_tensors
                or input in output_tensors
            ):
                tensor = (
                    input_tensors.get(input)
                    or value_tensors.get(input)
                    or output_tensors.get(input)
                )
                input_dims[-1].extend(
                    [
                        np.int64(dim.dim_value)
                        for dim in tensor.type.tensor_type.shape.dim
                    ]
                )
            elif input in init_tensors:
                input_dims[-1].extend([dim for dim in init_tensors.get(input).dims])

        attributes = {}
        for attribute in node.attribute:
            attributes.update(attribute_to_dict(attribute))

        current_op_flops = 0

        if node.op_type in unsupported_ops:
            return None

        elif (
            node.op_type == "MatMul"
            or node.op_type == "MatMulInteger"
            or node.op_type == "QLinearMatMul"
        ):
            input_a = input_dims[0]
            input_b = (
                input_dims[3] if node.op_type == "QLinearMatMul" else input_dims[1]
            )
            current_op_flops = 2 * np.prod(input_a, dtype=np.int64) * input_b[-1]

        elif node.op_type == "Mul" or node.op_type == "Div" or node.op_type == "Add":
            current_op_flops = np.prod(input_dims[0], dtype=np.int64) + np.prod(
                input_dims[1], dtype=np.int64
            )

        elif node.op_type == "Gemm" or node.op_type == "QGemm":
            x_shape = input_dims[0]
            w_shape = input_dims[1] if node.op_type == "Gemm" else input_dims[3]
            mm_dims = [
                x_shape[0] if not attributes.get("transA", 0) else x_shape[1],
                x_shape[1] if not attributes.get("transA", 0) else x_shape[0],
                w_shape[1] if not attributes.get("transB", 0) else w_shape[0],
            ]
            current_op_flops = 2 * np.prod(mm_dims, dtype=np.int64)
            if len(mm_dims) == 3:  # if there is a bias input
                current_op_flops += np.prod(input_dims[2], dtype=np.int64)

        elif (
            node.op_type == "Conv"
            or node.op_type == "ConvInteger"
            or node.op_type == "QLinearConv"
            or node.op_type == "ConvTranspose"
        ):
            x_shape = input_dims[0]  # N, C, d1, ..., dn
            w_shape = (
                input_dims[3] if node.op_type == "QLinearConv" else input_dims[1]
            )  # M, C/group, k1, ..., kn. Note C and M are swapped for ConvTranspose

            has_bias = False  # Note, ConvInteger has no bias
            if node.op_type == "Conv" and len(input_dims) == 3:
                has_bias = True
            elif node.op_type == "QLinearConv" and len(input_dims) == 9:
                has_bias = True

            num_dims = len(x_shape) - 2
            strides = attributes.get("strides", [1] * num_dims)
            dilation = attributes.get("dilations", [1] * num_dims)
            kernel_shape = w_shape[2:]
            batch_size = x_shape[0]
            out_channels = w_shape[0]
            out_dims = [batch_size, out_channels]
            output_shape = attributes.get("output_shape", [])

            # If output_shape is given then we do not need to compute it ourselves
            # The output_shape attribute does not include batch_size or channels and
            # is only valid for ConvTranspose
            if output_shape:
                out_dims.extend(output_shape)
            else:
                auto_pad = attributes.get("auto_pad", "NOTSET".encode()).decode()
                # SAME expects padding so that the output_shape = CEIL(input_shape / stride)
                if auto_pad == "SAME_UPPER" or auto_pad == "SAME_LOWER":
                    out_dims.extend([x * s for x, s in zip(x_shape[2:], strides)])
                else:
                    # NOTSET means just use pads attribute
                    if auto_pad == "NOTSET":
                        pads = attributes.get("pads", [0] * num_dims * 2)
                    # VALID essentially means no padding
                    elif auto_pad == "VALID":
                        pads = [0] * num_dims * 2

                    for i in range(num_dims):
                        dim_in = x_shape[i + 2]

                        if node.op_type == "ConvTranspose":
                            out_dim = (
                                strides[i] * (dim_in - 1)
                                + ((kernel_shape[i] - 1) * dilation[i] + 1)
                                - pads[i]
                                - pads[i + num_dims]
                            )
                        else:
                            out_dim = (
                                dim_in
                                + pads[i]
                                + pads[i + num_dims]
                                - dilation[i] * (kernel_shape[i] - 1)
                                - 1
                            ) // strides[i] + 1

                        out_dims.append(out_dim)

            kernel_flops = np.prod(kernel_shape, dtype=np.int64) * w_shape[1]
            output_points = np.prod(out_dims, dtype=np.int64)
            bias_ops = output_points if has_bias else 0
            current_op_flops = 2 * kernel_flops * output_points + bias_ops

        elif node.op_type == "LSTM" or node.op_type == "DynamicQuantizeLSTM":
            hidden_size = attributes.get("hidden_size")
            direction = (
                2 if attributes.get("direction") == "bidirectional".encode() else 1
            )
            bias_ops = 0 if not input_dims[3] else input_dims[3][1]
            seq_length, batch_size, input_dim = input_dims[0]
            num_gates = 4
            gate_input_flops = np.int64(2) * input_dim * hidden_size
            gate_hid_flops = np.int64(2) * hidden_size * hidden_size
            unit_flops = num_gates * (gate_input_flops + gate_hid_flops) + bias_ops
            current_op_flops = batch_size * seq_length * direction * unit_flops

        total_flops += current_op_flops

    return int(total_flops)


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
    for input in model.graph.input:  # pylint: disable=E1101
        shape = str(input.type.tensor_type.shape.dim)
        input_shape[input.name] = [int(s) for s in shape.split() if s.isdigit()]
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
        onnx_total_flops = get_onnx_total_flops(final_onnx_file)
        onnx_model_info = populate_onnx_model_info(final_onnx_file)
        input_dimensions = onnx_input_dimensions(final_onnx_file)

        stats.save_model_stat(
            fs.Keys.ONNX_OPS_COUNTER,
            onnx_ops_counter,
        )
        stats.save_model_stat(
            fs.Keys.ONNX_TOTAL_FLOPS,
            onnx_total_flops,
        )
        stats.save_model_stat(
            fs.Keys.ONNX_MODEL_INFO,
            onnx_model_info,
        )
        stats.save_model_stat(
            fs.Keys.ONNX_INPUT_DIMENSIONS,
            input_dimensions,
        )
