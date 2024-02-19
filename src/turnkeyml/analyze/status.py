import os
import dataclasses
import math
import platform
from typing import Dict, Union, List
from turnkeyml.common import printing
import turnkeyml.common.build as build
from turnkeyml.analyze.util import ModelInfo, BasicInfo


def update(
    models_found: Dict[str, ModelInfo],
    build_name: str,
    cache_dir: str,
    verbosity: str = "app",
) -> None:
    """
    Prints all models and submodels found
    """

    if verbosity == "app":
        if os.environ.get("TURNKEY_DEBUG") != "True":
            if platform.system() != "Windows":
                os.system("clear")
            else:
                os.system("cls")

        printing.logn(
            "\nModels discovered during profiling:\n",
            c=printing.Colors.BOLD,
        )
        recursive_print(models_found, build_name, cache_dir, None, None, [])


def recursive_print(
    models_found: Dict[str, ModelInfo],
    build_name: str,
    cache_dir: str,
    parent_model_hash: Union[str, None] = None,
    parent_invocation_hash: Union[str, None] = None,
    script_names_visited: List[str] = False,
) -> None:
    script_names_visited = []

    for model_hash in models_found.keys():
        model_visited = False
        model_info = models_found[model_hash]
        invocation_idx = 0
        for invocation_hash in model_info.unique_invocations.keys():
            unique_invocation = model_info.unique_invocations[invocation_hash]

            if (
                parent_model_hash == model_info.parent_hash
                and unique_invocation.executed > 0
                and (
                    model_info.unique_invocations[invocation_hash].parent_hash
                    == parent_invocation_hash
                )
            ):
                print_file_name = False
                if model_info.script_name not in script_names_visited:
                    script_names_visited.append(model_info.script_name)
                    if model_info.depth == 0:
                        print_file_name = True

                unique_invocation.print(
                    build_name=build_name,
                    cache_dir=cache_dir,
                    print_file_name=print_file_name,
                    invocation_idx=invocation_idx,
                    model_visited=model_visited,
                    multiple_unique_invocations=len(model_info.unique_invocations) > 1,
                )
                model_visited = True
                invocation_idx += 1

                if print_file_name:
                    script_names_visited.append(model_info.script_name)

                recursive_print(
                    models_found,
                    build_name,
                    cache_dir,
                    parent_model_hash=model_hash,
                    parent_invocation_hash=invocation_hash,
                    script_names_visited=script_names_visited,
                )


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


class Monitor:
    def __init__(self):
        self.skip = SkipFields()
        self.model_info = None
        self.unique_invocation = None
        self.extension = None
        self.indent = None
        self.exec_time = None
        self.invocation_hash = None
        self.build_name = None
        self.cache_dir = None
        self.invocation_idx = None

    def _print_heading(
        self,
        print_file_name: bool,
        model_visited: bool,
    ):
        if print_file_name:
            print(f"{self.model_info.script_name}{self.extension}:")

        # Print invocation about the model (only applies to scripts, not ONNX files)
        if self.model_info.model_type != build.ModelType.ONNX_FILE:
            if (
                self.model_info.depth == 0
                and len(self.model_info.unique_invocations) > 1
            ):
                if not model_visited:
                    printing.logn(f"{self.indent}{self.model_info.name}")
            else:
                printing.log(f"{self.indent}{self.model_info.name}")
                printing.logn(
                    f" (executed {self.unique_invocation.executed}x{self.exec_time})",
                    c=printing.Colors.OKGREEN,
                )

        self.skip.file_name = True
        self.skip.model_name = True

    def _print_model_type(
        self,
        model_visited: bool,
    ):

        if (self.model_info.depth == 0 and not model_visited) or (
            self.model_info.depth != 0
        ):
            if self.model_info.depth == 0:
                if self.model_info.model_type == build.ModelType.PYTORCH:
                    print(f"{self.indent}\tModel Type:\tPytorch (torch.nn.Module)")
                elif self.model_info.model_type == build.ModelType.KERAS:
                    print(f"{self.indent}\tModel Type:\tKeras (tf.keras.Model)")
                elif self.model_info.model_type == build.ModelType.ONNX_FILE:
                    print(f"{self.indent}\tModel Type:\tONNX File (.onnx)")

            self.skip.type = True

    def _print_class(self):
        if self.skip.class_name:
            return

        # Display class of model and where it was found, if
        # the an input script (and not an input onnx file) was used
        if self.model_info.model_type != build.ModelType.ONNX_FILE:
            model_class = type(self.model_info.model)
            print(f"{self.indent}\tClass:\t\t{model_class.__name__} ({model_class})")
            self.skip.class_name = True

    def _print_location(self):
        if self.skip.location:
            return

        if self.model_info.depth == 0:
            print(
                f"{self.indent}\tLocation:\t{self.model_info.file}, line {self.model_info.line}"
            )
            self.skip.location = True

    def _print_parameters(self):
        if self.skip.parameters:
            return

        # Display number of parameters and size
        parameters_size = parameters_to_size(self.model_info.params)
        print(
            f"{self.indent}\tParameters:\t{'{:,}'.format(self.model_info.params)} ({parameters_size})"
        )

        self.skip.parameters = True

    def _print_unique_input_shape(self):
        if self.skip.unique_input_shape:
            return

        if self.model_info.depth == 0 and len(self.model_info.unique_invocations) > 1:
            printing.logn(
                f"\n{self.indent}\tWith input shape {self.invocation_idx+1} "
                f"(executed {self.unique_invocation.executed}x{self.exec_time})",
                c=printing.Colors.OKGREEN,
            )

        self.skip.unique_input_shape = True

    def _print_input_shape(self):
        if self.skip.input_shape:
            return

        # Prepare input shape to be printed
        input_shape = dict(self.unique_invocation.input_shapes)
        input_shape = {key: value for key, value in input_shape.items() if value != ()}
        input_shape = str(input_shape).replace("{", "").replace("}", "")

        print(f"{self.indent}\tInput Shape:\t{input_shape}")

        self.skip.input_shape = True

    def _print_hash(self):
        if self.skip.hash:
            return

        print(f"{self.indent}\tHash:\t\t" + self.invocation_hash)

        self.skip.hash = True

    def _print_build_dir(self):
        if self.skip.build_dir:
            return

        print(f"{self.indent}\tBuild dir:\t" + self.cache_dir + "/" + self.build_name)

        self.skip.build_dir = True

    def _print_outcome(self):
        if self.skip.outcome:
            return

        # Print turnkey results if turnkey was run
        if self.unique_invocation.performance:
            printing.log(f"{self.indent}\tStatus:\t\t")
            printing.logn(
                f"Successfully benchmarked on {self.unique_invocation.performance.device} "
                f"({self.unique_invocation.performance.runtime} "
                f"v{self.unique_invocation.performance.runtime_version}) ",
                c=self.unique_invocation.status_message_color,
            )
            printing.logn(
                f"{self.indent}\t\t\tMean Latency:\t{self.unique_invocation.performance.mean_latency:.3f}"
                f"\t{self.unique_invocation.performance.latency_units}"
            )
            printing.logn(
                f"{self.indent}\t\t\tThroughput:\t{self.unique_invocation.performance.throughput:.1f}"
                f"\t{self.unique_invocation.performance.throughput_units}"
            )

            if self.unique_invocation.stats_keys is not None:
                for key in self.unique_invocation.stats_keys:
                    nice_key = _pretty_print_key(key)
                    value = self.unique_invocation.stats.evaluation_stats[key]
                    printing.logn(f"{self.indent}\t\t\t{nice_key}:\t{value}")
            print()
        else:
            if self.unique_invocation.is_target and self.model_info.build_model:
                printing.log(f"{self.indent}\tStatus:\t\t")
                printing.logn(
                    f"{self.unique_invocation.status_message}",
                    c=self.unique_invocation.status_message_color,
                )

                if self.unique_invocation.traceback is not None:
                    if os.environ.get("TURNKEY_TRACEBACK") != "False":
                        for line in self.unique_invocation.traceback:
                            for subline in line.split("\n")[:-1]:
                                print(f"{self.indent}\t{subline}")

                    else:
                        printing.logn(
                            f"{self.indent}\t\t\tTo see the full stack trace, "
                            "rerun with `export TURNKEY_TRACEBACK=True`.\n",
                            c=self.model_info.status_message_color,
                        )
                else:
                    print()

        self.skip.outcome = True

    def print(
        self,
        model_info: BasicInfo,
        build_name: str,
        cache_dir: str,
        invocation_hash: Union[str, None],
        print_file_name: bool = False,
        invocation_idx: int = 0,
        model_visited: bool = False,
    ):
        """
        Print information about a given model or submodel.
        """

        self.model_info = model_info

        if invocation_hash:
            self.unique_invocation = model_info.unique_invocations[invocation_hash]
        else:
            self.unique_invocation = model_info.unique_invocations[-1]

        if model_info.model_type == build.ModelType.ONNX_FILE:
            self.extension = ".onnx"
            self.indent = "\t" * (2 * model_info.depth)
        else:
            self.extension = ".py"
            self.indent = "\t" * (2 * model_info.depth + 1)

        if self.unique_invocation.exec_time == 0 or self.model_info.build_model:
            self.exec_time = ""
        else:
            self.exec_time = f" - {self.unique_invocation.exec_time:.2f}s"

        self.invocation_hash = invocation_hash
        self.build_name = build_name
        self.cache_dir = cache_dir
        self.invocation_idx = invocation_idx

        self._print_heading(print_file_name, model_visited)
        self._print_model_type(model_visited)
        self._print_class()
        self._print_location()
        self._print_parameters()
        self._print_unique_input_shape()
        self._print_input_shape()
        self._print_hash()
        self._print_build_dir()
        self._print_outcome()

        print()


def print_invocation(
    model_info: ModelInfo,
    build_name: str,
    cache_dir: str,
    invocation_hash: Union[str, None],
    print_file_name: bool = False,
    invocation_idx: int = 0,
    model_visited: bool = False,
) -> None:
    """
    Print information about a given model or submodel
    """
    if invocation_hash:
        unique_invocation = model_info.unique_invocations[invocation_hash]
        if model_info.model_type == build.ModelType.ONNX_FILE:
            extension = ".onnx"
            self.indent = "\t" * (2 * model_info.depth)
        else:
            extension = ".py"
            self.indent = "\t" * (2 * model_info.depth + 1)

    if print_file_name:
        print(f"{model_info.script_name}{extension}:")

    if (
        not invocation_hash
        or unique_invocation.exec_time == 0
        or model_info.build_model
    ):
        exec_time = ""
    else:
        exec_time = f" - {unique_invocation.exec_time:.2f}s"

    # Print invocation about the model (only applies to scripts, not ONNX files)
    if model_info.model_type != build.ModelType.ONNX_FILE:
        if model_info.depth == 0 and len(model_info.unique_invocations) > 1:
            if not model_visited:
                printing.logn(f"{self.indent}{model_info.name}")
        else:
            printing.log(f"{self.indent}{model_info.name}")
            printing.logn(
                f" (executed {unique_invocation.executed}x{exec_time})",
                c=printing.Colors.OKGREEN,
            )

    if (model_info.depth == 0 and not model_visited) or (model_info.depth != 0):
        if model_info.depth == 0:
            if model_info.model_type == build.ModelType.PYTORCH:
                print(f"{self.indent}\tModel Type:\tPytorch (torch.nn.Module)")
            elif model_info.model_type == build.ModelType.KERAS:
                print(f"{self.indent}\tModel Type:\tKeras (tf.keras.Model)")
            elif model_info.model_type == build.ModelType.ONNX_FILE:
                print(f"{self.indent}\tModel Type:\tONNX File (.onnx)")

        # Display class of model and where it was found, if
        # the an input script (and not an input onnx file) was used
        if model_info.model_type != build.ModelType.ONNX_FILE:
            model_class = type(model_info.model)
            print(f"{self.indent}\tClass:\t\t{model_class.__name__} ({model_class})")
            if model_info.depth == 0:
                print(
                    f"{self.indent}\tLocation:\t{model_info.file}, line {model_info.line}"
                )

        # Display number of parameters and size
        parameters_size = parameters_to_size(model_info.params)
        print(
            f"{self.indent}\tParameters:\t{'{:,}'.format(model_info.params)} ({parameters_size})"
        )

    if model_info.depth == 0 and len(model_info.unique_invocations) > 1:
        printing.logn(
            f"\n{self.indent}\tWith input shape {invocation_idx+1} "
            f"(executed {unique_invocation.executed}x{exec_time})",
            c=printing.Colors.OKGREEN,
        )

    # Prepare input shape to be printed
    input_shape = dict(model_info.unique_invocations[invocation_hash].input_shapes)
    input_shape = {key: value for key, value in input_shape.items() if value != ()}
    input_shape = str(input_shape).replace("{", "").replace("}", "")

    print(f"{self.indent}\tInput Shape:\t{input_shape}")
    print(f"{self.indent}\tHash:\t\t" + invocation_hash)
    print(f"{self.indent}\tBuild dir:\t" + cache_dir + "/" + build_name)

    # Print turnkey results if turnkey was run
    if unique_invocation.performance:
        printing.log(f"{self.indent}\tStatus:\t\t")
        printing.logn(
            f"Successfully benchmarked on {unique_invocation.performance.device} "
            f"({unique_invocation.performance.runtime} "
            f"v{unique_invocation.performance.runtime_version}) ",
            c=unique_invocation.status_message_color,
        )
        printing.logn(
            f"{self.indent}\t\t\tMean Latency:\t{unique_invocation.performance.mean_latency:.3f}"
            f"\t{unique_invocation.performance.latency_units}"
        )
        printing.logn(
            f"{self.indent}\t\t\tThroughput:\t{unique_invocation.performance.throughput:.1f}"
            f"\t{unique_invocation.performance.throughput_units}"
        )

        if unique_invocation.stats_keys is not None:
            for key in unique_invocation.stats_keys:
                nice_key = _pretty_print_key(key)
                value = unique_invocation.stats.evaluation_stats[key]
                printing.logn(f"{self.indent}\t\t\t{nice_key}:\t{value}")
        print()
    else:
        if unique_invocation.is_target and model_info.build_model:
            printing.log(f"{self.indent}\tStatus:\t\t")
            printing.logn(
                f"{unique_invocation.status_message}",
                c=unique_invocation.status_message_color,
            )

            if unique_invocation.traceback is not None:
                if os.environ.get("TURNKEY_TRACEBACK") != "False":
                    for line in unique_invocation.traceback:
                        for subline in line.split("\n")[:-1]:
                            print(f"{self.indent}\t{subline}")

                else:
                    printing.logn(
                        f"{self.indent}\t\t\tTo see the full stack trace, "
                        "rerun with `export TURNKEY_TRACEBACK=True`.\n",
                        c=model_info.status_message_color,
                    )
            else:
                print()
        print()
