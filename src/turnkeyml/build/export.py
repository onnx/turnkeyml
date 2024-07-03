import os
import inspect
import shutil
import warnings
import sys
import copy
import argparse
from typing import Union
import torch
import torch.onnx.verification
import numpy as np
import onnxruntime
import onnxmltools
import onnx
import turnkeyml.build.stage as stage
import turnkeyml.common.exceptions as exp
import turnkeyml.common.build as build
import turnkeyml.build.tensor_helpers as tensor_helpers
import turnkeyml.build.onnx_helpers as onnx_helpers
import turnkeyml.common.filesystem as fs
from turnkeyml.analyze.status import ModelInfo, UniqueInvocationInfo


def check_model(onnx_file, success_message, fail_message) -> bool:
    if os.path.isfile(onnx_file):
        print(success_message)
    else:
        print(fail_message)
        return False
    try:
        onnx.checker.check_model(onnx_file)
        print("\tSuccessfully checked onnx file")
        return True
    except onnx.checker.ValidationError as e:
        print("\tError while checking generated ONNX file")
        print(e)
        return False


def _warn_to_stdout(message, category, filename, line_number, _, line):
    sys.stdout.write(
        warnings.formatwarning(message, category, filename, line_number, line)
    )


def get_output_names(
    onnx_model: Union[str, onnx.ModelProto]
):  # pylint: disable=no-member
    # Get output names of ONNX file/model
    if not isinstance(onnx_model, onnx.ModelProto):  # pylint: disable=no-member
        onnx_model = onnx.load(onnx_model)
    return [node.name for node in onnx_model.graph.output]  # pylint: disable=no-member


def original_inputs_file(cache_dir: str, build_name: str):
    return os.path.join(build.output_dir(cache_dir, build_name), "inputs.npy")


def onnx_dir(state: fs.State):
    return os.path.join(build.output_dir(state.cache_dir, state.build_name), "onnx")


def base_onnx_file(state: fs.State):
    return os.path.join(
        onnx_dir(state),
        f"{state.build_name}-op{state.onnx_opset}-base.onnx",
    )


def opt_onnx_file(state: fs.State):
    return os.path.join(
        onnx_dir(state),
        f"{state.build_name}-op{state.onnx_opset}-opt.onnx",
    )


def converted_onnx_file(state: fs.State):
    return os.path.join(
        onnx_dir(state),
        f"{state.build_name}-op{state.onnx_opset}-opt-f16.onnx",
    )


class OnnxLoad(stage.Stage):
    """
    Stage that takes an ONNX model as input and passes it to the following
    stages.

    Expected inputs: None

    Outputs:
     - A *-base.onnx file that implements state.model given state.inputs.
    """

    unique_name = "onnx-load"

    def __init__(self):
        super().__init__(monitor_message="Loading ONNX Model")

    @staticmethod
    def parser(add_help: bool = True) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Load an ONNX model",
            add_help=add_help,
        )

        # Hidden argument required by TurnkeyML for any stage that starts a sequence
        parser.add_argument("--input", help=argparse.SUPPRESS)

        return parser

    def fire(self, state: fs.State, input: str = ""):

        onnx_file = input
        state.model_hash = build.hash_model(onnx_file)

        if not onnx_file.endswith(".onnx"):
            msg = f"""
            The current stage (ReceiveOnnxModel) expects a path to ONNX
            model, however the stage received {onnx_file}.
            """
            raise exp.StageError(msg)

        state.inputs = onnx_helpers.dummy_inputs(onnx_file)
        dummy_inputs = tuple(state.inputs.values())
        dummy_input_names = tuple(state.inputs.keys())
        state.inputs = dict(zip(dummy_input_names, dummy_inputs))

        model = onnx.load(onnx_file)
        opset = onnx_helpers.get_opset(model)
        state.onnx_opset = opset
        input_shapes = [
            [d.dim_value for d in _input.type.tensor_type.shape.dim]
            for _input in model.graph.input  # pylint: disable=no-member
        ]

        # Save output node names
        state.expected_output_names = get_output_names(model)

        # Check for Dynamic shapes in the model. They can be represented as 0, -1, "unk__".
        for input in input_shapes:
            for dimension in input:
                if dimension < 1 or not isinstance(dimension, int):
                    msg = f"""
                    The received model has dynamic input dimensions. Please freeze the model with static
                    input dimensions.
                    More information may be available in the log file at **{self.logfile_path}**
                    """
                    raise exp.StageError(msg)

        print(opset)
        if opset < build.DEFAULT_ONNX_OPSET and opset >= build.MINIMUM_ONNX_OPSET:
            print(
                f" \n The received model has an opset {opset}. Though this opset is supported \
                we recommend upgrading the model to opset {build.MINIMUM_ONNX_OPSET}"
            )
        elif opset < build.MINIMUM_ONNX_OPSET:
            msg = f"""
            The received model has an opset {opset}. Opset < {build.MINIMUM_ONNX_OPSET} 
            is not supported. Please try upgrading the model to opset {build.MINIMUM_ONNX_OPSET}.
            More information may be available in the log file at **{self.logfile_path}**
            """
            raise exp.StageError(msg)

        output_path = base_onnx_file(state)
        os.makedirs(onnx_dir(state), exist_ok=True)
        shutil.copy(onnx_file, output_path)

        tensor_helpers.save_inputs(
            [state.inputs],
            original_inputs_file(state.cache_dir, state.build_name),
            downcast=False,
        )

        # Check the if the base mode has been exported successfully
        success_msg = "\tSuccess receiving ONNX Model"
        fail_msg = "\tFailed receiving ONNX Model"

        if check_model(output_path, success_msg, fail_msg):
            state.results = output_path

            state.save_stat(
                fs.Keys.ONNX_FILE,
                output_path,
            )
        else:
            msg = f"""
            Unable to process ONNX Model. We recommend that you verify the source of the model.
            Any optimizations performed on the model could result in an error.
            More information may be available in the log file at **{self.logfile_path}**
            """
            raise exp.StageError(msg)

        # Create a UniqueInvocationInfo and ModelInfo so that we can display status
        # at the end of the sequence
        state.invocation_info = UniqueInvocationInfo(
            name=onnx_file,
            script_name=fs.clean_file_name(onnx_file),
            file=onnx_file,
            input_shapes={key: value.shape for key, value in state.inputs.items()},
            hash=state.model_hash,
            is_target=True,
            extension=".onnx",
            executed=1,
        )
        state.models_found = {
            "onnx_file": ModelInfo(
                model=onnx_file,
                name=onnx_file,
                script_name=onnx_file,
                file=onnx_file,
                unique_invocations={state.model_hash: state.invocation_info},
                hash=state.model_hash,
            )
        }
        state.invocation_info.params = state.models_found["onnx_file"].params

        return state


class ExportPytorchModel(stage.Stage):
    """
    Stage that takes a PyTorch model instance, in state.model, and
    exports it to an ONNX file.

    Expected inputs:
     - state.model is a torch.nn.Module or torch.jit.ScriptModule
     - state.inputs is a dict that represents valid kwargs to the forward
        function of state.model

    Outputs:
     - A *-base.onnx file that implements state.model given state.inputs
    """

    unique_name = "export-pytorch"

    def __init__(self):
        super().__init__(monitor_message="Exporting PyTorch to ONNX")

    @staticmethod
    def parser(add_help: bool = True) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Export a PyTorch model to ONNX",
            add_help=add_help,
        )

        parser.add_argument(
            "--opset",
            type=int,
            default=build.DEFAULT_ONNX_OPSET,
            help=f"ONNX opset to export into (default: {build.DEFAULT_ONNX_OPSET})",
        )

        return parser

    def fire(self, state: fs.State, opset: int = build.DEFAULT_ONNX_OPSET):
        if not isinstance(state.model, (torch.nn.Module, torch.jit.ScriptModule)):
            msg = f"""
            The current stage (ExportPytorchModel) is only compatible with
            models of type torch.nn.Module or torch.jit.ScriptModule, however
            the stage received a model of type {type(state.model)}.
            """
            raise exp.StageError(msg)

        state.onnx_opset = opset

        # The `torch.onnx.export()` function accepts a tuple of positional inputs
        # followed by a dictionary with all keyword inputs.
        # The dictionary must be last item in tuple.
        user_provided_args = list(state.inputs.keys())

        if isinstance(state.model, torch.nn.Module):
            # Validate user provided args
            all_args = list(inspect.signature(state.model.forward).parameters.keys())

            for inp in user_provided_args:
                if inp not in all_args:
                    msg = f"""
                    Input name {inp} not found in the model's forward method. Available
                    input names are: {all_args}"
                    """
                    raise ValueError(msg)

            # Most pytorch models have args that are kind = positional_or_keyword.
            # The `torch.onnx.export()` function accepts model args as
            #     (all_positional_args_value,{keyword_arg:value}).
            # To map the input_args correctly and to build an accurate model
            # the order of the input_names must reflect the order of the model args.

            # Collect order of pytorch model args.
            all_args_order_mapping = {arg: idx for idx, arg in enumerate(all_args)}

            # Sort the user provided inputs with respect to model args and store as tuple.
            sorted_user_inputs = sorted(
                user_provided_args, key=lambda x: all_args_order_mapping[x]
            )
            dummy_input_names = tuple(sorted_user_inputs)

            # If a single input is provided torch.onnx.export will
            # not accept a dictionary, so pop the first arg
            user_args = copy.deepcopy(state.inputs)
            first_input = user_args.pop(dummy_input_names[0])

            # Create tuple: (first input, {rest of user_args dict as keyword args})
            dummy_inputs = (first_input, user_args)

        else:  # state.model is a torch.jit.ScriptModule
            dummy_inputs = tuple(state.inputs.values())

            # Collect input names
            dummy_input_names = tuple(state.inputs.keys())

        # Send torch export warnings to stdout (and therefore the log file)
        # so that they don't fill up the command line
        default_warnings = warnings.showwarning
        warnings.showwarning = _warn_to_stdout

        # Verify if the exported model matches the input torch model
        try:
            # Tolerance levels for the torch export are recommended by Pytorch here:
            # https://pytorch.org/docs/stable/testing.html#module-torch.testing
            fp32_tolerance = torch.onnx.verification.VerificationOptions(
                rtol=1.3e-6, atol=1e-5
            )

            # The `torch.onnx.verification.find_mismatch()` takes input arguments to the
            # model as `input_args (Tuple[Any, ...])`
            export_verification = torch.onnx.verification.find_mismatch(
                state.model,
                tuple(state.inputs.values()),
                opset_version=opset,
                options=fp32_tolerance,
            )

            # `export_verification.has_mismatch()` returns True if a mismatch is found and
            # False otherwise. If no mismatch is found,# `is_export_valid` is set to "Valid",
            # indicating successful verification.
            # If a mismatch is found, `is_export_valid` is set to "Invalid", indicating
            # the verification failed.
            if not export_verification.has_mismatch():
                is_export_valid = "valid"
            else:
                is_export_valid = "invalid"

        # The except block catches any type of exception that might occur during the
        # verification process. If any exception occurs,`is_export_valid` is set to
        # "Unverified", indicating that the verification process could not be completed,
        # and therefore the model's export status is unverified.
        except Exception:  # pylint: disable=broad-except
            is_export_valid = "unverified"

        state.save_stat(
            fs.Keys.TORCH_ONNX_EXPORT_VALIDITY,
            is_export_valid,
        )

        # Export the model to ONNX
        output_path = base_onnx_file(state)
        os.makedirs(onnx_dir(state), exist_ok=True)

        torch.onnx.export(
            state.model,
            dummy_inputs,
            output_path,
            input_names=dummy_input_names,
            do_constant_folding=True,
            opset_version=opset,
            verbose=False,
        )

        # Save output names to ensure we are preserving the order of the outputs
        state.expected_output_names = get_output_names(output_path)

        # Restore default warnings behavior
        warnings.showwarning = default_warnings

        tensor_helpers.save_inputs(
            [state.inputs],
            original_inputs_file(state.cache_dir, state.build_name),
            downcast=False,
        )

        # Check the if the base mode has been exported successfully
        success_msg = "\tSuccess exporting model to ONNX"
        fail_msg = "\tFailed exporting model to ONNX"

        if check_model(output_path, success_msg, fail_msg):
            state.results = output_path

            state.save_stat(
                fs.Keys.ONNX_FILE,
                output_path,
            )
        else:
            msg = f"""
            Unable to export model to ONNX using Torch's ONNX exporter.
            We recommend that you modify your model until it is
            compatible with this third party software, then re-run.
            More information may be available in the log file at **{self.logfile_path}**
            """
            raise exp.StageError(msg)

        return state


class OptimizeOnnxModel(stage.Stage):
    """
    Stage that takes an ONNX file and uses ONNX Runtime to optimize it.
    Important because this helps to perform constant folding, Redundant
    node eliminations, Semantics-preserving node fusions

    Expected inputs:
     - state.results contains a single .onnx file

    Outputs:
     - A *-opt.onnx file
    """

    unique_name = "optimize-onnx"

    def __init__(self):
        super().__init__(monitor_message="Optimizing ONNX file")

    @staticmethod
    def parser(add_help: bool = True) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Use OnnxRuntime to optimize an ONNX model",
            add_help=add_help,
        )

        return parser

    def fire(self, state: fs.State):
        input_onnx = state.results
        output_path = opt_onnx_file(state)

        # Perform some basic optimizations on the model to remove shape related
        # information inserted for dynamic shape inference.
        # Given that we're compiling against a fixed sequence length the dynamic
        # shape information is not necessary
        session_options = onnxruntime.SessionOptions()

        # Set graph optimization level
        session_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
        )

        # To enable model serialization after graph optimization set this
        session_options.optimized_model_filepath = output_path

        # Optimize graph
        onnxruntime.InferenceSession(input_onnx, session_options)

        # Check that the converted model is still valid
        success_msg = "\tSuccess optimizing ONNX model"
        fail_msg = "\tFailed optimizing ONNX model"

        if check_model(output_path, success_msg, fail_msg):
            state.results = output_path

            state.save_stat(
                fs.Keys.ONNX_FILE,
                output_path,
            )
        else:
            msg = f"""
            Unable to optimize ONNX file using ONNX runtime.
            We recommend that you modify your model until it is
            compatible with this third party software, then re-run.
            More information may be available in the log file at **{self.logfile_path}**
            """
            raise exp.StageError(msg)

        return state


class ConvertOnnxToFp16(stage.Stage):
    """
    Stage that takes an ONNX file and converts its trained parameters
    to fp16.

    Expected inputs:
     - state.results contains a single .onnx file

    Outputs:
     - A *-f16.onnx file with FP16 trained parameters
    """

    unique_name = "fp16-conversion"

    def __init__(self):
        super().__init__(
            monitor_message="Converting to FP16",
        )

    @staticmethod
    def parser(add_help: bool = True) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Use OnnxMLTools to convert an ONNX model to fp16",
            add_help=add_help,
        )

        return parser

    def fire(self, state: fs.State):
        input_onnx = state.results

        # Convert the model to FP16
        # Some ops will not be converted to fp16 because they are in a block list
        # The latest list can be found here. It is not necessarily the list that
        # our version of onnxmltools sees
        # https://github.com/microsoft/onnxconverter-common/blob/master/onnxconverter_common/float16.py#L82

        # Send onnxmltools warnings to stdout (and therefore the log file)
        # so that they don't fill up the command line
        default_warnings = warnings.showwarning
        warnings.showwarning = _warn_to_stdout

        # Legalize ops are ops that have been or are currently in the block list
        # that we explicitly want removed
        legalize_ops = ["InstanceNormalization", "Resize", "Max"]
        op_block_list = onnxmltools.utils.float16_converter.DEFAULT_OP_BLOCK_LIST.copy()
        for op in legalize_ops:
            # Check to see that they are not in the block list before we remove them
            # Neccesary because the block list may be updated, and not in the state we expect
            if op in op_block_list:
                op_block_list.remove(op)

        # Infer shapes before converting to FP16 to enable models with >2GB
        onnx.shape_inference.infer_shapes_path(input_onnx)

        fp32_model = onnx.load_model(input_onnx)
        fp16_model = onnxmltools.utils.float16_converter.convert_float_to_float16(
            fp32_model, op_block_list=op_block_list, disable_shape_infer=True
        )

        # Load inputs and convert to fp16
        inputs_file = original_inputs_file(state.cache_dir, state.build_name)
        if os.path.isfile(inputs_file):
            inputs = np.load(inputs_file, allow_pickle=True)
            inputs_converted = tensor_helpers.save_inputs(
                inputs, inputs_file, downcast=True
            )
        else:
            raise exp.StageError(
                "Attempted to convert inputs to FP16, however inputs file was not found."
            )

        # Overwrite expected dtypes
        _, state.expected_input_dtypes = build.get_shapes_and_dtypes(
            inputs_converted[0]
        )

        # Indicate that inputs must be downcasted during inference
        state.downcast_applied = True

        # Save FP16 model (use external data format if needed)
        output_path = converted_onnx_file(state)
        try:
            onnxmltools.utils.save_model(fp16_model, output_path)
        except ValueError:
            onnx.save_model(fp16_model, output_path, save_as_external_data=True)

        # Restore default warnings behavior
        warnings.showwarning = default_warnings

        # Check that the converted model is still valid
        success_msg = "\tSuccess converting ONNX model to fp16"
        fail_msg = "\tFailed converting ONNX model to fp16"

        if check_model(output_path, success_msg, fail_msg):
            state.results = output_path

            state.save_stat(
                fs.Keys.ONNX_FILE,
                output_path,
            )
        else:
            msg = f"""
            Attempted to use onnxmltools, a third party library, to convert your
            model to the float16 datatype, however this operation was not successful.
            More information may be available in the log file at **{self.logfile_path}**
            """
            raise exp.StageError(msg)

        return state
