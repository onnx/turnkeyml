from typing import Optional, List, Tuple, Union, Dict, Any
import os
import copy
import torch
import turnkeyml.common.build as build
import turnkeyml.common.filesystem as fs
import turnkeyml.common.exceptions as exp
import turnkeyml.common.printing as printing
import turnkeyml.build.onnx_helpers as onnx_helpers
import turnkeyml.build.tensor_helpers as tensor_helpers
import turnkeyml.build.export as export
import turnkeyml.build.stage as stage
import turnkeyml.build.sequences as sequences
from turnkeyml.version import __version__ as turnkey_version


def decode_version_number(version: str) -> Dict[str, int]:
    numbers = [int(x) for x in version.split(".")]
    return {"major": numbers[0], "minor": numbers[1], "patch": numbers[0]}


def validate_cached_model(
    new_state: fs.State,
    cached_state: fs.State,
    model_type: build.ModelType,
    inputs: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Verify whether anything in the call to build_model() changed
    We require the user to resolve the discrepancy when such a
    change occurs, so the purpose of this function is simply to
    detect these conditions and raise an appropriate error.
    If this function returns without raising an exception then
    the cached model is valid to use in the build.
    """

    # The first few cache miss conditions (e.g., failed build, failed state load,
    # out-of-date package) return that specific condition as the reason the cache
    # missed. Any other conditions are aggregated into a multi-condition return
    # message.

    # We required that a build was successful to load it from cache
    if vars(cached_state).get(fs.Keys.BUILD_STATUS) != build.FunctionStatus.SUCCESSFUL:
        return ["Your cached build was not successful."]

    # All of the state properties analyzed by this function
    # should be listed here.
    cache_analysis_properties = [
        fs.Keys.BUILD_NAME,
        fs.Keys.ONNX_OPSET,
        fs.Keys.DEVICE,
        fs.Keys.SEQUENCE,
        fs.Keys.BUILD_STATUS,
        fs.Keys.TURNKEY_VERSION,
        fs.Keys.MODEL_HASH,
        fs.Keys.EXPECTED_INPUT_DTYPES,
        fs.Keys.EXPECTED_INPUT_SHAPES,
        fs.Keys.DOWNCAST_APPLIED,
        fs.Keys.UID,
        fs.Keys.EVALUATION_ID,
        fs.Keys.MODEL_TYPE,
    ]

    # Make sure the cached state contains all information needed to assess a cache hit
    # so that we don't hit an attribute error down the function
    for key in cache_analysis_properties:
        if key not in vars(cached_state).keys():
            return [
                (
                    f"Your cached build is missing state key {key}. This is a bug in "
                    "turnkey itself, please contact the developers."
                )
            ]

    current_version_decoded = decode_version_number(turnkey_version)
    state_version_decoded = decode_version_number(cached_state.turnkey_version)

    out_of_date: Union[str, bool] = False
    if current_version_decoded["major"] > state_version_decoded["major"]:
        out_of_date = "major"
    elif current_version_decoded["minor"] > state_version_decoded["minor"]:
        out_of_date = "minor"

    if out_of_date:
        return [
            (
                f"Your build {cached_state.build_name} was previously built against "
                f"turnkey version {cached_state.turnkey_version}, "
                f"however you are now using turnkey version {turnkey_version}. "
                "The previous build is "
                f"incompatible with this version of turnkey, as indicated by the {out_of_date} "
                "version number changing. See **docs/versioning.md** for details."
            )
        ]

    # Below are the cache miss properties that we will aggregate into
    # a multi-property cache miss message
    result = []

    if new_state.model is not None:
        model_changed = cached_state.model_hash != build.hash_model(
            new_state.model, model_type
        )
    else:
        model_changed = False

    if inputs is not None:
        (
            input_shapes_changed,
            input_dtypes_changed,
        ) = tensor_helpers.check_shapes_and_dtypes(
            inputs,
            cached_state.expected_input_shapes,
            cached_state.expected_input_dtypes,
            expect_downcast=cached_state.downcast_applied,
            raise_error=False,
        )
    else:
        input_shapes_changed = False
        input_dtypes_changed = False

    # Check if the results-impacting arguments have changed
    changed_args = []
    for key in [
        fs.Keys.BUILD_NAME,
        fs.Keys.ONNX_OPSET,
        fs.Keys.DEVICE,
        fs.Keys.SEQUENCE,
    ]:
        if vars(new_state)[key] != vars(cached_state)[key]:
            changed_args.append((key, vars(new_state)[key], vars(cached_state)[key]))

    # Show an error if the model changed

    # Show an error if build_name is not specified for different models on the same script
    if cached_state.uid == build.unique_id():
        msg = (
            "You are building multiple different models in the same script "
            "without specifying a unique build_model(..., build_name=) for each build."
        )
        result.append(msg)

    if model_changed:
        msg = (
            f'Model "{new_state.build_name}" changed since the last time it was built.'
        )
        result.append(msg)

    if input_shapes_changed:
        input_shapes, _ = build.get_shapes_and_dtypes(inputs)
        msg = (
            f'Input shape of model "{new_state.build_name}" changed from '
            f"{cached_state.expected_input_shapes} to {input_shapes} "
            f"since the last time it was built."
        )
        result.append(msg)

    if input_dtypes_changed:
        _, input_dtypes = build.get_shapes_and_dtypes(inputs)
        msg = (
            f'Input data type of model "{new_state.build_name}" changed from '
            f"{cached_state.expected_input_dtypes} to {input_dtypes} "
            f"since the last time it was built."
        )
        result.append(msg)

    if len(changed_args) > 0:
        for key_name, current_arg, previous_arg in changed_args:
            msg = (
                f'build_model() argument "{key_name}" for build '
                f"{new_state.build_name} changed from "
                f"{previous_arg} to {current_arg} since the last build."
            )
            result.append(msg)

    return result


def _begin_fresh_build(
    new_state: fs.State,
) -> fs.State:
    # Wipe everything in this model's build directory, except for the stats file,
    # start with a fresh State.
    stats = fs.Stats(new_state.cache_dir, new_state.build_name)

    build_dir = build.output_dir(new_state.cache_dir, new_state.build_name)

    fs.rmdir(
        build_dir,
        excludes=[
            stats.file,
            os.path.join(build_dir, fs.BUILD_MARKER),
        ],
    )
    new_state.save()

    return new_state


def _rebuild_if_needed(problem_report: str, state: fs.State):
    msg = (
        f"build_model() discovered a cached build of {state.build_name}, but decided to "
        "rebuild for the following reasons: \n\n"
        f"{problem_report} \n\n"
        "build_model() will now rebuild your model to ensure correctness. You can change this "
        "policy by setting the build_model(rebuild=...) argument."
    )
    printing.log_warning(msg)

    return _begin_fresh_build(state)


def load_from_cache(
    new_state: fs.State,
    rebuild: str,
    model_type: build.ModelType,
    inputs: Optional[Dict[str, Any]] = None,
) -> fs.State:
    """
    Decide whether we can load the model from the model cache
    (return a valid State instance) or whether we need to rebuild it (return
    a new State instance).

    We make this decision on the basis of whether the cached state used
    the same model, inputs, and build arguments as the new state generated
    by this call to the tool.
    """

    # Ensure that `rebuild` has a valid value
    if rebuild not in build.REBUILD_OPTIONS:
        raise ValueError(
            f"Received `rebuild` argument with value {rebuild}, "
            f"however the only allowed values of `rebuild` are {build.REBUILD_OPTIONS}"
        )

    if rebuild == "always":
        return _begin_fresh_build(new_state)
    else:
        # Try to load state and check if model successfully built before
        if os.path.isfile(build.state_file(new_state.cache_dir, new_state.build_name)):
            cached_state = fs.load_state(
                new_state.cache_dir,
                new_state.build_name,
            )

            cache_problems = validate_cached_model(
                new_state=new_state,
                cached_state=cached_state,
                model_type=model_type,
                inputs=inputs,
            )

            if len(cache_problems) > 0:
                cache_problems = [f"- {msg}" for msg in cache_problems]
                problem_report = "\n".join(cache_problems)

                if rebuild == "if_needed":
                    return _rebuild_if_needed(problem_report, new_state)
                if rebuild == "never":
                    msg = (
                        "build_model() discovered a cached build of "
                        f"{new_state.build_name}, and found that it "
                        "is likely invalid for the following reasons: \n\n"
                        f"{problem_report} \n\n"
                        "build_model() will raise a SkipBuild exception because you have "
                        "set rebuild=never. "
                    )
                    printing.log_warning(msg)

                    raise exp.SkipBuild(
                        "Skipping this build, by raising an exception, because it previously "
                        "failed and the `rebuild` argument is set to `never`."
                    )

            return cached_state

        else:
            # No state file found, so we have to build
            return _begin_fresh_build(new_state)


export_map = {
    build.ModelType.PYTORCH: export.ExportPytorchModel(),
    build.ModelType.ONNX_FILE: export.ReceiveOnnxModel(),
}


def validate_inputs(inputs: Dict):
    """
    Check the model's inputs and make sure they are legal. Raise an exception
    if they are not legal.
    TODO: it may be wise to validate the inputs against the model, or at least
    the type of model, as well.
    """

    if inputs is None:
        msg = """
        build_model() requires model inputs. Check your call to build_model() to make sure
        you are passing the inputs argument.
        """
        raise exp.IntakeError(msg)

    if not isinstance(inputs, dict):
        msg = f"""
        The "inputs" argument to build_model() is required to be a dictionary, where the
        keys map to the named arguments in the model's forward function. The inputs
        received by build_model() were of type {type(inputs)}, not dict.
        """
        raise exp.IntakeError(msg)


def identify_model_type(model) -> build.ModelType:
    # Validate that the model's type is supported by build_model()
    # and assign a ModelType tag
    if isinstance(model, (torch.nn.Module, torch.jit.ScriptModule)):
        model_type = build.ModelType.PYTORCH
    elif isinstance(model, str):
        if model.endswith(".onnx"):
            model_type = build.ModelType.ONNX_FILE
    else:
        raise exp.IntakeError(
            "Argument 'model' passed to build_model() is "
            f"of unsupported type {type(model)}"
        )

    return model_type


def model_intake(
    user_model,
    user_inputs,
    user_sequence: Optional[stage.Sequence],
) -> Tuple[Any, Any, stage.Sequence, build.ModelType, str]:
    # Model intake structure options:
    # user_model
    #    |
    #    |------- path to onnx model file
    #    |
    #    |------- pytorch model object

    if user_sequence is None or user_sequence.enable_model_validation:
        if user_model is None and user_inputs is None:
            msg = """
            You are running build_model() without any model, inputs, or custom Sequence. The purpose
            of non-customized build_model() is to build a model against some inputs, so you need to
            provide both.
            """
            raise exp.IntakeError(msg)

        # Make sure that if the model is a file path, it is valid
        if isinstance(user_model, str):
            if not os.path.isfile(user_model):
                msg = f"""
                build_model() model argument was passed a string (path to a model file),
                however no file was found at {user_model}.
                """
                raise exp.IntakeError(msg)

            if not user_model.endswith(".onnx"):
                msg = f"""
                build_model() received a model argument that was a string. However, model string
                arguments are required to be a path to a .onnx file, but the argument was: {user_model}
                """
                raise exp.IntakeError(msg)

            # Create dummy inputs based on the ONNX spec, if none were provided by the user
            if user_inputs is None:
                inputs = onnx_helpers.dummy_inputs(user_model)
            else:
                inputs = user_inputs
        else:
            inputs = user_inputs

        model_type = identify_model_type(user_model)

        sequence = copy.deepcopy(user_sequence)
        if sequence is None:
            sequence = stage.Sequence(
                "top_level_sequence",
                "Top Level Sequence",
                [sequences.onnx_fp32],
            )

        # If there is an ExportPlaceholder Stage in the sequence, replace it with
        # a framework-specific export Stage.
        # First, make a deepcopy of any sequence we bring in here. We do not want to modify
        # the original.
        sequence = copy.deepcopy(sequence)
        for index, stage_instance in enumerate(sequence.stages):
            if isinstance(stage_instance, export.ExportPlaceholder):
                sequence.stages[index] = export_map[model_type]

        validate_inputs(inputs)

    else:
        # We turn off a significant amount of automation and validation
        # to provide custom stages and sequences with maximum flexibility
        inputs = user_inputs
        sequence = user_sequence
        model_type = build.ModelType.UNKNOWN

    return (inputs, sequence, model_type)
