from typing import Optional, Dict, Any
import turnkeyml.sequence.ignition as ignition
from turnkeyml.sequence import Sequence
import turnkeyml.common.printing as printing
import turnkeyml.common.build as build
import turnkeyml.common.filesystem as fs
from turnkeyml.state import State


def build_model(
    sequence: Sequence,
    model: build.UnionValidModelInstanceTypes = None,
    inputs: Optional[Dict[str, Any]] = None,
    build_name: Optional[str] = None,
    cache_dir: str = fs.DEFAULT_CACHE_DIR,
    monitor: Optional[bool] = None,
    rebuild: Optional[str] = None,
    lean_cache: bool = False,
) -> State:
    """Use build a model instance into an optimized ONNX file.

    Args:
        sequence: the build tools and their arguments used to build the model.
        model: Model to be mapped to an optimized ONNX file, which can be a PyTorch
            model instance or a path to an ONNX file.
        inputs: Example inputs to the user's model. The ONNX file will be
            built to handle inputs with the same static shape only.
        build_name: Unique name for the model that will be
            used to store the ONNX file and build state on disk. Defaults to the
            name of the file that calls build_model().
        cache_dir: Directory to use as the cache for this build. Output files
            from this build will be stored at cache_dir/build_name/
            Defaults to the current working directory, but we recommend setting it to
            an absolute path of your choosing.
        monitor: Display a monitor on the command line that
            tracks the progress of this function as it builds the ONNX file.
        rebuild: determines whether to rebuild or load a cached build. Options:
            - "if_needed" (default): overwrite invalid cached builds with a rebuild
            - "always": overwrite valid cached builds with a rebuild
            - "never": load cached builds without checking validity, with no guarantee
                of functionality or correctness
            - None: Falls back to default
        lean_cache: delete build artifacts after the build has completed.

        More information is available in the Tools User Guide:
            https://github.com/onnx/turnkeyml/blob/main/docs/tools_user_guide.md
    """

    # Validate and apply defaults to the initial user arguments that
    # configure the build
    state = State(
        model=model,
        inputs=inputs,
        monitor=monitor,
        cache_dir=cache_dir,
        build_name=build_name,
        sequence_info=sequence.info,
        lean_cache=lean_cache,
    )

    # Get the state of the model from the cache if a valid build is available
    state = ignition.load_from_cache(
        new_state=state,
        rebuild=rebuild or build.DEFAULT_REBUILD_POLICY,
    )

    # Return a cached build if possible, otherwise prepare the model State for
    # a build
    if state.build_status == build.FunctionStatus.SUCCESSFUL:
        # Successful builds can be loaded from cache and returned with
        # no additional steps
        printing.log_success(
            f' Build "{state.build_name}" found in cache. Loading it!',
        )

        return state

    sequence.show_monitor(state, state.monitor)
    state = sequence.launch(state)

    printing.log_success(
        f"\n    Saved to **{build.output_dir(state.cache_dir, state.build_name)}**"
    )

    return state
