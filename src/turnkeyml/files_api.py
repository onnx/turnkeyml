import time
import os
import copy
import glob
from datetime import datetime
from typing import Tuple, List, Dict, Optional, Union
import git
import turnkeyml.common.printing as printing
import turnkeyml.common.exceptions as exceptions
import turnkeyml.build.stage as stage
import turnkeyml.cli.spawn as spawn
import turnkeyml.common.filesystem as fs
import turnkeyml.common.labels as labels_library
from turnkeyml.analyze.status import Verbosity
import turnkeyml.common.build as build
from turnkeyml.build_api import build_model

# The licensing for tqdm is confusing. Pending a legal scan,
# the following code provides tqdm to users who have installed
# it already, while being transparent to users who do not
# have tqdm installed.
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):  # pylint: disable=unused-argument
        return iterable


def _select_verbosity(
    verbosity: str, input_files_expanded: List[str], process_isolation: bool
) -> Tuple[Verbosity, bool]:
    """
    Choose verbosity based on the following policies:
        1. The explicit verbosity argument takes priority over AUTO and the env var
        2. The env var takes priority over AUTO
        3. Use STATIC when there are many inputs, or in process isolation mode,
            and use DYNAMIC otherwise

    Returns the selected verbosity.
    """

    verbosity_choices = {
        field.value: field for field in Verbosity if field != Verbosity.AUTO
    }
    verbosity_env_var = os.environ.get("TURNKEY_VERBOSITY")

    if verbosity != Verbosity.AUTO.value:
        # Specific verbosity argument takes priority over env var
        verbosity_selected = verbosity_choices[verbosity]
    elif verbosity_env_var in verbosity_choices.keys():
        # Env var takes priority over AUTO
        verbosity_selected = verbosity_choices[verbosity_env_var]
    else:
        # Verbosity.AUTO and no env var
        if len(input_files_expanded) > 4 or process_isolation:
            # Automatically select STATIC if:
            # - There are many evaluations (>4), since DYNAMIC mode works
            #       best when all results fit on one screen
            # - Process isolation mode is active, since DYNAMIC mode is
            #       incompatible with process isolation
            verbosity_selected = Verbosity.STATIC
        else:
            verbosity_selected = Verbosity.DYNAMIC

    # Use a progress bar in STATIC mode if there is more than 1 input
    use_progress_bar = (
        verbosity_selected == Verbosity.STATIC and len(input_files_expanded) > 1
    )

    return verbosity_selected, use_progress_bar


def unpack_txt_inputs(input_files: List[str]) -> List[str]:
    """
    Replace txt inputs with models listed inside those files
    Note: This implementation allows for nested .txt files
    """
    txt_files_expanded = sum(
        [glob.glob(f) for f in input_files if f.endswith(".txt")], []
    )
    processed_files = []
    for input_string in txt_files_expanded:
        if not os.path.exists(input_string):
            raise exceptions.ArgError(
                f"{input_string} does not exist. Please verify the file."
            )

        with open(input_string, "r", encoding="utf-8") as file:
            lines = [line.strip() for line in file if line.strip() != ""]
            processed_files.extend(unpack_txt_inputs(lines))

    return processed_files + [f for f in input_files if not f.endswith(".txt")]


# pylint: disable=unused-argument
def benchmark_files(
    input_files: List[str],
    use_slurm: bool = False,
    process_isolation: bool = False,
    lean_cache: bool = False,
    cache_dir: str = fs.DEFAULT_CACHE_DIR,
    labels: List[str] = None,
    rebuild: Optional[str] = None,
    timeout: Optional[int] = None,
    verbosity: str = Verbosity.STATIC.value,
    sequence: Union[Dict, stage.Sequence] = None,
):

    # Capture the function arguments so that we can forward them
    # to downstream APIs
    benchmarking_args = copy.deepcopy(locals())
    regular_files = []

    # Replace .txt files with the models listed inside them
    input_files = unpack_txt_inputs(input_files)

    # Iterate through each string in the input_files list
    for input_string in input_files:
        if not any(char in input_string for char in "*?[]"):
            regular_files.append(input_string)

    # Create a list of files that don't exist on the filesystem
    # Skip the files with "::" as hashes will be decoded later
    non_existent_files = [
        file for file in regular_files if not os.path.exists(file) and "::" not in file
    ]
    if non_existent_files:
        raise exceptions.ArgError(
            f"{non_existent_files} do not exist, please verify if the file(s) exists."
        )

    # Make sure that `timeout` is only being used with `process_isolation` or `use_slurm`
    # And then set a default for timeout if the user didn't set a value
    if timeout is not None:
        if not use_slurm and not process_isolation:
            raise exceptions.ArgError(
                "The `timeout` argument is only allowed when slurm "
                "or process isolation mode is activated."
            )

        timeout_to_use = timeout
    else:
        timeout_to_use = spawn.DEFAULT_TIMEOUT_SECONDS

    benchmarking_args["timeout"] = timeout_to_use

    # Convert regular expressions in input files argument
    # into full file paths (e.g., [*.py] -> [a.py, b.py] )
    input_files_expanded = fs.expand_inputs(input_files)

    # Do not forward arguments to downstream APIs
    # that will be decoded in this function body
    benchmarking_args.pop("input_files")
    benchmarking_args.pop("labels")
    benchmarking_args.pop("use_slurm")
    benchmarking_args.pop("process_isolation")

    # Make sure the cache directory exists
    fs.make_cache_dir(cache_dir)

    # Force the user to specify a legal cache dir in NFS if they are using slurm
    if cache_dir == fs.DEFAULT_CACHE_DIR and use_slurm:
        printing.log_warning(
            "Using the default cache directory when using Slurm will cause your cached "
            "files to only be available at the Slurm node. If this is not the behavior "
            "you desired, please se a --cache-dir that is accessible by both the slurm "
            "node and your local machine."
        )

    # Get list containing only file names
    clean_file_names = [
        fs.decode_input_arg(file_name)[0] for file_name in input_files_expanded
    ]

    # Validate that the files have supported file extensions
    # Note: We are not checking for .txt files here as those were previously handled
    for file_name in clean_file_names:
        if not file_name.endswith(".py") and not file_name.endswith(".onnx"):
            raise exceptions.ArgError(
                f"File extension must be .py, .onnx, or .txt (got {file_name})"
            )

    if use_slurm:
        jobs = spawn.slurm_jobs_in_queue()
        if len(jobs) > 0:
            printing.log_warning(f"There are already slurm jobs in your queue: {jobs}")
            printing.log_info(
                "Suggest quitting turnkey, running 'scancel -u $USER' and trying again."
            )

    verbosity_policy, use_progress_bar = _select_verbosity(
        verbosity, input_files_expanded, process_isolation
    )
    benchmarking_args["verbosity"] = verbosity_policy

    # Fork the args for analysis since they have differences from the spawn args:
    # build_only and analyze_only are encoded into actions
    analysis_args = copy.deepcopy(benchmarking_args)
    analysis_args.pop("timeout")

    for file_path_encoded in tqdm(input_files_expanded, disable=not use_progress_bar):

        printing.log_info(f"Running turnkey on {file_path_encoded}")

        file_path_absolute, targets, encoded_input = fs.decode_input_arg(
            file_path_encoded
        )

        file_labels = fs.read_labels(file_path_absolute)

        build_name = fs.get_build_name(
            fs.clean_file_name(file_path_absolute),
            file_labels,
            targets[0] if len(targets) > 0 else None,
        )

        if len(targets) > 1:
            raise exceptions.ArgError(
                "Only one target (number after the ::) is allowed, "
                f"but received {encoded_input}"
            )

        # Skip a file if the required_labels are not a subset of the script_labels.
        if labels:
            # Labels argument is not supported for ONNX files
            if file_path_absolute.endswith(".onnx"):
                raise ValueError(
                    "The labels argument is not supported for .onnx files, got",
                    file_path_absolute,
                )
            required_labels = labels_library.to_dict(labels)
            if not labels_library.is_subset(required_labels, file_labels):
                continue

        if use_slurm or process_isolation:
            # Decode args into spawn.Target
            if use_slurm and process_isolation:
                raise ValueError(
                    "use_slurm and process_isolation are mutually exclusive, but both are True"
                )
            elif use_slurm:
                process_type = spawn.Target.SLURM
            elif process_isolation:
                process_type = spawn.Target.LOCAL_PROCESS
            else:
                raise ValueError(
                    "This code path requires use_slurm or use_process to be True, "
                    "but both are False"
                )

            # We want to pass sequence in explicity
            benchmarking_args.pop("sequence")

            spawn.run_turnkey(
                build_name=build_name,
                sequence=sequence,
                target=process_type,
                file_name=encoded_input,
                **benchmarking_args,
            )

        else:
            # Forward the selected input to the first stage in the sequence
            first_stage_args = next(iter(sequence.stages.values()))
            first_stage_args.append("--input")
            first_stage_args.append(file_path_encoded)

            # Create a build directory and stats file in the cache
            fs.make_build_dir(cache_dir, build_name)
            stats = fs.Stats(cache_dir, build_name)

            # Save the system information used for this build
            system_info = build.get_system_info()
            stats.save_stat(
                fs.Keys.SYSTEM_INFO,
                system_info,
            )

            # Save lables info
            if fs.Keys.AUTHOR in file_labels:
                stats.save_stat(fs.Keys.AUTHOR, file_labels[fs.Keys.AUTHOR][0])
            if fs.Keys.TASK in file_labels:
                stats.save_stat(fs.Keys.TASK, file_labels[fs.Keys.TASK][0])

            # Save all of the lables in one place
            stats.save_stat(fs.Keys.LABELS, file_labels)

            # Save a timestamp so that we know the order of builds within a cache
            stats.save_stat(
                fs.Keys.TIMESTAMP,
                datetime.now(),
            )

            # If the input script is a built-in TurnkeyML model, make a note of
            # which one
            if os.path.abspath(fs.MODELS_DIR) in os.path.abspath(file_path_absolute):
                try:
                    # If this turnkey installation is in a git repo, use the
                    # specific git hash
                    git_repo = git.Repo(search_parent_directories=True)
                    git_hash = git_repo.head.object.hexsha
                except git.exc.InvalidGitRepositoryError:
                    # If we aren't in a git repo (e.g., PyPI package), point the user back to main
                    git_hash = "main"

                relative_path = file_path_absolute.replace(
                    fs.MODELS_DIR,
                    f"https://github.com/onnx/turnkeyml/tree/{git_hash}/models",
                ).replace("\\", "/")
                stats.save_stat(fs.Keys.MODEL_SCRIPT, relative_path)

            # Indicate that the build is running. If the build fails for any reason,
            # we will try to catch the exception and note it in the stats.
            # If a concluded build still has a status of "running", this means
            # there was an uncaught exception.
            stats.save_stat(fs.Keys.BUILD_STATUS, build.FunctionStatus.INCOMPLETE)

            build_model(
                build_name=build_name,
                model=file_path_absolute,
                sequence=sequence,
                cache_dir=cache_dir,
                rebuild=rebuild,
                lean_cache=lean_cache,
            )

    # Wait until all the Slurm jobs are done
    if use_slurm:
        while len(spawn.slurm_jobs_in_queue()) != 0:
            print(
                f"Waiting: {len(spawn.slurm_jobs_in_queue())} "
                f"jobs left in queue: {spawn.slurm_jobs_in_queue()}"
            )
            time.sleep(5)
