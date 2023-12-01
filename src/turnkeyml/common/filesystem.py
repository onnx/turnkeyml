import os
import shutil
import glob
import pathlib
from typing import Dict, List, Optional
import importlib.util
import yaml
import turnkeyml.common.printing as printing
import turnkeyml.common.build as build
import turnkeyml.common.exceptions as exp
from turnkeyml.common import labels

# Allow an environment variable to override the default
# location for the build cache
if os.environ.get("TURNKEY_CACHE_DIR"):
    DEFAULT_CACHE_DIR = os.path.expanduser(os.environ.get("TURNKEY_CACHE_DIR"))
else:
    DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/turnkey")

if " " in DEFAULT_CACHE_DIR:
    raise ValueError(
        f"Your turnkey cache directory is set to '{DEFAULT_CACHE_DIR}', "
        "however the space in that path will cause problems. Please set a cache directory "
        "that has no spaces by setting the TURNKEY_CACHE_DIR environment variable."
    )

if DEFAULT_CACHE_DIR.endswith("\\"):
    raise ValueError(
        f"Your turnkey cache directory is set to '{DEFAULT_CACHE_DIR}', "
        "however the trailing backslash (\\) in that path will cause problems. "
        "Please set a cache directory "
        "that has no trailing backslash by setting the TURNKEY_CACHE_DIR environment variable."
    )

CACHE_MARKER = ".turnkeycache"
BUILD_MARKER = ".turnkeybuild"

# Locate the models directory
MODELS_DIR = importlib.util.find_spec("turnkeyml_models").submodule_search_locations[0]


def rmdir(folder, exclude: Optional[str] = None):
    """
    Remove the contents of a directory from the filesystem.
    If `exclude=<name>`, the directory itself and the file named <name>
    are kept. Otherwise, the entire directory is removed.
    """
    if os.path.isdir(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if file_path != exclude:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

        if exclude is None:
            shutil.rmtree(folder)

        return True

    else:
        return False


def get_all(path, exclude_path=False, file_type="state.yaml", recursive=True):
    if recursive:
        files = [
            os.path.join(dp, f)
            for dp, dn, filenames in os.walk(path)
            for f in filenames
            if file_type in f
        ]
    else:
        files = []
        dp, _, filenames = os.walk(path)
        for f in filenames:
            if file_type in f:
                files.append(os.path.join(dp, f))

    if exclude_path:
        files = [os.path.basename(f) for f in files]

    return files


def clean_file_name(script_path: str) -> str:
    # Trim the ".py" / ".onnx"
    return pathlib.Path(script_path).stem


class CacheError(exp.Error):
    """
    Raise this exception when the cache is being accessed incorrectly
    """


def _load_yaml(file) -> Dict:
    if os.path.isfile(file):
        with open(file, "r", encoding="utf8") as stream:
            return yaml.load(stream, Loader=yaml.FullLoader)
    else:
        return {}


def _save_yaml(dict: Dict, file):
    with open(file, "w", encoding="utf8") as outfile:
        yaml.dump(dict, outfile)


def print_yaml_file(file_path, description):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            printing.log_info(f"The {description} for {file_path} are:")
            print(file.read())
    else:
        raise CacheError(
            f"No {description} found at {file_path}. "
            "Try running `turnkey cache list` to see the builds in your build cache."
        )


def make_cache_dir(cache_dir: str):
    """
    Create the build and cache directories, and put hidden files in them
    to mark them as such.
    """

    os.makedirs(cache_dir, exist_ok=True)

    # File that indicates that the directory is a cache directory
    cache_file_path = os.path.join(cache_dir, CACHE_MARKER)
    open(cache_file_path, mode="w", encoding="utf").close()


def make_build_dir(cache_dir: str, build_name: str):
    """
    Create the build and cache directories, and put hidden files in them
    to mark them as such.
    """
    make_cache_dir(cache_dir)

    build_dir = build.output_dir(cache_dir, build_name)
    os.makedirs(build_dir, exist_ok=True)

    # File that indicates that the directory is a build directory
    build_file_path = os.path.join(build_dir, BUILD_MARKER)
    open(build_file_path, mode="w", encoding="utf").close()


def check_cache_dir(cache_dir: str):
    cache_file_path = os.path.join(cache_dir, CACHE_MARKER)
    if not os.path.isfile(cache_file_path):
        raise CacheError(
            f"{cache_dir} is not a cache directory generated by TurnkeyML. "
            "You can only clean, delete and generate reports for directories that "
            "have been generated by TurnkeyML. Set a different --cache-dir before "
            "trying again."
        )


def is_build_dir(cache_dir: str, build_name: str):
    build_dir = build.output_dir(cache_dir, build_name)
    build_file_path = os.path.join(build_dir, BUILD_MARKER)
    return os.path.isfile(build_file_path)


def clean_output_dir(cache_dir: str, build_name: str) -> None:
    """
    Delete all elements of the output directory that are not human readable
    """
    output_dir = os.path.join(cache_dir, build_name)
    if os.path.isdir(output_dir) and is_build_dir(cache_dir, build_name):
        output_dir = os.path.expanduser(output_dir)
    else:
        raise CacheError(f"No build found at {output_dir}")

    # Remove files that do not have an allowed extension
    allowed_extensions = (".txt", ".out", ".yaml", ".json")
    all_paths = glob.glob(f"{output_dir}/**/*", recursive=True)
    for path in all_paths:
        if os.path.isfile(path) and not path.endswith(allowed_extensions):
            os.remove(path)

    # Remove all empty folders
    for path in all_paths:
        if os.path.isdir(path):
            if len(os.listdir(path)) == 0:
                shutil.rmtree(path)


def get_available_scripts(search_dir: str):
    scripts = [
        f
        for f in os.listdir(search_dir)
        if os.path.isfile(os.path.join(search_dir, f)) and ".py" in f
    ]

    return scripts


def get_available_builds(cache_dir):
    """
    Get all of the build directories within the build cache
    located at `cache_dir`
    """

    check_cache_dir(cache_dir)

    builds = [
        pathlib.PurePath(build).name
        for build in os.listdir(os.path.abspath(cache_dir))
        if os.path.isdir(os.path.join(cache_dir, build))
        and is_build_dir(cache_dir, build)
    ]
    builds.sort()

    return builds


def print_available_builds(args):
    printing.log_info(f"Builds available in cache {args.cache_dir}:")
    builds = get_available_builds(args.cache_dir)
    printing.list_table(builds, num_cols=1)
    print()


def delete_builds(args):
    check_cache_dir(args.cache_dir)

    if args.delete_all:
        builds = get_available_builds(args.cache_dir)
    else:
        builds = [args.build_name]

    for build in builds:
        build_path = os.path.join(args.cache_dir, build)
        if is_build_dir(args.cache_dir, build):
            rmdir(build_path)
            printing.log_info(f"Deleted build: {build}")
        else:
            raise CacheError(
                f"No build found with name: {build}. "
                "Try running `turnkey cache list` to see the builds in your build cache."
            )


def clean_builds(args):
    check_cache_dir(args.cache_dir)

    if args.clean_all:
        builds = get_available_builds(args.cache_dir)
    else:
        builds = [args.build_name]

    for build in builds:
        if is_build_dir(args.cache_dir, build):
            clean_output_dir(args.cache_dir, build)
            printing.log_info(f"Removed the build artifacts from: {build}")
        else:
            raise CacheError(
                f"No build found with name: {build}. "
                "Try running `turnkey cache list` to see the builds in your build cache."
            )


def clean_build_name(build_name: str) -> str:
    """
    Remove hash from build name
    Build names have the format: <script_name>_<author>_hash
    """

    # Get everything except the trailing _<hash>
    return "_".join(build_name.split("_")[:-1])


def get_build_name(
    script_name: str, script_labels: Dict[str, List], model_hash: str = None
):
    """
    Create build name from script_name, labels and model hash
    """
    build_name = script_name
    if "author" in script_labels:
        build_name += f"_{script_labels['author'][0]}"
    if model_hash:
        build_name += f"_{model_hash}"
    return build_name


def get_builds_from_file(cache_dir, script):
    script_name = clean_file_name(script)
    script_labels = labels.load_from_file(script)
    all_builds_in_cache = get_available_builds(cache_dir)

    script_builds = [
        x
        for x in all_builds_in_cache
        if get_build_name(script_name, script_labels) == clean_build_name(x)
    ]

    return script_builds, script_name


class Keys:
    # Unique hash value that identifies the model+inputs+args
    # for a workload
    HASH = "hash"
    # Number of parameters in the model
    PARAMETERS = "parameters"
    # Histogram of ONNX operators used in the model
    ONNX_OPS_COUNTER = "onnx_ops_counter"
    # ONNX model info: IR version, opset, and size on disk (KiB)
    ONNX_MODEL_INFO = "onnx_model_information"
    # ONNX model input tensor dimensions
    ONNX_INPUT_DIMENSIONS = "onnx_input_dimensions"
    # List of all build stages in the Sequence
    ALL_BUILD_STAGES = "all_build_stages"
    # Map of build stages that completed successfully to the
    # execution time for that stage. We can figure out if any build
    # stages failed if all_build_stages != completed_build_stages.keys().
    COMPLETED_BUILD_STAGES = "completed_build_stages"
    # Location of the most up-to-date ONNX file for this build. If the
    # build completed successfully, this is the final ONNX file.
    ONNX_FILE = "onnx_file"
    # MeasuredPerformance data for a benchmarked workload
    PERFORMANCE = "performance"
    # Runtime used for the benchmark
    RUNTIME = "runtime"
    # Device used for the benchmark
    DEVICE_TYPE = "device_type"
    # Name of the model
    MODEL_NAME = "model_name"
    # References the per-build stats section
    BUILDS = "builds"
    # Author of the model
    AUTHOR = "author"
    # Class type of the model
    CLASS = "class"
    # Task the model is meant to perform
    TASK = "task"
    # Number of iterations used in benchmarking
    ITERATIONS = "iterations"
    # System information to keep track of DUT
    SYSTEM_INFO = "system_info"
    # Path to the built-in model script used as input
    MODEL_SCRIPT = "builtin_model_script"
    # Indicates status of the most recent analysis tool run: FunctionStatus
    ANALYSIS_STATUS = "analysis_status"
    # Indicates status of the most recent build tool run: FunctionStatus
    BUILD_STATUS = "build_status"
    # Indicates status of the most recent benchmark tool run: FunctionStatus
    BENCHMARK_STATUS = "benchmark_status"


class FunctionStatus:
    RUNNING = "running"
    SUCCESSFUL = "successful"
    FAILED = "failed"
    KILLED = "killed"


class Stats:
    def __init__(self, cache_dir: str, build_name: str, stats_id: str = None):
        output_dir = build.output_dir(cache_dir, build_name)

        self.file = os.path.join(output_dir, "turnkey_stats.yaml")
        self.stats_id = stats_id

        os.makedirs(output_dir, exist_ok=True)
        if not os.path.exists(self.file):
            initial = {Keys.BUILDS: {}}
            _save_yaml(initial, self.file)

    @property
    def stats(self):
        return _load_yaml(self.file)

    def _set_key(self, dict, keys: List["str"], value):
        """
        Recursive approach to safely setting a key within any level of hierarchy
        in a dictionary. If a parent key of the desired key does not exist, create
        it and set it with an empty dictionary before proceeding.

        The end result is: dict[keys[0]][keys[1]]...[keys[-1]] = value
        """
        if len(keys) == 1:
            dict[keys[0]] = value

        else:
            if keys[0] not in dict.keys():
                dict[keys[0]] = {}

            self._set_key(dict[keys[0]], keys[1:], value)

    def save_stat(self, key: str, value):
        """
        Save statistics to an yaml file in the build directory
        """

        stats_dict = self.stats

        self._set_key(stats_dict, [key], value)

        _save_yaml(stats_dict, self.file)

    def add_sub_stat(self, parent_key: str, key: str, value):
        """
        Save nested statistics to an yaml file in the build directory

        stats[parent_key][key] = value
        """

        stats_dict = self.stats

        self._set_key(stats_dict, [parent_key, key], value)

        _save_yaml(stats_dict, self.file)

    def add_build_stat(self, key: str, value):
        stats_dict = self.stats

        self._set_key(stats_dict, [Keys.BUILDS, self.stats_id, key], value)

        _save_yaml(stats_dict, self.file)

    def add_build_sub_stat(self, parent_key: str, key: str, value):
        stats_dict = self.stats

        self._set_key(stats_dict, [Keys.BUILDS, self.stats_id, parent_key, key], value)

        _save_yaml(stats_dict, self.file)

    @property
    def build_stats(self):
        return self.stats[Keys.BUILDS][self.stats_id]


def print_cache_dir(_=None):
    printing.log_info(f"The default cache directory is: {DEFAULT_CACHE_DIR}")


def print_models_dir(args=None):
    if args.verbose:
        printing.log_info(f"The models directory is: {MODELS_DIR}")
    else:
        print(MODELS_DIR)


def expand_inputs(input_paths: List[str]) -> List[str]:
    """
    Convert regular expressions in input paths
    into full file/dir paths (e.g., [*.py] -> [a.py, b.py] )

    This makes up for Windows not resolving wildcards on the command line
    """
    input_paths_expanded = sum(
        [glob.glob(f) for f in input_paths if "::" not in f], []
    ) + [f for f in input_paths if "::" in f]

    if not input_paths_expanded:
        raise exp.ArgError("No files that match your inputs could be found.")

    return input_paths_expanded


def rebase_cache_dir(input_path: str, build_name: str, new_cache_dir: str):
    """
    Rebase a turnkey build path onto a new turnkey cache directory.

    For example:
        1. You built a model in cache_A
        2. The turnkey_stats.yaml references cache_A/build_dir/onnx/model.onnx
        3. You move the build to cache_B
        4. You now call this function to get a new path, cache_B/build_dir/onnx/model.onnx

    This works by:
        1. Split on the first occurance of the build name: [cache_A, onnx/model.onnx]
        2. Prepend the new cache dir, as well as the build dir (since it was removed in step 1)
    """

    relative_input_path = input_path.split(build_name, 1)[1][1:]
    return os.path.join(new_cache_dir, build_name, relative_input_path)
