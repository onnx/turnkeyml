import os
import logging
import sys
import traceback
import platform
import subprocess
import enum
from typing import Dict, Union
import hashlib
import pkg_resources
import psutil
import yaml
import torch
import numpy as np
import turnkeyml.common.exceptions as exp


UnionValidModelInstanceTypes = Union[
    None,
    str,
    torch.nn.Module,
    torch.jit.ScriptModule,
]

if os.environ.get("TURNKEY_ONNX_OPSET"):
    DEFAULT_ONNX_OPSET = int(os.environ.get("TURNKEY_ONNX_OPSET"))
else:
    DEFAULT_ONNX_OPSET = 14

MINIMUM_ONNX_OPSET = 11

DEFAULT_REBUILD_POLICY = "if_needed"
REBUILD_OPTIONS = ["if_needed", "always", "never"]


class ModelType(enum.Enum):
    PYTORCH = "pytorch"
    PYTORCH_COMPILED = "pytorch_compiled"
    ONNX_FILE = "onnx_file"
    UNKNOWN = "unknown"


# Indicates that the build should take take any specific device into account
DEFAULT_DEVICE = "default"


def load_yaml(file_path) -> Dict:
    with open(file_path, "r", encoding="utf8") as stream:
        try:
            return yaml.load(stream, Loader=yaml.FullLoader)
        except yaml.YAMLError as e:
            raise exp.IOError(
                f"Failed while trying to open {file_path}."
                f"The exception that triggered this was:\n{e}"
            )


def output_dir(cache_dir, build_name):
    path = os.path.join(cache_dir, build_name)
    return path


def state_file(cache_dir, build_name):
    state_file_name = f"{build_name}_state.yaml"
    path = os.path.join(output_dir(cache_dir, build_name), state_file_name)
    return path


def hash_model(model, model_type: ModelType, hash_params: bool = True):
    # If the model is a path to a file, hash the file
    if model_type == ModelType.ONNX_FILE:
        # TODO: Implement a way of hashing the models but not the parameters
        # of ONNX inputs.
        if not hash_params:
            msg = "hash_params must be True for model_type ONNX_FILE"
            raise ValueError(msg)
        if os.path.isfile(model):
            with open(model, "rb") as f:
                file_content = f.read()
            return hashlib.sha256(file_content).hexdigest()
        else:
            raise ValueError(
                "hash_model received str model that doesn't correspond to a file"
            )

    elif model_type in [ModelType.PYTORCH, ModelType.PYTORCH_COMPILED]:
        # Convert model parameters and topology to string
        hashable_params = {}
        for name, param in model.named_parameters():
            hashable_params[name] = param.data
        if hash_params:
            hashable_model = (str(model) + str(hashable_params)).encode()
        else:
            hashable_model = str(model).encode()

        # Return hash of topology and parameters
        return hashlib.sha256(hashable_model).hexdigest()

    else:
        msg = f"""
        model_type "{model_type}" unsupported by this hash_model function
        """
        raise ValueError(msg)


class FunctionStatus(enum.Enum):
    """
    Status values that are assigned to stages, builds, benchmarks, and other
    functionality to help the user understand whether that function completed
    successfully or not.
    """

    # SUCCESSFUL means the stage/build/benchmark completed successfully.
    SUCCESSFUL = "successful"

    # ERROR means the stage/build/benchmark failed and threw some error that
    # was caught by turnkey. You should proceed by looking at the build
    # logs to see what happened.

    ERROR = "error"

    # TIMEOUT means the stage/build/benchmark failed because it exceeded the timeout
    # set for the turnkey command.
    TIMEOUT = "timeout"

    # KILLED means the build/benchmark failed because the system killed it. This can
    # happen because of an out-of-memory (OOM), system shutdown, etc.
    # You should proceed by re-running the build and keeping an eye on it to observe
    # why it is being killed (e.g., watch the RAM utilization to diagnose an OOM).
    KILLED = "killed"

    # The NOT_STARTED status is applied to all stages/builds/benchmarks at startup.
    # It will be replaced by one of the other status values if the stage/build/benchmark
    # has a chance to start running.
    # A value of NOT_STARTED in the report CSV indicates that the stage/build/benchmark
    # never had a chance to start because turnkey exited before that functionality had
    # a chance to start running.
    NOT_STARTED = "not_started"

    # INCOMPLETE indicates that a stage/build/benchmark started running and did not complete.
    # Each stage, build, and benchmark are marked as INCOMPLETE when they start running.
    # If you open the turnkey_stats.yaml file while the stage/build/benchmark
    # is still running, the status will show as INCOMPLETE. If the stage/build/benchmark
    # is killed without the chance to do any stats cleanup, the status will continue to
    # show as INCOMPLETE in turnkey_stats.yaml.
    # When the report CSV is created, any instance of an INCOMPLETE stage/build/benchmark
    # status will be replaced by KILLED.
    INCOMPLETE = "incomplete"


# Create a unique ID from this run by hashing pid + process start time
def unique_id():
    pid = os.getpid()
    p = psutil.Process(pid)
    start_time = p.create_time()
    return hashlib.sha256(f"{pid}{start_time}".encode()).hexdigest()


def get_shapes_and_dtypes(inputs: dict):
    """
    Return the shape and data type of each value in the inputs dict
    """
    shapes = {}
    dtypes = {}
    for key in sorted(inputs):
        value = inputs[key]
        if isinstance(
            value,
            (list, tuple),
        ):
            for v, i in zip(value, range(len(value))):
                if isinstance(v, (list, tuple)):
                    # Handle nested lists/tuples, for example past_key_values
                    # in an LLM that has KV-caching enabled
                    for v2, i2 in zip(v, range(len(v))):
                        subsubkey = f"{key}[{i}][{i2}]"
                        shapes[subsubkey] = np.array(v2).shape
                        dtypes[subsubkey] = np.array(v2).dtype.name
                else:
                    # Handle single list/tuple
                    subkey = f"{key}[{i}]"
                    shapes[subkey] = np.array(v).shape
                    dtypes[subkey] = np.array(v).dtype.name
        elif torch.is_tensor(value):
            shapes[key] = np.array(value.detach()).shape
            dtypes[key] = np.array(value.detach()).dtype.name
        elif isinstance(value, np.ndarray):
            shapes[key] = value.shape
            dtypes[key] = value.dtype.name
        elif isinstance(value, (bool, int, float)):
            shapes[key] = (1,)
            dtypes[key] = type(value).__name__
        elif value is None:
            pass
        else:
            raise exp.Error(
                "One of the provided inputs contains the unsupported "
                f' type {type(value)} at key "{key}".'
            )

    return shapes, dtypes


class Logger:
    """
    Redirects stdout to to file (and console if needed)
    """

    def __init__(
        self,
        initial_message: str,
        log_path: str = None,
    ):
        self.debug = os.environ.get("TURNKEY_BUILD_DEBUG") == "True"
        self.terminal = sys.stdout
        self.terminal_err = sys.stderr
        self.log_path = log_path

        # Create the empty logfile
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"{initial_message}\n")

        # Disable any existing loggers so that we can capture all
        # outputs to a logfile
        self.root_logger = logging.getLogger()
        self.handlers = [handler for handler in self.root_logger.handlers]
        for handler in self.handlers:
            self.root_logger.removeHandler(handler)

        # Send any logger outputs to the logfile
        if not self.debug:
            self.file_handler = logging.FileHandler(filename=log_path)
            self.file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            self.file_handler.setFormatter(formatter)
            self.root_logger.addHandler(self.file_handler)

    def __enter__(self):
        sys.stdout = self
        sys.stderr = self

    def __exit__(self, _exc_type, _exc_value, _exc_tb):
        # Ensure we also capture the traceback as part of the logger when exceptions happen
        if _exc_type:
            traceback.print_exception(_exc_type, _exc_value, _exc_tb)

        # Stop redirecting stdout/stderr
        sys.stdout = self.terminal
        sys.stderr = self.terminal_err

        # Remove the logfile logging handler
        if not self.debug:
            self.file_handler.close()
            self.root_logger.removeHandler(self.file_handler)

            # Restore any pre-existing loggers
            for handler in self.handlers:
                self.root_logger.addHandler(handler)

    def write(self, message):
        if self.log_path is not None:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(message)
        if self.debug or self.log_path is None:
            self.terminal.write(message)
            self.terminal.flush()
            self.terminal_err.write(message)
            self.terminal_err.flush()

    def flush(self):
        # needed for python 3 compatibility.
        pass


def get_system_info():
    os_type = platform.system()
    info_dict = {}

    # Get OS Version
    try:
        info_dict["OS Version"] = platform.platform()
    except Exception as e:  # pylint: disable=broad-except
        info_dict["Error OS Version"] = str(e)

    if os_type == "Windows":
        # Get Processor Information
        try:
            proc_info = (
                subprocess.check_output("wmic cpu get name", shell=True)
                .decode()
                .split("\n")[1]
                .strip()
            )
            info_dict["Processor"] = proc_info
        except Exception as e:  # pylint: disable=broad-except
            info_dict["Error Processor"] = str(e)

        # Get OEM System Information
        try:
            oem_info = (
                subprocess.check_output("wmic computersystem get model", shell=True)
                .decode()
                .split("\n")[1]
                .strip()
            )
            info_dict["OEM System"] = oem_info
        except Exception as e:  # pylint: disable=broad-except
            info_dict["Error OEM System"] = str(e)

        # Get Physical Memory in GB
        try:
            mem_info_bytes = (
                subprocess.check_output(
                    "wmic computersystem get TotalPhysicalMemory", shell=True
                )
                .decode()
                .split("\n")[1]
                .strip()
            )
            mem_info_gb = round(int(mem_info_bytes) / (1024**3), 2)
            info_dict["Physical Memory"] = f"{mem_info_gb} GB"
        except Exception as e:  # pylint: disable=broad-except
            info_dict["Error Physical Memory"] = str(e)

    elif os_type == "Linux":
        # WSL has to be handled differently compared to native Linux
        if "microsoft" in str(platform.release()):
            try:
                oem_info = (
                    subprocess.check_output(
                        'powershell.exe -Command "wmic computersystem get model"',
                        shell=True,
                    )
                    .decode()
                    .strip()
                )
                oem_info = (
                    oem_info.replace("\r", "")
                    .replace("\n", "")
                    .split("Model")[-1]
                    .strip()
                )
                info_dict["OEM System"] = oem_info
            except Exception as e:  # pylint: disable=broad-except
                info_dict["Error OEM System (WSL)"] = str(e)

        else:
            # Get OEM System Information
            try:
                oem_info = (
                    subprocess.check_output(
                        "sudo -n dmidecode -s system-product-name",
                        shell=True,
                        stderr=subprocess.DEVNULL,
                    )
                    .decode()
                    .strip()
                    .replace("\n", " ")
                )
                info_dict["OEM System"] = oem_info
            except subprocess.CalledProcessError:
                # This catches the case where sudo requires a password
                info_dict["OEM System"] = "Unable to get oem info - password required"
            except Exception as e:  # pylint: disable=broad-except
                info_dict["Error OEM System"] = str(e)

        # Get CPU Information
        try:
            cpu_info = subprocess.check_output("lscpu", shell=True).decode()
            for line in cpu_info.split("\n"):
                if "Model name:" in line:
                    info_dict["Processor"] = line.split(":")[1].strip()
                    break
        except Exception as e:  # pylint: disable=broad-except
            info_dict["Error Processor"] = str(e)

        # Get Memory Information
        try:
            mem_info = (
                subprocess.check_output("free -m", shell=True)
                .decode()
                .split("\n")[1]
                .split()[1]
            )
            mem_info_gb = round(int(mem_info) / 1024, 2)
            info_dict["Memory Info"] = f"{mem_info_gb} GB"
        except Exception as e:  # pylint: disable=broad-except
            info_dict["Error Memory Info"] = str(e)

    else:
        info_dict["Error"] = "Unsupported OS"

    # Get Python Packages
    try:
        installed_packages = pkg_resources.working_set
        info_dict["Python Packages"] = [
            f"{i.key}=={i.version}"
            for i in installed_packages  # pylint: disable=not-an-iterable
        ]
    except Exception as e:  # pylint: disable=broad-except
        info_dict["Error Python Packages"] = str(e)

    return info_dict
