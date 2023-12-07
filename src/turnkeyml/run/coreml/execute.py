"""
The following script is used to get the latency and outputs of a given run on the x86 CPUs.
"""
# pylint: disable = no-name-in-module
# pylint: disable = import-error
import os
import subprocess
import json
from statistics import mean
import platform
import turnkeyml.run.plugin_helpers as plugin_helpers

COREML_VERSION = "7.1"

BATCHSIZE = 1


def create_conda_env(conda_env_name: str):
    """Create a Conda environment with the given name and install requirements."""
    conda_path = os.getenv("CONDA_EXE")
    if conda_path is None:
        raise EnvironmentError(
            "CONDA_EXE environment variable not set."
            "Make sure Conda is properly installed."
        )

    env_path = os.path.join(
        os.path.dirname(os.path.dirname(conda_path)), "envs", conda_env_name
    )

    # Only create the environment if it does not already exist
    if not os.path.exists(env_path):
        plugin_helpers.run_subprocess(
            [
                conda_path,
                "create",
                "--name",
                conda_env_name,
                "python=3.8",
                "-y",
            ]
        )

    # Using conda run to execute pip install within the environment
    setup_cmd = [
        conda_path,
        "run",
        "--name",
        conda_env_name,
        "pip",
        "install",
        f"coremltools=={COREML_VERSION}",
    ]
    plugin_helpers.run_subprocess(setup_cmd)


def execute_benchmark(
    coreml_file_path: str,
    outputs_file: str,
    output_dir: str,
    conda_env_name: str,
    iterations: int,
):
    """Execute the benchmark script and retrieve the output."""

    python_in_env = plugin_helpers.get_python_path(conda_env_name)
    iterations_file = os.path.join(output_dir, "per_iteration_latency.json")
    benchmarking_log_file = os.path.join(output_dir, "coreml_benchmarking_log.txt")

    cmd = [
        python_in_env,
        os.path.join(output_dir, "within_conda.py"),
        "--coreml-file",
        coreml_file_path,
        "--iterations",
        str(iterations),
        "--iterations-file",
        iterations_file,
    ]

    # Execute command and log stdout/stderr
    plugin_helpers.logged_subprocess(
        cmd=cmd,
        cwd=os.path.dirname(output_dir),
        log_to_std_streams=False,
        log_to_file=True,
        log_file_path=benchmarking_log_file,
    )

    # Parse per-iteration performance results and save aggregated results to a json file
    if os.path.exists(iterations_file):
        with open(iterations_file, "r", encoding="utf-8") as f:
            per_iteration_latency = json.load(f)
    else:
        raise ValueError(
            f"Execution of command {cmd} failed, see {benchmarking_log_file}"
        )

    cpu_performance = get_cpu_specs()
    cpu_performance["CoreML Version"] = str(COREML_VERSION)
    cpu_performance["Mean Latency(ms)"] = str(mean(per_iteration_latency) * 1000)
    cpu_performance["Throughput"] = str(BATCHSIZE / mean(per_iteration_latency))
    cpu_performance["Min Latency(ms)"] = str(min(per_iteration_latency) * 1000)
    cpu_performance["Max Latency(ms)"] = str(max(per_iteration_latency) * 1000)

    with open(outputs_file, "w", encoding="utf-8") as out_file:
        json.dump(cpu_performance, out_file, ensure_ascii=False, indent=4)


def get_cpu_specs() -> dict:
    # Check the operating system and define the command accordingly
    if platform.system() != "Darwin":
        raise OSError("You must se MacOS to run models with CoreML.")

    cpu_info_command = "sysctl -n machdep.cpu.brand_string"
    cpu_info = subprocess.Popen(cpu_info_command.split(), stdout=subprocess.PIPE)
    cpu_info_output, _ = cpu_info.communicate()
    if not cpu_info_output:
        raise EnvironmentError(
            f"Could not get CPU info using '{cpu_info_command.split()[0]}'. "
            "Please make sure this tool is correctly installed on your system before continuing."
        )

    # Store CPU specifications
    decoded_info = cpu_info_output.decode().strip().split("\n")
    cpu_spec = {"CPU Name": decoded_info[0]}

    return cpu_spec
