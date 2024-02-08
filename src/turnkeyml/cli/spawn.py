"""
Utilities for spawning new turnkey calls, in both local processes and on Slurm
"""

import os
import subprocess
import pathlib
import time
import shlex
import platform
from threading import Event, Lock, Thread
from time import monotonic
import getpass
from typing import List, Optional, Dict, Union
from enum import Enum
import turnkeyml.common.filesystem as filesystem
import turnkeyml.common.printing as printing
import turnkeyml.common.build as build
from turnkeyml.cli.parser_helpers import encode_args


class WatchdogTimer(Thread):
    """Run *callback* in *timeout* seconds unless the timer is restarted."""

    def __init__(self, timeout, callback, *args, timer=monotonic, **kwargs):
        super().__init__(**kwargs)
        self.timeout = timeout
        self.callback = callback
        self.args = args
        self.timer = timer
        self.cancelled = Event()
        self.blocked = Lock()
        self.timeout_reached = False

    def run(self):
        self.restart()  # don't start timer until `.start()` is called
        # wait until timeout happens or the timer is canceled
        while not self.cancelled.wait(self.deadline - self.timer()):
            # don't test the timeout while something else holds the lock
            # allow the timer to be restarted while blocked
            with self.blocked:
                if self.deadline <= self.timer() and not self.cancelled.is_set():
                    self.timeout_reached = True
                    return self.callback(*self.args)  # on timeout

    def restart(self):
        """Restart the watchdog timer."""
        self.deadline = self.timer() + self.timeout

    def cancel(self):
        self.cancelled.set()


if os.environ.get("TURNKEY_TIMEOUT_SECONDS"):
    timeout_env_var = os.environ.get("TURNKEY_TIMEOUT_SECONDS")
    SECONDS_IN_A_DAY = 60 * 60 * 24
    if timeout_env_var > SECONDS_IN_A_DAY:
        raise ValueError(
            f"Value of TURNKEY_TIMEOUT_SECONDS must be less than 1 day, got {timeout_env_var}"
        )
    DEFAULT_TIMEOUT_SECONDS = int(timeout_env_var)
else:
    DEFAULT_TIMEOUT_SECONDS = 3600


class Target(Enum):
    SLURM = "slurm"
    LOCAL_PROCESS = "local_process"


def slurm_jobs_in_queue(job_name=None) -> List[str]:
    """Return the set of slurm jobs that are currently pending/running"""
    user = getpass.getuser()
    if job_name is None:
        output = subprocess.check_output(["squeue", "-u", user])
    else:
        output = subprocess.check_output(["squeue", "-u", user, "--name", job_name])
    output = output.split(b"\n")
    output = [s.decode("utf").split() for s in output]

    # Remove headers
    output.pop(0)

    # Remove empty line at the end
    output.pop(-1)

    # Get just the job names
    if len(output) > 0:
        name_index_in_squeue = 2
        output = [s[name_index_in_squeue] for s in output]

    return output


def arg_format(name: str):
    name_underscores = name.replace("_", "-")
    return f"--{name_underscores}"


def list_arg(key: str, values: List):
    if values is not None:
        result = " ".join(values)
        return f"{key} {result}"
    else:
        return ""


def value_arg(key: str, value: Union[str, int]):
    if value is not None:
        return f'{key}="{value}"'
    else:
        return ""


def bool_arg(key: str, value: bool):
    if value:
        return f"{key}"
    else:
        return ""


def dict_arg(key: str, value: Dict):
    if value:
        return f"{key} {' '.join(encode_args(value))}"
    else:
        return ""


def run_turnkey(
    op: str,
    file_name: str,
    target: Target,
    cache_dir: str,
    timeout: Optional[int] = DEFAULT_TIMEOUT_SECONDS,
    working_dir: str = os.getcwd(),
    ml_cache_dir: Optional[str] = os.environ.get("SLURM_ML_CACHE"),
    max_jobs: int = 50,
    **kwargs,
):
    """
    Run turnkey on a single input file in a separate process (e.g., Slurm, subprocess).
    Any arguments that should be passed to the new turnkey process must be provided via kwargs.

    kwargs must also match the following format:
      The key must be the snake_case version of the CLI argument (e.g, build_only for --build-only)
    """

    type_to_formatter = {
        str: value_arg,
        int: value_arg,
        bool: bool_arg,
        list: list_arg,
        dict: dict_arg,
    }

    invocation_args = f"{op} {file_name}"

    for key, value in kwargs.items():
        if value is not None:
            arg_str = type_to_formatter[type(value)](arg_format(key), value)
            invocation_args = invocation_args + " " + arg_str

    if target == Target.SLURM:
        # Change args into the format expected by Slurm
        slurm_args = " ".join(shlex.split(invocation_args))

        # Remove the .py extension from the build name
        job_name = filesystem.clean_file_name(file_name)

        # Put the timeout into format days-hh:mm:ss
        hh_mm_ss = time.strftime("%H:%M:%S", time.gmtime(timeout))
        slurm_format_timeout = f"00-{hh_mm_ss}"

        while len(slurm_jobs_in_queue()) >= max_jobs:
            print(
                f"Waiting: Your number of jobs running ({len(slurm_jobs_in_queue())}) "
                "matches or exceeds the maximum "
                f"concurrent jobs allowed ({max_jobs}). "
                f"The jobs in queue are: {slurm_jobs_in_queue()}"
            )
            time.sleep(5)

        shell_script = os.path.join(
            pathlib.Path(__file__).parent.resolve(), "run_slurm.sh"
        )

        slurm_command = ["sbatch", "-c", "1"]
        if os.environ.get("TURNKEY_SLURM_USE_DEFAULT_MEMORY") != "True":
            slurm_command.append("--mem=128000")
        slurm_command.extend(
            [
                f"--time={slurm_format_timeout}",
                f"--job-name={job_name}",
                shell_script,
                "turnkey",
                slurm_args,
                working_dir,
            ]
        )
        if ml_cache_dir is not None:
            slurm_command.append(ml_cache_dir)

        print(f"Submitting job {job_name} to Slurm")
        subprocess.check_call(slurm_command)
    elif target == Target.LOCAL_PROCESS:
        command = "turnkey " + invocation_args
        printing.log_info(f"Starting process with command: {command}")

        # Linux and Windows want to handle some details differently
        if platform.system() != "Windows":
            command = shlex.split(command)

        # Launch a subprocess for turnkey to evaluate the script
        try:

            process_output = []
            with subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            ) as p:
                watchdog = WatchdogTimer(timeout, callback=p.kill, daemon=True)
                watchdog.start()

                for line in p.stdout:
                    print(line, end="")
                    process_output.append(line)

                p.wait()
                watchdog.cancel()

                if watchdog.timeout_reached:
                    raise subprocess.TimeoutExpired(p.args, timeout)

                if p.returncode != 0:
                    raise subprocess.CalledProcessError(p.returncode, p.args)

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            # Print some newlines so that our error messages don't end up in the
            # middle of the status monitor
            print("\n\n")
            printing.log_error(
                "The turnkey subprocess was terminated with the error shown below. "
                f"turnkey will move on to the next input.\n\n{e}"
            )

            if isinstance(e, subprocess.TimeoutExpired):
                evaluation_status = build.FunctionStatus.TIMEOUT
            else:
                evaluation_status = build.FunctionStatus.KILLED

            # If a build failed, it will be the last build mentioned in the
            # subprocess's stdout
            build_name = None
            evaluation_id = None
            for line in process_output:
                if "Capturing statistics in turnkey_stats.yaml" in line:
                    # This indicates a stats file was created for this evaluation
                    # Expected phrase: "Capturing statistics in turnkey_stats.yaml
                    #   under evaluation ID: {evaluation_id}"
                    evaluation_id = line.split("ID: ")[1].rstrip()
                if "Build dir:" in line:
                    # This declares the name of the build directory
                    # 'Build' directories are created for any evaluation, even
                    # if there is not actually a build (e.g., torch-eager runtime benchmark),
                    # which is why we use this line to find the build directory.
                    # Expected phrase:
                    #   "Build dir:      {cache_dir}/{build_name}"
                    build_name = os.path.basename(
                        os.path.normpath(line.split(":")[1].rstrip())
                    )

            if build_name:
                printing.log_info(
                    f"Detected failed build {build_name}. The parent process will attempt to clean up."
                )
                if "--lean-cache" in command:
                    printing.log_info("Removing build artifacts...")
                    filesystem.clean_output_dir(cache_dir, build_name)

                # Update the stats file for the build, if it exists
                if (
                    os.path.isfile(filesystem.stats_file(cache_dir, build_name))
                    and evaluation_id
                ):
                    try:
                        stats = filesystem.Stats(
                            cache_dir,
                            build_name,
                            evaluation_id,
                        )

                        for key in stats.evaluation_stats.keys():
                            if (
                                stats.evaluation_stats[key]
                                == build.FunctionStatus.INCOMPLETE.value
                            ):
                                stats.save_model_eval_stat(key, evaluation_status.value)

                    except Exception as stats_exception:  # pylint: disable=broad-except
                        printing.log_info(
                            "Stats file found, but unable to perform cleanup due to "
                            f"exception: {stats_exception}"
                        )

            else:
                printing.log_info(
                    "Turnkey subprocess was killed before any "
                    "build or benchmark could start."
                )

    else:
        raise ValueError(f"Unsupported value for target: {target}.")
