from typing import Dict, Optional, List
import multiprocessing
import argparse
import traceback
import psutil
import turnkeyml.common.build as build
import turnkeyml.common.exceptions as exp
import turnkeyml.common.filesystem as fs
import turnkeyml.common.printing as printing
import turnkeyml.cli.parser_helpers as parser_helpers
from turnkeyml.common.management_tools import ManagementTool
from turnkeyml.run.benchmark_model import Benchmark
from turnkeyml.run.devices import SUPPORTED_RUNTIMES

# The licensing for tqdm is confusing. Pending a legal scan,
# the following code provides tqdm to users who have installed
# it already, while being transparent to users who do not
# have tqdm installed.
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):  # pylint: disable=unused-argument
        return iterable


class SkippedBenchmark(Exception):
    """
    Indicates that a benchmark was skipped
    """


class Process(multiprocessing.Process):
    """
    Standardized way to make it possible to catch exceptions from a
    multiprocessing.Process.
    """

    def __init__(self, *args, **kwargs):
        multiprocessing.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = multiprocessing.Pipe()
        self._exception = None

    def run(self):
        try:
            multiprocessing.Process.run(self)
            self._cconn.send(None)
        except Exception as e:  # pylint: disable=broad-except
            tb = traceback.format_exc()
            self._cconn.send((e, tb))

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


def benchmark_build(
    cache_dir: str,
    build_name: str,
    runtime: str,
    iterations: int,
    rt_args: Optional[Dict] = None,
):
    """
    Benchmark the build artifact from a successful turnkey build.

    For example, `turnkey linear.py --build-only` would produce a build whose
    resulting artifact is an optimized ONNX file. This function would benchmark
    that optimized ONNX file.

    How it works:
        1. Attempt to load build state from the cache_dir/build_name specified
        2. Pass the build state directly into an instance of BaseRT and
            run the benchmark method
        3. Save stats to the same evaluation entry from the original build

    Args:
        cache_dir: same as turnkey
        build_name: same as turnkey
        runtime: same as turnkey
        iterations: same as turnkey
        rt_args: same as turnkey
    """

    state = fs.load_state(cache_dir, build_name)

    if state.build_status != build.FunctionStatus.SUCCESSFUL:
        raise SkippedBenchmark(
            "Only successful builds can be benchmarked with this "
            f"function, however selected build at {build_name} "
            f"has state: {state.build_status}"
        )

    state = Benchmark().fire(
        state, runtime=runtime, iterations=iterations, rt_args=rt_args
    )


skip_policy_default = "attempted"


class BenchmarkBuild(ManagementTool):
    """
    Benchmark pre-built models that are stored in a cache.
    """

    unique_name = "benchmark-build"

    @staticmethod
    def parser(add_help: bool = True) -> argparse.ArgumentParser:
        # NOTE: `--cache-dir` is set as a global input to the turnkey CLI and
        # passed directly to the `run()` method

        parser = argparse.ArgumentParser(
            description="Benchmark pre-built models that are stored in a cache",
            add_help=add_help,
        )

        cache_benchmark_group = parser.add_mutually_exclusive_group(required=True)

        cache_benchmark_group.add_argument(
            "--build-names",
            nargs="+",
            help="Name of the specific build to be benchmarked, within the cache directory",
        )

        cache_benchmark_group.add_argument(
            "--all",
            dest="benchmark_all",
            help="Benchmark all builds in the cache directory",
            action="store_true",
        )

        parser.add_argument(
            "--skip",
            choices=[skip_policy_default, "failed", "successful", "none"],
            dest="skip_policy",
            help="Sets the policy for skipping benchmark attempts "
            f"(defaults to {skip_policy_default})."
            "`attempted` means to skip any previously-attempted benchmark, "
            "whether it succeeded or failed."
            "`failed` skips benchmarks that have already failed once."
            "`successful` skips benchmarks that have already succeeded."
            "`none` will attempt all benchmarks, regardless of whether "
            "they were previously attempted.",
            required=False,
            default=skip_policy_default,
        )

        parser.add_argument(
            "--timeout",
            type=int,
            default=1800,
            help="Benchmark timeout, in seconds, after which each benchmark will be canceled "
            "(default: 30min).",
        )

        parser.add_argument(
            "--runtime",
            choices=SUPPORTED_RUNTIMES.keys(),
            dest="runtime",
            help="Software runtime that will be used to collect the benchmark. "
            "Must be compatible with the device chosen for the build. "
            "If this argument is not set, the default runtime of the selected device will be used.",
            required=False,
            default=None,
        )

        parser.add_argument(
            "--iterations",
            dest="iterations",
            type=int,
            default=100,
            help="Number of execution iterations of the model to capture\
                the benchmarking performance (e.g., mean latency)",
        )

        parser.add_argument(
            "--rt-args",
            dest="rt_args",
            type=str,
            nargs="*",
            help="Optional arguments provided to the runtime being used",
        )

        return parser

    def parse(self, args, known_only=True) -> argparse.Namespace:
        parsed_args = super().parse(args, known_only)
        parsed_args.rt_args = parser_helpers.decode_args(parsed_args.rt_args)
        return parsed_args

    def run(
        self,
        cache_dir: str,
        build_names: List[str] = None,
        benchmark_all: bool = False,
        skip_policy: str = skip_policy_default,
        runtime: Optional[str] = None,
        iterations: int = 100,
        timeout: Optional[int] = None,
        rt_args: Optional[Dict] = None,
    ):
        """
        Benchmark one or more builds in a cache using the benchmark_build()
        function.

        These benchmarks always run in process isolation mode because the purpose
        of this function is to quickly iterate over many builds.
        """

        printing.log_warning(
            "This is an experimental feature. Our plan is to deprecate it "
            "in favor of a new command, `turnkey benchmark cache/*`, ASAP. "
            "Please see https://github.com/onnx/turnkeyml/issues/115 "
            "for more info.\n\n"
        )

        if benchmark_all:
            builds = fs.get_available_builds(cache_dir)
        else:
            builds = build_names

        # Iterate over all of the selected builds and benchmark them
        for build_name in tqdm(builds):
            if not fs.is_build_dir(cache_dir, build_name):
                raise exp.CacheError(
                    f"No build found with name: {build_name}. "
                    "Try running `turnkey cache list` to see the builds in your build cache."
                )

            stats = fs.Stats(cache_dir, build_name)

            # Apply the skip policy by skipping over this iteration of the
            # loop if the evaluation's pre-existing benchmark status doesn't
            # meet certain criteria
            eval_stats = stats.stats
            benchmark_status_key = "stage_status:benchmark"
            if (
                benchmark_status_key in eval_stats
                and eval_stats[benchmark_status_key] != build.FunctionStatus.NOT_STARTED
            ):
                if skip_policy == "attempted":
                    printing.log_warning(
                        f"Skipping because it was previously attempted: {build_name}"
                    )
                    continue
                elif (
                    skip_policy == "successful"
                    and eval_stats[benchmark_status_key]
                    == build.FunctionStatus.SUCCESSFUL
                ):
                    printing.log_warning(
                        f"Skipping because it was already successfully benchmarked: {build_name}"
                    )
                    continue
                elif (
                    skip_policy == "failed"
                    and eval_stats[benchmark_status_key]
                    != build.FunctionStatus.SUCCESSFUL
                ):
                    printing.log_warning(
                        f"Skipping because it was previously attempted and failed: {build_name}"
                    )
                    continue
                elif skip_policy == "none":
                    # Skip policy of "none" means we should never skip over a build
                    pass

            printing.log_info(f"Attempting to benchmark: {build_name}")

            p = Process(
                target=benchmark_build,
                args=[cache_dir, build_name, runtime, iterations, rt_args],
            )
            p.start()
            p.join(timeout=timeout)

            if p.is_alive():
                # Handle the timeout, which is needed if the process is still alive after
                # waiting `timeout` seconds
                parent = psutil.Process(p.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
                stats.save_stat(benchmark_status_key, build.FunctionStatus.TIMEOUT)

                printing.log_warning(
                    f"Benchmarking {build_name} canceled because it exceeded the {timeout} "
                    "seconds timeout"
                )
            elif p.exception:
                # Handle any exception raised by the child process. In most cases, we should
                # move on to the next benchmark. However, if the exception was a
                # HardwareError that means the underlying runtime or device
                # is not able to conduct any more benchmarking. In this case the program
                # should exit and the user should follow the suggestion in the exception
                # message (e.g., restart their computer).

                if isinstance(p.exception[0], SkippedBenchmark):
                    stats.save_stat(
                        benchmark_status_key, build.FunctionStatus.NOT_STARTED
                    )
                else:
                    stats.save_stat(benchmark_status_key, build.FunctionStatus.ERROR)

                if isinstance(p.exception[0], exp.HardwareError):
                    stats.save_stat(fs.Keys.ERROR_LOG, p.exception[1])
                    raise p.exception[0]
                else:
                    printing.log_warning("Benchmarking failed with exception:")
                    print(p.exception[1])
            else:
                printing.log_success(f"Done benchmarking: {build_name}")
