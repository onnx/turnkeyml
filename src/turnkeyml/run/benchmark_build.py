from typing import List, Dict, Optional
from multiprocessing import Process, TimeoutError
import turnkeyml.common.build as build
import turnkeyml.common.exceptions as exp
import turnkeyml.common.filesystem as fs
import turnkeyml.common.printing as printing
from turnkeyml.analyze.script import set_status_on_exception
from turnkeyml.run.devices import SUPPORTED_RUNTIMES, apply_default_runtime
import turnkeyml.cli.parser_helpers as parser_helpers


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
    """

    build_path = build.output_dir(cache_dir, build_name)

    state = build.load_state(cache_dir, build_name)

    if state.build_status != build.FunctionStatus.SUCCESSFUL:
        raise exp.BenchmarkException(
            "Only successful builds can be benchmarked with this "
            f"function, however selected build at {build_path} "
            f"has state: {state.build_status}"
        )

    selected_runtime = apply_default_runtime(state.config.device, runtime)

    if rt_args is None:
        rt_args_to_use = {}
    else:
        rt_args_to_use = rt_args

    try:
        runtime_info = SUPPORTED_RUNTIMES[selected_runtime]
    except KeyError:
        # User should never get this far without hitting an actionable error message,
        # but let's raise an exception just in case.
        raise exp.BenchmarkException(
            f"Selected runtime is not supported: {selected_runtime}"
        )

    # Load the stats file using the same evaluation ID used in the original build.
    # This allows us to augment those stats with more data instead of starting a new
    # evaluation entry.
    stats = fs.Stats(cache_dir, build_name, state.evaluation_id)

    stats.save_model_eval_stat(
        fs.Keys.BENCHMARK_STATUS, build.FunctionStatus.INCOMPLETE.value
    )

    benchmark_logfile_path = ""
    try:
        # Instantiate BaseRT for the selected runtime
        runtime_handle = runtime_info["RuntimeClass"](
            cache_dir=cache_dir,
            build_name=build_name,
            stats=stats,
            iterations=iterations,
            model=state.results[0],
            # The `inputs` argument to BaseRT is only meant for
            # benchmarking runtimes that have to keep their inputs
            # in memory (e.g., `torch-eager`). We provide None here
            # because this function only works with runtimes that
            # keep their model and inputs on disk.
            inputs=None,
            device_type=state.config.device,
            runtime=selected_runtime,
            **rt_args_to_use,
        )
        benchmark_logfile_path = runtime_handle.logfile_path
        perf = runtime_handle.benchmark()

        for key, value in vars(perf).items():
            stats.save_model_eval_stat(
                key=key,
                value=value,
            )

        stats.save_model_eval_stat(
            fs.Keys.BENCHMARK_STATUS, build.FunctionStatus.SUCCESSFUL.value
        )
    except Exception as e:
        set_status_on_exception(
            runtime_info["build_required"], state, stats, benchmark_logfile_path
        )

        raise e


def benchmark_cache_cli(args):
    """
    Wrapper function for benchmark_cache() that passes in the CLI arguments
    """

    rt_args = parser_helpers.decode_args(args.rt_args)

    benchmark_cache(
        cache_dir=args.cache_dir,
        build_names=args.build_names,
        benchmark_all=args.benchmark_all,
        skip_policy=args.skip_policy,
        runtime=args.runtime,
        iterations=args.iterations,
        timeout=args.timeout,
        rt_args=rt_args,
    )


def benchmark_cache(
    cache_dir: str,
    build_names: List[str],
    benchmark_all: bool,
    skip_policy: str,
    runtime: str,
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

    if benchmark_all:
        builds = fs.get_available_builds(cache_dir)
    else:
        builds = build_names

    for build_name in builds:
        state = build.load_state(cache_dir, build_name)
        stats = fs.Stats(cache_dir, build_name, state.evaluation_id)

        eval_stats = stats.evaluation_stats
        if fs.Keys.BENCHMARK_STATUS in eval_stats:
            if skip_policy == "attempted":
                printing.log_warning(
                    f"Skipping because it was previously attempted: {build_name}"
                )
                continue
            elif (
                skip_policy == "successful"
                and eval_stats[fs.Keys.BENCHMARK_STATUS]
                == build.FunctionStatus.SUCCESSFUL.value
            ):
                printing.log_warning(
                    f"Skipping because it was already successfully benchmarked: {build_name}"
                )
                continue
            elif (
                skip_policy == "failed"
                and eval_stats[fs.Keys.BENCHMARK_STATUS]
                != build.FunctionStatus.SUCCESSFUL.value
            ):
                printing.log_warning(
                    f"Skipping because it was previously attempted and failed: {build_name}"
                )
                continue
            elif skip_policy == "none":
                pass
            else:
                raise ValueError(f"skip_policy has unsupported value {skip_policy}")

        printing.log_info(f"Attempting to benchmark: {build_name}")

        p = Process(
            target=benchmark_build,
            args=[cache_dir, build_name, runtime, iterations, rt_args],
        )
        p.start()

        try:
            p.join(timeout=timeout)
        except TimeoutError as e:
            # Set the timeout stat
            stats.save_model_eval_stat(
                fs.Keys.BENCHMARK_STATUS, build.FunctionStatus.TIMEOUT.value
            )

            print(e)

        printing.log_success(f"Done benchmarking: {build_name}")
