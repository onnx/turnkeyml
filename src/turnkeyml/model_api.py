from typing import Any, Dict, Optional, Union, List
import turnkeyml.common.printing as printing
import turnkeyml.common.filesystem as filesystem
from turnkeyml.common.performance import MeasuredPerformance


def benchmark_model(
    model: Any,
    inputs: Dict[str, Any],
    build_name: str,
    device: str,
    runtime: str,
    runtime_info: Dict,
    iterations: int = 100,
    stats_id: str = "build",
    cache_dir: str = filesystem.DEFAULT_CACHE_DIR,
    rt_args: Optional[Dict[str, Union[str, List[str]]]] = None,
) -> MeasuredPerformance:
    """
    Benchmark a model against some inputs on target hardware
    """

    if rt_args is None:
        rt_args_to_use = {}
    else:
        rt_args_to_use = rt_args

    printing.log_info(f"Benchmarking on {device}...")
    stats = filesystem.Stats(cache_dir, build_name, stats_id)
    model_handle = runtime_info["RuntimeClass"](
        cache_dir=cache_dir,
        build_name=build_name,
        stats=stats,
        iterations=iterations,
        model=model,
        inputs=inputs,
        device_type=device,
        runtime=runtime,
        **rt_args_to_use,
    )
    perf = model_handle.benchmark()
    stats.add_build_stat(
        filesystem.Keys.BENCHMARK_STATUS, filesystem.FunctionStatus.SUCCESSFUL
    )

    perf.print()
    return perf
