import os
import sys
import logging
import warnings
from typing import Dict, Any, List, Optional
from statistics import mean
import time
from packaging import version
import torch
import numpy as np
from turnkeyml.run.basert import BaseRT
from turnkeyml.common.performance import MeasuredPerformance
from turnkeyml.run.onnxrt.execute import get_cpu_specs
import turnkeyml.build.ignition as ignition
import turnkeyml.common.build as build
import turnkeyml.common.exceptions as exp
import turnkeyml.common.filesystem as fs
from turnkeyml.common.filesystem import Stats


def customwarn(message, category, filename, lineno, file=None, line=None):
    sys.stdout.write(warnings.formatwarning(message, category, filename, lineno))


class TorchRT(BaseRT):
    def __init__(
        self,
        cache_dir: str,
        build_name: str,
        stats: Stats,
        device_type: str,
        runtime: str,
        iterations: int,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        tensor_type=np.array,
        runtimes_supported: Optional[List[str]] = None,
        runtime_version: str = str(torch.__version__),
    ):
        # Torch Dynamo is pretty verbose with its warnings,
        # so we set the logging level to ERROR
        torch._logging.set_logs(dynamo=logging.ERROR)

        self.throughput_ips = None
        self.mean_latency_ms = None

        # Allow children of this class to pass different values than
        # the defaults for torch-eager and torch-compiled
        if runtimes_supported:
            init_runtimes_supported = runtimes_supported
        else:
            init_runtimes_supported = ["torch-eager", "torch-compiled"]

        super().__init__(
            cache_dir=cache_dir,
            build_name=build_name,
            stats=stats,
            device_type=device_type,
            runtime=runtime,
            iterations=iterations,
            runtimes_supported=init_runtimes_supported,
            runtime_version=runtime_version,
            base_path=os.path.dirname(__file__),
            tensor_type=tensor_type,
            model=model,
            inputs=inputs,
        )

    def _compile(self) -> None:
        self.model.eval()

        if self.runtime == "torch-compiled":
            raise ValueError("gotcha!")
            # First ensure we have the required version of Pytorch
            clean_torch_version = self.runtime_version.split("+")[0]
            if version.parse(clean_torch_version) < version.parse("2.0.0"):
                raise exp.BenchmarkException(
                    (
                        f"{self.runtime} can only be used with Pytorch 2.0.0 or above. "
                        f"However, version {self.runtime_version} was found."
                    )
                )

            self.model = torch.compile(self.model)

    def _setup(self) -> None:
        # Ensure we have the correct model type
        model_type = ignition.identify_model_type(self.model)
        if model_type != build.ModelType.PYTORCH:
            raise exp.IntakeError(
                f"Only Pytorch models are valid when runtime is {self.runtime}"
            )

        start_time = time.perf_counter()
        with build.Logger("Preparing torch model", self.logfile_path):
            self._compile()
        end_time = time.perf_counter()
        total_time = end_time - start_time

        self.stats.save_model_eval_stat("torch_compilation_seconds", total_time)

    def _calculate_performance(
        self, per_iteration_latency: List[float]
    ) -> MeasuredPerformance:
        """
        Calculate performance statistics from per_iteration_latency
        """

        self.mean_latency_ms = mean(per_iteration_latency) * 1000
        self.throughput_ips = float(
            1 / (np.sum(per_iteration_latency) / len(per_iteration_latency))
        )

        return MeasuredPerformance(
            mean_latency=self.mean_latency,
            throughput=self.throughput,
            device=self.device_name,
            device_type=self.device_type,
            runtime=self.runtime,
            runtime_version=self.runtime_version,
            build_name=self.build_name,
        )

    def _run_model(self, iterations: int, time_limit: int) -> List[float]:
        """
        Run the model repeatedly, collecting the performance of each
        iteration. Stop running when the iterations target or time limit
        is reached, whichever comes first.
        """

        counter = 0
        total_time = 0
        per_iteration_latency = []

        while counter < iterations and total_time < time_limit:
            start_time = time.perf_counter()
            self.model(**self.inputs)
            end_time = time.perf_counter()
            total_time = total_time + end_time - start_time
            counter = counter + 1
            per_iteration_latency.append(end_time - start_time)

        return per_iteration_latency

    def _execute(self) -> MeasuredPerformance:
        """
        The logic for executing a torch model to collect performance data
        """

        # Cache warmup for 1 minute or 10 iterations, whichever
        # comes first
        self._run_model(iterations=10, time_limit=60)

        # Run the benchmark for the specified amount of iterations,
        # or 2 minutes, whichever comes first
        per_iteration_latency = self._run_model(
            iterations=self.iterations, time_limit=120
        )

        # Record the number of iterations actually used for the benchmark,
        # which will be less than the `iterations` argument if the time
        # limit was reached
        self.stats.save_model_eval_stat(fs.Keys.ITERATIONS, len(per_iteration_latency))

        return self._calculate_performance(per_iteration_latency)

    def benchmark(self) -> MeasuredPerformance:
        """
        Wrapper function for self._execute()

        The reason this wrapper exists is to allow developers to apply various
        settings to execution on a per-runtime basis. For example, selectively
        enabling torch.no_grad().
        """
        with torch.no_grad():
            return self._execute()

    @property
    def mean_latency(self) -> float:
        if self.mean_latency_ms is not None:
            return self.mean_latency_ms
        else:
            raise exp.BenchmarkException(
                "Queried mean latency before self.benchmark() was called"
            )

    @property
    def throughput(self) -> float:
        if self.throughput_ips is not None:
            return self.throughput_ips
        else:
            raise exp.BenchmarkException(
                "Queried throughput before self.benchmark() was called"
            )

    @property
    def device_name(self) -> str:
        return get_cpu_specs()["CPU Name"]
