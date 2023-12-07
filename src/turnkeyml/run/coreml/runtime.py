import platform
import os
import shutil
import numpy as np
from turnkeyml.run.basert import BaseRT
import turnkeyml.common.exceptions as exp
from turnkeyml.run.coreml.execute import COREML_VERSION
from turnkeyml.common.filesystem import Stats, rebase_cache_dir
import turnkeyml.common.build as build
from turnkeyml.common.performance import MeasuredPerformance
from turnkeyml.run.coreml.execute import create_conda_env, execute_benchmark
import turnkeyml.run.plugin_helpers as plugin_helpers


class CoreML(BaseRT):
    def __init__(
        self,
        cache_dir: str,
        build_name: str,
        stats: Stats,
        iterations: int,
        device_type: str,
        runtime: str = "coreml",
        tensor_type=np.array,
        model=None,
        inputs=None,
    ):
        super().__init__(
            cache_dir=cache_dir,
            build_name=build_name,
            stats=stats,
            tensor_type=tensor_type,
            device_type=device_type,
            iterations=iterations,
            runtime=runtime,
            runtimes_supported=["coreml"],
            runtime_version=COREML_VERSION,
            base_path=os.path.dirname(__file__),
            model=model,
            inputs=inputs,
            requires_docker=False,
            model_filename="model.mlmodel",
            model_dirname="mlmodel",
        )

    def _setup(self):
        # Check OS
        if platform.system() != "Darwin":
            msg = "Only MacOS is supported for CoreML Runtime"
            raise exp.ModelRuntimeError(msg)

        # Check silicon
        if "Apple M" not in self.device_name:
            msg = f"You need an 'Apple M*' processor to run using apple_silicon, got '{self.device_name}'"
            raise exp.ModelRuntimeError(msg)

        self._transfer_files([self.conda_script])

    def benchmark(self) -> MeasuredPerformance:
        """
        Transfer input artifacts, execute model on hardware, analyze output artifacts,
        and return the performance.
        """

        # Remove previous benchmarking artifacts
        if os.path.exists(self.local_outputs_file):
            os.remove(self.local_outputs_file)

        # Transfer input artifacts
        state = build.load_state(self.cache_dir, self.build_name)

        # Just in case the model file was generated on a different machine:
        # strip the state's cache dir, then prepend the current cache dir
        model_file = rebase_cache_dir(
            state.results[0], state.config.build_name, self.cache_dir
        )

        if not os.path.exists(model_file):
            msg = "Model file not found"
            raise exp.ModelRuntimeError(msg)

        os.makedirs(self.local_output_dir, exist_ok=True)
        os.makedirs(self.local_model_dir, exist_ok=True)
        shutil.copy(model_file, self.local_model_file)

        # Execute benchmarking in hardware
        self._execute(
            output_dir=self.local_output_dir,
            coreml_file_path=self.local_model_file,
            outputs_file=self.local_outputs_file,
        )

        if not os.path.isfile(self.local_outputs_file):
            raise exp.BenchmarkException(
                "No benchmarking outputs file found after benchmarking run. "
                "Sorry we don't have more information."
            )

        # Call property methods to analyze the output artifacts for performance stats
        # and return them
        return MeasuredPerformance(
            mean_latency=self.mean_latency,
            throughput=self.throughput,
            device=self.device_name,
            device_type=self.device_type,
            runtime=self.runtime,
            runtime_version=self.runtime_version,
            build_name=self.build_name,
        )

    def _execute(
        self,
        output_dir: str,
        coreml_file_path: str,
        outputs_file: str,
    ):
        conda_env_name = "turnkey-coreml-ep"

        try:
            # Create and setup the conda env
            create_conda_env(conda_env_name)
        except Exception as e:
            raise plugin_helpers.CondaError(
                f"Conda env setup failed with exception: {e}"
            )

        # Execute the benchmark script in the conda environment
        execute_benchmark(
            coreml_file_path=coreml_file_path,
            outputs_file=outputs_file,
            output_dir=output_dir,
            conda_env_name=conda_env_name,
            iterations=self.iterations,
        )

    @property
    def mean_latency(self):
        return float(self._get_stat("Mean Latency(ms)"))

    @property
    def throughput(self):
        return float(self._get_stat("Throughput"))

    @property
    def device_name(self):
        return self._get_stat("CPU Name")
