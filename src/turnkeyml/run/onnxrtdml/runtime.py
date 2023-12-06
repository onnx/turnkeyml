import platform
import os
import subprocess
import numpy as np
from turnkeyml.run.basert import BaseRT
import turnkeyml.common.exceptions as exp
import turnkeyml.common.build as build
from turnkeyml.run.onnxrtdml.execute import ORT_DML_VERSION
from turnkeyml.common.filesystem import Stats
from turnkeyml.run.onnxrtdml.execute import create_conda_env, execute_benchmark
import turnkeyml.run.plugin_helpers as plugin_helpers

rt_name = "ortdml"
class OnnxRTDML(BaseRT):
    def __init__(
        self,
        cache_dir: str,
        build_name: str,
        stats: Stats,
        iterations: int,
        device_type: str,
        runtime: str = "ortdml",
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
            runtimes_supported=[rt_name],
            runtime_version=ORT_DML_VERSION,
            base_path=os.path.dirname(__file__),
            model=model,
            inputs=inputs,
            requires_docker=False,
        )

    def _setup(self):

        # Check if DirectX12 is supported
        enumerate_dx12_devices = """
import torch_directml
if not torch_directml.is_available() or torch_directml.device_count() == 0:
    print('no_devices')
else:
    for i in range(torch_directml.device_count()):
        print(f'device_id: {i}, device_name: {torch_directml.device_name(i)}')
"""
        conda_env_name = "turnkey-torch-dml-helper"
        dml_helper_requirements = ["torch_directml"]
        try:
            # Create and setup the conda env
            create_conda_env(conda_env_name, dml_helper_requirements)
        except Exception as e:
            raise plugin_helpers.CondaError(
                f"Conda env setup failed with exception: {e}"
            )
        python_in_env = plugin_helpers.get_python_path(conda_env_name)
        
        cmd = [
            python_in_env,
            '-c', 
            enumerate_dx12_devices.strip()
        ]

        try:
            output = subprocess.check_output(cmd, text=True)
            if output.strip() == 'no_devices':
                raise RuntimeError('DirectML is not available or no Directx 12 supported devices found')
        except Exception as e:
            raise plugin_helpers.CondaError(
                f"Checking for DirectX12 devices failed: {e}"
            )

        dx12_devices = []
        for line in output.splitlines():
            if line.startswith('device_id'):
                parts = line.split(',')
                device_id = parts[0].split(':')[1].strip()
                device_name = parts[1].split(':')[1].strip()
                dx12_devices.append({'device_id': device_id, 'device_name': device_name})

        self._transfer_files([self.conda_script])

    def _execute(
        self,
        output_dir: str,
        onnx_file: str,
        outputs_file: str,
    ):
        conda_env_name = "turnkey-onnxruntime-dml-ep"
        dml_requirements = [f"onnxruntime-directml=={ORT_DML_VERSION}"]

        try:
            # Create and setup the conda env
            create_conda_env(conda_env_name, dml_requirements)
        except Exception as e:
            raise plugin_helpers.CondaError(
                f"Conda env setup failed with exception: {e}"
            )

        # Execute the benchmark script in the conda environment
        execute_benchmark(
            onnx_file=onnx_file,
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
