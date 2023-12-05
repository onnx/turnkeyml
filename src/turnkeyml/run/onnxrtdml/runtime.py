import platform
import os
import subprocess
import numpy as np
from turnkeyml.run.basert import BaseRT
import xml.etree.ElementTree as ET
import turnkeyml.common.exceptions as exp
from turnkeyml.run.onnxrtdml.execute import ORT_VERSION
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
            runtime_version=ORT_VERSION,
            base_path=os.path.dirname(__file__),
            model=model,
            inputs=inputs,
            requires_docker=False,
        )

    def _setup(self):
        # Check if DirectX12 is supported
        dxdiag_xml_file = 'dxdiag_output.xml'
        try:
            # Run dxdiag and output to an XML file
            subprocess.run(['dxdiag', '/x',  '/whql:off', dxdiag_xml_file])

            # Parse the XML file
            tree = ET.parse(dxdiag_xml_file)
            root = tree.getroot()

            # Find DirectX version
            dx_version = root.find(".//DirectXVersion")
            if dx_version is  None or 'DirectX 12' not in dx_version.text:
                msg = (
                    f"System under test does not support Directx 12, {rt_name} "
                    "needs Directx 12 for execution"
                )
                raise exp.ModelRuntimeError(msg)
        except Exception as e:
            msg = (
                    f"dxdiag command to verify Directx 12 support failed"
                )
            raise exp.ModelRuntimeError(msg)
        finally:
            # Delete the XML file
            if os.path.exists(dxdiag_xml_file):
                os.remove(dxdiag_xml_file)

        self._transfer_files([self.conda_script])

    def _execute(
        self,
        output_dir: str,
        onnx_file: str,
        outputs_file: str,
    ):
        conda_env_name = "turnkey-onnxruntime-dml-ep"

        try:
            # Create and setup the conda env
            create_conda_env(conda_env_name)
        except PermissionError as pe:
            os_type = platform.system()
            if os_type == "Windows":
                raise plugin_helpers.CondaError(
                    f"Conda environment setup encountered a permission issue: {pe}. "
                    "Ensure you have write permissions for the Conda installation directory. "
                    "If Conda is installed for 'All Users' on a Windows machine, it defaults "
                    "to 'C:\\ProgramData', where you may not have the necessary permissions."
                    "To resolve this, consider reinstalling Conda for 'Just Me' instead."
                )
            else:
                raise plugin_helpers.CondaError(
                    f"Conda env setup failed due to permission error: {pe}"
                )
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
