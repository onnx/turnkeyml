from typing import Optional
from typing import List, Dict, Tuple
import turnkeyml.run.onnxrt as onnxrt
import turnkeyml.run.tensorrt as tensorrt
import turnkeyml.run.torchrt as torchrt
import turnkeyml.common.plugins as plugins
from turnkeyml.sequence import Sequence
import turnkeyml.common.exceptions as exp


def supported_devices_list(data: Dict, parent_key: str = "") -> List:
    """Recursive function to generate all device::part::config pairs"""
    result = []
    for key, value in data.items():
        current_key = f"{parent_key}::{key}" if parent_key else key
        result.append(current_key)
        if isinstance(value, dict):
            result.extend(supported_devices_list(value, current_key))
        elif isinstance(value, list):
            for item in value:
                result.append(f"{current_key}::{item}")
    return result


discovered_plugins = plugins.discover()

DEFAULT_RUNTIME = 0

# Note: order matters here. We append the discovered_plugins after builtin so
# that the default runtime for each device will come from a builtin, whenever
# available.
builtin_runtimes = [onnxrt, tensorrt, torchrt]
plugin_modules = builtin_runtimes + list(discovered_plugins.values())

SUPPORTED_RUNTIMES = {}

for module in plugin_modules:
    if "runtimes" in module.implements.keys():
        for runtime_name, runtime_info in module.implements["runtimes"].items():
            if runtime_name in SUPPORTED_RUNTIMES:
                raise ValueError(
                    f"Your turnkeyml installation has two runtimes named '{runtime_name}' "
                    "installed. You must uninstall one of your plugins that includes "
                    f"{runtime_name}. Your imported runtime plugins are: {SUPPORTED_RUNTIMES}\n"
                    f"This error was thrown while trying to import {module}"
                )
            if isinstance(runtime_info["supported_devices"], set):
                runtime_info["supported_devices"] = {
                    item: {} for item in runtime_info["supported_devices"]
                }
            SUPPORTED_RUNTIMES[runtime_name] = runtime_info

# Get the list of supported devices by checking which devices each runtime supports
SUPPORTED_DEVICES = []
for runtime_info in SUPPORTED_RUNTIMES.values():
    SUPPORTED_DEVICES.extend(supported_devices_list(runtime_info["supported_devices"]))
SUPPORTED_DEVICES = list(set(SUPPORTED_DEVICES))

# Organizing the supported devices that are in the "long form".
# Those are not the elements used for setting the defaults.
# This is useful for nicely showing the list of supported devices as part of the help menu.
SUPPORTED_DEVICES.sort()

# Create a map of devices to the runtimes that support them
DEVICE_RUNTIME_MAP = {key: [] for key in SUPPORTED_DEVICES}
for device in SUPPORTED_DEVICES:
    for runtime_name, runtime_info in SUPPORTED_RUNTIMES.items():
        if device in supported_devices_list(runtime_info["supported_devices"]):
            DEVICE_RUNTIME_MAP[device].append(runtime_name)


def apply_default_runtime(device: str, runtime: Optional[str] = None):
    if runtime is None:
        return DEVICE_RUNTIME_MAP[str(device)][DEFAULT_RUNTIME]
    else:
        return runtime


def _check_suggestion(value: str):
    return (
        f"You may need to check the spelling of '{value}', install a "
        "plugin, or update the turnkeyml package."
    )


def select_runtime(device: str, runtime: Optional[str]) -> Tuple[str, str, Sequence]:
    # Convert to str in case its an instance of Device
    device_str = str(device)

    selected_runtime = apply_default_runtime(device_str, runtime)

    # Validate device and runtime selections
    if device_str not in SUPPORTED_DEVICES:
        raise exp.ArgError(
            f"Device argument '{device_str}' is not one of the available "
            f"supported devices {SUPPORTED_DEVICES}\n"
            f"{_check_suggestion(device_str)}"
        )
    if selected_runtime not in DEVICE_RUNTIME_MAP[device_str]:
        raise exp.ArgError(
            f"Runtime argument '{selected_runtime}' is not one of the available "
            f"runtimes supported for device '{device_str}': {DEVICE_RUNTIME_MAP[device_str]}\n"
            f"{_check_suggestion(selected_runtime)}"
        )

    # Get the plugin module for the selected runtime
    runtime_info = SUPPORTED_RUNTIMES[selected_runtime]

    return selected_runtime, runtime_info
