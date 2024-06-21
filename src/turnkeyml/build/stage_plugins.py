import turnkeyml.build.export as export
import turnkeyml.common.plugins as plugins
import turnkeyml.common.management_tools as mgmt
import turnkeyml.run.benchmark_build as bench

# Plugin interface for sequences
discovered_plugins = plugins.discover()

# Populated supported sequences dict with builtin sequences
SUPPORTED_STAGES = [
    mgmt.Version,
    mgmt.Cache,
    mgmt.ModelsLocation,
    mgmt.Report,
    bench.BenchmarkBuild,
    export.ExportPytorchModel,
    export.OptimizeOnnxModel,
    export.OnnxLoad,
    export.ConvertOnnxToFp16,
]

# Add sequences from plugins to supported sequences dict
for module in discovered_plugins.values():
    if "stages" in module.implements.keys():
        for stage_class in module.implements["stages"]:
            if stage_class in SUPPORTED_STAGES:
                name = stage_class.__class__.unique_name
                raise ValueError(
                    f"Your turnkeyml installation has two stages named '{name}' "
                    "installed. You must uninstall one of your plugins that includes "
                    f"{name}. Your imported sequence plugins are: {SUPPORTED_STAGES}\n"
                    f"This error was thrown while trying to import {module}"
                )

            SUPPORTED_STAGES.append(stage_class)