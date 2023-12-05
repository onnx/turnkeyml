import turnkeyml.build.sequences as sequences
from .runtime import OnnxRTDML

implements = {
    "runtimes": {
        "ortdml": {
            "build_required": True,
            "RuntimeClass": OnnxRTDML,
            "supported_devices": {"nvidia", "amd_igpu", "intel_igpu", "qualcomm"},
            "default_sequence": sequences.optimize_fp16,
        }
    }
}
