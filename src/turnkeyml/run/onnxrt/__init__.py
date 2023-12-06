import turnkeyml.build.sequences as sequences
from .runtime import OnnxRT

implements = {
    "runtimes": {
        "ort": {
            "build_required": True,
            "RuntimeClass": OnnxRT,
            "supported_devices": {"x86","apple_silicon"},
            "default_sequence": sequences.optimize_fp32,
        }
    }
}
