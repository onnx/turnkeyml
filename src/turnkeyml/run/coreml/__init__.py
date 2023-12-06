import turnkeyml.build.sequences as sequences
from .runtime import CoreML

implements = {
    "runtimes": {
        "coreml": {
            "build_required": True,
            "RuntimeClass": CoreML,
            "supported_devices": {"apple_silicon"},
            "default_sequence": sequences.coreml,
        }
    }
}
