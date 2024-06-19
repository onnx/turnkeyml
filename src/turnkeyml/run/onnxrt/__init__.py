from .runtime import OnnxRT

implements = {
    "runtimes": {
        "ort": {
            "build_required": True,
            "RuntimeClass": OnnxRT,
            "supported_devices": {"x86"},
        }
    }
}
