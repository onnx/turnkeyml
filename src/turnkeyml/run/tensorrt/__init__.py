from .runtime import TensorRT


implements = {
    "runtimes": {
        "trt": {
            "build_required": True,
            "RuntimeClass": TensorRT,
            "supported_devices": {"nvidia"},
        }
    }
}
