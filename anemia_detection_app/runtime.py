"""TensorFlow runtime selection and accelerator diagnostics."""

from __future__ import annotations

from typing import Any

import tensorflow as tf


def configure_accelerator() -> dict[str, Any]:
    """Enable safe GPU memory allocation when TensorFlow exposes a GPU.

    This is intentionally a no-op on CPU-only systems.  It lets the identical
    training code use CUDA automatically when the application runs in a GPU
    capable TensorFlow environment (for example, WSL2 on Windows).
    """

    gpus = tf.config.list_physical_devices("GPU")
    devices: list[str] = []
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            # A device may already be initialized by another TensorFlow call.
            pass
        details = tf.config.experimental.get_device_details(gpu)
        devices.append(str(details.get("device_name") or gpu.name))
    return {
        "available": bool(gpus),
        "device_count": len(gpus),
        "devices": devices,
        "tensorflow_version": tf.__version__,
    }
