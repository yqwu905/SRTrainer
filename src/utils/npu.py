import torch


def setup_npu() -> bool:
    is_npu = hasattr(torch, "npu") and torch.npu.is_available()
    if is_npu:
        print("Ascend NPU Environment detected.")
        from torch_npu.contrib import transfer_to_npu  # noqa: F401 - registers CUDA-to-NPU redirects
    return is_npu


def max_memory_allocated_bytes(device: torch.device | str | None) -> int:
    """Return the peak allocated device memory (bytes) for ``device``.

    Mirrors ``torch.cuda.max_memory_allocated`` across backends: CUDA, Ascend NPU
    (``torch_npu``) and XPU expose the same allocator-statistics API under their
    own module. Unsupported devices (e.g. CPU/MPS) return 0 so callers can log
    ``0.0`` instead of crashing.
    """
    try:
        dev = torch.device(device) if device is not None else torch.device("cuda")
    except (TypeError, RuntimeError):
        return 0

    device_type = dev.type
    index = dev.index if dev.index is not None else 0

    if device_type == "cuda":
        module = getattr(torch, "cuda", None)
        available = bool(module is not None and module.is_available())
    elif device_type == "npu":
        module = getattr(torch, "npu", None)
        available = bool(module is not None and module.is_available())
    elif device_type == "xpu":
        module = getattr(torch, "xpu", None)
        available = bool(module is not None and module.is_available())
    else:
        return 0

    query = getattr(module, "max_memory_allocated", None)
    if query is None or not available:
        return 0

    # torch_npu/XPU accept the device index as a positional argument; CUDA also
    # accepts a torch.device. The index form works uniformly across all three.
    try:
        return int(query(index))
    except (TypeError, RuntimeError, AssertionError):
        try:
            return int(query(dev))
        except Exception:
            return 0
