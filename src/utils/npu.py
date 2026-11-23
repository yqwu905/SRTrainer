import torch

def setup_npu() -> bool:
    is_npu = hasattr(torch, "npu") and torch.npu.is_available()
    if is_npu:
        print("Ascend NPU Environment detected.")
        import torch_npu
        from torch_npu.contrib import transfer_to_npu
    return is_npu
