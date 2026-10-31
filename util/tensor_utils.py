import torch


def as_index(indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    """保留已对齐的索引；需要转换时一次完成设备和 long 类型转换。"""
    if indices.device == device and indices.dtype == torch.long:
        return indices
    return indices.to(device=device, dtype=torch.long, non_blocking=True)
