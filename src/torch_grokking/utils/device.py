import torch


def get_device(device: str | None = None, cpu: bool = False) -> str:
    """Select compute device."""
    if device is not None:
        return device
    if cpu:
        return 'cpu'
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def gpu_type(device: str | None = None) -> str:
    """Return a short label for the active accelerator (e.g. 'H100', 'A100', 'mps', 'cpu')."""
    dev = device or get_device()
    if dev.startswith('cuda'):
        try:
            idx = int(dev.split(':')[1]) if ':' in dev else 0
        except ValueError:
            idx = 0
        try:
            name = torch.cuda.get_device_name(idx)
        except Exception:
            return 'cuda'
        # 'NVIDIA H100 80GB HBM3' -> 'H100'
        for marker in ('H200', 'H100', 'A100', 'A10', 'L40', 'L4', 'V100', 'T4', 'RTX', '4090', '3090'):
            if marker in name:
                return marker
        return name
    if dev == 'mps':
        return 'mps'
    return 'cpu'
