import math
from typing import Tuple


def compute_dataset_size_bits(
    p: int, op: str = '/', training_fraction: float = 0.5
) -> Tuple[int, float]:
    """Return (n_samples, dataset_bits) for a modular-arithmetic split.

    n = p * (p - 1) when op == '/' (no division by zero), else p * p.
    bits = n * log2(p + 2) over the full vocabulary.
    """
    n = p * (p - 1) if op == '/' else p * p
    n *= training_fraction
    size = n * math.log2(p + 2)
    return int(n), size
