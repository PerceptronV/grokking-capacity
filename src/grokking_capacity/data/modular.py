# Author Attribution
# Adapted from Amund Tveit, https://github.com/atveit/torch_grokking (MIT)
# itself a PyTorch port of Jason Stock's MLX code, https://github.com/stockeh/mlx-grokking
# Modified to add split types and random data generation.

import numpy as np
import torch


def split_indices(n: int, train_fraction: float, p: int, op: str, type: str = 'random'):
    """Split indices into train and test sets."""

    if type == 'random':
        n_train = int(train_fraction * n)
        inds = np.random.permutation(n)
        return inds[:n_train], inds[n_train:]

    cyc_length = p - 1 if op == '/' else p
    if type == 'sequential':  # hold out later right-hand operands
        boundary = int(train_fraction * cyc_length)
        train_columns, test_columns = range(boundary), range(boundary, cyc_length)
    elif type == 'alternating':  # fixed even/odd split; independent of train_fraction
        train_columns, test_columns = range(0, cyc_length, 2), range(1, cyc_length, 2)
    else:
        raise ValueError(f"Invalid type: {type}")

    # Both structured splits use the same row-major ordering. Keep training
    # permutation before test permutation to preserve the seeded protocol.
    partitions = [
        np.array([[a * cyc_length + i for i in columns] for a in range(p)]).flatten()
        for columns in (train_columns, test_columns)
    ]
    return tuple(np.random.permutation(indices) for indices in partitions)


def grokking_data_torch(
    p: int,
    op: str = '/',
    split_type: str = 'random',
    train_fraction: float = 0.5,
    device='cpu',
):
    """Generate modular arithmetic dataset, returning torch tensors."""
    if op not in {'*', '/', '+', '-'}:
        raise ValueError("Unsupported operation, choose from ['*', '/', '+', '-']")

    rhs = np.arange(1 if op == '/' else 0, p)
    a = np.repeat(np.arange(p), len(rhs))
    b = np.tile(rhs, p)
    if op == '/':
        # Each denominator's inverse is shared by all p left-hand operands.
        inverse = np.array([pow(int(value), p - 2, p) for value in rhs])
        T = (a * np.tile(inverse, p)) % p
    else:
        operation = {'*': np.multiply, '+': np.add, '-': np.subtract}[op]
        T = operation(a, b) % p
    X = np.column_stack((a, np.full_like(a, p), b, np.full_like(a, p + 1)))

    train_inds, test_inds = split_indices(len(X), train_fraction, p, op, split_type)
    Xtrain, Ttrain = X[train_inds], T[train_inds]
    Xtest, Ttest = X[test_inds], T[test_inds]

    return (
        torch.tensor(Xtrain, dtype=torch.long, device=device),
        torch.tensor(Ttrain, dtype=torch.long, device=device),
        torch.tensor(Xtest, dtype=torch.long, device=device),
        torch.tensor(Ttest, dtype=torch.long, device=device),
    )


def random_target_data_torch(n_samples: int, p: int, seq_len: int = 4, device='cpu'):
    """Generate random input-output pairs for measuring memorisation capacity.

    Targets are uniform over the full vocabulary [0, p+2), forcing pure
    memorisation rather than learned structure.
    """
    n_tokens = p + 2  # p digits + operator + equals
    X = np.random.randint(0, n_tokens, size=(n_samples, seq_len))
    T = np.random.randint(0, n_tokens, size=n_samples)
    return (
        torch.tensor(X, dtype=torch.long, device=device),
        torch.tensor(T, dtype=torch.long, device=device),
    )
