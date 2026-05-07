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

    elif type == 'sequential':  # hold out later moduli
        cyc_length = p - 1 if op == '/' else p
        cyc_train = int(train_fraction * cyc_length)
        train_inds = np.array([
            [a * cyc_length + i for i in range(cyc_train)]
            for a in range(p)
        ]).flatten()
        test_inds = np.array([
            [a * cyc_length + i for i in range(cyc_train, cyc_length)]
            for a in range(p)
        ]).flatten()
        return np.random.permutation(train_inds), np.random.permutation(test_inds)

    elif type == 'alternating':  # alternate holding out moduli between train and test
        cyc_length = p - 1 if op == '/' else p
        cyc_train = list(range(0, cyc_length, 2))
        cyc_test = list(range(1, cyc_length, 2))
        train_inds = np.array([
            [a * cyc_length + i for i in cyc_train]
            for a in range(p)
        ]).flatten()
        test_inds = np.array([
            [a * cyc_length + i for i in cyc_test]
            for a in range(p)
        ]).flatten()
        return np.random.permutation(train_inds), np.random.permutation(test_inds)

    else:
        raise ValueError(f"Invalid type: {type}")


def grokking_data_torch(
    p: int,
    op: str = '/',
    split_type: str = 'random',
    train_fraction: float = 0.5,
    device='cpu',
):
    """Generate modular arithmetic dataset, returning torch tensors."""
    operations = {
        '*': lambda a, b: (a * b) % p,
        '/': lambda a, b: (a * pow(int(b), p - 2, p)) % p,
        '+': lambda a, b: (a + b) % p,
        '-': lambda a, b: (a - b) % p,
    }
    if op not in operations:
        raise ValueError("Unsupported operation, choose from ['*', '/', '+', '-']")

    X = np.array([(a, b) for a in range(p) for b in range(1 if op == '/' else 0, p)])
    T = np.array([operations[op](a, b) for a, b in X])

    embed = {'*': p, '/': p, '+': p, '-': p, '=': p + 1}
    X = np.array([[a, embed[op], b, embed['=']] for (a, b) in X])

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
