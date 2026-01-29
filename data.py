# ------------ Author Attribution ------------
# This file is adapted from the original code by Amund Tveit, available at
#       https://github.com/atveit/torch_grokking (MIT License)
# which itself is a PyTorch port of the original MLX code by Jason Stock, available at
#       https://github.com/stockeh/mlx-grokking
# We have modified this file to add different split types and random data generation.
# ---------------------------------------------


import numpy as np
import torch

def split_indices(n: int, train_fraction: float, p: int, op: str, type: str = 'random'):
    """
    Split indices into train and test sets.
    """

    if type == 'random':
        n_train = int(train_fraction * n)
        inds = np.random.permutation(n)
        return inds[:n_train], inds[n_train:]
    
    elif type == 'sequential':  # hold out later moduli
        if op == '/':
            cyc_length = p - 1
        else:
            cyc_length = p
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
        if op == '/':
            cyc_length = p - 1
        else:
            cyc_length = p
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

def grokking_data_torch(p: int, op: str = '/', split_type: str = 'random', train_fraction: float = 0.5, device='cpu'):
    """
    Same logic as grokking_data, but returns PyTorch tensors.
    """
    operations = {
        '*': lambda a, b: (a * b) % p,
        '/': lambda a, b: (a * pow(int(b), p-2, p)) % p,
        '+': lambda a, b: (a + b) % p,
        '-': lambda a, b: (a - b) % p
    }

    if op not in operations:
        raise ValueError(
            "Unsupported operation, choose from ['*', '/', '+', '-']")

    X = np.array([(a, b) for a in range(p)
                 for b in range(1 if op == '/' else 0, p)])
    T = np.array([operations[op](a, b) for a, b in X])

    embed = {'*': p, '/': p, '+': p, '-': p, '=': p + 1}
    X = np.array([[a, embed[op], b, embed['=']]
                  for (a, b) in X])

    train_inds, test_inds = split_indices(len(X), train_fraction, p, op, split_type)
    Xtrain, Ttrain = X[train_inds], T[train_inds]
    Xtest, Ttest = X[test_inds], T[test_inds]

    # Convert to torch
    Xtrain_torch = torch.tensor(Xtrain, dtype=torch.long, device=device)
    Ttrain_torch = torch.tensor(Ttrain, dtype=torch.long, device=device)
    Xtest_torch = torch.tensor(Xtest, dtype=torch.long, device=device)
    Ttest_torch = torch.tensor(Ttest, dtype=torch.long, device=device)

    return Xtrain_torch, Ttrain_torch, Xtest_torch, Ttest_torch


# --- Random target data for memorisation capacity experiments ---
def random_target_data_torch(n_samples: int, p: int, seq_len: int = 4, device='cpu'):
    """
    Generate random input-output pairs for measuring model memorisation capacity.
    
    The targets are uniformly random over the full vocabulary [0, p+2), which means
    a model must memorize the training data rather than learn any pattern.
    
    Args:
        n_samples: Number of training examples
        p: Prime number (determines vocabulary size: p tokens for digits + 2 for op/=)
        seq_len: Sequence length (default 4 for [a, op, b, =] format)
        device: PyTorch device
    
    Returns:
        X: Input sequences of shape (n_samples, seq_len)
        T: Random target labels uniformly distributed over [0, p+2)
    """
    n_tokens = p + 2  # p digits + operator + equals
    
    # Generate random input sequences (each token from 0 to n_tokens-1)
    X = np.random.randint(0, n_tokens, size=(n_samples, seq_len))
    
    # Generate random targets uniformly over [0, n_tokens) - full vocabulary
    T = np.random.randint(0, n_tokens, size=n_samples)
    
    X_torch = torch.tensor(X, dtype=torch.long, device=device)
    T_torch = torch.tensor(T, dtype=torch.long, device=device)
    
    return X_torch, T_torch


# Optional: quick equivalence check
if __name__ == '__main__':
    X_t, T_t, Xtest_t, Ttest_t = grokking_data_torch(11, op='/', train_fraction=0.5)
    print("Torch shapes:", X_t.shape, T_t.shape, Xtest_t.shape, Ttest_t.shape)

    # Check close in shape & values
    # Note: order might differ if random perm changed seeds
    # This is purely a demonstration
    print("Sample Torch data:", X_t[0], T_t[0])
    
    # Test random target data
    X_rand, T_rand = random_target_data_torch(100, 11)
    print("\nRandom target data shapes:", X_rand.shape, T_rand.shape)
    print("Sample random data:", X_rand[0], T_rand[0])
