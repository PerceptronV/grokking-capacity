import os

import torch

from ..models import TransformerTorch


def save_model(model, metadata: dict, save_path: str) -> None:
    """Save model weights and metadata to a .pt file."""
    save_dict = {'model_state_dict': model.state_dict(), 'metadata': metadata}
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(save_dict, save_path)


def load_model(model_path: str, device: str = 'cpu'):
    """Reconstruct a TransformerTorch from a saved checkpoint.

    The checkpoint must contain a 'metadata' dict with depth/dim/heads/n_tokens/
    seq_len/dropout. Returns (model, metadata).
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    metadata = checkpoint['metadata']

    model = TransformerTorch(
        depth=metadata['depth'],
        dim=metadata['dim'],
        heads=metadata['heads'],
        n_tokens=metadata['n_tokens'],
        seq_len=metadata['seq_len'],
        dropout=metadata['dropout'],
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, metadata
