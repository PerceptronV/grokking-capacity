import os
import math
import torch
from models import TransformerTorch


def save_model(model, metadata, save_path):
    """
    Save model weights and metadata to a file.
    
    Args:
        model: PyTorch model to save
        metadata: Dictionary containing hyperparameters and training info
        save_path: Path to save the model
    """
    save_dict = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(save_dict, save_path)
    print(f"\nModel saved to: {save_path}")
    print(f"Metadata keys: {list(metadata.keys())}")


def load_model(model_path, device='cpu'):
    """
    Load a trained model from a saved file.
    
    This function recreates the model architecture from saved hyperparameters
    and loads the trained weights. It can be imported and used in other scripts.
    
    Args:
        model_path: Path to the saved model file
        device: Device to load the model on ('cpu', 'cuda', or 'mps')
    
    Returns:
        model: Loaded PyTorch model
        metadata: Dictionary containing all hyperparameters and training info
    
    Example:
        from single import load_model
        model, metadata = load_model('data/single/model.pt')
        print(f"Model dimension: {metadata['dim']}")
        print(f"Final validation accuracy: {metadata['val_acc'][-1]:.2f}%")
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    metadata = checkpoint['metadata']
    
    # Recreate model architecture from metadata
    model = TransformerTorch(
        depth=metadata['depth'],
        dim=metadata['dim'],
        heads=metadata['heads'],
        n_tokens=metadata['n_tokens'],
        seq_len=metadata['seq_len'],
        dropout=metadata['dropout']
    ).to(device)
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    print(f"Model loaded from: {model_path}")
    print(f"Architecture: depth={metadata['depth']}, dim={metadata['dim']}, heads={metadata['heads']}")
    print(f"Parameters: {metadata['param_count']:,}")
    print(f"Final validation accuracy: {metadata['val_acc'][-1]:.2f}%")
    
    return model, metadata


def compute_dataset_size_bits(p: int, op: str = '/', training_fraction: float = 0.5) -> float:
    """
    Compute dataset size in samples and bits.
    
    Args:
        p: Prime number
        op: Operation ('/' for division, others for full table)
        training_fraction: Fraction of data used for training
    
    Returns:
        Dataset size in samples and bits
    """
    n = p * (p - 1) if op == '/' else p * p
    n *= training_fraction
    size = n * math.log2(p + 2)
    return int(n), size
