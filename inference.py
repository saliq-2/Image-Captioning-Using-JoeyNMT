import torch

from models import create_model


def load_trained_model(checkpoint_path, device):
    # PyTorch 2.6+ defaults to weights_only=True; we need full checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab = checkpoint['vocab']
    model = create_model(vocab, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {checkpoint_path}")
    print(f"Training completed at epoch: {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    return model, vocab, checkpoint.get('history', {})


