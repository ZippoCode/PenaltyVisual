import os
from pathlib import Path

import torch

from penalty_vision.utils import logger


def save_checkpoint(model, optimizer, epoch, metrics, checkpoint_dir='checkpoints', filename='best_model.pth'):
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }

    filepath = checkpoint_path / filename
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, checkpoint_path, device='cpu'):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']

    logger.info(f"Checkpoint loaded from epoch {epoch}")
    logger.info(f"Metrics: {metrics}")

    return epoch, metrics
