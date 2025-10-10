# scripts/train_yolo.py
import torch
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

from penalty_vision.utils import Config, logger


def train_yolo(config: Config, run_name: str = None):
    data_path = Path(config.paths.data_yaml)
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_name or f"train_{timestamp}"

    model = YOLO(config.model.weights)

    train_params = {
        'data': str(data_path),
        'epochs': config.training.epochs,
        'imgsz': config.model.imgsz,
        'batch': config.training.batch,
        'project': config.paths.runs_dir,
        'name': run_name,
        'device': config.training.device if hasattr(config.training, 'device') else 'cuda',
        'patience': config.training.patience,
        'save_period': config.training.save_period,
        'save': True,
        'plots': True,
        'verbose': True,
        **config.to_dict()['augmentation']
    }

    model.train(**train_params)

    run_dir = Path(config.paths.runs_dir) / run_name
    best_model_path = run_dir / "weights" / "best.pt"

    logger.info(f"Training complete: {best_model_path}")
    
    return best_model_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    config = Config.from_yaml(args.config)
    
    if args.device:
        config.training.device = args.device
    elif torch.cuda.is_available():
        config.training.device = "cuda"
    elif torch.backends.mps.is_available():
        config.training.device = "mps"
    else:
        config.training.device = "cpu"

    print(f"Device: {config.training.device}")
    
    best_model = train_yolo(config, run_name=args.name)
    print(f"Model: {best_model}")