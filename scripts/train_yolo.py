import argparse
import os
import random
import torch
import wandb

import numpy as np

from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from ultralytics import YOLO

from penalty_vision.utils import Config, logger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to: {seed}")

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

    set_seed(config.training.seed)
    results = model.train(**train_params)
    best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    
    if not best_model_path.exists():
        logger.warning(f"Best model not found at {best_model_path}")
        return None

    logger.info(f"Training complete: {best_model_path}")
    
    return best_model_path


if __name__ == "__main__":
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
    
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        entity = os.getenv("WANDB_ENTITY", "wandb-user")
        project = os.getenv("WANDB_PROJECT", "penaltykickvisual-yolotraining")
        
        try:
            wandb.login(key=wandb_api_key)
            wandb.init(
                project=project,
                entity=entity,
                name=args.name,
                config=config.to_dict()
            )
            logger.info(f"WandB configured: {entity}/{project}")
        except Exception as e:
            logger.warning(f"WandB login failed: {e}. Continuing without WandB")
    else:
        logger.info("WandB API key not found, running without WandB")
        
    logger.info(f"Device: {config.training.device}")
    
    best_model = train_yolo(config, run_name=args.name)
    logger.info(f"Model: {best_model}")