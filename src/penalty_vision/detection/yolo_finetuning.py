from datetime import datetime
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

from penalty_vision.utils import logger


def train_yolo(data_yaml: str, runs_dir: str, base_model: str = "yolo11n.pt", epochs: int = 50, imgsz: int = 640,
               batch: int = 16, name: str = "penalty_detector", device: str = "0", patience: int = 10,
               save_period: int = 5, **kwargs):
    data_path = Path(data_yaml)
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    with open(data_path, 'r') as f:
        data_config = yaml.safe_load(f)

    logger.info("\n" + "=" * 60)
    logger.info("YOLO FINE-TUNING CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Base model: {base_model}")
    logger.info(f"Dataset: {data_yaml}")
    logger.info(f"Classes: {data_config['names']}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Image size: {imgsz}")
    logger.info(f"Batch size: {batch}")
    logger.info(f"Device: {device}")
    logger.info(f"Output: {runs_dir}/{name}")
    logger.info("=" * 60)

    logger.info(f"\nLoading base model: {base_model}...")
    model = YOLO(base_model)

    logger.info("\nStarting training...")
    logger.info("This may take a while. Monitor progress in terminal.\n")

    model.train(data=str(data_path), epochs=epochs, imgsz=imgsz, batch=batch, project=runs_dir, name=name,
                device=device, patience=patience, save=True, save_period=save_period, plots=True,
                verbose=True, **kwargs)

    run_dir = Path(runs_dir) / name
    best_model_path = run_dir / "weights" / "best.pt"
    last_model_path = run_dir / "weights" / "last.pt"

    logger.info("\n" + "=" * 60)
    logger.info("✓ TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best model: {best_model_path}")
    logger.info(f"Last model: {last_model_path}")
    logger.info(f"Results: {run_dir}")
    logger.info(f"Plots: {run_dir}/*.png")
    logger.info("=" * 60)

    return best_model_path


def training(data_yaml: str, runs_dir: str, device: str = "0", num_epochs: int = 50):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return train_yolo(data_yaml=data_yaml, runs_dir=runs_dir, base_model="yolo11n.pt", epochs=num_epochs, imgsz=640,
                      batch=16, name=f"full_train_{timestamp}", device=device, patience=15, save_period=10, hsv_h=0.015,
                      hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, flipud=0.0, fliplr=0.5, mosaic=1.0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune YOLO on soccer dataset")
    parser.add_argument("--runs_dir", type=str, required=True, help="The directory to save results"),
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml"),
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs (for custom mode)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="Device (0 for GPU, cpu or mps for Apple Silicon)")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("YOLO FINE-TUNING")
    print("=" * 60)

    if args.device != "cpu":
        if torch.cuda.is_available():
            print(f"NVIDIA GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif torch.backends.mps.is_available():
            print("Apple Silicon GPU (MPS) detected")
            args.device = "mps"
        else:
            print("⚠No GPU detected, falling back to CPU")
            args.device = "cpu"

    print(f"Dataset: {args.data}")
    print(f"Device: {args.device}")
    print("=" * 60)

    print("\nFull training mode (100 epochs)")
    best_model = training(args.data, runs_dir=args.runs_dir, device=args.device, num_epochs=args.epochs)
    print(f"\n✓ Training complete!")
    print(f"Best model saved at: {best_model}")
