import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._generate_schema")

import argparse
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

from penalty_vision.config.training_config import get_training_config
from penalty_vision.dataset.dataloaders import create_dataloaders
from penalty_vision.models.losses import get_loss_function
from penalty_vision.models.metrics import MetricsCalculator
from penalty_vision.models.optimizer import get_optimizer, get_scheduler
from penalty_vision.models.two_stream_lstm import TwoStreamLSTM
from penalty_vision.training.trainer import Trainer
from penalty_vision.utils.logger import logger
from penalty_vision.utils.seed import set_seed
from penalty_vision.utils.wandb_logger import WandBLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, help='Path to config YAML file')
    args = parser.parse_args()

    config = get_training_config(args.config)
    device = torch.device(config.device)
    logger.info(f"Using device: {device}")

    set_seed(config.seed)

    train_loader, val_loader, test_loader, dataset_info = create_dataloaders(
        data_dir=str(config.data.data_dir),
        label_field=config.data.label_field,
        batch_size=config.data.batch_size,
        train_size=config.data.train_split,
        val_size=config.data.val_split,
        test_size=config.data.test_split,
        num_workers=config.data.num_workers
    )

    wandb_logger = WandBLogger(
        experiment_name=config.experiment_name,
        config={
            "learning_rate": config.training.learning_rate,
            "batch_size": config.data.batch_size,
            "num_epochs": config.training.num_epochs,
            "optimizer": config.training.optimizer,
            "scheduler": config.training.scheduler,
            "hidden_size": config.model.hidden_size,
            "num_layers": config.model.num_layers,
            "dropout": config.model.dropout,
            "gradient_clip_val": config.training.gradient_clip_val,
            "mixed_precision": config.training.mixed_precision,
            "early_stopping_patience": config.training.early_stopping_patience,
            "seed": config.seed,
            "dataset/total_samples": dataset_info['total_samples'],
            "dataset/train_samples": dataset_info['train_samples'],
            "dataset/val_samples": dataset_info['val_samples'],
            "dataset/test_samples": dataset_info['test_samples'],
            "dataset/num_classes": dataset_info['num_classes'],
            "dataset/label_field": config.data.label_field,
            "dataset/label_names": dataset_info['label_names'],
            "dataset/label_distribution": dataset_info['label_distribution']
        }
    )

    model = TwoStreamLSTM(
        input_size=config.model.input_size,
        hidden_size=config.model.hidden_size,
        num_classes=config.model.num_classes,
        dropout=config.model.dropout,
        metadata_size=config.model.metadata_size
    )
    model = model.to(device)

    wandb_logger.watch_model(model)

    train_labels = train_loader.dataset.labels
    criterion = get_loss_function(
        num_classes=config.model.num_classes,
        labels=train_labels,
        device=device
    )

    optimizer = get_optimizer(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    scheduler = get_scheduler(
        optimizer=optimizer,
        patience=config.training.scheduler_patience,
        factor=config.training.scheduler_factor
    )

    metrics_calculator = MetricsCalculator(
        num_classes=config.model.num_classes,
        device=device
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics_calculator=metrics_calculator,
        device=device,
        checkpoint_dir=str(config.checkpoint_dir),
        gradient_clip_val=config.training.gradient_clip_val,
        mixed_precision=config.training.mixed_precision,
        wandb_logger=wandb_logger
    )

    logger.info("Starting training...")
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(
        f"Dataset: {dataset_info['total_samples']} samples ({dataset_info['train_samples']} train, \
        {dataset_info['val_samples']} val, {dataset_info['test_samples']} test)")
    logger.info(f"Classes: {dataset_info['num_classes']} - {dataset_info['label_names']}")
    logger.info(f"Label distribution: {dataset_info['label_distribution']}")
    logger.info(f"Batch size: {config.data.batch_size}")
    logger.info(f"Number of epochs: {config.training.num_epochs}")
    logger.info(f"Learning rate: {config.training.learning_rate}")
    logger.info(f"Optimizer: {config.training.optimizer}")
    logger.info(f"Scheduler: {config.training.scheduler}")
    logger.info(f"Gradient clipping: {config.training.gradient_clip_val}")
    logger.info(f"Mixed precision: {config.training.mixed_precision}")
    logger.info(f"Early stopping patience: {config.training.early_stopping_patience}")
    logger.info("-" * 50)

    best_accuracy = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.training.num_epochs,
        early_stopping_patience=config.training.early_stopping_patience
    )

    logger.info(f"\nTraining finished! Best validation accuracy: {best_accuracy:.4f}")

    wandb_logger.finish()


if __name__ == '__main__':
    main()
