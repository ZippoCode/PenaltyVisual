import argparse
from pathlib import Path

import torch

from penalty_vision.config.training_config import get_training_config
from penalty_vision.dataset.dataloaders import create_dataloaders
from penalty_vision.models.losses import get_loss_function
from penalty_vision.models.metrics import MetricsCalculator
from penalty_vision.models.optimizer import get_optimizer
from penalty_vision.models.two_stream_lstm import TwoStreamLSTM
from penalty_vision.training.trainer import Trainer
from penalty_vision.utils.logger import logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, help='Path to config YAML file')
    args = parser.parse_args()

    config = get_training_config(args.config)
    device = torch.device(config.device)
    logger.info(f"Using device: {device}")

    torch.manual_seed(config.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(config.seed)

    model = TwoStreamLSTM(
        input_size=config.model.input_size,
        hidden_size=config.model.hidden_size,
        num_classes=config.model.num_classes,
        dropout=config.model.dropout
    )
    model = model.to(device)

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=str(config.data.data_dir),
        label_field=config.data.label_field,
        batch_size=config.data.batch_size,
        train_size=config.data.train_split,
        val_size=config.data.val_split,
        test_size=config.data.test_split,
        num_workers=config.data.num_workers
    )

    train_labels = train_loader.dataset.labels
    criterion = get_loss_function(
        num_classes=config.model.num_classes,
        labels=train_labels,
        device=device
    )

    optimizer = get_optimizer(
        model.parameters(),
        config.training.learning_rate
    )

    metrics_calculator = MetricsCalculator(
        num_classes=config.model.num_classes,
        device=device
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        metrics_calculator=metrics_calculator,
        device=device,
        checkpoint_dir=str(config.checkpoint_dir)
    )

    logger.info("Starting training...")
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Batch size: {config.data.batch_size}")
    logger.info(f"Number of epochs: {config.training.num_epochs}")
    logger.info(f"Learning rate: {config.training.learning_rate}")
    logger.info("-" * 50)

    best_accuracy = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.training.num_epochs
    )

    logger.info(f"\nTraining finished! Best validation accuracy: {best_accuracy:.4f}")


if __name__ == '__main__':
    main()
