import torch

from penalty_vision.dataset.dataloaders import create_dataloaders
from penalty_vision.models.losses import get_loss_function
from penalty_vision.models.metrics import MetricsCalculator
from penalty_vision.models.optimizer import get_optimizer
from penalty_vision.training.trainer import Trainer
from penalty_vision.models.two_stream_lstm import TwoStreamLSTM


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    batch_size = 32
    num_epochs = 100
    learning_rate = 0.001
    checkpoint_dir = '/home/sprochilo/ai_project/PenaltyVisual/checkpoints'

    num_classes = 3
    label_field = "side"

    model = TwoStreamLSTM(
        input_size=768,
        num_classes=num_classes
    )
    model = model.to(device)

    data_dir = "/home/sprochilo/ai_project/PenaltyVisual/data/embedding"
    train_loader, val_loader, _ = create_dataloaders(data_dir, label_field)
    train_labels = train_loader.dataset.labels
    criterion = get_loss_function(num_classes=num_classes,
                                  labels=train_labels,
                                  device=device)
    optimizer = get_optimizer(model.parameters(), learning_rate)
    metrics_calculator = MetricsCalculator(num_classes=num_classes, device=device)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        metrics_calculator=metrics_calculator,
        device=device,
        checkpoint_dir=checkpoint_dir
    )

    print("Starting training...")
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print("-" * 50)

    best_accuracy = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs
    )

    print(f"\nTraining finished! Best validation accuracy: {best_accuracy:.4f}")


if __name__ == '__main__':
    main()
