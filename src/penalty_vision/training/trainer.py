import torch
from tqdm import tqdm

from penalty_vision.training.checkpoints import save_checkpoint
from penalty_vision.training.early_stopping import EarlyStopping


class Trainer:
    def __init__(self, model, criterion, optimizer, metrics_calculator, device, checkpoint_dir='checkpoints'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics_calculator = metrics_calculator
        self.device = device
        self.checkpoint_dir = checkpoint_dir

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        self.metrics_calculator.reset()

        for (running_embeds, kicking_embeds, metadata), labels in tqdm(dataloader, desc='Training'):
            running_embeds = running_embeds.to(self.device)
            kicking_embeds = kicking_embeds.to(self.device)
            metadata = metadata.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(running_embeds, kicking_embeds, metadata)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            self.metrics_calculator.update(outputs, labels)

        avg_loss = total_loss / len(dataloader)
        metrics = self.metrics_calculator.compute()
        metrics['loss'] = avg_loss

        return metrics

    def validate_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        self.metrics_calculator.reset()

        with torch.no_grad():
            for (running_embeds, kicking_embeds, metadata), labels in tqdm(dataloader, desc='Training'):
                running_embeds = running_embeds.to(self.device)
                kicking_embeds = kicking_embeds.to(self.device)
                metadata = metadata.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(running_embeds, kicking_embeds, metadata)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                self.metrics_calculator.update(outputs, labels)

        avg_loss = total_loss / len(dataloader)
        metrics = self.metrics_calculator.compute()
        metrics['loss'] = avg_loss

        return metrics

    def train(self, train_loader, val_loader, num_epochs):
        best_val_accuracy = 0.0
        early_stopping = EarlyStopping(patience=10, mode='max')

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            train_metrics = self.train_epoch(train_loader)
            print(f"Train Loss: {train_metrics['loss']:.4f} | Accuracy: {train_metrics['accuracy']:.4f}")

            val_metrics = self.validate_epoch(val_loader)
            print(f"Val Loss: {val_metrics['loss']:.4f} | Accuracy: {val_metrics['accuracy']:.4f}")

            val_accuracy = val_metrics['accuracy']

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                save_checkpoint(self.model, self.optimizer, epoch, val_metrics, self.checkpoint_dir, 'best_model.pth')
                print(f"Best model saved with accuracy: {best_val_accuracy:.4f}")

            if early_stopping(val_accuracy):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        print(f"\nTraining completed. Best validation accuracy: {best_val_accuracy:.4f}")
        return best_val_accuracy
