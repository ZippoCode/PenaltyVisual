import torch
from tqdm import tqdm

from penalty_vision.training.checkpoints import save_checkpoint
from penalty_vision.training.early_stopping import EarlyStopping


class Trainer:
    def __init__(
            self,
            model,
            criterion,
            optimizer,
            scheduler,
            metrics_calculator,
            device,
            checkpoint_dir='checkpoints',
            gradient_clip_val=None,
            mixed_precision=False,
            wandb_logger=None
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics_calculator = metrics_calculator
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.gradient_clip_val = gradient_clip_val
        self.mixed_precision = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
        self.wandb_logger = wandb_logger

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

            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(running_embeds, kicking_embeds, metadata)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()

                if self.gradient_clip_val is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(running_embeds, kicking_embeds, metadata)
                loss = self.criterion(outputs, labels)
                loss.backward()

                if self.gradient_clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)

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
            for (running_embeds, kicking_embeds, metadata), labels in tqdm(dataloader, desc='Validation'):
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

    def train(self, train_loader, val_loader, num_epochs, early_stopping_patience):
        best_val_accuracy = 0.0
        early_stopping = EarlyStopping(patience=early_stopping_patience, mode='max')

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            train_metrics = self.train_epoch(train_loader)
            print(f"Train Loss: {train_metrics['loss']:.4f} | Accuracy: {train_metrics['accuracy']:.4f}")

            val_metrics = self.validate_epoch(val_loader)
            print(f"Val Loss: {val_metrics['loss']:.4f} | Accuracy: {val_metrics['accuracy']:.4f}")

            val_accuracy = val_metrics['accuracy']

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Learning Rate: {current_lr:.6f}")

            if self.wandb_logger:
                self.wandb_logger.log({
                    "epoch": epoch + 1,
                    "train/loss": train_metrics['loss'],
                    "train/accuracy": train_metrics['accuracy'],
                    "val/loss": val_metrics['loss'],
                    "val/accuracy": val_metrics['accuracy'],
                    "learning_rate": current_lr
                })

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                save_checkpoint(self.model, self.optimizer, epoch, val_metrics, self.checkpoint_dir, 'best_model.pth')
                print(f"Best model saved with accuracy: {best_val_accuracy:.4f}")

                if self.wandb_logger:
                    self.wandb_logger.log_summary({
                        "best_val_accuracy": best_val_accuracy,
                        "best_epoch": epoch + 1
                    })

            if early_stopping(val_accuracy):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        print(f"\nTraining completed. Best validation accuracy: {best_val_accuracy:.4f}")
        return best_val_accuracy
