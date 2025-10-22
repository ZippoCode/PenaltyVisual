from typing import Dict, List

import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
from torchmetrics import MetricCollection


class MetricsCalculator:

    def __init__(self, num_classes: int, class_names: List[str], device: torch.device = None):
        self.num_classes = num_classes
        self.device = device
        self.class_names = class_names

        task = 'binary' if num_classes == 2 else 'multiclass'

        self.metrics = MetricCollection({
            'accuracy': Accuracy(task=task, num_classes=num_classes),
            'precision_macro': Precision(task=task, num_classes=num_classes, average='macro'),
            'precision_weighted': Precision(task=task, num_classes=num_classes, average='weighted'),
            'recall_macro': Recall(task=task, num_classes=num_classes, average='macro'),
            'recall_weighted': Recall(task=task, num_classes=num_classes, average='weighted'),
            'f1_macro': F1Score(task=task, num_classes=num_classes, average='macro'),
            'f1_weighted': F1Score(task=task, num_classes=num_classes, average='weighted'),
        }).to(device)

        self.precision_per_class = Precision(task=task, num_classes=num_classes, average=None).to(device)
        self.recall_per_class = Recall(task=task, num_classes=num_classes, average=None).to(device)
        self.f1_per_class = F1Score(task=task, num_classes=num_classes, average=None).to(device)
        self.confusion_matrix = ConfusionMatrix(task=task, num_classes=num_classes).to(device)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        if predictions.dim() > 1 and predictions.shape[1] > 1:
            predictions = torch.argmax(predictions, dim=1)

        self.metrics.update(predictions, targets)
        self.precision_per_class.update(predictions, targets)
        self.recall_per_class.update(predictions, targets)
        self.f1_per_class.update(predictions, targets)
        self.confusion_matrix.update(predictions, targets)

    def compute(self) -> Dict:
        results = self.metrics.compute()

        precision_per_class = self.precision_per_class.compute()
        if precision_per_class.dim() == 0:
            precision_per_class = precision_per_class.unsqueeze(0)
        results['precision_per_class'] = {
            self.class_names[i]: precision_per_class[i].item()
            for i in range(len(precision_per_class))
        }

        recall_per_class = self.recall_per_class.compute()
        if recall_per_class.dim() == 0:
            recall_per_class = recall_per_class.unsqueeze(0)
        results['recall_per_class'] = {
            self.class_names[i]: recall_per_class[i].item()
            for i in range(len(recall_per_class))
        }

        f1_per_class = self.f1_per_class.compute()
        if f1_per_class.dim() == 0:
            f1_per_class = f1_per_class.unsqueeze(0)
        results['f1_per_class'] = {
            self.class_names[i]: f1_per_class[i].item()
            for i in range(len(f1_per_class))
        }

        results['confusion_matrix'] = self.confusion_matrix.compute().cpu().numpy()

        for key in results:
            if isinstance(results[key], torch.Tensor) and results[key].numel() == 1:
                results[key] = results[key].item()

        return results

    def log_to_wandb(self, wandb_logger, metrics: Dict, prefix: str, extra_metrics: Dict = None):
        if wandb_logger is None:
            return

        log_dict = {}

        if extra_metrics:
            log_dict.update(extra_metrics)

        log_dict[f"{prefix}/loss"] = metrics.get('loss')
        log_dict[f"{prefix}/accuracy"] = metrics.get('accuracy')

        if metrics.get('precision_macro'):
            log_dict[f"{prefix}/precision_macro"] = metrics['precision_macro']
        if metrics.get('recall_macro'):
            log_dict[f"{prefix}/recall_macro"] = metrics['recall_macro']
        if metrics.get('f1_macro'):
            log_dict[f"{prefix}/f1_macro"] = metrics['f1_macro']

        if metrics.get('precision_weighted'):
            log_dict[f"{prefix}/precision_weighted"] = metrics['precision_weighted']
        if metrics.get('recall_weighted'):
            log_dict[f"{prefix}/recall_weighted"] = metrics['recall_weighted']
        if metrics.get('f1_weighted'):
            log_dict[f"{prefix}/f1_weighted"] = metrics['f1_weighted']

        for metric_name in ['precision_per_class', 'recall_per_class', 'f1_per_class']:
            if metric_name in metrics:
                metric_short = metric_name.replace('_per_class', '')
                for class_name, value in metrics[metric_name].items():
                    log_dict[f"{prefix}/{metric_short}_{class_name}"] = value

        if 'confusion_matrix' in metrics:
            try:
                import wandb
                import matplotlib.pyplot as plt
                import seaborn as sns

                cm = metrics['confusion_matrix']

                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=self.class_names,
                            yticklabels=self.class_names, ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_title(f'{prefix.capitalize()} Confusion Matrix')

                log_dict[f"confusion_matrix/{prefix}"] = wandb.Image(fig)
                plt.close(fig)
            except (ImportError, Exception):
                pass

        try:
            import wandb
            import matplotlib.pyplot as plt

            if 'precision_per_class' in metrics and 'recall_per_class' in metrics and 'f1_per_class' in metrics:
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))

                metrics_data = [
                    (metrics['precision_per_class'], 'Precision', axes[0]),
                    (metrics['recall_per_class'], 'Recall', axes[1]),
                    (metrics['f1_per_class'], 'F1-Score', axes[2])
                ]

                for data, title, ax in metrics_data:
                    classes = list(data.keys())
                    values = list(data.values())
                    bars = ax.bar(classes, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(classes)])
                    ax.set_title(f'{prefix.capitalize()} {title} per Class')
                    ax.set_ylabel(title)
                    ax.set_ylim(0, 1)
                    ax.grid(axis='y', alpha=0.3)

                    for bar, value in zip(bars, values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2., height,
                                f'{value:.3f}', ha='center', va='bottom', fontsize=9)

                plt.tight_layout()
                log_dict[f"metrics_summary/{prefix}"] = wandb.Image(fig)
                plt.close(fig)
        except (ImportError, Exception):
            pass

        wandb_logger.log(log_dict)

    def reset(self):
        self.metrics.reset()
        self.precision_per_class.reset()
        self.recall_per_class.reset()
        self.f1_per_class.reset()
        self.confusion_matrix.reset()

    def compute_and_reset(self) -> Dict:
        results = self.compute()
        self.reset()
        return results
