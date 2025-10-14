from typing import Optional, Dict

import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
from torchmetrics import MetricCollection


class MetricsCalculator:

    def __init__(self, num_classes: int, class_names: Optional[list] = None, device: str = 'cpu'):
        self.num_classes = num_classes
        self.device = device

        if class_names is None:
            if num_classes == 2:
                self.class_names = ['left', 'right']
            elif num_classes == 3:
                self.class_names = ['left', 'center', 'right']
            else:
                self.class_names = [f'class_{i}' for i in range(num_classes)]
        else:
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
        results['precision_per_class'] = {
            self.class_names[i]: precision_per_class[i].item()
            for i in range(len(precision_per_class))
        }

        recall_per_class = self.recall_per_class.compute()
        results['recall_per_class'] = {
            self.class_names[i]: recall_per_class[i].item()
            for i in range(len(recall_per_class))
        }

        f1_per_class = self.f1_per_class.compute()
        results['f1_per_class'] = {
            self.class_names[i]: f1_per_class[i].item()
            for i in range(len(f1_per_class))
        }

        results['confusion_matrix'] = self.confusion_matrix.compute().cpu().numpy()

        for key in results:
            if isinstance(results[key], torch.Tensor) and results[key].numel() == 1:
                results[key] = results[key].item()

        return results

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


def get_metrics_calculator(num_classes: int, class_names: Optional[list] = None,
                           device: str = 'cpu') -> MetricsCalculator:
    return MetricsCalculator(num_classes, class_names, device)
