from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn


class WeightedCrossEntropyLoss(nn.Module):

    def __init__(self, num_classes: int, class_counts: Optional[Union[list, np.ndarray]] = None,
                 device: torch.device = 'cpu'):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.device = device

        if class_counts is not None:
            class_counts = np.array(class_counts)
            total_samples = np.sum(class_counts)
            weights = total_samples / (self.num_classes * class_counts)
            weights = weights / np.sum(weights) * self.num_classes
            self.weights = torch.FloatTensor(weights).to(self.device)
        else:
            self.weights = None

        self.criterion = nn.CrossEntropyLoss(weight=self.weights)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(predictions, targets)

    def get_weights(self) -> Optional[torch.Tensor]:
        return self.weights


def get_loss_function(
        num_classes: int,
        labels: Union[list, np.ndarray, torch.Tensor],
        device: torch.device = None
) -> WeightedCrossEntropyLoss:
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    elif isinstance(labels, list):
        labels = np.array(labels)

    class_counts = np.bincount(labels, minlength=num_classes)

    return WeightedCrossEntropyLoss(num_classes, class_counts, device)
