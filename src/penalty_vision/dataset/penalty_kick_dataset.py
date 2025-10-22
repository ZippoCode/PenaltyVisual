from pathlib import Path
from typing import List

import numpy as np
from torch.utils.data import Dataset

from penalty_vision.dataset.penalty_kick import PenaltyKickSample


class PenaltyKickDataset(Dataset):

    def __init__(self, file_paths: List[Path], labels: List[int]):
        self.file_paths = file_paths
        self.labels = labels

        if not self.file_paths:
            raise ValueError("file_paths list is empty")

        if len(self.file_paths) != len(self.labels):
            raise ValueError(f"Mismatch: {len(self.file_paths)} files but {len(self.labels)} labels")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        data = np.load(self.file_paths[idx], allow_pickle=True)
        metadata = data['metadata'].item()

        return PenaltyKickSample(
            running_embeddings=data['running_embeddings'],
            kicking_embeddings=data['kicking_embeddings'],
            metadata=metadata
        ), self.labels[idx]
