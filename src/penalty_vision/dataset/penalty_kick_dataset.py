from pathlib import Path
from typing import List

import numpy as np
from torch.utils.data import Dataset

from penalty_vision.dataset.dataset_utils import encode_field_side_label
from penalty_vision.dataset.penalty_kick import PenaltyKickSample


class PenaltyKickDataset(Dataset):
    def __init__(self, file_paths: List[Path]):
        self.file_paths = file_paths

        if not self.file_paths:
            raise ValueError("file_paths list is empty")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        data = np.load(self.file_paths[idx], allow_pickle=True)
        metadata = data['metadata'].item()

        label = encode_field_side_label(metadata['lato'])

        return PenaltyKickSample(
            running_frames=data['running_frames'],
            kicking_frames=data['kicking_frames'],
            metadata=metadata
        ), label
