from dataclasses import dataclass

import numpy as np


@dataclass
class PenaltyKickSample:
    running_embeddings: np.ndarray
    kicking_embeddings: np.ndarray
    metadata: dict
