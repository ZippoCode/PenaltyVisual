from dataclasses import dataclass

import numpy as np


@dataclass
class PenaltyKickSample:
    running_frames: np.ndarray
    kicking_frames: np.ndarray
    metadata: dict
