from pathlib import Path
from typing import List

import cv2
import numpy as np

from penalty_vision.utils import logger


def save_frames(frames: List[np.ndarray], output_dir: str, prefix: str = "frame") -> List[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for i, frame in enumerate(frames):
        filename = f"{prefix}_{i:04d}.jpg"
        filepath = output_path / filename
        cv2.imwrite(str(filepath), frame)
        saved_paths.append(filepath)

    logger.info(f"Saved {len(frames)} frame in {output_dir}")
    return saved_paths
