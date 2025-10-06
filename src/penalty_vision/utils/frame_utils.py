from typing import Tuple

import cv2
import numpy as np


def resize_frame(frame: np.ndarray, target_size: Tuple[int, int] = (640, 480), keep_aspect: bool = True) -> np.ndarray:
    if not keep_aspect:
        return cv2.resize(frame, target_size)

    h, w = frame.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2

    padded = cv2.copyMakeBorder(resized, pad_h, target_h - new_h - pad_h, pad_w, target_w - new_w - pad_w,
                                cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return padded