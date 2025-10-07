from typing import Optional, Dict
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


def draw_detection(frame: np.ndarray, detection: Optional[Dict], color: tuple = (0, 255, 0),
                   thickness: int = 2) -> np.ndarray:
    vis_frame = frame.copy()

    if detection is None:
        cv2.putText(vis_frame, "NO DETECTION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return vis_frame

    x1, y1, x2, y2 = detection['bbox']
    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)

    conf_text = f"Confidence: {detection['confidence']:.2f}"
    cv2.putText(vis_frame, conf_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    area_text = f"Area: {detection['area']}"
    cv2.putText(vis_frame, area_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return vis_frame
