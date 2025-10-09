from typing import List, Dict

import cv2
import numpy as np
from ultralytics import YOLO


def draw_kicker(frame: np.ndarray, detections: List[Dict], color: tuple = (0, 255, 0), thickness: int = 2,
                show_conf: bool = True, show_id: bool = False) -> np.ndarray:
    if not detections:
        return frame
    annotated_frame = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
        label_parts = []
        if show_conf: label_parts.append(f"{detection['confidence']:.2f}")
        if show_id and 'track_id' in detection and detection['track_id'] != -1: label_parts.append(
            f"ID:{detection['track_id']}")
        if label_parts:
            label = " ".join(label_parts)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return annotated_frame


def draw_detection(frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
    annotated_frame = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        track_id = det['track_id']
        conf = det['confidence']
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"ID:{track_id} Conf:{conf:.2f}"
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return annotated_frame


def draw_detections_on_frames(frames: np.ndarray, detections: List[Dict]) -> np.ndarray:
    annotated_frames = []

    for frame_num, frame in enumerate(frames):
        if frame_num < len(detections):
            frame_detections = detections[frame_num]['detections']
            annotated_frame = draw_detection(frame, frame_detections)
        else:
            annotated_frame = frame.copy()

        annotated_frames.append(annotated_frame)

    return np.array(annotated_frames)
