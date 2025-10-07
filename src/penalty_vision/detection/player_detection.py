from typing import List, Dict

import cv2
import numpy as np
from ultralytics import YOLO

from penalty_vision.utils import logger


class PlayerDetector:
    def __init__(self, model_name: str = "yolo11n.pt", tracker: str = "bytetrack.yaml", confidence: float = 0.5):
        logger.info(f"Loading YOLO model: {model_name}...")

        self.model = YOLO(model_name)
        self.confidence = confidence
        self.tracker = tracker
        self.KICKER_CLASS_ID = 0
        self.BALL_CLASS_ID = 1

        logger.info(f"Model loaded: {model_name}")
        logger.info(f"Confidence threshold: {confidence}")

    def detect(self, frame: np.ndarray, class_id: int) -> List[Dict]:
        results = self.model(frame, verbose=False)

        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                if int(box.cls) == class_id:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf_score = float(box.conf)

                    area = int((x2 - x1) * (y2 - y1))

                    detections.append(
                        {'bbox': (int(x1), int(y1), int(x2), int(y2)), 'confidence': conf_score, 'area': area})
        return detections

    def detect_kicker(self, frame: np.ndarray) -> List[Dict]:
        return self.detect(frame, self.KICKER_CLASS_ID)

    def detect_ball(self, frame: np.ndarray) -> List[Dict]:
        return self.detect(frame, self.BALL_CLASS_ID)

    def track(self, frame: np.ndarray, class_id: int, persist: bool = True) -> List[Dict]:
        results = self.model.track(frame, persist=persist, verbose=False, conf=self.confidence, tracker=self.tracker)

        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                if int(box.cls) == class_id:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf_score = float(box.conf)
                    area = int((x2 - x1) * (y2 - y1))
                    track_id = int(box.id) if box.id is not None else -1

                    detections.append(
                        {'bbox': (int(x1), int(y1), int(x2), int(y2)), 'confidence': conf_score, 'area': area,
                         'track_id': track_id})

        return detections

    def track_kicker(self, frame: np.ndarray, persist: bool = True) -> List[Dict]:
        return self.track(frame, self.KICKER_CLASS_ID, persist)

    @staticmethod
    def draw_kicker(frame: np.ndarray, detections: List[Dict], color: tuple = (0, 255, 0), thickness: int = 2,
                    show_conf: bool = True, show_id: bool = False) -> np.ndarray:
        if not detections: return frame
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
