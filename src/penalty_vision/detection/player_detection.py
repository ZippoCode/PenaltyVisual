from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from ultralytics import YOLO, settings

from penalty_vision.utils import logger


class PlayerDetector:
    def __init__(self, model_name: str = "yolo11n.pt", confidence: float = 0.5, weights_dir: str = "models/weights"):
        weights_path = Path(weights_dir)
        weights_path.mkdir(parents=True, exist_ok=True)
        settings.update({'weights_dir': weights_dir})

        logger.info(f"Loading YOLO model: {model_name}...")
        logger.info(f"Weights directory: {weights_path}")
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.PERSON_CLASS_ID = 0

        logger.info(f"Model loaded: {model_name}")
        logger.info(f"Confidence threshold: {confidence}")

    def detect_people(self, frame: np.ndarray) -> List[Dict]:
        results = self.model(frame, verbose=False)

        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                if int(box.cls) == self.PERSON_CLASS_ID:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf_score = float(box.conf)

                    area = int((x2 - x1) * (y2 - y1))

                    detections.append(
                        {'bbox': (int(x1), int(y1), int(x2), int(y2)), 'confidence': conf_score, 'area': area})

        return detections

