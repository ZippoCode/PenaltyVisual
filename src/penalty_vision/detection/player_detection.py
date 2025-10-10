from typing import List, Dict
from pathlib import Path
import numpy as np
from ultralytics import YOLO


from penalty_vision.utils import Config, logger


class PlayerDetector:
    def __init__(self, config_path: str = None):
        if config_path:
            self.config = Config.from_yaml(config_path)
        else:
            self.config = Config()
            
        self._load_model()
        self._load_tracker()
        
        self.KICKER_CLASS_ID = self.config.classes.kicker
        self.BALL_CLASS_ID = self.config.classes.ball

        logger.info(f"Confidence threshold: {self.confidence}")

    def _load_model(self):
        model_path = Path(self.config.model.weights)
        
        if model_path.exists():
            logger.info(f"Loading model: {model_path}")
            self.model = YOLO(str(model_path))
            logger.info(f"Model loaded successfully: {model_path.name}")
        else:
            logger.warning(f"Model not found: {model_path}")
            logger.info(f"Using fallback model: {self.config.model.fallback}")
            self.model = YOLO(self.config.model.fallback)

        total_params = sum(p.numel() for p in self.model.model.parameters())
        logger.info(f"Model parameters: {total_params:,}")
    
    def _load_tracker(self):
        tracker_path = Path(self.config.tracking.tracker)
        
        if tracker_path.exists():
            logger.info(f"Using custom tracker: {tracker_path}")
            self.tracker = str(tracker_path)
        else:
            logger.info(f"Using default tracker: {self.config.tracking.tracker}")
            self.tracker = self.config.tracking.tracker
    
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
