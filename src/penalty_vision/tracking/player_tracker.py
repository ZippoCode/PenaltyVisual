from typing import List, Dict

import numpy as np

from penalty_vision.detection.object_detector import ObjectDetector
from penalty_vision.utils import logger


class PlayerTracker:

    def __init__(self, detector: ObjectDetector):
        self.detector = detector

    def track_frames(self, frames: np.ndarray) -> List[Dict]:
        all_tracks = []

        for frame_num, frame in enumerate(frames):
            detections = self.detector.detect_kicker(frame)
            all_tracks.append({'frame': frame_num, 'detections': detections})

        logger.info(f"Tracking completed: {len(frames)} frames processed")
        return all_tracks
