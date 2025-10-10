from typing import List, Dict

import numpy as np

from penalty_vision.detection.penalty_kick_detector import PenaltyKickDetector


class BallTracker:
    def __init__(self, detector: PenaltyKickDetector):
        self.detector = detector

    def track_frames(self, frames: np.ndarray) -> List[Dict]:
        detections = []
        for frame_idx, frame in enumerate(frames):
            ball_detections = self.detector.track_ball(frame)
            detections.append({'frame': frame_idx, 'detections': ball_detections})
        return detections
