from typing import List, Dict

import cv2
import numpy as np

from penalty_vision.detection.player_detection import PlayerDetector
from penalty_vision.utils import logger


class PlayerTracker:

    def __init__(self, detector: PlayerDetector):
        self.detector = detector
        self.track_history = []

    def track_frames(self, frames: np.ndarray) -> List[Dict]:
        all_tracks = []

        for frame_num, frame in enumerate(frames):
            detections = self.detector.track_kicker(frame, persist=True)
            all_tracks.append({'frame': frame_num, 'detections': detections})

            # if frame_num % 30 == 0:
            #     logger.info(f"Tracked {frame_num}/{total_frames} frames...")

        self.track_history = all_tracks
        logger.info(f"Tracking completed: {len(frames)} frames processed")
        return all_tracks

    @staticmethod
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

    @staticmethod
    def draw_detections_on_frames(frames: np.ndarray, detections: List[Dict]) -> np.ndarray:
        annotated_frames = []

        for frame_num, frame in enumerate(frames):
            if frame_num < len(detections):
                frame_detections = detections[frame_num]['detections']
                annotated_frame = PlayerTracker.draw_detection(frame, frame_detections)
            else:
                annotated_frame = frame.copy()

            annotated_frames.append(annotated_frame)

        return np.array(annotated_frames)

    def reset(self):
        self.track_history = []
        logger.info("Track history reset")
