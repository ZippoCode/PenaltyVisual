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

    def track_and_save(self, frames: np.ndarray, output_path: str, fps: float = 30.0) -> List[Dict]:
        total_frames, height, width = frames.shape[:3]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        all_detections = []

        for frame_num, frame in enumerate(frames):
            detections = self.detector.track_kicker(frame, persist=True)
            all_detections.append({'frame': frame_num, 'detections': detections})

            frame_to_write = frame.copy()
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                track_id = det['track_id']
                conf = det['confidence']
                cv2.rectangle(frame_to_write, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID:{track_id} Conf:{conf:.2f}"
                cv2.putText(frame_to_write, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            out.write(frame_to_write)

            # if frame_num % 30 == 0:
            #     logger.info(f"Processed {frame_num}/{total_frames} frames...")

        out.release()
        self.track_history = all_detections
        logger.info(f"Video saved: {output_path}")

        return all_detections

    def reset(self):
        self.track_history = []
        logger.info("Track history reset")
