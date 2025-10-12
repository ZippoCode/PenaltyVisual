from typing import List, Dict, Optional

import numpy as np


class BallMotionAnalyzer:
    def __init__(self, frames: np.ndarray, ball_detections: List[Dict]):
        self.frames = frames
        self.ball_detections = ball_detections
        self.num_frames = len(frames)

    def detect_kick_frame(self, min_consecutive_before=5, offset_frames=5) -> Optional[int]:
        consecutive_detections = 0
        last_detection_frame = None

        for i in range(len(self.ball_detections) - 1, -1, -1):
            detections_list = self.ball_detections[i]['detections']
            has_detection = len(detections_list) > 0

            if has_detection:
                consecutive_detections += 1
                if last_detection_frame is None:
                    last_detection_frame = i
            else:
                if consecutive_detections >= min_consecutive_before and last_detection_frame is not None:
                    kick_frame = min(last_detection_frame + offset_frames, len(self.ball_detections) - 1)
                    return kick_frame
                consecutive_detections = 0
                last_detection_frame = None

        return None
