# penalty_vision/preprocessing/context_constraint.py

import numpy as np
from typing import List, Dict, Tuple
from penalty_vision.utils import logger


class ContextConstraint:

    def __init__(self, frames: np.ndarray):
        if frames.ndim != 4:
            raise ValueError(f"Expected 4D array (num_frames, height, width, channels), got {frames.ndim}D")

        self.frames = frames
        self.average_frame = None
        self.num_frames, self.height, self.width, self.channels = frames.shape
        logger.info(f"ContextConstraint initialized with {self.num_frames} frames")

    def compute_average_frame(self) -> np.ndarray:
        self.average_frame = np.mean(self.frames, axis=0, dtype=np.float32).astype(np.uint8)
        return self.average_frame

    def process_tracked_sequence(self, detections: List[Dict], max_propagation_frames: int = 10) -> np.ndarray:
        if self.average_frame is None:
            self.compute_average_frame()

        processed_frames = np.tile(self.average_frame, (self.num_frames, 1, 1, 1))

        last_valid_bbox = None
        frames_since_last_detection = 0

        for track_data in detections:
            frame_idx = track_data['frame']
            detection_list = track_data['detections']

            current_bbox = None

            if len(detection_list) > 0:
                bbox = detection_list[0].get('bbox', None)
                if bbox is not None:
                    current_bbox = bbox
                    last_valid_bbox = bbox
                    frames_since_last_detection = 0

            if current_bbox is None and last_valid_bbox is not None:
                if frames_since_last_detection < max_propagation_frames:
                    current_bbox = last_valid_bbox
                    frames_since_last_detection += 1
                    # logger.debug(f"Frame {frame_idx}: propagating bbox from {frames_since_last_detection} frames ago")
                # else:
                    # logger.warning(f"Frame {frame_idx}: no bbox and max propagation limit reached")

            if current_bbox is not None:
                x1, y1, x2, y2 = current_bbox
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(self.width, int(x2)), min(self.height, int(y2))

                if x2 > x1 and y2 > y1:
                    processed_frames[frame_idx, y1:y2, x1:x2] = self.frames[frame_idx, y1:y2, x1:x2]

        logger.info(f"Processed {self.num_frames} frames with bbox propagation")
        return processed_frames
