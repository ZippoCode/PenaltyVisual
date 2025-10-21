import cv2
import numpy as np

from typing import List, Dict
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
        logger.info("Computing background using Average Filter")
        self.average_frame = np.mean(self.frames, axis=0, dtype=np.float32).astype(np.uint8)
        return self.average_frame

    def compute_background_median_bilateral(self, d: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0) -> np.ndarray:
        logger.info("Computing background using MEDIAN + Bilateral Filter")
        median_frame = np.median(self.frames, axis=0).astype(np.uint8)
        self.average_frame = cv2.bilateralFilter(median_frame, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        return self.average_frame

    def create_smooth_mask(self, bbox_shape, feather_pixels=10):
        h, w = bbox_shape[:2]
        mask = np.ones((h, w, 1), dtype=np.float32)

        for i in range(feather_pixels):
            alpha = i / feather_pixels
            mask[i, :] *= alpha
            mask[h - 1 - i, :] *= alpha
            mask[:, i] *= alpha
            mask[:, w - 1 - i] *= alpha

        return mask

    def expand_bbox_adaptive(self, bbox, frame_idx, total_running_frames=32):
        x1, y1, x2, y2 = bbox

        if frame_idx < total_running_frames:
            scale_factor = 1.3
        else:
            scale_factor = 2.0

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1

        new_width = width * scale_factor
        new_height = height * scale_factor

        new_x1 = max(0, int(center_x - new_width / 2))
        new_y1 = max(0, int(center_y - new_height / 2))
        new_x2 = min(self.width, int(center_x + new_width / 2))
        new_y2 = min(self.height, int(center_y + new_height / 2))

        return (new_x1, new_y1, new_x2, new_y2)

    def process_tracked_sequence(self, detections: List[Dict], max_propagation_frames: int = 10) -> np.ndarray:
        if self.average_frame is None:
            self.compute_background_median_bilateral()

        processed_frames = np.tile(self.average_frame, (self.num_frames, 1, 1, 1))
        last_valid_bbox = None
        last_valid_frame_idx = None
        frames_since_last_detection = 0

        for track_data in detections:
            frame_idx = track_data['frame']
            detection_list = track_data['detections']
            current_bbox = None
            frame_idx_to_copy = frame_idx

            if len(detection_list) > 0:
                bbox = detection_list[0].get('bbox', None)
                if bbox is not None:
                    current_bbox = bbox
                    last_valid_bbox = bbox
                    last_valid_frame_idx = frame_idx
                    frames_since_last_detection = 0
                    frame_idx_to_copy = frame_idx

            if current_bbox is None and last_valid_bbox is not None:
                if frames_since_last_detection < max_propagation_frames:
                    current_bbox = last_valid_bbox
                    frame_idx_to_copy = last_valid_frame_idx
                    frames_since_last_detection += 1

            if current_bbox is not None:
                x1, y1, x2, y2 = self.expand_bbox_adaptive(current_bbox, frame_idx, total_running_frames=32)

                if x2 > x1 and y2 > y1:
                    processed_frames[frame_idx, y1:y2, x1:x2] = self.frames[frame_idx_to_copy, y1:y2, x1:x2]

        logger.info(f"Processed {self.num_frames} frames with bbox propagation")
        return processed_frames
