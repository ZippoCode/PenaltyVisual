from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np

from penalty_vision.utils import logger


class VideoProcessor:

    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            self.cap.release()
            raise ValueError(f"Cannot open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0

        logger.info(f"Video: {self.video_path.name} | "
                    f"Frame count: {self.frame_count} | "
                    f"{self.width}x{self.height} | "
                    f"{self.fps:.1f}fps | "
                    f"{self.duration:.1f}s")

    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        if frame_number < 0 or frame_number >= self.frame_count:
            logger.warning(f"Frame {frame_number} out range [0, {self.frame_count - 1}]")
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()

        if not ret:
            logger.warning(f"Cannot read the frame: {frame_number}")
            return None

        return frame

    def extract_frames(self, start_frame: int, end_frame: int) -> List[np.ndarray]:
        frames = []

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_num in range(start_frame, end_frame):
            ret, frame = self.cap.read()
            if frame is not None:
                frames.append(frame)
            else:
                logger.warning(f"Skipped frame {frame_num}")
                break
        logger.info(f"Extracted {len(frames)} frame from {start_frame} to {end_frame - 1}")
        return frames

    def extract_frames_by_time(self, start_sec: float, end_sec: float) -> List[np.ndarray]:
        start_frame = int(start_sec * self.fps)
        end_frame = int(end_sec * self.fps)

        return self.extract_frames(start_frame, end_frame)

    def release(self):
        if hasattr(self, "cap") and self.cap and self.cap.isOpened():
            self.cap.release()
            logger.info(f"Released video: {self.video_path.name}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False

    def __del__(self):
        self.release()
