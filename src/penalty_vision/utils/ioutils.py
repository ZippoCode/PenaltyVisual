import os
import random
from pathlib import Path
from typing import List

import cv2
import numpy as np

from penalty_vision.utils import logger


def save_frames(frames: List[np.ndarray], output_dir: str, prefix: str = "frame") -> List[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for i, frame in enumerate(frames):
        filename = f"{prefix}_{i:04d}.jpg"
        filepath = output_path / filename
        cv2.imwrite(str(filepath), frame)
        saved_paths.append(filepath)

    logger.info(f"Saved {len(frames)} frame in {output_dir}")
    return saved_paths


def choice_random_video(video_dir: str) -> str:
    if not Path(video_dir).is_dir():
        raise NotADirectoryError(f"{video_dir} is not a directory")
    extensions = (".mp4", ".mov", ".avi", ".mkv")
    videos = [f for f in os.listdir(video_dir) if f.lower().endswith(extensions)]
    random_video = random.choice(videos)
    video_path = os.path.join(video_dir, random_video)
    return video_path


def choice_random_image(frame_dir: str) -> str:
    if not Path(frame_dir).is_dir():
        raise NotADirectoryError(f"{frame_dir} is not a directory")
    extensions = (".jpg", ".png")
    frames = [f for f in os.listdir(frame_dir) if f.lower().endswith(extensions)]
    random_frame = random.choice(frames)
    frame_path = os.path.join(frame_dir, random_frame)
    return frame_path
