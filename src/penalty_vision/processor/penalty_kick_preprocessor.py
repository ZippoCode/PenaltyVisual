import os
from pathlib import Path
from typing import Dict

from penalty_vision.detection.kick_detector import KickDetector
from penalty_vision.detection.object_detector import ObjectDetector
from penalty_vision.detection.pose_detector import PoseDetector
from penalty_vision.processor.video_processor import VideoProcessor
from penalty_vision.tracking.ball_tracker import BallTracker
from penalty_vision.tracking.player_tracker import PlayerTracker
from penalty_vision.utils import Config, logger


class PenaltyKickPreprocessor:
    def __init__(self, config_path: str):
        self.config = Config.from_yaml(config_path)

        self.output_dir = Path(self.config.paths.output)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.player_detector = ObjectDetector(config_path=config_path)
        self.player_tracker = PlayerTracker(self.player_detector)
        self.ball_tracker = BallTracker(self.player_detector)
        self.pose_detector = PoseDetector()

        logger.info("PenaltyKickPreprocessor initialized")

    def extract_embeddings_data(self, video_path: str) -> Dict:
        logger.info(f"Extracting embeddings data: {video_path}")
        video_name = os.path.basename(video_path).split('.')[0]

        with VideoProcessor(str(video_path)) as vp:
            frames = vp.extract_all_frames_as_array()

            player_detections = self.player_tracker.track_frames(frames)
            ball_detections = self.ball_tracker.track_frames(frames)

            kick_detector = KickDetector(frames, player_detections, ball_detections)
            temporal_segmentation = kick_detector.segment_penalty_phases()
            constrained_frames = kick_detector.constrained_frames

        return {
            "video_name": video_name,
            "constrained_frames": constrained_frames,
            "temporal_segmentation": temporal_segmentation
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pose_detector.release()
        return False

    def release(self):
        self.pose_detector.release()
