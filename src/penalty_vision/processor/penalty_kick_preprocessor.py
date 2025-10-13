import json
import os
from pathlib import Path
from typing import Dict

from penalty_vision.detection.object_detector import ObjectDetector
from penalty_vision.detection.pose_detector import PoseDetector
from penalty_vision.processor.context_constraint import ContextConstraint
from penalty_vision.processor.video_processor import VideoProcessor
from penalty_vision.tracking.ball_tracker import BallTracker
from penalty_vision.tracking.player_tracker import PlayerTracker
from penalty_vision.utils import Config, logger
from penalty_vision.utils.drawing import draw_detections_on_frames
from penalty_vision.utils.ioutils import save_video


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

    def process_video(self, video_path: str) -> Dict:
        logger.info(f"Processing video: {video_path}")

        video_name = os.path.basename(video_path).split('.')[0]

        video_output_dir = self.output_dir / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)

        vp = VideoProcessor(str(video_path))
        frames = vp.extract_all_frames_as_array()
        fps = vp.fps
        vp.release()

        player_detections = self.player_tracker.track_frames(frames)
        ball_detections = self.ball_tracker.track_frames(frames)

        tracked_frames = draw_detections_on_frames(frames, player_detections, ball_detections)
        poses_detected = self.pose_detector.extract_poses_from_detections(frames, player_detections)
        dp_frames = self.pose_detector.draw_poses_on_frames(tracked_frames, poses_detected)

        output_pose_path = video_output_dir / "pose_detected.mp4"
        save_video(dp_frames, str(output_pose_path), fps=fps)

        context_constraint = ContextConstraint(frames)
        constrained_frames = context_constraint.process_tracked_sequence(player_detections)

        constrained_output = video_output_dir / "context_constrained.mp4"
        save_video(constrained_frames, str(constrained_output), fps=fps)

        poses_on_constrained = self.pose_detector.draw_poses_on_frames(constrained_frames, poses_detected)
        constrained_pose_output = video_output_dir / "constrained_with_poses.mp4"
        save_video(poses_on_constrained, str(constrained_pose_output), fps=fps)

        result = {
            "video_name": video_name,
            "video_path": str(video_path),
            "total_frames": len(frames),
            "fps": fps,
            "ball_detections": ball_detections,
            "outputs": {
                "pose_detected": str(output_pose_path),
                "context_constrained": str(constrained_output),
                "constrained_with_poses": str(constrained_pose_output)
            }
        }

        info_path = video_output_dir / "info.json"
        with open(info_path, 'w') as f:
            json.dump(result, f, indent=2)

        logger.info(f"Video processed successfully: {video_name}")
        return result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pose_detector.release()
        return False

    def release(self):
        self.pose_detector.release()
