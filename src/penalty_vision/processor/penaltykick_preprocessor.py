import os
import json
from pathlib import Path
from typing import Dict

from penalty_vision.detection.player_detection import PlayerDetector
from penalty_vision.detection.pose_detection import PoseDetection
from penalty_vision.processor.context_constraint import ContextConstraint
from penalty_vision.processor.video_processor import VideoProcessor
from penalty_vision.tracking.player_tracking import PlayerTracker
from penalty_vision.utils import Config, logger
from penalty_vision.utils.drawing import draw_detections_on_frames, save_video


class PenaltyKickPreprocessor:
    def __init__(self, config_path: str, output_dir: str = None):
        self.config = Config.from_yaml(config_path)
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(self.config.paths.output)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.player_detector = PlayerDetector(config_path=config_path)
        self.player_tracker = PlayerTracker(self.player_detector)
        self.pose_detector = PoseDetection()
        
        logger.info("PenaltyKickPreprocessor initialized")
    
    def process_video(self, video_path: str) -> Dict:
        logger.info(f"Processing video: {video_path}")
        
        video_name = os.path.basename(video_path).split('.')[0]
        
        vp = VideoProcessor(str(video_path))
        frames = vp.extract_all_frames_as_array()
        fps = vp.fps
        vp.release()
        
        detections = self.player_tracker.track_frames(frames)
        
        tracked_frames = draw_detections_on_frames(frames, detections)
        poses_detected = self.pose_detector.extract_poses_from_detections(frames, detections)
        dp_frames = self.pose_detector.draw_poses_on_frames(tracked_frames, poses_detected)
        
        output_pose_path = self.output_dir / f"{video_name}_pose_detected.mp4"
        save_video(dp_frames, str(output_pose_path), fps=fps)
        
        context_constraint = ContextConstraint(frames)
        constrained_frames = context_constraint.process_tracked_sequence(detections)
        
        constrained_output = self.output_dir / f"{video_name}_context_constrained.mp4"
        save_video(constrained_frames, str(constrained_output), fps=fps)
        
        poses_on_constrained = self.pose_detector.draw_poses_on_frames(constrained_frames, poses_detected)
        constrained_pose_output = self.output_dir / f"{video_name}_constrained_with_poses.mp4"
        save_video(poses_on_constrained, str(constrained_pose_output), fps=fps)
        
        result = {
            "video_name": video_name,
            "video_path": str(video_path),
            "total_frames": len(frames),
            "fps": fps,
            "outputs": {
                "pose_detected": str(output_pose_path),
                "context_constrained": str(constrained_output),
                "constrained_with_poses": str(constrained_pose_output)
            }
        }
        
        info_path = self.output_dir / f"{video_name}_info.json"
        with open(info_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Video processed successfully: {video_name}")
        return result