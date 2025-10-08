import argparse
import os

from penalty_vision.detection import PlayerDetector, PoseDetection
from penalty_vision.modules.player_tracking import PlayerTracker
from penalty_vision.processor.video_processor import VideoProcessor
from penalty_vision.utils import Config
from penalty_vision.utils.ioutils import choice_random_video, save_video


def run_video(video_dir: str, output: str, checkpoint_path: str, tracker_config: str):
    random_video_path = choice_random_video(video_dir=video_dir)
    video_name = os.path.basename(random_video_path).split('.')[0]

    output_path = os.path.join(output, f"{video_name}_detected.mp4")
    player_detector = PlayerDetector(model_name=checkpoint_path, tracker=tracker_config)
    frames = VideoProcessor(str(random_video_path)).extract_all_frames_as_array()

    tracker = PlayerTracker(player_detector)
    detections = tracker.track_frames(frames=frames)
    tracked_frames = tracker.draw_detections_on_frames(frames, detections)
    save_video(tracked_frames, output_path)


def run_video_pose(video_dir: str, output: str, checkpoint_path: str, tracker_config: str):
    random_video_path = choice_random_video(video_dir=video_dir)
    video_name = os.path.basename(random_video_path).split('.')[0]
    frames = VideoProcessor(str(random_video_path)).extract_all_frames_as_array()

    player_detector = PlayerDetector(model_name=checkpoint_path, tracker=tracker_config)
    player_tracker = PlayerTracker(player_detector)
    detections = player_tracker.track_frames(frames)
    tracked_frames = player_tracker.draw_detections_on_frames(frames, detections)

    pose_detection = PoseDetection()
    poses_detected = pose_detection.extract_poses_from_detections(frames, detections)
    dp_frames = pose_detection.draw_poses_on_frames(tracked_frames, poses_detected)

    output_path = os.path.join(output, f"{video_name}_pose_detected.mp4")
    save_video(dp_frames, output_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config path')
    parser.add_argument('--output', type=str, required=True, help='Output folder path')
    args = parser.parse_args()

    config = Config(args.config)
    os.makedirs(args.output, exist_ok=True)

    # run_video(config.video_dir, args.output, config.checkpoint_path, config.tracker_config)
    run_video_pose(config.video_dir, args.output, config.checkpoint_path, config.tracker_config)
