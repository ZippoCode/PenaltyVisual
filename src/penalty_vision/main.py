import argparse
import os
import random

from penalty_vision import PlayerDetector, PlayerTracker, VideoProcessor
from penalty_vision.utils import Config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config path')
    parser.add_argument('--output', type=str, required=True, help='Config path')
    args = parser.parse_args()

    config_path = args.config
    config = Config(config_path)

    frame_dir = config.frame_dir

    extensions = (".mp4", ".mov", ".avi", ".mkv")
    videos = [f for f in os.listdir(config.video_dir) if f.lower().endswith(extensions)]
    random_video = random.choice(videos)
    video_path = os.path.join(config.video_dir, random_video)

    video_name = random_video.split(".")[0]
    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, f"{video_name}_detected.mp4")
    video_processor = VideoProcessor(str(video_path))
    player_detector = PlayerDetector(model_name=config.checkpoint_path, tracker=config.tracker_config)
    tracker = PlayerTracker(player_detector)

    with VideoProcessor(str(video_path)) as vp:
        tracks = tracker.track_and_save(vp, output_path)

    tracker.reset()
