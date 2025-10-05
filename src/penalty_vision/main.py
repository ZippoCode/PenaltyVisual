import argparse
import os
import random

from penalty_vision import PlayerDetector, VideoProcessor
from penalty_vision.detection.detection_utils import visualize_video_detection
from penalty_vision.utils import Config
from penalty_vision.video.frames import resize_frame

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config path')
    args = parser.parse_args()

    config_path = args.config
    config = Config(config_path)

    frame_dir = config.frame_dir

    extensions = (".mp4", ".mov", ".avi", ".mkv")
    videos = [f for f in os.listdir(config.video_dir) if f.lower().endswith(extensions)]
    random_video = random.choice(videos)
    video_path = os.path.join(config.video_dir, random_video)
    vp = VideoProcessor(str(video_path))

    pd = PlayerDetector(model_name=config.checkpoint_path, weights_dir=config.weights_dir)
    visualize_video_detection(vp, pd, max_frames=50)
