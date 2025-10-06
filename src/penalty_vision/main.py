import argparse
import os
import random
import cv2

from penalty_vision import PlayerDetector, PlayerTracker, VideoProcessor
from penalty_vision.utils import Config
from penalty_vision.utils.visualize import visualize_detection


def test_frame(frame_path: str, checkpoint_path: str):
    player_detector = PlayerDetector(model_name=checkpoint_path)
    img = cv2.imread(frame_path)
    player_detections = player_detector.detect_kicker(frame=img)
    visualize_detection(img, player_detections[0])
    exit(1)
    

def test_video(video_path, checkpoint_path, tracker_path):
    extensions = (".mp4", ".mov", ".avi", ".mkv")
    videos = [f for f in os.listdir(config.video_dir) if f.lower().endswith(extensions)]
    random_video = random.choice(videos)
    video_path = os.path.join(config.video_dir, random_video)

    video_name = random_video.split(".")[0]
    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, f"{video_name}_detected.mp4")
    player_detector = PlayerDetector(model_name=checkpoint_path, tracker=tracker_path)
    tracker = PlayerTracker(player_detector)

    with VideoProcessor(str(video_path)) as vp:
        tracker.track_and_save(vp, output_path)
    tracker.reset()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config path')
    parser.add_argument('--output', type=str, required=True, help='Output folder path')
    parser.add_argument('--image', type=str, required=True, help='Image path')
    args = parser.parse_args()

    config_path = args.config
    config = Config(config_path)
    test_frame(args.image, config.checkpoint_path)
