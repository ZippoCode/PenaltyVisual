import argparse
import os
import random

import cv2

from penalty_vision import PlayerDetector, PlayerTracker, VideoProcessor
from penalty_vision.utils import Config
from penalty_vision.utils.frame_utils import draw_detection, resize_frame
from penalty_vision.utils.visualize import visualize_frame


def run_frame(frame_dir: str, checkpoint_path: str):
    extensions = (".jpg", ".png")
    frames = [f for f in os.listdir(frame_dir) if f.lower().endswith(extensions)]
    random_frame = random.choice(frames)
    frame_path = os.path.join(frame_dir, random_frame)

    img = cv2.imread(frame_path)
    img = resize_frame(frame=img, target_size=(1420, 780))

    player_detector = PlayerDetector(model_name=checkpoint_path)
    player_detections = player_detector.detect_kicker(frame=img)
    vis_frame = draw_detection(img, player_detections[0])
    visualize_frame(vis_frame)


def run_video(video_dir, checkpoint_path, tracker_config):
    extensions = (".mp4", ".mov", ".avi", ".mkv")
    videos = [f for f in os.listdir(video_dir) if f.lower().endswith(extensions)]
    random_video = random.choice(videos)
    video_path = os.path.join(video_dir, random_video)

    video_name = random_video.split(".")[0]
    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, f"{video_name}_detected.mp4")
    player_detector = PlayerDetector(model_name=checkpoint_path, tracker=tracker_config)
    tracker = PlayerTracker(player_detector)

    with VideoProcessor(str(video_path)) as vp:
        tracker.track_and_save(vp, output_path)
    tracker.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config path')
    parser.add_argument('--output', type=str, required=True, help='Output folder path')
    args = parser.parse_args()

    config = Config(args.config)
    # run_frame(config.frame_dir, config.checkpoint_path)
    run_video(config.video_dir, config.checkpoint_path, config.tracker_config)
