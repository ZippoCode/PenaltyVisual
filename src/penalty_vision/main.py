import argparse
import os
import random

import cv2

from penalty_vision.detection import PlayerDetector, PoseDetection
from penalty_vision.modules.player_tracking import PlayerTracker
from penalty_vision.processor.penalty_kick_processor import PenaltyKickProcessor
from penalty_vision.processor.video_processor import VideoProcessor
from penalty_vision.utils import Config
from penalty_vision.utils.frame_utils import resize_frame
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
    frame = player_detector.draw_kicker(img, detections=player_detections)

    pose_detection = PoseDetection()
    landmarks = pose_detection.extract_pose_landmarks(frame=img, bbox=player_detections[0]['bbox'])
    frame = pose_detection.draw_pose(frame, landmarks)

    visualize_frame(frame)


def run_video(video_dir, output, checkpoint_path, tracker_config):
    extensions = (".mp4", ".mov", ".avi", ".mkv")
    videos = [f for f in os.listdir(video_dir) if f.lower().endswith(extensions)]
    random_video = random.choice(videos)
    video_path = os.path.join(video_dir, random_video)

    video_name = random_video.split(".")[0]
    os.makedirs(output, exist_ok=True)
    output_path = os.path.join(output, f"{video_name}_detected.mp4")
    player_detector = PlayerDetector(model_name=checkpoint_path, tracker=tracker_config)
    tracker = PlayerTracker(player_detector)

    with VideoProcessor(str(video_path)) as vp:
        tracker.track_and_save(vp, output_path)
    tracker.reset()


def run_video_pose(video_dir: str, output: str, checkpoint_path: str, tracker_config: str):
    extensions = (".mp4", ".mov", ".avi", ".mkv")
    videos = [f for f in os.listdir(video_dir) if f.lower().endswith(extensions)]
    random_video = random.choice(videos)
    video_path = os.path.join(video_dir, random_video)

    video_name = random_video.split(".")[0]
    player_detector = PlayerDetector(model_name=checkpoint_path, tracker=tracker_config)
    player_tracker = PlayerTracker(player_detector)
    pose_detection = PoseDetection()
    processor = PenaltyKickProcessor(player_tracker, pose_detection)
    os.makedirs(output, exist_ok=True)
    output_path = os.path.join(output, f"{video_name}_pose_detected.mp4")
    processor.process_and_save(video_path, output_path=output_path, start_frame=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config path')
    parser.add_argument('--output', type=str, required=True, help='Output folder path')
    args = parser.parse_args()

    config = Config(args.config)
    # run_frame(config.frame_dir, config.checkpoint_path)
    run_video_pose(config.video_dir, args.output, config.checkpoint_path, config.tracker_config)
