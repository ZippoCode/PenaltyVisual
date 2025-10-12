import argparse
import os

from penalty_vision.detection import ObjectDetector
from penalty_vision.processor.penaltykick_preprocessor import PenaltyKickPreprocessor
from penalty_vision.processor.video_processor import VideoProcessor
from penalty_vision.tracking.player_tracker import PlayerTracker
from penalty_vision.utils import Config, logger
from penalty_vision.utils.drawing import draw_detections_on_frames
from penalty_vision.utils.ioutils import choice_random_video, save_video


def run_video(config_path: str):
    config = Config.from_yaml(config_path)
    random_video_path = choice_random_video(video_dir=config.paths.video_dir)
    video_name = os.path.basename(random_video_path).split('.')[0]

    output_path = os.path.join(config.paths.output, f"{video_name}_detected.mp4")
    player_detector = ObjectDetector(config_path=config_path)
    frames = VideoProcessor(str(random_video_path)).extract_all_frames_as_array()

    tracker = PlayerTracker(player_detector)
    detections = tracker.track_frames(frames=frames)
    tracked_frames = draw_detections_on_frames(frames, detections)
    save_video(tracked_frames, output_path)


if __name__ == '__main__':
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config path')
    args = parser.parse_args()

    # run_video(config.video_dir, args.output, config.checkpoint_path, config.tracker_config)
    config = Config.from_yaml(args.config)
    random_video_path = choice_random_video(video_dir=config.paths.video_dir)

    with PenaltyKickPreprocessor(config_path=args.config) as preprocessor:
        result = preprocessor.process_video(random_video_path)
        logger.info(json.dumps(result, indent=2))
