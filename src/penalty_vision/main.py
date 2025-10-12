import argparse
import os
from pathlib import Path

from penalty_vision.detection.kick_detector import KickDetector
from penalty_vision.detection.object_detector import ObjectDetector
from penalty_vision.processor.penaltykick_preprocessor import PenaltyKickPreprocessor
from penalty_vision.processor.phase_frame_extractor import PhaseFrameExtractor
from penalty_vision.processor.video_processor import VideoProcessor
from penalty_vision.tracking.ball_tracker import BallTracker
from penalty_vision.tracking.player_tracker import PlayerTracker
from penalty_vision.utils import Config, logger
from penalty_vision.utils.drawing import draw_detections_on_frames
from penalty_vision.utils.ioutils import choice_random_video, save_video, save_frames


def run_save_frames(video_name: Path, config_path: str, output_dir: Path):
    logger.info("=" * 70)
    logger.info("SAVE EXTRACTED FRAMES TEST")
    logger.info("=" * 70)

    logger.info("\nInitializing...")
    player_detector = ObjectDetector(config_path=config_path)
    player_tracker = PlayerTracker(player_detector)
    ball_tracker = BallTracker(player_detector)

    video_output_dir = output_dir / video_name.name.split(".")[0]
    video_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\nExtracting frames from video...")
    with VideoProcessor(str(video_name)) as vp:
        frames = vp.extract_all_frames_as_array()

    logger.info(f"Total frames: {len(frames)}")

    logger.info("\nTracking...")
    player_detections = player_tracker.track_frames(frames)
    ball_detections = ball_tracker.track_frames(frames)

    logger.info("\nSegmenting phases...")
    kick_detector = KickDetector(frames, player_detections, ball_detections)
    temporal_segmentation = kick_detector.segment_penalty_phases()

    logger.info(f"Kick frame: {temporal_segmentation['kick_frame']}")
    logger.info(f"Runup start: {temporal_segmentation['runup_start']}")

    logger.info("\nExtracting training frames...")
    extractor = PhaseFrameExtractor(frames, temporal_segmentation)
    training_frames = extractor.extract_training_frames(n_running=32, n_kicking=16)

    running_frames = training_frames['running_frames']
    kicking_frames = training_frames['kicking_frames']

    logger.info(f"Running frames shape: {running_frames.shape}")
    logger.info(f"Kicking frames shape: {kicking_frames.shape}")

    logger.info("\n" + "=" * 70)
    logger.info("SAVING RUNNING FRAMES")
    logger.info("=" * 70)

    running_dir = video_output_dir / 'running'
    running_paths = save_frames(list(running_frames), str(running_dir), prefix="running")
    logger.info(f"✓ Running frames saved: {len(running_paths)} files")

    kicking_dir = video_output_dir / 'kicking'
    kicking_paths = save_frames(list(kicking_frames), str(kicking_dir), prefix="kicking")
    logger.info(f"✓ Kicking frames saved: {len(kicking_paths)} files")

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Video: {video_name}")
    logger.info(f"Output directory: {video_output_dir}")
    logger.info(f"\nFiles created:")
    logger.info(f"  - {running_dir}/ (32 frames)")
    logger.info(f"  - {kicking_dir}/ (16 frames)")
    logger.info(f"  - running_montage.jpg")
    logger.info(f"  - kicking_montage.jpg")
    logger.info("\n" + "=" * 70)
    logger.info("✓ COMPLETED")
    logger.info("=" * 70)

    return video_output_dir


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config path')
    args = parser.parse_args()

    # run_video(config.video_dir, args.output, config.checkpoint_path, config.tracker_config)
    config = Config.from_yaml(args.config)
    random_video_path = choice_random_video(video_dir=config.paths.video_dir)

    with PenaltyKickPreprocessor(config_path=args.config) as preprocessor:
        result = preprocessor.process_video(random_video_path)
    #     logger.info(json.dumps(result, indent=2))
    run_save_frames(video_name=Path(random_video_path), config_path=args.config, output_dir=Path(config.paths.output))
