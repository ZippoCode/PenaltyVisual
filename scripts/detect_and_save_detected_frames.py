import argparse

from pathlib import Path
from penalty_vision.processing.penalty_kick_video_analyzer import PenaltyKickVideoAnalyzer
from penalty_vision.processing.phase_frame_sampler import PhaseFrameSampler
from penalty_vision.utils.ioutils import save_frames


def extract_and_save_penalty_frames(video_path, config_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = PenaltyKickVideoAnalyzer(config_path)
    results = preprocessor.analyze_video(video_path=video_path)
    sampler = PhaseFrameSampler(results['constrained_frames'], results['temporal_segmentation'])
    phase_frames = sampler.extract_training_frames(n_running=32, n_kicking=16)

    running_frames = phase_frames['running_frames']
    kicking_frames = phase_frames['kicking_frames']

    running_dir = output_dir / "running"
    kicking_dir = output_dir / "kicking"

    save_frames(running_frames, running_dir, "frame")
    save_frames(kicking_frames, kicking_dir, "frame")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and save penalty kick frames")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the penalty kick video file")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the detection configuration YAML file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where frames will be saved")

    args = parser.parse_args()

    extract_and_save_penalty_frames(args.video_path, args.config_path, args.output_dir)
