from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from penalty_vision.processor.penalty_kick_preprocessor import PenaltyKickPreprocessor
from penalty_vision.processor.phase_frame_extractor import PhaseFrameExtractor
from penalty_vision.utils import logger


class PenaltyKickFeatureExtractor:
    def __init__(self, config_path: str, output_dir: str):
        self.preprocessor = PenaltyKickPreprocessor(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_single_video(self, video_path: str, metadata: Dict) -> Dict:
        result = self.preprocessor.extract_embeddings_data(video_path)

        phase_extractor = PhaseFrameExtractor(result['constrained_frames'], result['temporal_segmentation'])
        phase_frames = phase_extractor.extract_training_frames()

        video_name = result['video_name']
        output_path = self.output_dir / f"{video_name}.npz"

        np.savez(
            output_path,
            running_frames=phase_frames['running_frames'],
            kicking_frames=phase_frames['kicking_frames'],
            metadata=metadata
        )

        logger.info(f"Saved {video_name}.npz")

        return {
            'video_name': video_name,
            'output_path': str(output_path),
            'running_frames_shape': phase_frames['running_frames'].shape,
            'kicking_frames_shape': phase_frames['kicking_frames'].shape
        }

    def process_dataset_from_csv(self, csv_path: str, video_dir: str) -> Dict:
        df = pd.read_csv(csv_path)
        results = {'successful': [], 'failed': []}

        for idx, row in df.iterrows():
            video_name = Path(row['video_file']).stem
            video_path = Path(video_dir) / row['video_file']

            if not video_path.exists():
                results['failed'].append(video_name)
                continue

            metadata = {k: row[k] for k in ['dentro_fuori', 'piede', 'lato', 'altezza', 'parato',
                                            'angolo_camera', 'visibilita_giocatore', 'velocita_rincorsa', 'fake']}

            try:
                self.process_single_video(str(video_path), metadata)
                results['successful'].append(video_name)
                logger.info(f"✓ [{idx + 1}/{len(df)}] {video_name}")

            except Exception as e:
                results['failed'].append(video_name)
                logger.error(f"✗ [{idx + 1}/{len(df)}] {video_name} - {e}")

        return results

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.preprocessor.release()
        return False
