from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from penalty_vision.processing.har_embedding_extractor import HAREmbeddingExtractor
from penalty_vision.processing.penalty_kick_video_analyzer import PenaltyKickVideoAnalyzer
from penalty_vision.processing.phase_frame_sampler import PhaseFrameSampler
from penalty_vision.utils import logger


class PenaltyKickPipeline:

    def __init__(
        self,
        config_path: str,
        har_extractor: HAREmbeddingExtractor,
        output_dir: Path,
        metadata_columns: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None
    ):
        self.preprocessor = PenaltyKickVideoAnalyzer(config_path)
        self.har_extractor = har_extractor
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_columns = metadata_columns
        self.exclude_columns = exclude_columns or ['video_file']

    def _extract_metadata_from_row(self, row: pd.Series) -> Dict:
        if self.metadata_columns:
            return {col: row[col] for col in self.metadata_columns if col in row.index}
        else:
            return {col: row[col] for col in row.index if col not in self.exclude_columns}

    def process_single_video(self, video_path: str, metadata: Dict) -> Dict:
        result = self.preprocessor.analyze_video(video_path)

        phase_extractor = PhaseFrameSampler(result['constrained_frames'], result['temporal_segmentation'])
        phase_frames = phase_extractor.extract_training_frames()

        embeddings = self.har_extractor.process_penalty_kick(
            phase_frames['running_frames'],
            phase_frames['kicking_frames']
        )

        video_name = result['video_name']
        output_path = self.output_dir / f"{video_name}.npz"

        np.savez(
            output_path,
            running_embeddings=embeddings['running_embedding'].numpy(),
            kicking_embeddings=embeddings['kicking_embedding'].numpy(),
            metadata=np.array(metadata, dtype=object)
        )

        logger.info(f"Saved {video_name}.npz")

        return {
            'video_name': video_name,
            'output_path': str(output_path),
            'running_embeddings_shape': embeddings['running_embedding'].shape,
            'kicking_embeddings_shape': embeddings['kicking_embedding'].shape
        }

    def process_dataset_from_csv(self, csv_path: str, video_dir: str) -> Dict:
        csv_path = Path(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        results = {'successful': [], 'failed': []}

        for i, (_, row) in enumerate(df.iterrows()):
            video_name = Path(row['video_file']).stem
            video_path = Path(video_dir) / row['video_file']
            if not video_path.exists():
                logger.error(f"Video file not found: {video_path}")
                results['failed'].append(video_name)
                continue

            metadata = self._extract_metadata_from_row(row)

            try:
                self.process_single_video(str(video_path), metadata)
                results['successful'].append(video_name)
                logger.info(f"SUCCESS [{i + 1}/{len(df)}] {video_name}")

            except Exception as e:
                results['failed'].append(video_name)
                logger.error(f"FAILED [{i + 1}/{len(df)}] {video_name} - {e}")

        return results

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.preprocessor.release()
        return False
