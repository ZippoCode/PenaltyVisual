import numpy as np
from pathlib import Path
from typing import Dict

from penalty_vision.processor.har_feature_extractor import HARFeatureExtractor
from penalty_vision.processor.penalty_kick_preprocessor import PenaltyKickPreprocessor
from penalty_vision.processor.phase_frame_extractor import PhaseFrameExtractor
from penalty_vision.processor.video_processor import VideoProcessor


class PenaltyKickFeatureExtractor:
    def __init__(self, config_path: str, output_dir: str):
        self.preprocessor = PenaltyKickPreprocessor(config_path)
        self.extractor = HARFeatureExtractor()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_single_video(self, video_path: str, temporal_segmentation: Dict, metadata: Dict) -> Dict:
        result = self.preprocessor.process_video(video_path)
        
        video_name = result['video_name']
        constrained_path = result['outputs']['context_constrained']
        
        vp = VideoProcessor(constrained_path)
        frames = vp.extract_all_frames_as_array()
        vp.release()
        
        phase_extractor = PhaseFrameExtractor(frames, temporal_segmentation)
        phase_frames = phase_extractor.extract_training_frames()
        
        embeddings = self.extractor.process_penalty_kick(
            phase_frames['running_frames'],
            phase_frames['kicking_frames']
        )
        
        output_data = {
            'video_name': video_name,
            'running_embedding': embeddings['running_embedding'].numpy(),
            'kicking_embedding': embeddings['kicking_embedding'].numpy(),
            'metadata': metadata
        }
        
        output_path = self.output_dir / f"{video_name}.npz"
        np.savez(output_path, **output_data)
        
        return output_data