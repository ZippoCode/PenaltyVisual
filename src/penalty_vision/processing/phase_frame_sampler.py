from typing import Dict

import numpy as np


class PhaseFrameSampler:

    def __init__(self, frames: np.ndarray, temporal_segmentation: Dict):
        self.frames = frames
        self.temporal_segmentation = temporal_segmentation

    def extract_training_frames(self, n_running: int = 32, n_kicking: int = 16) -> Dict[str, np.ndarray]:
        phases = self.temporal_segmentation['phases']
        running_start, running_end = phases['runup']
        kicking_start, kicking_end = phases['kick']

        running_frames = self._extract_running_frames(running_start, running_end, n_running)
        kicking_frames = self._extract_kicking_frames(kicking_start, kicking_end, n_kicking)

        return {
            'running_frames': running_frames,
            'kicking_frames': kicking_frames,
            'n_running': len(running_frames),
            'n_kicking': len(kicking_frames)
        }

    def _extract_running_frames(self, start_frame: int, end_frame: int, target_n: int) -> np.ndarray:
        available_frames = self.frames[start_frame:end_frame]
        n_available = len(available_frames)
        if n_available == 0:
            return np.array([])
        if n_available >= target_n:
            indices = np.linspace(0, n_available - 1, target_n, dtype=int)
            return available_frames[indices]
        else:
            n_padding = target_n - n_available
            padded = np.concatenate([
                np.repeat(available_frames[0:1], n_padding, axis=0),
                available_frames
            ])
            return padded

    def _extract_kicking_frames(self, start_frame: int, end_frame: int, target_n: int) -> np.ndarray:
        available_frames = self.frames[start_frame:end_frame]
        n_available = len(available_frames)
        if n_available == 0:
            return np.array([])
        if n_available >= target_n:
            indices = np.linspace(0, n_available - 1, target_n, dtype=int)
            return available_frames[indices]
        else:
            n_padding = target_n - n_available
            padded = np.concatenate([
                available_frames,
                np.repeat(available_frames[-1:], n_padding, axis=0)
            ])
            return padded
