from typing import Optional

import cv2
import numpy as np

from penalty_vision.utils.signal_processing import smooth_signal


class PlayerMotionAnalyzer:
    def __init__(self, frames: np.ndarray):
        self.frames = frames
        self.num_frames = len(frames)
        self.motion_magnitude = None
        self.motion_smoothed = None

    def compute_optical_flow(self) -> np.ndarray:
        motion_magnitudes = []

        for i in range(self.num_frames - 1):
            frame1_gray = cv2.cvtColor(self.frames[i], cv2.COLOR_BGR2GRAY)
            frame2_gray = cv2.cvtColor(self.frames[i + 1], cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                frame1_gray, frame2_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            avg_magnitude = np.mean(magnitude)
            motion_magnitudes.append(avg_magnitude)

        self.motion_magnitude = np.array(motion_magnitudes)
        return self.motion_magnitude

    def smooth_motion(self, window_length=7, poly_order=2) -> np.ndarray:
        if self.motion_magnitude is None:
            raise ValueError("Compute optical flow first")

        self.motion_smoothed = smooth_signal(
            self.motion_magnitude,
            window_length=window_length,
            poly_order=poly_order
        )
        return self.motion_smoothed

    def detect_kick_frame(self, min_prominence=0.5, window_for_baseline=10) -> Optional[int]:
        if self.motion_smoothed is None:
            raise ValueError("Apply smoothing first")

        baseline = np.mean(self.motion_smoothed[:min(window_for_baseline, len(self.motion_smoothed))])
        motion_above_baseline = self.motion_smoothed - baseline

        potential_kicks = []
        for i in range(1, len(motion_above_baseline) - 1):
            if (motion_above_baseline[i] > motion_above_baseline[i - 1] and
                    motion_above_baseline[i] > motion_above_baseline[i + 1] and
                    motion_above_baseline[i] > min_prominence):
                potential_kicks.append(i)

        if not potential_kicks:
            return np.argmax(self.motion_smoothed)

        return potential_kicks[0]

    def detect_runup_start(self, kick_frame: int, threshold_ratio=0.15, min_baseline_frames=5, min_consecutive_frames=3,
                           velocity_threshold=0.05) -> Optional[int]:
        if self.motion_smoothed is None:
            raise ValueError("Apply smoothing first")
        if kick_frame < min_baseline_frames:
            return 0
        search_window = self.motion_smoothed[:kick_frame]
        baseline = np.mean(search_window[:min(min_baseline_frames, len(search_window))])
        baseline_std = np.std(search_window[:min(min_baseline_frames, len(search_window))])
        threshold = baseline + max(baseline_std * 2, (np.max(search_window) - baseline) * threshold_ratio)
        velocity = np.diff(search_window)
        velocity = np.insert(velocity, 0, 0)
        consecutive_count = 0
        candidate_start = None
        for i in range(len(search_window)):
            if search_window[i] > threshold and velocity[i] > velocity_threshold:
                if consecutive_count == 0:
                    candidate_start = i
                consecutive_count += 1
                if consecutive_count >= min_consecutive_frames:
                    return max(0, candidate_start - 2)
            else:
                consecutive_count = 0
                candidate_start = None
        for i in range(len(search_window)):
            if search_window[i] > threshold:
                return max(0, i - 2)
        return 0
