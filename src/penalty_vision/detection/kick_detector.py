from typing import List, Optional, Dict

import numpy as np

from penalty_vision.analysis import BallMotionAnalyzer, PlayerMotionAnalyzer
from penalty_vision.processing.context_constraint import ContextConstraint
from penalty_vision.utils import logger


class KickDetector:
    def __init__(self, frames: np.ndarray, player_detections: List[Dict], ball_detections: List[Dict]):
        self.frames = frames
        self.player_detections = player_detections
        self.ball_detections = ball_detections
        self._constrained_frames = None

    @property
    def constrained_frames(self) -> np.ndarray:
        if self._constrained_frames is None:
            context_constraint = ContextConstraint(self.frames)
            self._constrained_frames = context_constraint.process_tracked_sequence(self.player_detections)
        return self._constrained_frames

    def detect_kick_frame(self, ball_weight=0.8, player_weight=0.2) -> int:
        ball_kick = self._analyze_ball_motion()
        player_kick = self._analyze_player_motion()
        logger.info(f"Ball kick {ball_kick} - Player kick {player_kick}")

        if ball_kick is not None and player_kick is not None:
            weighted_kick = int(ball_kick * ball_weight + player_kick * player_weight)
            return weighted_kick

        if ball_kick is not None:
            return ball_kick

        if player_kick is not None:
            return player_kick

        return len(self.frames) // 2

    def detect_runup_start(self, kick_frame: Optional[int] = None) -> int:
        if kick_frame is None:
            kick_frame = self.detect_kick_frame()

        motion_analyzer = PlayerMotionAnalyzer(self.constrained_frames)
        motion_analyzer.compute_optical_flow()
        motion_analyzer.smooth_motion()

        runup_start = motion_analyzer.detect_runup_start(kick_frame)

        return runup_start if runup_start is not None else 0

    def segment_penalty_phases(self, frame_after_kick: int = 3) -> Dict:
        kick_frame = self.detect_kick_frame()
        runup_start = self.detect_runup_start(kick_frame)

        return {
            'kick_frame': int(kick_frame),
            'runup_start': int(runup_start),
            'phases': {
                'static': [int(0), int(runup_start)],
                'runup': [int(runup_start), int(kick_frame)],
                'kick': [int(kick_frame), int(kick_frame + frame_after_kick)]
            }
        }

    def _analyze_ball_motion(self) -> Optional[int]:
        ball_analyzer = BallMotionAnalyzer(self.frames, self.ball_detections)
        kick_frame = ball_analyzer.detect_kick_frame()
        return kick_frame

    def _analyze_player_motion(self) -> Optional[int]:
        motion_analyzer = PlayerMotionAnalyzer(self.constrained_frames)
        motion_analyzer.compute_optical_flow()

        if motion_analyzer.motion_magnitude is None or len(motion_analyzer.motion_magnitude) == 0:
            return None

        motion_analyzer.smooth_motion()
        return motion_analyzer.detect_kick_frame()
