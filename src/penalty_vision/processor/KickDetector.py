from typing import List, Optional, Dict

import numpy as np

from penalty_vision.analysis import BallMotionAnalyzer, PlayerMotionAnalyzer
from penalty_vision.utils import logger


class KickDetector:
    def __init__(self, frames: np.ndarray, player_detections: List[Dict], ball_detections: List[Dict]):
        self.frames = frames
        self.player_detections = player_detections
        self.ball_detections = ball_detections

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

    def _analyze_ball_motion(self) -> Optional[int]:
        ball_analyzer = BallMotionAnalyzer(self.frames, self.ball_detections)
        kick_frame = ball_analyzer.detect_kick_frame()
        return kick_frame

    def _analyze_player_motion(self) -> Optional[int]:
        from penalty_vision.processor.context_constraint import ContextConstraint

        context_constraint = ContextConstraint(self.frames)
        constrained_frames = context_constraint.process_tracked_sequence(self.player_detections)

        motion_analyzer = PlayerMotionAnalyzer(constrained_frames)
        motion_analyzer.compute_optical_flow()

        if motion_analyzer.motion_magnitude is None or len(motion_analyzer.motion_magnitude) == 0:
            return None

        motion_analyzer.smooth_motion()
        return motion_analyzer.detect_kick_frame()
