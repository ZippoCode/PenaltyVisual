from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from penalty_vision.detection.pose_detection import PoseDetection
from penalty_vision.modules.player_tracking import PlayerTracker
from penalty_vision.processor.video_processor import VideoProcessor
from penalty_vision.utils import logger


class PenaltyKickProcessor:
    def __init__(self, player_tracker: PlayerTracker, pose_detection: PoseDetection):
        self.player_tracker = player_tracker
        self.pose_detection = pose_detection

    def _extract_frames_and_track(self, vp: VideoProcessor, target_track_id: Optional[int] = None) -> Tuple[
        List[np.ndarray], List[Optional[Tuple[int, int, int, int]]], int]:
        self.player_tracker.reset()

        all_tracks = []
        frames = []

        for frame_num in range(vp.frame_count):
            frame = vp.get_frame(frame_num)
            if frame is None:
                continue

            frames.append(frame)
            detections = self.player_tracker.detector.track_kicker(frame, persist=True)
            all_tracks.append(detections)

        player_found = False
        bboxes = []

        for frame_idx, detections in enumerate(all_tracks):
            if not player_found and target_track_id is None:
                if len(detections) > 0:
                    largest_det = max(detections,
                                      key=lambda d: (d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1]))
                    target_track_id = largest_det['track_id']
                    player_found = True
                    logger.info(f"Player detected at frame {frame_idx} with track_id {target_track_id}")

            bbox = self._get_bbox_for_track_id(detections, target_track_id) if player_found else None

            if bbox is None and len(bboxes) > 0:
                bbox = bboxes[-1]
                logger.warning(f"Lost tracking at frame {frame_idx}, using last known bbox")

            bboxes.append(bbox)

        if not player_found:
            raise ValueError(f"No player detected in the video")

        self.player_tracker.reset()

        return frames, bboxes, target_track_id

    def process_penalty_kick(self, video_path: str, target_track_id: Optional[int] = None) -> Dict:
        with VideoProcessor(video_path) as vp:
            frames, bboxes, target_track_id = self._extract_frames_and_track(vp, target_track_id)

            poses_data = self.pose_detection.extract_poses_from_frames(frames, bboxes)

            return {
                'frames': frames,
                'bboxes': bboxes,
                'poses': poses_data,
                'tracked_player_id': target_track_id
            }

    def process_and_save(self, video_path: str, output_path: str, target_track_id: Optional[int] = None,
                         show_info: bool = True) -> Dict:
        with VideoProcessor(video_path) as vp:
            frames, bboxes, target_track_id = self._extract_frames_and_track(vp, target_track_id)

            poses_data = self.pose_detection.extract_poses_from_frames(frames, bboxes)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, vp.fps, (vp.width, vp.height))

            for frame_idx, (frame, bbox, pose) in enumerate(zip(frames, bboxes, poses_data['all_poses'])):
                annotated_frame = frame.copy()

                if bbox is not None:
                    annotated_frame = self._draw_bbox(annotated_frame, bbox)

                if pose is not None:
                    annotated_frame = self.pose_detection.draw_pose(annotated_frame, pose)

                if show_info:
                    info_text = f"Frame: {frame_idx} | ID: {target_track_id}"
                    cv2.putText(annotated_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                out.write(annotated_frame)

            out.release()
            logger.info(f"Video saved: {output_path}")

            return {
                'frames': frames,
                'bboxes': bboxes,
                'poses': poses_data,
                'tracked_player_id': target_track_id
            }

    def _get_bbox_for_track_id(self, detections: List[Dict], track_id: int) -> Optional[Tuple[int, int, int, int]]:
        for det in detections:
            if det['track_id'] == track_id:
                return tuple(det['bbox'])
        return None

    def _draw_bbox(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], color: Tuple[int, int, int] = (0, 255, 0),
                   thickness: int = 2) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        return frame