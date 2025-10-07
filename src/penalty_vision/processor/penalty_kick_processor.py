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

    def process_penalty_kick(self, video_path: str, running_frames: int = 32, kicking_frames: int = 16,
                             start_frame: int = 0, target_track_id: Optional[int] = None) -> Dict:
        with VideoProcessor(video_path) as vp:
            total_frames = running_frames + kicking_frames
            end_frame = start_frame + total_frames

            if end_frame > vp.frame_count:
                raise ValueError(f"Video contains only {vp.frame_count} frames, required up to frame {end_frame}")

            frames = vp.extract_frames(start_frame, end_frame)

            if len(frames) < total_frames:
                raise ValueError(f"Extracted only {len(frames)} frames, required {total_frames}")

            bboxes = []
            player_found = False

            for frame_idx, frame in enumerate(frames):
                detections = self.player_tracker.detector.track_kicker(frame, persist=True)

                if not player_found and target_track_id is None:
                    if len(detections) > 0:
                        target_track_id = detections[0]['track_id']
                        player_found = True
                        logger.info(
                            f"Player detected at frame {start_frame + frame_idx} with track_id {target_track_id}")

                bbox = self._get_bbox_for_track_id(detections, target_track_id) if player_found else None

                if bbox is None and len(bboxes) > 0:
                    bbox = bboxes[-1]

                bboxes.append(bbox)

            if not player_found:
                raise ValueError(f"No player detected in any of the {total_frames} frames")

            self.player_tracker.reset()

            poses_data = self.pose_detection.extract_poses_from_frames(frames, bboxes, running_frames, kicking_frames)
            features = self.pose_detection.extract_temporal_features(poses_data['running_poses'],
                                                                     poses_data['kicking_poses'])

            return {
                'frames': frames,
                'bboxes': bboxes,
                'poses': poses_data,
                'features': features,
                'running_poses': poses_data['running_poses'],
                'kicking_poses': poses_data['kicking_poses'],
                'tracked_player_id': target_track_id
            }

    def process_with_visualization(self, video_path: str, running_frames: int = 32, kicking_frames: int = 16,
                                   start_frame: int = 0, target_track_id: Optional[int] = None) -> Tuple[
        Dict, List[np.ndarray]]:
        result = self.process_penalty_kick(video_path, running_frames, kicking_frames, start_frame, target_track_id)

        annotated_frames = []
        for frame, bbox, pose in zip(result['frames'], result['bboxes'], result['poses']['all_poses']):
            annotated_frame = frame.copy()

            if bbox is not None:
                annotated_frame = self._draw_bbox(annotated_frame, bbox)

            if pose is not None:
                annotated_frame = self.pose_detection.draw_pose(annotated_frame, pose)

            annotated_frames.append(annotated_frame)

        return result, annotated_frames

    def process_and_save(self, video_path: str, output_path: str, running_frames: int = 32,
                         kicking_frames: int = 16, start_frame: int = 0,
                         target_track_id: Optional[int] = None, show_info: bool = True) -> Dict:

        result = self.process_penalty_kick(video_path, running_frames, kicking_frames, start_frame, target_track_id)

        with VideoProcessor(video_path) as vp:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, vp.fps, (vp.width, vp.height))

            for frame_idx, (frame, bbox, pose) in enumerate(
                    zip(result['frames'], result['bboxes'], result['poses']['all_poses'])):
                annotated_frame = frame.copy()

                if bbox is not None:
                    annotated_frame = self._draw_bbox(annotated_frame, bbox)

                if pose is not None:
                    annotated_frame = self.pose_detection.draw_pose(annotated_frame, pose)

                if show_info:
                    phase = "RUNNING" if frame_idx < running_frames else "KICKING"
                    info_text = f"Frame: {start_frame + frame_idx} | Phase: {phase} | ID: {result['tracked_player_id']}"
                    cv2.putText(annotated_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                out.write(annotated_frame)

            out.release()
            logger.info(f"Video salvato: {output_path}")

        return result

    def _get_bbox_for_track_id(self, detections: List[Dict], track_id: int) -> Optional[Tuple[int, int, int, int]]:
        for det in detections:
            if det['track_id'] == track_id:
                return tuple(det['bbox'])
        return None

    def _draw_bbox(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], color: Tuple[int, int, int] = (0, 255, 0),
                   thickness: int = 2) -> np.ndarray:
        import cv2
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        return frame
