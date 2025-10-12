from typing import List, Dict, Tuple, Optional

import cv2
import mediapipe as mp
import numpy as np


class PoseDetector:

    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5, static_image_mode: bool = False):
        self.mp_pose = mp.solutions.pose
        self.PoseLandmark = self.mp_pose.PoseLandmark
        self.pose = self.mp_pose.Pose(model_complexity=model_complexity,
                                      min_detection_confidence=min_detection_confidence,
                                      min_tracking_confidence=min_tracking_confidence,
                                      static_image_mode=static_image_mode)
        self.mp_drawing = mp.solutions.drawing_utils

    def extract_pose_from_bbox(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[Dict]:
        x1, y1, x2, y2 = bbox
        cropped_frame = frame[y1:y2, x1:x2]

        frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append({
                    'x': landmark.x * cropped_frame.shape[1] + x1,
                    'y': landmark.y * cropped_frame.shape[0] + y1,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            return {
                'landmarks': landmarks,
                'world_landmarks': results.pose_world_landmarks
            }
        return None

    def extract_poses_from_detections(self, frames: np.ndarray, detections: List[Dict]) -> Dict[int, Optional[Dict]]:
        poses = {}

        for detection_data in detections:
            frame_num = detection_data['frame']
            frame_detections = detection_data['detections']

            if len(frame_detections) > 0 and frame_num < len(frames):
                bbox = frame_detections[0]['bbox']
                frame = frames[frame_num]
                pose_data = self.extract_pose_from_bbox(frame, bbox)
                poses[frame_num] = pose_data
            else:
                poses[frame_num] = None

        return poses

    def normalize_poses(self, poses: Dict[int, Dict]) -> Dict[int, Dict]:
        normalized_poses = {}

        for frame_idx, pose_data in poses.items():
            if not pose_data or 'landmarks' not in pose_data or len(pose_data['landmarks']) == 0:
                normalized_poses[frame_idx] = pose_data
                continue

            landmarks = pose_data['landmarks']
            left_hip = landmarks[self.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[self.PoseLandmark.RIGHT_HIP]

            ref_x = (left_hip['x'] + right_hip['x']) / 2
            ref_y = (left_hip['y'] + right_hip['y']) / 2

            left_shoulder = landmarks[self.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.PoseLandmark.RIGHT_SHOULDER]

            scale = np.sqrt((left_shoulder['x'] - right_shoulder['x']) ** 2 +
                            (left_shoulder['y'] - right_shoulder['y']) ** 2)

            if scale == 0:
                scale = 1.0

            normalized_landmarks = []
            for landmark in landmarks:
                normalized_landmarks.append({
                    'x': (landmark['x'] - ref_x) / scale,
                    'y': (landmark['y'] - ref_y) / scale,
                    'z': landmark['z'] / scale if 'z' in landmark else 0,
                    'visibility': landmark.get('visibility', 1.0)
                })

            normalized_poses[frame_idx] = {'landmarks': normalized_landmarks}
        return normalized_poses

    def draw_pose(self, frame: np.ndarray, pose_data: Dict) -> np.ndarray:
        if not pose_data or 'landmarks' not in pose_data:
            return frame

        annotated_frame = frame.copy()
        landmarks = pose_data['landmarks']

        for landmark in landmarks:
            cv2.circle(annotated_frame, (int(landmark['x']), int(landmark['y'])), 3, (0, 255, 0), -1)

        connections = self.mp_pose.POSE_CONNECTIONS
        for connection in connections:
            start_idx, end_idx = connection
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            cv2.line(annotated_frame, (int(start['x']), int(start['y'])), (int(end['x']), int(end['y'])), (255, 0, 0),
                     2)

        return annotated_frame

    def draw_poses_on_frames(self, frames: np.ndarray, poses: Dict[int, Dict]) -> np.ndarray:
        annotated_frames = []

        for frame_idx in range(len(frames)):
            frame = frames[frame_idx].copy()

            if frame_idx in poses:
                frame = self.draw_pose(frame, poses[frame_idx])

            annotated_frames.append(frame)

        return np.array(annotated_frames)

    def release(self):
        self.pose.close()

    def __del__(self):
        try:
            self.pose.close()
        except Exception:
            pass
