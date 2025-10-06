from typing import Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np


class PoseEstimator:

    def __init__(self,
                 model_complexity: int = 1,
                 smooth_landmarks: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self.mp_pose = mp.solutions.pose

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.results = None
        self.current_bbox = None

    def estimate_pose_from_detection(self, frame: np.ndarray, detection: Dict) -> Optional[Dict]:
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        self.current_bbox = bbox

        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return None

        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(roi_rgb)

        if self.results.pose_landmarks:
            return self._get_absolute_landmarks(x1, y1, x2 - x1, y2 - y1)

        return None

    def _get_absolute_landmarks(self, x_offset: int, y_offset: int,
                                width: int, height: int) -> Dict:
        if not self.results or not self.results.pose_landmarks:
            return None

        landmarks = {}
        for idx, landmark in enumerate(self.results.pose_landmarks.landmark):
            landmarks[idx] = {
                'x': landmark.x * width + x_offset,
                'y': landmark.y * height + y_offset,
                'z': landmark.z,
                'visibility': landmark.visibility
            }

        return landmarks

    def extract_keypoints(self, landmarks: Optional[Dict] = None) -> Optional[np.ndarray]:
        if landmarks is None:
            if not self.results or not self.results.pose_landmarks:
                return None

            keypoints = []
            for landmark in self.results.pose_landmarks.landmark:
                keypoints.append([
                    landmark.x,
                    landmark.y,
                    landmark.z,
                    landmark.visibility
                ])
            return np.array(keypoints)

        else:
            keypoints = []
            for idx in sorted(landmarks.keys()):
                lm = landmarks[idx]
                keypoints.append([
                    lm['x'],
                    lm['y'],
                    lm['z'],
                    lm['visibility']
                ])
            return np.array(keypoints)

    def extract_keypoints_flat(self, landmarks: Optional[Dict] = None) -> Optional[np.ndarray]:
        keypoints = self.extract_keypoints(landmarks)
        if keypoints is not None:
            return keypoints.flatten()
        return None

    def extract_specific_keypoints(self, landmarks: Dict, keypoint_indices: List[int]) -> Optional[np.ndarray]:
        if not landmarks:
            return None

        keypoints = []
        for idx in keypoint_indices:
            if idx in landmarks:
                lm = landmarks[idx]
                keypoints.append([
                    lm['x'],
                    lm['y'],
                    lm['z'],
                    lm['visibility']
                ])
            else:
                keypoints.append([0.0, 0.0, 0.0, 0.0])

        return np.array(keypoints)

    def get_lower_body_keypoints(self, landmarks: Dict) -> Optional[np.ndarray]:
        lower_body_indices = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        return self.extract_specific_keypoints(landmarks, lower_body_indices)

    def get_upper_body_keypoints(self, landmarks: Dict) -> Optional[np.ndarray]:
        upper_body_indices = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        return self.extract_specific_keypoints(landmarks, upper_body_indices)

    def get_joint_angle(self, landmarks: Dict, joint_name: str) -> Optional[float]:
        joint_configs = {
            'left_elbow': (11, 13, 15),  # shoulder, elbow, wrist
            'right_elbow': (12, 14, 16),
            'left_knee': (23, 25, 27),  # hip, knee, ankle
            'right_knee': (24, 26, 28),
            'left_hip': (11, 23, 25),  # shoulder, hip, knee
            'right_hip': (12, 24, 26),
        }

        if joint_name not in joint_configs or not landmarks:
            return None

        p1_id, p2_id, p3_id = joint_configs[joint_name]

        if p1_id not in landmarks or p2_id not in landmarks or p3_id not in landmarks:
            return None

        p1 = landmarks[p1_id]
        p2 = landmarks[p2_id]
        p3 = landmarks[p3_id]

        a = np.array([p1['x'], p1['y']])
        b = np.array([p2['x'], p2['y']])
        c = np.array([p3['x'], p3['y']])

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        return np.degrees(angle)

    def is_pose_detected(self) -> bool:
        return self.results is not None and self.results.pose_landmarks is not None

    def release(self):
        self.pose.close()
