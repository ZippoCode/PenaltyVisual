from typing import List, Dict, Tuple, Optional

import cv2
import mediapipe as mp
import numpy as np


class PoseDetection:

    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5, static_image_mode: bool = False):
        self.mp_pose = mp.solutions.pose
        self.PoseLandmark = self.mp_pose.PoseLandmark
        self.pose = self.mp_pose.Pose(model_complexity=model_complexity,
                                      min_detection_confidence=min_detection_confidence,
                                      min_tracking_confidence=min_tracking_confidence,
                                      static_image_mode=static_image_mode)
        self.mp_drawing = mp.solutions.drawing_utils

    def extract_pose_landmarks(self, frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> Optional[
        Dict]:
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cropped_frame = frame[y1:y2, x1:x2]
        else:
            cropped_frame = frame
            x1, y1 = 0, 0
        frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark: landmarks.append(
                {'x': landmark.x * cropped_frame.shape[1] + x1, 'y': landmark.y * cropped_frame.shape[0] + y1,
                 'z': landmark.z, 'visibility': landmark.visibility})
            return {'landmarks': landmarks, 'world_landmarks': results.pose_world_landmarks}
        return None

    def extract_poses_from_frames(self, frames: List[np.ndarray],
                                  bboxes: Optional[List[Tuple[int, int, int, int]]] = None, running_frames: int = 32,
                                  kicking_frames: int = 16) -> Dict[str, List[Dict]]:
        total_frames = min(len(frames), running_frames + kicking_frames)
        all_poses = []

        for frame_idx in range(total_frames):
            bbox = bboxes[frame_idx] if bboxes and frame_idx < len(bboxes) else None
            pose_data = self.extract_pose_landmarks(frames[frame_idx], bbox)
            all_poses.append(pose_data)

        running_poses = all_poses[:running_frames]
        kicking_poses = all_poses[running_frames:running_frames + kicking_frames]

        return {
            'running_poses': running_poses,
            'kicking_poses': kicking_poses,
            'all_poses': all_poses
        }

    def extract_key_angles(self, pose_data: Dict) -> Dict[str, float]:
        if not pose_data or 'landmarks' not in pose_data: return {}
        landmarks = pose_data['landmarks']
        angles = {}
        angles['left_knee'] = self._calculate_angle(landmarks[self.PoseLandmark.LEFT_HIP],
                                                    landmarks[self.PoseLandmark.LEFT_KNEE],
                                                    landmarks[self.PoseLandmark.LEFT_ANKLE])
        angles['right_knee'] = self._calculate_angle(landmarks[self.PoseLandmark.RIGHT_HIP],
                                                     landmarks[self.PoseLandmark.RIGHT_KNEE],
                                                     landmarks[self.PoseLandmark.RIGHT_ANKLE])
        angles['left_hip'] = self._calculate_angle(landmarks[self.PoseLandmark.LEFT_SHOULDER],
                                                   landmarks[self.PoseLandmark.LEFT_HIP],
                                                   landmarks[self.PoseLandmark.LEFT_KNEE])
        angles['right_hip'] = self._calculate_angle(landmarks[self.PoseLandmark.RIGHT_SHOULDER],
                                                    landmarks[self.PoseLandmark.RIGHT_HIP],
                                                    landmarks[self.PoseLandmark.RIGHT_KNEE])
        angles['left_ankle'] = self._calculate_angle(landmarks[self.PoseLandmark.LEFT_KNEE],
                                                     landmarks[self.PoseLandmark.LEFT_ANKLE],
                                                     landmarks[self.PoseLandmark.LEFT_FOOT_INDEX])
        angles['right_ankle'] = self._calculate_angle(landmarks[self.PoseLandmark.RIGHT_KNEE],
                                                      landmarks[self.PoseLandmark.RIGHT_ANKLE],
                                                      landmarks[self.PoseLandmark.RIGHT_FOOT_INDEX])
        angles['torso_lean'] = self._calculate_torso_angle(landmarks[self.PoseLandmark.LEFT_SHOULDER],
                                                           landmarks[self.PoseLandmark.RIGHT_SHOULDER],
                                                           landmarks[self.PoseLandmark.LEFT_HIP],
                                                           landmarks[self.PoseLandmark.RIGHT_HIP])
        angles['left_elbow'] = self._calculate_angle(landmarks[self.PoseLandmark.LEFT_SHOULDER],
                                                     landmarks[self.PoseLandmark.LEFT_ELBOW],
                                                     landmarks[self.PoseLandmark.LEFT_WRIST])
        angles['right_elbow'] = self._calculate_angle(landmarks[self.PoseLandmark.RIGHT_SHOULDER],
                                                      landmarks[self.PoseLandmark.RIGHT_ELBOW],
                                                      landmarks[self.PoseLandmark.RIGHT_WRIST])
        return angles

    def _calculate_angle(self, point1: Dict, point2: Dict, point3: Dict) -> float:
        vector1 = np.array([point1['x'] - point2['x'], point1['y'] - point2['y']])
        vector2 = np.array([point3['x'] - point2['x'], point3['y'] - point2['y']])
        cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def _calculate_torso_angle(self, left_shoulder: Dict, right_shoulder: Dict, left_hip: Dict,
                               right_hip: Dict) -> float:
        shoulder_mid = {'x': (left_shoulder['x'] + right_shoulder['x']) / 2,
                        'y': (left_shoulder['y'] + right_shoulder['y']) / 2}
        hip_mid = {'x': (left_hip['x'] + right_hip['x']) / 2, 'y': (left_hip['y'] + right_hip['y']) / 2}
        vector = np.array([shoulder_mid['x'] - hip_mid['x'], shoulder_mid['y'] - hip_mid['y']])
        vertical = np.array([0, -1])
        cosine_angle = np.dot(vector, vertical) / (np.linalg.norm(vector) * np.linalg.norm(vertical) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def extract_velocity_features(self, pose_sequence: List[Dict]) -> Dict[str, List[float]]:
        if len(pose_sequence) < 2: return {}
        velocities = {'ankle_left': [], 'ankle_right': [], 'knee_left': [], 'knee_right': [], 'hip_left': [],
                      'hip_right': [], 'wrist_left': [], 'wrist_right': []}
        key_points = {'ankle_left': self.PoseLandmark.LEFT_ANKLE, 'ankle_right': self.PoseLandmark.RIGHT_ANKLE,
                      'knee_left': self.PoseLandmark.LEFT_KNEE, 'knee_right': self.PoseLandmark.RIGHT_KNEE,
                      'hip_left': self.PoseLandmark.LEFT_HIP, 'hip_right': self.PoseLandmark.RIGHT_HIP,
                      'wrist_left': self.PoseLandmark.LEFT_WRIST, 'wrist_right': self.PoseLandmark.RIGHT_WRIST}
        for i in range(1, len(pose_sequence)):
            if pose_sequence[i] and pose_sequence[i - 1]:
                for joint_name, idx in key_points.items():
                    curr = pose_sequence[i]['landmarks'][idx];
                    prev = pose_sequence[i - 1]['landmarks'][idx]
                    velocity = np.sqrt((curr['x'] - prev['x']) ** 2 + (curr['y'] - prev['y']) ** 2)
                    velocities[joint_name].append(velocity)
        return velocities

    def extract_temporal_features(self, running_poses: List[Dict], kicking_poses: List[Dict]) -> Dict:
        features = {}
        if running_poses: features['running_angles'] = [self.extract_key_angles(pose) for pose in running_poses if
                                                        pose]; features[
            'running_velocity'] = self.extract_velocity_features(running_poses)
        if kicking_poses: features['kicking_angles'] = [self.extract_key_angles(pose) for pose in kicking_poses if
                                                        pose]; features[
            'kicking_velocity'] = self.extract_velocity_features(kicking_poses)
        return features

    def draw_pose(self, frame: np.ndarray, pose_data: Dict) -> np.ndarray:
        if not pose_data or 'landmarks' not in pose_data: return frame
        annotated_frame = frame.copy()
        landmarks = pose_data['landmarks']
        for landmark in landmarks: cv2.circle(annotated_frame, (int(landmark['x']), int(landmark['y'])), 3, (0, 255, 0),
                                              -1)
        connections = self.mp_pose.POSE_CONNECTIONS
        for connection in connections:
            start_idx, end_idx = connection
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            cv2.line(annotated_frame, (int(start['x']), int(start['y'])), (int(end['x']), int(end['y'])), (255, 0, 0),
                     2)
        return annotated_frame

    @staticmethod
    def get_landmark_position(self, pose_data: Dict, landmark_type: int) -> Optional[Dict]:
        if not pose_data or 'landmarks' not in pose_data: return None
        return pose_data['landmarks'][landmark_type]

    def release(self):
        self.pose.close()
        
    
    def __del__(self):
        try:
            self.pose.close()
        except Exception:
            pass
