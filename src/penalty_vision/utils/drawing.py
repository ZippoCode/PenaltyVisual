from typing import List, Dict

import cv2
import numpy as np


def draw_kicker(frame: np.ndarray, detections: List[Dict], color: tuple = (0, 255, 0), thickness: int = 2,
                show_conf: bool = True, show_id: bool = False) -> np.ndarray:
    if not detections:
        return frame
    annotated_frame = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
        label_parts = []
        if show_conf: label_parts.append(f"{detection['confidence']:.2f}")
        if show_id and 'track_id' in detection and detection['track_id'] != -1: label_parts.append(
            f"ID:{detection['track_id']}")
        if label_parts:
            label = " ".join(label_parts)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return annotated_frame


def draw_detection(frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
    annotated_frame = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        track_id = det['track_id']
        conf = det['confidence']
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"ID:{track_id} Conf:{conf:.2f}"
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return annotated_frame


def draw_detections_on_frames(
        frames: np.ndarray,
        player_detections: List[Dict],
        ball_detections: List[Dict] = None
) -> np.ndarray:
    drawn_frames = []

    for frame_idx, frame in enumerate(frames):
        frame_copy = frame.copy()

        # Draw players
        if player_detections is not None and len(player_detections) > 0:
            player_data = player_detections[frame_idx]
            for detection in player_data['detections']:
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if 'track_id' in detection:
                    cv2.putText(frame_copy, f"ID: {detection['track_id']}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw ball
        if ball_detections is not None and len(ball_detections) > 0:
            ball_data = ball_detections[frame_idx]
            for ball in ball_data['detections']:
                bbox = ball['bbox']
                x1, y1, x2, y2 = bbox
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                radius = max((x2 - x1), (y2 - y1)) // 2
                cv2.circle(frame_copy, center, radius, (0, 0, 255), 2)

                if 'track_id' in ball:
                    cv2.putText(frame_copy, f"Ball: {ball['track_id']}",
                                (center[0] - 20, center[1] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        drawn_frames.append(frame_copy)

    return np.array(drawn_frames)
