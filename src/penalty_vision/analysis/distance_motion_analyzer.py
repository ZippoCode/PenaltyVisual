from typing import List, Dict

import numpy as np


class DistanceMotionAnalyzer:

    def __init__(self, ball_detections: List[Dict], player_detections: List[Dict]):
        self.ball_detections = ball_detections
        self.player_detections = player_detections

    def calculate_euclidean_distance(self, point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def get_center_point(self, bbox):
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return (center_x, center_y)

    def find_closest_frame(self):
        min_distance = float('inf')
        closest_frame = None

        for ball_frame_data in self.ball_detections:
            frame_id = ball_frame_data['frame']
            ball_dets = ball_frame_data['detections']

            if not ball_dets:
                continue

            player_frame_data = next((p for p in self.player_detections if p['frame'] == frame_id), None)

            if not player_frame_data or not player_frame_data['detections']:
                continue

            ball_bbox = ball_dets[0]['bbox']
            player_bbox = player_frame_data['detections'][0]['bbox']

            ball_center = self.get_center_point(ball_bbox)
            player_center = self.get_center_point(player_bbox)

            distance = self.calculate_euclidean_distance(ball_center, player_center)

            if distance < min_distance:
                min_distance = distance
                closest_frame = frame_id

        return closest_frame
