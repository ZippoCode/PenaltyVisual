from typing import List, Dict

import cv2

from penalty_vision.detection.player_detection import PlayerDetector
from penalty_vision.processor.video_processor import VideoProcessor
from penalty_vision.utils import logger


class PlayerTracker:

    def __init__(self, detector: PlayerDetector):
        self.detector = detector
        self.track_history = []

    def track_video(self, vp: VideoProcessor) -> List[Dict]:
        all_tracks = []

        for frame_num in range(vp.frame_count):
            frame = vp.get_frame(frame_num)
            if frame is None:
                continue

            detections = self.detector.track_kicker(frame, persist=True)

            all_tracks.append({
                'frame': frame_num,
                'detections': detections,
                'timestamp': frame_num / vp.fps
            })

            if frame_num % 30 == 0:
                logger.info(f"Tracked {frame_num}/{vp.frame_count} frames...")

        self.track_history = all_tracks
        return all_tracks

    def track_and_save(self, vp: VideoProcessor, output_path: str, visualize: bool = True):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, vp.fps, (vp.width, vp.height))

        all_detections = []

        for frame_num in range(vp.frame_count):
            frame = vp.get_frame(frame_num)
            if frame is None:
                continue

            detections = self.detector.track_kicker(frame, persist=True)
            all_detections.append({'frame': frame_num, 'detections': detections})

            if visualize:
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    track_id = det['track_id']
                    conf = det['confidence']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"ID:{track_id} Conf:{conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            out.write(frame)

            if frame_num % 30 == 0:
                logger.info(f"Processed {frame_num}/{vp.frame_count} frames...")

        out.release()
        self.track_history = all_detections
        logger.info(f"Video saved: {output_path}")

        return all_detections

    def reset(self):
        self.track_history = []
