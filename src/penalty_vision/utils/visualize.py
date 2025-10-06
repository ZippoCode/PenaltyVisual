from typing import Tuple, Optional, Dict

import cv2
import numpy as np

from penalty_vision.modules.detection_utils import get_main_player
from penalty_vision.utils import logger


def visualize_frame(frame: np.ndarray, window_name: str = "Frame", wait_time: int = 0) -> int:
    cv2.imshow(window_name, frame)
    return cv2.waitKey(wait_time)


def visualize_detection(frame: np.ndarray, detection: Optional[Dict], color: tuple = (0, 255, 0),
                        thickness: int = 2) -> np.ndarray:
    vis_frame = frame.copy()

    if detection is None:
        cv2.putText(vis_frame, "NO DETECTION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return vis_frame

    x1, y1, x2, y2 = detection['bbox']
    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)

    conf_text = f"Confidence: {detection['confidence']:.2f}"
    cv2.putText(vis_frame, conf_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    area_text = f"Area: {detection['area']}"
    cv2.putText(vis_frame, area_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    cv2.imshow("Detection", vis_frame)
    cv2.waitKey(30)
    cv2.destroyAllWindows()


def visualize_video_detection(video_processor, player_detector, max_frames: int = 100):
    logger.info(f"Processing {max_frames} frames...")
    logger.info("Press 'q' to quit, 'p' to pause")

    frame_count = min(max_frames, video_processor.frame_count)
    paused = False

    for i in range(frame_count):
        if not paused:
            frame = video_processor.get_frame(i)
            if frame is None:
                break

            detections = player_detector.detect_kicker(frame)
            main_player = get_main_player(detections)
            vis_frame = visualize_detection(frame, main_player)

            cv2.putText(vis_frame, f"Frame: {i}/{frame_count}", (10, vis_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)

        cv2.imshow("Video Detection", vis_frame)

        key = cv2.waitKey(30)  # 30ms = ~33fps playback
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("Paused" if paused else "Resumed")

    cv2.destroyAllWindows()
