import cv2
import numpy as np

from penalty_vision.detection import PlayerDetector
from penalty_vision.detection.detection_utils import get_main_player
from penalty_vision.utils import logger


def visualize_frame(frame: np.ndarray, window_name: str = "Frame", wait_time: int = 0) -> int:
    cv2.imshow(window_name, frame)
    return cv2.waitKey(wait_time)


def visualize_video_detection(video_processor, player_detector: PlayerDetector, max_frames: int = 100):
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
            main_player = get_main_player(detections)  # To FIX
            vis_frame = player_detector.draw_kicker(frame, detections)

            cv2.putText(vis_frame, f"Frame: {i}/{frame_count}", (10, vis_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)

        cv2.imshow("Video Detection", vis_frame)

        key = cv2.waitKey(30)
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("Paused" if paused else "Resumed")

    cv2.destroyAllWindows()
