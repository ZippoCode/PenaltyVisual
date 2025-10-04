from typing import List, Dict, Optional, Tuple

import cv2

from penalty_vision.utils import logger
from penalty_vision.video.frames import visualize_detection


def get_main_player(
        detections: List[Dict],
        strategy: str = 'largest',
        frame_size: Tuple[int, int] = None
) -> Optional[Dict]:
    if not detections:
        return None

    if strategy == 'largest':
        return max(detections, key=lambda d: d['area'])

    elif strategy == 'center':

        if frame_size is None:
            raise ValueError("frame_size must be provided when using strategy='center'")

        def distance_from_center(det):
            x1, y1, x2, y2 = det['bbox']
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            return ((cx - frame_size[0]) ** 2 + (cy - frame_size[1]) ** 2) ** 0.5

        return min(detections, key=distance_from_center)

    elif strategy == 'bottom':
        return max(detections, key=lambda d: d['bbox'][3])

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


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

            detections = player_detector.detect_people(frame)
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
