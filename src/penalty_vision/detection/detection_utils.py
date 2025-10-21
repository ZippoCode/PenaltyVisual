from typing import List, Dict, Optional, Tuple


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