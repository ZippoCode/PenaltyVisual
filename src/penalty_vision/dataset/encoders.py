import numpy as np


def encode_field_side_label(direction: str) -> int:
    if isinstance(direction, (int, np.integer)):
        return int(direction)

    direction = str(direction).lower().strip()

    if direction == "left" or direction == "0":
        return 0
    elif direction == "right" or direction == "1":
        return 1
    elif direction == "center":
        return 2
    else:
        raise ValueError(f"Invalid direction '{direction}'. Expected 'left', 'right' or integer")


def encode_foot_feature(value: str) -> np.ndarray:
    if isinstance(value, (int, np.integer)):
        value = int(value)
        if value == 0:
            return np.array([1, 0], dtype=np.float32)
        elif value == 1:
            return np.array([0, 1], dtype=np.float32)

    value = str(value).lower().strip()

    if value == "left" or value == "0" or value == "sinistro":
        return np.array([1, 0], dtype=np.float32)
    elif value == "right" or value == "1" or value == "destro":
        return np.array([0, 1], dtype=np.float32)
    else:
        raise ValueError(f"Invalid binary feature value '{value}'. Expected 'left'/'right' or 0/1")


def encode_in_out_feature(value: str) -> np.ndarray:
    if isinstance(value, (int, np.integer)):
        value = int(value)
        if value == 0:
            return np.array([1, 0], dtype=np.float32)
        elif value == 1:
            return np.array([0, 1], dtype=np.float32)

    value = str(value).lower().strip()

    if value == "dentro" or value == "0" or value == "in":
        return np.array([1, 0], dtype=np.float32)
    elif value == "fuori" or value == "1" or value == "out":
        return np.array([0, 1], dtype=np.float32)
    else:
        raise ValueError(f"Invalid binary feature value '{value}'. Expected 'left'/'right' or 0/1")


def encode_metadata(metadata: dict) -> np.ndarray:
    foot = metadata['foot']
    pitch_side = metadata['in_out']

    foot_onehot = encode_foot_feature(foot)
    pitch_side_onehot = encode_in_out_feature(pitch_side)

    return np.concatenate([foot_onehot, pitch_side_onehot])
