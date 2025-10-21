import numpy as np


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


def encode_fake_feature(value: str) -> np.ndarray:
    if isinstance(value, (int, np.integer)):
        value = int(value)
        if value == 0:
            return np.array([1, 0], dtype=np.float32)
        elif value == 1:
            return np.array([0, 1], dtype=np.float32)

    value = str(value).lower().strip()

    if value == "yes" or value == "0" or value == "si":
        return np.array([1, 0], dtype=np.float32)
    elif value == "no" or value == "1" or value == "no":
        return np.array([0, 1], dtype=np.float32)
    else:
        raise ValueError(f"Invalid binary feature value '{value}'. Expected 'yes'/'no' or 0/1")


def encode_camera_angle_feature(value: str) -> np.ndarray:
    if isinstance(value, (int, np.integer)):
        value = int(value)
        if value == 0:
            return np.array([1, 0, 0, 0], dtype=np.float32)
        elif value == 1:
            return np.array([0, 1, 0, 0], dtype=np.float32)
        elif value == 2:
            return np.array([0, 0, 1, 0], dtype=np.float32)
        elif value == 3:
            return np.array([0, 0, 0, 1], dtype=np.float32)

    value = str(value).lower().strip()

    if value == "diagonale" or value == "0" or value == "diagonal":
        return np.array([1, 0, 0, 0], dtype=np.float32)
    elif value == "laterale" or value == "1" or value == "lateral":
        return np.array([0, 1, 0, 0], dtype=np.float32)
    elif value == "frontale" or value == "2" or value == "frontal":
        return np.array([0, 0, 1, 0], dtype=np.float32)
    elif value == "dietro" or value == "2" or value == "behind":
        return np.array([0, 0, 0, 1], dtype=np.float32)
    else:
        raise ValueError(
            f"Invalid binary feature value '{value}'. Expected 'diagonal'/'lateral'/'frontal'/'behind' or 0/1/2/3")


def encode_player_visibility_feature(value: str) -> np.ndarray:
    if isinstance(value, (int, np.integer)):
        value = int(value)
        if value == 0:
            return np.array([1, 0, 0], dtype=np.float32)
        elif value == 1:
            return np.array([0, 1, 0], dtype=np.float32)
        elif value == 2:
            return np.array([0, 0, 1], dtype=np.float32)

    value = str(value).lower().strip()

    if value == "full" or value == "0" or value == "completa":
        return np.array([1, 0, 0], dtype=np.float32)
    elif value == "partial" or value == "1" or value == "parziale":
        return np.array([0, 1, 0], dtype=np.float32)
    elif value == "obstructed" or value == "2" or value == "ostruita" or value == 'poor':
        return np.array([0, 0, 1], dtype=np.float32)
    else:
        raise ValueError(f"Invalid feature value '{value}'. Expected 'full'/'partial/'/'obstructed' or 0/1/2")


def encode_run_speed_feature(value: str) -> np.ndarray:
    if isinstance(value, (int, np.integer)):
        value = int(value)
        if value == 0:
            return np.array([1, 0, 0], dtype=np.float32)
        elif value == 1:
            return np.array([0, 1, 0], dtype=np.float32)
        elif value == 2:
            return np.array([0, 0, 1], dtype=np.float32)

    value = str(value).lower().strip()

    if value == "fast" or value == "0" or value == "veloce":
        return np.array([1, 0, 0], dtype=np.float32)
    elif value == "medium" or value == "1" or value == "media":
        return np.array([0, 1, 0], dtype=np.float32)
    elif value == "slow" or value == "2" or value == "lenta":
        return np.array([0, 0, 1], dtype=np.float32)
    else:
        raise ValueError(f"Invalid feature value '{value}'. Expected 'fast'/'medium/'/'slow' or 0/1/2")


def encode_metadata(metadata: dict) -> np.ndarray:
    foot = metadata['foot']
    in_out = metadata['in_out']
    camera_angle = metadata['camera_angle']
    player_visibility = metadata['player_visibility']
    run_speed = metadata['run_speed']
    fake = metadata['fake']

    foot_onehot = encode_foot_feature(foot)
    in_out_onehot = encode_in_out_feature(in_out)
    camera_angle_onehot = encode_camera_angle_feature(camera_angle)
    player_visibility_onehot = encode_player_visibility_feature(player_visibility)
    run_speed_onehot = encode_run_speed_feature(run_speed)
    fake_onehot = encode_fake_feature(fake)

    encoded_metadata = np.concatenate([
        foot_onehot,
        in_out_onehot,
        camera_angle_onehot,
        player_visibility_onehot,
        run_speed_onehot,
        fake_onehot,
    ])
    return encoded_metadata
