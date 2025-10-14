from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.model_selection import train_test_split


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
    foot = metadata['piede']
    pitch_side = metadata['dentro_fuori']

    foot_onehot = encode_foot_feature(foot)
    pitch_side_onehot = encode_in_out_feature(pitch_side)

    return np.concatenate([foot_onehot, pitch_side_onehot])


def create_stratified_split(
        data_dir: str,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
) -> Dict[str, List[Path]]:
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError(f"Splits must sum to 1.0, got {train_size + val_size + test_size}")

    data_dir = Path(data_dir)
    npz_files = sorted(list(data_dir.glob("*.npz")))

    if not npz_files:
        raise ValueError(f"No .npz files found in {data_dir}")

    labels = []
    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)
        metadata = data['metadata'].item()
        label = encode_field_side_label(metadata['lato'])
        labels.append(label)

    indices = np.arange(len(npz_files))

    train_indices, temp_indices, train_labels, temp_labels = train_test_split(
        indices, labels, train_size=train_size, stratify=labels, shuffle=True
    )

    val_relative_size = val_size / (val_size + test_size)
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=1 - val_relative_size, stratify=temp_labels, shuffle=True
    )

    return {
        'train': [npz_files[i] for i in train_indices],
        'val': [npz_files[i] for i in val_indices],
        'test': [npz_files[i] for i in test_indices]
    }
