from pathlib import Path
from typing import Dict, List

import numpy as np
from penalty_vision.dataset.encoders import encode_side_labels
from penalty_vision.utils.logger import logger
from sklearn.model_selection import train_test_split, StratifiedKFold


def create_kfold_splits(data_dir: str, n_folds: int = 10, seed: int = 32) -> List[Dict]:
    data_dir = Path(data_dir)
    npz_files = sorted(list(data_dir.glob("*.npz")))

    if not npz_files:
        raise ValueError(f"No .npz files found in {data_dir}")

    logger.info(f"Found {len(npz_files)} files")

    encoded_labels, label_to_int = encode_side_labels(npz_files)
    logger.info(f"Label mapping: {label_to_int}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(npz_files, encoded_labels)):
        val_size = len(test_idx)
        train_size = len(train_val_idx) - val_size

        train_idx = train_val_idx[:train_size]
        val_idx = train_val_idx[train_size:]

        fold_data = {
            'fold': fold_idx,
            'train': {
                'files': [npz_files[i] for i in train_idx],
                'labels': encoded_labels[train_idx].tolist()
            },
            'val': {
                'files': [npz_files[i] for i in val_idx],
                'labels': encoded_labels[val_idx].tolist()
            },
            'test': {
                'files': [npz_files[i] for i in test_idx],
                'labels': encoded_labels[test_idx].tolist()
            },
            'label_mapping': label_to_int
        }

        folds.append(fold_data)

        logger.info(f"Fold {fold_idx}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    return folds


def create_stratified_split(
        data_dir: str,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1
) -> Dict[str, Dict[str, List]]:
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError(f"Splits must sum to 1.0, got {train_size + val_size + test_size}")

    data_dir = Path(data_dir)
    npz_files = sorted(list(data_dir.glob("*.npz")))

    if not npz_files:
        raise ValueError(f"No .npz files found in {data_dir}")

    logger.info(f"Found {len(npz_files)} .npz files in {data_dir}")

    encoded_labels, label_to_int = encode_side_labels(npz_files)
    logger.info(f"Label mapping: {label_to_int}")

    label_counts = dict(zip(*np.unique(encoded_labels, return_counts=True)))
    logger.info(f"Label distribution: {label_counts}")

    indices = np.arange(len(npz_files))

    train_indices, temp_indices, train_labels, temp_labels = train_test_split(
        indices, encoded_labels, train_size=train_size, stratify=encoded_labels, shuffle=True
    )

    val_relative_size = val_size / (val_size + test_size)
    val_indices, test_indices, val_labels, test_labels = train_test_split(
        temp_indices, temp_labels, test_size=1 - val_relative_size, stratify=temp_labels, shuffle=True
    )

    logger.info(
        f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}, Test samples: {len(test_indices)}")

    return {
        'train': {
            'files': [npz_files[i] for i in train_indices],
            'labels': train_labels,
            'label_mapping': label_to_int
        },
        'val': {
            'files': [npz_files[i] for i in val_indices],
            'labels': val_labels,
            'label_mapping': label_to_int
        },
        'test': {
            'files': [npz_files[i] for i in test_indices],
            'labels': test_labels,
            'label_mapping': label_to_int
        },
        'dataset_info': {
            'total_samples': len(npz_files),
            'num_classes': len(label_to_int),
            'label_names': list(label_to_int.keys()),
            'label_mapping': label_to_int,
            'label_distribution': label_counts
        }
    }
