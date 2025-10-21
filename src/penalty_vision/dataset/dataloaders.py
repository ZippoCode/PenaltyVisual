from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from penalty_vision.dataset.encoders import encode_metadata
from penalty_vision.dataset.penalty_kick_dataset import PenaltyKickDataset
from penalty_vision.utils.logger import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def collate_fn(batch):
    valid_batch = []

    for sample, label in batch:
        if sample.running_frames.size > 0 and sample.kicking_frames.size > 0:
            valid_batch.append((sample, label))

    if len(valid_batch) == 0:
        return None

    samples, labels = zip(*valid_batch)

    running_embeddings = torch.stack([torch.from_numpy(s.running_frames) for s in samples])
    kicking_embeddings = torch.stack([torch.from_numpy(s.kicking_frames) for s in samples])
    metadata = torch.stack([torch.from_numpy(encode_metadata(s.metadata)) for s in samples])
    labels = torch.tensor(labels, dtype=torch.long)

    return (running_embeddings, kicking_embeddings, metadata), labels


def create_stratified_split(data_dir: str, label_field: Union[str, List[str]], train_size: float = 0.8,
                            val_size: float = 0.1, test_size: float = 0.1) -> Dict[str, Dict[str, List]]:
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError(f"Splits must sum to 1.0, got {train_size + val_size + test_size}")

    data_dir = Path(data_dir)
    npz_files = sorted(list(data_dir.glob("*.npz")))

    if not npz_files:
        raise ValueError(f"No .npz files found in {data_dir}")

    logger.info(f"Found {len(npz_files)} .npz files in {data_dir}")

    raw_labels = []
    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)
        metadata = data['metadata'].item()

        if isinstance(label_field, list):
            label = "_".join(str(metadata[field]) for field in label_field)
        else:
            label = metadata[label_field]

        raw_labels.append(label)

    unique_labels = sorted(set(raw_labels))
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    logger.info(f"Unique labels found: {unique_labels}")
    logger.info(f"Label encoding: {label_to_int}")

    encoded_labels = [label_to_int[label] for label in raw_labels]

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
            'num_classes': len(unique_labels),
            'label_names': unique_labels,
            'label_mapping': label_to_int,
            'label_distribution': label_counts
        }
    }


def create_dataloaders(data_dir: str, label_field: str, batch_size: int = 32, train_size: float = 0.8,
                       val_size: float = 0.1, test_size: float = 0.1, num_workers: int = 0) -> Tuple[
    DataLoader, DataLoader, DataLoader, Dict]:
    logger.info(f"Creating dataloaders with batch_size={batch_size}, num_workers={num_workers}")

    splits = create_stratified_split(data_dir, label_field, train_size, val_size, test_size)

    train_dataset = PenaltyKickDataset(splits['train']['files'], splits['train']['labels'])
    val_dataset = PenaltyKickDataset(splits['val']['files'], splits['val']['labels'])
    test_dataset = PenaltyKickDataset(splits['test']['files'], splits['test']['labels'])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )

    logger.info("Dataloaders created successfully")

    dataset_info = splits['dataset_info']
    dataset_info['train_samples'] = len(train_dataset)
    dataset_info['val_samples'] = len(val_dataset)
    dataset_info['test_samples'] = len(test_dataset)

    return train_loader, val_loader, test_loader, dataset_info
