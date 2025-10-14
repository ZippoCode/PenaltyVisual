from typing import Tuple

import torch
from torch.utils.data import DataLoader

from penalty_vision.dataset.dataset_utils import create_stratified_split, encode_metadata
from penalty_vision.dataset.penalty_kick_dataset import PenaltyKickDataset


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


def create_dataloaders(
        data_dir: str,
        batch_size: int = 32,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    splits = create_stratified_split(data_dir, train_size, val_size, test_size)

    train_dataset = PenaltyKickDataset(splits['train'])
    val_dataset = PenaltyKickDataset(splits['val'])
    test_dataset = PenaltyKickDataset(splits['test'])

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

    return train_loader, val_loader, test_loader
