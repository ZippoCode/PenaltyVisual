from typing import Dict, Tuple

import torch
from penalty_vision.dataset.dataset_utils import create_stratified_split, create_kfold_splits
from penalty_vision.dataset.encoders import encode_metadata
from penalty_vision.dataset.penalty_kick_dataset import PenaltyKickDataset
from penalty_vision.utils.logger import logger
from torch.utils.data import DataLoader


def collate_fn(batch):
    valid_batch = []

    for sample, label in batch:
        if sample.running_embeddings.size > 0 and sample.kicking_embeddings.size > 0:
            valid_batch.append((sample, label))

    if len(valid_batch) == 0:
        return None

    samples, labels = zip(*valid_batch)

    running_embeddings = torch.stack([torch.from_numpy(s.running_embeddings) for s in samples])
    kicking_embeddings = torch.stack([torch.from_numpy(s.kicking_embeddings) for s in samples])
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
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    logger.info(f"Creating dataloaders with batch_size={batch_size}, num_workers={num_workers}")

    splits = create_stratified_split(data_dir, train_size, val_size, test_size)

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


def create_dataloaders_from_fold(
        fold_data: Dict,
        batch_size: int = 32,
        num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:

    train_dataset = PenaltyKickDataset(fold_data['train']['files'], fold_data['train']['labels'])
    val_dataset = PenaltyKickDataset(fold_data['val']['files'], fold_data['val']['labels'])
    test_dataset = PenaltyKickDataset(fold_data['test']['files'], fold_data['test']['labels'])

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

    info = {
        'fold': fold_data['fold'],
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
        'label_mapping': fold_data['label_mapping'],
        'num_classes': len(fold_data['label_mapping'])
    }

    return train_loader, val_loader, test_loader, info
