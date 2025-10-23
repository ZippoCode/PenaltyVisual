import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from penalty_vision.config.training_config import get_training_config
from penalty_vision.dataset.dataloaders import create_dataloaders_from_fold
from penalty_vision.dataset.dataset_utils import create_kfold_splits
from penalty_vision.models.losses import get_loss_function
from penalty_vision.models.metrics import MetricsCalculator
from penalty_vision.models.optimizer import get_optimizer, get_scheduler
from penalty_vision.models.two_stream_lstm import TwoStreamLSTM
from penalty_vision.training.trainer import Trainer
from penalty_vision.utils.logger import logger
from penalty_vision.utils.seed import set_seed
from penalty_vision.utils.wandb_logger import WandBLogger

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._generate_schema")


def train_single_fold(fold_idx, fold_data, config, device):
    logger.info(f"\n{'=' * 60}")
    logger.info(f"FOLD {fold_idx}")
    logger.info(f"{'=' * 60}")

    set_seed(config.seed + fold_idx)

    train_loader, val_loader, test_loader, dataset_info = create_dataloaders_from_fold(
        fold_data=fold_data,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers
    )

    wandb_logger = WandBLogger(
        experiment_name=f"{config.experiment_name}_fold{fold_idx}",
        config={
            "fold": fold_idx,
            "learning_rate": config.training.learning_rate,
            "batch_size": config.data.batch_size,
            "num_epochs": config.training.num_epochs,
            "optimizer": config.training.optimizer,
            "scheduler": config.training.scheduler,
            "hidden_size": config.model.hidden_size,
            "dropout": config.model.dropout,
            "seed": config.seed + fold_idx,
            "train_samples": dataset_info['train_samples'],
            "val_samples": dataset_info['val_samples'],
            "test_samples": dataset_info['test_samples'],
            "num_classes": dataset_info['num_classes']
        }
    )

    model = TwoStreamLSTM(
        input_size=config.model.input_size,
        hidden_size=config.model.hidden_size,
        num_classes=config.model.num_classes,
        dropout=config.model.dropout,
        metadata_size=config.model.metadata_size
    )
    model = model.to(device)

    wandb_logger.watch_model(model)

    criterion = get_loss_function(
        num_classes=config.model.num_classes,
        labels=train_loader.dataset.labels,
        device=device
    )

    optimizer = get_optimizer(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    scheduler = get_scheduler(
        optimizer=optimizer,
        patience=config.training.scheduler_patience,
        factor=config.training.scheduler_factor
    )

    metrics_calculator = MetricsCalculator(
        num_classes=config.model.num_classes,
        class_names=list(dataset_info['label_mapping'].keys()),
        device=device
    )

    fold_checkpoint_dir = Path(config.checkpoint_dir) / f"fold_{fold_idx}"
    fold_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics_calculator=metrics_calculator,
        device=device,
        checkpoint_dir=str(fold_checkpoint_dir),
        gradient_clip_val=config.training.gradient_clip_val,
        mixed_precision=config.training.mixed_precision,
        wandb_logger=wandb_logger
    )

    logger.info(
        f"Train: {dataset_info['train_samples']}, Val: {dataset_info['val_samples']}, Test: {dataset_info['test_samples']}")

    best_val_accuracy = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.training.num_epochs,
        early_stopping_patience=config.training.early_stopping_patience
    )

    best_checkpoint = fold_checkpoint_dir / "best_model.pt"
    if best_checkpoint.exists():
        model.load_state_dict(torch.load(best_checkpoint, map_location=device, weights_only=False))

    test_metrics = trainer.evaluate(test_loader, phase='test')

    wandb_logger.finish()

    results = {
        'fold': fold_idx,
        'best_val_accuracy': best_val_accuracy,
        'test_accuracy': test_metrics['accuracy'],
        'test_precision_macro': test_metrics['precision_macro'],
        'test_precision_weighted': test_metrics['precision_weighted'],
        'test_recall_macro': test_metrics['recall_macro'],
        'test_recall_weighted': test_metrics['recall_weighted'],
        'test_f1_macro': test_metrics['f1_macro'],
        'test_f1_weighted': test_metrics['f1_weighted']
    }

    logger.info(
        f"Fold {fold_idx}: Val Acc={best_val_accuracy:.4f}, Test Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1_macro']:.4f}")

    return results


def run_kfold_cross_validation(config, device):
    logger.info(f"\n{'=' * 60}")
    logger.info("K-FOLD CROSS-VALIDATION")
    logger.info(f"{'=' * 60}")
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Folds: {config.data.n_folds}")
    logger.info(f"Epochs: {config.training.num_epochs}")
    logger.info(f"Batch size: {config.data.batch_size}")

    folds = create_kfold_splits(
        data_dir=str(config.data.data_dir),
        n_folds=config.data.n_folds
    )

    all_results = []

    for fold_idx in range(config.data.n_folds):
        fold_results = train_single_fold(fold_idx, folds[fold_idx], config, device)
        all_results.append(fold_results)

    logger.info(f"\n{'=' * 60}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 60}")

    val_accs = [r['best_val_accuracy'] for r in all_results]
    test_accs = [r['test_accuracy'] for r in all_results]
    test_precs = [r['test_precision'] for r in all_results]
    test_recs = [r['test_recall'] for r in all_results]
    test_f1s = [r['test_f1'] for r in all_results]

    summary = {
        'experiment_name': config.experiment_name,
        'n_folds': config.data.n_folds,
        'fold_results': all_results,
        'mean_val_accuracy': float(np.mean(val_accs)),
        'std_val_accuracy': float(np.std(val_accs)),
        'mean_test_accuracy': float(np.mean(test_accs)),
        'std_test_accuracy': float(np.std(test_accs)),
        'mean_test_precision': float(np.mean(test_precs)),
        'std_test_precision': float(np.std(test_precs)),
        'mean_test_recall': float(np.mean(test_recs)),
        'std_test_recall': float(np.std(test_recs)),
        'mean_test_f1': float(np.mean(test_f1s)),
        'std_test_f1': float(np.std(test_f1s))
    }

    logger.info(f"\nVal Accuracy: {summary['mean_val_accuracy']:.4f} ± {summary['std_val_accuracy']:.4f}")
    logger.info(f"Test Accuracy: {summary['mean_test_accuracy']:.4f} ± {summary['std_test_accuracy']:.4f}")
    logger.info(f"Test Precision: {summary['mean_test_precision']:.4f} ± {summary['std_test_precision']:.4f}")
    logger.info(f"Test Recall: {summary['mean_test_recall']:.4f} ± {summary['std_test_recall']:.4f}")
    logger.info(f"Test F1: {summary['mean_test_f1']:.4f} ± {summary['std_test_f1']:.4f}")

    logger.info(f"\nPer-fold:")
    for r in all_results:
        logger.info(
            f"  Fold {r['fold']}: Val={r['best_val_accuracy']:.4f}, Test={r['test_accuracy']:.4f}, F1={r['test_f1']:.4f}")

    results_dir = Path(config.checkpoint_dir) / "kfold_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "cv_results.json"

    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nResults saved: {results_file}")
    logger.info(f"{'=' * 60}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='K-Fold Cross-Validation Training')
    parser.add_argument('--config', type=Path, required=True, help='Config YAML file')
    args = parser.parse_args()

    config = get_training_config(args.config)
    device = torch.device(config.device)

    logger.info(f"Device: {device}")
    set_seed(config.seed)

    summary = run_kfold_cross_validation(config, device)

    logger.info(
        f"\nCompleted! Final Test Accuracy: {summary['mean_test_accuracy']:.4f} ± {summary['std_test_accuracy']:.4f}")


if __name__ == '__main__':
    main()
