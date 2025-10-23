from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal

import torch


@dataclass
class ModelConfig:
    input_size: int = 768
    num_classes: int = 3
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    metadata_size: int = 270


@dataclass
class DataConfig:
    data_dir: Path = field(default_factory=Path)
    label_field: Literal["side", "direction", "outcome"] = "side"
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    n_folds: int = 10

@dataclass
class TrainingConfig:
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    optimizer: Literal["adam", "sgd", "adamw"] = "adam"
    scheduler: Optional[Literal["step", "cosine", "reduce"]] = "reduce"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 20
    gradient_clip_val: Optional[float] = 1.0
    mixed_precision: bool = False


@dataclass
class LossConfig:
    use_class_weights: bool = True
    focal_loss: bool = False
    focal_gamma: float = 2.0
    focal_alpha: Optional[float] = None
    label_smoothing: float = 0.1


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    checkpoint_dir: Path = field(default_factory=Path)
    log_dir: Path = field(default_factory=Path)
    experiment_name: str = "two_stream_lstm"
    seed: int = 42
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    save_best_only: bool = True
    save_every_n_epochs: Optional[int] = None
    resume_from_checkpoint: Optional[Path] = None
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    debug: bool = False


def get_training_config(config_path: Optional[Path] = None) -> Config:
    if config_path and config_path.exists():
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        config = Config()

        if 'model' in config_dict:
            config.model = ModelConfig(**config_dict['model'])
        if 'data' in config_dict:
            config.data = DataConfig(**{k: Path(v) if k.endswith('_dir') else v
                                        for k, v in config_dict['data'].items()})
        if 'training' in config_dict:
            config.training = TrainingConfig(**config_dict['training'])
        if 'loss' in config_dict:
            config.loss = LossConfig(**config_dict['loss'])

        for k, v in config_dict.items():
            if k not in ['model', 'data', 'training', 'loss']:
                if k.endswith('_dir') or k.endswith('_path'):
                    setattr(config, k, Path(v) if v else None)
                else:
                    setattr(config, k, v)

        return config

    return Config()
