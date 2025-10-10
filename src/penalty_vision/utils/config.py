from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any
import yaml


@dataclass
class ModelConfig:
    weights: str = "yolo11n.pt"
    fallback: str = "yolo11n.pt"
    imgsz: int = 640


@dataclass
class DetectionConfig:
    confidence: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 300


@dataclass
class TrackingConfig:
    tracker: str = "bytetrack.yaml"
    persist: bool = True


@dataclass
class ClassConfig:
    kicker: int = 0
    ball: int = 1


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch: int = 16
    patience: int = 15
    save_period: int = 10


@dataclass
class AugmentationConfig:
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0


@dataclass
class PathConfig:
    data_yaml: str = ""
    runs_dir: str = "runs"
    frame_dir: str = ""
    video_dir: str = ""


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    classes: ClassConfig = field(default_factory=ClassConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        config_file = Path(config_path)
        
        if not config_file.exists():
            example_file = Path(str(config_file) + ".example")
            if example_file.exists():
                config_file = example_file
            else:
                raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**data.get('model', {})),
            detection=DetectionConfig(**data.get('detection', {})),
            tracking=TrackingConfig(**data.get('tracking', {})),
            classes=ClassConfig(**data.get('classes', {})),
            training=TrainingConfig(**data.get('training', {})),
            augmentation=AugmentationConfig(**data.get('augmentation', {})),
            paths=PathConfig(**data.get('paths', {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)