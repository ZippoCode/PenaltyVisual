from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Config:

    def __init__(self, config_path: str):
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        self.frame_dir = config.get("frame_dir")
        self.video_dir = config.get("video_dir")
        self.checkpoint_path = config.get("checkpoint_path")
        self.tracker_config = config.get("tracker_config")
