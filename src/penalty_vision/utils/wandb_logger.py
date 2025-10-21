import os
from typing import Dict, Any, Optional

import wandb
from dotenv import load_dotenv
from penalty_vision.utils.logger import logger


class WandBLogger:

    def __init__(self, experiment_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        load_dotenv()
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key:
            wandb.login(key=wandb_api_key)

        self.project = os.getenv('WANDB_PROJECT')
        self.entity = os.getenv('WANDB_ENTITY')
        print(self.project, self.entity)
        self.enabled = self.project is not None and self.entity is not None

        if self.enabled:
            wandb.init(
                project=self.project,
                entity=self.entity,
                name=experiment_name,
                config=config
            )
            logger.info(f"WandB logging enabled - Project: {self.project}, Entity: {self.entity}")
        else:
            logger.warning("WandB logging disabled - WANDB_PROJECT or WANDB_ENTITY not found in environment")

    def watch_model(self, model, log: str = "all", log_freq: int = 100):
        if self.enabled:
            wandb.watch(model, log=log, log_freq=log_freq)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        if self.enabled:
            wandb.log(metrics, step=step)

    def log_summary(self, summary: Dict[str, Any]):
        if self.enabled:
            for key, value in summary.items():
                wandb.run.summary[key] = value

    def finish(self):
        if self.enabled:
            wandb.finish()
