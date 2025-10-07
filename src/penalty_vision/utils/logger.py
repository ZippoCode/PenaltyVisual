import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


class Logger:
    def __init__(self, name: str, log_file: Optional[str] = None, max_bytes: int = 10485760, backup_count: int = 5,
                 console_level: int = logging.INFO, file_level: int = logging.DEBUG):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8')
            file_handler.setLevel(file_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def get_logger(self):
        return self.logger


logger = Logger("PenaltyVision").get_logger()
