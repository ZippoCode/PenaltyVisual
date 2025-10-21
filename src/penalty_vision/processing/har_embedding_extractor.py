from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights


class HAREmbeddingExtractor:
    def __init__(self, num_frames: int = 16, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.weights = MViT_V2_S_Weights.KINETICS400_V1
        self.model = self._load_model()
        self.transform = self.weights.transforms()
        self.num_frames = num_frames

    def _load_model(self) -> nn.Module:
        model = mvit_v2_s(weights=self.weights)
        model.head = nn.Identity()
        model = model.to(self.device)
        model.eval()
        return model

    def _resample_frames(self, frames: np.ndarray) -> np.ndarray:
        n_available = len(frames)
        if n_available == self.num_frames:
            return frames
        indices = np.linspace(0, n_available - 1, self.num_frames, dtype=int)
        return frames[indices]

    def extract_embeddings(self, frames: np.ndarray) -> torch.Tensor:
        frames = self._resample_frames(frames)
        video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0)
        video_tensor = video_tensor.float() / 255.0
        video_tensor = self.transform(video_tensor).to(self.device)

        with torch.no_grad():
            features = self.model(video_tensor)

        return features.cpu()

    def process_penalty_kick(self, running_frames: np.ndarray, kicking_frames: np.ndarray) -> Dict[str, torch.Tensor]:
        return {
            "running_embedding": self.extract_embeddings(running_frames),
            "kicking_embedding": self.extract_embeddings(kicking_frames)
        }
