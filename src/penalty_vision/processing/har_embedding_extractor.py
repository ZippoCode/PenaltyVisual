from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights


class HAREmbeddingExtractor:
    def __init__(self, num_frames: int = 16, stride: int = 8, batch_size: int = 8, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.weights = MViT_V2_S_Weights.KINETICS400_V1
        self.model = self._load_model()
        self.transform = self.weights.transforms()
        self.num_frames = num_frames
        self.stride = stride
        self.batch_size = batch_size

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

    def _prepare_clip_tensor(self, frames: np.ndarray) -> torch.Tensor:
        video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0)
        video_tensor = video_tensor.float() / 255.0
        video_tensor = self.transform(video_tensor)
        return video_tensor

    def _process_clips_batch(self, clips: List[np.ndarray]) -> torch.Tensor:
        batch_tensors = []

        for clip in clips:
            tensor = self._prepare_clip_tensor(clip)
            batch_tensors.append(tensor)

        batch_tensor = torch.cat(batch_tensors, dim=0).to(self.device)

        with torch.no_grad():
            features = self.model(batch_tensor)

        return features

    def extract_embeddings(self, frames: np.ndarray) -> torch.Tensor:
        n_available = len(frames)

        if n_available < self.num_frames:
            frames = self._resample_frames(frames)
            return self._process_clips_batch([frames]).cpu()

        if n_available == self.num_frames:
            return self._process_clips_batch([frames]).cpu()

        clips = []
        for start_idx in range(0, n_available - self.num_frames + 1, self.stride):
            end_idx = start_idx + self.num_frames
            clip = frames[start_idx:end_idx]
            clips.append(clip)

        all_embeddings = []

        for i in range(0, len(clips), self.batch_size):
            batch_clips = clips[i:i + self.batch_size]
            batch_embeddings = self._process_clips_batch(batch_clips)
            all_embeddings.append(batch_embeddings)

        all_embeddings = torch.cat(all_embeddings, dim=0)
        final_embedding = torch.mean(all_embeddings, dim=0, keepdim=True)

        return final_embedding.cpu()

    def process_penalty_kick(self, running_frames: np.ndarray, kicking_frames: np.ndarray) -> Dict[str, torch.Tensor]:
        running_embedding = self.extract_embeddings(running_frames)
        kicking_embedding = self.extract_embeddings(kicking_frames)

        return {
            "running_embedding": running_embedding,
            "kicking_embedding": kicking_embedding
        }
