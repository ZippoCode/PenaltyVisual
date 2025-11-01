import torch
import torch.nn as nn


class PenaltyKickClassifier(nn.Module):

    def __init__(
            self,
            embedding_size,
            hidden_size=128,
            num_classes=3,
            dropout=0.3,
            metadata_size=4,
            metadata_dense=6
    ):
        super(PenaltyKickClassifier, self).__init__()
        self.metadata_dense = nn.Linear(metadata_size, metadata_dense)

        self.kicking_branch = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )

        self.running_branch = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )

        combined_size = hidden_size + hidden_size + metadata_dense

        self.classifier = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, num_classes)
        )

    def forward(self, running_embeddings, kicking_embeddings, metadata):
        if running_embeddings.dim() == 3:
            running_embeddings = running_embeddings.squeeze(1)
        if kicking_embeddings.dim() == 3:
            kicking_embeddings = kicking_embeddings.squeeze(1)

        kicking_features = self.kicking_branch(kicking_embeddings)
        running_features = self.running_branch(running_embeddings)
        metadata_features = self.metadata_dense(metadata)

        combined = torch.cat([
            running_features,
            kicking_features,
            metadata_features
        ], dim=1)

        output = self.classifier(combined)

        return output
