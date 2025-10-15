import torch
import torch.nn as nn


class TwoStreamLSTM(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size=128,
            num_classes=3,
            dropout=0.3,
            metadata_size=4,
            fc1_size=256,
            fc2_size=128
    ):
        super(TwoStreamLSTM, self).__init__()
        self.lstm_run = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm_kick = nn.LSTM(input_size, hidden_size, batch_first=True)

        combined_size = hidden_size * 2 + metadata_size
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, fc1_size),
            nn.LayerNorm(fc1_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc1_size, fc2_size),
            nn.LayerNorm(fc2_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc2_size, num_classes)
        )

    def forward(self, x_run, x_kick, metadata):
        _, (h_run, _) = self.lstm_run(x_run)
        _, (h_kick, _) = self.lstm_kick(x_kick)

        combined = torch.cat([h_run[-1], h_kick[-1], metadata], dim=1)
        output = self.classifier(combined)

        return output
