import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_size, use_cuda):
        super().__init__()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.gru_layers = nn.Sequential(
            nn.GRU(input_size=input_size, hidden_size=1024, num_layers=1, batch_first=True),
            nn.Dropout(0.2),
            nn.GRU(input_size=1024, hidden_size=512, num_layers=1, batch_first=True),
            nn.Dropout(0.2),
            nn.GRU(input_size=512, hidden_size=256, num_layers=1, batch_first=True),
            nn.Dropout(0.2)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 1024).to(self.device)
        out, _ = self.gru_layers(x, h0)
        out = self.linear_layers(out[:, -1, :])
        return out
    