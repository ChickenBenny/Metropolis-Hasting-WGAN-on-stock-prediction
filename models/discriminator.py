import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=5, stride=1, padding='same'),
            nn.LeakyReLU(0.01),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding='same'),
            nn.LeakyReLU(0.01),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding='same'),
            nn.LeakyReLU(0.01)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128, 220),
            nn.LeakyReLU(0.01),
            nn.Linear(220, 220),
            nn.ReLU(),
            nn.Linear(220, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        out = self.fc_layers(x)
        return out
    