import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, config, latent_dim):
        super().__init__()

        modules = []
        self.decoder_input = nn.Linear(latent_dim, config[-1])

        for i in range(len(config) - 1, 1, -1):
            modules.append(
                nn.Sequential(
                    nn.Linear(config[i], config[i - 1]),
                    nn.ReLU()
                )
            )
        modules.append(
            nn.Sequential(
                nn.Linear(config[1], config[0]),
                nn.Sigmoid()
            )
        )
        self.decoder = nn.Sequential(*modules)

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded
    