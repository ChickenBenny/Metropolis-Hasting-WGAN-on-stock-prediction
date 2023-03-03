import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, config, latent_dim):
        super().__init__()

        modules = []
        prev_dim = latent_dim

        for dim in config:
            modules.append(
                nn.Sequential(
                    nn.Linear(prev_dim, dim),
                    nn.ReLU()
                )
            )
            prev_dim = dim

        self.decoder_input = nn.Linear(latent_dim, config[-1])
        self.decoder = nn.Sequential(*modules)

    def forward(self, x):
        x = self.decoder_input(x)
        decoded = self.decoder(x)
        return decoded