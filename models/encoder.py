import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, config, latent_dim):
        super().__init__()

        modules = []
        for i in range(1, len(config)):
            modules.append(
                nn.Sequential(
                    nn.Linear(config[i - 1], config[i]),
                    nn.ReLU()
                )
            )
        
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(config[-1], latent_dim)
        self.fc_var = nn.Linear(config[-1], latent_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logVar = self.fc_var(encoded)
        return mu, logVar

