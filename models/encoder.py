import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, config, latent_dim):
        super().__init__()
        layers = []
        for i in range(len(config)-1):
            layers.append(nn.Linear(config[i], config[i+1]))
            layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(config[-1], latent_dim)
        self.fc_var = nn.Linear(config[-1], latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
