import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .encoder import Encoder
from .decoder import Decoder

class VAE(nn.Module):
    def __init__(self, config, latent_dim, use_cuda=1):
        super().__init__()
        self.encoder = Encoder(config, latent_dim)
        self.decoder = Decoder(config, latent_dim)
        self.device = torch.device("cuda" if (torch.cuda.is_available() and use_cuda) else "cpu")

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        output = self.decoder(z)
        return output, z, mu, log_var

    def kl_divergence(self, mu, log_var):
        return 0.5 * torch.sum(-1 - log_var + mu.pow(2) + log_var.exp())

    def criterion(self, x_hat, x, mu, log_var):
        return F.binary_cross_entropy(x_hat, x) + self.kl_divergence(mu, log_var)
    