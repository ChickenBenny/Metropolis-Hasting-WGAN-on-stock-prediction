import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Encoder
from .decoder import Decoder

class VAE(nn.Module):
    def __init__(self, config, latent_dim):
        super().__init__()
        self.encoder = Encoder(config, latent_dim)
        self.decoder = Decoder(config, latent_dim)

    def reparameterize(self, mu, logVar):
        std = torch.exp(0.5* logVar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        output = self.decoder(z)
        return output, z, mu, logVar

    def kl_divergence(self, mu, logVar):
        return 0.5* torch.sum(-1 -logVar + mu.pow(2) + logVar.exp())

    def vae_loss(self, x_hat, x, mu, logVar):
        return F.binary_cross_entropy(x_hat, x) + self.kl_divergence(mu, logVar)