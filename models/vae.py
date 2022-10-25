import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .encoder import Encoder
from .decoder import Decoder

class VAE_Network(nn.Module):
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

class VAE(nn.Module):
    def __init__(self, config, latent_dim, use_cuda):
        super().__init__()
        self.vae = VAE_Network(config, latent_dim)
        self.device = torch.device("cuda" if (torch.cuda.is_available() & use_cuda) else "cpu")

    def kl_divergence(self, mu, logVar):
        return 0.5* torch.sum(-1 -logVar + mu.pow(2) + logVar.exp())

    def vae_loss(self, x_hat, x, mu, logVar):
        return F.binary_cross_entropy(x_hat, x) + self.kl_divergence(mu, logVar)

    def optimizers(self, lr):
        self.opt = torch.optim.Adam(self.vae.parameters(), lr = lr)

    def training_step(self, epochs, train_dataloader, real_tick):
        self.vae.train()
        hist = np.zeros(epochs)
        for epoch in range(epochs):
            loss_VAE = []
            for (x,) in train_dataloader:
                x = x.to(self.device)
                x_hat, z, mu, logVar = self.vae(x)
                loss = self.vae_loss(x_hat, x, mu, logVar)
                loss.backward()
                self.opt.step()
                loss_VAE.append(loss.item())
            hist[epoch] = sum(loss_VAE)
            print(f'[{epoch + 1}/{epochs}] LossD: {sum(loss_VAE):.5f}')

    def predict(self, x):
        self.vae.eval()
        x_hat, z, mu, logVar = self.vae(x.to(self.device))
        return x_hat.cpu().detach().numpy(), z.cpu().detach().numpy()