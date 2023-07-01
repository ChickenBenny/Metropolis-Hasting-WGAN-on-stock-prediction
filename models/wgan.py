import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from .generator import Generator
from .discriminator import Discriminator


class WGAN(nn.Module):
    def __init__(self, input_size, seq_length, use_cuda=1):
        super().__init__()
        self.input_size = input_size
        self.device = torch.device("cuda" if (torch.cuda.is_available() and use_cuda) else "cpu")
        self.gen = Generator(self.input_size, use_cuda)
        self.dis = Discriminator(seq_length)

    def forward(self, z):
        return self.gen(z)

    def discriminator_loss(self, y, y_hat):
        return -(torch.mean(y) - torch.mean(y_hat))

    def generator_loss(self, y_hat):
        return -torch.mean(y_hat)

    def optimizers(self, lr):
        self.opt_G = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.0, 0.9), weight_decay=1e-3)
        self.opt_D = torch.optim.Adam(self.dis.parameters(), lr=lr, betas=(0.0, 0.9), weight_decay=1e-3)
