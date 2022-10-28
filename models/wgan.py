import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from .generator import Generator
from .discriminator import Discriminator

class WGAN(nn.Module):
    def __init__(self, input_size, use_cuda):
        super().__init__()
        self.input_size = input_size
        self.device = torch.device("cuda" if (torch.cuda.is_available() & use_cuda) else "cpu")
        self.gen = Generator(self.input_size, use_cuda)
        self.dis = Discriminator()

    def forward(self, z):
        return self.gen(z)

    def discriminator_loss(self, y, y_hat):
        return -(torch.mean(y) - torch.mean(y_hat))

    def generator_loss(self, y_hat):
        return -torch.mean(y_hat)

    def optimizers(self, lr):
        self.opt_G = torch.optim.Adam(self.gen.parameters(), lr = lr, betas = (0.0, 0.9), weight_decay = 1e-3)
        self.opt_D = torch.optim.Adam(self.dis.parameters(), lr = lr, betas = (0.0, 0.9), weight_decay = 1e-3)

    def training_step(self, epochs, train_dataloader, real_tick):
        self.gen.train()
        self.dis.train()
        hist_G = np.zeros(epochs)
        hist_D = np.zeros(epochs)
        for epoch in range(epochs):
            loss_G = []
            loss_D = []
            for (x, y) in train_dataloader:
                x = x.to(self.device)
                y = y.to(self.device)

                fake_data = self.gen(x)
                fake_data = torch.cat([y[:, :real_tick, :], fake_data.reshape(-1, 1, 1)], axis = 1)

                cirtic_real = self.dis(y)
                critic_fake = self.dis(fake_data)

                loss_d = self.discriminator_loss(cirtic_real, critic_fake)
                self.dis.zero_grad()
                loss_d.backward(retain_graph = True)
                self.opt_D.step()

                output_fake = self.dis(fake_data)
                loss_g = self.generator_loss(output_fake)
                self.gen.zero_grad()
                loss_g.backward()
                self.opt_G.step()

                loss_G.append(loss_g.item())
                loss_D.append(loss_d.item())
            hist_G[epoch] = sum(loss_G)
            hist_D[epoch] = sum(loss_D)
            print(f'[{epoch + 1}/{epochs}] LossD: {sum(loss_D):.5f} LossG:{sum(loss_G):.5f}')
        self.plot(hist_G, hist_D)

    def generator_samples(self, x):
        self.gen.eval()
        pred_y = self.gen(x.to(self.device))
        return pred_y.cpu()

    def score_sample(self, x):
        self.dis.eval()
        score = self.dis(x.to(self.device))
        return score.cpu()

    def plot(self, histG, histD):
        plt.figure(figsize = (12, 6))
        plt.plot(histG, color = 'blue', label = 'Generator Loss')
        plt.plot(histD, color = 'black', label = 'Discriminator Loss')
        plt.title('WGAN-GP Loss')
        plt.xlabel('Days')
        plt.legend(loc = 'upper right')
