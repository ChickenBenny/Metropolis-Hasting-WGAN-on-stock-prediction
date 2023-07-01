from pickle import FALSE
import torch
import random
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.isotonic import IsotonicRegression
from .wgan import WGAN

class MHWGAN(WGAN):
    def __init__(self, input_size, seq_length, use_cuda=1):
        super().__init__(input_size, seq_length, use_cuda)
        self.seq_length = seq_length
        self.calibrator = None

    def generator_samples(self, train_x):
        self.gen.eval()
        with torch.no_grad():
            train_x_tensor = torch.from_numpy(train_x).to(self.device).float()
            mh_sample = self.gen(train_x_tensor)
        return mh_sample

    def score_sample(self, mh_sample_cat):
        self.dis.eval()
        with torch.no_grad():
            mh_sample_dis = self.dis(mh_sample_cat.to(self.device))
        return mh_sample_dis

    def calibrate_discriminator(self, train_x, train_y):
        self.calibrator = IsotonicRegression()
        mh_sample_predict = self.generator_samples(train_x)

        train_y_gpu = torch.from_numpy(train_y[:, :self.seq_length, :]).to(self.device)
        mh_sample_cat = torch.cat([train_y_gpu, mh_sample_predict.reshape(-1, 1, 1)], axis=1)

        mh_sample_dis = self.score_sample(mh_sample_cat.float())
        self.calibrator.fit(mh_sample_dis.cpu().detach().numpy(), train_y[:, self.seq_length, :].reshape(-1))
        mh_sample_cal = self.calibrator.predict(mh_sample_dis.cpu().detach().numpy())
        return mh_sample_cal

    def num_sampler(self, n_samples, x_shape):
        random_pick = []
        for _ in range(n_samples):
            random_pick.append(np.random.randint(0, x_shape - 1))
        return random_pick

    def mh_sample(self, train_x, train_y, mh_sample_cal):
        y_score_base = mh_sample_cal[0]

        x_samples = []
        y_samples = []

        while len(x_samples) < 2000:
            idx = self.num_sampler(1, train_x.shape[0])

            x_sample = train_x[idx, :, :]
            y_sample = train_y[idx, :, :]

            pred_x = self.generator_samples(x_sample)
            x_smaple_cat = torch.cat([torch.from_numpy(y_sample[:, :self.seq_length, :]).float().to(self.device),
                                      pred_x.reshape(-1, 1, 1)], axis=1)
            x_dis_score = self.score_sample(x_smaple_cat)
            x_score = self.calibrator.transform(x_dis_score.cpu().detach().numpy())

            u = np.random.uniform(0, 1, (1,))[0]
            if u <= np.fmin(1., (1. / y_score_base - 1.) / (1. / x_score - 1.)):
                y_score_base = x_score
                x_samples.append(x_sample)
                y_samples.append(y_sample)

        return np.array(x_samples).reshape(-1, train_x.shape[1], train_x.shape[2]), np.array(y_samples).reshape(-1, train_y.shape[1], train_y.shape[2])


    def check_type(self, type):
        if type == 'mh':
            return True
        else:
            return False

    def training_model(self, epochs, type, batch_size, train_x, train_y, seq_length):
        history_G = np.zeros(epochs)
        history_D = np.zeros(epochs)
        for epoch in range(epochs):
            total_loss_G = 0
            total_loss_D = 0

            if self.check_type(type):
                mh_sample_cal = self.calibrate_discriminator(train_x, train_y)
                train_x, train_y = self.mh_sample(train_x, train_y, mh_sample_cal)

            dataset = DataLoader(TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float()),
                                 batch_size=batch_size, shuffle=False)

            self.dis.train()
            self.gen.train()
            for x, y in dataset:
                x = x.to(self.device)
                y = y.to(self.device)
                fake_data = self.gen(x)
                fake_data = torch.cat([y[:, :seq_length, :], fake_data.reshape(-1, 1, 1)], axis=1)

                cirtic_real = self.dis(y)
                critic_fake = self.dis(fake_data)

                loss_d = self.discriminator_loss(cirtic_real, critic_fake)
                self.dis.zero_grad()
                loss_d.backward(retain_graph=True)
                self.opt_D.step()

                output_fake = self.dis(fake_data)
                loss_g = self.generator_loss(output_fake)
                self.gen.zero_grad()
                loss_g.backward()
                self.opt_G.step()

                total_loss_G += loss_g.item()
                total_loss_D += loss_d.item()
            history_G[epoch] = total_loss_G
            history_D[epoch] = total_loss_D
            print(f'[{epoch + 1}/{epochs}] LossD: {total_loss_D:.5f} LossG: {total_loss_G:.5f}')
        return history_G, history_D

    def mh_training(self, epochs_wgan, epochs_mhgan, batch_size, train_x, train_y, seq_length, plot_loss=False):
        hist_G = np.zeros(epochs_wgan + epochs_mhgan)
        hist_D = np.zeros(epochs_wgan + epochs_mhgan)

        print(f'Training Basic for {epochs_wgan} epochs')
        wgan_loss_G, wgan_loss_D = self.training_model(epochs_wgan, 'wgan', batch_size, train_x, train_y, seq_length)

        for i in range(epochs_wgan):
            hist_G[i] = wgan_loss_G[i]
            hist_D[i] = wgan_loss_D[i]

        print(f'Training MH for {epochs_mhgan} epochs')
        mhgan_loss_G, mhgan_loss_D = self.training_model(epochs_mhgan, 'mh', batch_size, train_x, train_y, seq_length)

        for i in range(epochs_wgan, epochs_wgan + epochs_mhgan):
            hist_G[i] = mhgan_loss_G[i - epochs_wgan]
            hist_D[i] = mhgan_loss_D[i - epochs_wgan]

        if plot_loss:
            self.plot(hist_G, hist_D)

    def plot(self, hist_G, hist_D):
        plt.figure(figsize=(12, 6))
        plt.plot(hist_G, color='blue', label='Generator Loss')
        plt.plot(hist_D, color='orange', label='Discriminator Loss')
        plt.title('MHWGAN Loss')
        plt.xlabel('Epochs')
        plt.legend(loc='upper right')
        plt.show()
