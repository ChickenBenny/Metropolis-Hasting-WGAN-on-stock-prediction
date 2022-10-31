from pickle import FALSE
import torch
import random
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.isotonic import IsotonicRegression
from .wgan import WGAN

class MHGAN(WGAN):
    
    def calibrate_discriminator(self, train_x, train_y):
        self.calibrator = IsotonicRegression()
        mh_sample_predict = self.generator_samples(train_x)
        mh_sample_cat = torch.cat([train_y[:, :3, :], mh_sample_predict.cpu().detach().reshape(-1, 1, 1)], axis = 1)

        mh_sample_dis = self.score_sample(mh_sample_cat)

        self.calibrator.fit(mh_sample_dis.cpu().detach().numpy(), train_y[:, 3, :].numpy().reshape(-1))
        mh_sample_cal = self.calibrator.predict(mh_sample_dis.cpu().detach().numpy())
        return mh_sample_cal

    def num_sampler(self, n_samples, x_shape):
        random_pick = []
        for i in range(n_samples):
            random_pick.append(random.randint(0, x_shape - 1))
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
            x_smaple_cat = torch.cat([y_sample[:, :3, :], pred_x.cpu().detach().reshape(-1, 1, 1)], axis = 1)
            x_dis_score = self.score_sample(x_smaple_cat.to(self.device))
            x_score = self.calibrator.transform(x_dis_score.cpu().detach().numpy())

            u = np.random.uniform(0, 1, (1,))[0]
            if u <= np.fmin(1., (1./ y_score_base - 1.)/ (1./ x_score - 1.)):
                y_score_base = x_score
                x_samples.append(x_sample.numpy())
                y_samples.append(y_sample.numpy())
        
        return np.array(x_samples).reshape(-1, train_x.shape[1], train_x.shape[2]), np.array(y_samples).reshape(-1, train_y.shape[1], train_y.shape[2])

    def mh_enhance(self, epochs, batch_size, train_x, train_y, real_tick, plot_loss = False):
        hist_G = np.zeros(epochs)
        hist_D = np.zeros(epochs)        
        for epoch in range(epochs):
            loss_G = []
            loss_D = []

            mh_sample_cal = self.calibrate_discriminator(train_x, train_y)
            mh_sample_x, mh_sample_y = self.mh_sample(train_x, train_y, mh_sample_cal)
            mh_dataset = DataLoader(TensorDataset(torch.from_numpy(mh_sample_x).float(), torch.from_numpy(mh_sample_y).float()), batch_size = batch_size, shuffle = False)

            self.dis.train()
            self.gen.train()
            for x, y in mh_dataset:
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
        if plot_loss:
            self.plot(hist_G, hist_D) 
        return hist_G, hist_D


    def mh_training(self, epochs_wgan, epochs_mhgan, train_dataloader, batch_size, train_x, train_y, real_tick):         
        hist_G = np.zeros(epochs_wgan + epochs_mhgan)
        hist_D = np.zeros(epochs_wgan + epochs_mhgan)

        print("WGAN init training")
        wgan_loss_G, wgan_loss_D = self.training_step(epochs_wgan, train_dataloader, real_tick, False)
        
        for i in range(epochs_wgan):
            hist_G[i] = wgan_loss_G[i]
            hist_D[i] = wgan_loss_D[i]

        print("Use metropolis-Hasting enhance training")
        mhgan_loss_G, mhgan_loss_D = self.mh_enhance(epochs_mhgan, batch_size, train_x, train_y, real_tick, False)

        for i in range(epochs_wgan, epochs_wgan + epochs_mhgan):
            hist_G[i] = mhgan_loss_G[i - epochs_wgan]
            hist_D[i] = mhgan_loss_D[i - epochs_wgan]]

        print("plot the training result")
        print(f'Train in {epochs_wgan} epochs and enhance in {epochs_mhgan} epochs')
        self.plot(hist_G, hist_D)        



    def plot_prob_original(self, x, y):
        plt.subplots(1, 2)

        plt.title('Probability density')
        plt.subplot(1, 2, 1)
        sns.distplot(x.cpu().detach().numpy())

        plt.title('Original scatter')      
        plt.subplot(1, 2, 2)
        plt.scatter(x.cpu().detach().numpy(), y.numpy())
  
    def plot_prob_cal(self, x, y):
        plt.scatter(x.cpu().detach().numpy(), y)    