import torch
import random
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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