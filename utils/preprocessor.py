import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler


class Preprocessor():
    
    def __init__(self, windows, split_rate):
        self.windows = windows
        self.split_rate = split_rate

    def get_data(self, data_name):
        data = self._get_data(data_name)
        train_x, train_y, test_x, test_y = self._make_train_test(data)
        return train_x, train_y, test_x, test_y

    def _get_data(self, data_name):
        return pd.read_csv(f'./assets/{data_name}.csv', index_col = 'Date')

    def _make_train_test(self, data):
        data['y'] = data['Close']

        x = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        split = int(data.shape[0] * self.split_rate)
        train_x, test_x = x[:split, :], x[split:, :]
        train_y, test_y = y[:split,], y[split:,]

        self.x_scaler = MinMaxScaler(feature_range = (0, 1))
        self.y_scaler = MinMaxScaler(feature_range = (0, 1))

        train_x = self.x_scaler.fit_transform(train_x)
        test_x = self.x_scaler.transform(test_x)

        train_y = self.y_scaler.fit_transform(train_y.reshape(-1, 1))
        test_y = self.y_scaler.transform(test_y.reshape(-1, 1))     
        return train_x, train_y, test_x, test_y   

    def _is_torch(self, target):
        if type(target) != torch.Tensor:
            return torch.from_numpy(target).float()
        else:
            return target

    def sliding_windows(self, x, y, windows):
        x_ = []
        y_1d = []
        y_nd = []
        for i in range(windows, x.shape[0]):
            x_.append(x[i-windows: i, :])
            y_1d.append(y[i])
            y_nd.append(y[i-windows: i, :])
        x_ = torch.from_numpy(np.array(x_))
        y_1d = torch.from_numpy(np.array(y_1d))
        y_nd = torch.from_numpy(np.array(y_nd))
        return x_, y_1d, y_nd

    def make_torch_data(self, x, y, batch_size, shuffle):
        x = self._is_torch(x)
        y = self._is_torch(y)
        return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)