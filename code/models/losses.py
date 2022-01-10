# coding: utf-8

import numpy as np
import torch.nn as nn

class distance_loss(nn.Module):

    def __init__(self):
        with open('../data/raw/Cellular_Baselocation_baidu') as f:
            gpss = f.readlines()
        self.X = []
        self.Y = []
        for gps in gpss:
            x, y = float(gps.split()[0]), float(gps.split()[1])
            self.X.append(x)
            self.Y.append(y)
        self.X = torch.Tensor(np.array(self.X)).float()
        self.Y = torch.Tensor(np.array(self.Y)).float()

    def forward(self, x):
        """

        :param x: generated sequence, batch_size * seq_len
        :return:
        """
        x1 = torch.index_select(self.X, 0, x[:, :-1].view(-1))
        x2 = torch.index_select(self.X, 0, x[:, 1:].view(-1))
        y1 = torch.index_select(self.Y, 0, x[:, :-1].view(-1))
        y2 = torch.index_select(self.Y, 0, x[:, :-1].view(-1))
        dx = x1 - x2
        dy = y1 - y2
        loss = dx**2 + dy**2
        return loss


class period_loss(nn.Module):

    def __init__(self, time_interval):
        self.time_interval = time_interval
        self.mse = nn.MSELoss()

    def forward(self, x):
        """

        :param x: generated sequence, batch_size * seq_len
        :return:
        """
        loss = 0.
        for i in range(0, x.size(1) - self.time_interval):
            loss += self.mse(x[:, i], x[:, i + self.time_interval])
        return loss
