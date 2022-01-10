# coding: utf-8
import pdb
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


class GANLoss(nn.Module):
    """Reward-Refined NLLLoss Function for adversial training of Gnerator"""

    def __init__(self):
        super(GANLoss, self).__init__()

    def forward(self, prob, target, reward, device, ploss=False):
        """
        Args:
            prob: (N, C), torch Variable
            target : (N, ), torch Variable
            reward : (N, ), torch Variable
        """
        N = target.size(0)
        C = prob.size(1)
        one_hot = torch.zeros((N, C))
        if prob.is_cuda:
            one_hot = one_hot.to(device)
        one_hot.scatter_(1, target.data.view((-1, 1)), 1)
        one_hot = one_hot.type(torch.ByteTensor)
        one_hot = Variable(one_hot)
        if prob.is_cuda:
            one_hot = one_hot.to(device)
        loss = torch.masked_select(prob, one_hot)
        loss = loss * reward
        loss = -torch.sum(loss)
        return loss

class distance_loss(nn.Module):

    def __init__(self, datasets, device):
        super(distance_loss, self).__init__()
        if datasets == 'mobile':
            with open('../data/mobile/gps') as f:
                gpss = f.readlines()
        else:
             with open('../data/geolife/gps') as f:
                gpss = f.readlines()
        self.X = []
        self.Y = []
        for gps in gpss:
            x, y = float(gps.split()[0]), float(gps.split()[1])
            self.X.append(x)
            self.Y.append(y)
        self.X = torch.Tensor(np.array(self.X)).float()
        self.Y = torch.Tensor(np.array(self.Y)).float()
        self.X = self.X.to(device)
        self.Y = self.Y.to(device)

    def forward(self, x):
        """

        :param x: generated sequence, batch_size * seq_len
        :return:
        """
        x = x.long()
        x1 = torch.index_select(self.X, 0, x[:, :-1].contiguous().view(-1))
        x2 = torch.index_select(self.X, 0, x[:, 1:].contiguous().view(-1))
        y1 = torch.index_select(self.Y, 0, x[:, :-1].contiguous().view(-1))
        y2 = torch.index_select(self.Y, 0, x[:, 1:].contiguous().view(-1))
        dx = x1 - x2
        dy = y1 - y2
        loss = dx**2 + dy**2
        loss = torch.sum(loss) / loss.size(0)
        return loss


class period_loss(nn.Module):

    def __init__(self, time_interval):
        super(period_loss, self).__init__()
        self.time_interval = time_interval
        self.mse = nn.MSELoss()

    def forward(self, x):
        """

        :param x: generated sequence, batch_size * seq_len
        :return:
        """
        loss = 0.
        for i in range(0, x.size(1) - self.time_interval):
            loss += torch.sum(torch.ne(x[:, i], x[:, i + self.time_interval]))
        return loss


class embd_distance_loss(nn.Module):

    def __init__(self, embd):
        super(embd_distance_loss, self).__init__()
        self.embd = embd

    def forward(self, x, embd_size):
        """

        :param x: generated sequence, batch_size * seq_len
        :return:
        """
        emb = self.embd(x)
        emb = emb.permute(1, 0, 2)
        curr = emb[: x.size(1) - 1].contiguous().view(-1, embd_size)
        next = emb[1: x.size(1)].contiguous().view(-1, embd_size)
        loss = torch.nn.functional.mse_loss(curr, next, reduction='sum')
        return loss


class embd_period_loss(nn.Module):

    def __init__(self, embd):
        super(embd_period_loss, self).__init__()
        self.embd = embd

    def forward(self, x, embd_size):
        """

        :param x: generated sequence, batch_size * seq_len
        :return:
        """
        emb = self.embd(x)
        emb = emb.permute(1, 0, 2)
        curr = emb[: 24].contiguous().view(-1, embd_size)
        next = emb[24:].contiguous().view(-1, embd_size)
        loss = torch.nn.functional.mse_loss(curr, next, reduction='sum')
        return loss
