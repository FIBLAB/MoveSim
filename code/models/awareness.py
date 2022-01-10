# coding: utf-8

import torch.nn as nn


class DAN(nn.Module):
    """Distance awareness network.
    """

    def __init__(self, total_locs):
        super(DAN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=total_locs, out_features=total_locs),
            nn.Sigmoid(),
        )
        self.weight_init()

    def forward(self, x):
        weights = self.net(x)
        return weights

    def weight_init(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)