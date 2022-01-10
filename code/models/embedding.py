# coding: utf-8

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class Embedding(nn.Module):
    """Common embedding network.
    """

    def __init__(self, total_locations, embedding_dim):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=total_locations, embedding_dim=embedding_dim)

    def forward(self, x):
        return self.embedding(x)
