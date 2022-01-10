# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    """Basic discriminator.
    """

    def __init__(
            self,
            total_locations=8606,
            embedding_net=None,
            embedding_dim=64,
            dropout=0.6):
        super(Discriminator, self).__init__()
        num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
        filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        if embedding_net:
            self.embedding = embedding_net
        else:
            self.embedding = nn.Embedding(
                num_embeddings=total_locations,
                embedding_dim=embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, n, (f, embedding_dim)) for (
            n, f) in zip(num_filters, filter_sizes)])
        self.highway = nn.Linear(sum(num_filters), sum(num_filters))
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(sum(num_filters), 2)
        self.init_parameters()

    def forward(self, x):
        """
        Args:
            x: (batch_size * seq_len)
        """
        emb = self.embedding(x).unsqueeze(
            1)  # batch_size * 1 * seq_len * emb_dim
        # [batch_size * num_filter * length]
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2)
                 for conv in convs]  # [batch_size * num_filter]
        pred = torch.cat(pools, 1)  # batch_size * num_filters_sum
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) * F.relu(highway) + \
            (1. - torch.sigmoid(highway)) * pred
        pred = F.log_softmax(self.linear(self.dropout(pred)), dim=-1)
        return pred

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)


class TCDiscriminator(nn.Module):

    def __init__(self,
            total_locations=8606,
            embedding_net=None,
            sembedding_dim=64,
            tembedding_dim=16,
            dropout=0.6):
        super(TCDiscriminator, self).__init__()
        num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
        filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        if embedding_net:
            self.tembedding = embedding_net[0]
            self.sembedding = embedding_net[1]
        else:
            self.tembedding = nn.Embedding(
                num_embeddings=total_locations,
                embedding_dim=tembedding_dim)
            self.sembedding = nn.Embedding(
                num_embeddings=total_locations,
                embedding_dim=sembedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, n, (f, tembedding_dim + sembedding_dim)) for (
            n, f) in zip(num_filters, filter_sizes)])
        self.highway = nn.Linear(sum(num_filters), sum(num_filters))
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(sum(num_filters), 2)
        self.init_parameters()

    def forward(self, xt, xs):
        """
        Args:
            x: (batch_size * seq_len)
        """
        temb = self.tembedding(xt)
        semb = self.sembedding(xs)
        emb = torch.cat([temb, semb], dim=-1).unsqueeze(1)
        # [batch_size * num_filter * length]
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2)
                 for conv in convs]  # [batch_size * num_filter]
        pred = torch.cat(pools, 1)  # batch_size * num_filters_sum
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) * F.relu(highway) + \
               (1. - torch.sigmoid(highway)) * pred
        pred = F.log_softmax(self.linear(self.dropout(pred)), dim=-1)
        return pred

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)
