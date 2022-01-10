# -*- coding:utf-8 -*-

import os
import pdb
import random
import math
import numpy as np
import torch


class GenDataIter(object):
    """ Toy data iter to load digits"""

    def __init__(self, data_file, batch_size):
        super(GenDataIter, self).__init__()
        self.batch_size = batch_size
        self.data_lis = self.read_file(data_file)
        self.data_num = len(self.data_lis)
        self.indices = range(self.data_num)
        self.num_batches = int(
            math.ceil(float(self.data_num) / self.batch_size))
        self.idx = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self):
        self.idx = 0
        random.shuffle(self.data_lis)

    def next(self):
        if self.idx >= self.data_num - self.batch_size:
            raise StopIteration
        index = self.indices[self.idx:self.idx + self.batch_size]
        d = [self.data_lis[i] for i in index]
        d = torch.LongTensor(np.asarray(d, dtype='int64'))
        data = torch.cat([torch.zeros(self.batch_size, 1).long(), d], dim=1)
        target = torch.cat([d, torch.zeros(self.batch_size, 1).long()], dim=1)
        self.idx += self.batch_size
        return data, target

    def read_file(self, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        lis = []
        for line in lines:
            l = line.strip().split(' ')
            l = [int(s) for s in l]
            lis.append(l)
        return lis


class NewGenIter(object):
    """ Toy data iter to load digits"""

    def __init__(self, data_file, batch_size):
        super(NewGenIter, self).__init__()
        self.batch_size = batch_size
        self.data_lis = self.read_file(data_file)
        self.data_num = len(self.data_lis)
        self.indices = range(self.data_num)
        self.num_batches = int(
            math.ceil(float(self.data_num) / self.batch_size))
        self.idx = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self):
        self.idx = 0
        random.shuffle(self.data_lis)

    def next(self):
        if self.idx >= self.data_num - self.batch_size:
            raise StopIteration
        index = self.indices[self.idx:self.idx + self.batch_size]
        d = [self.data_lis[i] for i in index]
        d = np.asarray(d, dtype='int64')
        data = torch.LongTensor(d[:, :-1])
        target = torch.LongTensor(d[:, 1:])
        self.idx += self.batch_size
        return data, target

    def read_file(self, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        lis = []
        for line in lines:
            l = line.strip().split(' ')
            l = [int(s) for s in l]
            lis.append(l)
        return lis


class DisDataIter(object):
    """ Toy data iter to load digits"""

    def __init__(self, real_data_file, fake_data_file, batch_size,length):
        super(DisDataIter, self).__init__()
        self.batch_size = batch_size
        real_data_lis = self.read_file(real_data_file)
        t2 = real_data_lis

        if len(t2[0])!=length:
            if length == 24:
                t3 = [t[0:24] for t in t2]
            elif length==72:
                t3 = []
                for t in t2:
                    t.extend(t[0:24])
                    t3.append(t)
            elif length==96:
                t3 = []
                for t in t2:
                    t.extend(t)
                    t3.append(t)
            elif length==120:
                t3 = []
                for t in t2:
                    t.extend(t)
                    t.extend(t[0:24])
                    t3.append(t)
            elif length==168:
                t3 = []
                for t in t2:
                    t.extend(t)
                    t.extend(t[0:48])
                    t.extend(t[0:24])                    
                    t3.append(t)
                    
            real_data_lis = t3
        fake_data_lis = self.read_file(fake_data_file)
        t2 = fake_data_lis
        if len(t2[0])!=length:
            if length == 24:
                t3 = [t[0:24] for t in t2]
            elif length==72:
                t3 = []
                for t in t2:
                    t.extend(t[0:24])
                    t3.append(t)
            elif length==96:
                t3 = []
                for t in t2:
                    t.extend(t)
                    t3.append(t)
            elif length==120:
                t3 = []
                for t in t2:
                    t.extend(t)
                    t.extend(t[0:24])
                    t3.append(t)
            elif length==168:
                t3 = []
                for t in t2:
                    t.extend(t)
                    t.extend(t[0:48])
                    t.extend(t[0:24])                    
                    t3.append(t)
                    
            fake_data_lis = t3
        self.data = real_data_lis + fake_data_lis
        self.labels = [1 for _ in range(len(real_data_lis))] + \
                      [0 for _ in range(len(fake_data_lis))]
        self.pairs = list(zip(self.data, self.labels))
        self.data_num = len(self.pairs)
        self.indices = range(self.data_num)
        self.num_batches = int(
            math.ceil(float(self.data_num) / self.batch_size))
        self.idx = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self):
        self.idx = 0
        random.shuffle(self.pairs)

    def next(self):
        if self.idx >= self.data_num - self.batch_size:
            raise StopIteration
        index = self.indices[self.idx:self.idx + self.batch_size]
        pairs = [self.pairs[i] for i in index]
        data = [p[0] for p in pairs]
        label = [p[1] for p in pairs]
        data = torch.LongTensor(np.asarray(data, dtype='int64'))
        label = torch.LongTensor(np.asarray(label, dtype='int64'))
        self.idx += self.batch_size
        return data, label

    def read_file(self, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        lis = []
        for line in lines:
            l = line.strip().split(' ')
            l = [int(s) for s in l]
            lis.append(l)
        return lis


class TCGenDataIter(object):
    """data iterator with time context.
    """

    def __init__(self, data_file, batch_size):
        super(TCGenDataIter, self).__init__()
        self.batch_size = batch_size
        self.data_lis = self.read_file(data_file)
        self.data_num = len(self.data_lis)
        self.indices = range(self.data_num)
        self.num_batches = int(
            math.ceil(float(self.data_num) / self.batch_size))
        self.idx = 0
        time_list = [23] + [i % 24 for i in range(48)]
        self.time = torch.stack(
            [torch.ones(self.batch_size) * i for i in time_list], dim=1).long()

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self):
        self.idx = 0
        random.shuffle(self.data_lis)

    def next(self):
        if self.idx >= self.data_num - self.batch_size:
            raise StopIteration
        index = self.indices[self.idx:self.idx + self.batch_size]
        d = [self.data_lis[i] for i in index]
        d = torch.LongTensor(np.asarray(d, dtype='int64'))
        data = torch.cat([torch.zeros(self.batch_size, 1).long(), d], dim=1)
        target = torch.cat([d, torch.zeros(self.batch_size, 1).long()], dim=1)
        self.idx += self.batch_size
        return self.time, data, target

    def read_file(self, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        lis = []
        for line in lines:
            l = line.strip().split(' ')
            l = [int(s) for s in l]
            lis.append(l)
        return lis


class TCDisDataIter(object):
    """ Toy data iter to load digits"""

    def __init__(self, real_data_file, fake_data_file, batch_size):
        super(TCDisDataIter, self).__init__()
        self.batch_size = batch_size
        real_data_lis = self.read_file(real_data_file)
        fake_data_lis = self.read_file(fake_data_file)
        self.data = real_data_lis + fake_data_lis
        self.labels = [1 for _ in range(len(real_data_lis))] + \
                      [0 for _ in range(len(fake_data_lis))]
        self.pairs = list(zip(self.data, self.labels))
        self.data_num = len(self.pairs)
        self.indices = range(self.data_num)
        self.num_batches = int(
            math.ceil(float(self.data_num) / self.batch_size))
        self.idx = 0
        time_list = [i % 24 for i in range(48)]
        self.time = torch.stack(
            [torch.ones(self.batch_size) * i for i in time_list], dim=1).long()

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self):
        self.idx = 0
        random.shuffle(self.pairs)

    def next(self):
        if self.idx >= self.data_num - self.batch_size:
            raise StopIteration
        index = self.indices[self.idx:self.idx + self.batch_size]
        pairs = [self.pairs[i] for i in index]
        data = [p[0] for p in pairs]
        label = [p[1] for p in pairs]
        data = torch.LongTensor(np.asarray(data, dtype='int64'))
        label = torch.LongTensor(np.asarray(label, dtype='int64'))
        self.idx += self.batch_size
        return self.time, data, label

    def read_file(self, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        lis = []
        for line in lines:
            l = line.strip().split(' ')
            l = [int(s) for s in l]
            lis.append(l)
        return lis
