import numpy as np

import torch
from torch._C import device
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable

def createLissajous(a, b, r, data_num):
    t = np.linspace(-2.5*np.pi, 2.5*np.pi, data_num)
    x = r*np.sin(a*t)
    y = r*np.sin(b*t)
    xy = np.stack([x, y], 1)
    x = np.expand_dims(x,1)
    y = np.expand_dims(y,1)
    t = np.expand_dims(t,1)
    return t, xy, x, y

def createBatches(data, batch_size, torch_device):
    sequence_length, data_dim = data.shape
    batch_num = int(sequence_length / batch_size)
    data_batch = torch.Tensor(data.reshape(batch_size, batch_num, data_dim)).to(torch_device)
    return data_batch