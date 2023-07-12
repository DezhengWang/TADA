import pandas as pd
import numpy as np
import os
# from torch.utils.data import Dataset
# import joblib
# from sklearn import preprocessing
# import random
import torch
from torch import nn
from IPython import display
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline

def adjust_learning_rate(optimizer, epoch, args, type="type2"):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if type=='type1':
        lr_adjust = {epoch: args.lr * (0.5 ** ((epoch-1) // 1))}
    elif type=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif type=='type3':
        lr_adjust = {
            8: 5e-5, 10: 1e-5, 12: 5e-6, 13: 1e-6,
            14: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        args.lr = lr

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def grad_clipping(net, theta):  #@save
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def save_model(net, path):
    with open(os.path.join(path, "net.pt"), 'wb') as f:
        torch.save(net, f)

def mkdir(model, dataset, pred_len, augmentation="None", index=0):
    path = os.path.join("Result", pred_len, model, dataset, augmentation, str(index))
    if not os.path.exists("Result"):
        os.mkdir("Result")
    if not os.path.exists(os.path.join("Result", pred_len)):
        os.mkdir(os.path.join("Result", pred_len))
    if not os.path.exists(os.path.join("Result", pred_len, model)):
        os.mkdir(os.path.join("Result", pred_len, model))
    if not os.path.exists(os.path.join("Result", pred_len, model, dataset)):
        os.mkdir(os.path.join("Result", pred_len, model, dataset))
    if not os.path.exists(os.path.join("Result", pred_len, model, dataset, augmentation)):
        os.mkdir(os.path.join("Result", pred_len, model, dataset, augmentation))
    if not os.path.exists(path):
        os.mkdir(path)
        print("the file folder was created: {}".format(path))
    return path


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


class Animator:  #@save
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        backend_inline.set_matplotlib_formats('svg')
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)