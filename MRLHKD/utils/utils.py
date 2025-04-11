import torch
from torch import nn
import numpy as np
from copy import deepcopy
from MRLHKD.utils.logger import my_logger


def act_str2obj(activation):
    act = nn.ReLU()
    if activation == 'ReLU':
        act = nn.ReLU()
    elif activation == 'SeLU':
        act = nn.SELU()
    elif activation == 'Sigmoid':
        act = nn.Sigmoid()
    elif activation == 'Tanh':
        act = nn.Tanh()
    return act


def get_optimizer(model, lr, optimizer_str):
    if optimizer_str == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_str == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_str == 'RMSprop':
        return torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise NotImplementedError('Optimizer ' + model.optimizer +
                                  ' is not implemented')


class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, dics):
        score = -val_loss
        if np.isnan(val_loss):
            self.counter += 1
            my_logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        elif self.best_score is None:
            self.best_score = score
            dics[0] = deepcopy(model.state_dict())
        elif score <= self.best_score + self.delta:
            self.counter += 1
            my_logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            dics[0] = deepcopy(model.state_dict())


