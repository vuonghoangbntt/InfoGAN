import torch.nn as nn
import torch
import numpy as np


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class NLL_gaussian:
    def __call__(self, x, mu, var):
        """
        Compute negative log-likelihood for Gaussian distribution
        """
        eps = 1e-6
        l = (x - mu) ** 2
        l /= (2 * var + eps)
        l += 0.5 * torch.log(2 * np.pi * var + eps)

        return l
