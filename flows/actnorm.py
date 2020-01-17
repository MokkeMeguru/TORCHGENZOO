from torch import nn
import torch
import numpy as np


class Actnorm(nn.Module):
    def __init__(self, channels, scale=1.0):
        """
        """
        super(Actnorm, self).__init__()
        self.channels = channels
        size = [1, channels, 1, 1]
        self.register_parameter('bias', nn.Parameter(torch.zeros(*size)))
        self.register_parameter('logs', nn.Parameter(torch.zeros(*size)))
        self.scale = float(scale)
        self.inited = False
        self._logdet_jacobian = None

    def initialize_parameters(self, x):
        if not self.training:
            return
        assert x.device == self.bias.device
        with torch.no_grad():
            bias = torch.mean(x.clone(), dim=[0, 2, 3], keepdim=True) * -1.0
            vars = torch.mean((x.clone() + bias) ** 2,
                              dim=[0, 2, 3], keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-7))
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.inited = True

    def forward(self, x, compute_jacobian=True):
        if not self.inited:
            self.initialize_parameters(x)
        z = (x + self.bias) * torch.exp(self.logs)
        if compute_jacobian:
            self._log_det_jacobian = torch.sum(
                self.logs) * np.prod(x.size()[2:])
        return z

    def inverse(self, z):
        x = z * torch.exp(- self.logs) - self.bias
        return x
