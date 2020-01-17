from torch import nn
import torch
from torch.nn import functional as F

import numpy as np
from flows.flow import Flow


class ChannelConv(Flow):
    def __init__(self, in_features):
        """
        in_featrues: list
        [C, H, W]
        """
        super(ChannelConv, self).__init__(in_features)
        w_shape = [in_features[0], in_features[0]]
        w_init = np.linalg.qr(np.random.normal(*w_shape))[0].astype(np.float32)
        self.register_parameter('weight', nn.Parameter(torch.Tensor(w_init)))
        self.w_shape = w_shape

    def get_parameters(self, x, inverse):
        pixels = np.prod(x.size()[2:])
        device = x.device
        logdet_jacobian  = torch.slogdet(self.weight.cpu())[1].to(device) * pixels
        if not inverse:
            weight = self.weight.view(self.w_shape[0], self.w_shape[1], 1, 1)
        else:
            weight = torch.inverse(self.weight.double()).float().view(self.w_shape[0], self.w_shape[1], 1, 1)
        return weight, logdet_jacobian

    def forward(self, x, compute_jacobian):
        weight, logdet_jacobian = self.get_parameters(x, inverse=False)
        z = F.conv2d(x, weight)
        if compute_jacobian:
            self._logdet_jacobian = logdet_jacobian
        return z

    def inverse(self, x, compute_jacobian):
        weight, _ = self.get_parameters(x, inverse=True)
        z = F.conv2d(x, weight)
        return z
