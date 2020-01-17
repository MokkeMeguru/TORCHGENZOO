from torch import nn
import torch


class Flow(nn.Module):
    def __init__(self, in_features):
        """
        in_features: [C H W]
        ex. mnist => [1, 32, 32]
        """
        super().__init__()
        self.in_features = in_features
        self._logdet_jacobian = None

    @property
    def logdet_jacobian(self):
        return self._logdet_jacobian


class FlowList(Flow):
    def __init__(self, flow_list):
        super(FlowList, self).__init__()
        self.flow_list = nn.ModuleList(flow_list)

    def forward(self, x, compute_jacobian=True):
        logdet_jacobian = 0.0
        z = x
        for flow in self.flow_list:
            z = flow.forward(z, compute_jacobian)
            if compute_jacobian:
                logdet_jacobian = logdet_jacobian + flow.logdet_jacobian
        if compute_jacobian:
            self._logdet_jacobian = logdet_jacobian
        return z

    def inverse(self, z):
        x = z
        for flow in reversed(self.flow_list):
            x = flow.inverse(x)
        return x


class FlowBlock(Flow):
    def __init__(self, flow_list):
        super(FlowBlock, self).__init__(
            flow_list[0].in_features + flow_list[1].in_features)
        assert len(flow_list) == 2, 'flow block require two flow layer'

    def forward(self, x, compute_jacobian=True):
        logdet_jacobian = 0.0
        x1, x2 = torch.chunk(x, 2)
        z1 = self.flow_list[0](x1, compute_jacobian)
        z2 = self.flow_list[1](x2, compute_jacobian)
        if compute_jacobian:
            logdet_jacobian = logdet_jacobian + \
                self.flow_list[0].logdet_jacobian()
            logdet_jacobian = logdet_jacobian + \
                self.flow_list[1].logdet_jacobian()
            self._logdet_jacobian = logdet_jacobian
        z = torch.cat([z1, z2], 1)
        return z

    def inverse(self, z):
        z1, z2 = torch.chunk(z, 2, dim=1)
        x1 = self.flow_list[0].inverse(z1)
        x2 = self.flow_list[1].inverse(z2)
        x = torch.cat([x1, x2], 1)
        return x
