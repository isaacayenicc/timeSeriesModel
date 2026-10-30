"""Compact past-window neural predictors; no future padding or bidirectionality."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class CausalBlock(nn.Module):
    """Two left-padded convolutions plus pointwise residual projection."""
    def __init__(self, channels, dilation):
        super().__init__()
        self.dilation = dilation
        self.a = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.b = nn.Conv1d(channels, channels, 3, dilation=dilation)

    def forward(self, x):
        y = F.relu(self.a(F.pad(x, (2*self.dilation, 0))))
        y = self.b(F.pad(y, (2*self.dilation, 0)))
        return F.relu(x+y)


class Predictor(nn.Module):
    def __init__(self, kind, lag, channels=8):
        super().__init__()
        self.kind, self.lag = kind, lag
        if kind == "mlp":
            self.net = nn.Sequential(nn.Linear(lag, 32), nn.ReLU(), nn.Linear(32, 1))
        elif kind == "gru":
            self.net = nn.GRU(1, 16, batch_first=True)
            self.head = nn.Linear(16, 1)
        else:
            self.stem = nn.Conv1d(1, channels, 1)
            self.blocks = nn.ModuleList([CausalBlock(channels, 2**i) for i in range(math.ceil(math.log2((lag-1)/4+1)))])
            self.receptive_field = 1+4*sum(2**i for i in range(len(self.blocks)))
            if kind in ("speed_tcn", "shift_tcn"):
                self.film = nn.ModuleList([nn.Linear(1, channels*2) for _ in self.blocks])
            self.head = nn.Linear(channels+(1 if kind=="late_tcn" else 0), 1)

    def forward(self, x):
        if self.kind == "mlp":
            return self.net(x[:, 0]).squeeze(-1)
        if self.kind == "gru":
            y, _ = self.net(x[:, :1].transpose(1, 2))
            return self.head(y[:, -1]).squeeze(-1)
        y = self.stem(x[:, :1])
        for i, block in enumerate(self.blocks):
            y = block(y)
            if self.kind in ("speed_tcn", "shift_tcn"):
                # Each past position gets only its own observed speed.
                g, b = self.film[i](x[:, 1:2].transpose(1, 2)).transpose(1, 2).chunk(2, dim=1)
                y = y*(1+0.1*torch.tanh(g))+0.1*b
        z = y[:, :, -1]
        if self.kind == "late_tcn":
            z = torch.cat((z, x[:, 1, -1:]), dim=1)
        return self.head(z).squeeze(-1)


def build_network(kind, lag, channels=8):
    """Construct a CPU-compatible fresh model; channels ignored for GRU/MLP."""
    return Predictor(kind, lag, channels)
