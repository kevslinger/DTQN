import torch
import torch.nn as nn


class GRUGate(nn.Module):
    """GRU Gating from GTrXL (https://arxiv.org/pdf/1910.06764.pdf)"""

    def __init__(self, **kwargs):
        super().__init__()

        embed_size = kwargs["embed_size"]

        self.w_r = nn.Linear(embed_size, embed_size, bias=False)
        self.u_r = nn.Linear(embed_size, embed_size, bias=False)
        self.w_z = nn.Linear(embed_size, embed_size)
        self.u_z = nn.Linear(embed_size, embed_size, bias=False)
        self.w_g = nn.Linear(embed_size, embed_size, bias=False)
        self.u_g = nn.Linear(embed_size, embed_size, bias=False)

        self.init_bias()

    def init_bias(self):
        with torch.no_grad():
            self.w_z.bias.fill_(-2)  # This is the value set by GTrXL paper

    def forward(self, x, y):
        z = torch.sigmoid(self.w_z(y) + self.u_z(x))
        r = torch.sigmoid(self.w_r(y) + self.u_r(x))
        h = torch.tanh(self.w_g(y) + self.u_g(r * x))

        return (1.0 - z) * x + z * h


class ResGate(nn.Module):
    """Residual skip connection"""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, y):
        return x + y
