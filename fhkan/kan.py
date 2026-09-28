from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def chebyshev_polynomials(x: Tensor, degree: int) -> Tensor:
    t0 = torch.ones_like(x)
    if degree == 0:
        return t0.unsqueeze(-1)
    t1 = x
    terms = [t0, t1]
    for _ in range(2, degree + 1):
        terms.append(2.0 * x * terms[-1] - terms[-2])
    return torch.stack(terms, dim=-1)

class ChebyKANLayer(nn.Module):

    def __init__(self, in_features: int, out_features: int, degree: int=4):
        super().__init__()
        self.degree = int(degree)
        self.theta = nn.Parameter(torch.empty(out_features, in_features, self.degree + 1))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.theta.reshape(out_features, -1))

    def forward(self, x: Tensor) -> Tensor:
        t = chebyshev_polynomials(torch.tanh(x), self.degree)
        flat = t.reshape(*t.shape[:-2], -1)
        w = self.theta.reshape(self.theta.shape[0], -1)
        return F.linear(flat, w, self.bias)

class ChebyKAN(nn.Module):

    def __init__(self, in_features: int, out_features: int, hidden: int=5, degree: int=4, n_layers: int=1):
        super().__init__()
        n_hidden = max(int(n_layers), 1)
        self.in_layer = ChebyKANLayer(in_features, hidden, degree)
        self.hidden = nn.ModuleList([ChebyKANLayer(hidden, hidden, degree) for _ in range(n_hidden - 1)])
        self.out_layer = ChebyKANLayer(hidden, out_features, degree)

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_layer(x)
        for layer in self.hidden:
            x = x + layer(x)
        return self.out_layer(x)
