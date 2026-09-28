from __future__ import annotations
from typing import Callable, Dict, Sequence
import torch
import torch.nn as nn
from torch import Tensor
from .layers import FHNB, DeepFHKAN
ACTIVATIONS: Dict[str, Callable[[], nn.Module]] = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'gelu': nn.GELU, 'silu': nn.SiLU, 'elu': nn.ELU, 'leakyrelu': nn.LeakyReLU, 'softplus': nn.Softplus, 'selu': nn.SELU}

def make_activation(name: str, **fhnb_kwargs) -> nn.Module:
    key = name.lower().replace(' ', '').replace('_', '')
    if key.startswith('fhnb') or key.startswith('fhkan'):
        degree = int(''.join((ch for ch in key if ch.isdigit())) or '4')
        return FHNB(degree, **fhnb_kwargs)
    if key not in ACTIVATIONS:
        raise ValueError(f'unknown activation {name!r}')
    return ACTIVATIONS[key]()

class MLP(nn.Module):

    def __init__(self, in_features: int, out_features: int, hidden: Sequence[int]=(16,), activation: str='relu', **act_kwargs) -> None:
        super().__init__()
        widths = [in_features, *list(hidden), out_features]
        layers = []
        for i in range(len(widths) - 2):
            layers.append(nn.Linear(widths[i], widths[i + 1]))
            layers.append(make_activation(activation, **act_kwargs))
        layers.append(nn.Linear(widths[-2], widths[-1]))
        self.net = nn.Sequential(*layers)
        for mod in self.net:
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

class FHKANRegressor(nn.Module):

    def __init__(self, in_features: int=1, out_features: int=1, hidden: Sequence[int]=(5,), degree: int=5, **kwargs) -> None:
        super().__init__()
        self.core = DeepFHKAN(in_features, out_features, hidden=hidden[0] if hidden else 5, degree=degree, n_layers=max(len(hidden), 1), **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.core(x)
