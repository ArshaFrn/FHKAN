from __future__ import annotations
from typing import List, Optional, Sequence, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .hermite import fractional_hermite_functions

def _reparam_mu(raw: Tensor, mu_min: float) -> Tensor:
    return F.softplus(raw) + mu_min

def _reparam_gamma(raw: Tensor) -> Tensor:
    return torch.sigmoid(raw)

def _reparam_lambda(raw: Tensor) -> Tensor:
    return F.softplus(raw)

class FHNB(nn.Module):

    def __init__(self, degree: int=4, *, mu_min: float=-0.5, init_gamma: float=1.556, init_scale: float=1.843, init_mu: float=0.738, normalize: bool=True, freeze_gamma: Optional[float]=None) -> None:
        super().__init__()
        if degree < 0:
            raise ValueError('degree must be non-negative')
        self.degree = int(degree)
        self.mu_min = float(mu_min)
        self.normalize = bool(normalize)
        self.gamma_frozen = freeze_gamma is not None
        self.raw_mu = nn.Parameter(torch.tensor(float(init_mu)))
        self.raw_gamma = nn.Parameter(torch.tensor(float(init_gamma)), requires_grad=freeze_gamma is None)
        self.raw_lambda = nn.Parameter(torch.tensor(float(init_scale)))
        if freeze_gamma is None:
            self.register_buffer('_frozen_gamma', torch.empty(0))
        else:
            self.register_buffer('_frozen_gamma', torch.tensor(float(freeze_gamma)))

    @property
    def mu_hat(self) -> Tensor:
        return _reparam_mu(self.raw_mu, self.mu_min)

    @property
    def gamma_hat(self) -> Tensor:
        if self.gamma_frozen:
            return self._frozen_gamma
        return _reparam_gamma(self.raw_gamma)

    @property
    def lambda_hat(self) -> Tensor:
        return _reparam_lambda(self.raw_lambda)

    def extra_repr(self) -> str:
        frozen = f', freeze_gamma={float(self._frozen_gamma):g}' if self.gamma_frozen else ''
        return f'degree={self.degree}, mu_min={self.mu_min}, normalize={self.normalize}{frozen}'

    def forward(self, zeta: Tensor) -> Tensor:
        scaled = self.lambda_hat * zeta
        basis = fractional_hermite_functions(scaled, self.degree, self.mu_hat, self.gamma_hat, normalize=self.normalize)
        return basis[..., self.degree]

class FHKANLayer(nn.Module):

    def __init__(self, in_features: int, out_features: int, degree: int=4, *, mu_min: float=-0.5, bias: bool=True, normalize: bool=True, freeze_gamma: Optional[float]=None) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.degree = int(degree)
        self.mu_min = float(mu_min)
        self.normalize = bool(normalize)
        self.gamma_frozen = freeze_gamma is not None
        n_basis = self.degree + 1
        self.raw_mu = nn.Parameter(torch.full((n_basis,), 0.738))
        self.raw_gamma = nn.Parameter(torch.full((n_basis,), 1.556), requires_grad=freeze_gamma is None)
        self.raw_lambda = nn.Parameter(torch.full((n_basis,), 1.843))
        if freeze_gamma is None:
            self.register_buffer('_frozen_gamma', torch.empty(0))
        else:
            self.register_buffer('_frozen_gamma', torch.full((n_basis,), float(freeze_gamma)))
        self.theta = nn.Parameter(torch.empty(out_features, in_features, n_basis))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.xavier_uniform_(self.theta.reshape(out_features, -1))

    def hats(self):
        mu = _reparam_mu(self.raw_mu, self.mu_min)
        gamma = self._frozen_gamma if self.gamma_frozen else _reparam_gamma(self.raw_gamma)
        lam = _reparam_lambda(self.raw_lambda)
        return (mu, gamma, lam)

    def forward(self, zeta: Tensor) -> Tensor:
        mu, gamma, lam = self.hats()
        scaled = zeta.unsqueeze(-1) * lam
        pieces = []
        for q in range(self.degree + 1):
            fn = fractional_hermite_functions(scaled[..., q], q, mu[q], gamma[q], normalize=self.normalize)[..., q]
            pieces.append(fn)
        basis = torch.stack(pieces, dim=-1)
        flat_basis = basis.reshape(*basis.shape[:-2], -1)
        flat_theta = self.theta.reshape(self.out_features, -1)
        out = F.linear(flat_basis, flat_theta, self.bias)
        return out

class SequentialFHKAN(nn.Module):

    def __init__(self, in_features: int, out_features: int, hidden: Sequence[int]=(16, 16), degrees: Union[int, Sequence[int]]=4, *, mu_min: float=-0.5, normalize: bool=True, freeze_gamma: Optional[float]=None) -> None:
        super().__init__()
        widths: List[int] = [in_features, *list(hidden), out_features]
        if isinstance(degrees, int):
            deg_list = [degrees] * (len(widths) - 2)
        else:
            deg_list = list(degrees)
            if len(deg_list) != len(widths) - 2:
                raise ValueError('degrees must match the number of hidden layers')
        layers: List[nn.Module] = []
        for i in range(len(widths) - 2):
            layers.append(nn.Linear(widths[i], widths[i + 1]))
            layers.append(FHNB(deg_list[i], mu_min=mu_min, normalize=normalize, freeze_gamma=freeze_gamma))
        layers.append(nn.Linear(widths[-2], widths[-1]))
        self.net = nn.Sequential(*layers)
        for mod in self.net:
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

def _mlp_head(in_dim: int, out_features: int, hidden_dims: Optional[Sequence[int]]):
    if not hidden_dims:
        return nn.Linear(in_dim, out_features)
    dims = [in_dim, *list(hidden_dims), out_features]
    layers: List[nn.Module] = []
    for i in range(len(dims) - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.Tanh())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)

class ParallelFHKAN(nn.Module):

    def __init__(self, in_features: int, out_features: int, hidden: int=10, degrees: Sequence[int]=(1, 2, 3, 4, 5, 6), *, n_blocks: Optional[int]=None, mu_min: float=-0.5, normalize: bool=True, head_hidden: Optional[int]=None, head_layers: Optional[Sequence[int]]=None, per_block_linear: bool=True, zero_output: bool=False, init_mu: float=0.738, init_gamma: float=1.556, init_scale: float=1.843, freeze_gamma: Optional[float]=None) -> None:
        super().__init__()
        if n_blocks is not None:
            degrees = tuple(range(1, n_blocks + 1))
        self.degrees = tuple((int(d) for d in degrees))
        self.per_block_linear = bool(per_block_linear)
        if self.per_block_linear:
            self.stems = nn.ModuleList([nn.Linear(in_features, hidden) for _ in self.degrees])
            self.stem = None
        else:
            self.stem = nn.Linear(in_features, hidden)
            self.stems = None
        self.blocks = nn.ModuleList([FHNB(q, mu_min=mu_min, normalize=normalize, init_mu=init_mu, init_gamma=init_gamma, init_scale=init_scale, freeze_gamma=freeze_gamma) for q in self.degrees])
        fused = hidden * len(self.blocks)
        if head_layers is not None:
            self.head = _mlp_head(fused, out_features, head_layers)
        elif head_hidden is None:
            self.head = nn.Linear(fused, out_features)
        else:
            self.head = nn.Sequential(nn.Linear(fused, head_hidden), nn.Tanh(), nn.Linear(head_hidden, out_features))
        for mod in list(self.stems or []) + ([self.stem] if self.stem is not None else []):
            nn.init.xavier_uniform_(mod.weight)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        last = self.head[-1] if isinstance(self.head, nn.Sequential) else self.head
        if isinstance(last, nn.Linear):
            nn.init.xavier_uniform_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)
        if zero_output:
            self._zero_output()

    def _zero_output(self) -> None:
        last = self.head[-1] if isinstance(self.head, nn.Sequential) else self.head
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x: Tensor) -> Tensor:
        if self.stems is not None:
            feats = [blk(stem(x)) for stem, blk in zip(self.stems, self.blocks)]
        else:
            h = self.stem(x)
            feats = [blk(h) for blk in self.blocks]
        return self.head(torch.cat(feats, dim=-1))

class ResParallelFHKAN(nn.Module):

    def __init__(self, in_features: int=1, out_features: int=1, hidden: int=10, degrees: Sequence[int]=(1, 2, 3, 4, 5), depth: int=1, **parallel_kwargs) -> None:
        super().__init__()
        degrees = tuple((int(d) for d in degrees))
        fused = hidden * len(degrees)
        self.in_layer = ParallelFHKAN(in_features, fused, hidden=hidden, degrees=degrees, **parallel_kwargs)
        if isinstance(self.in_layer.head, nn.Linear) and self.in_layer.head.out_features == fused:
            with torch.no_grad():
                self.in_layer.head.weight.copy_(torch.eye(fused, dtype=self.in_layer.head.weight.dtype))
                if self.in_layer.head.bias is not None:
                    self.in_layer.head.bias.zero_()
        self.hidden_layers = nn.ModuleList([ParallelFHKAN(fused, fused, hidden=hidden, degrees=degrees, zero_output=True, **parallel_kwargs) for _ in range(max(int(depth) - 1, 0))])
        self.out = nn.Linear(fused, out_features)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: Tensor) -> Tensor:
        h = self.in_layer(x)
        for layer in self.hidden_layers:
            h = h + layer(h)
        return self.out(h)

class DeepFHKAN(nn.Module):

    def __init__(self, in_features: int, out_features: int, hidden: int=5, degree: int=4, n_layers: int=1, **layer_kwargs) -> None:
        super().__init__()
        n_hidden = max(int(n_layers), 1)
        layers: List[nn.Module] = [FHKANLayer(in_features, hidden, degree, **layer_kwargs)]
        for _ in range(n_hidden - 1):
            layers.append(FHKANLayer(hidden, hidden, degree, **layer_kwargs))
        self.hidden = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, out_features)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.out(self.hidden(x))
