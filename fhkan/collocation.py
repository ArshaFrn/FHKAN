from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import torch
from numpy.polynomial.hermite import hermgauss
from torch import Tensor
from .hermite import dehghan_map_inverse, hermite_gauss_nodes

def mapped_hermite_positive(n: int, *, k: float=1.0, n_raw: Optional[int]=None, device=None, dtype=torch.float32) -> Tensor:
    raw = n_raw if n_raw is not None else max(2 * n, n + 8)
    x = hermite_gauss_nodes(raw, device=device, dtype=dtype)
    z = dehghan_map_inverse(x, k=k)
    z = z[z > 0]
    z, _ = torch.sort(z)
    return z[:n]

def hermite_nodes_interval(n: int, a: float=0.0, b: float=1.0, device=None, dtype=None) -> Tensor:
    x = hermite_gauss_nodes(n, device=device, dtype=dtype or torch.float32)
    xmin, xmax = (x.min(), x.max())
    u = (x - xmin) / (xmax - xmin).clamp_min(1e-12)
    return u * (b - a) + a

def mapped_hermite_nodes(n: int, *, k: float=1.0, z_max: float=16.0, device=None, dtype=torch.float64) -> Tensor:
    x = hermite_gauss_nodes(n, device=device, dtype=dtype)
    z = dehghan_map_inverse(x, k=k)
    z = z[z > 0]
    z = z[z <= z_max]
    z, _ = torch.sort(z)
    return z

def chebyshev_nodes(n: int, a: float=0.0, b: float=1.0, device=None, dtype=None):
    k = torch.arange(n, device=device, dtype=dtype or torch.float32)
    x = torch.cos(torch.pi * k / max(n - 1, 1))
    return 0.5 * (b - a) * (x + 1.0) + a

def tensor_grid(x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
    X, T = torch.meshgrid(x, t, indexing='ij')
    pts = torch.stack([X.reshape(-1), T.reshape(-1)], dim=-1)
    return (pts, X, T)

def caputo_l1_matrix(t: Tensor, alpha: float) -> Tensor:
    t = t.reshape(-1)
    n = t.numel()
    dtype, device = (t.dtype, t.device)
    D = torch.zeros(n, n, dtype=dtype, device=device)
    if n < 2:
        return D
    g = torch.exp(torch.lgamma(torch.tensor(2.0 - alpha, dtype=dtype, device=device)))
    one_m_a = 1.0 - alpha
    for i in range(1, n):
        ti = t[i]
        for k in range(i):
            tau = (t[k + 1] - t[k]).clamp_min(1e-16)
            w = ((ti - t[k]).clamp_min(0.0).pow(one_m_a) - (ti - t[k + 1]).clamp_min(0.0).pow(one_m_a)) / tau
            coeff = w / g
            D[i, k + 1] += coeff
            D[i, k] -= coeff
    return D
