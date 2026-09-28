from __future__ import annotations
import math
import torch
from torch import Tensor

def fractional_map(x: Tensor, gamma: Tensor) -> Tensor:
    gamma = torch.as_tensor(gamma, device=x.device, dtype=x.dtype)
    ax = x.abs().clamp_min(1e-12)
    return torch.sign(x) * ax.pow(gamma)

def _parity_term(n: int, mu: Tensor) -> Tensor:
    return n + mu * (1.0 - (-1.0) ** n)

def generalized_hermite_polynomials(x: Tensor, degree: int, mu: Tensor) -> Tensor:
    if degree < 0:
        raise ValueError('degree must be non-negative')
    mu = torch.as_tensor(mu, device=x.device, dtype=x.dtype)
    h0 = torch.ones_like(x)
    if degree == 0:
        return h0.unsqueeze(-1)
    h1 = 2.0 * x
    terms = [h0, h1]
    for n in range(1, degree):
        coeff = _parity_term(n, mu)
        h_next = 2.0 * x * terms[-1] - 2.0 * coeff * terms[-2]
        terms.append(h_next)
    return torch.stack(terms, dim=-1)

def hermite_normalization(degree: int, mu: Tensor) -> Tensor:
    mu = torch.as_tensor(mu)
    norms = []
    log2 = math.log(2.0)
    for n in range(degree + 1):
        k = n // 2
        if n % 2 == 0:
            log_rho = 4 * k * log2 + math.lgamma(k + 1) + torch.lgamma(k + mu + 0.5)
        else:
            log_rho = (4 * k + 2) * log2 + math.lgamma(k + 1) + torch.lgamma(k + mu + 1.5)
        norms.append(torch.exp(0.5 * log_rho))
    return torch.stack(norms, dim=-1)

def generalized_hermite_functions(x: Tensor, degree: int, mu: Tensor, *, normalize: bool=True) -> Tensor:
    mu = torch.as_tensor(mu, device=x.device, dtype=x.dtype)
    polys = generalized_hermite_polynomials(x, degree, mu)
    envelope = x.abs().clamp_min(1e-12).pow(mu) * torch.exp(-0.5 * x.square())
    psi = envelope.unsqueeze(-1) * polys
    if normalize:
        scale = hermite_normalization(degree, mu)
        while scale.ndim < psi.ndim:
            scale = scale.unsqueeze(0)
        psi = psi / scale.to(dtype=psi.dtype, device=psi.device)
    return psi

def fractional_hermite_functions(x: Tensor, degree: int, mu: Tensor, gamma: Tensor, *, normalize: bool=True) -> Tensor:
    return generalized_hermite_functions(fractional_map(x, gamma), degree, mu, normalize=normalize)

def hermite_gauss_nodes(n: int, device=None, dtype=None) -> Tensor:
    import numpy as np
    from numpy.polynomial.hermite import hermgauss
    x, _w = hermgauss(n)
    return torch.as_tensor(x, device=device, dtype=dtype or torch.float32)

def dehghan_map(zeta: Tensor, k: float=1.0) -> Tensor:
    arg = (k * zeta).clamp_min(1e-06)
    return torch.log(torch.sinh(arg))

def dehghan_map_inverse(omega: Tensor, k: float=1.0) -> Tensor:
    e = torch.exp(omega)
    return torch.log(e + torch.sqrt(e.square() + 1.0)) / k
