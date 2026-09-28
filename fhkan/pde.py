from __future__ import annotations
import math
import numpy as np
import torch
from torch import Tensor
from scipy.special import gammaln

def burgers_exact(z: Tensor, t: Tensor, m0: float, m1: float, m2: float) -> Tensor:
    return m2 / m0 - 2.0 * (m1 / m0) * torch.tanh(z - m2 * t)

def burgers_residual(chi: Tensor, z: Tensor, t: Tensor, m0: float, m1: float) -> Tensor:
    ones = torch.ones_like(chi)
    chi_z = torch.autograd.grad(chi, z, ones, create_graph=True)[0]
    chi_t = torch.autograd.grad(chi, t, ones, create_graph=True)[0]
    chi_zz = torch.autograd.grad(chi_z, z, ones, create_graph=True)[0]
    return chi_t + m0 * chi * chi_z - m1 * chi_zz

def lane_emden_exact(m: int, zeta: Tensor) -> Tensor:
    z = zeta
    if m == 0:
        return 1.0 - z.square() / 6.0
    if m == 1:
        return torch.where(z.abs() < 1e-12, torch.ones_like(z), torch.sin(z) / z)
    if m == 5:
        return (1.0 + z.square() / 3.0).pow(-0.5)
    raise ValueError(f'no closed form for m={m}')
HOREDT_FIRST_ROOT = {0: math.sqrt(6.0), 1: math.pi, 2: 4.3528745959, 3: 6.8968486194, 4: 14.9715463488}

def fde_forcing(zeta: Tensor, alpha: float=0.3) -> Tensor:
    n = 3.0
    caputo_power = math.gamma(n + 1.0) / math.gamma(n + 1.0 - alpha) * zeta.clamp_min(0.0).pow(n - alpha)
    return 1.0 - 3.0 * zeta + 3.0 * zeta.square() + caputo_power

def fde_delay_history(zeta: Tensor) -> Tensor:
    return (zeta - 1.0) ** 3

def mittag_leffler_e_alpha(z: np.ndarray, alpha: float, k_max: int=250) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    acc = np.zeros_like(z, dtype=np.float64)
    for k in range(int(k_max)):
        coeff = float(np.exp(-gammaln(alpha * k + 1.0)))
        if k == 0:
            term = np.full_like(z, coeff)
        else:
            term = coeff * np.power(z, k)
        acc = acc + term
        if np.max(np.abs(term)) < 1e-16 * (1.0 + np.max(np.abs(acc))):
            break
    return acc

def caputo_relaxation_exact(t: Tensor, alpha: float=0.4) -> Tensor:
    t_np = t.detach().cpu().numpy()
    z = -np.clip(t_np, 0.0, None) ** alpha
    y = mittag_leffler_e_alpha(z, alpha)
    return torch.as_tensor(y, device=t.device, dtype=t.dtype)
