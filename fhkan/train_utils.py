from __future__ import annotations
from typing import Callable, List, Optional
import torch
import torch.nn as nn
from torch import Tensor

def adam_then_lbfgs(net: nn.Module, loss_fn: Callable[[], Tensor], *, adam_steps: int=150, lbfgs_steps: int=40, adam_lr: float=0.008, lbfgs_lr: float=0.8, lbfgs_max_iter: int=20, clip: float=5.0, log_every: int=25, tag: str='', lbfgs_plateau_loss: Optional[float]=None, lbfgs_plateau_rel: float=0.001) -> List[float]:
    losses: List[float] = []
    opt = torch.optim.Adam(net.parameters(), lr=adam_lr)
    for k in range(adam_steps):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        opt.step()
        losses.append(float(loss.detach()))
        if log_every and k % log_every == 0:
            print(f'  {tag}adam step={k} loss={losses[-1]:.3e}', flush=True)
        if losses[-1] < 1e-12:
            return losses
    if lbfgs_steps <= 0:
        return losses
    opt = torch.optim.LBFGS(net.parameters(), lr=lbfgs_lr, max_iter=lbfgs_max_iter, history_size=50, line_search_fn='strong_wolfe', tolerance_grad=1e-14, tolerance_change=1e-14)
    for k in range(lbfgs_steps):

        def closure():
            opt.zero_grad()
            loss = loss_fn()
            if torch.isfinite(loss):
                loss.backward()
            return loss
        try:
            loss = opt.step(closure)
        except (RuntimeError, ValueError) as exc:
            print(f'  {tag}L-BFGS stopped: {exc}', flush=True)
            break
        val = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
        losses.append(val)
        if k % max(log_every // 5, 1) == 0 or k == 0:
            print(f'  {tag}L-BFGS step={k} loss={val:.3e}', flush=True)
        if val < 1e-12:
            break
        if lbfgs_plateau_loss is not None and val < lbfgs_plateau_loss and (len(losses) >= 2):
            rel = abs(losses[-2] - val) / max(abs(losses[-2]), 1e-16)
            if rel < lbfgs_plateau_rel:
                print(f'  {tag}L-BFGS plateau at step={k} loss={val:.3e} (rel={rel:.2e}); stopping', flush=True)
                break
    return losses
