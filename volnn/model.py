"""Stage 1: a small MLP that maps (log-moneyness, maturity) to implied vol."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class VolSurfaceMLP(nn.Module):
    """IV(k, T) = softplus(MLP(features)).

    Features are [k / sqrt(T), sqrt(T)], standardised with statistics from the training set.
    Scaling k by sqrt(T) makes smiles at different maturities look alike, which is most of
    what lets one small network cover the whole surface.
    SiLU is smooth, so the network can be differentiated twice in stage 2 (no-arbitrage penalties).
    """

    def __init__(self, hidden: int = 64, depth: int = 3):
        super().__init__()
        layers, d = [], 2
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        self.register_buffer("mu", torch.zeros(2))
        self.register_buffer("sd", torch.ones(2))

    @staticmethod
    def features(k: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        s = torch.sqrt(T)
        return torch.stack([k / s, s], dim=-1)

    def set_scaler(self, k: torch.Tensor, T: torch.Tensor) -> None:
        x = self.features(k, T)
        self.mu.copy_(x.mean(0))
        self.sd.copy_(x.std(0))

    def forward(self, k: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        x = (self.features(k, T) - self.mu) / self.sd
        return nn.functional.softplus(self.net(x)).squeeze(-1)


def fit(model, k, T, iv, k_val=None, T_val=None, iv_val=None, weight=None,
        epochs: int = 4000, lr: float = 3e-3, seed: int = 0, log_every: int = 500):
    """Full-batch Adam with cosine learning-rate decay. Returns the loss history."""
    torch.manual_seed(seed)
    to_t = lambda a: torch.tensor(np.array(a, dtype=np.float32))
    k, T, iv = to_t(k), to_t(T), to_t(iv)
    w = torch.ones_like(iv) if weight is None else to_t(weight)
    w = w / w.mean()
    has_val = k_val is not None
    if has_val:
        k_val, T_val, iv_val = to_t(k_val), to_t(T_val), to_t(iv_val)

    model.set_scaler(k, T)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    history = {"epoch": [], "train": [], "val": []}
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        loss = (w * (model(k, T) - iv) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
        if ep % 50 == 0 or ep == 1:
            with torch.no_grad():
                tr = torch.sqrt(((model(k, T) - iv) ** 2).mean()).item()
                va = torch.sqrt(((model(k_val, T_val) - iv_val) ** 2).mean()).item() if has_val else float("nan")
            history["epoch"].append(ep); history["train"].append(tr); history["val"].append(va)
            if log_every and ep % log_every == 0:
                print(f"epoch {ep:5d}  train RMSE {100*tr:6.3f}  val RMSE {100*va:6.3f}  (vol points)")
    return history


def predict(model, k, T) -> np.ndarray:
    with torch.no_grad():
        return model(torch.tensor(np.array(k, dtype=np.float32)),
                     torch.tensor(np.array(T, dtype=np.float32))).numpy()
