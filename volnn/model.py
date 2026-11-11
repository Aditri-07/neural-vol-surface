"""Stage 1: a small MLP that maps (log-moneyness, maturity, optional extra features) to implied vol."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _t(a) -> torch.Tensor:
    return torch.tensor(np.array(a, dtype=np.float32))


class VolSurfaceMLP(nn.Module):
    """Neural implied-vol surface.

    Inputs:  [k / sqrt(T), sqrt(T)] plus `n_extra` optional features, all standardised with
             training-set statistics. Scaling k by sqrt(T) makes smiles at different maturities
             look alike, which is most of what lets one small network cover the whole surface.
    Output:  output="iv"      -> sigma = softplus(net)
             output="totvar"  -> total variance w = w_scale * softplus(net), sigma = sqrt(w / T)
    SiLU is smooth, so the network can be differentiated twice in stage 2 (no-arbitrage penalties).
    """

    def __init__(self, hidden: int = 64, depth: int = 3, n_extra: int = 0, output: str = "iv"):
        super().__init__()
        assert output in ("iv", "totvar")
        self.output = output
        layers, d = [], 2 + n_extra
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        self.register_buffer("mu", torch.zeros(2 + n_extra))
        self.register_buffer("sd", torch.ones(2 + n_extra))
        self.register_buffer("w_scale", torch.ones(()))

    @staticmethod
    def features(k, T, extra=None):
        s = torch.sqrt(T)
        x = torch.stack([k / s, s], dim=-1)
        return x if extra is None else torch.cat([x, extra], dim=-1)

    def set_scaler(self, k, T, iv, extra=None):
        x = self.features(k, T, extra)
        self.mu.copy_(x.mean(0))
        self.sd.copy_(x.std(0).clamp_min(1e-6))
        self.w_scale.copy_((iv ** 2 * T).mean())

    def forward(self, k, T, extra=None):
        x = (self.features(k, T, extra) - self.mu) / self.sd
        z = nn.functional.softplus(self.net(x)).squeeze(-1)
        if self.output == "iv":
            return z
        return torch.sqrt(self.w_scale * z / T + 1e-12)


def spread_weights(iv_bid, iv_ask, floor: float = 0.0005) -> np.ndarray:
    """Loss weights 1 / spread^2, so errors are measured in units of the bid-ask width.
    `floor` (0.05 vol points) stops a handful of very tight quotes from dominating."""
    s = np.maximum(np.asarray(iv_ask) - np.asarray(iv_bid), floor)
    return 1.0 / s ** 2


def fit(model, k, T, iv, extra=None, weight=None, val=None,
        epochs: int = 4000, lr: float = 3e-3, seed: int = 0, log_every: int = 500):
    """Full-batch Adam with cosine learning-rate decay. The loss is (weighted) squared error in IV.

    val: optional dict with keys k, T, iv and (if used) extra, for tracking held-out RMSE.
    Returns the loss history (unweighted RMSE in IV units)."""
    torch.manual_seed(seed)
    k, T, iv = _t(k), _t(T), _t(iv)
    extra = None if extra is None else _t(extra)
    w = torch.ones_like(iv) if weight is None else _t(weight)
    w = w / w.mean()
    if val is not None:
        vk, vT, viv = _t(val["k"]), _t(val["T"]), _t(val["iv"])
        vx = None if val.get("extra") is None else _t(val["extra"])

    model.set_scaler(k, T, iv, extra)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    history = {"epoch": [], "train": [], "val": []}
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        loss = (w * (model(k, T, extra) - iv) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
        if ep % 50 == 0 or ep == 1:
            with torch.no_grad():
                tr = torch.sqrt(((model(k, T, extra) - iv) ** 2).mean()).item()
                va = torch.sqrt(((model(vk, vT, vx) - viv) ** 2).mean()).item() if val is not None else float("nan")
            history["epoch"].append(ep); history["train"].append(tr); history["val"].append(va)
            if log_every and ep % log_every == 0:
                print(f"epoch {ep:5d}  train RMSE {100*tr:6.3f}  val RMSE {100*va:6.3f}  (vol points)")
    return history


def predict(model, k, T, extra=None) -> np.ndarray:
    with torch.no_grad():
        return model(_t(k), _t(T), None if extra is None else _t(extra)).numpy()
