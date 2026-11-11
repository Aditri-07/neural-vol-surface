"""Stage 2: total-variance surfaces with static no-arbitrage penalties.

Notation: k = ln(K / F), T = calendar time to expiry, w(k, T) = sigma^2 * T (total variance).

No static arbitrage (Gatheral & Jacquier 2014) requires
  calendar:   w non-decreasing in T at fixed k
  butterfly:  g(k) = (1 - k w' / (2w))^2 - (w'^2 / 4)(1 / w + 1/4) + w'' / 2 >= 0,   ' = d/dk
The second condition is the same as the risk-neutral density being non-negative.

Both are added to the loss as soft penalties on points where no quote is needed, following
Ackerer, Tagasovska & Vatter, "Deep Smoothing of the Implied Volatility Surface" (2020).

Every model takes k and a "time" tensor with columns [T, T_trading, n_fomc, n_cpi]. These depend only
on the quote date and the expiry date, so they can be built for any calendar date, quoted or not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .features import CPI, FOMC, HOLIDAYS, _count_between

TIME_COLS = ["T", "T_trading", "n_fomc", "n_cpi"]


def _t(a) -> torch.Tensor:
    return torch.tensor(np.array(a, dtype=np.float32))


# ------------------------------------------------------------------ time inputs
def time_inputs(quote_date, expire_dates) -> np.ndarray:
    """Columns [T, T_trading, n_fomc, n_cpi] for one quote date and an array of expiry dates."""
    end = np.asarray(pd.to_datetime(expire_dates)).astype("datetime64[D]")
    start = np.full(end.shape, np.datetime64(pd.Timestamp(quote_date), "D"))
    one = np.timedelta64(1, "D")
    hol = HOLIDAYS.values.astype("datetime64[D]")
    T = (end - start).astype(float) / 365.0
    T_tr = np.busday_count(start + one, end + one, holidays=hol) / 252.0
    return np.column_stack([T, T_tr, _count_between(FOMC, start, end), _count_between(CPI, start, end)])


def calendar_grid(quote_date, min_days=7, max_days=365) -> np.ndarray:
    """Time inputs for every trading day between min_days and max_days ahead, in order.
    Used as collocation dates: the penalties are enforced here whether or not an expiry is listed."""
    q = pd.Timestamp(quote_date)
    days = pd.bdate_range(q + pd.Timedelta(days=min_days), q + pd.Timedelta(days=max_days))
    return time_inputs(q, days.difference(HOLIDAYS))


# ------------------------------------------------------------------------ models
class TotalVarianceSurface(nn.Module):
    """w(k, time) >= 0.

    mode="free":        w = w_scale * softplus(net(k/sqrt(T), sqrt(T), T_trading/T, n_fomc/T, n_cpi/T))
                        The network is free to use the time features however it likes (stage 1, variant I).
    mode="structured":  w = w_scale * softplus(net(k/sqrt(tau), sqrt(tau))) + v_fomc * n_fomc + v_cpi * n_cpi
                        with tau = T_trading. Variance accrues smoothly in trading time, and each scheduled
                        event adds a fixed, learned chunk of variance. Two event parameters instead of an
                        arbitrary function of the time features.
    """

    def __init__(self, mode: str = "free", hidden: int = 128, depth: int = 4):
        super().__init__()
        assert mode in ("free", "structured")
        self.mode = mode
        d_in = 5 if mode == "free" else 2
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        self.register_buffer("mu", torch.zeros(d_in))
        self.register_buffer("sd", torch.ones(d_in))
        self.register_buffer("w_scale", torch.ones(()))
        if mode == "structured":
            # softplus(-3) ~ 0.05, i.e. events start small; multiplied by an event-variance unit
            self.event_raw = nn.Parameter(torch.full((2,), -3.0))
            self.register_buffer("event_unit", torch.ones(()))

    def _features(self, k, time):
        T, tau = time[:, 0], time[:, 1]
        if self.mode == "free":
            s = torch.sqrt(T)
            return torch.stack([k / s, s, tau / T, time[:, 2] / T, time[:, 3] / T], dim=-1)
        s = torch.sqrt(tau)
        return torch.stack([k / s, s], dim=-1)

    def set_scaler(self, k, time, iv):
        x = self._features(k, time)
        self.mu.copy_(x.mean(0))
        self.sd.copy_(x.std(0).clamp_min(1e-6))
        self.w_scale.copy_((iv ** 2 * time[:, 0]).mean())
        if self.mode == "structured":
            # unit = variance of a 1% one-day move, so softplus(event_raw) is in (1% move)^2 units
            self.event_unit.copy_(torch.tensor(1e-4))

    def event_variance(self) -> torch.Tensor:
        """[v_fomc, v_cpi] in total-variance units (structured mode only)."""
        return nn.functional.softplus(self.event_raw) * self.event_unit

    def total_variance(self, k, time):
        x = (self._features(k, time) - self.mu) / self.sd
        w = self.w_scale * nn.functional.softplus(self.net(x)).squeeze(-1)
        if self.mode == "structured":
            v = self.event_variance()
            w = w + v[0] * time[:, 2] + v[1] * time[:, 3]
        return w + 1e-10

    def forward(self, k, time):
        """Implied vol."""
        return torch.sqrt(self.total_variance(k, time) / time[:, 0])


# --------------------------------------------------------------------- penalties
def butterfly_g(model, k, time):
    """Durrleman's g(k). Negative values mean butterfly arbitrage. Uses autograd for w' and w''."""
    k = k.clone().requires_grad_(True)
    w = model.total_variance(k, time)
    dw = torch.autograd.grad(w.sum(), k, create_graph=True)[0]
    d2w = torch.autograd.grad(dw.sum(), k, create_graph=True)[0]
    return (1 - k * dw / (2 * w)) ** 2 - dw ** 2 / 4 * (1 / w + 0.25) + d2w / 2


def calendar_drop(model, k, time_a, time_b):
    """Relative fall in total variance from date a to the next date b at the same k.
    Positive values mean calendar arbitrage."""
    wa, wb = model.total_variance(k, time_a), model.total_variance(k, time_b)
    return torch.relu(wa - wb) / wa.detach()


# ---------------------------------------------------------------------- training
def fit_surface(model, k, time, iv, iv_bid, iv_ask, grid, lam_cal=0.0, lam_bf=0.0,
                epochs=5000, lr=3e-3, n_colloc=512, seed=0, log_every=0):
    """Full-batch Adam on  data + lam_cal * calendar + lam_bf * butterfly.

    data      mean of ((sigma_hat - sigma_mid) / spread)^2, so it is in units of the bid-ask width
    calendar  mean relative fall in w between consecutive trading days of `grid`, at random k
    butterfly mean of max(0, -g) at random (date, k) points of `grid`
    """
    torch.manual_seed(seed)
    k, time, iv = _t(k), _t(time), _t(iv)
    spread = torch.clamp(_t(iv_ask) - _t(iv_bid), min=0.0005)
    grid = _t(grid)
    model.set_scaler(k, time, iv)
    u = k / torch.sqrt(time[:, 0])                       # standardised moneyness range of the data
    u_lo, u_hi = u.min().item(), u.max().item()
    use_pen = lam_cal > 0 or lam_bf > 0

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        data = (((model(k, time) - iv) / spread) ** 2).mean()
        cal = bf = torch.zeros(())
        if use_pen:
            j = torch.randint(0, len(grid) - 1, (n_colloc,))
            uu = u_lo + (u_hi - u_lo) * torch.rand(n_colloc)
            kk = uu * torch.sqrt(grid[j, 0])
            if lam_cal > 0:
                cal = calendar_drop(model, kk, grid[j], grid[j + 1]).mean()
            if lam_bf > 0:
                bf = torch.relu(-butterfly_g(model, kk, grid[j])).mean()
        (data + lam_cal * cal + lam_bf * bf).backward()
        opt.step()
        sched.step()
        if log_every and ep % log_every == 0:
            print(f"epoch {ep:5d}  data {data.item():.4f}  calendar {cal.item():.2e}  butterfly {bf.item():.2e}")
    return model


def predict_iv(model, k, time) -> np.ndarray:
    with torch.no_grad():
        return model(_t(k), _t(time)).numpy()


# ------------------------------------------------------------------- diagnostics
def arbitrage_report(model, grid, u_lo, u_hi, n_k=41) -> dict:
    """Share of a dense (date, k) grid where the fitted surface violates each condition."""
    grid = _t(grid)
    uu = torch.linspace(u_lo, u_hi, n_k)
    j = torch.arange(len(grid) - 1).repeat_interleave(n_k)
    kk = uu.repeat(len(grid) - 1) * torch.sqrt(grid[j, 0])
    with torch.no_grad():
        drop = calendar_drop(model, kk, grid[j], grid[j + 1])
    g = butterfly_g(model, kk, grid[j]).detach()
    return {"calendar_viol_pct": 100 * (drop > 1e-4).float().mean().item(),
            "butterfly_viol_pct": 100 * (g < -1e-4).float().mean().item(),
            "worst_calendar_drop_pct": 100 * drop.max().item(),
            "min_g": g.min().item()}
