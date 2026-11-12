"""Stage 3: one decoder network for all days, with a small latent vector per day.

    w(k, time, day) = w_scale * softplus( net([k/sqrt(T), sqrt(T), T_trading/T, n_fomc/T, n_cpi/T, z_day]) )

z_day is a learned vector (the "state" of the surface that day). The network weights are shared by
every day, so everything that is common to all days (smile shape, how events enter, how weekends
enter) is learned from all of them, and everything that is specific to a day has to fit in z_day.

This is an auto-decoder: there is no encoder network. During training each day's z is a free
parameter updated by backpropagation along with the weights. To "encode" a day the model has never
seen, freeze the network and optimise only that day's z (see `encode_day`). A classical autoencoder
would instead need the surface resampled onto a fixed grid to feed an encoder; this avoids that and
works directly on irregular quotes.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn



def _t(a, dtype=np.float32) -> torch.Tensor:
    return torch.tensor(np.array(a, dtype=dtype))


class LatentSurface(nn.Module):
    def __init__(self, n_days: int, latent_dim: int = 4, hidden: int = 128, depth: int = 4):
        super().__init__()
        self.latent_dim = latent_dim
        self.z = nn.Embedding(n_days, latent_dim)
        nn.init.normal_(self.z.weight, std=0.01)
        layers, d = [], 5 + latent_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        self.register_buffer("mu", torch.zeros(5))
        self.register_buffer("sd", torch.ones(5))
        self.register_buffer("w_scale", torch.ones(()))

    @staticmethod
    def _features(k, time):
        T = time[:, 0]
        s = torch.sqrt(T)
        return torch.stack([k / s, s, time[:, 1] / T, time[:, 2] / T, time[:, 3] / T], dim=-1)

    def set_scaler(self, k, time, iv):
        x = self._features(k, time)
        self.mu.copy_(x.mean(0))
        self.sd.copy_(x.std(0).clamp_min(1e-6))
        self.w_scale.copy_((iv ** 2 * time[:, 0]).mean())

    def total_variance(self, k, time, z):
        """z has one row per point (use self.z(day_index) for training days)."""
        x = torch.cat([(self._features(k, time) - self.mu) / self.sd, z], dim=-1)
        return self.w_scale * nn.functional.softplus(self.net(x)).squeeze(-1) + 1e-10

    def iv(self, k, time, z):
        return torch.sqrt(self.total_variance(k, time, z) / time[:, 0])


def fit_latent(model, k, time, iv, spread, day, grids, steps=8000, batch=16384, lr=3e-3,
               lam_cal=100.0, lam_z=1e-4, n_colloc=512, seed=0, log_every=1000):
    """Mini-batch Adam on  data + lam_cal * calendar + lam_z * |z|^2.

    k, time, iv, spread, day: tensors over all training quotes (day = integer day index)
    grids: tensor [n_days, G, 4], calendar grid of time inputs for each day (for the calendar penalty)
    """
    torch.manual_seed(seed)
    model.set_scaler(k, time, iv)
    u = k / torch.sqrt(time[:, 0])
    u_lo, u_hi = torch.quantile(u[::50], 0.005).item(), torch.quantile(u[::50], 0.995).item()
    n, n_days, G = len(k), grids.shape[0], grids.shape[1]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.01)
    perm, pos = torch.randperm(n), 0
    for step in range(1, steps + 1):
        if pos + batch > n:
            perm, pos = torch.randperm(n), 0
        idx = perm[pos:pos + batch]
        pos += batch
        opt.zero_grad()
        z = model.z(day[idx])
        data = (((model.iv(k[idx], time[idx], z) - iv[idx]) / spread[idx]) ** 2).mean()
        cal = torch.zeros(())
        if lam_cal > 0:
            d = torch.randint(0, n_days, (n_colloc,))
            j = torch.randint(0, G - 1, (n_colloc,))
            ta, tb = grids[d, j], grids[d, j + 1]
            kk = (u_lo + (u_hi - u_lo) * torch.rand(n_colloc)) * torch.sqrt(ta[:, 0])
            zc = model.z(d)
            wa, wb = model.total_variance(kk, ta, zc), model.total_variance(kk, tb, zc)
            cal = (torch.relu(wa - wb) / wa.detach()).mean()
        loss = data + lam_cal * cal + lam_z * (z ** 2).sum(-1).mean()
        loss.backward()
        opt.step()
        sched.step()
        if log_every and step % log_every == 0:
            print(f"step {step:6d}  data {data.item():.4f}  calendar {cal.item():.2e}", flush=True)
    return u_lo, u_hi


def encode_day(model, k, time, iv, spread, z_init, steps=300, lr=0.05):
    """Find the latent vector for one day with the network frozen. Returns a 1-D tensor."""
    z = z_init.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    for p in model.parameters():
        p.requires_grad_(False)
    for _ in range(steps):
        opt.zero_grad()
        loss = (((model.iv(k, time, z.expand(len(k), -1)) - iv) / spread) ** 2).mean()
        loss.backward()
        opt.step()
    for p in model.parameters():
        p.requires_grad_(True)
    return z.detach()


def predict_iv(model, k, time, z) -> np.ndarray:
    with torch.no_grad():
        return model.iv(k, time, z.expand(len(k), -1)).numpy()


def calendar_violations(model, grid, z, u_lo, u_hi, n_k=41) -> float:
    """Share of a dense (date, k) grid where total variance falls from one trading day to the next."""
    uu = torch.linspace(u_lo, u_hi, n_k)
    j = torch.arange(len(grid) - 1).repeat_interleave(n_k)
    kk = uu.repeat(len(grid) - 1) * torch.sqrt(grid[j, 0])
    with torch.no_grad():
        zz = z.expand(len(kk), -1)
        wa, wb = model.total_variance(kk, grid[j], zz), model.total_variance(kk, grid[j + 1], zz)
    return 100 * ((wa - wb) / wa > 1e-4).float().mean().item()
