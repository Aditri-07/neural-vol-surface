"""Stage 3: one model for the whole year, with a latent state per day.

Three out-of-sample tests:
  1. held-out quotes     random 20% of quotes on training days
  2. held-out expiries   four whole expiries per training day
  3. held-out days       every tenth day is never used to train the network; its latent vector is
                         found afterwards with the network frozen, from part of that day's quotes,
                         and the rest of that day's quotes and expiries are scored

Usage:
    python vet_data.py                       (once, writes quotes_clean.pkl)
    python stage3_latent.py --latent 8
"""
import argparse
import time as clock

import numpy as np
import pandas as pd
import torch

from stage2_arbitrage import split
from volnn.latent import LatentSurface, calendar_violations, encode_day, fit_latent, predict_iv
from volnn.surface import TIME_COLS, calendar_grid, time_inputs


def _t(a, dtype=np.float32):
    return torch.tensor(np.array(a, dtype=dtype))


def tensors(df):
    spread = np.maximum(df.iv_ask.values - df.iv_bid.values, 0.0005)
    return _t(df.k.values), _t(df[TIME_COLS].values), _t(df.iv_mid.values), _t(spread)


def score(pred, q):
    err = pred - q.iv_mid.values
    return 100 * np.sqrt(np.mean(err ** 2)), 100 * np.mean((pred >= q.iv_bid.values) & (pred <= q.iv_ask.values))


def surface_descriptors(day: pd.DataFrame) -> dict:
    """Simple hand-made descriptors of a day's surface, to see what the latent vector encodes."""
    exp = day.groupby("expire_date").agg(T=("T", "first")).reset_index()
    out = {}
    for label, target in [("30d", 30 / 365), ("180d", 180 / 365)]:
        e = exp.iloc[(exp["T"] - target).abs().argmin()].expire_date
        g = day[day.expire_date == e].sort_values("k")
        out[f"atm_{label}"] = float(np.interp(0.0, g.k, g.iv_mid))
        if label == "30d":
            s = out["atm_30d"] * np.sqrt(g["T"].iloc[0])
            out["skew_30d"] = float(np.interp(-s, g.k, g.iv_mid) - np.interp(s, g.k, g.iv_mid))
    out["term_slope"] = out["atm_180d"] - out["atm_30d"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quotes", default="quotes_clean.pkl")
    ap.add_argument("--latent", type=int, default=8)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lam_cal", type=float, default=100.0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"stage3_latent{args.latent}"
    torch.set_num_threads(args.threads)

    q = pd.read_pickle(args.quotes)
    dates = np.sort(q.quote_date.unique())
    test_dates = set(dates[4::10])
    day_frames, grids = {}, {}
    for d, g in q.groupby("quote_date"):
        g = g.reset_index(drop=True)
        g[TIME_COLS] = time_inputs(d, g.expire_date.values)
        day_frames[d] = split(g)                      # (train quotes, held-out quotes, held-out expiries)
        grids[d] = calendar_grid(d)
    G = min(len(g) for g in grids.values())
    train_dates = [d for d in dates if d not in test_dates]
    index = {d: i for i, d in enumerate(train_dates)}
    print(f"{len(train_dates)} training days, {len(test_dates)} held-out days")

    tr = pd.concat([day_frames[d][0].assign(day=index[d]) for d in train_dates], ignore_index=True)
    k, time, iv, spread = tensors(tr)
    day = _t(tr.day.values, np.int64)
    grid_t = _t(np.stack([grids[d][:G] for d in train_dates]))
    print(f"{len(tr):,} training quotes")

    model = LatentSurface(len(train_dates), args.latent, args.hidden)
    t0 = clock.time()
    u_lo, u_hi = fit_latent(model, k, time, iv, spread, day, grid_t, steps=args.steps, lam_cal=args.lam_cal)
    train_minutes = (clock.time() - t0) / 60
    print(f"trained in {train_minutes:.1f} minutes")

    rows = []
    z_mean = model.z.weight.detach().mean(0, keepdim=True)
    for d in dates:
        trn, val, exp = day_frames[d]
        if d in test_dates:   # encode from this day's training quotes only, network frozen
            z = encode_day(model, *tensors(trn), z_init=z_mean)
            kind = "held-out day"
        else:
            z = model.z.weight[index[d]].detach().unsqueeze(0)
            kind = "training day"
        vr, vi = score(predict_iv(model, _t(val.k.values), _t(val[TIME_COLS].values), z), val)
        er, ei = score(predict_iv(model, _t(exp.k.values), _t(exp[TIME_COLS].values), z), exp)
        row = {"date": pd.Timestamp(d).date(), "kind": kind, "val_rmse": vr, "val_inside": vi,
               "exp_rmse": er, "exp_inside": ei,
               "calendar_viol_pct": calendar_violations(model, _t(grids[d]), z, u_lo, u_hi)}
        row |= {f"z{i}": float(z[0, i]) for i in range(args.latent)}
        row |= surface_descriptors(pd.concat([trn, val, exp]))
        rows.append(row)

    res = pd.DataFrame(rows)
    res.to_csv(f"{out}_by_day.csv", index=False)
    torch.save(model.state_dict(), f"{out}_model.pt")
    cols = ["val_rmse", "val_inside", "exp_rmse", "exp_inside", "calendar_viol_pct"]
    summary = res.groupby("kind")[cols].mean().assign(days=res.groupby("kind").size())
    summary.to_csv(f"{out}_summary.csv")
    pd.set_option("display.width", 200)
    print(f"\nlatent dim {args.latent}. Mean over days. RMSE in vol points; inside = % within bid-ask.\n")
    print(summary.round(3).to_string())
    print("\nworst five days by held-out-quote RMSE:")
    print(res.nlargest(5, "val_rmse")[["date", "kind"] + cols[:4]].round(3).to_string(index=False))

    zc = [f"z{i}" for i in range(args.latent)]
    desc = ["atm_30d", "skew_30d", "term_slope"]
    corr = res[zc + desc].corr().loc[zc, desc]
    corr.to_csv(f"{out}_latent_correlations.csv")
    print("\ncorrelation of each latent coordinate with hand-made surface descriptors:")
    print(corr.round(2).to_string())
    # how much of each descriptor a linear combination of the latents explains
    Z = np.column_stack([res[zc].values, np.ones(len(res))])
    r2 = {c: 1 - np.sum((res[c] - Z @ np.linalg.lstsq(Z, res[c], rcond=None)[0]) ** 2)
             / np.sum((res[c] - res[c].mean()) ** 2) for c in desc}
    print("\nR^2 of a linear fit of each descriptor on the latent vector:", {c: round(v, 3) for c, v in r2.items()})


if __name__ == "__main__":
    main()
