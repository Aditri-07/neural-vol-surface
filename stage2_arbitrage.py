"""Stage 2: do no-arbitrage penalties and structure across maturity fix unseen expiries?

For each day: hold out four whole expiries spread along the curve plus a random 20% of the remaining
quotes, fit each variant, and report fit quality and how much static arbitrage the surface contains.

Usage:
    python vet_data.py                       (once, writes quotes_clean.pkl)
    python stage2_arbitrage.py --quotes quotes_clean.pkl
"""
import argparse
import time as clock
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

from volnn.surface import (TIME_COLS, TotalVarianceSurface, arbitrage_report, calendar_grid, fit_surface,
                           predict_iv, time_inputs)

LAM_CAL, LAM_BF = 100.0, 10.0   # not tuned
VARIANTS = {
    "free, no penalty":                  dict(mode="free",       lam_cal=0.0,     lam_bf=0.0),
    "free + calendar":                   dict(mode="free",       lam_cal=LAM_CAL, lam_bf=0.0),
    "free + calendar + butterfly":       dict(mode="free",       lam_cal=LAM_CAL, lam_bf=LAM_BF),
    "structured, no penalty":            dict(mode="structured", lam_cal=0.0,     lam_bf=0.0),
    "structured + calendar + butterfly": dict(mode="structured", lam_cal=LAM_CAL, lam_bf=LAM_BF),
}
DEFAULT_DATES = ["2023-02-15", "2023-06-21", "2023-10-18"]


def split(day: pd.DataFrame, seed: int = 0):
    """Hold out four expiries (at 20/40/60/80% along the curve) and a random 20% of the other quotes."""
    expiries = np.sort(day.expire_date.unique())
    held = expiries[(np.array([0.2, 0.4, 0.6, 0.8]) * (len(expiries) - 1)).round().astype(int)]
    rng = np.random.default_rng(seed)
    is_exp = day.expire_date.isin(held).values
    is_val = ~is_exp & (rng.random(len(day)) < 0.20)
    return day[~is_exp & ~is_val], day[is_val], day[is_exp]


def score(pred, q, prefix):
    err = pred - q.iv_mid.values
    return {f"{prefix}_rmse": 100 * np.sqrt(np.mean(err ** 2)),
            f"{prefix}_inside": 100 * np.mean((pred >= q.iv_bid.values) & (pred <= q.iv_ask.values))}


def run_one(args):
    date, name, cfg, day, epochs = args
    torch.set_num_threads(1)
    train, val, exp = split(day)
    grid = calendar_grid(date)
    model = TotalVarianceSurface(cfg["mode"])
    t0 = clock.time()
    fit_surface(model, train.k, train[TIME_COLS].values, train.iv_mid, train.iv_bid, train.iv_ask, grid,
                lam_cal=cfg["lam_cal"], lam_bf=cfg["lam_bf"], epochs=epochs)
    u = train.k / np.sqrt(train["T"])
    row = {"date": date, "variant": name, "seconds": round(clock.time() - t0)}
    row |= score(predict_iv(model, val.k, val[TIME_COLS].values), val, "val")
    row |= score(predict_iv(model, exp.k, exp[TIME_COLS].values), exp, "exp")
    row |= arbitrage_report(model, grid, float(u.min()), float(u.max()))
    if cfg["mode"] == "structured":
        v = model.event_variance().detach().numpy()
        row |= {"fomc_move_pct": 100 * float(np.sqrt(v[0])), "cpi_move_pct": 100 * float(np.sqrt(v[1]))}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quotes", default="quotes_clean.pkl")
    ap.add_argument("--dates", nargs="*", default=DEFAULT_DATES)
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="stage2")
    args = ap.parse_args()

    q = pd.read_pickle(args.quotes)
    q = q[q.quote_date.isin(pd.to_datetime(args.dates))].copy()
    days = {}
    for d, g in q.groupby(q.quote_date.dt.strftime("%Y-%m-%d")):
        g = g.reset_index(drop=True)
        g[TIME_COLS] = time_inputs(d, g.expire_date.values)
        days[d] = g

    jobs = [(d, name, cfg, day, args.epochs) for d, day in days.items() for name, cfg in VARIANTS.items()]
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for r in pool.map(run_one, jobs):
            rows.append(r)
            print(f"{r['date']}  {r['variant']:<34} quotes {r['val_rmse']:.3f} ({r['val_inside']:.0f}%)  "
                  f"expiries {r['exp_rmse']:.3f} ({r['exp_inside']:.0f}%)  "
                  f"cal viol {r['calendar_viol_pct']:.1f}%  bf viol {r['butterfly_viol_pct']:.1f}%  "
                  f"({r['seconds']}s)", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(f"{args.out}_by_day.csv", index=False)
    summary = res.drop(columns=["date", "seconds"]).groupby("variant").mean().loc[list(VARIANTS)]
    summary.to_csv(f"{args.out}_summary.csv")
    pd.set_option("display.width", 250)
    print(f"\nMean over {len(days)} days. RMSE in vol points; (%) = share inside bid-ask; "
          f"viol = share of a dense grid violating the condition.\n")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
