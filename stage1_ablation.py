"""Stage 1 ablation: which changes fix the underfit?

Runs a grid of model variants on several days and reports out-of-sample error for each.
Variants differ in loss weighting, output parametrisation, time features and network size.

Usage:
    python vet_data.py                       (once, writes quotes_clean.pkl)
    python stage1_ablation.py --quotes quotes_clean.pkl
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

from volnn.features import TIME_FEATURES, add_time_features
from volnn.model import VolSurfaceMLP, fit, predict, spread_weights

# name -> settings
SMALL = dict(hidden=64, depth=3, epochs=4000)
BIG = dict(hidden=128, depth=4, epochs=8000)
VARIANTS = {
    "A baseline (small)":                               dict(weighted=False, output="iv",     extra=False, **SMALL),
    "B small + spread weights":                         dict(weighted=True,  output="iv",     extra=False, **SMALL),
    "C small + weights + total variance":               dict(weighted=True,  output="totvar", extra=False, **SMALL),
    "D small + weights + time features":                dict(weighted=True,  output="iv",     extra=True,  **SMALL),
    "E big + weights + time features":                  dict(weighted=True,  output="iv",     extra=True,  **BIG),
    "F big network only":                               dict(weighted=False, output="iv",     extra=False, **BIG),
    "G big + spread weights":                           dict(weighted=True,  output="iv",     extra=False, **BIG),
    "H big + weights + total variance":                 dict(weighted=True,  output="totvar", extra=False, **BIG),
    "I big + weights + total variance + time features": dict(weighted=True,  output="totvar", extra=True,  **BIG),
}
DEFAULT_DATES = ["2023-02-15", "2023-04-19", "2023-06-21", "2023-08-16", "2023-10-18", "2023-12-06"]


def split(day: pd.DataFrame, seed: int = 0):
    """Hold out one mid-curve expiry entirely, and a random 20% of the remaining quotes."""
    expiries = np.sort(day.expire_date.unique())
    held = expiries[len(expiries) // 2]
    rng = np.random.default_rng(seed)
    is_exp = (day.expire_date == held).values
    is_val = ~is_exp & (rng.random(len(day)) < 0.20)
    return day[~is_exp & ~is_val], day[is_val], day[is_exp]


def score(pred, q, prefix):
    err = pred - q.iv_mid.values
    spread = np.maximum(q.iv_ask.values - q.iv_bid.values, 1e-6)
    return {f"{prefix}_rmse": 100 * np.sqrt(np.mean(err ** 2)),
            f"{prefix}_inside": 100 * np.mean((pred >= q.iv_bid.values) & (pred <= q.iv_ask.values)),
            f"{prefix}_err_in_spreads": float(np.median(np.abs(err) / spread))}


def poly_benchmark(train, test, deg=4):
    pred = np.full(len(test), np.nan)
    for e, g in train.groupby("expire_date"):
        m = (test.expire_date == e).values
        if m.any() and len(g) > deg + 2:
            pred[m] = np.polyval(np.polyfit(g.k, g.iv_mid, deg), test.k.values[m])
    return pred


def run_one(args):
    date, name, cfg, day = args
    torch.set_num_threads(1)
    train, val, exp = split(day)
    X = (lambda d: d[TIME_FEATURES].values) if cfg["extra"] else (lambda d: None)
    model = VolSurfaceMLP(cfg["hidden"], cfg["depth"], n_extra=len(TIME_FEATURES) if cfg["extra"] else 0,
                          output=cfg["output"])
    t0 = time.time()
    fit(model, train.k, train["T"], train.iv_mid, extra=X(train),
        weight=spread_weights(train.iv_bid, train.iv_ask) if cfg["weighted"] else None,
        epochs=cfg["epochs"], log_every=0)
    row = {"date": date, "variant": name, "seconds": round(time.time() - t0)}
    row |= score(predict(model, train.k, train["T"], X(train)), train, "train")
    row |= score(predict(model, val.k, val["T"], X(val)), val, "val")
    row |= score(predict(model, exp.k, exp["T"], X(exp)), exp, "exp")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quotes", default="quotes_clean.pkl")
    ap.add_argument("--dates", nargs="*", default=DEFAULT_DATES)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="results/stage1_ablation")
    args = ap.parse_args()

    q = pd.read_pickle(args.quotes)
    q = add_time_features(q[q.quote_date.isin(pd.to_datetime(args.dates))])
    days = {d: g.reset_index(drop=True) for d, g in q.groupby(q.quote_date.dt.strftime("%Y-%m-%d"))}

    rows = []
    for d, day in days.items():  # benchmark
        train, val, _ = split(day)
        rows.append({"date": d, "variant": "Polynomial per expiry", "seconds": 0}
                    | score(poly_benchmark(train, train), train, "train")
                    | score(poly_benchmark(train, val), val, "val"))
    jobs = [(d, name, cfg, day) for d, day in days.items() for name, cfg in VARIANTS.items()]
    with ProcessPoolExecutor(args.workers) as pool:
        for r in pool.map(run_one, jobs):
            rows.append(r)
            print(f"{r['date']}  {r['variant']:<50} val RMSE {r['val_rmse']:.3f}  inside {r['val_inside']:.0f}%  "
                  f"expiry RMSE {r['exp_rmse']:.3f}  ({r['seconds']}s)", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(f"{args.out}_by_day.csv", index=False)
    order = ["Polynomial per expiry"] + list(VARIANTS)
    summary = res.drop(columns=["date", "seconds"]).groupby("variant").mean().loc[order]
    summary.to_csv(f"{args.out}_summary.csv")
    pd.set_option("display.width", 250)
    print(f"\nMean over {len(days)} days (RMSE in vol points, inside = % within bid-ask, "
          f"err_in_spreads = median |error| / spread):\n")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
