"""Stage 1: fit a neural implied-vol surface to one day of SPX quotes.

Usage:
    python stage1_fit_one_day.py --data data/spx_2023q1 --date 2023-02-15

Model: 4 x 128 MLP, total-variance output, loss weighted by 1 / bid-ask width^2, with time features
(trading-day time, FOMC and CPI events before expiry). This is the best variant from
stage1_ablation.py for held-out quotes. Pass --baseline for the original small plain MLP.

Two out-of-sample tests:
  1. random 20% of quotes held out            (interpolation within smiles)
  2. one whole expiry held out                (interpolation across maturity - the harder test)
Benchmark: a separate quartic polynomial in log-moneyness per expiry (it cannot do test 2).
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from volnn.data import prepare
from volnn.features import TIME_FEATURES, add_time_features
from volnn.model import VolSurfaceMLP, fit, predict, spread_weights

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def metrics(pred, q):
    err = pred - q.iv_mid.values
    inside = (pred >= q.iv_bid.values) & (pred <= q.iv_ask.values)
    return {"n": len(q), "rmse_volpts": 100 * np.sqrt(np.mean(err ** 2)),
            "mae_volpts": 100 * np.mean(np.abs(err)), "inside_bidask_pct": 100 * inside.mean()}


def poly_benchmark(train, test, deg=4):
    """Per-expiry polynomial in k, fitted on train quotes, evaluated on test quotes."""
    pred = np.full(len(test), np.nan)
    for e, g in train.groupby("expire_date"):
        m = (test.expire_date == e).values
        if m.any() and len(g) > deg + 2:
            pred[m] = np.polyval(np.polyfit(g.k, g.iv_mid, deg), test.k.values[m])
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/spx_2023q1")
    ap.add_argument("--date", default="2023-02-15")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--baseline", action="store_true", help="original 3 x 64 MLP, unweighted, no time features")
    ap.add_argument("--out", default="results/stage1")
    args = ap.parse_args()

    q = prepare(args.data)
    day = add_time_features(q[q.quote_date == pd.Timestamp(args.date)]).reset_index(drop=True)
    if day.empty:
        raise SystemExit(f"no quotes on {args.date}")
    expiries = np.sort(day.expire_date.unique())
    print(f"\n{args.date}: {len(day)} clean quotes, {len(expiries)} expiries, "
          f"DTE {day.dte.min():.0f}-{day.dte.max():.0f}")

    # hold out one mid-curve expiry entirely, then a random 20% of the rest
    held_exp = expiries[len(expiries) // 2]
    rng = np.random.default_rng(0)
    is_exp = (day.expire_date == held_exp).values
    is_val = ~is_exp & (rng.random(len(day)) < 0.20)
    train, val, exp = day[~is_exp & ~is_val], day[is_val], day[is_exp]

    if args.baseline:
        X = lambda d: None
        model, weight, epochs = VolSurfaceMLP(), None, args.epochs or 4000
    else:
        X = lambda d: d[TIME_FEATURES].values
        model = VolSurfaceMLP(hidden=128, depth=4, n_extra=len(TIME_FEATURES), output="totvar")
        weight, epochs = spread_weights(train.iv_bid, train.iv_ask), args.epochs or 8000
    hist = fit(model, train.k, train["T"], train.iv_mid, extra=X(train), weight=weight,
               val=dict(k=val.k, T=val["T"], iv=val.iv_mid, extra=X(val)), epochs=epochs, log_every=1000)

    rows = {
        "NN  train": metrics(predict(model, train.k, train["T"], X(train)), train),
        "NN  held-out quotes": metrics(predict(model, val.k, val["T"], X(val)), val),
        "NN  held-out expiry": metrics(predict(model, exp.k, exp["T"], X(exp)), exp),
        "Poly train": metrics(poly_benchmark(train, train), train),
        "Poly held-out quotes": metrics(poly_benchmark(train, val), val),
    }
    table = pd.DataFrame(rows).T
    table["n"] = table["n"].astype(int)
    print("\n" + table.round(3).to_string())
    print(f"\nmedian bid-ask width: {100 * (day.iv_ask - day.iv_bid).median():.3f} vol points")
    print(f"held-out expiry: {pd.Timestamp(held_exp).date()} ({exp.dte.iloc[0]:.0f} DTE)")
    table.to_csv(f"{args.out}_metrics.csv")

    # ---- figure 1: smiles, market band vs network
    show = list(expiries[np.linspace(0, len(expiries) - 1, 5).round().astype(int)])
    if held_exp not in show:
        show[2] = held_exp
    show = sorted(set(show))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), facecolor=SURFACE)
    axes = axes.ravel()
    for ax, e in zip(axes, show):
        g = day[day.expire_date == e].sort_values("k")
        kk = np.linspace(g.k.min(), g.k.max(), 200)
        xx = None if args.baseline else np.tile(g[TIME_FEATURES].iloc[0].values, (len(kk), 1))
        ax.fill_between(g.k, 100 * g.iv_bid, 100 * g.iv_ask, color=MUTED, alpha=0.30, lw=0,
                        label="Market bid-ask")
        ax.plot(kk, 100 * predict(model, kk, np.full_like(kk, g["T"].iloc[0]), xx), color=BLUE, lw=2,
                label="Neural network")
        tag = "  (held out of training)" if e == held_exp else ""
        ax.set_title(f"{pd.Timestamp(e).date()}  ·  {g.dte.iloc[0]:.0f} days{tag}", fontsize=10,
                     color=INK, loc="left")
    # last panel: training curve
    ax = axes[len(show)]
    ax.plot(hist["epoch"], 100 * np.array(hist["train"]), color=BLUE, lw=2, label="Train")
    ax.plot(hist["epoch"], 100 * np.array(hist["val"]), color=ORANGE, lw=2, label="Held-out quotes")
    ax.set_yscale("log")
    ax.set_title("Training curve  ·  RMSE, vol points", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("Epoch", color=MUTED)
    ax.legend(frameon=False, fontsize=9)
    for a in axes[:len(show)]:
        a.set_xlabel("Log-moneyness  ln(K / F)", color=MUTED)
        a.set_ylabel("Implied vol, %", color=MUTED)
    axes[0].legend(frameon=False, fontsize=9)
    for a in axes[len(show) + 1:]:
        a.axis("off")
    for a in axes:
        a.set_facecolor(SURFACE)
        a.grid(True, color=GRID, lw=0.8)
        a.tick_params(colors=MUTED, labelsize=9)
        for s in a.spines.values():
            s.set_visible(False)
    fig.suptitle(f"SPX implied-vol surface on {args.date}: one MLP fitted to all expiries",
                 fontsize=13, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f"{args.out}_smiles.png", dpi=150, facecolor=SURFACE)
    print(f"\nsaved {args.out}_smiles.png and {args.out}_metrics.csv")


if __name__ == "__main__":
    main()
