"""Load and clean OptionsDX SPX end-of-day chains.

Pipeline:
  load_chain()      raw monthly files -> one tidy DataFrame (one row per date/expiry/strike)
  add_forwards()    forward F and discount factor D per (date, expiry) from put-call parity
  build_quotes()    out-of-the-money quotes with our own Black-76 implied vols (bid / mid / ask)
  clean_quotes()    liquidity and sanity filters, each documented and counted
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from scipy.special import ndtr

NUMERIC = [
    "UNDERLYING_LAST", "DTE", "STRIKE",
    "C_BID", "C_ASK", "C_IV", "C_VOLUME",
    "P_BID", "P_ASK", "P_IV", "P_VOLUME",
]


# --------------------------------------------------------------------------- load
def load_chain(path: str) -> pd.DataFrame:
    """Read every spx_eod_*.txt under `path` (or a single file)."""
    files = sorted(glob.glob(os.path.join(path, "spx_eod_*.txt"))) if os.path.isdir(path) else [path]
    if not files:
        raise FileNotFoundError(f"no spx_eod_*.txt files in {path}")
    frames = []
    for f in files:
        df = pd.read_csv(f, skipinitialspace=True, low_memory=False)
        df.columns = [c.strip().strip("[]") for c in df.columns]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["QUOTE_DATE"] = pd.to_datetime(df["QUOTE_DATE"].astype(str).str.strip())
    df["EXPIRE_DATE"] = pd.to_datetime(df["EXPIRE_DATE"].astype(str).str.strip())
    keep = ["QUOTE_DATE", "EXPIRE_DATE", "QUOTE_TIME_HOURS"] + NUMERIC
    df = df[keep].rename(columns=str.lower)
    df["T"] = df["dte"] / 365.0
    return df


# ----------------------------------------------------------------------- forwards
def add_forwards(df: pd.DataFrame, band: float = 0.10, min_strikes: int = 6) -> pd.DataFrame:
    """Implied forward and discount factor per (date, expiry).

    Put-call parity for European options: C - P = D * (F - K).
    Regressing mid(C) - mid(P) on K over near-the-money strikes gives
    slope = -D and intercept = D * F, so no external rate or dividend curve is needed.
    """
    two_sided = (df.c_bid > 0) & (df.p_bid > 0) & (df.c_ask > df.c_bid) & (df.p_ask > df.p_bid)
    near = (df.strike / df.underlying_last - 1).abs() < band
    sub = df[two_sided & near & (df["T"] > 0)].copy()
    sub["cp"] = 0.5 * (sub.c_bid + sub.c_ask) - 0.5 * (sub.p_bid + sub.p_ask)

    rows = []
    for (d, e), g in sub.groupby(["quote_date", "expire_date"], sort=False):
        if len(g) < min_strikes:
            continue
        slope, intercept = np.polyfit(g.strike.values, g.cp.values, 1)
        D = -slope
        if not 0.90 < D < 1.02:
            continue
        rows.append((d, e, intercept / D, D, len(g)))
    fwd = pd.DataFrame(rows, columns=["quote_date", "expire_date", "F", "D", "n_parity"])
    return df.merge(fwd, on=["quote_date", "expire_date"], how="inner")


# ------------------------------------------------------------------- implied vols
def black76(F, K, T, sigma, is_call):
    """Undiscounted Black-76 price."""
    sT = sigma * np.sqrt(T)
    d1 = np.log(F / K) / sT + 0.5 * sT
    d2 = d1 - sT
    call = F * ndtr(d1) - K * ndtr(d2)
    return np.where(is_call, call, call - (F - K))


def implied_vol(price, F, K, T, is_call, lo=1e-4, hi=5.0, iters=60):
    """Vectorised bisection. `price` is undiscounted. Returns NaN where no solution."""
    price, F, K, T = map(np.asarray, (price, F, K, T))
    intrinsic = np.where(is_call, np.maximum(F - K, 0), np.maximum(K - F, 0))
    ok = (price > intrinsic) & (price < np.where(is_call, F, K)) & (T > 0)
    lo_v = np.full(price.shape, lo)
    hi_v = np.full(price.shape, hi)
    with np.errstate(all="ignore"):
        for _ in range(iters):
            mid = 0.5 * (lo_v + hi_v)
            too_low = black76(F, K, T, mid, is_call) < price
            lo_v = np.where(too_low, mid, lo_v)
            hi_v = np.where(too_low, hi_v, mid)
    iv = 0.5 * (lo_v + hi_v)
    iv[~ok | (iv > hi - 1e-3) | (iv < lo + 1e-5)] = np.nan
    return iv


def build_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """One out-of-the-money quote per (date, expiry, strike): puts below F, calls above."""
    is_call = (df.strike >= df.F).values
    bid = np.where(is_call, df.c_bid, df.p_bid)
    ask = np.where(is_call, df.c_ask, df.p_ask)
    q = pd.DataFrame({
        "quote_date": df.quote_date.values,
        "expire_date": df.expire_date.values,
        "dte": df.dte.values,
        "T": df["T"].values,
        "strike": df.strike.values,
        "F": df.F.values,
        "D": df.D.values,
        "is_call": is_call,
        "bid": bid,
        "ask": ask,
        "volume": np.where(is_call, df.c_volume, df.p_volume),
        "vendor_iv": np.where(is_call, df.c_iv, df.p_iv),
    })
    q["mid"] = 0.5 * (q.bid + q.ask)
    q["k"] = np.log(q.strike / q.F)  # log-moneyness
    for name in ("bid", "mid", "ask"):
        q[f"iv_{name}"] = implied_vol(q[name].values / q.D.values, q.F.values, q.strike.values,
                                      q["T"].values, q.is_call.values)
    return q


# ------------------------------------------------------------------------ filters
def clean_quotes(q: pd.DataFrame, min_dte=7, max_dte=365, min_bid=0.10,
                 max_rel_spread=0.50, max_abs_std_k=4.0, verbose=True) -> pd.DataFrame:
    """Apply filters in order and report how many quotes each removes.

    min_dte / max_dte    drop the very short end (0DTE and weeklies behave differently) and LEAPS
    min_bid              drop zero and near-zero bids (no real market on the wing)
    max_rel_spread       drop quotes whose (ask - bid) / mid is too wide to locate a price
    iv solvable          need a valid implied vol at bid, mid and ask
    max_abs_std_k        keep |k| / (atm_vol * sqrt(T)) within a few standard deviations
    """
    steps = []
    n0 = len(q)

    def keep(mask, label):
        nonlocal q
        before = len(q)
        q = q[mask(q)]
        steps.append((label, before - len(q), len(q)))

    keep(lambda x: (x.dte >= min_dte) & (x.dte <= max_dte), f"DTE in [{min_dte}, {max_dte}]")
    keep(lambda x: x.bid >= min_bid, f"bid >= {min_bid}")
    keep(lambda x: (x.ask > x.bid) & ((x.ask - x.bid) / x.mid <= max_rel_spread),
         f"relative spread <= {max_rel_spread}")
    keep(lambda x: x[["iv_bid", "iv_mid", "iv_ask"]].notna().all(axis=1), "implied vol solvable")

    # ATM vol per slice = mid IV at the strike nearest the forward
    idx = q.assign(absk=q.k.abs()).groupby(["quote_date", "expire_date"]).absk.idxmin()
    atm = q.loc[idx, ["quote_date", "expire_date", "iv_mid"]].rename(columns={"iv_mid": "atm_iv"})
    q = q.merge(atm, on=["quote_date", "expire_date"])
    q["std_k"] = q.k / (q.atm_iv * np.sqrt(q["T"]))
    keep(lambda x: x.std_k.abs() <= max_abs_std_k, f"|standardised k| <= {max_abs_std_k}")

    if verbose:
        print(f"{'filter':<34}{'removed':>10}{'left':>10}")
        print(f"{'start':<34}{'':>10}{n0:>10}")
        for label, removed, left in steps:
            print(f"{label:<34}{removed:>10}{left:>10}")
    return q.reset_index(drop=True)


def prepare(path: str, **filter_kwargs) -> pd.DataFrame:
    """Raw files -> cleaned quotes, in one call."""
    return clean_quotes(build_quotes(add_forwards(load_chain(path))), **filter_kwargs)
