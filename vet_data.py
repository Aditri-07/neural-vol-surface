"""Vet OptionsDX SPX end-of-day files month by month.

Usage:
    python vet_data.py data/spx_2023q1 data/spx_2023q2 ...   (folders holding spx_eod_YYYYMM.txt)

Writes vetting_by_month.csv, missing_days.csv and the cleaned quotes (quotes_clean.pkl).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

from volnn.data import add_forwards, build_quotes, clean_quotes, load_chain

# NYSE full-day closures in 2023
HOLIDAYS_2023 = pd.to_datetime(["2023-01-02", "2023-01-16", "2023-02-20", "2023-04-07", "2023-05-29",
                                "2023-06-19", "2023-07-04", "2023-09-04", "2023-11-23", "2023-12-25"])


def expected_days(month: pd.Period) -> pd.DatetimeIndex:
    days = pd.bdate_range(month.start_time, month.end_time)
    return days.difference(HOLIDAYS_2023)


def butterfly_stats(q: pd.DataFrame) -> tuple[float, float]:
    """Share of strike triples where call prices are not convex in strike.

    at_mid:      convexity fails using mid prices (noise or arbitrage)
    executable:  still fails buying the wings at the ask and selling the body at the bid
                 (a real, tradeable arbitrage - should be essentially zero in clean data)
    """
    n = at_mid = executable = 0
    for _, g in q.groupby(["quote_date", "expire_date"], sort=False):
        if len(g) < 3:
            continue
        g = g.sort_values("strike")
        K = g.strike.values
        # undiscounted call prices; puts converted through parity C = P + (F - K)
        shift = np.where(g.is_call.values, 0.0, g.F.values - K)
        mid, bid, ask = (g[c].values / g.D.values + shift for c in ("mid", "bid", "ask"))
        w1 = (K[2:] - K[1:-1]) / (K[2:] - K[:-2])
        w3 = 1 - w1
        n += len(w1)
        at_mid += np.sum(w1 * mid[:-2] + w3 * mid[2:] - mid[1:-1] < -1e-8)
        executable += np.sum(w1 * ask[:-2] + w3 * ask[2:] - bid[1:-1] < -1e-8)
    return 100 * at_mid / max(n, 1), 100 * executable / max(n, 1)


def calendar_stats(q: pd.DataFrame) -> float:
    """Share of points where total variance (iv^2 * T) falls from one expiry to the next at the same
    log-moneyness. Total variance must rise with maturity, otherwise there is calendar arbitrage."""
    n = bad = 0
    for _, day in q.groupby("quote_date", sort=False):
        slices = [g.sort_values("k") for _, g in day.groupby("expire_date")]
        slices.sort(key=lambda g: g["T"].iloc[0])
        for a, b in zip(slices[:-1], slices[1:]):
            lo, hi = max(a.k.min(), b.k.min()), min(a.k.max(), b.k.max())
            m = (a.k >= lo) & (a.k <= hi)
            if m.sum() < 3:
                continue
            w_a = (a.iv_mid ** 2 * a["T"]).values[m.values]
            w_b = np.interp(a.k.values[m.values], b.k.values, (b.iv_mid ** 2 * b["T"]).values)
            n += len(w_a)
            bad += np.sum(w_b < w_a - 1e-10)
    return 100 * bad / max(n, 1)


def vet_month(path: str):
    month = pd.Period(os.path.basename(path)[8:14], freq="M")
    raw = load_chain(path)
    days = pd.DatetimeIndex(sorted(raw.quote_date.unique()))
    missing = expected_days(month).difference(days)
    extra = days.difference(expected_days(month))

    n_slices = raw.groupby(["quote_date", "expire_date"]).ngroups
    f = add_forwards(raw)
    s = f.drop_duplicates(["quote_date", "expire_date"])
    s = s[s["T"] > 30 / 365]
    q = clean_quotes(build_quotes(f), verbose=False)
    per_day = q.groupby("quote_date").size()
    bf_mid, bf_exec = butterfly_stats(q)

    row = {
        "month": str(month),
        "raw_rows": len(raw),
        "days_found": len(days),
        "days_expected": len(expected_days(month)),
        "days_missing": len(missing),
        "quote_times": ",".join(f"{t:g}" for t in sorted(raw.quote_time_hours.dropna().unique())),
        "duplicate_rows": int(raw.duplicated(["quote_date", "expire_date", "strike"]).sum()),
        "crossed_or_locked_pct": 100 * float((((raw.c_bid > 0) & (raw.c_ask <= raw.c_bid)) |
                                              ((raw.p_bid > 0) & (raw.p_ask <= raw.p_bid))).mean()),
        "slices_dropped_pct": 100 * (1 - f.groupby(["quote_date", "expire_date"]).ngroups / n_slices),
        "implied_rate_median_pct": 100 * float((-np.log(s.D) / s["T"]).median()),
        "implied_rate_iqr_pct": 100 * float((-np.log(s.D) / s["T"]).quantile(.75) - (-np.log(s.D) / s["T"]).quantile(.25)),
        "own_vs_vendor_iv_median_volpts": 100 * float((q.iv_mid - q.vendor_iv).median()),
        "bidask_median_volpts": 100 * float((q.iv_ask - q.iv_bid).median()),
        "clean_quotes": len(q),
        "clean_per_day_min": int(per_day.min()),
        "clean_per_day_median": int(per_day.median()),
        "zero_volume_pct": 100 * float((q.volume.fillna(0) == 0).mean()),
        "butterfly_at_mid_pct": bf_mid,
        "butterfly_executable_pct": bf_exec,
        "calendar_at_mid_pct": calendar_stats(q),
    }
    return row, missing, extra, q


def main(folders):
    files = sorted(f for d in folders for f in glob.glob(os.path.join(d, "spx_eod_*.txt")))
    if not files:
        raise SystemExit(
            "No spx_eod_*.txt files found. Expected them at data/spx_2023q1/spx_eod_202301.txt etc.\n"
            f"Looked in: {list(folders) or 'data/spx_2023q* (no such folders)'}\n"
            "Folder names are case-sensitive: the folder must be 'data', not 'Data'.\n"
            "Or pass the folders explicitly: python vet_data.py path/to/q1 path/to/q2 ...")
    rows, all_missing, all_q = [], [], []
    for f in files:
        row, missing, extra, q = vet_month(f)
        rows.append(row); all_q.append(q)
        all_missing += [(d.date(), "missing") for d in missing] + [(d.date(), "unexpected") for d in extra]
        print(f"{row['month']}: {row['days_found']}/{row['days_expected']} days, {row['clean_quotes']} clean quotes")
    table = pd.DataFrame(rows).set_index("month")
    table.to_csv("vetting_by_month.csv")
    pd.DataFrame(all_missing, columns=["date", "issue"]).to_csv("missing_days.csv", index=False)
    pd.concat(all_q, ignore_index=True).to_pickle("quotes_clean.pkl")
    pd.set_option("display.width", 250)
    print("\n" + table.T.round(3).to_string())
    print("\nmissing / unexpected days:", all_missing or "none")


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(glob.glob("data/spx_2023q*")))
