# Changelog

One entry per push to GitHub: what was added, what changed, and what was found. Nothing is deleted
between iterations. Every earlier state is kept as a snapshot branch (`git checkout iteration-1`), and
each iteration's outputs are kept side by side under `results/`.

For the reasoning behind each decision, see [LOG.md](LOG.md).

---

## Iteration 2 - Fixing the stage 1 underfit  (branch `iteration-2`, 2026-11-11)

**What we worked on:** why the first network lost to a polynomial, and what fixes it.

**Added**
- `volnn/features.py`: trading-day time, and FOMC / CPI events before expiry.
- `stage1_ablation.py`: runs nine model variants across several days.
- `volnn/model.py`: spread-weighted loss, total-variance output, extra input features.
- `results/iteration-2-ablation/`: ablation tables and the new fit figure.
- This changelog, and a `results/` folder per iteration.

**Changed**
- `stage1_fit_one_day.py` now defaults to the best variant (4 x 128 network, spread weights,
  total-variance output, time features). `--baseline` still runs the iteration 1 model unchanged.
- Scripts write outputs to the folder they are run from, so committed results are never overwritten.

**Found** (mean of three days, RMSE in vol points)

| | Iteration 1 baseline | Iteration 2 best | Polynomial benchmark |
|---|---|---|---|
| Held-out quotes RMSE | 0.249 | 0.056 | 0.240 |
| Held-out quotes inside bid-ask | 26% | 91% | 38% |
| Held-out expiry RMSE | 0.418 | 0.357 | n/a |
| Held-out expiry inside bid-ask | 23% | 11% | n/a |

- Time features plus a bigger network fixed the fit on held-out quotes. Neither helped alone.
- An expiry the network has never seen is still fitted badly: overfitting along the maturity axis.

**Not done / caveats**
- Four variants ran on three days only; single random seed; event dates not checked against sources.

## Iteration 1c - Log update  (commit `77f2e26`, 2026-11-11)

- Recorded that `vet_data.py` reproduced exactly on a local MacBook.
- Added a "where to pick up" list to `LOG.md`.

## Iteration 1b - Local-run fix  (commit `d71b8ff`, 2026-11-11)

- `vet_data.py` crashed with `KeyError: 'month'` when it found no data files. It now stops with a
  message saying where it looked.

## Iteration 1 - Data pipeline and first network  (branch `iteration-1`, 2026-11-11)

**What we worked on:** choosing the project and data, building and vetting the data pipeline, and a
first neural fit.

**Added**
- `volnn/data.py`: loader for OptionsDX SPX end-of-day files, forwards and discount factors from
  put-call parity, Black-76 implied vols at bid / mid / ask, five cleaning filters.
- `vet_data.py`: month-by-month vetting, including butterfly and calendar arbitrage counts.
- `volnn/model.py` and `stage1_fit_one_day.py`: 3 x 64 MLP fitted to one day, two holdout tests,
  per-expiry polynomial benchmark.
- `results/data-vetting/` and `results/iteration-1-baseline/`.

**Found**
- 2023 data is clean: 249 of 250 trading days, 1,234,564 quotes after filters.
- The first network underfits: 0.33 vol points RMSE against a 0.12 bid-ask width on 2023-02-15,
  worse than the polynomial (0.29 on held-out quotes), and 1.03 on a held-out expiry.
