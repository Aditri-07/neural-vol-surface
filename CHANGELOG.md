# Changelog

One entry per push to GitHub: what was added, what changed, and what was found. Nothing is deleted
between iterations. Every earlier state is kept as a snapshot branch (`git checkout iteration-1`), and
each iteration's outputs are kept side by side under `results/`.

For the reasoning behind each decision, see [LOG.md](LOG.md).

---

## Iteration 3 - No-arbitrage penalties  (branch `iteration-3`, 2026-11-11)

**What we worked on:** stage 2. Adding calendar and butterfly conditions to the loss, and testing
whether they, or a more structured treatment of events, help on expiries the network has not seen.

**Added**
- `volnn/surface.py`: total-variance surface models, calendar and butterfly penalties computed with
  autograd, and a report of how much arbitrage a fitted surface contains.
- `stage2_arbitrage.py`: five variants on three days, with four held-out expiries per day.
- `results/iteration-3-arbitrage/`.

**Changed**
- The unseen-expiry test now holds out four expiries per day instead of one. Stage 1 code is untouched.

**Found** (mean of three days, RMSE in vol points, share inside bid-ask in brackets)

| Variant | Held-out quotes | Held-out expiries | Calendar violations | Butterfly violations |
|---|---|---|---|---|
| Free network, no penalty | 0.061 (90%) | 0.097 (52%) | 2.9% | 0% |
| Free + calendar penalty | 0.066 (87%) | 0.085 (50%) | 0.6% | 0% |
| Free + calendar + butterfly | 0.064 (90%) | 0.096 (51%) | 0.9% | 0% |
| Structured events, no penalty | 0.113 (66%) | 0.120 (49%) | 0.07% | 0% |
| Structured + calendar + butterfly | 0.111 (66%) | 0.124 (47%) | 0.02% | 0% |

- The calendar penalty removes most calendar arbitrage at almost no cost to the fit.
- The fitted surfaces had no butterfly arbitrage even without a penalty.
- Neither the penalties nor the structured event model improved unseen expiries.
- **Correction to iteration 2:** "fails on unseen expiries" was overstated. That came from holding out
  one expiry per day, which on 2023-02-15 was the first expiry after a Fed meeting. With four expiries
  held out, the error is 0.10 vol points, not 0.36. Weaker than on quotes, but not a failure.

**Not done / caveats**
- Penalty weights (100 and 10) were not tuned; calendar violations are reduced, not eliminated.
- Arbitrage is checked on a grid inside the quoted strike range only, not in the far wings.
- Three days, one seed. Expiries just after an event are probably still the weak spot; not tested.

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
