# Project log

Newest entries at the bottom. Each entry records what was decided or done, the result, and what is
still open. Numbers are copied from script output; the script that produced them is named.

---

## 2026-11-11 - Scoping

**Decisions**
- Goal is to learn neural networks on real market data, not simulated data.
- Dropped the original idea (Heston calibration / identifiability paper) as the centre of the project.
  Reasons: a neural Heston calibrator has to be trained on model-generated prices, and an ML correction
  of Heston residuals is already published and teaches little about neural networks.
- New centre: a neural implied-vol surface learned directly from SPX quotes, in five stages
  (see README). Heston/SVI stay as benchmarks.
- Framing: a learning project with an honest write-up. Arbitrage-aware neural vol surfaces are not novel.

**Data source**
- No WRDS/OptionMetrics access. Considered Kaggle option dumps (mostly SPY, unclear provenance) and
  Carbon Arc (alternative data, no option chains found).
- Chose OptionsDX SPX end-of-day chains: European-style index options, bid/ask included.
- Open: OptionsDX licence terms not yet read. Raw data is kept out of this repo until they are.

## 2026-11-11 - Data pipeline (`volnn/data.py`)

**What is computed, per (date, expiry)**
1. Forward F and discount factor D from put-call parity: regress mid(C) - mid(P) on strike over
   strikes within 10% of spot; slope = -D, intercept = D * F. No external rate or dividend input.
2. One out-of-the-money quote per strike (puts below F, calls above).
3. Own Black-76 implied vols at bid, mid and ask (vectorised bisection).
4. Log-moneyness k = ln(K / F) and maturity T = DTE / 365.

**Filters, in order**
1. 7 <= DTE <= 365
2. bid >= 0.10
3. (ask - bid) / mid <= 0.50
4. implied vol solvable at bid, mid and ask
5. |k| / (ATM vol * sqrt(T)) <= 4

Q1 2023: 448,287 OTM quotes -> 312,426 after filters.

## 2026-11-11 - Stage 1: MLP on one day (`stage1_fit_one_day.py`)

**Setup:** 2023-02-15, 4,768 clean quotes, 38 expiries. MLP 3 x 64, SiLU, softplus output, inputs
[k / sqrt(T), sqrt(T)], full-batch Adam, 4,000 epochs, unweighted MSE on mid IV.
Held out: a random 20% of quotes, and separately the whole 2023-03-24 expiry.
Benchmark: quartic polynomial in k fitted per expiry.

| | RMSE (vol pts) | Inside bid-ask |
|---|---|---|
| NN, train | 0.33 | 15% |
| NN, held-out quotes | 0.34 | 14% |
| NN, held-out expiry | 1.03 | 1% |
| Polynomial, train | 0.13 | 52% |
| Polynomial, held-out quotes | 0.29 | 48% |

Median bid-ask width that day: 0.12 vol points. Figure: `results/iteration-1-baseline/stage1_smiles.png`.

**Reading**
- The network underfits: train and held-out errors match but are ~3x the bid-ask width.
  One larger run (4 x 128, 8,000 epochs) reached 0.23 RMSE, 37% inside the spread. Still short.
- The held-out expiry fails because the ATM term structure is not smooth: 15.9% at the 20 Mar expiry,
  17.1% at 24 Mar. Likely the 22 Mar FOMC meeting; not verified beyond the date coincidence.
- The network currently loses to the benchmark. That is the honest state of stage 1.

**Next**
1. Weight the loss by inverse bid-ask width.
2. Fit total variance (iv^2 * T) instead of IV.
3. Handle event dates (per-expiry level offset or event feature).
4. Evaluate across many days, not one.

## 2026-11-11 - Full-year vetting (`vet_data.py`)

Loaded Q2-Q4 2023. Totals: 249 trading days, 9,869 expiry slices, 1,234,564 clean quotes.
Full table: `results/data-vetting/vetting_by_month.csv`.

| Check | Result |
|---|---|
| Trading days | 249 of 250; 2023-01-03 missing |
| Quote time | 16:00 on every row |
| Duplicate rows / crossed quotes | 0 / 0 |
| Slices dropped at the forward step | 6.5%-8.2% per month |
| Implied rate (median, T > 30d) | 4.8% in Jan rising to ~5.6% by Aug |
| Own IV minus vendor IV (median) | 0.13-0.19 vol points |
| Bid-ask width (median) | 0.12-0.16 vol points |
| Clean quotes per day | min 4,293 |
| Calendar violations at mid | 0.02%-0.17% |
| Butterfly violations, executable (wings at ask, body at bid) | 0.04%-0.8% |
| Butterfly violations at mid | 19%-29% |

**Reading**
- Data is usable. Butterfly violations at mid are tick-rounding noise (few are executable), but they
  mean raw mids are not an arbitrage-free surface; relevant for stage 2.
- Implied rate sits slightly above Fed funds (5.25%-5.5% in H2). Possibly day-count: T uses calendar
  days / 365 and treats every expiry as 16:00. Not resolved.
- Share of zero-volume quotes rises in H2 (10% -> 15-22%). Kept for now.

**Open**
- Stale closing quotes on the wings: not tested.
- AM- vs PM-settled expiries are not distinguished.
- OptionsDX licence terms: not read.

## 2026-11-11 - First local run

- Repo pushed to GitHub (private). Raw data stays local under `data/`, which Git ignores.
- `vet_data.py` crashed with `KeyError: 'month'` when it found no data files. It now stops with a
  message saying where it looked. Cause on the local machine: folder named `Data`, script expects `data`.
- Local run of `vet_data.py` on a MacBook reproduced every number in `results/data-vetting/vetting_by_month.csv`.
  `stage1_fit_one_day.py` not yet run locally.

## Where to pick up

1. Run `python stage1_fit_one_day.py --date 2023-02-15` locally; try a few other dates.
2. Fix the stage 1 underfit, one change at a time, recording the metrics after each:
   a. weight the loss by inverse bid-ask width
   b. fit total variance (iv^2 * T) instead of IV
   c. handle event dates (per-expiry level offset or event feature)
3. Evaluate across many days (e.g. 20 spread over 2023), not one.
4. Target before stage 2: beat the per-expiry polynomial on held-out quotes and get most
   predictions inside the bid-ask band.
5. Housekeeping: read the OptionsDX licence terms; decide whether to buy/download 2021-2022.

## 2026-11-11 - Stage 1 underfit: ablation (`stage1_ablation.py`)

**Changes tried**
- Spread weights: loss weighted by 1 / (bid-ask width)^2, floored at 0.05 vol points.
- Total-variance output: network outputs w = sigma^2 * T; loss still measured in IV.
- Time features (`volnn/features.py`): trading-day time / calendar time, FOMC decisions per year
  before expiry, CPI releases per year before expiry. Event dates were typed in by hand and have
  not been checked against the Fed and BLS calendars.
- Bigger network trained longer: 4 x 128, 8,000 epochs (small = 3 x 64, 4,000 epochs).

**Results** - mean over the three days all variants were run on (2023-02-15, 06-21, 10-18).
RMSE in vol points; inside = % of predictions within bid-ask. Variants A-E and the polynomial were
also run on three more days (04-19, 08-16, 12-06); six-day means are in
`results/iteration-2-ablation/stage1_ablation_summary.csv` and tell the same story.

| Variant | Held-out quotes RMSE | inside | Held-out expiry RMSE | inside |
|---|---|---|---|---|
| Polynomial per expiry | 0.240 | 38% | n/a | n/a |
| A baseline (small) | 0.249 | 26% | 0.418 | 23% |
| B small + spread weights | 0.351 | 22% | 0.436 | 21% |
| C small + weights + total variance | 0.217 | 33% | 0.244 | 31% |
| D small + weights + time features | 0.522 | 19% | 0.293 | 23% |
| F big network only | 0.201 | 41% | 0.226 | 34% |
| G big + spread weights | 0.197 | 42% | 0.191 | 27% |
| H big + weights + total variance | 0.194 | 47% | 0.230 | 8% |
| E big + weights + time features | 0.076 | 79% | 0.278 | 12% |
| I big + weights + total variance + time features | 0.056 | 91% | 0.357 | 11% |

**Reading**
- The underfit on held-out quotes is fixed, and it took two things together: time features and a
  bigger network. Either alone does little (F: 0.20; D: worse than baseline).
- Spread weighting hurt the small network and was roughly neutral for the big one. Total variance
  helped the small network and was roughly neutral for the big one.
- The held-out expiry did not improve and got worse with time features. With one day of data there are
  only ~38 expiries, and the time features nearly identify each one, so the network can learn each
  expiry's level separately without learning a rule that carries to an unseen expiry. This is
  overfitting along the maturity axis, invisible in the random-quote holdout.
- Held-out-expiry numbers are noisy: one expiry per day, and the result moved between runs
  (2023-02-15: 0.53 and 0.64 for variant I under different thread settings).
- F-I were run on three days only. Single seed throughout. No hyperparameter search.

**Decision**
- `stage1_fit_one_day.py` now defaults to variant I. `--baseline` reproduces the original model.
- Stage 1 is good enough on quotes. The unseen-expiry problem is not a stage 1 tuning problem: it
  needs constraints across maturity (stage 2) and training on many days so event effects are learned
  from hundreds of expiries instead of 38 (stage 3).

**Third-party code:** none used. Candidates noted in README (QuantLib, an SVI/SSVI package).

## Where to pick up (replaces the list above)

1. `git pull`, then run `python stage1_fit_one_day.py --date 2023-02-15` locally.
2. Check the FOMC and CPI dates in `volnn/features.py` against the Fed and BLS calendars.
3. Stage 2: calendar and butterfly penalties via autograd. Test: does the calendar penalty improve the
   held-out expiry?
4. Make the held-out-expiry test less noisy: hold out several expiries per day, several seeds.
5. Housekeeping: OptionsDX licence terms; 2021-2022 data.

## 2026-11-11 - Repo housekeeping

- Results are now kept per iteration under `results/` and never overwritten; the iteration 1 figure
  and metrics were restored from history next to the iteration 2 ones.
- `CHANGELOG.md` added: one entry per push. Snapshot branches `iteration-1` and `iteration-2` mark each state (tags could not be pushed from the Claude session).
- Convention from here: each push gets a changelog entry, a snapshot branch, and its own results folder.

## 2026-11-11 - Stage 2: no-arbitrage penalties (`volnn/surface.py`, `stage2_arbitrage.py`)

**Math**
- Total variance w(k, T) = sigma^2 * T.
- Calendar: w must not fall as T rises at fixed k. Penalty = mean relative fall in w between
  consecutive trading days, at random k. Enforced on every trading day 7-365 days ahead, listed or
  not, because the model's time inputs depend only on dates.
- Butterfly: g(k) = (1 - k w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2 >= 0. Penalty = mean max(0, -g).
  w' and w'' come from autograd.
- Loss = mean(((sigma_hat - sigma_mid) / spread)^2) + 100 * calendar + 10 * butterfly.

**Models**
- Free: the stage 1 variant I network, written to output total variance directly.
- Structured: w = smooth network in trading time + v_fomc * (FOMC count) + v_cpi * (CPI count).
  Two learned event variances instead of free-form time features.

**Test:** per day, hold out four whole expiries (20/40/60/80% along the curve) and a random 20% of
the remaining quotes. Days: 2023-02-15, 06-21, 10-18. 4 x 128 network, 5,000 epochs.

**Results:** table in CHANGELOG.md, iteration 3; files in `results/iteration-3-arbitrage/`.

**Reading**
- Calendar penalty: violations 2.9% -> 0.6% of the grid, worst single fall 7.4% -> 2.4%, fit unchanged.
  Soft penalty with an untuned weight, so not zero.
- Butterfly: no violations with or without the penalty (minimum g about 0.1). The raw mid quotes had
  19-29% butterfly violations from tick rounding; the smooth network removes them by itself. The
  penalty costs 2x training time and currently buys nothing inside the quoted strike range.
- Unseen expiries: no variant helps. Hypothesis that the calendar penalty would fix them: rejected.
  Hypothesis that structured event variance would generalise better: rejected; it fits quotes worse
  (0.11 vs 0.06) because one flat variance number per event type is too crude. Learned event sizes were
  about a 0.30% move for FOMC and 0.24% for CPI.
- Correction: the iteration 2 conclusion that unseen expiries "fail" rested on one held-out expiry per
  day and was too strong. See CHANGELOG.
- A figure of day-by-day ATM variance growth was tried and dropped: violations are mostly away from
  the money and the picture did not show the effect.

## Where to pick up (replaces the lists above)

1. `git pull`; run `python stage2_arbitrage.py` locally if you want to reproduce (about 15 minutes).
2. Check the FOMC and CPI dates in `volnn/features.py` against the Fed and BLS calendars.
3. Stage 3: one model across all 249 days with a latent state per day. This is the real test of
   learning event effects, with thousands of expiries instead of ~38.
4. Optional before stage 3: tune the calendar weight; test expiries just after events specifically;
   check butterfly in the far wings.
5. Housekeeping: OptionsDX licence terms; 2021-2022 data.
