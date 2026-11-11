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

Median bid-ask width that day: 0.12 vol points. Figure: `results/stage1_smiles.png`.

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
Full table: `results/vetting_by_month.csv`.

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
