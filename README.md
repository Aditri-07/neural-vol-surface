# Neural implied-volatility surface on SPX options

A learning project: fit and forecast the SPX implied-vol surface with neural networks,
using real end-of-day option quotes. Classical models (per-expiry polynomial now, Heston/SVI later)
are benchmarks, not the subject.

**Status:** data pipeline and vetting done for 2023. Stages 1 and 2 done on single days. The network
beats the per-expiry polynomial on held-out quotes (about 0.06 vol points RMSE, about 90% inside the
bid-ask band). On expiries it has never seen it is weaker (about 0.10 RMSE, about 50% inside). The
fitted surfaces have no butterfly arbitrage; a calendar penalty cuts calendar violations from about
3% of the grid to under 1% without hurting the fit, but does not improve unseen expiries. See [CHANGELOG.md](CHANGELOG.md) for what changed in each push and [LOG.md](LOG.md) for the reasoning.

## Plan
1. Fit one day's surface with an MLP - *done*
2. Add no-arbitrage penalties (calendar, butterfly) to the loss - *done; penalties not tuned*
3. One model across all days via a latent state per day (autoencoder / VAE)
4. Forecast next-day surfaces; test against carry-forward and parametric fits
5. Optional: uncertainty bands (ensembles / MC dropout)

## Data
SPX end-of-day option chains for 2023 from [OptionsDX](https://www.optionsdx.com/product/spx-option-chain/)
(monthly `spx_eod_YYYYMM.txt` files). The raw files are **not** in this repo. To reproduce, download
them and place them under `data/` in any sub-folders, e.g. `data/spx_2023q1/spx_eod_202301.txt`.

## Run
```
pip install -r requirements.txt
python vet_data.py data/spx_2023q1 data/spx_2023q2 data/spx_2023q3 data/spx_2023q4
python stage1_fit_one_day.py --data data/spx_2023q1 --date 2023-02-15
python stage1_ablation.py            # all model variants on six days; slow (about an hour on 2 cores)
python stage2_arbitrage.py           # no-arbitrage penalties on three days (about 15 minutes on 2 cores)
```

## Layout
| Path | What it is |
|---|---|
| `volnn/data.py` | Loader, parity-implied forwards, Black-76 implied vols, cleaning filters |
| `volnn/model.py` | Stage 1 MLP and training loop |
| `volnn/features.py` | Trading-day time and FOMC / CPI event features |
| `vet_data.py` | Month-by-month data vetting, including static-arbitrage counts |
| `stage1_fit_one_day.py` | Stage 1 fit of one day with two out-of-sample tests and a benchmark |
| `stage1_ablation.py` | Grid of model variants across several days |
| `volnn/surface.py` | Stage 2 total-variance models, calendar and butterfly penalties, arbitrage diagnostics |
| `stage2_arbitrage.py` | Stage 2 experiment: penalties and a structured event model, four held-out expiries per day |
| `results/` | Outputs kept per iteration: `data-vetting/`, `iteration-1-baseline/`, `iteration-2-ablation/`, `iteration-3-arbitrage/` |
| `CHANGELOG.md` | What changed in each push |
| `LOG.md` | Dated log of decisions, results and open issues |

## Third-party code
None is used yet. Candidates noted for later benchmarks (not yet evaluated in depth):
[QuantLib](https://www.quantlib.org/) (BSD 3-clause) for Heston pricing, and an SVI/SSVI fitter such as
[ssvi-surface](https://pypi.org/project/ssvi-surface/) (MIT). The stage 2 design follows
Ackerer, Tagasovska and Vatter, "Deep Smoothing of the Implied Volatility Surface" (NeurIPS 2020).
