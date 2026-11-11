# Neural implied-volatility surface on SPX options

A learning project: fit and forecast the SPX implied-vol surface with neural networks,
using real end-of-day option quotes. Classical models (per-expiry polynomial now, Heston/SVI later)
are benchmarks, not the subject.

**Status:** data pipeline and vetting done for 2023. Stage 1 (plain MLP fit of one day) runs but
underfits and loses to the polynomial benchmark. See [LOG.md](LOG.md) for the running log.

## Plan
1. Fit one day's surface with a small MLP - *in progress, underfitting*
2. Add no-arbitrage penalties (calendar, butterfly) to the loss
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
```

## Layout
| Path | What it is |
|---|---|
| `volnn/data.py` | Loader, parity-implied forwards, Black-76 implied vols, cleaning filters |
| `volnn/model.py` | Stage 1 MLP and training loop |
| `vet_data.py` | Month-by-month data vetting, including static-arbitrage counts |
| `stage1_fit_one_day.py` | Stage 1 experiment with two out-of-sample tests and a benchmark |
| `results/` | Figures and tables produced by the scripts |
| `LOG.md` | Dated log of decisions, results and open issues |
