# Project 09 — Weather Data Analysis & Prediction

Explore a city's historical weather and forecast the **next-day mean
temperature** with a model trained on real observations. The project is a
regression / time-series exercise whose whole point is an honest comparison
against a persistence baseline.

## What it does

1. Downloads daily observations for a city from the **Open-Meteo historical
   archive** (ERA5 reanalysis) — free, no API key, no registration.
2. Engineers **leakage-free** features from information known at the end of each
   day (lagged temperatures, trailing rolling statistics and calendar terms).
3. Trains three regressors and compares them to a **persistence baseline**
   (`tomorrow = today`) on the same chronological split.
4. Saves the winner and reports its held-out metrics in degrees Celsius.

## Data source

| Field | Value |
| --- | --- |
| Dataset | Open-Meteo historical archive (ERA5 reanalysis) |
| Endpoint | `https://archive-api.open-meteo.com/v1/archive` |
| Variables | daily max/min/mean 2 m temperature, precipitation, max wind speed |
| Default window | 2000-01-01 to 2023-12-31 |
| Licence | Open-Meteo (CC-BY-4.0 attribution) |

The response is cached as `data/weather_<city>.csv`, so training and the app work
offline after the first download. Cities offered: London, New York, Tokyo,
Sydney, Mumbai and Cairo.

## Method

- **Target.** The mean temperature `horizon` days ahead, encoded by default as a
  *change* from today (`delta`) so tree ensembles, which cannot extrapolate, are
  not handicapped by an upward-trending series.
- **Features.** `lag_1/2/3/7`, rolling means over 7/14/30 days, rolling
  std/min/max over 7 days, the day's temperature range, precipitation, wind speed
  and cyclical calendar terms (`day_of_year_sin/cos`, `month`, `day_of_week`).
- **Models.** Linear Regression, Random Forest and Gradient Boosting, all fitted
  on the same 80/10/10 chronological split; the winner is chosen by test RMSE.
- **Baseline.** The persistence forecast, reported next to every model.

Because temperature is strongly autocorrelated, the baseline is genuinely hard
to beat — and the interface says so whenever the model does not beat it.

## Running

```bash
cd 09-weather-analysis-prediction
python train.py                      # trains the default city (London)
python train.py --city "Tokyo, Japan" --horizon 3
streamlit run app.py                 # launches the interface
```

Useful flags: `--city`, `--start`, `--end`, `--horizon`, `--target-mode`,
`--force-download`.

## Outputs

- `models/weather_model.joblib` — the selected model plus its metadata.
- `models/training_metrics.json` — held-out metrics, the full model comparison,
  feature importances and dataset details.

## Limitations

- This is an educational next-day temperature forecast, **not** a meteorological
  product, and must not be used for safety-critical decisions.
- It predicts a single city's mean temperature; it does not model fronts,
  storms or spatial structure.
- Reported accuracy is on a held-out *final block* of the series, which is the
  correct protocol for time series but means the number is specific to the
  chosen city and date range.

## Files

| File | Purpose |
| --- | --- |
| `src/data.py` | Open-Meteo download, atomic CSV caching and loading. |
| `src/features.py` | Leakage-free feature engineering. |
| `src/model.py` | Estimators, persistence baseline, training and persistence. |
| `train.py` | Training / evaluation entry point. |
| `app.py` | Streamlit interface. |
| `tests/test_weather.py` | Offline tests on synthetic data. |
