# IoT Device Reading Forecaster

A production-grade time-series forecasting pipeline for industrial IoT telemetry data. Predicts device readings at 5-second resolution for a 3-hour horizon (2,160 steps) using classical ML and deep learning approaches.

---

## Problem Statement

Given 30 days of device sensor readings sampled every 5 seconds (~500,000+ observations), forecast the next 3 hours of readings at the same 5-second frequency. The data contains structural outages, spikes, strong short-term autocorrelation, and weak long-range seasonality.

---

## Repository Structure

```
IOT-forecasting/
│
├── input_data/
│   └── problem_data.xlsx                     # Raw input telemetry dataset
│
├── models/
│   ├── lgb_production.pkl                    # Saved LightGBM production model
│   └── lgb_production_metadata.json          # Model metadata & feature registry
│
├── notebooks/
│   ├── 01_data_analysis_and_preprocessing.ipynb        # Data cleaning, preprocessing, EDA
│   ├── 02_time_series_analysis_and_baselines.ipynb     # Statistical TS analysis, ARIMA & baselines
│   ├── 03_ml_forecasting_pipeline.ipynb                # ML forecasting pipeline (XGBoost, LightGBM, RF)
│   └── 04_final_forecasting_and_outputs.ipynb          # Final production forecast & outputs
│
├── output_data/
│   ├── feature_importance.png                # Feature importance visualisation
│   ├── forecast_from_arima.csv               # ARIMA forecast output
│   ├── forecast_output.csv                   # Final production forecast output
│   └── forecast_plot.png                     # Forecast visualisation plot
│
├── .gitignore                                # Git ignore rules
├── environment.yml                           # Conda environment specification
├── README.md                                 # Project documentation
└── requirements.txt                          # Python dependencies
```

---

## Pipeline Overview

### Notebook 1 — Data Analysis & Preprocessing (`01_data_analysis_and_preprocessing.ipynb`)

Foundational analysis and preprocessing of the raw telemetry signal:

- **Data loading and cleaning** — null removal, deduplication, sorting
- **Resampling** — aggregating to a strict 5-second grid; zero-filling structural outages
- **Missing value analysis** — gap detection and imputation strategy

---

### Notebook 2 — Time Series Analysis & Baselines (`02_time_series_analysis_and_baselines.ipynb`)

Deeper signal characterisation:

- **Stationarity testing** — ADF and KPSS tests with differencing
- **Decomposition** — trend, seasonality, and residual separation
- **Outlier analysis** — IQR and Z-score based spike detection, - Rolling statistics and regime change detection
- **ACF / PACF analysis** — identifying significant lag structure
- **Baseline forecasting** — Simple Exponential Smoothing, ARIMA
- **SARIMA evaluation** — tested and ruled out due to computational infeasibility at large seasonal periods
- **Split** - Train / test split design (last 7 days held out)

Key finding: strong autocorrelation at short lags (1–100 steps), weak long-range seasonality, predictable zero-run outage patterns.

Key conclusion: SARIMA becomes computationally infeasible at large seasonal periods (daily seasonality = 17,280 steps at 5-second resolution).

---

### Notebook 3 — ML Forecasting Pipeline (`03_ml_forecasting_pipeline.ipynb`)

**Stage 1 — Feature Engineering**

| Feature Family | Examples |
|---|---|
| Lag features | `lag_1`, `lag_12`, `lag_720`, `lag_2160` |
| Rolling statistics | `roll_mean_60`, `roll_std_360`, `roll_range_720` |
| Exponentially weighted | `ewm_mean_12`, `ewm_std_60` |
| Rate of change | `diff_1`, `diff_12`, `diff_60` |
| Cyclical time | `hour_sin/cos`, `minute_sin/cos`, `dow_sin/cos` |
| Outage indicators | `is_zero`, `zero_run_length`, `steps_since_nonzero` |
| Local trend | `local_trend` (normalised mean difference) |

All features are strictly backward-looking — no target leakage.

**Stage 2 — Forecasting Strategy**

Three strategies compared: Recursive, Direct, and MIMO. Recursive forecasting chosen as primary strategy due to strong short-lag autocorrelation and the practicality of maintaining a single model for a 2,160-step horizon.

**Stage 3 — Models**

- **XGBoost** — histogram-based gradient boosting, L1/L2 regularisation, walk-forward validation
- **LightGBM** — leaf-wise tree growth, faster training on large datasets, native quantile support
- **Random Forest** — bagging baseline, free uncertainty proxy via tree variance

A `RecursiveForecaster` class handles inference: maintains a stateful rolling buffer and updates all lag, rolling, and EWM features incrementally at each step — no train/inference skew.

**Stage 4 — Evaluation**

Metrics used:
- MAE
- RMSE
- SMAPE
- MASE
- MAPE

Horizon-sliced SMAPE curves show how error accumulates across the 3-hour window. Model leaderboard and horizon-wise SMAPE degradation curves compare all approaches. 

**Stage 5 — Advanced**

- **Quantile forecasting** — LightGBM trained at q10/q50/q90 for 80% prediction intervals
- **Outage-aware regime masking** — Logistic classifier detects predicted outage windows and overrides forecast with zero, preventing hallucinated readings during structural downtime

---

### Notebook 4 — Final Forecasting & Outputs (`04_final_forecasting_and_outputs.ipynb`)

Final production notebook. Selects LightGBM as the best-performing model and retrains on the full 30-day dataset.

**Steps:**
1. Reload and validate full dataset
2. Rebuild feature matrix on all available data
3. Retrain LightGBM with validated hyperparameters (no test split)
4. Save model to `lgb_production.pkl` + companion `lgb_production_metadata.json`
5. Verify round-trip: reload model from disk, run smoke test
6. Generate 2,160-step (3-hour) recursive forecast from last observed timestamp
7. Export forecast to `forecast_output.csv`
8. Produce visualisations: forecast overlay, hourly breakdown, feature importances

---

### Notebook 5 — Deep Learning Pipeline (`dl_forecasting_pipeline.ipynb`)
 
A separate DL track (extra). Justified by dataset size (>500k observations) and strong sequential structure. All models use MIMO (multi-output) strategy to avoid recursive error accumulation.
 
**Data preparation**
- Z-score normalisation fit on training data only
- Sliding window dataset (`SlidingWindowDataset`) with configurable lookback, horizon, and stride
- Shared training loop with AdamW, cosine LR annealing, Huber loss, and early stopping

**Models**

| Model | Key Design | Suitability |
|---|---|---|
| **Temporal CNN (TCN)** | Causal + dilated convolutions, residual blocks, exponential receptive field growth | Best speed, no hidden state, fully parallelisable |
| **GRU Encoder-Decoder** | Seq2Seq with teacher forcing annealing | Good outage recovery modelling via gating |
| **LSTM + Bahdanau Attention** | Attention over encoder states at each decoder step | Better long-range recall, interpretable attention maps |
| **PatchTransformer** | PatchTST-style: input split into patches, reducing attention cost by patch_size² | Efficient Transformer for long sequences |
| **N-BEATS** | Doubly residual FC stack with backcast/forecast decomposition | Interpretable, no convolution or recurrence needed |

All architectures compared on a shared DL leaderboard with horizon-sliced SMAPE decay curves.

---

## Quickstart

### Prerequisites

```bash
pip install pandas numpy matplotlib lightgbm xgboost scikit-learn torch openpyxl
```

### Run order

Run notebooks in sequence:

```
1. 01_data_analysis_and_preprocessing.ipynb
2. 02_time_series_analysis_and_baselines.ipynb
3. 03_ml_forecasting_pipeline.ipynb
4. 04_final_forecasting_and_outputs.ipynb
```

Notebook 4 is fully self-contained and can be run independently after placing `problem_data.xlsx` in the working directory.

### Input

| File | Description |
|---|---|
| `problem_data.xlsx` | Raw device readings with `timestamp` and `Reading` columns |

### Outputs

| File | Description |
|---|---|
| `lgb_production.pkl` | Serialised production LightGBM model |
| `lgb_production_metadata.json` | Feature registry, hyperparameters, training stats |
| `forecast_output.csv` | 2,160-row forecast: timestamp, reading, step, minutes ahead |
| `forecast_plot.png` | 6-hour history + 3-hour forecast overlay |
| `feature_importance.png` | Top-20 LightGBM feature importances |
| `forecast_hourly_breakdown.png` | Mean forecast by hour block |

---

## Model Selection Rationale

| Criterion | Winner | Notes |
|---|---|---|
| Forecast accuracy (SMAPE) | LightGBM | Lowest error on 7-day hold-out |
| Training speed | LightGBM | Leaf-wise growth fastest on 500k+ rows |
| Inference speed | LightGBM | ~2s for 2,160-step recursive forecast on CPU |
| Uncertainty quantification | LightGBM (quantile) | Native q10/q50/q90 support |
| Interpretability | LightGBM | SHAP-compatible feature importances |
| GPU dependency | None | Runs on CPU in production |

---

## Key Design Decisions

**Why not SARIMA?** Seasonal periods at 5-second resolution (e.g., daily = 17,280 steps) make SARIMA computationally infeasible. Tested and ruled out in Notebook 1.

**Why recursive over direct forecasting?** The signal has very strong short-lag autocorrelation. A single recursive model exploits this directly and avoids training 2,160 independent models.

**Why include a horizon-aligned lag (`lag_2160`)?** The reading from exactly 3 hours ago acts as an anchor feature, partially compensating for recursive error accumulation at long horizons.

**Why Huber loss for DL models?** The signal contains outage spikes and structural zeros. Huber loss (δ=1.0) is robust to these outliers without fully ignoring them like MAE.

**Why outage masking?** Zero-run outages are structurally predictable. A logistic regime classifier detects predicted outage windows and overrides the forecast with zero, preventing the model from hallucinating positive readings during known downtime.

---

## Computational Notes

| Component | Approximate time (CPU) |
|---|---|
| Feature engineering (full dataset) | 2–5 min |
| LightGBM training (full dataset) | 3–8 min |
| Recursive 3-hour forecast | ~2 sec |
| TCN training (50 epochs) | 10–20 min |
| GRU / LSTM training (50 epochs) | 20–40 min |

GPU is optional. All ML notebooks run on CPU. DL notebooks benefit from a GPU but are not required.

---

## License

MIT
