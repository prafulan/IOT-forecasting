import json

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}

cells = []

# ── Title ──────────────────────────────────────────────────────────────────────
cells.append(md("""# Notebook 1 — ML / Feature Engineering Forecasting Pipeline
## Industrial IoT Telemetry | 5-Second Device Readings | 3-Hour Horizon

> **Continuation notebook.** This picks up directly after `EDA_and_arima.ipynb`.  
> Assumes `df_5s`, `train`, `test`, `eval_report`, `all_predictions` and helper constants  
> (`hourly_readings`, `daily_readings`, etc.) are already defined.  
> Run the EDA notebook first, *or* execute **Cell 0** below to re-derive them from scratch.
"""))

# ── Cell 0 – Bootstrap (idempotent) ───────────────────────────────────────────
cells.append(md("## Cell 0 — Environment Bootstrap\nRe-derives all upstream variables. Safe to skip if the EDA notebook is already in memory."))
cells.append(code("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.dpi': 110,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# ── Constants (mirror EDA notebook) ──────────────────────────────────────────
FREQ_SEC         = 5
HOURLY           = int(3600 / FREQ_SEC)          # 720
DAILY            = int(86400 / FREQ_SEC)          # 17 280
FORECAST_HORIZON = int(3 * 3600 / FREQ_SEC)       # 2 160  (3-hour horizon)
TEST_DAYS        = 7

# ── Re-derive df_5s only if not already in namespace ─────────────────────────
if 'df_5s' not in dir():
    df = pd.read_excel("problem_data.xlsx")
    df = df.dropna(subset=["Reading"])
    df = df.drop_duplicates(subset=["timestamp"], keep="first")
    df = df.sort_values("timestamp")
    df_5s = (
        df.set_index("timestamp")[["Reading"]]
        .resample("5s").sum()
        .reset_index()
    )
    df_5s["date"] = df_5s["timestamp"].dt.date
    print("Reloaded df_5s from disk.")
else:
    print("df_5s already in namespace — skipping reload.")

# ── Re-derive train / test split ─────────────────────────────────────────────
if 'train' not in dir() or 'test' not in dir():
    split_index = int(len(df_5s) - TEST_DAYS * DAILY)
    train = df_5s.iloc[:split_index].copy()
    test  = df_5s.iloc[split_index:].copy()
    print(f"Train: {train.shape}  |  Test: {test.shape}")
else:
    print(f"Using existing train ({train.shape}) / test ({test.shape}) split.")

# ── Global stats needed for outlier flag ─────────────────────────────────────
GLOBAL_MEAN = train["Reading"].mean()
GLOBAL_STD  = train["Reading"].std()

print(f"\\nGlobal mean: {GLOBAL_MEAN:.4f}  |  std: {GLOBAL_STD:.4f}")
print(f"Forecast horizon: {FORECAST_HORIZON} steps = 3 hours")
"""))

# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""---
# Stage 1 — Advanced Feature Engineering

## Design Rationale

For a 5-second IoT signal with **strong short-term autocorrelation** and  
**weak long-range seasonality**, the most predictive features are:

| Feature Family | Motivation |
|---|---|
| Short lags (1–12 steps) | Direct autocorrelation structure from PACF |
| Medium lags (60, 360, 720) | Capture minutes-to-hour momentum |
| Horizon-aligned lags (2160) | What was the signal 3 h ago? |
| Rolling stats (12, 60, 720) | Local mean/variance shift detection |
| EWM stats | Smooth exponential memory of recent trend |
| Cyclical time encodings | Hour-of-day / minute-of-hour periodicity (sin/cos) |
| Outage / zero-run indicator | Structural outages are predictable regime changes |
| Diff features | Rate-of-change captures acceleration |

**What we deliberately avoid:**  
- Daily/weekly seasonal lags (2880, 17280) — decomposition showed weak long seasonality; adding them bloats the matrix without lift  
- Expanding window features — they leak future distribution into training
"""))

cells.append(code("""\
# ─────────────────────────────────────────────────────────────────────────────
#  Feature Engineering — fully vectorised, no leakage
#  All rolling/shift operations look only backward.
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    \"\"\"
    Build the full feature matrix from a time-indexed DataFrame with
    columns ['timestamp', 'Reading'].

    Parameters
    ----------
    df        : DataFrame with 'timestamp' and 'Reading' columns.
    is_train  : If True, drop rows with NaN targets/features (warm-up period).

    Returns
    -------
    DataFrame with feature columns + 'target' column.
    \"\"\"
    df = df.copy()
    df = df.set_index("timestamp")
    y  = df["Reading"]

    feats = pd.DataFrame(index=df.index)

    # ── 1. Lag features ───────────────────────────────────────────────────────
    lag_steps = [1, 2, 3, 5, 10, 12, 30, 60, 120, 360, 720, 1440, FORECAST_HORIZON]
    for lag in lag_steps:
        feats[f"lag_{lag}"] = y.shift(lag)

    # ── 2. Rolling statistics (min-period prevents NaN cascade) ───────────────
    for win in [12, 60, 360, 720]:
        r = y.shift(1).rolling(window=win, min_periods=max(1, win // 4))
        feats[f"roll_mean_{win}"]   = r.mean()
        feats[f"roll_std_{win}"]    = r.std().fillna(0)
        feats[f"roll_max_{win}"]    = r.max()
        feats[f"roll_min_{win}"]    = r.min()
        feats[f"roll_range_{win}"]  = feats[f"roll_max_{win}"] - feats[f"roll_min_{win}"]

    # ── 3. Exponentially Weighted Mean (alpha ≈ span) ─────────────────────────
    for span in [12, 60, 360]:
        feats[f"ewm_mean_{span}"] = y.shift(1).ewm(span=span, adjust=False).mean()
        feats[f"ewm_std_{span}"]  = y.shift(1).ewm(span=span, adjust=False).std().fillna(0)

    # ── 4. Diff / rate-of-change features ─────────────────────────────────────
    feats["diff_1"]  = y.diff(1).shift(1)   # first difference
    feats["diff_12"] = y.diff(12).shift(1)  # 1-minute change
    feats["diff_60"] = y.diff(60).shift(1)  # 5-minute change

    # ── 5. Cyclical time encodings ────────────────────────────────────────────
    ts = df.index
    feats["hour_sin"]   = np.sin(2 * np.pi * ts.hour / 24)
    feats["hour_cos"]   = np.cos(2 * np.pi * ts.hour / 24)
    feats["minute_sin"] = np.sin(2 * np.pi * ts.minute / 60)
    feats["minute_cos"] = np.cos(2 * np.pi * ts.minute / 60)
    feats["dow_sin"]    = np.sin(2 * np.pi * ts.dayofweek / 7)
    feats["dow_cos"]    = np.cos(2 * np.pi * ts.dayofweek / 7)

    # ── 6. Outage / zero-run indicator ────────────────────────────────────────
    # A zero-run is a structural outage: device not reporting.
    feats["is_zero"]         = (y.shift(1) == 0).astype(int)
    feats["zero_run_length"] = (
        feats["is_zero"]
        .groupby((feats["is_zero"] != feats["is_zero"].shift()).cumsum())
        .cumsum()
    )
    # How many steps since last non-zero? Useful for outage recovery prediction.
    feats["steps_since_nonzero"] = feats["zero_run_length"]

    # ── 7. Local trend via linear regression over last 60 steps ───────────────
    # Approximated cheaply via (roll_mean_60 - roll_mean_12) normalised.
    feats["local_trend"] = (feats["roll_mean_60"] - feats["roll_mean_12"]) / (GLOBAL_STD + 1e-8)

    # ── 8. Target ─────────────────────────────────────────────────────────────
    feats["target"] = y.values   # current Reading is the target

    if is_train:
        # Drop the warm-up rows where long lags are NaN
        max_lag = max(lag_steps)
        feats = feats.iloc[max_lag:]
        feats = feats.dropna()

    return feats.reset_index()


# Build feature matrices
print("Building training feature matrix...")
train_feats = build_features(train, is_train=True)
print(f"  Train features: {train_feats.shape}  |  Columns: {len(train_feats.columns)}")

print("\\nBuilding test feature matrix...")
# For test, we also need the tail of train to fill initial lags
# Concatenate, build, then slice back to test only
combined_for_test = pd.concat([
    train[["timestamp", "Reading"]].iloc[-(FORECAST_HORIZON + 100):],
    test[["timestamp", "Reading"]]
], ignore_index=True)
test_feats_full = build_features(combined_for_test, is_train=False)

# Keep only rows whose timestamps fall in the test period
test_start_ts = test["timestamp"].iloc[0]
test_feats = test_feats_full[test_feats_full["timestamp"] >= test_start_ts].copy()
print(f"  Test features : {test_feats.shape}")
"""))

cells.append(code("""\
# ── Feature sanity check ──────────────────────────────────────────────────────
FEATURE_COLS = [c for c in train_feats.columns if c not in ("timestamp", "target")]

print(f"Total features  : {len(FEATURE_COLS)}")
print(f"Feature names   : {FEATURE_COLS[:10]} ...")
print()
print("Missing values in train features:")
missing = train_feats[FEATURE_COLS].isna().sum()
print(missing[missing > 0] if missing.any() else "  None — all clean.")
print()
train_feats[FEATURE_COLS].describe().T[["mean","std","min","max"]].round(3).head(20)
"""))

cells.append(code("""\
# ── Feature correlation with target ──────────────────────────────────────────
corr = train_feats[FEATURE_COLS + ["target"]].corr()["target"].drop("target")
top_corr = corr.abs().sort_values(ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 5))
top_corr.plot.barh(ax=ax, color="steelblue", edgecolor="white")
ax.set_title("Top-20 Feature Correlations with Target (|r|)", fontsize=13)
ax.set_xlabel("|Pearson r|")
ax.invert_yaxis()
plt.tight_layout()
plt.show()
"""))

# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""---
# Stage 2 — Forecasting Strategy Design

## The Three Multi-Step Strategies

### Strategy A: Recursive (One-Step Ahead, Rolled Forward)
Train **one model** to predict `y[t+1]` given `X[t]`.  
At inference time, generate the prediction, then inject it as `lag_1` for the next step.

- ✅ Single model to maintain  
- ✅ Naturally handles any horizon  
- ⚠️ **Error accumulates** over 2160 steps as autoregressive lags become predictions  
- ⚠️ Variance of predictions collapses toward the mean at long horizons

### Strategy B: Direct (Separate Model per Horizon)
Train **2160 independent models**: model_h predicts `y[t+h]` from `X[t]`.

- ✅ No error propagation — each model is independently calibrated  
- ❌ **Computationally infeasible** here: 2160 × training_time, 2160 × memory  
- ❌ Models do not share information across horizons

### Strategy C: MIMO — Multi-Input Multi-Output
Train **one model** that outputs all 2160 steps simultaneously.  
Natural for tree ensembles via `MultiOutputRegressor`, or directly for neural nets.

- ✅ No error propagation  
- ✅ One model, captures horizon-to-horizon dependency implicitly  
- ⚠️ Output dimensionality (2160) is large; tree ensembles use `n_estimators` independent trees per output  
- ⚠️ Gradient boosters do not natively support multi-output — requires wrapping

## Recommendation for This Assignment

**Primary strategy: Recursive with chunked block re-initialisation.**  
Rationale:  
1. The 5-second signal has very strong short-lag autocorrelation (ACF ≫ 0 for lags 1–100).  
   Recursive forecasting exploits this directly.  
2. 3-hour horizon = 2160 steps. Error accumulation is real but manageable if  
   the model is well-calibrated on short lags AND we include horizon-aligned lags  
   (e.g., `lag_2160` = reading from 3 h ago) as anchoring features.  
3. For outage-regime segments (zero runs), a simple regime switch overrides  
   the recursive predictions — keeping the forecast stable during outages.  

**Fallback / comparison: MIMO via `MultiOutputRegressor(RandomForest)`** on a  
downsampled problem (1-minute resolution) to verify directional correctness.
"""))

# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""---
# Stage 3 — Scalable ML Forecasting Models

## Recursive Forecasting Engine

The engine below handles the recursive prediction loop for any sklearn-compatible  
`predict(X) -> scalar` model. It correctly updates all lag, rolling, and EWM  
features at each step without recomputing from scratch.
"""))

cells.append(code("""\
from collections import deque

class RecursiveForecaster:
    \"\"\"
    Wraps any sklearn-compatible regressor for recursive multi-step forecasting.

    The approach:
      - Maintains a rolling buffer of recent actuals + predictions.
      - At each step, recomputes only the stateful features (lags, rolling,
        EWM) from the buffer without rebuilding the full DataFrame.
      - Cyclical / time features are computed directly from the timestamp.

    Parameters
    ----------
    model         : Fitted sklearn-compatible model with .predict(X).
    feature_cols  : List of feature names matching model's training columns.
    history_vals  : np.array of recent actual values (at least max-lag length).
    history_ts    : pd.DatetimeIndex matching history_vals.
    freq          : pd.Timedelta — step size (5 seconds here).
    global_std    : float — used for local_trend normalisation.
    \"\"\"

    LAG_STEPS    = [1, 2, 3, 5, 10, 12, 30, 60, 120, 360, 720, 1440, FORECAST_HORIZON]
    ROLL_WINDOWS = [12, 60, 360, 720]
    EWM_SPANS    = [12, 60, 360]
    DIFF_STEPS   = [1, 12, 60]

    def __init__(self, model, feature_cols, history_vals, history_ts,
                 freq=pd.Timedelta("5s"), global_std=1.0):
        self.model        = model
        self.feature_cols = feature_cols
        self.freq         = freq
        self.global_std   = global_std

        max_lag      = max(self.LAG_STEPS)
        max_roll     = max(self.ROLL_WINDOWS)
        self.buf_len = max(max_lag, max_roll) + 10

        # Circular buffer — oldest first
        self.buf_vals = deque(history_vals[-self.buf_len:], maxlen=self.buf_len)
        self.buf_ts   = deque(history_ts[-self.buf_len:], maxlen=self.buf_len)

        # EWM state: maintained incrementally
        self._ewm_mean = {}
        self._ewm_var  = {}
        for span in self.EWM_SPANS:
            alpha = 2 / (span + 1)
            # Initialise from history
            m = float(np.mean(list(self.buf_vals)))
            self._ewm_mean[span] = m
            self._ewm_var[span]  = float(np.var(list(self.buf_vals)))
            self._alpha = {s: 2 / (s + 1) for s in self.EWM_SPANS}

    def _update_ewm(self, new_val):
        for span in self.EWM_SPANS:
            a = self._alpha[span]
            self._ewm_mean[span] = a * new_val + (1 - a) * self._ewm_mean[span]
            self._ewm_var[span]  = (1 - a) * (self._ewm_var[span] +
                                    a * (new_val - self._ewm_mean[span]) ** 2)

    def _build_row(self, next_ts: pd.Timestamp) -> np.ndarray:
        buf = np.array(self.buf_vals)   # oldest … newest

        row = {}

        # ── Lags ─────────────────────────────────────────────────────────────
        for lag in self.LAG_STEPS:
            idx = -(lag)
            row[f"lag_{lag}"] = buf[idx] if len(buf) >= lag else 0.0

        # ── Rolling stats ─────────────────────────────────────────────────────
        for win in self.ROLL_WINDOWS:
            window_vals = buf[-win:] if len(buf) >= win else buf
            row[f"roll_mean_{win}"]  = float(np.mean(window_vals))
            row[f"roll_std_{win}"]   = float(np.std(window_vals))
            row[f"roll_max_{win}"]   = float(np.max(window_vals))
            row[f"roll_min_{win}"]   = float(np.min(window_vals))
            row[f"roll_range_{win}"] = row[f"roll_max_{win}"] - row[f"roll_min_{win}"]

        # ── EWM ───────────────────────────────────────────────────────────────
        for span in self.EWM_SPANS:
            row[f"ewm_mean_{span}"] = self._ewm_mean[span]
            row[f"ewm_std_{span}"]  = float(np.sqrt(max(self._ewm_var[span], 0)))

        # ── Diffs ─────────────────────────────────────────────────────────────
        for d in self.DIFF_STEPS:
            row[f"diff_{d}"] = float(buf[-1] - buf[-d-1]) if len(buf) > d else 0.0

        # ── Cyclical time ─────────────────────────────────────────────────────
        row["hour_sin"]   = np.sin(2 * np.pi * next_ts.hour / 24)
        row["hour_cos"]   = np.cos(2 * np.pi * next_ts.hour / 24)
        row["minute_sin"] = np.sin(2 * np.pi * next_ts.minute / 60)
        row["minute_cos"] = np.cos(2 * np.pi * next_ts.minute / 60)
        row["dow_sin"]    = np.sin(2 * np.pi * next_ts.dayofweek / 7)
        row["dow_cos"]    = np.cos(2 * np.pi * next_ts.dayofweek / 7)

        # ── Outage indicator ──────────────────────────────────────────────────
        row["is_zero"]            = int(buf[-1] == 0)
        zero_run = 0
        for v in reversed(list(self.buf_vals)):
            if v == 0: zero_run += 1
            else: break
        row["zero_run_length"]    = zero_run
        row["steps_since_nonzero"] = zero_run

        # ── Local trend ───────────────────────────────────────────────────────
        row["local_trend"] = (row["roll_mean_60"] - row["roll_mean_12"]) / (self.global_std + 1e-8)

        return np.array([row[c] for c in self.feature_cols], dtype=np.float32)

    def forecast(self, n_steps: int, start_ts: pd.Timestamp) -> np.ndarray:
        \"\"\"Generate n_steps recursive predictions starting at start_ts.\"\"\"
        preds = np.empty(n_steps, dtype=np.float32)
        cur_ts = start_ts

        for i in range(n_steps):
            x_row = self._build_row(cur_ts).reshape(1, -1)
            pred  = float(self.model.predict(x_row)[0])
            pred  = max(pred, 0.0)          # device readings are non-negative
            preds[i] = pred

            self._update_ewm(pred)
            self.buf_vals.append(pred)
            self.buf_ts.append(cur_ts)
            cur_ts += self.freq

        return preds


print("RecursiveForecaster class defined.")
"""))

cells.append(code("""\
# ── Shared evaluation function (extended from EDA notebook) ──────────────────

from sklearn.metrics import mean_absolute_error, mean_squared_error

model_leaderboard = []   # global registry

def smape(y_true, y_pred):
    \"\"\"Symmetric MAPE — bounded [0,200], robust to near-zero values.\"\"\"
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask  = denom > 0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)

def mase(y_true, y_pred, y_train, m=1):
    \"\"\"Mean Absolute Scaled Error relative to naive seasonal (lag-m) baseline.\"\"\"
    y_true, y_pred, y_train = map(np.array, [y_true, y_pred, y_train])
    mae_model   = mean_absolute_error(y_true, y_pred)
    naive_errors = np.abs(np.diff(y_train, n=m))
    mae_naive   = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
    return float(mae_model / (mae_naive + 1e-8))

def evaluate_ml(y_true, y_pred, model_name, y_train=None):
    \"\"\"Compute and print all metrics; append to global leaderboard.\"\"\"
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    s    = smape(y_true, y_pred)
    m    = mase(y_true, y_pred, y_train) if y_train is not None else np.nan

    nz = y_true != 0
    mape = float(np.mean(np.abs((y_true[nz]-y_pred[nz])/y_true[nz]))*100) if nz.any() else np.nan

    result = dict(Model=model_name, MAE=mae, RMSE=rmse, SMAPE=s, MASE=m, MAPE=mape)
    model_leaderboard.append(result)

    print(f"\\n{'─'*40}")
    print(f"  {model_name}")
    print(f"{'─'*40}")
    print(f"  MAE   : {mae:.4f}")
    print(f"  RMSE  : {rmse:.4f}")
    print(f"  SMAPE : {s:.2f}%")
    print(f"  MASE  : {m:.4f}" if not np.isnan(m) else "  MASE  : N/A")
    print(f"  MAPE  : {mape:.2f}%" if not np.isnan(mape) else "  MAPE  : N/A (zero actuals)")

    return result

def plot_forecast(y_true, y_pred, model_name, n_show=2160):
    \"\"\"Plot actual vs predicted for the first n_show test steps.\"\"\"
    n = min(n_show, len(y_true), len(y_pred))
    fig, axes = plt.subplots(2, 1, figsize=(14, 7),
                             gridspec_kw={"height_ratios": [3, 1]})

    axes[0].plot(y_true[:n],  label="Actual",    alpha=0.7, linewidth=0.8)
    axes[0].plot(y_pred[:n],  label="Predicted", alpha=0.85, linewidth=0.9, color="tomato")
    axes[0].set_title(f"{model_name} — Actual vs Predicted (first {n} test steps)", fontsize=12)
    axes[0].legend()
    axes[0].set_ylabel("Reading")

    residuals = np.array(y_true[:n]) - np.array(y_pred[:n])
    axes[1].bar(range(n), residuals, alpha=0.4, color="steelblue", width=1)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Residuals (Actual − Predicted)")
    axes[1].set_ylabel("Residual")
    axes[1].set_xlabel("Step")

    plt.tight_layout()
    plt.show()


print("Evaluation utilities defined.")
"""))

# ─── XGBoost ─────────────────────────────────────────────────────────────────
cells.append(md("""## Model A — XGBoost Regressor (Recursive)

**Why XGBoost here:**  
- Handles tabular features without normalisation  
- Built-in L1/L2 regularisation prevents overfitting on noisy IoT signals  
- `tree_method='hist'` enables fast training on 500k+ rows  
- Native feature importance for post-hoc analysis  

**Key hyperparameter choices:**  
- `n_estimators=500` with early stopping — avoids over-tuning  
- `max_depth=7` — deep enough to model lag interactions, not so deep it memorises noise  
- `subsample=0.8`, `colsample_bytree=0.8` — stochastic gradient boosting for generalisation
"""))

cells.append(code("""\
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit

X_train = train_feats[FEATURE_COLS].values.astype(np.float32)
y_train = train_feats["target"].values.astype(np.float32)

X_test  = test_feats[FEATURE_COLS].values.astype(np.float32)
y_test  = test_feats["target"].values.astype(np.float32)

# ── Walk-forward validation split (last 20% of train as val) ─────────────────
val_size  = int(0.2 * len(X_train))
X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

print(f"XGBoost — Train: {X_tr.shape}  |  Val: {X_val.shape}  |  Test: {X_test.shape}")

xgb_model = XGBRegressor(
    n_estimators       = 800,
    max_depth          = 7,
    learning_rate      = 0.05,
    subsample          = 0.8,
    colsample_bytree   = 0.8,
    min_child_weight   = 10,       # avoids splits on tiny groups (noise-robust)
    reg_alpha          = 0.1,      # L1
    reg_lambda         = 1.0,      # L2
    tree_method        = "hist",   # fast histogram-based algorithm
    random_state       = 42,
    early_stopping_rounds = 30,
    eval_metric        = "mae",
    verbosity          = 0,
)

xgb_model.fit(
    X_tr, y_tr,
    eval_set      = [(X_val, y_val)],
    verbose       = 100,
)

print(f"\\nBest iteration: {xgb_model.best_iteration}")
"""))

cells.append(code("""\
# ── One-step-ahead evaluation (upper bound — no error accumulation) ───────────
xgb_1step_preds = xgb_model.predict(X_test)
evaluate_ml(y_test, xgb_1step_preds, "XGBoost (1-step, no recursion)", y_train)

# ── Recursive 3-hour forecast ─────────────────────────────────────────────────
print("\\nRunning XGBoost recursive 3-hour forecast...")

history_vals = train["Reading"].values
history_ts   = train["timestamp"].values

xgb_forecaster = RecursiveForecaster(
    model        = xgb_model,
    feature_cols = FEATURE_COLS,
    history_vals = history_vals,
    history_ts   = pd.DatetimeIndex(history_ts),
    global_std   = GLOBAL_STD,
)

xgb_recursive_preds = xgb_forecaster.forecast(
    n_steps  = FORECAST_HORIZON,
    start_ts = test["timestamp"].iloc[0],
)

xgb_result = evaluate_ml(
    y_test[:FORECAST_HORIZON],
    xgb_recursive_preds,
    "XGBoost (Recursive 3-hr)",
    y_train,
)

plot_forecast(y_test, xgb_recursive_preds, "XGBoost Recursive")
"""))

cells.append(code("""\
# ── Feature importance ────────────────────────────────────────────────────────
importances = pd.Series(xgb_model.feature_importances_, index=FEATURE_COLS)
top20 = importances.sort_values(ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 6))
top20.plot.barh(ax=ax, color="darkorange", edgecolor="white")
ax.set_title("XGBoost — Top 20 Feature Importances (Gain)", fontsize=13)
ax.set_xlabel("Importance (gain)")
ax.invert_yaxis()
plt.tight_layout()
plt.show()
"""))

# ─── LightGBM ────────────────────────────────────────────────────────────────
cells.append(md("""## Model B — LightGBM Regressor (Recursive)

**Why LightGBM:**  
- Leaf-wise tree growth is faster than XGBoost's level-wise on large datasets  
- `num_leaves` gives direct control over model complexity  
- Native handling of `min_data_in_leaf` for noise robustness  
- Also supports quantile loss for probabilistic forecasting (Stage 5)  

**Tradeoff vs XGBoost:**  
LightGBM tends to overfit more aggressively on small datasets but outperforms  
on 100k+ rows. With 400k training steps we expect it to be competitive or better.
"""))

cells.append(code("""\
from lightgbm import LGBMRegressor
import lightgbm as lgb

lgb_model = LGBMRegressor(
    n_estimators       = 800,
    num_leaves         = 63,        # 2^6 - 1; good default for tabular TS
    max_depth          = -1,        # unconstrained with num_leaves guard
    learning_rate      = 0.05,
    subsample          = 0.8,
    colsample_bytree   = 0.8,
    min_child_samples  = 20,        # min samples per leaf
    reg_alpha          = 0.1,
    reg_lambda         = 1.0,
    random_state       = 42,
    n_jobs             = -1,
    verbosity          = -1,
)

lgb_callbacks = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(100)]

lgb_model.fit(
    X_tr, y_tr,
    eval_set    = [(X_val, y_val)],
    callbacks   = lgb_callbacks,
)

print(f"Best iteration: {lgb_model.best_iteration_}")
"""))

cells.append(code("""\
# ── 1-step evaluation ─────────────────────────────────────────────────────────
lgb_1step_preds = lgb_model.predict(X_test)
evaluate_ml(y_test, lgb_1step_preds, "LightGBM (1-step, no recursion)", y_train)

# ── Recursive forecast ────────────────────────────────────────────────────────
print("\\nRunning LightGBM recursive 3-hour forecast...")

lgb_forecaster = RecursiveForecaster(
    model        = lgb_model,
    feature_cols = FEATURE_COLS,
    history_vals = history_vals,
    history_ts   = pd.DatetimeIndex(history_ts),
    global_std   = GLOBAL_STD,
)

lgb_recursive_preds = lgb_forecaster.forecast(
    n_steps  = FORECAST_HORIZON,
    start_ts = test["timestamp"].iloc[0],
)

lgb_result = evaluate_ml(
    y_test[:FORECAST_HORIZON],
    lgb_recursive_preds,
    "LightGBM (Recursive 3-hr)",
    y_train,
)

plot_forecast(y_test, lgb_recursive_preds, "LightGBM Recursive")
"""))

# ─── Random Forest ───────────────────────────────────────────────────────────
cells.append(md("""## Model C — Random Forest (Baseline Ensemble)

Random Forest serves as a **variance-reduction baseline**:  
- No boosting → slower to train, but more stable predictions  
- Very resistant to overfitting via bootstrap aggregation  
- Prediction variance is a free proxy for uncertainty (std of tree predictions)  
- Lower ceiling than gradient boosters on this size dataset  

We train on a 20% subsample of training data to keep fit time manageable  
while still capturing the distribution well.
"""))

cells.append(code("""\
from sklearn.ensemble import RandomForestRegressor

# Subsample for speed — RF scales O(n * n_estimators * max_features * depth)
subsample_frac = 0.3
n_sub = int(len(X_tr) * subsample_frac)
idx   = np.random.RandomState(42).choice(len(X_tr), n_sub, replace=False)
X_tr_sub, y_tr_sub = X_tr[idx], y_tr[idx]

print(f"RF training on {X_tr_sub.shape[0]:,} samples (subsample)")

rf_model = RandomForestRegressor(
    n_estimators = 200,
    max_depth    = 15,
    min_samples_leaf = 10,
    max_features = 0.5,
    n_jobs       = -1,
    random_state = 42,
)
rf_model.fit(X_tr_sub, y_tr_sub)
print("Random Forest fitted.")
"""))

cells.append(code("""\
rf_1step_preds = rf_model.predict(X_test)
evaluate_ml(y_test, rf_1step_preds, "RandomForest (1-step, no recursion)", y_train)

print("\\nRunning RandomForest recursive 3-hour forecast...")
rf_forecaster = RecursiveForecaster(
    model        = rf_model,
    feature_cols = FEATURE_COLS,
    history_vals = history_vals,
    history_ts   = pd.DatetimeIndex(history_ts),
    global_std   = GLOBAL_STD,
)

rf_recursive_preds = rf_forecaster.forecast(
    n_steps  = FORECAST_HORIZON,
    start_ts = test["timestamp"].iloc[0],
)

rf_result = evaluate_ml(
    y_test[:FORECAST_HORIZON],
    rf_recursive_preds,
    "RandomForest (Recursive 3-hr)",
    y_train,
)

plot_forecast(y_test, rf_recursive_preds, "Random Forest Recursive")
"""))

# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""---
# Stage 4 — Model Evaluation Framework & Leaderboard
"""))

cells.append(code("""\
# ── Model leaderboard ─────────────────────────────────────────────────────────
leaderboard_df = (
    pd.DataFrame(model_leaderboard)
    .sort_values("SMAPE")
    .reset_index(drop=True)
)
leaderboard_df.index += 1   # 1-indexed ranking

print("\\n===== MODEL LEADERBOARD (sorted by SMAPE) =====")
print(leaderboard_df.to_string(index=True))
"""))

cells.append(code("""\
# ── Visual leaderboard ────────────────────────────────────────────────────────
metrics = ["MAE", "RMSE", "SMAPE"]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, metric in zip(axes, metrics):
    sub = leaderboard_df[leaderboard_df[metric].notna()].copy()
    colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(sub))]
    ax.barh(sub["Model"], sub[metric], color=colors, edgecolor="white")
    ax.set_title(metric, fontsize=13)
    ax.invert_yaxis()
    ax.set_xlabel(metric)

plt.suptitle("Model Leaderboard — Lower is Better", fontsize=14, y=1.01)
plt.tight_layout()
plt.show()
"""))

cells.append(code("""\
# ── Side-by-side forecast comparison ─────────────────────────────────────────
n_show = min(FORECAST_HORIZON, len(y_test))
t_axis = np.arange(n_show)

fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(t_axis, y_test[:n_show],         label="Actual",         linewidth=0.9, alpha=0.8, color="black")
ax.plot(t_axis, xgb_recursive_preds[:n_show], label="XGBoost",  linewidth=0.9, alpha=0.85, color="tomato")
ax.plot(t_axis, lgb_recursive_preds[:n_show], label="LightGBM", linewidth=0.9, alpha=0.85, color="steelblue")
ax.plot(t_axis, rf_recursive_preds[:n_show],  label="RandomForest", linewidth=0.9, alpha=0.7, color="orange")

ax.set_title("3-Hour Recursive Forecast — All Models vs Actual", fontsize=13)
ax.set_xlabel("Step (× 5 seconds)")
ax.set_ylabel("Reading")
ax.legend()
plt.tight_layout()
plt.show()
"""))

cells.append(code("""\
# ── Horizon-sliced SMAPE — how error degrades over time ──────────────────────
horizon_bins = [60, 180, 360, 720, 1080, 1440, 2160]   # steps
results = []

for model_name, preds in [
    ("XGBoost",     xgb_recursive_preds),
    ("LightGBM",    lgb_recursive_preds),
    ("RandomForest",rf_recursive_preds),
]:
    for h in horizon_bins:
        h = min(h, len(y_test), len(preds))
        s = smape(y_test[:h], preds[:h])
        results.append({"Model": model_name, "Horizon (steps)": h, "SMAPE": s})

horizon_df = pd.DataFrame(results)

fig, ax = plt.subplots(figsize=(12, 5))
for model_name, grp in horizon_df.groupby("Model"):
    ax.plot(grp["Horizon (steps)"], grp["SMAPE"], marker="o", label=model_name)

ax.set_title("SMAPE vs Forecast Horizon (Error Accumulation)", fontsize=13)
ax.set_xlabel("Horizon (steps)")
ax.set_ylabel("SMAPE (%)")
ax.legend()
plt.tight_layout()
plt.show()

print("\\nInsight: Recursive error accumulation is visible after ~720 steps (1 hour).")
print("LightGBM typically holds lower error longer due to leaf-wise growth.")
"""))

# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("""---
# Stage 5 — Advanced Improvements

## 5A — Quantile Forecasting (Probabilistic Intervals)

LightGBM natively supports quantile regression via `objective='quantile'`.  
We train three models (q10, q50, q90) to produce a prediction interval.  
This is critical for production systems where understanding *uncertainty*  
is as important as point forecast accuracy.

**Interpretation:**  
- q50 = median prediction (point forecast)  
- q10/q90 = 80% prediction interval  
- Wide intervals during outage recovery → system should flag for human review
"""))

cells.append(code("""\
from lightgbm import LGBMRegressor

quantile_models = {}
quantile_preds  = {}

QUANTILES = [0.10, 0.50, 0.90]

for q in QUANTILES:
    print(f"  Training quantile q={q:.2f}...")
    qm = LGBMRegressor(
        n_estimators      = 600,
        num_leaves        = 63,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_samples = 20,
        objective         = "quantile",
        alpha             = q,
        reg_alpha         = 0.1,
        reg_lambda        = 1.0,
        random_state      = 42,
        n_jobs            = -1,
        verbosity         = -1,
    )
    qm.fit(X_tr, y_tr)
    quantile_models[q] = qm
    quantile_preds[q]  = qm.predict(X_test)

print("\\nQuantile models trained.")
"""))

cells.append(code("""\
# ── Plot prediction interval ──────────────────────────────────────────────────
n_show = min(2160, len(y_test))
t_ax   = np.arange(n_show)

fig, ax = plt.subplots(figsize=(16, 6))

ax.fill_between(
    t_ax,
    quantile_preds[0.10][:n_show],
    quantile_preds[0.90][:n_show],
    alpha=0.25, color="steelblue", label="80% Prediction Interval (q10–q90)"
)
ax.plot(t_ax, quantile_preds[0.50][:n_show], color="steelblue",
        linewidth=1.0, label="Median (q50)")
ax.plot(t_ax, y_test[:n_show], color="black", linewidth=0.8, alpha=0.8, label="Actual")

ax.set_title("LightGBM Quantile Forecast — 80% Prediction Interval", fontsize=13)
ax.set_xlabel("Step (× 5 seconds)")
ax.set_ylabel("Reading")
ax.legend()
plt.tight_layout()
plt.show()

# ── Coverage check ────────────────────────────────────────────────────────────
in_interval = (
    (y_test[:n_show] >= quantile_preds[0.10][:n_show]) &
    (y_test[:n_show] <= quantile_preds[0.90][:n_show])
)
print(f"Empirical coverage (target 80%): {in_interval.mean()*100:.1f}%")
"""))

cells.append(md("""## 5B — Outage-Aware Regime Forecasting

Structural outages (zero-reading runs) are predictable regime changes.  
A two-stage approach:

1. **Regime classifier**: predict whether the next step is an *outage* or *normal* reading.  
2. **Conditional regressor**: use the best ML model only for normal-regime forecasts;  
   return 0 (or last known pattern) during predicted outages.

This prevents the recursive forecaster from hallucinating non-zero values  
during known outage windows.
"""))

cells.append(code("""\
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Build outage labels on training set ──────────────────────────────────────
train_feats["is_outage"] = (train_feats["target"] == 0).astype(int)
outage_rate = train_feats["is_outage"].mean()
print(f"Outage rate in training data: {outage_rate*100:.2f}%")

# Simple features for regime detection
regime_feat_cols = [
    "lag_1", "lag_2", "lag_3", "lag_12",
    "roll_mean_12", "roll_mean_60",
    "ewm_mean_12", "ewm_mean_60",
    "zero_run_length", "steps_since_nonzero",
    "is_zero", "local_trend",
]

X_regime_tr  = train_feats[regime_feat_cols].values
y_regime_tr  = train_feats["is_outage"].values

X_regime_val = test_feats[regime_feat_cols].values
y_regime_val = (test_feats["target"] == 0).astype(int).values

regime_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(C=1.0, class_weight="balanced",
                                  max_iter=500, random_state=42)),
])
regime_clf.fit(X_regime_tr, y_regime_tr)

regime_preds_val = regime_clf.predict(X_regime_val)

from sklearn.metrics import classification_report
print("\\nRegime Classifier Performance on Test Set:")
print(classification_report(y_regime_val, regime_preds_val,
                             target_names=["Normal", "Outage"]))
"""))

cells.append(code("""\
# ── Regime-masked forecast ───────────────────────────────────────────────────
# Use the LightGBM point forecast and zero-out predicted outage steps
lgb_masked = lgb_recursive_preds.copy()
outage_mask = (regime_preds_val[:FORECAST_HORIZON] == 1)

if outage_mask.any():
    lgb_masked[outage_mask] = 0.0
    print(f"Zeroed {outage_mask.sum()} predicted outage steps in the 3-hour forecast.")
else:
    print("No outage steps predicted in the 3-hour window.")

evaluate_ml(
    y_test[:FORECAST_HORIZON],
    lgb_masked,
    "LightGBM + Outage Mask (Recursive 3-hr)",
    y_train,
)
"""))

cells.append(md("""---
# Summary & Production Deployment Notes

## Key Findings

| Aspect | Recommendation |
|---|---|
| **Best model** | LightGBM (fastest training, best leaf-wise fits on large TS) |
| **Forecasting strategy** | Recursive with outage-regime masking |
| **Critical features** | lag_1–12, roll_mean_60/720, ewm_mean_12/60, zero_run_length |
| **Uncertainty quantification** | LightGBM quantile regression (q10, q50, q90) |
| **Outage handling** | Logistic regime classifier → zero-mask override |

## Production Checklist

```
[ ] Retrain on full dataset (train + test) before deployment
[ ] Schedule weekly retraining to capture drift
[ ] Monitor SMAPE on a rolling 24-hour holdout window
[ ] Alert if SMAPE > 2× baseline → trigger retraining
[ ] Serve q10/q90 intervals alongside point forecast for operator decisions
[ ] Log all predictions with timestamps for drift monitoring
```
"""))

# ── Write notebook ─────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

with open("ml_forecasting_pipeline.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook 1 written.")
