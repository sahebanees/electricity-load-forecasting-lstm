"""
Short-Term Electricity Load Forecasting: LSTM vs Machine Learning baselines
===========================================================================
Forecasts the NEXT HOUR of electricity demand from past demand + calendar features.

Pipeline
  1. Load hourly demand data, sort by time, remove duplicates, fill missing hours
  2. Feature engineering: calendar features, lag features, rolling mean
  3. Time-based split (70% train / 15% validation / 15% test) - NO random shuffling
  4. Models: Naive baseline, Linear Regression, Random Forest, LSTM
  5. Evaluate on the unseen test set with MAE, RMSE and MAPE
  6. Save metrics, predictions and figures into the results/ folder

Usage
  python forecast.py --data PJME_hourly.csv
  python forecast.py --data PJME_hourly.csv --epochs 5      (faster)
  python forecast.py --data PJME_hourly.csv --skip-lstm     (only ML baselines)
"""
import argparse
import os
import random

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # save figures to files, no pop-up windows
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

SEED = 42
WINDOW = 24  # LSTM looks at the previous 24 hours
np.random.seed(SEED)
random.seed(SEED)


# ----------------------------------------------------------------------------
# 1. Data loading and cleaning
# ----------------------------------------------------------------------------
def load_data(path, datetime_col=None, target_col=None, last_n=None):
    """Read the CSV and return a clean, regular hourly Series of demand."""
    df = pd.read_csv(path)
    datetime_col = datetime_col or df.columns[0]  # PJM files: first column = Datetime
    target_col = target_col or df.columns[1]      # second column = demand in MW
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    s = df.set_index(datetime_col)[target_col].sort_index()
    s = s[~s.index.duplicated(keep="first")]       # remove duplicate timestamps
    s = s.asfreq("h")                              # force a regular hourly index
    n_missing = int(s.isna().sum())
    s = s.interpolate(limit_direction="both")      # fill missing hours linearly
    print(f"Loaded {len(s)} hourly rows ({s.index.min()} -> {s.index.max()}); "
          f"filled {n_missing} missing hours.")
    if last_n:
        s = s.iloc[-last_n:]                       # keep the most recent N hours
        print(f"Using the most recent {len(s)} hours.")
    s.name = "demand_mw"
    return s


# ----------------------------------------------------------------------------
# 2. Feature engineering
# ----------------------------------------------------------------------------
def make_features(s):
    """Calendar, lag and rolling features. Lags use only PAST values (no leakage)."""
    X = pd.DataFrame(index=s.index)
    X["hour"] = s.index.hour
    X["day_of_week"] = s.index.dayofweek
    X["month"] = s.index.month
    X["is_weekend"] = (s.index.dayofweek >= 5).astype(int)
    X["lag_1h"] = s.shift(1)        # demand 1 hour ago
    X["lag_24h"] = s.shift(24)      # same hour yesterday
    X["lag_168h"] = s.shift(168)    # same hour last week
    X["roll_mean_24h"] = s.shift(1).rolling(24).mean()  # average of previous 24 hours
    X = X.dropna()
    y = s.loc[X.index]
    return X, y


def make_windows(values, window):
    """Turn a 1-D array into (samples, window, 1) inputs and next-step targets."""
    xs = np.array([values[k - window:k] for k in range(window, len(values))])
    ys = values[window:]
    return xs[..., np.newaxis], ys


# ----------------------------------------------------------------------------
# 3. Metrics
# ----------------------------------------------------------------------------
def evaluate(name, y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "model": name,
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE": round(mean_squared_error(y_true, y_pred) ** 0.5, 2),
        "MAPE_%": round(float(np.mean(np.abs((y_true - y_pred) / y_true))) * 100, 2),
    }


# ----------------------------------------------------------------------------
# 4. LSTM model
# ----------------------------------------------------------------------------
def train_lstm(y, train_end, val_end, epochs, batch_size=128):
    """Train an LSTM on scaled demand. Returns (test predictions in MW, history)."""
    import tensorflow as tf  # imported here so --skip-lstm works without TensorFlow

    tf.random.set_seed(SEED)

    # Scale to 0-1 using TRAINING data only (otherwise information leaks from the future)
    scaler = MinMaxScaler().fit(y.iloc[:train_end].values.reshape(-1, 1))
    scaled = scaler.transform(y.values.reshape(-1, 1)).ravel()

    xs, ys = make_windows(scaled, WINDOW)
    # window k predicts position k+WINDOW of y, so shift the split points
    tr = slice(0, train_end - WINDOW)
    va = slice(train_end - WINDOW, val_end - WINDOW)
    te = slice(val_end - WINDOW, None)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(WINDOW, 1)),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    history = model.fit(
        xs[tr], ys[tr],
        validation_data=(xs[va], ys[va]),
        epochs=epochs, batch_size=batch_size,
        callbacks=[early_stop], verbose=2,
    )
    pred_scaled = model.predict(xs[te], verbose=0)
    return scaler.inverse_transform(pred_scaled).ravel(), history


# ----------------------------------------------------------------------------
# 5. Figures
# ----------------------------------------------------------------------------
def save_figures(s, y_test, preds, results, history, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)

    # (a) overview of the whole series (daily average, so it is readable)
    plt.figure(figsize=(12, 4))
    s.resample("D").mean().plot(color="steelblue", linewidth=0.8)
    plt.title("Electricity demand - daily average")
    plt.ylabel("Demand (MW)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "01_demand_overview.png"), dpi=130)
    plt.close()

    # (b) actual vs predicted for the first week of the test set
    n = 168
    plt.figure(figsize=(12, 4.5))
    plt.plot(y_test.index[:n], y_test.values[:n], label="Actual", color="black", linewidth=2)
    for name, p in preds.items():
        if name.startswith("Naive") or name == "Linear Regression":
            continue
        plt.plot(y_test.index[:n], np.asarray(p)[:n], label=name, linewidth=1.3)
    plt.title("Actual vs predicted demand - first week of the test set")
    plt.ylabel("Demand (MW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "02_forecast_week.png"), dpi=130)
    plt.close()

    # (c) model comparison
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(results["model"], results["MAE"], color="steelblue")
    ax[0].set_title("MAE (MW) - lower is better")
    ax[1].bar(results["model"], results["MAPE_%"], color="darkorange")
    ax[1].set_title("MAPE (%) - lower is better")
    for a in ax:
        a.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "03_model_comparison.png"), dpi=130)
    plt.close()

    # (d) error by hour of day -> shows peak vs off-peak behaviour
    plt.figure(figsize=(10, 4))
    for name, p in preds.items():
        if name.startswith("Naive"):
            continue
        err = pd.Series(np.abs(y_test.values - np.asarray(p)), index=y_test.index)
        err.groupby(err.index.hour).mean().plot(label=name)
    plt.title("Mean absolute error by hour of day")
    plt.xlabel("Hour of day")
    plt.ylabel("MAE (MW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "04_error_by_hour.png"), dpi=130)
    plt.close()

    # (e) LSTM training curve
    if history is not None:
        plt.figure(figsize=(7, 4))
        plt.plot(history.history["loss"], label="Training loss")
        plt.plot(history.history["val_loss"], label="Validation loss")
        plt.title("LSTM training curve")
        plt.xlabel("Epoch")
        plt.ylabel("MSE (scaled data)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "05_lstm_training_loss.png"), dpi=130)
        plt.close()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Short-term electricity load forecasting")
    ap.add_argument("--data", required=True, help="CSV with a datetime column and a demand column")
    ap.add_argument("--datetime-col", default=None)
    ap.add_argument("--target-col", default=None)
    ap.add_argument("--last-n", type=int, default=35000, help="use only the most recent N hours")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--skip-lstm", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    s = load_data(args.data, args.datetime_col, args.target_col, args.last_n)
    X, y = make_features(s)

    # Time-based split: train (oldest) -> validation -> test (newest)
    n = len(X)
    train_end, val_end = int(n * 0.70), int(n * 0.85)
    X_tr, X_te = X.iloc[:train_end], X.iloc[val_end:]
    y_tr, y_te = y.iloc[:train_end], y.iloc[val_end:]
    print(f"Split -> train: {train_end}, validation: {val_end - train_end}, test: {n - val_end}")

    preds, rows = {}, []

    preds["Naive (same hour yesterday)"] = X_te["lag_24h"].values
    lin = LinearRegression().fit(X_tr, y_tr)
    preds["Linear Regression"] = lin.predict(X_te)
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=SEED).fit(X_tr, y_tr)
    preds["Random Forest"] = rf.predict(X_te)

    history = None
    if not args.skip_lstm:
        try:
            preds["LSTM"], history = train_lstm(y, train_end, val_end, args.epochs)
        except ImportError:
            print("TensorFlow is not installed - skipping the LSTM. "
                  "Run: pip install tensorflow")

    for name, p in preds.items():
        rows.append(evaluate(name, y_te.values, p))
    results = pd.DataFrame(rows)

    os.makedirs(args.out, exist_ok=True)
    results.to_csv(os.path.join(args.out, "metrics.csv"), index=False)
    pd.DataFrame({"actual_mw": y_te.values, **{k: np.asarray(v) for k, v in preds.items()}},
                 index=y_te.index).to_csv(os.path.join(args.out, "predictions.csv"))
    save_figures(s, y_te, preds, results, history, os.path.join(args.out, "figures"))

    print("\n=== Test-set results (lower is better) ===")
    print(results.to_string(index=False))
    print(f"\nSaved metrics, predictions and figures to '{args.out}/'")


if __name__ == "__main__":
    main()
