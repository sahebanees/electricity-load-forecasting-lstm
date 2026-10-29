# Short-Term Electricity Load Forecasting: LSTM vs Machine Learning Baselines

Forecasts the next hour of electricity demand from past demand and calendar features, and compares a deep learning model (LSTM) against simpler baselines.

## Problem
Utilities need accurate short-term demand forecasts for generation scheduling and grid planning. This project builds an end-to-end Python pipeline for hourly load forecasting and evaluates it honestly on unseen, later-in-time data.

## Dataset
- **PJM Hourly Energy Consumption** (PJME region), hourly demand in MW, from Kaggle:
  https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
- Download `PJME_hourly.csv` and place it next to `forecast.py`. The file is not stored in this repo because of its size.
- The most recent 35,000 hours are used by default (change with `--last-n`).

## Approach
1. **Cleaning:** sort by time, remove duplicate timestamps, fill missing hours by interpolation.
2. **Features:** hour, day of week, month, weekend flag, demand 1 hour / 24 hours / 168 hours ago, and the previous 24-hour rolling mean.
3. **Split:** time-based, 70% train / 15% validation / 15% test. No random shuffling, so the model never sees the future.
4. **Models:** Naive baseline (same hour yesterday), Linear Regression, Random Forest (scikit-learn), and an LSTM (TensorFlow/Keras) that reads the previous 24 hours. Scaling is fitted on training data only.
5. **Evaluation:** MAE, RMSE and MAPE on the test set.

## Results
<!-- Paste the table from results/metrics.csv after you run the code. Use YOUR numbers only. -->

| Model | MAE (MW) | RMSE (MW) | MAPE (%) |
|---|---|---|---|
| Naive (same hour yesterday) | <<fill>> | <<fill>> | <<fill>> |
| Linear Regression | <<fill>> | <<fill>> | <<fill>> |
| Random Forest | <<fill>> | <<fill>> | <<fill>> |
| LSTM | <<fill>> | <<fill>> | <<fill>> |

**Observations:** <<Write 2-3 sentences on which model performed best and why, based on your table and the plots in `results/figures/`.>>

![Actual vs predicted](results/figures/02_forecast_week.png)
![Model comparison](results/figures/03_model_comparison.png)
![Error by hour](results/figures/04_error_by_hour.png)

## How to run
```bash
pip install -r requirements.txt
python forecast.py --data PJME_hourly.csv
```
Optional flags: `--epochs 5` (faster LSTM training), `--skip-lstm` (baselines only), `--last-n 20000` (less data).
Outputs are saved to `results/` (metrics, predictions and figures).

## Limitations and future work
- Uses a single region and no weather or holiday features.
- The LSTM is a small single-layer model; hyperparameters were not extensively tuned.
- Forecasts only one hour ahead; multi-step horizons would be a natural extension.
- Possible next steps: add temperature data, tune the models, and serve the best model through a simple API.

## Tech stack
Python, NumPy, pandas, scikit-learn, TensorFlow/Keras, Matplotlib
