# LSTM Autoencoder — HVAC Anomaly Detection (this folder)

Unsupervised anomaly detection for Air Handling Units using an LSTM encoder–decoder (autoencoder).
The model learns to reconstruct *normal* multivariate time-series windows; **high reconstruction error** indicates abnormal behaviour.

## What to run

- **Main entry point**: `1_main.py` (runs the full pipeline end-to-end)
- **Optional tuner**: `10_optuna_tuner.py` (searches model hyperparameters to reduce validation reconstruction loss)
- **Optional context plots**: `9_plot_anomaly_context.py` (deep-dive plots around detected anomalies)
- **Setup check**: `0_test_setup.py`

## File map

```
0_test_setup.py         verify environment + config access
1_main.py               run: load → preprocess → train → evaluate → plots → report
2_data_loader.py        load feature CSV, gap handling, date filtering
3_preprocessor.py       scaling, split strategy, gap-safe windowing, clean-training filters
4_model.py              stacked LSTM autoencoder (encoder/bottleneck/decoder)
5_trainer.py            training loop + callbacks (EarlyStopping, ReduceLROnPlateau, checkpoint)
6_evaluator.py          reconstruction error, thresholds, smoothing/persistence, severity
7_visualizer.py         plots (loss, error distribution, anomaly timeline, reconstructions)
8_report_generator.py   Word report
9_plot_anomaly_context.py  anomaly-by-anomaly context figures
10_optuna_tuner.py      Optuna hyperparameter search (val_loss proxy)
```

## Quick start (PowerShell)

1) Edit the four global variables in the project root `config.ini`:

- `building_id`, `ahu_unit`, `season`, `year`

2) Run:

```powershell
cd "scripts/5_lstm_anomaly_detection"
python 1_main.py
```

## Configuration (project root `config.ini`)

All parameters live in the project-level `config.ini` under `[lstm_autoencoder]`.

### Features

The active features are selected automatically by season:

```ini
features_summer = ...
features_winter = ...
```

All feature lists include **thermal sensors** (TM, TR, TS) plus optional seasonal modulation, and **calendar awareness**:

- `is_weekend`: Binary flag (0=Monday–Friday, 1=Saturday/Sunday)
  - **Why**: HVAC systems exhibit distinct usage patterns on weekends vs. weekdays
  - **Simplicity**: Single boolean is more interpretable than trigonometric encoding
  - **Trade-off**: Loses fine-grained day-of-week context (e.g., Monday vs. Friday specifics)

This simple weekday/weekend distinction is sufficient to improve anomaly detection by helping the model learn different "normal" behaviors for occupied vs. unoccupied periods.

### Split strategy (important)

Your pipeline uses chronological evaluation:

- `split_strategy = chronological` (strict out-of-time generalization)

This is set in `config.ini` under `[lstm_autoencoder]`.

### Thresholding

Thresholding is controlled by:

- `threshold_mode = percentile | std_dev | ema | manual`

Notes:
- `percentile` / `std_dev` compute a **training-anchored scalar threshold**.
- `ema` computes a **dynamic threshold on test errors** (plus a training EMA “anchor” for warm-start).

### Training details that matter

- **Scaler** is fit on training data only (prevents leakage).
- **Windows are gap-safe**: sequences are built only inside contiguous blocks (no window spans a time jump).
- **Shuffle** in `5_trainer.py` is `shuffle=True`, which shuffles the *order of windows* (not timesteps inside a window).

## Outputs

Relative to the HVAC project root:

- `processed_data/lstm_autoencoder/{building}_{unit}_{season}{year}/`
  - `anomaly_results_YYYYMMDD_HHMMSS.csv`
  - `summary_YYYYMMDD_HHMMSS.txt`
- `plots/lstm_autoencoder/{building}_{unit}_{season}{year}/`
  - training + anomaly plots
- `plots/lstm_autoencoder/models/`
  - `best_autoencoder_{building}_{unit}_{season}{year}.keras` plus versioned snapshots

## Notes on Optuna

`10_optuna_tuner.py` optimizes **validation reconstruction loss** (`val_loss`).
This is useful for model quality, but it is not a guarantee of “best anomaly alarms” because unsupervised anomaly detection has no ground-truth labels.
