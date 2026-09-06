# Outlier Detection (`4.outlier_detection.py`) — `config.ini` Quick Reference

This guide documents **only what is implemented in this repository** (and matches the current `config.ini`).

## What the outlier step does (high level)

`scripts/1_data_preprocessing/4.outlier_detection.py` flags outliers using:

- **Physical limits** (hard bounds from `[physical_thresholds]`, plus optional seasonal overrides)
- **IQR** (statistical outliers; should be used only on stable sensor channels)
- **Valid states** (optional; only for discrete/step-like signals)

It produces plots/reports (if enabled) and a CSV of outlier flags used later by cleaning/interpolation and (optionally) LSTM training.

---

## Minimal recommended configuration (safe defaults)

Start conservative: physical limits + IQR only on stable sensors; do not enable valid-states unless you confirm discreteness.

```ini
[outlier_detection]
enable_physical_limits = 1
enable_iqr = 1
enable_valid_states = 0

# IQR sensitivity (higher = fewer flags)
iqr_multiplier = 3.5

# Only include stable continuous sensors (avoid setpoints and modulations)
iqr_columns = Temperatura Saturazione, Temperatura Esterna, Umidita Esterna

# Optional “data quality too bad” warning threshold
drop_threshold_percentage = 10

# Outputs
create_outlier_plots = 1
save_outlier_reports = 1
```

---

## Physical limits (hard bounds)

Physical bounds are defined in `config.ini` under:

- `[physical_thresholds]` (base limits)
- `[physical_thresholds_summer]` and `[physical_thresholds_winter]` (seasonal overrides, if needed)

Format:

```ini
[physical_thresholds]
Temperatura Ripresa = 10, 40
Temperatura Mandata = 10, 40
Temperatura Esterna = -10, 45
Modulazione Ventilatore Mandata = 0, 100
```

**Rules of thumb**
- Use physical limits for **everything** (sensors, modulations, setpoints).
- Keep limits wide enough to avoid flagging normal operating extremes, but narrow enough to catch obvious impossible values.

---

## IQR (statistical outliers) — use cautiously

IQR is useful for **stable sensors** with a single-regime distribution (e.g., weather channels, coil/saturation temperature).

Not recommended for:
- **Setpoints** (step-like changes are normal)
- **Control outputs** like fan modulation (often discrete/step control or multi-regime)
- Variables whose distribution changes by mode (heating/cooling/occupancy)

Config keys used:
- `enable_iqr`
- `iqr_multiplier`
- `iqr_columns`

Example:

```ini
[outlier_detection]
enable_iqr = 1
iqr_multiplier = 3.5
iqr_columns = Temperatura Saturazione, Temperatura Esterna, Umidita Esterna
```

---

## Valid states (discrete signals only)

Valid-states checking is **only correct** if the signal takes a small number of discrete values (e.g., staged fan control).

If the signal is continuously modulated (VFD), valid-states will create many false outliers.

Config keys:
- `enable_valid_states`
- `valid_states_columns`
- `<ColumnName>_valid_states` (per-column list)
- `valid_states_tolerance`

Template:

```ini
[outlier_detection]
enable_valid_states = 1
valid_states_columns = Modulazione Ventilatore Mandata
valid_states_tolerance = 0.5

Modulazione Ventilatore Mandata_valid_states = 0, 20, 50, 70, 80, 100
```

---

## Practical tuning checklist

- **Too many outliers**
  - widen `[physical_thresholds]` slightly if they are unrealistically tight
  - increase `iqr_multiplier` (e.g. 3.5 → 4.0)
  - remove unstable variables from `iqr_columns`
  - disable valid-states unless you have confirmed discreteness

- **Too few outliers / obvious errors not flagged**
  - check that the column names in `config.ini` match the post-merge names used in your CSVs
  - verify the column is included in the dataset (after renaming)
  - tighten physical limits slightly if they are too wide

---

## Where outputs go

Outputs are written under your configured `processed_data` / `plots` folders (see `config.ini` `[paths]`).

The next step (`5.cleaning and interpolation.py`) can read the outlier flag CSV and preserve those flags during interpolation/cleaning.

