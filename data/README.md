# Data

## ⚠️ Real data is not included

The analyses in this project were developed on **real Building Management System
(BMS) sensor data** from an Air Handling Unit (AHU UTA1, Building C1) collected
through an industry collaboration (Eurix / University of Turin). That data is
confidential and is **not** distributed with this repository.

## Synthetic sample data

To make the pipeline fully runnable, this folder contains a generator that
produces physically plausible synthetic data in the **exact format of the real
BMS export** — same columns, same units, same quirks (UTF-8 BOM, `sep=,` hint
line, ` °C` value suffixes, irregular gaps, sensor spikes):

```bash
python generate_sample_data.py
```

This creates:

| File | Contents |
|---|---|
| `C1_UTA1_Summer2025.csv` | AHU sensors, cooling season (2025-07-09 → 2025-10-14, 30-min) |
| `C1_UTA1_Winter2026.csv` | AHU sensors, heating season (2025-10-15 → 2026-02-18, 30-min) |
| `LIMF_from_2025-07-01_to_2026-02-18.csv` | Weather station data (temperature, relative humidity) |

The generator injects realistic problems for the pipeline to find: multi-hour
transmission gaps, sensor spike outliers, and a **synthetic spike** around
**2026-02-12** (supply temperature drops while the fan keeps running). That
spike is not the internship event and is not a confirmed coil, sensor, or
actuator fault.

## Data dictionary (AHU files)

| Column | Meaning | Unit |
|---|---|---|
| `Time` | Timestamp, local time, 30-min resolution | — |
| `Temp. Mandata` | Supply air temperature | °C |
| `Temp. Ripresa` | Return air temperature | °C |
| `Temp. Saturazione` | Coil saturation temperature | °C |
| `Temp. Esterna` | Outdoor air temperature (AHU sensor) | °C |
| `SP Temp. Mand.` | Supply temperature setpoint | °C |
| `SP Temp. Mand. Comp.` | Weather-compensated supply setpoint | °C |
| `Mod. V. Mandata` | Supply fan modulation | % |
| `Mod. V. Ripresa` | Return fan modulation | % |

> Note: figures in the project README and in
> `scripts/5_lstm_anomaly_detection/LSTM_AE_Results_Report.md` are real-data
> results from the confidential Eurix extract. The public sample cannot
> reproduce them. Running the pipeline here reproduces the workflow only.
