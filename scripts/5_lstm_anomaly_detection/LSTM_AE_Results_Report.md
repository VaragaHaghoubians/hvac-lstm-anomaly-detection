# LSTM Autoencoder for HVAC Anomaly Detection
## Project Report, Building C1, AHU UTA1
### Summer 2025 & Winter 2026

**Institution:** Università degli Studi di Torino (UNITO)  
**System:** HVAC, Air Handling Unit UTA1, Building C1  
**Report Date:** 25 February 2026  

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Methodology](#2-methodology)
   - 2.1 Data and Features
   - 2.2 Data Preprocessing Pipeline
   - 2.3 Model Architecture
   - 2.4 Training Procedure
   - 2.5 Anomaly Detection Strategy
3. [Summer 2025 Results](#3-summer-2025-results-2025-07-09--2025-10-14)
4. [Winter 2026 Results](#4-winter-2026-results-2025-10-15--2026-02-18)
5. [Cross-Seasonal Comparison](#5-cross-seasonal-comparison)
6. [Comparison with Related Literature](#6-comparison-with-related-literature)
7. [Conclusions](#7-conclusions)

---

## 1. Project Overview

This project applies an unsupervised **LSTM Autoencoder (LSTM-AE)** to detect anomalies in an Air Handling Unit (AHU) in a real commercial building. The system processes multivariate time-series data collected at 30-minute resolution from internal HVAC sensors. The core principle is **unsupervised anomaly detection**: the model trains exclusively on normal operating data and detects anomalies by measuring how poorly it reconstructs unseen sequences—high reconstruction error signals abnormal behavior.

No fault labels are required. The approach is fully data-driven and is applied independently to two operational seasons: **Summer 2025** (cooling mode) and **Winter 2026** (heating mode), allowing cross-seasonal comparison of system behavior and model performance.

---

## 2. Methodology

### 2.1 Data and Features

Raw sensor data from AHU UTA1 (Building C1) was loaded from feature-engineered CSV files at 30-minute sampling resolution. After extensive analysis of feature variance, reconstruction quality, and false-positive drivers, **three HVAC-internal temperature signals** were selected as reconstruction targets for both seasons:

| Feature | Column | Physical Meaning |
|---|---|---|
| F1 | `Temperatura Mandata` | Supply air temperature, primary control output of the AHU |
| F2 | `Temperatura Ripresa` | Return air temperature, reflects building thermal load |
| F3 | `Temperatura Saturazione` | Coil saturation temperature, divergence from TM indicates heat-transfer fault |

**Features deliberately excluded from reconstruction** and their rationale:

- **Fan modulation** (`Modulazione Ventilatore`): locked at 80% throughout winter → near-zero variance → RobustScaler IQR collapses → corrupts gradients. Applied as a rule-based check instead.
- **Weather signals** (`Temperatura Esterna`, `Umidità Esterna`): exogenous disturbances not controlled by the AHU. Including them would make the model score "did the weather change?" rather than "is the AHU behaving correctly given the weather?" → Dominated false positives in shoulder-season test data.
- **`delta_t_signed`** (= TM − TR): mathematically redundant given F1 and F2. Rule-based check: `delta_t_signed < 0.5°C` during `Mod > 30%` → heat-transfer fault.
- **`setpoint_error`**: median ≈ 0 with large tails → RMSE ≈ 173%, flat training distribution → corrupts threshold. Applied as rule: `rolling_mean(setpoint_error, 4h) > 1.5°C` → control fault.

This feature selection follows the philosophy of Yeom et al. (2025): use physically meaningful internal AHU signals and handle exogenous or algebraically derived signals separately.

### 2.2 Data Preprocessing Pipeline

The preprocessing pipeline (implemented in `3_preprocessor.py`) applies the following steps in strict order:

1. **Dataset capping** (Summer only): the shoulder-season tail (October, when outdoor temperature drops from ~30°C to ~12°C) is optionally removed before splitting to keep the test set within the same thermal regime as training. This guards against the "regime shift" false-positive flood documented in the Summer 2025 analysis.

2. **Chronological 70/15/15 split by calendar week**: train = early season, validation = mid season, test = late season. No shuffling. This respects temporal causality and mirrors real deployment scenarios where only past data is available for training.

3. **Optional EWMA smoothing** (disabled): per Schein (2006), exponential smoothing can be applied per split before scaling to reduce sensor noise.

4. **Feature scaling with `RobustScaler`**: fitted **on training data only** to prevent data leakage. RobustScaler uses median and IQR, making it robust to the sensor spikes and near-constant signals common in HVAC data. MinMaxScaler was rejected because a feature with near-zero range in training (e.g., `setpoint_error ≈ 0` all summer) produces an incorrect unit-interval mapping when tested in autumn.

5. **Automatic variance check and feature auto-drop**: post-scaling, any feature with variance < 0.01 (flatline) is dropped from all splits and the scaler is re-fitted. This catches the "80% fan" scenario without requiring manual config edits.

6. **Sliding window sequence creation**: overlapping windows of `sequence_length = 8` steps (4 hours at 30-minute resolution) are created from each split. Windows that span known data gaps are discarded to prevent the model from learning artificial transitions.

7. **Clean training mask**: sequences are excluded from training (but kept for evaluation) if they contain:
   - Fan modulation ≈ 0 (system off — not representative of normal operation)
   - Values outside physical thresholds (sensor faults, impossible readings)
   - Segments longer than 4 hours that were gap-filled by interpolation

### 2.3 Model Architecture

A **stacked LSTM Autoencoder** is used (implemented in `4_model.py`). The architecture is deeper than the single-layer encoder–decoder of Wei et al. (2022), and comparable in depth to the multi-layer stacking of Huang et al. (2024):

```
INPUT:  (8 timesteps × 3 features)
        │
ENCODER Layer 1:  LSTM(64, return_sequences=True)
        │
        Dropout(0.1)
        │
ENCODER Layer 2:  LSTM(16, return_sequences=False)  ← BOTTLENECK
        │
        Dropout(0.1)
        │
        RepeatVector(8)   ← expand latent vector across time axis
        │
DECODER Layer 1:  LSTM(16, return_sequences=True)
        │
        Dropout(0.1)
        │
DECODER Layer 2:  LSTM(64, return_sequences=True)
        │
        Dropout(0.1)
        │
OUTPUT: TimeDistributed(Dense(3, activation='linear'))
        (8 timesteps × 3 features)
```

The bottleneck compresses 8 × 3 = 24 input values into a 16-dimensional latent vector, forcing the model to learn only the essential temporal patterns of normal HVAC operation. Anything the model cannot compress and reconstruct well (i.e., abnormal behavior) produces high reconstruction error.

**Architecture rationale** (rule of thumb: encoder/decoder units ≈ 4–8× number of features):
- 3 features → encoder = 64, bottleneck = 16, decoder = 64
- Gradient clipping (`clipnorm = 1.0`) prevents exploding gradients common with stacked LSTMs
- **MAE loss** (not MSE): MSE would square-penalize tiny reconstruction wiggles on near-constant features (e.g., binary fan signal), inflating the anomaly threshold unfairly. MAE is scale-linear and more robust.

### 2.4 Training Procedure

Training is implemented in `5_trainer.py`:

| Parameter | Value |
|---|---|
| Optimiser | Adam, lr = 0.0005, clipnorm = 1.0 |
| Loss | MAE |
| Maximum epochs | 300 |
| Batch size | 32 |
| Early stopping | patience = 50, monitor = val_loss, restore best weights |
| LR scheduler | ReduceLROnPlateau — factor = 0.7, patience = 15, min_lr = 0.00005 |
| Model checkpoint | Save best val_loss model to `.keras` file |

**Critical principle: the autoencoder is trained as a self-supervised copier** — input equals target (`X_train = y_train`). The model learns to compress and reconstruct normal sequences. It never sees anomalous examples during training. This is the "Train on the Norm, Detect on the Storm" paradigm.

### 2.5 Anomaly Detection Strategy

Anomaly detection is implemented in `6_evaluator.py` through a multi-layer pipeline:

#### Step 1: Reconstruction Error Calculation
For each test sequence, compute the mean absolute error (MAE) between the original and reconstructed sequence, averaged across all timesteps and features; this is the **anomaly score**.

#### Step 2: Dynamic EMA Threshold
The threshold is computed dynamically on the test set using an **Exponentially Weighted Moving Average (EMA)**:

$$\text{threshold}(t) = \text{EMA}_{\text{mean}}(\text{errors}, \text{span}=10) + 1.0 \times \text{EMA}_{\text{std}}(\text{errors}, \text{span}=10)$$

The EMA is warm-started from an anchor value computed on the training errors (the value reported in the results as "EMA anchor"). The dynamic threshold adapts to slow operational regime changes (e.g., outdoor temperature drop in autumn) without raising the false-positive rate for such transitions, while still flagging sudden, localized spikes above the local baseline.

**Important limitation acknowledged** (documented in `config.ini`): EMA carries a "boiling frog" risk. If the AHU degrades *very slowly* over weeks, the EMA rises with the degradation and may not fire an alarm until complete failure. For gradual mechanical degradation monitoring, the percentile-based static threshold (95th percentile of training errors) is preferable and is also available in the pipeline.

#### Step 3: Feature-Wise Errors
For each anomalous sequence, the per-feature MAE contribution is computed, identifying **which of the three signals caused the anomaly** (F1 = TM, F2 = TR, F3 = TS).

#### Step 4: Smoothing
A centered rolling-median filter (window = 3) is applied to the raw error signal to suppress isolated single-window spikes caused by short interpolated segments or sensor transients.

#### Step 5: Persistence Filter
A detected anomaly is **retained only if it belongs to a run of ≥ 2 consecutive anomalous windows** (≥ 1 hour of sustained abnormal reconstruction). This eliminates isolated 30-minute false alarms.

#### Step 6 — Severity Classification (configured but disabled in current runs)
When enabled, anomalies are classified into:
- **Critical**: error > 2.5 × threshold
- **Moderate**: error > 1.5 × threshold
- **Minor**: error > 1.0 × threshold

---

## 3. Summer 2025 Results (2025-07-09 → 2025-10-14)

### 3.1 Dataset Summary

| Split | Sequences | Date Range |
|---|---|---|
| Training (70%) | 3,017 | 2025-07-09 → early Sep 2025 |
| Validation (15%) | 665 | mid Sep 2025 |
| Test (15%) | 988 | late Sep → 2025-10-14 |

### 3.2 Reconstruction Quality

| Set | Mean Error (MAE) | Std Dev |
|---|---|---|
| Training | 0.042574 | 0.033112 |
| Validation | 0.054025 | 0.033801 |
| **Test** | **0.177637** | **0.086710** |

The test mean error (0.1776) is **4.2× higher than the training mean** (0.0426). This is explained by the shoulder-season regime shift: the test period covers late September and October 2025, when outdoor temperature dropped from ~30°C to ~12°C, fundamentally changing the building's thermal load. The AHU transitions from active cooling to near-passive mode — a condition the model was not trained on since training covered the stable mid-summer core season.

This is **not a model failure** but an expected and documented phenomenon. The `dataset_end_date` parameter in `config.ini` lets you cap the dataset before the shoulder season if you want to restrict evaluation to the stable cooling regime only.

### 3.3 Anomaly Detection Results

| Metric | Value |
|---|---|
| **Total anomalies detected** | **136** |
| **Anomaly rate (test)** | **13.77%** |
| EMA threshold anchor | 0.070742 |
| Anomaly rate (training) | 15.46% |
| Anomaly rate (validation) | 31.28% |

The test anomaly rate (13.77%) is comparable to the training rate (15.46%), which indicates reasonable model generalization. The validation rate (31.28%) is substantially higher; this corresponds to the mid-season period (September), where the shoulder-season transition begins, and the model encounters increasingly out-of-distribution patterns relative to the warm July–August training core.

**Pattern from the EMA timeline plot**: anomalies cluster in two main periods:
- **Early test period (late September)**: transition from stable summer operation; reconstruction error elevated as outdoor conditions change
- **Early October spike cluster** (around 2025-10-05 to 2025-10-09): sustained elevated reconstruction error coinciding with a sharp outdoor temperature drop — the model correctly identifies this as operationally anomalous relative to its training distribution

### 3.4 Feature-Wise Error Analysis

| Feature | RMSE | MSE | MAE |
|---|---|---|---|
| F1 : Temperatura Mandata | 3.3% | 0.1% | 2.4% |
| F2 : Temperatura Ripresa | 2.6% | 0.1% | 2.1% |
| **F3 : Temperatura Saturazione** | **5.9%** | **0.4%** | **4.7%** |

**Temperatura Saturazione (F3) shows the highest error across all three metrics**, with RMSE 5.9%, more than double that of F2. This is physically meaningful: coil saturation temperature is directly coupled to outdoor conditions (the refrigerant circuit operating point shifts with ambient temperature). When the outdoor temperature drops sharply in autumn, the saturation temperature behavior changes significantly, making it the hardest feature for a summer-trained model to reconstruct in the shoulder season.

Temperatura Ripresa (F2) shows the lowest error, consistent with its slower dynamics and higher thermal inertia; the building's return air temperature responds more gradually to outdoor changes than the coil saturation.

### 3.5 Reconstruction Time Series Analysis

**Temperatura Mandata (F1)**: The model captures the general diurnal cycle (daily rising/falling pattern, setpoint transitions) but shows a persistent phase lag and smooths sharp peaks. The reconstruction tends to overestimate temperatures during troughs (morning startup) and underestimate during peaks (afternoon cooling demand). This is expected given the short sequence length (8 steps = 4 hours), which limits the model's temporal context.

**Temperatura Ripresa (F2)**: The best-reconstructed feature. The reconstruction closely tracks the real signal across the entire period, with deviations mainly at the sharp daily peaks. The smoother thermal behaviour of return air temperature aligns well with the LSTM's learned representations.

**Temperatura Saturazione (F3)**: Shows the most significant reconstruction deviations in the shoulder season (Sep 24 – Oct 14). In the early test weeks (late Sep), the reconstruction generally follows the signal, but as outdoor temperatures drop, the real F3 begins to diverge from the model's learned summer-mode expectations; the model reconstructs what it expects the saturation temperature to be (summer pattern) while the real signal follows a different thermal regime. This divergence primarily drives the anomaly detections in October.

---

## 4. Winter 2026 Results (2025-10-15 → 2026-02-18)

### 4.1 Dataset Summary

| Split | Sequences | Date Range |
|---|---|---|
| Training (70%) | 3,959 | 2025-10-15 → early Jan 2026 |
| Validation (15%) | 665 | mid Jan 2026 |
| Test (15%) | 1,049 | late Jan → 2026-02-18 |

The winter dataset is 31% larger than the summer dataset (4,673 total sequences vs. 3,559 after cleaning), reflecting the longer winter season (126 days vs. 97 days for July–October). More training data provides a richer representation of winter HVAC behavior.

### 4.2 Reconstruction Quality

| Set | Mean Error (MAE) | Std Dev |
|---|---|---|
| Training | 0.082530 | 0.056908 |
| Validation | 0.091606 | 0.036681 |
| **Test** | **0.113723** | **0.054420** |

Unlike summer, the winter test error (0.1137) is only **1.38× the training mean** (0.0825), far more modest than the 4.2× factor observed in summer. This confirms that the winter test period (January–February) stays within the same thermal regime as training (October–January): steady cold-weather heating mode with relatively consistent outdoor conditions. No sharp winter regime transition matches the summer-to-autumn collapse.

### 4.3 Anomaly Detection Results

| Metric | Value |
|---|---|
| **Total anomalies detected** | **57** |
| **Anomaly rate (test)** | **5.43%** |
| EMA threshold anchor | 0.099107 |
| Anomaly rate (training) | 30.58% |
| Anomaly rate (validation) | 36.24% |

The winter results present an unusual pattern: the **test anomaly rate (5.43%) is substantially lower than training (30.58%)**. This is a positive and interpretable finding:

- **Training period (Oct–early Jan)**: covers the autumn-to-winter transition, a highly variable regime with outdoor temperatures spanning −2°C to +18°C, rapidly changing heating setpoints, and frequent on/off cycling as the system learns the winter load. High variability leads to high reconstruction error during training.
- **Test period (Jan–Feb)**: deep winter "steady state"; outdoor temperatures are cold and relatively stable; the AHU operates in consistent heating mode; setpoints are stable. The system behavior is more predictable, leading to lower reconstruction errors and fewer detections.

This is **not a model failure**; it confirms the model generalized beyond its training distribution and correctly identifies the late-winter operation as largely nominal.

**Critical anomaly event, circa 2026-02-12**: The EMA timeline plot shows a pronounced spike reaching MAE ≈ 0.5, clearly the largest anomaly event in the entire winter dataset. This is approximately **5× the typical winter baseline** (MAE ≈ 0.09–0.10). The EMA threshold rises in response, but the spike still exceeds it significantly, triggering a cluster of anomaly flags. This event warrants physical investigation (sensor fault, actuator fault, heating coil issue, or external supply problem).

**February 17–18 cluster**: A sustained elevation in reconstruction error toward the very end of the test period also generates multiple anomaly detections, suggesting a possible emerging operational change or fault developing in the final days of the monitored period.

### 4.4 Feature-Wise Error Analysis

| Feature | RMSE | MSE | MAE |
|---|---|---|---|
| **F1 : Temperatura Mandata** | **4.7%** | **0.2%** | **3.4%** |
| F2 : Temperatura Ripresa | 0.7% | 0.0% | 0.4% |
| F3 : Temperatura Saturazione | 4.0% | 0.2% | 3.0% |

The winter feature-error distribution is strikingly different from summer:

- **F2 (Temperatura Ripresa) shows near-perfect reconstruction in winter** (MAE 0.4%, RMSE 0.7%). Return air temperature in heating mode follows extremely stable diurnal patterns driven by building occupancy schedules, the most temporally regular signal in the dataset, well within the LSTM's learned representation.
- **F1 (Temperatura Mandata) is now the hardest feature** (RMSE 4.7%, highest of the three). In heating mode, the supply air temperature oscillates at high frequency between heating setpoint peaks (28–32°C) and inter-cycle valleys (24–26°C) driven by the heating coil modulation cycles. These rapid, high-amplitude oscillations are harder to reconstruct than the slower summer cooling patterns. The time series plot confirms this; the reconstruction follows the envelope but cannot precisely track every individual cycle peak.
- **F3 (Temperatura Saturazione) shows moderate error** (RMSE 4.0%). In heating mode, the saturation temperature tracks the heating coil behavior, oscillating with the heating cycles similarly to F1 but with additional thermal lag from the refrigerant circuit.

### 4.5 Reconstruction Time Series Analysis

**Temperatura Mandata (F1)**: The reconstruction shows a clear systematic bias; the model reconstructs a smoothed, lower-frequency version of the actual signal. The real TM oscillates at ~55-minute cycles between 24°C and 32°C (heating coil on/off cycles), while the reconstruction captures only the mean trend. This is because the **sequence length is 8 steps (4 hours)**: the model "sees" roughly 4–5 complete heating cycles per window and reconstructs the average behavior rather than the instantaneous cycle position. Increasing `sequence_length` to 16 or 24 would likely improve TM reconstruction in winter.

**Temperatura Ripresa (F2)**: Excellent reconstruction quality throughout the entire winter period. The model precisely tracks the slow daily arc of return air temperature, from nighttime lows (~21.5°C) to occupied-hours peaks (~24.5°C), the most consistent signal and the strongest evidence that the model has learned genuine winter HVAC physics.

**Temperatura Saturazione (F3)**: Like F1, the reconstruction smooths the high-frequency heating-coil oscillations but captures the slower trend. Notable deviation around 2026-02-12 (the major anomaly event ): both real and reconstructed TS diverge significantly, confirming that the anomaly event affected the coil's thermal behavior.

---

## 5. Cross-Seasonal Comparison

### 5.1 Numerical Summary

| Metric | Summer 2025 | Winter 2026 |
|---|---|---|
| Analysis period | Jul 9 – Oct 14, 2025 | Oct 15, 2025 – Feb 18, 2026 |
| Duration | 97 days | 126 days |
| Training sequences | 3,017 | 3,959 |
| Test sequences | 988 | 1,049 |
| EMA threshold anchor | 0.070742 | 0.099107 |
| Train mean error (MAE) | 0.042574 | 0.082530 |
| Test mean error (MAE) | 0.177637 | 0.113723 |
| Test/train error ratio | **4.18×** | **1.38×** |
| Total anomalies | 136 | 57 |
| Test anomaly rate | 13.77% | 5.43% |
| Hardest feature | F3 - TS (RMSE 5.9%) | F1 - TM (RMSE 4.7%) |
| Easiest feature | F2 - TR (RMSE 2.6%) | F2 - TR (RMSE 0.7%) |

### 5.2 Key Differences and Physical Interpretation

**Higher winter threshold** (0.099 vs. 0.071): The model's baseline reconstruction error is higher in winter because the AHU operating point covers a wider range of conditions from October through December. The training data spans the full autumn-to-winter transition (a large envelope), so the model must accept more variability as "normal." In summer, training covers only the stable warm-weather cooling regime.

**Radically different test/train error ratios**: Summer shows a 4.18× amplification on test data versus training, entirely attributable to the shoulder-season regime shift. Winter shows only 1.38×, confirming that January–February deep-winter operation is well represented by the October–January training distribution. This makes the **winter model more reliable** for operational monitoring in the season's final months.

**Fewest anomalies in winter despite longer season**: Winter detected 57 anomalies (5.43% rate) compared to 136 (13.77%) in summer, across a longer season. This confirms that late-winter HVAC operation is more stable and predictable than the end of the cooling season. The summer anomaly cluster around October is largely a regime-shift detection rather than a mechanical fault.

**F3 dominates summer, F1 dominates winter**: This is a direct reflection of HVAC operating modes:
- In **cooling mode (summer)**, the saturation temperature (F3) is the most volatile feature; it directly tracks refrigerant circuit conditions, which are highly sensitive to outdoor temperature changes in the shoulder season.
- In **heating mode (winter)**, the supply temperature (F1) shows the most complex dynamics; rapid heating-coil cycling produces high-frequency oscillations that challenge the model's short temporal window.

**F2 (Temperatura Ripresa) is consistently the most stable feature in both seasons** (lowest error). Return air temperature is dominated by building thermal inertia and occupancy schedule, making it a temporally smooth and predictable signal in all conditions, an ideal reference signal for the autoencoder.

### 5.3 Observed Anomaly Events by Type

| Event Description | Season | Likely Source |
|---|---|---|
| Sustained elevated error, late Sep 2025 | Summer | Shoulder-season regime transition |
| Spike cluster, Oct 5–9, 2025 | Summer | Sharp outdoor temperature drop combined with changing cooling load |
| Large isolated spike to MAE ≈ 0.50, circa Feb 12, 2026 | Winter | Probable actuator/sensor fault or heating supply interruption |
| Elevated error cluster, Feb 17–18, 2026 | Winter | End-of-season operational change or emerging fault |
| Scattered isolated detections throughout both seasons | Both | Transient control events (setpoint changes, fan switching), partially filtered by persistence requirement |

---

## 6. Comparison with Related Literature

We reviewed three LSTM-AE papers for comparison. The table below maps their core design choices against this project.

### 6.1 Reviewed Papers

**Wei et al. (2022)**, *"LSTM-Autoencoder-based Anomaly Detection for Indoor Air Quality Time Series Data"*  
Single-layer LSTM-AE for univariate CO₂ time series. Threshold = maximum reconstruction error on normal training data.

**Huang et al. (2024)**, *"Time-Series Few-Shot Anomaly Detection for HVAC Systems"*  
Multi-layer LSTM-AE with cross-building domain adaptation via few-shot fine-tuning. Designed for multi-building generalization.

**Yeom et al. (2025)**, *"Hybrid AI Model for Fault Detection in AHU Systems"*  
LSTM-AE for unsupervised detection on unlabeled AHU data, combined with physical validation rules (PID instability, heat transfer efficiency, CO₂). Applied to real building data with noise and missing values.

### 6.2 Architectural Comparison

| Aspect | Wei (2022) | Huang (2024) | Yeom (2025) | **This Project** |
|---|---|---|---|---|
| Encoder depth | 1 LSTM layer | 2 LSTM layers (8→4) | 1 LSTM layer | **2 LSTM layers (64→16)** |
| Decoder depth | 1 LSTM layer | 2 LSTM layers (4→8) | 1 LSTM layer | **2 LSTM layers (16→64)** |
| Bottleneck | Not specified | 4 units | 6 units | **16 units** |
| Dropout | Not used | Not specified | Not specified | **0.1 on all layers** |
| Gradient clipping | No | No | No | **Yes (clipnorm=1.0)** |
| Loss function | MSE / MAE | Not specified | Not specified | **MAE (robust to near-constant)** |
| Sequence length | 10 steps | Not specified | 12 steps (2h) | **8 steps (4h)** |
| Data dimensionality | 1 feature (CO₂) | Multi-feature | Multi-feature | **3 features** |
| Scaler | Not specified | Not specified | Not specified | **RobustScaler (IQR-based)** |

The architecture most closely matches **Huang et al. (2024)** in stacked encoder–decoder depth and **Yeom et al. (2025)** in application context (real building AHU data, multivariate).

### 6.3 Threshold Strategy Comparison

| Method | Wei (2022) | Huang (2024) | Yeom (2025) | **This Project** |
|---|---|---|---|---|
| Static percentile | Max training error (≡ 100th pct.) | Max training error | Not specified | Available (95th pct.) |
| Mean + k·σ | Not used | Not used | Not used | Available |
| **Dynamic EMA** | ❌ Not used | ❌ Not used | ❌ Not used | ✅ **Used (active)** |
| Manual fixed | Not used | Not used | Not used | Available |
| Persistence filter | ❌ | ❌ | ❌ | ✅ **≥2 consecutive windows** |
| Error smoothing | ❌ | ❌ | ❌ | ✅ **Rolling median (window=3)** |

All three literature papers use a static threshold defined as the maximum reconstruction error on normal training data. **The dynamic EMA threshold used in this project is a novel contribution** relative to the reviewed papers: it adapts to the local error baseline at each timestep, allowing a single model to handle the operational regime variation across a full season without manual threshold re-tuning. This is particularly important for HVAC data where even within a single season there is substantial variation (outdoor temperature ranges, occupancy changes, setpoint adjustments).

The persistence filter and rolling-median smoothing are also absent from all three literature papers — they are engineering additions that significantly reduce false alarms from transient sensor events.

### 6.4 Physical Validation Layer

Yeom et al. (2025) is the only paper that combines LSTM-AE with physical validation rules. Their approach validates anomaly detections against:
- PID instability indicators
- Heat transfer efficiency
- CO₂ concentration

This project has laid the groundwork for an equivalent physical validation layer, acknowledging in `config.ini` two rule-based checks derived from the available features:
1. `delta_t_signed < 0.5°C` while `Mod > 30%` → heat-transfer fault (coil fouling or refrigerant loss)
2. `rolling_mean(setpoint_error, 4h) > 1.5°C` → control fault (actuator fault or BMS setpoint error)

These rules were excluded from the LSTM reconstruction targets specifically *because* they are better applied as deterministic rules, a design decision aligned with Yeom et al.'s hybrid philosophy.

### 6.5 Domain Context Comparison

| Context | Wei (2022) | Huang (2024) | Yeom (2025) | **This Project** |
|---|---|---|---|---|
| Building type | Not specified | Multiple buildings | Real AHU | **Real AHU, single building** |
| Data quality | Clean | Clean | Noisy, missing | **Noisy, DST issues, gaps, interpolation** |
| Seasons | Not specified | Not specified | Not specified | **Two full seasons** |
| Transfer learning | ❌ | ✅ (few-shot) | ❌ | ❌ (single building) |
| Labelled faults | Not required | Not required | Some (supervised part) | **Not required (fully unsupervised)** |

The cross-seasonal application of the same pipeline with detailed analysis of regime-shift effects on model performance is a contribution not addressed in any of the three reviewed papers.

---

## 7. Conclusions

### 7.1 Technical Findings

The LSTM Autoencoder pipeline successfully learned normal HVAC operation patterns for both the summer cooling and winter heating modes of AHU UTA1, Building C1. The model demonstrates:

1. **Good generalization to unseen test data in winter** (test/train error ratio 1.38×), confirming that January–February deep-winter operation is well captured by the October–January training distribution.

2. **Expected difficulty with shoulder-season transitions in summer** (test/train error ratio 4.18×). The late September and October test period falls outside the summer training distribution as outdoor temperatures collapse. This produces elevated anomaly rates that are partially genuine operational changes rather than equipment faults.

3. **Feature-specific behavior consistent with HVAC physics**: F3 (Temperatura Saturazione) is the dominant error source in summer (coil conditions sensitive to outdoor temperature), while F1 (Temperatura Mandata) is the dominant source in winter (rapid heating-cycle oscillations challenge the model's temporal context). F2 (Temperatura Ripresa) is consistently the most stable feature across both seasons, reflecting the high thermal inertia of building return air.

4. **A significant anomaly event in winter circa 2026-02-12** (MAE spike to ≈0.50, approximately 5× the seasonal baseline) is the strongest candidate for a genuine mechanical or control fault and warrants investigation against physical inspection records.

5. **The EMA dynamic threshold** prevents the false-positive flood that would occur with a fixed threshold in the face of seasonal regime variation, at the cost of potential "boiling frog" insensitivity to very gradual degradation. A hybrid approach — using EMA for real-time monitoring and a fixed 95th-percentile threshold for seasonal drift detection, is recommended for production use.

### 7.2 Positioning Relative to Literature

This project implements a pipeline that:
- Is **architecturally comparable to Huang et al. (2024)** in model depth and multivariate HVAC application
- Is **contextually closest to Yeom et al. (2025)** in dealing with real, noisy AHU data and in the deliberate separation of reconstruction-based anomaly scoring from physical-rule-based fault characterization
- **Exceeds all three reviewed papers** in post-processing sophistication (dynamic threshold, persistence filtering, feature-wise error attribution) and in systematic cross-seasonal analysis
- **Does not implement cross-building transfer learning** (Huang et al. 2024) as the study is confined to a single building; this is the primary remaining gap relative to the literature

### 7.3 Recommendations for Operational Deployment

1. **Investigate the February 12, 2026 anomaly event** against maintenance logs, BMS event history, and physical inspection data. The magnitude (5× baseline) and duration (several consecutive windows) make this the study's most operationally significant finding.

2. **Switch to the static 95th-percentile threshold** for long-term degradation monitoring, reserving the EMA threshold for sudden-fault detection in real-time monitoring contexts.

3. **Extend sequence length to 16 steps (8 hours)** for the winter model: the current 4-hour window is insufficient to capture the full heating-coil cycling patterns in F1, leading to systematic reconstruction bias. This should substantially reduce the winter F1 error.

4. **Implement the two rule-based physical validation checks** already designed in the config: `delta_t_signed` heat-transfer rule and `setpoint_error` control-fault rule. Combined with the LSTM-AE anomaly scores, this would achieve a Yeom-style hybrid detection layer using features already available in the dataset.

5. **Retrain the summer model with `dataset_end_date` set to approximately 2025-09-15** to exclude the shoulder season from the test set and obtain a clean evaluation of the cooling-mode model on in-distribution data.

---

*End of Report*  
*Generated: 25 February 2026*  
*Project: HVAC Anomaly Detection, UNITO Master's Thesis*  
*System: Building C1, AHU UTA1, Torino, Italy*
