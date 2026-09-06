# Interpolation Script - Correct Usage Guide

## Critical Fixes Applied

### ✅ Fix #1: Now Uses Reindexed Data (Regular Time Grid)

**Problem**: Previously interpolated from merged file with irregular timestamps, so missing timestamps weren't visible as NaNs.

**Solution**: Script now:
1. **Prioritizes** the reindexed file from Step 3 (has regular time grid)
2. **Falls back** to merged file and reindexes on-the-fly if Step 3 wasn't run
3. **Converts missing timestamps to NaNs** so they can be properly interpolated

```
Input priority:
1. reindex_for_interp/{dataset}_reindexed_for_interpolation.csv  ← Best
2. merged/{dataset}_merged.csv (reindexed on-the-fly)            ← OK
3. raw/{dataset}.csv (reindexed on-the-fly)                       ← Fallback
```

### ✅ Fix #2: Proper Setpoint & Control Signal Detection

**Problem**: Setpoints were being linearly interpolated (creating artificial ramps).

**Solution**: Enhanced detection now identifies:
- **Setpoints**: "set point", "setpoint", "compensat", "sp " → Forward-fill
- **Control signals**: "modul", "fan", "ventil", "valve", "damper" → Forward-fill  
- **Sensors**: Everything else → Linear/time interpolation

## Paper-Consistent Interpolation Methods

### Sensor Readings (Linear/Time Interpolation)
- **Columns**: Temperatura Ripresa, Mandata, Saturazione, Outdoor Temp/RH
- **Method**: `interpolate(method='time', limit=X)`
- **Fills**: Short gaps only (≤ max_gap_hours)
- **Rationale**: Smooth transitions between measurements

### Setpoints & Controls (Forward-Fill)
- **Columns**: Set Points, Modulation, Fan speeds, Valve positions
- **Method**: `ffill(limit=X)` then `bfill(limit=X)`
- **Fills**: Short gaps only (≤ max_gap_hours)
- **Rationale**: These stay constant until changed by control system

### Long Gaps
- **Rows deleted** from the output dataset — not left as NaN
- Any row where at least one processed column is still NaN after interpolation is dropped
- Controlled by `max_gap_hours` in `config.ini` (default: 4h)

## Output Files

### 1. `{dataset}_interpolated_with_masks.csv`
Contains all data + boolean mask columns:
- `{variable}_is_interpolated` = 1 if point was filled, 0 if original

**Use for LSTM training**: Exclude interpolated points if desired
```python
# Train only on original (non-interpolated) data
mask = df['Supply_Temp_is_interpolated'] == 0
train_data = df[mask]
```

> Note: Long-gap rows have already been **deleted** from this file. No NaN rows for unreliable periods remain.

### 2. `{dataset}_interpolated.csv`
Simplified version without mask columns (just the values)

### 3. `{dataset}_interpolation_summary.json`
Detailed process log including:
- Input/output files
- Sampling interval detected
- Interpolation methods used per column
- Number of points filled per column
- Gap information

## Recommended Workflow

```
Step 1: Merge data
  → scripts/1_data_preprocessing/1.merged_data.py

Step 2: (Optional) quick visual QA on raw data
  → scripts/1_data_preprocessing/2.scatter_plot_and_distributions.py

Step 3: Missing-data analysis + reindex to regular grid (CRITICAL for interpolation!)
  → scripts/1_data_preprocessing/3.missing_data_analysis.py
  → Output: processed_data/reindex_for_interp/<dataset>/<dataset>_reindexed_for_interpolation.csv
  
Step 4: Detect outliers (with boolean flags)
  → scripts/1_data_preprocessing/4.outlier_detection.py
  
Step 5: Interpolate (THIS SCRIPT)
  → scripts/1_data_preprocessing/5.cleaning and interpolation.py
  
Step 6: (Optional) Use the outlier/interpolation masks downstream
  → LSTM pipeline can exclude interpolated / outlier-flagged points during training
  → Long-gap rows are already deleted from the output — no further filtering needed for those
```

## Configuration (`config.ini`)

```ini
[interpolation]
# Maximum gap to fill (in hours)
max_gap_hours = 4

# Columns to process (comma-separated)
columns_to_process = Temperatura Ripresa, Temperatura Mandata, Temperatura Saturazione, 
                     Temperatura Esterna, Umidita Esterna,
                     Set Points Temperatura Mandata, Set Points Temperatura Compensata,
                     Modulazione Ventilatore Mandata, Modulazione Ventilatore Ripresa
```

## Common Issues

### Issue: "total_gaps_found = 0" but I know there are gaps
**Cause**: Using merged file without reindexing
**Fix**: Run Step 3 (reindex) first, or script will reindex on-the-fly

### Issue: Setpoints have artificial ramps
**Cause**: Column name not detected for forward-fill
**Fix**: Check if column contains keywords: "set point", "setpoint", "compensat", "modul", "fan"

### Issue: Too many points interpolated
**Cause**: `max_gap_hours` too large
**Fix**: Reduce `max_gap_hours` in config (typical: 2-4 hours)

### Issue: Not enough points interpolated
**Cause**: `max_gap_hours` too small
**Fix**: Increase `max_gap_hours` (but be careful - only fill "short" gaps)

## Validation Checklist

After running, check the JSON summary:

```json
{
  "sampling_interval_detected": "0 days 00:30:00",  ← Should match your data
  "interpolation_limit_steps": 8,                   ← max_gap_hours / interval
  "interpolation_methods": {
    "linear_time": ["Temperatura Ripresa", ...],    ← Sensors here
    "forward_fill": ["Set Points...", "Modul..."]   ← Setpoints/controls here
  },
  "points_filled_per_column": {                     ← Should be reasonable
    "Temperatura Ripresa": 38,
    "Set Points...": 12
  }
}
```

**Red flags**:
- `forward_fill: []` → Setpoint detection failed
- `points_filled = 0` for all → Reindexing issue
- `points_filled > 10% of data` → Gap threshold too high

## Paper Justification

This approach follows standard practices in building energy research:

1. **Regular sampling**: Ensures consistent temporal resolution
2. **Short gap filling**: Preserves data quality (only fills ≤2-4h)
3. **Method selection**: Appropriate for signal type (sensor vs control)
4. **Transparency**: Masks track which points were filled
5. **Long gap deletion**: Rows with gaps ≥ max_gap_hours are dropped — avoids artificial data and ensures a clean, NaN-free output dataset

References commonly use similar approaches for AHU/chiller fault detection and LSTM modeling.

---

**Key takeaway**: Always run the reindexing script (Step 3) before interpolation, or accept the on-the-fly reindex. This ensures missing timestamps become visible as NaNs.
