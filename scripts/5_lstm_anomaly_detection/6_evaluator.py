"""
6_evaluator.py — Anomaly Scoring, Decision Engine, and Post-Processing
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Role in the pipeline
--------------------
This module is the **decision and post-processing core** of the system.
It is not model evaluation in the classical supervised sense (no ground-truth
labels, no accuracy/F1 computation). Instead it performs the full chain of
steps that turn raw reconstruction errors into validated, semantically labelled
fault events:

  1. Reconstruction-error scoring
       Compute per-window MAE for train / val / test from the autoencoder output.

  2. Error smoothing
       Apply rolling-median smoothing to test errors before thresholding, to
       reduce spike sensitivity.

  3. Adaptive EMA thresholding
       Build a causal, per-window threshold:  threshold(t) = EMA_mean + k × EMA_std
       Anchored to the training-set error distribution; resets at data gaps to
       prevent holiday/shutdown artifacts from falsely elevating the baseline.

  4. Candidate screening (two-layer filter)
       A lenient first pass checks feature-level agreement (≥ N features above
       their individual thresholds) and short persistence (≥ M consecutive
       windows), producing a candidate anomaly mask.

  5. Transient / TR-guard suppression
       Removes candidate windows that coincide with steep supply-temperature
       (TM) ramps where the return temperature (TR) is stable — these are
       predictable morning startup transients, not faults.

  6. Semantic and physics-based fault filtering
       Classifies each candidate as:
         • "Operational peak"  — recurring daily pattern (periodic), no physics violation
         • "Potential fault"   — non-periodic deviation, or physics-violating event
       In strict season mode (currently Winter) only non-periodic windows survive.

  7. Persistence filter (final gate)
       Requires N consecutive anomalous windows before issuing a confirmed flag.

  8. Event construction
       Merges consecutive flagged windows into discrete fault events, assigns
       event IDs, computes duration and peak error per event.

CRITICAL PRINCIPLE — "Train on the Norm, Detect on the Storm"
-------------------------------------------------------------
  • The EMA threshold anchor is derived from TRAINING errors (normal behaviour).
  • It is applied to TEST data only — zero look-ahead leakage.

NOTE ON EWMA vs EMA
-------------------
  The pipeline uses exponential weighting in two distinct, separate places:
    • Input EWMA smoothing (3_preprocessor.py): smooths raw sensor channels
      *before* scaling and training, to reduce measurement noise.
    • EMA adaptive thresholding (this module): adapts the anomaly-detection
      threshold to the rolling error level *after* reconstruction.
  These are independent mechanisms with different spans and purposes.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd


def calculate_reconstruction_error(model, X, metric='mse'):
    """
    Calculate reconstruction error for sequences
    
    Args:
        model: Trained autoencoder
        X: Input sequences (n_samples, seq_len, n_features)
        metric: 'mse', 'rmse', or 'mae'
        
    Returns:
        errors: Array of reconstruction errors (one per sample)
        X_reconstructed: Reconstructed sequences
    """
    
    # Reconstruct data
    X_reconstructed = model.predict(X, verbose=0)
    
    # Calculate error per sample
    errors = []
    for i in range(len(X)):
        if metric == 'mse':
            error = mean_squared_error(X[i], X_reconstructed[i])
        elif metric == 'rmse':
            error = np.sqrt(mean_squared_error(X[i], X_reconstructed[i]))
        elif metric == 'mae':
            error = mean_absolute_error(X[i], X_reconstructed[i])
        else:
            raise ValueError(f"Unknown metric: {metric}")
        errors.append(error)
    
    return np.array(errors), X_reconstructed


def calculate_feature_wise_errors(X_original, X_reconstructed):
    """
    Calculate reconstruction error per feature (for severity classification)
    
    Args:
        X_original: Original sequences (n_samples, seq_len, n_features)
        X_reconstructed: Reconstructed sequences
        
    Returns:
        feature_errors: (n_samples, n_features) - MAE per feature per sample
    """
    
    n_samples, seq_len, n_features = X_original.shape
    feature_errors = np.zeros((n_samples, n_features))
    
    for i in range(n_samples):
        for j in range(n_features):
            # MAE for feature j in sample i (consistent with training loss)
            feature_errors[i, j] = mean_absolute_error(
                X_original[i, :, j], 
                X_reconstructed[i, :, j]
            )
    
    return feature_errors


def calculate_threshold(errors, config):
    """
    Calculate anomaly threshold from training reconstruction errors
    
     CRITICAL INTERPRETATION: Expected False Positive Rate
    
    Autoencoders threshold normal variability vs. anomalies.
    
    Percentile Method (most common):
    
     99.5% percentile  ~0.5% of NORMAL data will exceed threshold
     99.0% percentile  ~1.0% of NORMAL data will exceed threshold
     99.9% percentile  ~0.1% of NORMAL data will exceed threshold
    
    This is BY DESIGN, not a bug:
    - Threshold captures "typical" reconstruction error
    - Rare normal patterns may exceed it (false positives)
    - Trade-off: Tighter threshold = more sensitivity BUT more false alarms
    
    For thesis/publication:
    
    "We expect ~{100 - percentile}% of normal samples to exceed the threshold
    by definition. This controlled false positive rate is inherent to the
    quantile-based thresholding approach and represents the model's sensitivity
    to rare but normal operational patterns."
    
    Methods:
        - 'percentile': Direct quantile control (e.g., top 1%)
        - 'std_dev': Statistical approach (mean + k*sigma)
        - 'manual': Fixed threshold value
    
    Args:
        errors: Training reconstruction errors
        config: ConfigParser object with ANOMALY section
        
    Returns:
        threshold: Anomaly detection threshold value
        threshold_info: Dict with interpretation details
    """
    
    threshold_mode = config.get('lstm_autoencoder', 'threshold_mode', fallback='percentile')
    threshold_info = {'mode': threshold_mode}
    
    if threshold_mode == 'percentile':
        # Use percentile of training errors (quantile method)
        percentile = config.getfloat('lstm_autoencoder', 'anomaly_threshold_percentile', fallback=99.5)
        threshold = np.percentile(errors, percentile)
        
        expected_fp_rate = 100 - percentile
        
        print(f"    THRESHOLD CALCULATION (Percentile/Quantile Method):")
        print(f"       Percentile: {percentile}%")
        print(f"       Training errors range: [{errors.min():.6f}, {errors.max():.6f}]")
        print(f"       Training errors mean: {errors.mean():.6f}")
        print(f"       Training errors std: {errors.std():.6f}")
        print(f"       Threshold: {threshold:.6f}")
        print(f"")
        print(f"       INTERPRETATION:")
        print(f"       Expected false positive rate: ~{expected_fp_rate:.1f}% of normal samples")
        print(f"       This is BY DESIGN (captures typical reconstruction error)")
        print(f"       Trade-off: Higher percentile = fewer false alarms BUT may miss subtle anomalies")
        
        threshold_info.update({
            'percentile': percentile,
            'expected_fp_rate': expected_fp_rate,
            'train_error_mean': errors.mean(),
            'train_error_std': errors.std(),
            'train_error_range': (errors.min(), errors.max())
        })
        
    elif threshold_mode == 'std_dev':
        # Use standard deviation method (mean + k*sigma)
        sigma_multiplier = config.getfloat('lstm_autoencoder', 'threshold_sigma_multiplier', fallback=3.0)
        mean_error = errors.mean()
        std_error = errors.std()
        threshold = mean_error + sigma_multiplier * std_error
        
        # Estimate expected FP rate assuming normal distribution
        # For Gaussian: 3  99.7% within range, 0.3% beyond
        if sigma_multiplier == 3.0:
            expected_fp_rate = 0.15  # One-sided tail
        elif sigma_multiplier == 2.0:
            expected_fp_rate = 2.3
        else:
            expected_fp_rate = None  # Unknown for arbitrary multiplier
        
        print(f"    THRESHOLD CALCULATION (Standard Deviation Method):")
        print(f"       Mean error: {mean_error:.6f}")
        print(f"       Std deviation: {std_error:.6f}")
        print(f"       Sigma multiplier: {sigma_multiplier}")
        print(f"       Threshold (mean + {sigma_multiplier}): {threshold:.6f}")
        if expected_fp_rate:
            print(f"       Expected FP rate (if Gaussian): ~{expected_fp_rate:.2f}%")
        
        threshold_info.update({
            'sigma_multiplier': sigma_multiplier,
            'train_error_mean': mean_error,
            'train_error_std': std_error,
            'expected_fp_rate': expected_fp_rate
        })
        
    elif threshold_mode == 'ema':
        # 
        # EMA DYNAMIC THRESHOLD (training-side anchor)
        # 
        # Here we compute an EMA over the TRAINING errors to obtain a
        # stable anchor value.  The real dynamic threshold is built in
        # detect_anomalies() by running EMA on the TEST errors step-by-step.
        # This anchor is used for:
        #    fallback scalar (train/val anomaly rate comparison)
        #    initialising the test EMA (warm-start)
        # Formula: threshold = EMA_mean(train) + k * EMA_std(train)
        # 
        span = config.getint('lstm_autoencoder', 'ema_span', fallback=20)
        _season   = config.get('global', 'season', fallback='').lower()
        _seas_key = f'ema_sigma_multiplier_{_season}'
        if config.has_option('lstm_autoencoder', _seas_key):
            k = config.getfloat('lstm_autoencoder', _seas_key)
        else:
            k = config.getfloat('lstm_autoencoder', 'ema_sigma_multiplier', fallback=3.0)

        s        = pd.Series(errors)
        ema_mean = s.ewm(span=span, adjust=False).mean()
        ema_std  = s.ewm(span=span, adjust=False).std().fillna(errors.std())
        # Anchor = EMA-based threshold at the last training step
        threshold = float((ema_mean + k * ema_std).iloc[-1])

        print(f"    THRESHOLD CALCULATION (EMA Dynamic Method):")
        print(f"       Span: {span} steps  (~{span * 0.5:.0f} h at 30-min resolution)")
        print(f"       Sigma multiplier k: {k}")
        print(f"       Training errors range : [{errors.min():.6f}, {errors.max():.6f}]")
        print(f"       Training EMA anchor   : {threshold:.6f}  (used as warm-start)")
        print(f"")
        print(f"       INTERPRETATION:")
        print(f"       On TEST data, EMA is recomputed step-by-step  dynamic threshold")
        print(f"       Adapts to slow operational drift (autumn cool-down, etc.)")
        print(f"       Sudden spikes above local EMA baseline  anomaly")

        threshold_info.update({
            'span': span,
            'ema_sigma_multiplier': k,
            'train_error_mean': float(errors.mean()),
            'train_error_std' : float(errors.std()),
            'expected_fp_rate': None,   # no closed-form estimate for EMA
        })

    elif threshold_mode == 'manual':
        # Use manually specified threshold
        threshold = config.getfloat('lstm_autoencoder', 'manual_threshold', fallback=0.01)
        
        print(f"    THRESHOLD (Manual Mode): {threshold:.6f}")
        print(f"       Note: Cannot estimate false positive rate for manual threshold")
        
        threshold_info.update({
            'manual_value': threshold,
            'expected_fp_rate': None
        })
    
    else:
        raise ValueError(f"Unknown threshold_mode: {threshold_mode}")
    
    return threshold, threshold_info


# 
# POST-PROCESSING HELPERS  (smoothing  persistence  gap diagnostic)
# 

def smooth_errors(errors, window):
    """
    Apply a centred rolling-median to anomaly scores.

    Motivation: isolated single-window spikes caused by short interpolated
    segments or sensor noise should not trigger anomaly flags on their own.
    A median filter is preferred over mean because it is robust to the
    occasional extreme outlier in the error signal.

    Args:
        errors : 1-D numpy array of raw reconstruction errors.
        window : odd integer (centred window size, e.g. 3 or 5).
                 1 = no smoothing (identity).

    Returns:
        smoothed : 1-D numpy array, same length as errors.
    """
    if window <= 1:
        return errors.copy()
    s = pd.Series(errors).rolling(window=window, center=True, min_periods=1).median()
    return s.to_numpy()


def compute_occupied_mask(timestamps, config):
    """
    Build a simple occupied/unoccupied mask from timestamp schedule.

    Returns
    -------
    mask : np.ndarray[bool]
        True for occupied windows, False otherwise.
    """
    ts = pd.DatetimeIndex(timestamps)
    if len(ts) == 0:
        return np.array([], dtype=bool)

    start_h = config.getint('lstm_autoencoder', 'occupied_start_hour', fallback=8)
    end_h = config.getint('lstm_autoencoder', 'occupied_end_hour', fallback=18)
    include_weekends = config.getboolean('lstm_autoencoder', 'occupied_include_weekends', fallback=False)

    if start_h == end_h:
        hour_mask = np.ones(len(ts), dtype=bool)
    elif start_h < end_h:
        hour_mask = (ts.hour >= start_h) & (ts.hour < end_h)
    else:
        # Overnight schedule, e.g. 22 -> 06
        hour_mask = (ts.hour >= start_h) | (ts.hour < end_h)

    if include_weekends:
        day_mask = np.ones(len(ts), dtype=bool)
    else:
        day_mask = ts.dayofweek < 5  # Monday=0 ... Sunday=6

    return np.asarray(hour_mask & day_mask, dtype=bool)


def apply_persistence_filter(anomaly_mask, min_consecutive):
    """
    Suppress isolated anomaly windows that are not part of a sustained run.

    A detected anomaly is retained only when it belongs to a run of at least
    `min_consecutive` consecutive anomalous windows.  This reduces false alarms
    caused by individual windows straddling gap boundaries or containing a
    short transient not indicative of a sustained operational anomaly.

    Args:
        anomaly_mask    : Boolean 1-D numpy array (True = anomaly before filter).
        min_consecutive : Minimum run length required to keep the anomaly flag.
                          1 = no filtering (identity).

    Returns:
        filtered : Boolean 1-D numpy array with isolated detections suppressed.
    """
    if min_consecutive <= 1:
        return anomaly_mask.copy()

    filtered = np.zeros_like(anomaly_mask, dtype=bool)
    n = len(anomaly_mask)
    i = 0
    while i < n:
        if anomaly_mask[i]:
            j = i
            while j < n and anomaly_mask[j]:
                j += 1
            run_len = j - i
            if run_len >= min_consecutive:
                filtered[i:j] = True
            i = j
        else:
            i += 1
    return filtered


def gap_vs_anomaly_diagnostic(filtered_anomalies, test_timestamps, gap_info):
    """
    Cross-reference detected anomalies against known removed-gap periods.

    Even after gap removal and NaN-safe windowing, the windows immediately
    *adjacent* to a removed segment can have elevated reconstruction error
    because the model was never trained on such boundary patterns.  This
    function quantifies how many flagged anomalies are temporally close to
    a removed gap, allowing the analyst to separate:

         Non-gap-adjacent anomalies  (error not close to known gaps)
         Gap-boundary FPs   (error inflated by gap proximity)

    Args:
        filtered_anomalies : Boolean array (post-persistence filter).
        test_timestamps    : pd.DatetimeIndex aligned to filtered_anomalies
                             (end-timestamp of each test sequence).
        gap_info           : dict returned by characterize_gaps()  must contain
                             'removed_timestamps' (pd.DatetimeIndex) and
                             'episodes' (list of episode dicts).

    Returns:
        dict with:
        - 'n_anomalies'              : total anomalies after filtering
        - 'n_gap_adjacent'           : anomalies within gap_proximity_steps of a gap
        - 'gap_adjacent_pct'         : proportion (%) of gap-adjacent anomalies
        - 'n_genuine_estimate'       : estimated genuine anomalies (non-gap-adjacent)
        - 'gap_proximity_steps'      : window steps used as proximity threshold
    """
    print(f"\n{'='*70}")
    print(f"[DIAGNOSTIC] GAP vs. ANOMALY DIAGNOSTIC")
    print(f"{'='*70}")

    n_anomalies = int(filtered_anomalies.sum())
    print(f"    Total anomalies (post-filter): {n_anomalies}")

    if n_anomalies == 0 or not gap_info or 'removed_timestamps' not in gap_info:
        print(f"    Nothing to cross-reference (no anomalies or no gap info).")
        print(f"{'='*70}\n")
        return {
            'n_anomalies': n_anomalies,
            'n_gap_adjacent': 0,
            'gap_adjacent_pct': 0.0,
            'n_genuine_estimate': n_anomalies,
            'gap_proximity_steps': 0
        }

    removed_ts = gap_info['removed_timestamps']
    if len(removed_ts) == 0:
        print(f"    No removed gaps to compare against.")
        print(f"{'='*70}\n")
        return {
            'n_anomalies': n_anomalies,
            'n_gap_adjacent': 0,
            'gap_adjacent_pct': 0.0,
            'n_genuine_estimate': n_anomalies,
            'gap_proximity_steps': 0
        }

    # Infer sampling interval from test_timestamps
    if len(test_timestamps) > 1:
        diffs      = np.diff(test_timestamps.to_numpy())
        median_ns  = np.median(diffs[diffs > 0]).astype('float64')
        step_td    = pd.Timedelta(int(median_ns), 'ns')
    else:
        step_td = pd.Timedelta(minutes=30)

    # Proximity window: 2 sequence-steps before/after a gap boundary
    proximity_steps = 2
    prox_td         = step_td * proximity_steps

    # Build a set of gap-boundary times (start and end of each episode)
    boundary_times = set()
    for ep in gap_info.get('episodes', []):
        boundary_times.add(ep['start'])
        boundary_times.add(ep['end'])

    # For each anomaly timestamp, check if it falls within prox_td
    # of any gap boundary OR within the removed set itself
    anomaly_ts   = test_timestamps[filtered_anomalies]
    removed_set  = set(removed_ts.to_list())

    n_gap_adjacent = 0
    for ts in anomaly_ts:
        # Within the removed set (direct overlap)
        if ts in removed_set:
            n_gap_adjacent += 1
            continue
        # Within proximity of a gap boundary
        for bt in boundary_times:
            if abs(ts - bt) <= prox_td:
                n_gap_adjacent += 1
                break

    gap_adjacent_pct   = n_gap_adjacent / n_anomalies * 100 if n_anomalies > 0 else 0.0
    n_genuine_estimate = n_anomalies - n_gap_adjacent

    print(f"    Gap-adjacent anomalies ({proximity_steps} steps \u2248 {prox_td}): "
          f"{n_gap_adjacent} / {n_anomalies}  ({gap_adjacent_pct:.1f}%)")
    print(f"    Non-gap-adjacent anomaly estimate: {n_genuine_estimate}  "
          f"({100 - gap_adjacent_pct:.1f}%)")

    if gap_adjacent_pct > 30:
        print(f"   \u26a0\ufe0f  >30% of anomalies are gap-adjacent \u2014 consider widening the gap "
              f"exclusion window or inspecting neighbouring windows manually.")
    elif gap_adjacent_pct > 0:
        print(f"   \u2139\ufe0f  Some anomalies coincide with gap boundaries (expected at segment edges).")
    else:
        print(f"   \u2705  No anomalies overlap with known gap periods.")

    print(f"{'='*70}\n")

    return {
        'n_anomalies'         : n_anomalies,
        'n_gap_adjacent'      : n_gap_adjacent,
        'gap_adjacent_pct'    : round(gap_adjacent_pct, 2),
        'n_genuine_estimate'  : n_genuine_estimate,
        'gap_proximity_steps' : proximity_steps
    }


def compute_episode_stats(anomaly_mask, timestamps):
    """
    Compute event-level statistics from a boolean per-window anomaly mask.

    A single anomaly *episode* is a contiguous run of consecutive anomalous
    windows.  Point-wise anomaly rates (e.g., 5.43%) only say what fraction
    of windows are flagged  episode stats give the operational answer:
    "how many distinct anomalous periods occurred and how long did they last?"

    Args:
        anomaly_mask : Boolean 1-D numpy array (True = anomalous window).
        timestamps   : pd.DatetimeIndex aligned to anomaly_mask
                       (end-timestamp of each window).

    Returns:
        dict with:
        - 'n_episodes'          : distinct anomaly episodes (contiguous runs)
        - 'mean_duration_steps' : mean episode length in windows
        - 'max_duration_steps'  : longest episode in windows
        - 'mean_duration_h'     : mean episode length in hours
        - 'max_duration_h'      : longest episode in hours
        - 'episodes'            : list of (start_ts, end_ts, n_windows) strings
    """
    n = len(anomaly_mask)
    if n == 0 or int(anomaly_mask.sum()) == 0:
        return {
            'n_episodes': 0, 'mean_duration_steps': 0.0,
            'max_duration_steps': 0, 'mean_duration_h': 0.0,
            'max_duration_h': 0.0, 'episodes': []
        }

    # Infer nominal step size in hours from the timestamp index.
    # Use second-resolution conversion so this is robust to timedelta units
    # (ns/us/ms/s), which can vary by platform/pandas internals.
    if len(timestamps) > 1:
        diffs_s = np.diff(timestamps.to_numpy()).astype('timedelta64[s]').astype(float)
        diffs_s = diffs_s[diffs_s > 0]
        step_h = float(np.median(diffs_s)) / 3600.0 if len(diffs_s) > 0 else 0.5
    else:
        step_h = 0.5  # 30-min fallback

    episodes = []
    i = 0
    while i < n:
        if anomaly_mask[i]:
            j = i
            while j < n and anomaly_mask[j]:
                j += 1
            n_steps = j - i
            ts_start = timestamps[i]   if len(timestamps) > i  else None
            ts_end   = timestamps[j-1] if len(timestamps) >= j else None
            episodes.append((ts_start, ts_end, n_steps))
            i = j
        else:
            i += 1

    lengths    = [ep[2] for ep in episodes]
    mean_steps = float(np.mean(lengths)) if lengths else 0.0
    max_steps  = int(np.max(lengths))    if lengths else 0

    return {
        'n_episodes'          : len(episodes),
        'mean_duration_steps' : round(mean_steps, 1),
        'max_duration_steps'  : max_steps,
        'mean_duration_h'     : round(mean_steps * step_h, 2),
        'max_duration_h'      : round(max_steps  * step_h, 2),
        'episodes'            : [(str(s), str(e), k) for s, e, k in episodes]
    }


def apply_operational_fault_filter(anomaly_mask, test_errors_smoothed, processed_data, config):
    """
    Split statistical anomalies into:
      - operational/periodic peaks
      - potential faults (non-periodic + physics-breaking)

    Returns
    -------
    filtered_mask : np.ndarray[bool]
        Final anomaly mask after semantic filtering.
    semantic_labels : np.ndarray[object]
        Per-window label in {'Normal','Operational peak','Potential fault'}.
    stats : dict
        Diagnostic counters for report/CSV.
    """
    n = len(anomaly_mask)
    semantic_labels = np.array(['Normal'] * n, dtype=object)
    stats = {
        'enabled': False,
        'mode': 'disabled',
        'n_input_anomalies': int(anomaly_mask.sum()),
        'n_periodic_operational': 0,
        'n_physics_break': 0,
        'n_potential_fault': 0,
        'n_suppressed_by_semantic_filter': 0,
    }

    if not config.getboolean('lstm_autoencoder', 'enable_operational_fault_filter', fallback=False):
        return anomaly_mask.copy(), semantic_labels, stats

    stats['enabled'] = True
    _season = config.get('global', 'season', fallback='').strip().lower()
    _strict_key = f'fault_filter_strict_mode_{_season}'
    if config.has_option('lstm_autoencoder', _strict_key):
        strict_mode = config.getboolean('lstm_autoencoder', _strict_key)
    else:
        strict_mode = config.getboolean('lstm_autoencoder', 'fault_filter_strict_mode', fallback=True)
    stats['mode'] = 'strict' if strict_mode else 'label_only'

    test_ts = pd.DatetimeIndex(processed_data.get('test_indices', pd.DatetimeIndex([])))
    df_test = processed_data.get('df_test', pd.DataFrame())
    if len(test_ts) != n or df_test.empty:
        print("     Operational/fault semantic filter skipped: missing aligned test timeline/dataframe")
        return anomaly_mask.copy(), semantic_labels, stats

    # Column selection
    tm_col = config.get('lstm_autoencoder', 'transition_tm_column', fallback='Temperatura Mandata')
    tr_col = config.get('lstm_autoencoder', 'transition_tr_column', fallback='Temperatura Ripresa')
    ts_col = config.get('lstm_autoencoder', 'semantic_ts_column', fallback='Temperatura Saturazione')
    cols = list(df_test.columns)
    if tm_col not in cols or tr_col not in cols or ts_col not in cols:
        print("     Operational/fault semantic filter skipped: TM/TR/TS columns not found in df_test")
        return anomaly_mask.copy(), semantic_labels, stats

    # Align physical signals to sequence-end timestamps.
    tm = df_test[tm_col].reindex(test_ts).ffill().bfill().astype(float)
    tr = df_test[tr_col].reindex(test_ts).ffill().bfill().astype(float)
    ts = df_test[ts_col].reindex(test_ts).ffill().bfill().astype(float)
    d_tm = tm.diff().abs().fillna(0.0).to_numpy()
    d_tr = tr.diff().abs().fillna(0.0).to_numpy()
    abs_tm_ts = (tm - ts).abs().to_numpy()

    # Physics-breaking criteria
    tm_ts_divergence = config.getfloat('lstm_autoencoder', 'physics_tm_ts_divergence_degC', fallback=2.0)
    tm_ramp_min = config.getfloat('lstm_autoencoder', 'physics_tm_ramp_min_degC_per_step', fallback=1.0)
    tr_response_min = config.getfloat('lstm_autoencoder', 'physics_tr_response_min_degC_per_step', fallback=0.25)
    physics_break = (abs_tm_ts >= tm_ts_divergence) | ((d_tm >= tm_ramp_min) & (d_tr <= tr_response_min))

    # Periodicity criteria: repeat near same hour around previous day.
    period_h = config.getfloat('lstm_autoencoder', 'periodicity_hours', fallback=24.0)
    tol_h = config.getfloat('lstm_autoencoder', 'periodicity_tolerance_hours', fallback=1.0)
    err_ratio_tol = config.getfloat('lstm_autoencoder', 'periodicity_error_ratio_tolerance', fallback=0.5)
    ts_ns = test_ts.view('i8')
    period_ns = int(period_h * 3600 * 1e9)
    tol_ns = int(tol_h * 3600 * 1e9)

    periodic = np.zeros(n, dtype=bool)
    active_idx = np.where(anomaly_mask)[0]
    for i in active_idx:
        target_ns = ts_ns[i] - period_ns
        left = np.searchsorted(ts_ns, target_ns - tol_ns, side='left')
        right = np.searchsorted(ts_ns, target_ns + tol_ns, side='right')
        if right <= left:
            continue
        prev_idxs = np.arange(left, right)
        prev_idxs = prev_idxs[prev_idxs < i]
        if len(prev_idxs) == 0:
            continue

        cur_err = float(test_errors_smoothed[i]) + 1e-9
        prev_err_window = test_errors_smoothed[prev_idxs]
        prev_err = float(np.median(prev_err_window)) + 1e-9
        ratio = max(cur_err, prev_err) / min(cur_err, prev_err)

        # Robust periodicity check:
        # 1) similar error magnitude to previous-day same-hour window
        # 2) previous-day window had either an anomaly OR comparable high error
        prev_had_anomaly = bool(np.any(anomaly_mask[prev_idxs]))
        prev_high_error = bool(np.any(prev_err_window >= (cur_err / (1.0 + err_ratio_tol))))
        if ratio <= (1.0 + err_ratio_tol) and (prev_had_anomaly or prev_high_error):
            periodic[i] = True

    # Semantic labels for input anomalies.
    # Periodic + no physics violation = expected daily startup/shutdown peak.
    semantic_labels[anomaly_mask & periodic & ~physics_break] = 'Operational peak'
    # Non-periodic without physics violation = unexplained deviation; most likely a real fault.
    semantic_labels[anomaly_mask & ~periodic & ~physics_break] = 'Potential fault'
    # Physics-violating always overrides (physics_break wins over periodicity).
    semantic_labels[anomaly_mask & physics_break] = 'Potential fault'

    # Final mask
    if strict_mode:
        # Keep all non-periodic anomalies (unexplained AND physics-violating).
        # Periodic windows, even physics-violating ones, are recurring operational events.
        filtered_mask = anomaly_mask & (~periodic)
    else:
        filtered_mask = anomaly_mask.copy()

    stats['n_periodic_operational'] = int(np.sum(anomaly_mask & periodic & ~physics_break))
    stats['n_physics_break'] = int(np.sum(anomaly_mask & physics_break))
    stats['n_potential_fault'] = int(np.sum(semantic_labels == 'Potential fault'))
    stats['n_suppressed_by_semantic_filter'] = int(np.sum(anomaly_mask) - np.sum(filtered_mask))

    print(
        f"    Operational/fault semantic filter ({stats['mode']}): "
        f"input={stats['n_input_anomalies']}, potential_fault={stats['n_potential_fault']}, "
        f"suppressed={stats['n_suppressed_by_semantic_filter']}"
    )
    return filtered_mask, semantic_labels, stats


def build_semantic_events(timestamps, anomaly_mask, semantic_labels, severity_labels, errors, processed_data, config):
    """
    Group final anomaly windows into event-level records (operational/fault).
    """
    ts = pd.DatetimeIndex(timestamps)
    n = len(ts)
    event_ids = np.zeros(n, dtype=int)
    anomaly_mask = np.asarray(anomaly_mask, dtype=bool)
    if len(anomaly_mask) != n:
        return event_ids, [], {'n_events': 0, 'n_fault_events': 0, 'n_operational_events': 0}

    merge_gap_h = config.getfloat('lstm_autoencoder', 'event_merge_gap_hours', fallback=1.0)
    merge_gap = pd.Timedelta(hours=merge_gap_h)
    start_min_len = config.getint('lstm_autoencoder', 'event_start_min_len', fallback=3)
    end_len = config.getint('lstm_autoencoder', 'event_end_len', fallback=2)

    # Bridge short "off" gaps so one event is not fragmented by threshold chatter.
    bridged = anomaly_mask.copy()
    if end_len > 0:
        i = 0
        while i < n:
            if bridged[i]:
                i += 1
                continue
            j = i
            while j < n and not bridged[j]:
                j += 1
            false_len = j - i
            left_true = (i - 1) >= 0 and bridged[i - 1]
            right_true = j < n and bridged[j]
            if left_true and right_true and false_len <= end_len:
                bridged[i:j] = True
            i = j

    # Build contiguous runs and keep only persistent ones.
    runs = []
    i = 0
    while i < n:
        if not bridged[i]:
            i += 1
            continue
        j = i
        while j < n and bridged[j]:
            j += 1
        if (j - i) >= start_min_len:
            runs.append(np.arange(i, j))
        i = j

    if len(runs) == 0:
        return event_ids, [], {'n_events': 0, 'n_fault_events': 0, 'n_operational_events': 0}

    # Merge runs separated by short time gaps.
    merged_runs = [runs[0]]
    for r in runs[1:]:
        prev = merged_runs[-1]
        if (ts[r[0]] - ts[prev[-1]]) <= merge_gap:
            merged_runs[-1] = np.concatenate([prev, r])
        else:
            merged_runs.append(r)

    active_idx = np.concatenate(merged_runs)
    if len(active_idx) == 0:
        return event_ids, [], {'n_events': 0, 'n_fault_events': 0, 'n_operational_events': 0}

    severity_rank = {'Normal': 0, 'Anomaly': 1, 'Minor': 1, 'Moderate': 2, 'Critical': 3}
    events = []
    for ev_id, run in enumerate(merged_runs, start=1):
        event_ids[run] = ev_id
        events.append((ev_id, run.tolist()))

    tm_col = config.get('lstm_autoencoder', 'transition_tm_column', fallback='Temperatura Mandata')
    tr_col = config.get('lstm_autoencoder', 'transition_tr_column', fallback='Temperatura Ripresa')
    ts_col = config.get('lstm_autoencoder', 'semantic_ts_column', fallback='Temperatura Saturazione')
    df_test = processed_data.get('df_test', pd.DataFrame())
    cols = list(df_test.columns)
    have_phys = (tm_col in cols and tr_col in cols and ts_col in cols)
    if have_phys:
        tm_al = df_test[tm_col].reindex(ts).ffill().bfill().astype(float).to_numpy()
        tr_al = df_test[tr_col].reindex(ts).ffill().bfill().astype(float).to_numpy()
        ts_al = df_test[ts_col].reindex(ts).ffill().bfill().astype(float).to_numpy()
    else:
        tm_al = tr_al = ts_al = None

    event_rows = []
    n_fault = 0
    n_oper = 0
    for ev_id, ridx in events:
        ev_ts = ts[ridx]
        ev_sem = semantic_labels[ridx]
        ev_sev = severity_labels[ridx] if severity_labels is not None else np.array(['Normal'] * len(ridx))
        ev_err = errors[ridx]

        has_fault = np.any(ev_sem == 'Potential fault')
        event_type = 'Potential fault' if has_fault else 'Operational transient'
        if has_fault:
            n_fault += 1
        else:
            n_oper += 1

        sev = max(ev_sev, key=lambda x: severity_rank.get(str(x), 0)) if len(ev_sev) else 'Normal'
        peak_pos = int(np.argmax(ev_err))
        peak_idx = ridx[peak_pos]

        # infer step hours for duration conversion
        if len(ev_ts) > 1:
            step_h = float(np.median(np.diff(ev_ts.to_numpy()).astype('timedelta64[s]').astype(float))) / 3600.0
            step_h = max(step_h, 0.5)
        else:
            step_h = 0.5

        event_rows.append({
            'Event_ID': ev_id,
            'Event_Type': event_type,
            'Event_Severity': sev,
            'Start': ev_ts[0],
            'End': ev_ts[-1],
            'Num_Windows': len(ridx),
            'Duration_h': round(len(ridx) * step_h, 2),
            'Peak_Error': float(np.max(ev_err)),
            'Peak_Timestamp': ts[peak_idx],
        })
        if have_phys:
            tm_ev = tm_al[ridx]
            tr_ev = tr_al[ridx]
            ts_ev = ts_al[ridx]
            dtm = np.abs(np.diff(tm_ev)) if len(tm_ev) > 1 else np.array([0.0])
            event_rows[-1].update({
                'TM_TS_MeanAbs': float(np.mean(np.abs(tm_ev - ts_ev))),
                'TM_TR_MeanAbs': float(np.mean(np.abs(tm_ev - tr_ev))),
                'dTMdt_MaxAbs': float(np.max(dtm)),
            })

    stats = {
        'n_events': len(event_rows),
        'n_fault_events': n_fault,
        'n_operational_events': n_oper,
    }
    return event_ids, event_rows, stats


def detect_anomalies(model, processed_data, config):
    """
    Detect anomalies on test set using reconstruction errors
    
    ANOMALY DETECTION WORKFLOW:
    
    1. Calculate reconstruction errors on TRAINING set
    2. Determine threshold from VALIDATION errors (better for distribution shift)
    3. Calculate reconstruction errors on TEST set
    4. Flag test samples with error > threshold as anomalies
    5. Calculate per-feature errors for severity classification
    6. Filter false positives based on min_anomalous_features
    
    Args:
        model: Trained autoencoder
        processed_data: Output from preprocess_data()
        config: ConfigParser object
        
    Returns:
        dict with:
        - 'train_errors': Reconstruction errors on training set
        - 'val_errors': Validation errors
        - 'test_errors': Test errors
        - 'threshold': Anomaly threshold
        - 'anomalies': Boolean array (True = anomaly)
        - 'X_test_reconstructed': Reconstructed test sequences
        - 'feature_errors': Per-feature errors for test set
        - 'severity_labels': Critical/Moderate/Minor classification
        - 'filtered_anomalies': After applying min_features filter
    """
    
    print(f"\n{'='*70}")
    print(f" ANOMALY DETECTION")
    print(f"{'='*70}")
    
    X_train = processed_data['X_train']
    X_val = processed_data['X_val']
    X_test = processed_data['X_test']
    clean_mask = processed_data['clean_sequence_mask']
    
    # Use only clean training data for threshold calculation
    X_train_clean = X_train[clean_mask]

    # Guard: same fallback as trainer  if clean mask emptied everything, use full X_train
    if len(X_train_clean) == 0:
        print(f"     WARNING: clean mask removed ALL {len(X_train)} training sequences.")
        print(f"   Falling back to unfiltered training data for threshold calculation.")
        X_train_clean = X_train

    # 
    # STEP 1: Calculate reconstruction errors
    # 
    print(f"\n   Calculating reconstruction errors...")
    
    train_errors, _ = calculate_reconstruction_error(model, X_train_clean, metric='mae')
    val_errors, _ = calculate_reconstruction_error(model, X_val, metric='mae')
    test_errors, X_test_reconstructed = calculate_reconstruction_error(model, X_test, metric='mae')
    
    print(f"    Train errors: mean={train_errors.mean():.6f}, std={train_errors.std():.6f}")
    print(f"    Val errors:   mean={val_errors.mean():.6f}, std={val_errors.std():.6f}")
    print(f"    Test errors:  mean={test_errors.mean():.6f}, std={test_errors.std():.6f}")

    # 
    # STEP 1b: Rolling-median smoothing on test errors
    # Reduces isolated spikes from gap boundaries or transient sensor noise
    # before thresholding, without distorting sustained anomalous episodes.
    # 
    smooth_window = config.getint('lstm_autoencoder', 'error_smoothing_window', fallback=3)
    test_errors_smoothed = smooth_errors(test_errors, smooth_window)
    if smooth_window > 1:
        print(f"   \u2713 Test errors smoothed (rolling median, window={smooth_window}): "
              f"mean={test_errors_smoothed.mean():.6f}, std={test_errors_smoothed.std():.6f}")
    else:
        print(f"   \u2139\ufe0f  Error smoothing disabled (window=1)")
    # 
    # train_errors was computed from X_train_clean (already filtered) so no
    # second masking is needed  use all train_errors directly.
    threshold, threshold_info = calculate_threshold(train_errors, config)

    # 
    # STEP 2b: EMA DYNAMIC THRESHOLD (only when threshold_mode = 'ema')
    # 
    # Build a per-step threshold series from the TEST (smoothed) errors:
    #   threshold(t) = EMA_mean(test_errors, span) + k * EMA_std(test_errors, span)
    #
    # This lets the baseline adapt to slow operational regime shifts (e.g.
    # autumn temperature drop) without flagging them as anomalies, while
    # still detecting sudden spikes above the local EMA.
    #
    # The EMA uses adjust=False (recursive / "online" style) so each step
    # only depends on past data  no look-ahead bias.
    # 
    threshold_mode_used = config.get('lstm_autoencoder', 'threshold_mode', fallback='percentile')
    mode_threshold_stats = {
        'enabled': False,
        'occupied_windows': 0,
        'unoccupied_windows': 0,
        'occupied_multiplier': 1.0,
        'unoccupied_multiplier': 1.0,
    }

    if threshold_mode_used == 'ema':
        span = config.getint('lstm_autoencoder', 'ema_span', fallback=20)
        _seas_ema     = config.get('global', 'season', fallback='').strip().lower()
        _seas_key_ema = f'ema_sigma_multiplier_{_seas_ema}'
        if config.has_option('lstm_autoencoder', _seas_key_ema):
            k = config.getfloat('lstm_autoencoder', _seas_key_ema)
        else:
            k = config.getfloat('lstm_autoencoder', 'ema_sigma_multiplier', fallback=2.0)
        s = pd.Series(test_errors_smoothed)

        # Optional: reset EMA threshold at discontinuities in the test timeline.
        ema_reset_after_gap = config.getboolean('lstm_autoencoder', 'ema_reset_after_gap', fallback=False)
        ema_reset_gap_hours = config.getfloat('lstm_autoencoder', 'ema_reset_gap_hours', fallback=3.0)
        test_ts = processed_data.get('test_indices', None)

        if ema_reset_after_gap and test_ts is not None and len(test_ts) == len(test_errors_smoothed):
            ts = pd.DatetimeIndex(test_ts)
            deltas = ts.to_series().diff()
            reset_mask = (deltas > pd.Timedelta(hours=ema_reset_gap_hours)).fillna(False).to_numpy()

            threshold_dynamic = np.empty(len(test_errors_smoothed), dtype=float)
            start = 0
            while start < len(test_errors_smoothed):
                next_resets = np.where(reset_mask[start + 1:])[0]
                if len(next_resets) == 0:
                    end = len(test_errors_smoothed)
                else:
                    end = start + 1 + int(next_resets[0])

                s_blk = pd.Series(test_errors_smoothed[start:end])
                ema_mean_blk = s_blk.ewm(span=span, adjust=False).mean()
                ema_std_blk = (
                    s_blk.ewm(span=span, adjust=False).std()
                    .fillna(threshold_info.get('train_error_std', train_errors.std()))
                )
                thr_raw_blk = (ema_mean_blk + k * ema_std_blk).to_numpy()

                threshold_dynamic[start] = threshold
                if end - start > 1:
                    threshold_dynamic[start + 1:end] = thr_raw_blk[:-1]
                start = end
        else:
            ema_mean = s.ewm(span=span, adjust=False).mean()
            ema_std = s.ewm(span=span, adjust=False).std().fillna(
                threshold_info.get('train_error_std', train_errors.std())
            )
            threshold_raw = (ema_mean + k * ema_std).to_numpy()
            threshold_dynamic = np.empty_like(threshold_raw)
            threshold_dynamic[0] = threshold
            threshold_dynamic[1:] = threshold_raw[:-1]

        threshold_series = threshold_dynamic.copy()
        print(
            f"    EMA dynamic threshold (test, causal): "
            f"range [{threshold_series.min():.6f}, {threshold_series.max():.6f}], "
            f"mean {threshold_series.mean():.6f}"
        )
    else:
        # Static threshold (scalar): percentile / std_dev / manual
        threshold_series = np.full(len(test_errors_smoothed), float(threshold), dtype=float)

    # Optional: occupied/unoccupied threshold scaling.
    if config.getboolean('lstm_autoencoder', 'enable_mode_aware_threshold', fallback=False):
        test_ts = pd.DatetimeIndex(processed_data.get('test_indices', pd.DatetimeIndex([])))
        if len(test_ts) == len(threshold_series):
            occ_mask = compute_occupied_mask(test_ts, config)
            occ_mult = config.getfloat('lstm_autoencoder', 'occupied_threshold_multiplier', fallback=1.0)
            unocc_mult = config.getfloat('lstm_autoencoder', 'unoccupied_threshold_multiplier', fallback=1.0)
            threshold_series = threshold_series * np.where(occ_mask, occ_mult, unocc_mult).astype(float)
            mode_threshold_stats.update({
                'enabled': True,
                'occupied_windows': int(np.sum(occ_mask)),
                'unoccupied_windows': int(np.sum(~occ_mask)),
                'occupied_multiplier': float(occ_mult),
                'unoccupied_multiplier': float(unocc_mult),
            })
            print(
                f"    Mode-aware threshold: occupied={mode_threshold_stats['occupied_windows']}, "
                f"unoccupied={mode_threshold_stats['unoccupied_windows']}, "
                f"mult=({occ_mult:.2f}/{unocc_mult:.2f})"
            )
        else:
            print("     Mode-aware threshold skipped: test timestamp alignment mismatch")

    threshold_info['threshold_series'] = threshold_series
    anomalies_pre_persist = test_errors_smoothed > threshold_series

    # Warm-start grace period for EMA only.
    grace_period = config.getint('lstm_autoencoder', 'ema_grace_period_windows', fallback=0)
    if threshold_mode_used == 'ema' and grace_period > 0:
        anomalies_pre_persist[:grace_period] = False
        print(
            f"     EMA grace period: first {grace_period} windows ({grace_period * 0.5:.1f} h) "
            f"excluded from anomaly detection"
        )
    min_consec = config.getint('lstm_autoencoder', 'min_consecutive_anomaly_windows', fallback=1)
    _season = config.get('global', 'season', fallback='').strip().lower()
    _cand_consec_key = f'candidate_min_consecutive_anomaly_windows_{_season}'
    if config.has_option('lstm_autoencoder', _cand_consec_key):
        candidate_min_consec = config.getint('lstm_autoencoder', _cand_consec_key)
    else:
        candidate_min_consec = config.getint(
            'lstm_autoencoder',
            'candidate_min_consecutive_anomaly_windows',
            fallback=max(1, min_consec - 1)
        )
    anomalies  = apply_persistence_filter(anomalies_pre_persist, min_consec)
    candidate_anomalies = apply_persistence_filter(anomalies_pre_persist, candidate_min_consec)
    if min_consec > 1:
        n_suppressed = int(anomalies_pre_persist.sum()) - int(anomalies.sum())
        print(f"   \u2713 Persistence filter (min {min_consec} consecutive windows): "
              f"suppressed {n_suppressed} isolated anomalies")
    if candidate_min_consec != min_consec:
        n_suppressed_candidate = int(anomalies_pre_persist.sum()) - int(candidate_anomalies.sum())
        print(f"   \u2713 Candidate persistence (min {candidate_min_consec} consecutive windows): "
              f"suppressed {n_suppressed_candidate} isolated anomalies")
    n_anomalies = anomalies.sum()
    anomaly_rate = n_anomalies / len(test_errors) * 100
    
    # Compare observed anomaly rate with expected false positive rate
    expected_fp_rate = threshold_info.get('expected_fp_rate', None)
    
    print(f"\n    ANOMALIES DETECTED (Initial):")
    print(f"       Total test samples: {len(test_errors):,}")
    print(f"       Anomalies found: {n_anomalies:,} ({anomaly_rate:.2f}%)")
    if expected_fp_rate is not None:
        print(f"       Expected FP rate: ~{expected_fp_rate:.2f}% ({int(len(test_errors) * expected_fp_rate / 100)} samples)")
        excess_rate = anomaly_rate - expected_fp_rate
        if excess_rate > 0:
            print(f"       Excess anomalies: {excess_rate:.2f}% ({int(len(test_errors) * excess_rate / 100)} samples)")
            print(f"         Potential anomalies beyond the expected false-positive baseline")
        else:
            print(f"       Anomaly rate within expected false positive range")
    
    #  FIX 2  FAIR SPLIT-RATE COMPARISON 
    # For EMA mode the test rate uses causal-EMA + smoothing + persistence;
    # applying the same logic to train and val makes all three rates comparable.
    # For static modes (percentile / std_dev / manual) the scalar is used throughout.
    if threshold_mode_used == 'ema':
        _span         = config.getint('lstm_autoencoder', 'ema_span', fallback=12)
        _seas_key_cmp = f'ema_sigma_multiplier_{_season}'
        if config.has_option('lstm_autoencoder', _seas_key_cmp):
            _k = config.getfloat('lstm_autoencoder', _seas_key_cmp)
        else:
            _k = config.getfloat('lstm_autoencoder', 'ema_sigma_multiplier', fallback=2.0)
        _fallback_std = threshold_info.get('train_error_std', float(train_errors.std()))

        def _split_ema_flags(errors):
            """Causal EMA threshold + smoothing + persistence  mirrors test logic."""
            err_s    = smooth_errors(errors, smooth_window)
            s_       = pd.Series(err_s)
            thr_raw_ = (
                s_.ewm(span=_span, adjust=False).mean()
                + _k * s_.ewm(span=_span, adjust=False).std().fillna(_fallback_std)
            ).to_numpy()
            thr_     = np.empty_like(thr_raw_)
            thr_[0]  = threshold          # training anchor as warm-start
            thr_[1:] = thr_raw_[:-1]
            flags_   = err_s > thr_
            # Apply grace period to train/val splits as well for fair comparison
            if grace_period > 0 and len(flags_) > grace_period:
                flags_[:grace_period] = False
            return apply_persistence_filter(flags_, min_consec)

        train_anomalies = _split_ema_flags(train_errors)
        val_anomalies   = _split_ema_flags(val_errors)
        _rate_note = ' (EMA causal + persistence  same logic as test)'
    else:
        # Static threshold  apply the same smoothing + persistence as test
        # so all three rates are measured with the same ruler (Fix 2).
        train_errors_s  = smooth_errors(train_errors, smooth_window)
        val_errors_s    = smooth_errors(val_errors,   smooth_window)
        train_anomalies = apply_persistence_filter(train_errors_s > threshold, min_consec)
        val_anomalies   = apply_persistence_filter(val_errors_s   > threshold, min_consec)
        _rate_note = ' (static threshold + same smoothing/persistence as test)'

    train_anomaly_rate = train_anomalies.sum() / len(train_errors) * 100
    val_anomaly_rate   = val_anomalies.sum()   / len(val_errors)   * 100

    print(f"\n    ANOMALY RATES BY SPLIT{_rate_note}:")
    print(f"       Training set: {train_anomalies.sum():,}/{len(train_errors):,} ({train_anomaly_rate:.2f}%)")
    print(f"       Validation set: {val_anomalies.sum():,}/{len(val_errors):,} ({val_anomaly_rate:.2f}%)")
    print(f"       Test set: {n_anomalies:,}/{len(test_errors):,} ({anomaly_rate:.2f}%)")
    
    # Check if test anomaly rate is significantly higher (suggests possible issues)
    if train_anomaly_rate == 0:
        if anomaly_rate > 0:
            print(f"        Test anomaly rate is {anomaly_rate:.2f}% while training had 0% (threshold may be tight)")
            print(f"          Test period may contain more anomalous operating patterns")
    elif anomaly_rate > 2 * train_anomaly_rate:
        print(f"        Test anomaly rate is {anomaly_rate / train_anomaly_rate:.1f} higher than training")
        print(f"          Test period may contain more anomalous operating patterns")
    elif anomaly_rate < 0.5 * train_anomaly_rate:
        print(f"       Test anomaly rate is lower than training (good generalization)")
    
    # 
    # STEP 4: Calculate per-feature errors (for severity)
    # 
    feature_errors = calculate_feature_wise_errors(X_test, X_test_reconstructed)
    
    # Calculate per-feature thresholds from TRAINING set (same source as overall threshold)
    # FIXED: was using val_feature_errors which inflates thresholds 9 when val set has
    # distribution shift (e.g. seasonal transition), causing ALL anomalies to be suppressed.
    train_reconstructed_for_features = model.predict(X_train_clean, verbose=0)
    train_feature_errors = calculate_feature_wise_errors(X_train_clean, train_reconstructed_for_features)
    # Feature thresholds: default to training percentile (stable, interpretable).
    # Optionally calibrate to the EMA anchor to keep the per-feature filter on the
    # same scale as the overall threshold when threshold_mode = ema.
    feature_thr_mode = config.get('lstm_autoencoder', 'feature_threshold_mode', fallback='percentile').strip().lower()
    if threshold_mode_used == 'ema' and feature_thr_mode in {'scaled_anchor', 'scaled_to_anchor'}:
        train_feat_means = train_feature_errors.mean(axis=0)
        overall_train_mean = float(train_errors.mean())
        ratio = float(threshold) / (overall_train_mean + 1e-9)
        feature_thresholds = train_feat_means * ratio
        print(f"    Feature thresholds (scaled_anchor, ratio={ratio:.3f}): {feature_thresholds.round(6)}")
    else:
        percentile = config.getfloat('lstm_autoencoder', 'anomaly_threshold_percentile', fallback=99.5)
        feature_thresholds = np.percentile(train_feature_errors, percentile, axis=0)
        print(f"    Feature thresholds (from training {percentile}th pct): {feature_thresholds.round(6)}")
    
    # Candidate feature-agreement filter (screening layer)
    _cand_feat_key = f'candidate_min_anomalous_features_{_season}'
    if config.has_option('lstm_autoencoder', _cand_feat_key):
        candidate_min_features = config.getint('lstm_autoencoder', _cand_feat_key)
    else:
        candidate_min_features = config.getint(
            'lstm_autoencoder',
            'candidate_min_anomalous_features',
            fallback=config.getint('lstm_autoencoder', 'min_anomalous_features', fallback=1)
        )
    if candidate_min_features > 1:
        candidate_feature_count = np.sum(feature_errors > feature_thresholds, axis=1)
        candidate_before = int(candidate_anomalies.sum())
        candidate_anomalies = candidate_anomalies & (candidate_feature_count >= candidate_min_features)
        candidate_after = int(candidate_anomalies.sum())
        if candidate_after != candidate_before:
            print(f"   \u2713 Candidate feature filter (min {candidate_min_features} anomalous features): "
                  f"suppressed {candidate_before - candidate_after} windows")

    # 
    # STEP 5: Filter false positives & classify severity (validated layer)
    # 
    # For EMA mode, pass the dynamic threshold array so that
    # Critical/Moderate boundaries scale to the local EMA baseline.
    # For all other modes, use the scalar threshold.
    # 
    severity_threshold = threshold_info.get('threshold_series', threshold)  # array or scalar
    filtered_results = classify_and_filter_anomalies(
        anomalies, test_errors, feature_errors, 
        severity_threshold, feature_thresholds, config
    )
    severity_labels_pre_semantic = filtered_results.get('severity_labels', None)
    if severity_labels_pre_semantic is not None:
        severity_labels_pre_semantic = severity_labels_pre_semantic.copy()

    # -----------------------------------------------------------------
    # OPTIONAL POST-FILTER: transition masking + TR-stability guard
    # (summer shoulder-season false-positive mitigation)
    # Suppress anomaly flags when TM is in a fast transition AND TR remains stable.
    # -----------------------------------------------------------------
    transition_filter_stats = {
        'enabled': False,
        'n_candidates': 0,
        'n_suppressed': 0,
        'tm_transition_threshold': None,
        'tr_stability_threshold': None
    }
    enable_transition_filter = config.getboolean('lstm_autoencoder', 'enable_transition_tr_guard_filter', fallback=False)
    if enable_transition_filter:
        feature_names = list(processed_data.get('df_test', pd.DataFrame()).columns)
        tm_col = config.get('lstm_autoencoder', 'transition_tm_column', fallback='Temperatura Mandata')
        tr_col = config.get('lstm_autoencoder', 'transition_tr_column', fallback='Temperatura Ripresa')
        tm_roc_threshold = config.getfloat('lstm_autoencoder', 'transition_tm_roc_threshold', fallback=1.5)
        tr_stability_pct = config.getfloat('lstm_autoencoder', 'tr_stability_error_percentile', fallback=99.0)

        transition_filter_stats['enabled'] = True
        transition_filter_stats['tm_transition_threshold'] = float(tm_roc_threshold)

        if tm_col in feature_names and tr_col in feature_names and len(processed_data.get('test_indices', [])) == len(filtered_results['filtered_anomalies']):
            tm_idx = feature_names.index(tm_col)
            tr_idx = feature_names.index(tr_col)

            # Build TM series aligned to sequence end timestamps.
            ts_idx = pd.DatetimeIndex(processed_data['test_indices'])
            tm_series = (
                processed_data['df_test'][tm_col]
                .reindex(ts_idx)
                .ffill()
                .bfill()
            )
            tm_vals = tm_series.to_numpy(dtype=float)
            tm_roc = np.abs(np.gradient(tm_vals))
            transition_mask = tm_roc > tm_roc_threshold

            # TR stability threshold from training feature-error distribution.
            tr_stability_threshold = float(np.percentile(train_feature_errors[:, tr_idx], tr_stability_pct))
            tr_stable_mask = feature_errors[:, tr_idx] < tr_stability_threshold
            transition_filter_stats['tr_stability_threshold'] = tr_stability_threshold

            suppress_mask = transition_mask & tr_stable_mask & filtered_results['filtered_anomalies']
            n_suppressed = int(suppress_mask.sum())
            transition_filter_stats['n_candidates'] = int((transition_mask & tr_stable_mask).sum())
            transition_filter_stats['n_suppressed'] = n_suppressed

            if n_suppressed > 0:
                filtered_results['filtered_anomalies'][suppress_mask] = False
                if filtered_results.get('severity_labels') is not None:
                    filtered_results['severity_labels'] = np.where(
                        filtered_results['filtered_anomalies'],
                        filtered_results['severity_labels'],
                        'Normal'
                    )
                print(f"   \u2713 Transition/TR guard filter: suppressed {n_suppressed} anomalies "
                      f"(TM ROC > {tm_roc_threshold:.2f} and TR error < {tr_stability_threshold:.4f})")
            else:
                print(f"   \u2139\ufe0f  Transition/TR guard filter enabled: no anomalies matched suppression criteria")
        else:
            print(f"   \u26a0\ufe0f  Transition/TR guard filter skipped: TM/TR columns not found or test index mismatch")

    # -----------------------------------------------------------------
    # OPTIONAL SEMANTIC FILTER: operational peaks vs potential faults
    # -----------------------------------------------------------------
    semantic_labels = np.array(['Normal'] * len(test_errors), dtype=object)
    semantic_filter_stats = {
        'enabled': False,
        'mode': 'disabled',
        'n_input_anomalies': int(filtered_results['filtered_anomalies'].sum()),
        'n_periodic_operational': 0,
        'n_physics_break': 0,
        'n_potential_fault': int(filtered_results['filtered_anomalies'].sum()),
        'n_suppressed_by_semantic_filter': 0,
    }
    sem_mask, sem_labels, sem_stats = apply_operational_fault_filter(
        filtered_results['filtered_anomalies'],
        test_errors_smoothed,
        processed_data,
        config
    )
    filtered_results['filtered_anomalies'] = sem_mask
    semantic_labels = sem_labels
    semantic_filter_stats = sem_stats
    if filtered_results.get('severity_labels') is not None:
        filtered_results['severity_labels'] = np.where(
            filtered_results['filtered_anomalies'],
            filtered_results['severity_labels'],
            'Normal'
        )

    #  FIX 1b  PERSISTENCE ON FINAL FILTERED MASK 
    # The severity/feature-count filter can "break" a multi-window run by
    # suppressing one window inside it, leaving behind a single isolated window.
    # Re-apply the persistence constraint on the final mask so that
    # min_consecutive_anomaly_windows truly eliminates all single-window episodes.
    if min_consec > 1:
        filtered_final = apply_persistence_filter(filtered_results['filtered_anomalies'], min_consec)
        if int(filtered_final.sum()) != int(filtered_results['filtered_anomalies'].sum()):
            n_supp_final = int(filtered_results['filtered_anomalies'].sum()) - int(filtered_final.sum())
            print(f"    Persistence (final mask {min_consec}): suppressed {n_supp_final} post-filter isolated anomalies")

        filtered_results['filtered_anomalies'] = filtered_final

        # Keep labels consistent with the final mask
        if 'severity_labels' in filtered_results and filtered_results['severity_labels'] is not None:
            filtered_results['severity_labels'] = np.where(filtered_final, filtered_results['severity_labels'], 'Normal')

        # Recompute severity counts if they exist (safe when enable_severity=false)
        if 'n_critical' in filtered_results:
            filtered_results['n_critical'] = int(np.sum((filtered_results.get('severity_labels') == 'Critical') & filtered_final))
        if 'n_moderate' in filtered_results:
            filtered_results['n_moderate'] = int(np.sum((filtered_results.get('severity_labels') == 'Moderate') & filtered_final))
        if 'n_minor' in filtered_results:
            filtered_results['n_minor'] = int(np.sum((filtered_results.get('severity_labels') == 'Minor') & filtered_final))

    # Keep severity counters consistent with the current final anomaly mask.
    if filtered_results.get('severity_labels') is not None:
        final_mask = filtered_results['filtered_anomalies']
        if 'n_critical' in filtered_results:
            filtered_results['n_critical'] = int(np.sum((filtered_results['severity_labels'] == 'Critical') & final_mask))
        if 'n_moderate' in filtered_results:
            filtered_results['n_moderate'] = int(np.sum((filtered_results['severity_labels'] == 'Moderate') & final_mask))
        if 'n_minor' in filtered_results:
            filtered_results['n_minor'] = int(np.sum((filtered_results['severity_labels'] == 'Minor') & final_mask))
    
    print(f"\n   \U0001f3af ANOMALIES AFTER FILTERING:")
    print(f"      \u2022 Filtered anomalies: {filtered_results['filtered_anomalies'].sum()}")
    _sev_enabled = config.getboolean('lstm_autoencoder', 'enable_severity', fallback=True)
    if _sev_enabled:
        print(f"      \u2022 Critical: {filtered_results['n_critical']}")
        print(f"      \u2022 Moderate: {filtered_results['n_moderate']}")
        print(f"      \u2022 Minor: {filtered_results['n_minor']}")
    print(f"{'='*70}\n")

    # =================================================================
    # STEP 5b: Event-level grouping (merge adjacent anomaly windows)
    # =================================================================
    event_ids, event_rows, event_stats = build_semantic_events(
        processed_data.get('test_indices', pd.DatetimeIndex([])),
        filtered_results['filtered_anomalies'],
        semantic_labels,
        severity_labels_pre_semantic,
        test_errors,
        processed_data,
        config
    )
    print(f"    EVENT GROUPING: {event_stats['n_events']} events "
          f"(fault={event_stats['n_fault_events']}, operational={event_stats['n_operational_events']})")

    # Candidate screening statistics (before strict semantic/validation layer)
    candidate_test_timestamps = processed_data.get('test_indices', pd.DatetimeIndex([]))
    candidate_episode_stats = compute_episode_stats(candidate_anomalies, candidate_test_timestamps)
    candidate_anomaly_rate = (candidate_anomalies.sum() / len(test_errors) * 100) if len(test_errors) else 0.0
    print(
        f"    CANDIDATE SCREENING: {int(candidate_anomalies.sum())} windows "
        f"({candidate_anomaly_rate:.2f}%), episodes={candidate_episode_stats['n_episodes']}"
    )

    # =================================================================
    # STEP 6: Gap vs. Anomaly FP Diagnostic
    # Cross-reference detected anomalies against known removed-gap periods
    # to estimate how many flags are gap-boundary artefacts vs non-gap-adjacent anomalies.
    # =================================================================
    test_timestamps = processed_data.get('test_indices', pd.DatetimeIndex([]))
    gap_info        = processed_data.get('gap_info', {})
    gap_diagnostic  = gap_vs_anomaly_diagnostic(
        filtered_results['filtered_anomalies'], test_timestamps, gap_info
    )

    # =================================================================
    # STEP 7: Episode-level statistics (FIX 3)
    # Convert the point-wise anomaly mask into distinct anomaly episodes
    # (contiguous runs) with count and duration metrics.
    # =================================================================
    episode_stats = compute_episode_stats(
        filtered_results['filtered_anomalies'], test_timestamps
    )
    if episode_stats['n_episodes'] > 0:
        print(f"\n    ANOMALY EPISODE STATISTICS:")
        print(f"       Distinct episodes : {episode_stats['n_episodes']}")
        print(f"       Mean duration     : {episode_stats['mean_duration_steps']} windows "
              f"({episode_stats['mean_duration_h']} h)")
        print(f"       Longest episode   : {episode_stats['max_duration_steps']} windows "
              f"({episode_stats['max_duration_h']} h)")
        if episode_stats['episodes']:
            print(f"       Episode list (start  end, windows):")
            for start_s, end_s, n_win in episode_stats['episodes']:
                print(f"           {start_s}    {end_s}  ({n_win} windows)")
    else:
        print(f"\n    ANOMALY EPISODE STATISTICS: no episodes detected.")

    return {
        'train_errors'           : train_errors,
        'val_errors'             : val_errors,
        'test_errors'            : test_errors,            # raw per-window MAE (for CSV/viz)
        'test_errors_smoothed'   : test_errors_smoothed,   # rolling-median smoothed
        'threshold'              : threshold,
        'threshold_info'         : threshold_info,
        'anomalies'              : anomalies,              # after persistence filter
        'candidate_anomalies'    : candidate_anomalies,    # candidate screening layer
        'anomalies_pre_persist'  : anomalies_pre_persist,  # before persistence (diagnostic)
        'X_test_reconstructed'   : X_test_reconstructed,
        'feature_errors'         : feature_errors,
        'feature_thresholds'     : feature_thresholds,
        # Anomaly rates for comparison
        'train_anomaly_rate'     : train_anomaly_rate,
        'val_anomaly_rate'       : val_anomaly_rate,
        'test_anomaly_rate'      : anomaly_rate,
        'candidate_anomaly_rate' : candidate_anomaly_rate,
        'candidate_episode_stats': candidate_episode_stats,
        # Gap FP diagnostic
        'gap_diagnostic'         : gap_diagnostic,
        'transition_filter_stats': transition_filter_stats,
        'mode_threshold_stats'   : mode_threshold_stats,
        'semantic_filter_stats'  : semantic_filter_stats,
        'semantic_labels'        : semantic_labels,
        'event_ids'              : event_ids,
        'event_rows'             : event_rows,
        'event_stats'            : event_stats,
        # Episode-level statistics
        'episode_stats'          : episode_stats,
        **filtered_results  # Unpack severity classification results
    }


def classify_and_filter_anomalies(anomalies, test_errors, feature_errors, 
                                   threshold, feature_thresholds, config):
    """
    Filter false positives and classify anomaly severity
    
    FILTERING LOGIC (Manager's Request - "un compromesso"):
    
    1. Count how many features are anomalous per sample
    2. Only flag as anomaly if >= min_anomalous_features are bad
       (Prevents single noisy sensor from triggering false alarm)
    
    SEVERITY CLASSIFICATION:
    
    - Critical: Error > critical_multiplier  threshold (e.g., 3)
    - Moderate: Error > moderate_multiplier  threshold (e.g., 2)
    - Minor: Error > 1 threshold
    
    Args:
        anomalies: Initial anomaly flags
        test_errors: Overall reconstruction errors
        feature_errors: Per-feature errors (n_samples, n_features)
        threshold: Overall threshold
        feature_thresholds: Per-feature thresholds
        config: ConfigParser object
        
    Returns:
        dict with filtered anomalies and severity classification
    """
    
    min_features = config.getint('lstm_autoencoder', 'min_anomalous_features', fallback=2)
    critical_mult = config.getfloat('lstm_autoencoder', 'critical_severity_multiplier', fallback=3.0)
    moderate_mult = config.getfloat('lstm_autoencoder', 'moderate_severity_multiplier', fallback=2.0)
    # enable_severity = false   all anomalies labelled 'Anomaly' (no Minor/Moderate/Critical)
    # enable_severity = true    full three-level classification restored
    enable_severity = config.getboolean('lstm_autoencoder', 'enable_severity', fallback=True)

    # Count anomalous features per sample
    features_anomalous = (feature_errors > feature_thresholds).sum(axis=1)
    
    # Filter: Keep only if >= min_features are anomalous
    filtered_anomalies = anomalies & (features_anomalous >= min_features)
    
    # CRITICAL: use dtype=object so long strings ('Moderate','Critical') are never truncated
    severity_labels = np.array(['Normal'] * len(test_errors), dtype=object)

    if enable_severity:
        # Full three-level classification
        severity_labels[filtered_anomalies] = 'Minor'
        severity_labels[filtered_anomalies & (test_errors > moderate_mult * threshold)] = 'Moderate'
        severity_labels[filtered_anomalies & (test_errors > critical_mult * threshold)] = 'Critical'
        n_critical = (severity_labels == 'Critical').sum()
        n_moderate = (severity_labels == 'Moderate').sum()
        n_minor    = (severity_labels == 'Minor').sum()
    else:
        # Binary mode: anomaly is simply 'Anomaly', no sub-levels
        severity_labels[filtered_anomalies] = 'Anomaly'
        n_critical = 0
        n_moderate = 0
        n_minor    = 0

    #  report_minor_anomalies = false: hide Minor from filtered_anomalies
    # (Minor anomalies are still written to CSV as Severity='Minor', but they
    #  are excluded from filtered_anomalies so plots and summary counts don't
    #  include them.  This honours the config key that previously had no effect.)
    report_minor = config.getboolean('lstm_autoencoder', 'report_minor_anomalies', fallback=True)
    if not report_minor:
        minor_only_mask = (severity_labels == 'Minor')
        filtered_anomalies = filtered_anomalies & ~minor_only_mask
        # n_minor stays as-is so the count is still reported for reference

    return {
        'filtered_anomalies': filtered_anomalies,
        'severity_labels': severity_labels,
        'features_anomalous': features_anomalous,
        'n_critical': n_critical,
        'n_moderate': n_moderate,
        'n_minor': n_minor
    }


def calculate_metrics(results, processed_data):
    """
    Calculate evaluation metrics (R, etc.)
    
    Args:
        results: Output from detect_anomalies()
        processed_data: Preprocessor output
        
    Returns:
        dict with metrics
    """
    
    print(f"\n{'='*70}")
    print(f" EVALUATION METRICS")
    print(f"{'='*70}")
    
    X_test = processed_data['X_test']
    X_reconstructed = results['X_test_reconstructed']
    scaler = processed_data['scaler']
    n_features = X_test.shape[-1]

    #  Flatten scaled sequences (for R only  R is scale-invariant) 
    X_test_flat  = X_test.reshape(-1, n_features)
    X_recon_flat = X_reconstructed.reshape(-1, n_features)

    # R is computed in scaled space (scale-invariant, correct either way)
    r2_overall     = r2_score(X_test_flat, X_recon_flat)
    r2_per_feature = [r2_score(X_test_flat[:, i], X_recon_flat[:, i])
                      for i in range(n_features)]

    #  Inverse-transform to physical units (C) for RMSE / MSE / MAE 
    X_test_phys  = scaler.inverse_transform(X_test_flat)
    X_recon_phys = scaler.inverse_transform(X_recon_flat)

    # Per-feature metrics in C
    rmse_per_feature, mse_per_feature, mae_per_feature = [], [], []
    for i in range(n_features):
        mse  = mean_squared_error(X_test_phys[:, i], X_recon_phys[:, i])
        mae  = mean_absolute_error(X_test_phys[:, i], X_recon_phys[:, i])
        rmse = np.sqrt(mse)
        rmse_per_feature.append(rmse)
        mse_per_feature.append(mse)
        mae_per_feature.append(mae)

    # Overall metrics in C (mean across all features)
    mse_overall  = mean_squared_error(X_test_phys, X_recon_phys)
    mae_overall  = mean_absolute_error(X_test_phys, X_recon_phys)
    rmse_overall = np.sqrt(mse_overall)

    print(f"\n   Reconstruction Quality (physical units  C):")
    print(f"       Overall R:   {r2_overall:.4f}")
    print(f"       Overall RMSE: {rmse_overall:.4f} C")
    print(f"       Overall MSE:  {mse_overall:.4f} (scaled)")
    print(f"       Overall MAE:  {mae_overall:.4f} C")
    print(f"       Per-feature R:   {[f'{r:.4f}' for r in r2_per_feature]}")
    print(f"       Per-feature RMSE: {[f'{v:.4f} C' for v in rmse_per_feature]}")
    print(f"       Per-feature MSE:  {[f'{v:.4f}' for v in mse_per_feature]} (scaled)")
    print(f"       Per-feature MAE:  {[f'{v:.4f} C' for v in mae_per_feature]}")

    print(f"\n   Anomaly Detection:")
    print(f"       Test samples: {len(X_test)}")
    print(f"       Anomalies: {results['filtered_anomalies'].sum()} ({results['filtered_anomalies'].sum()/len(X_test)*100:.2f}%)")
    if results['n_critical'] or results['n_moderate'] or results['n_minor']:
        print(f"       Severity breakdown:")
        print(f"         - Critical: {results['n_critical']}")
        print(f"         - Moderate: {results['n_moderate']}")
        print(f"         - Minor: {results['n_minor']}")
    print(f"{'='*70}\n")

    return {
        'r2_overall': r2_overall,
        'r2_per_feature': r2_per_feature,
        'rmse_overall': rmse_overall,
        'mse_overall': mse_overall,
        'mae_overall': mae_overall,
        'rmse_per_feature': rmse_per_feature,
        'mse_per_feature': mse_per_feature,
        'mae_per_feature': mae_per_feature,
        'anomaly_rate': results['filtered_anomalies'].sum() / len(X_test) * 100
    }


# ═════════════════════════════════════════════════════════════════════════════
# PHYSICS RULE CHECKS  (Yeom et al. 2025 — hybrid layer)
# Two deterministic fault indicators applied independently of the LSTM-AE score.
# ═════════════════════════════════════════════════════════════════════════════

def apply_physics_rule_checks(df_full, test_timestamps, config):
    """
    Evaluate two rule-based physical fault indicators at each test-window timestamp.

    Rule 1 — Heat-transfer fault:
        delta_t_signed (TM − TR) < threshold  while  supply-fan Mod > mod_threshold
        Physical meaning: when the fan is running at >30 % but supply air is barely
        warmer/cooler than return air, the coil is not transferring heat — indicates
        fouling, refrigerant loss, or actuator failure.

    Rule 2 — Control fault:
        rolling mean( |TM − SP_TM| , window=4 h ) > setpoint_err_threshold
        Physical meaning: sustained setpoint error over 4 hours means the controller
        cannot reach its target — indicates actuator fault, stuck valve, or BMS error.

    Both rules are evaluated on the ORIGINAL (physical-unit) dataset at each
    sequence-end timestamp, so they are independent of the LSTM scaling/windowing.

    Configurable via config.ini [lstm_autoencoder] keys:
        rule_delta_t_threshold    (°C,   default 0.5)
        rule_mod_threshold_pct    (%,    default 30.0)
        rule_setpoint_err_thresh  (°C,   default 1.5)
        rule_setpoint_roll_hours  (h,    default 4.0)

    Args:
        df_full         : full physical-units DataFrame indexed by DatetimeIndex
                          (the original feature-engineered dataset, unscaled)
        test_timestamps : array-like of timestamps, one per test window
        config          : ConfigParser

    Returns:
        pd.DataFrame with boolean columns ['Rule_HeatTransfer', 'Rule_Control']
        aligned to test_timestamps (integer index 0..n-1).
    """
    TM_COL  = 'Temperatura Mandata'
    TR_COL  = 'Temperatura Ripresa'
    MOD_COL = 'Modulazione Ventilatore Mandata'
    SP_COL  = 'Set Points Temperatura Mandata'

    dt_threshold  = config.getfloat('lstm_autoencoder', 'rule_delta_t_threshold',   fallback=0.5)
    mod_threshold = config.getfloat('lstm_autoencoder', 'rule_mod_threshold_pct',   fallback=30.0)
    sp_err_thresh = config.getfloat('lstm_autoencoder', 'rule_setpoint_err_thresh', fallback=1.5)
    sp_roll_h     = config.getfloat('lstm_autoencoder', 'rule_setpoint_roll_hours', fallback=4.0)

    n    = len(test_timestamps)
    rule1 = np.zeros(n, dtype=bool)
    rule2 = np.zeros(n, dtype=bool)

    avail    = set(df_full.columns)
    has_r1   = (TM_COL in avail) and (TR_COL in avail) and (MOD_COL in avail)
    has_r2   = (TM_COL in avail) and (SP_COL in avail)

    if not has_r1 and not has_r2:
        print("   [Rule checks] Required columns not found in df_full — skipping both rules.")
        return pd.DataFrame({'Rule_HeatTransfer': rule1, 'Rule_Control': rule2})

    df  = df_full.sort_index()
    ts  = pd.DatetimeIndex(test_timestamps)

    # ── Rule 1: heat-transfer fault ──────────────────────────────────────────
    if has_r1:
        delta_t = (df[TM_COL] - df[TR_COL]).reindex(ts, method='ffill')
        mod     = df[MOD_COL].reindex(ts, method='ffill')
        # Modulation column may be on a 0-1 or 0-100 scale; detect and normalise.
        q99 = float(mod.abs().quantile(0.99)) if len(mod.dropna()) else 0
        if q99 <= 1.5:
            mod = mod * 100.0   # convert 0-1 to 0-100 for consistent threshold
        rule1 = ((delta_t < dt_threshold) & (mod > mod_threshold)).fillna(False).to_numpy()
    else:
        print(f"   [Rule checks] Rule 1 skipped — missing: "
              f"{[c for c in (TM_COL, TR_COL, MOD_COL) if c not in avail]}")

    # ── Rule 2: control fault ────────────────────────────────────────────────
    if has_r2:
        setpoint_err = (df[TM_COL] - df[SP_COL]).abs()
        # Rolling window size in samples (30-min resolution → 2 samples/h)
        roll_samples = max(1, int(sp_roll_h * 2))
        setpoint_roll = setpoint_err.rolling(window=roll_samples, min_periods=1).mean()
        rule2 = (setpoint_roll.reindex(ts, method='ffill') > sp_err_thresh).fillna(False).to_numpy()
    else:
        print(f"   [Rule checks] Rule 2 skipped — missing: "
              f"{[c for c in (TM_COL, SP_COL) if c not in avail]}")

    n_r1 = int(rule1.sum())
    n_r2 = int(rule2.sum())
    n_both = int((rule1 & rule2).sum())
    print(f"\n   [Physics Rule Checks]")
    print(f"      Rule 1 (heat-transfer, dT<{dt_threshold}°C & Mod>{mod_threshold}%): "
          f"{n_r1}/{n} windows flagged ({n_r1/n*100:.1f}%)")
    print(f"      Rule 2 (control fault, |SP_error| 4h-mean>{sp_err_thresh}°C): "
          f"{n_r2}/{n} windows flagged ({n_r2/n*100:.1f}%)")
    if n_both > 0:
        print(f"      Both rules simultaneously: {n_both} windows — highest-confidence fault candidates")

    return pd.DataFrame({'Rule_HeatTransfer': rule1, 'Rule_Control': rule2})


#
# TEST SCRIPT - loads saved model + data and runs full evaluation
#
if __name__ == "__main__":
    import configparser
    import importlib
    import sys
    from pathlib import Path
    from tensorflow.keras.models import load_model

    print("\n*** TESTING EVALUATOR.PY (standalone) ***\n")

    #  1. Config 
    config = configparser.ConfigParser()
    config_path = Path(__file__).parent.parent.parent / 'config.ini'
    config.read(config_path, encoding='utf-8-sig')
    print(f" Config loaded: {config_path}")

    #  2. Load & preprocess data 
    sys.path.insert(0, str(Path(__file__).parent))
    data_loader   = importlib.import_module('2_data_loader')
    preprocessor  = importlib.import_module('3_preprocessor')

    data_dict      = data_loader.load_and_prepare_data(config)
    processed_data = preprocessor.preprocess_data(data_dict, config)
    print(f" Data preprocessed")
    print(f"   X_train: {processed_data['X_train'].shape}")
    print(f"   X_val:   {processed_data['X_val'].shape}")
    print(f"   X_test:  {processed_data['X_test'].shape}")

    #  3. Load saved model 
    base_folder  = config.get('paths', 'plots_folder', fallback='./plots')
    building_id  = config.get('global', 'building_id', fallback='C1')
    ahu_unit     = config.get('global', 'ahu_unit',     fallback='UTA1')
    season       = config.get('global', 'season',       fallback='Summer')
    year         = config.get('global', 'year',         fallback='2025')
    model_name   = f'best_autoencoder_{building_id}_{ahu_unit}_{season}{year}.keras'
    model_path   = (
        Path(__file__).parent.parent.parent
        / base_folder.replace('./', '')
        / 'lstm_autoencoder' / 'models' / model_name
    )
    if not model_path.exists():
        print(f"\n Model not found: {model_path}")
        print("   Run 1_main.py first to train and save the model.")
        sys.exit(1)

    model = load_model(str(model_path))
    print(f"\n Model loaded: {model_path}")
    model.summary()

    #  4. Detect anomalies 
    results = detect_anomalies(model, processed_data, config)

    #  5. Metrics 
    metrics = calculate_metrics(results, processed_data)

    #  6. Summary 
    print("\n" + "="*70)
    print(" EVALUATION SUMMARY")
    print("="*70)
    print(f"   Overall R:          {metrics['r2_overall']:.4f}")
    print(f"   Anomaly rate:        {metrics['anomaly_rate']:.2f}%")
    print(f"   Threshold:           {results['threshold']:.6f}")
    print(f"   Train err mean:      {results['train_errors'].mean():.6f}")
    print(f"   Val   err mean:      {results['val_errors'].mean():.6f}")
    print(f"   Test  err mean:      {results['test_errors'].mean():.6f}")
    print(f"   Critical anomalies:  {results['n_critical']}")
    print(f"   Moderate anomalies:  {results['n_moderate']}")
    print(f"   Minor anomalies:     {results['n_minor']}")
    print("="*70 + "\n")

