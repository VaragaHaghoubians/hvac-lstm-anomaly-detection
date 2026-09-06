"""
DATA LOADER - Load and prepare HVAC sensor data
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Purpose:
    1. Load feature-engineered CSV data
    2. Extract specified feature columns
    3. Create sliding window sequences for LSTM input
    4. Validate data quality

Input:  CSV file with timestamp and sensor readings
Output: Dictionary with cleaned data and metadata
"""

import pandas as pd
import numpy as np
import pytz
from pathlib import Path


def load_and_prepare_data(config):
    """
    Load CSV, handle missing values, return clean dataframe
    
    Args:
        config: ConfigParser object with DATA section
        
    Returns:
        dict with:
        - 'df': cleaned dataframe (datetime index)
        - 'features': list of feature column names
        - 'n_features': number of features
        - 'date_range': tuple of (start_date, end_date)
    """
    
    # Get parameters from config
    # Note: Using main config.ini structure
    # Data path from paths section (auto-generated)
    building_id = config.get('global', 'building_id')
    ahu_unit = config.get('global', 'ahu_unit')
    season = config.get('global', 'season').strip()
    year = config.get('global', 'year')
    
    # Build data path
    base_folder = config.get('paths', 'processed_folder')
    data_filename = f"{building_id}_{ahu_unit}_{season}{year}_features.csv"
    data_path = f"{base_folder}/feature_engineering/{building_id}_{ahu_unit}_{season}{year}/{data_filename}"
    
    # Get column names from data section
    date_column = config.get('data', 'time_column', fallback='Time')
    
    # Auto-select features based on season
    season_key = f"features_{season.lower()}"
    if config.has_option('lstm_autoencoder', season_key):
        feature_columns = [f.strip() for f in config.get('lstm_autoencoder', season_key).split(',')]
        print(f"   Season-specific features loaded: [{season}] → {season_key}")
    else:
        # Fallback to generic 'features' key if season-specific not found
        feature_columns = [f.strip() for f in config.get('lstm_autoencoder', 'features').split(',')]
        print(f"   Generic features loaded (no season-specific key found for '{season_key}')")
    
    print(f"\n{'='*70}")
    print(f"📂 LOADING DATA")
    print(f"{'='*70}")
    print(f"   File: {data_path}")
    print(f"   Features: {feature_columns}")
    
    # Resolve path relative to PROJECT ROOT (2 levels up from scripts/lstm_anomaly_detection)
    project_root = Path(__file__).resolve().parent.parent.parent
    if not Path(data_path).is_absolute():
        filepath = project_root / data_path.replace('./', '')
    else:
        filepath = Path(data_path)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    # Load CSV - first without parsing dates to handle timezone issues
    df = pd.read_csv(filepath)
    
    # Convert date column to datetime and handle timezone.
    # utc=True correctly collapses mixed-offset strings (+01:00/+02:00 from DST).
    # tz_convert('Europe/Rome') restores correct local wall-clock time before
    # stripping timezone — without this step the index is shifted by 1-2 hours.
    ts = pd.to_datetime(df[date_column], utc=True, errors='coerce')
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert(pytz.timezone('Europe/Rome')).dt.tz_localize(None)
    df[date_column] = ts
    df.set_index(date_column, inplace=True)

    # Drop duplicate timestamps created by the DST fold (end of daylight saving time):
    # tz_convert('Europe/Rome') maps two distinct UTC instants to the same naive local
    # wall-clock time (e.g. 2025-10-26 02:00 and 02:30 each appear twice).
    # Keep the first occurrence (the summer-time / +02:00 reading) and discard the repeat.
    if df.index.duplicated().any():
        n_dupes = int(df.index.duplicated().sum())
        print(f"   ⚠️  Dropped {n_dupes} duplicate timestamp(s) from DST fold "
              f"(keeping first / summer-time occurrence)")
        df = df[~df.index.duplicated(keep='first')]

    # ========================================================================
    # PHASE 1: Filter to stable period only (exclude Nov 20-27 distribution shift)
    # NOTE: This block was specific to Winter 2025 dataset. Disabled for Winter 2026.
    # Uncomment ONLY if running on the Winter 2025 dataset (Oct 15 - Nov 27, 2025).
    # ========================================================================
    # stable_end = pd.Timestamp('2025-11-14')
    # original_samples = len(df)
    # df = df[df.index <= stable_end]
    # filtered_samples = len(df)
    # print(f"\n   🔧 FILTERED to stable period (Oct 15 - Nov 14, 2025)")
    # print(f"   Reason: Exclude Nov 20-27 distribution shift (first winter cold snap)")
    # print(f"   Original samples: {original_samples:,}")
    # print(f"   After filtering: {filtered_samples:,}")
    # print(f"   Removed: {original_samples - filtered_samples:,} samples")
    # ========================================================================
    
    # ── On-the-fly derived features ─────────────────────────────────────────
    # Compute setpoint_error (TM - SP_Mandata) if it's requested but not in the CSV.
    # This happens when feature engineering was run before the winter fallback fix.
    if 'setpoint_error' in feature_columns and 'setpoint_error' not in df.columns:
        sp_cols = [c for c in df.columns if ('set point' in c.lower() or 'setpoint' in c.lower())
                   and 'compensat' not in c.lower()]
        if not sp_cols:
            sp_cols = [c for c in df.columns if 'set point' in c.lower() or 'setpoint' in c.lower()]
        if sp_cols and 'Temperatura Mandata' in df.columns:
            chosen_sp = next((c for c in sp_cols if 'mandata' in c.lower()), sp_cols[0])
            df['setpoint_error'] = df['Temperatura Mandata'] - df[chosen_sp]
            print(f"   ℹ️  Computed setpoint_error on-the-fly = TM − '{chosen_sp}'")
        else:
            raise ValueError("setpoint_error requested but no setpoint column found in CSV. "
                             "Re-run feature engineering (12.feature_engineering.py) first.")

    # Check if all features exist
    missing_features = [f for f in feature_columns if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in CSV: {missing_features}\nAvailable: {df.columns.tolist()}")
    
    # Extract only needed features
    df_features = df[feature_columns].copy()
    
    # Sort by time (critical for time series!)
    df_features = df_features.sort_index()

    # Exclude Christmas / New Year shutdown gap (configurable via config.ini)
    # This period contains ~380 rows filled by forward-fill from a sensor outage.
    # Forward-filled flatlines corrupt training sequences in week 11.
    _gap_start_str = config.get('lstm_autoencoder', 'dataset_gap_exclude_start', fallback='').strip()
    _gap_end_str   = config.get('lstm_autoencoder', 'dataset_gap_exclude_end', fallback='').strip()
    removed_info = {}
    if _gap_start_str and _gap_end_str:
        christmas_start = pd.Timestamp(_gap_start_str)
        christmas_end   = pd.Timestamp(_gap_end_str)
        christmas_mask  = (df_features.index >= christmas_start) & (df_features.index <= christmas_end)
        if christmas_mask.any():
            removed_info['christmas'] = df_features.index[christmas_mask].copy()
            df_features = df_features[~christmas_mask].copy()
            print(f"   🚫 Christmas/New Year gap excluded: {christmas_mask.sum()} rows removed "
                  f"({christmas_start.date()} – {christmas_end.date()})")
        else:
            removed_info['christmas'] = pd.DatetimeIndex([])
    else:
        removed_info['christmas'] = pd.DatetimeIndex([])

    # Detect interpolated periods on RAW data before any scaling
    if config.getboolean('lstm_autoencoder', 'exclude_raw_flatline_periods', fallback=True):
        flat_window = config.getint('lstm_autoencoder', 'raw_flatline_window', fallback=12)
        flat_std_threshold = config.getfloat('lstm_autoencoder', 'raw_flatline_std_threshold', fallback=0.05)
        flat_min_samples = config.getint('lstm_autoencoder', 'raw_flatline_min_samples', fallback=24)

        print(f"\n   🔍 DETECTING INTERPOLATED GAPS (raw data):")
        flatline_mask = pd.Series(False, index=df_features.index)

        for col in df_features.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['set point', 'setpoint', 'sp ', 'compensata', 'modul', 'ventilat', 'fan']):
                continue

            raw_rolling_std = df_features[col].rolling(window=flat_window, center=True, min_periods=flat_window).std()
            col_flat = (raw_rolling_std < flat_std_threshold).fillna(False)
            n_flat = int(col_flat.sum())

            if n_flat > flat_min_samples:
                print(f"      ⚠️  {col}: {n_flat} flat samples")
                flatline_mask |= col_flat

        removed_info['flatline'] = df_features.index[flatline_mask].copy()
        n_before = len(df_features)
        df_features = df_features[~flatline_mask].copy()
        n_removed = n_before - len(df_features)

        if n_removed > 0:
            pct_removed = (n_removed / n_before * 100) if n_before > 0 else 0.0
            print(f"      Removed {n_removed:,} interpolated samples ({pct_removed:.1f}%)")
        else:
            print(f"      No significant raw flatline periods detected")
        print(f"      Remaining: {len(df_features):,} samples of real data")
    else:
        removed_info['flatline'] = pd.DatetimeIndex([])
    
    # Check for missing values
    nan_count = df_features.isna().sum().sum()
    if nan_count > 0:
        print(f"\n   ⚠️  WARNING: Found {nan_count} missing values")
        print(f"   Missing values per feature:")
        for col in df_features.columns:
            missing = df_features[col].isna().sum()
            if missing > 0:
                print(f"      • {col}: {missing} ({missing/len(df_features)*100:.2f}%)")
        
        # Apply LIMITED forward/backward fill to respect the "keep long gaps as NaN"
        # policy from the interpolation step.  Unlimited ffill() would silently fill
        # multi-day gaps that were intentionally left as NaN, producing fake flat-line
        # segments and hiding real data-quality periods from the sequence builder.
        #
        # Limit = max_interpolation_gap_hours expressed in number of samples.
        # Any gap longer than this stays as NaN and will be skipped by create_sequences.
        _gap_h  = config.getfloat('lstm_autoencoder', 'max_interpolation_gap_hours', fallback=4.0)
        _sr_min = config.getint('lstm_autoencoder', 'sampling_interval_minutes', fallback=30)
        _fill_limit = max(1, int(_gap_h * 60 / _sr_min))
        print(f"\n   🔧 Applying limited fill (limit={_fill_limit} steps = {_gap_h:.0f} h) to short gaps...")
        df_features = df_features.ffill(limit=_fill_limit).bfill(limit=_fill_limit)
        
        # Report how many NaNs remain (long gaps — intentionally preserved)
        remaining_nan = df_features.isna().sum().sum()
        if remaining_nan > 0:
            print(f"   ℹ️  {remaining_nan} NaN(s) remain after limited fill (gap > {_gap_h:.0f} h — correctly preserved)")
            print(f"      These will be skipped by the sequence builder (no data leakage).")
    
    # ── (C) Deadband on setpoint_error ──────────────────────────────────────
    # Zero out |e| <= deadband to suppress noise-driven false alarms (Salsbury)
    sp_deadband = config.getfloat('lstm_autoencoder', 'setpoint_error_deadband_degC', fallback=0.0)
    if sp_deadband > 0.0 and 'setpoint_error' in df_features.columns:
        before_zeros = (df_features['setpoint_error'] == 0.0).sum()
        df_features['setpoint_error'] = df_features['setpoint_error'].where(
            df_features['setpoint_error'].abs() > sp_deadband, 0.0
        )
        new_zeros = (df_features['setpoint_error'] == 0.0).sum() - before_zeros
        print(f"   (C) Deadband ±{sp_deadband}°C on setpoint_error: {new_zeros:,} new zeros ({new_zeros/len(df_features)*100:.1f}%)")

    # ── (D) On-the-fly differencing features ────────────────────────────────
    # Supports dT_mandata and dMod_mandata if they appear in feature_columns
    # but were not created by the feature-engineering script (e.g., older CSV).
    diff_sources = {
        'dT_mandata':   'Temperatura Mandata',
        'dMod_mandata': 'Modulazione Ventilatore Mandata',
    }
    for feat_name, source_col in diff_sources.items():
        if feat_name in feature_columns and feat_name not in df_features.columns:
            if source_col in df_features.columns:
                df_features[feat_name] = df_features[source_col].diff().fillna(0.0)
            elif source_col in df.columns:
                src = df.loc[df_features.index, source_col]
                df_features[feat_name] = src.diff().fillna(0.0)
            else:
                raise ValueError(
                    f"Column '{source_col}' not found for on-the-fly '{feat_name}'. "
                    "Re-run feature engineering (12.feature_engineering.py) first."
                )
            print(f"   (D) Computed {feat_name} on-the-fly = \u0394{source_col}")

    # ── Gap characterization report ─────────────────────────────────────────
    # Also include NaN rows that were preserved by the limited fill step.
    # These are long gaps (> max_gap_hours) intentionally kept as NaN so the
    # sequence builder skips them.  Without this step they were invisible to the
    # report (removed_info only tracks rows that were *removed* from the index).
    nan_mask = df_features.isna().any(axis=1)
    if nan_mask.sum() > 0:
        removed_info['nan_preserved'] = pd.DatetimeIndex(df_features.index[nan_mask])
    sampling_minutes_cfg = config.getint('lstm_autoencoder', 'sampling_interval_minutes', fallback=30)
    gap_info = characterize_gaps(removed_info, df_features, sampling_minutes_cfg)

    print(f"\n   ✓ Loaded successfully!")
    print(f"   • Total samples: {len(df_features):,}")
    print(f"   • Date range: {df_features.index.min()} → {df_features.index.max()}")
    print(f"   • Features: {len(feature_columns)}")
    print(f"   • Sampling interval: {pd.infer_freq(df_features.index) or 'irregular'}")
    print(f"{'='*70}\n")
    
    return {
        'df': df_features,
        'features': feature_columns,
        'n_features': len(feature_columns),
        'date_range': (df_features.index.min(), df_features.index.max()),
        'gap_info': gap_info
    }


def characterize_gaps(removed_info, df_clean, sampling_minutes=30):
    """
    Characterize data gaps removed before LSTM processing.

    Reports: total removal %, max consecutive run, number of distinct gap
    episodes, and a per-episode table (start, end, duration, type).

    Args:
        removed_info     : dict {label: pd.DatetimeIndex} of removed row groups.
                           Keys are meaningful labels, e.g. 'christmas', 'flatline'.
        df_clean         : The cleaned DataFrame after all removals — used to
                           compute the total timeline denominator.
        sampling_minutes : Expected sampling interval in minutes (default 30).

    Returns:
        dict with:
        - 'removed_timestamps' : union of all removed timestamps (pd.DatetimeIndex)
        - 'gap_rate_pct'       : % of total timeline rows that were removed
        - 'n_gap_episodes'     : number of distinct contiguous removed blocks
        - 'max_gap_hours'      : duration of the longest single gap in hours
        - 'mean_gap_hours'     : average gap episode duration in hours
        - 'clustering_ratio'   : max_gap_hours / mean_gap_hours
                                 (1 = uniform spacing; >1 = concentrated in one block)
        - 'episodes'           : list of per-episode dicts (start, end, duration_h, type)
    """
    print(f"\n{'='*70}")
    print(f"📊 GAP CHARACTERIZATION REPORT")
    print(f"{'='*70}")

    # Union of all removed timestamps
    all_removed = pd.DatetimeIndex([])
    for label, idx in removed_info.items():
        all_removed = all_removed.union(idx)
    all_removed = all_removed.sort_values()

    n_removed = len(all_removed)
    n_total   = n_removed + len(df_clean)
    gap_rate  = (n_removed / n_total * 100) if n_total > 0 else 0.0

    print(f"   • Rows removed   : {n_removed:,} / {n_total:,}  ({gap_rate:.2f}%)")
    for label, idx in removed_info.items():
        pct = len(idx) / n_total * 100 if n_total > 0 else 0.0
        print(f"     – {label:<15s}: {len(idx):>5,} rows  ({pct:.2f}%)")

    # Detect distinct episodes (gap between consecutive removed timestamps
    # > 1.5× the expected interval = new episode)
    episodes        = []
    max_gap_hours   = 0.0
    total_gap_hours = 0.0

    if n_removed > 0:
        step_ns    = np.timedelta64(int(sampling_minutes * 1.5 * 60 * 1e9), 'ns')
        removed_np = all_removed.to_numpy()
        ep_start   = removed_np[0]
        ep_end     = removed_np[0]

        # Map removed timestamp → source label (first match wins)
        label_map = {}
        for label, idx in removed_info.items():
            for ts in idx:
                if ts not in label_map:
                    label_map[ts] = label

        for ts in removed_np[1:]:
            if ts - ep_end <= step_ns:
                ep_end = ts
            else:
                dur_h = (ep_end - ep_start) / np.timedelta64(1, 'h')
                episodes.append({
                    'start'      : pd.Timestamp(ep_start),
                    'end'        : pd.Timestamp(ep_end),
                    'duration_h' : round(dur_h, 2),
                    'type'       : label_map.get(pd.Timestamp(ep_start), 'unknown')
                })
                max_gap_hours   = max(max_gap_hours, dur_h)
                total_gap_hours += dur_h
                ep_start = ts
                ep_end   = ts

        # Flush last episode
        dur_h = (ep_end - ep_start) / np.timedelta64(1, 'h')
        episodes.append({
            'start'      : pd.Timestamp(ep_start),
            'end'        : pd.Timestamp(ep_end),
            'duration_h' : round(dur_h, 2),
            'type'       : label_map.get(pd.Timestamp(ep_start), 'unknown')
        })
        max_gap_hours   = max(max_gap_hours, dur_h)
        total_gap_hours += dur_h

    n_ep             = len(episodes)
    mean_gap_hours   = total_gap_hours / n_ep if n_ep > 0 else 0.0
    clustering_ratio = max_gap_hours / mean_gap_hours if mean_gap_hours > 0 else 1.0

    print(f"\n   • Distinct gap episodes  : {n_ep}")
    print(f"   • Longest gap            : {max_gap_hours:.1f} h")
    print(f"   • Mean gap duration      : {mean_gap_hours:.1f} h")
    print(f"   • Clustering ratio       : {clustering_ratio:.2f}  "
          f"(1 = uniform  │ >1 = concentrated in one block)")
    if n_ep > 0:
        print(f"\n   {'Start':<20s}  {'End':<20s}  {'Hours':>6s}  Type")
        print(f"   {'-'*20}  {'-'*20}  {'-'*6}  {'-'*12}")
        for ep in episodes:
            print(f"   {str(ep['start']):<20s}  {str(ep['end']):<20s}  "
                  f"{ep['duration_h']:>6.1f}  {ep['type']}")

    print(f"{'='*70}\n")

    return {
        'removed_timestamps' : all_removed,
        'gap_rate_pct'       : round(gap_rate, 4),
        'n_gap_episodes'     : n_ep,
        'max_gap_hours'      : round(max_gap_hours, 2),
        'mean_gap_hours'     : round(mean_gap_hours, 2),
        'clustering_ratio'   : round(clustering_ratio, 2),
        'episodes'           : episodes
    }


def create_sequences(data, sequence_length):
    """
    Convert dataframe to sequences for LSTM using sliding window
    
    🎬 WINDOWING EXPLANATION:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    This creates "movie clips" from continuous sensor readings.
    
    Example with sequence_length=24 (12 hours):
    • Sample 1: Timesteps [0-23]   → Monday 00:00 - 11:30
    • Sample 2: Timesteps [1-24]   → Monday 00:30 - 12:00
    • Sample 3: Timesteps [2-25]   → Monday 01:00 - 12:30
    
    Each "clip" captures temporal patterns:
    - Daily cycles (morning warmup, afternoon peak, night cooldown)
    - Correlations between variables over time
    - Sequential dependencies the LSTM learns
    
    Args:
        data: pandas DataFrame or numpy array (T timesteps, n_features)
        sequence_length: number of timesteps per sequence
        
    Returns:
        X: shape (n_samples, sequence_length, n_features)
           3D tensor ready for LSTM input
    """
    
    # Convert to numpy if pandas DataFrame
    if isinstance(data, pd.DataFrame):
        values = data.values
    else:
        values = data
    
    X = []
    
    # Sliding window: move one step at a time
    for i in range(len(values) - sequence_length + 1):
        sequence = values[i:i + sequence_length]
        X.append(sequence)
    
    X = np.array(X)
    
    print(f"\n   📊 SEQUENCE CREATION:")
    print(f"   • Original data: {values.shape[0]} timesteps × {values.shape[1]} features")
    print(f"   • Sequence length: {sequence_length} timesteps")
    print(f"   • Created sequences: {X.shape[0]}")
    print(f"   • Output shape: {X.shape} (samples, seq_len, features)")
    
    return X


def create_sequences_from_blocks(df, seq_length, sampling_minutes=30, clean_series=None):
    """
    Create sliding-window sequences from a regular-grid DataFrame, skipping
    any window that contains a NaN OR spans a time gap > 1.5× the expected
    sampling interval.

    APPROACH (Method A + time-continuity):
    ──────────────────────────────────────
    • The full regular 30-min timeline is preserved (no rows deleted).
    • A prefix-sum over the NaN indicator gives O(1) per-window NaN detection.
    • A precomputed delta array flags time jumps (e.g. after Christmas gap
      removal) so no window ever bridges two non-contiguous blocks.

    Args:
        df              : Scaled DataFrame with a regular DatetimeIndex.
                          May contain NaNs; those windows are skipped.
        seq_length      : Timesteps per sequence (e.g. 48 = 24 h at 30 min).
        sampling_minutes: Expected sampling interval in minutes (default 30).
        clean_series    : Optional boolean pd.Series aligned to df.index.
                          When supplied, returns a per-sequence boolean mask
                          (True ↔ every row in the window is marked clean).

    Returns (always a 3-tuple):
        X          : np.ndarray  (n_seq, seq_length, n_features)  float32
        timestamps : pd.DatetimeIndex  — end-timestamp of each sequence
        clean_mask : np.ndarray of bool, or None if clean_series is None
    """
    n          = len(df)
    n_features = df.shape[1]
    data       = df.values.astype(np.float64)
    times      = df.index.to_numpy()  # numpy datetime64

    if n < seq_length:
        X  = np.empty((0, seq_length, n_features), dtype=np.float32)
        ts = pd.DatetimeIndex([])
        print(f"   ⚠️  Not enough rows ({n}) for seq_length={seq_length}")
        return X, ts, None

    # ── NaN prefix sum ─────────────────────────────────────────────
    # has_nan[i] = 1 if row i contains any NaN, else 0
    has_nan = np.isnan(data).any(axis=1).astype(np.int32)
    csum    = np.empty(n + 1, dtype=np.int32)
    csum[0] = 0
    np.cumsum(has_nan, out=csum[1:])

    # ── Time-gap flag ───────────────────────────────────────────────
    # gap_after[i] = True  →  gap between times[i] and times[i+1] is too large
    max_ns  = np.timedelta64(int(sampling_minutes * 1.5 * 60 * 1e9), 'ns')
    deltas  = np.diff(times)                          # length n-1
    gap_after = deltas > max_ns                       # True where a gap starts

    # Prefix sum of gap_after so we can check any sub-range in O(1)
    gsum     = np.empty(n, dtype=np.int32)
    gsum[0]  = 0
    np.cumsum(gap_after, out=gsum[1:])

    # ── Sliding window ──────────────────────────────────────────────
    all_sequences  = []
    all_timestamps = []
    all_clean      = []

    n_nan_skip = 0
    n_gap_skip = 0

    for end in range(seq_length - 1, n):
        start = end - seq_length + 1

        # NaN check: any NaN in [start, end]?
        if csum[end + 1] - csum[start] > 0:
            n_nan_skip += 1
            continue

        # Time-gap check: any large gap in [start, end-1]?
        # gsum[end] - gsum[start] counts gaps between times[start..end]
        if gsum[end] - gsum[start] > 0:
            n_gap_skip += 1
            continue

        all_sequences.append(data[start:end + 1])
        all_timestamps.append(times[end])

        if clean_series is not None:
            window_idx   = df.index[start:end + 1]
            window_clean = clean_series.reindex(window_idx).fillna(False).values
            all_clean.append(bool(window_clean.all()))

    if not all_sequences:
        X  = np.empty((0, seq_length, n_features), dtype=np.float32)
        ts = pd.DatetimeIndex([])
        print(f"   ⚠️  No valid sequences (all {n - seq_length + 1} windows skipped)")
        return X, ts, None

    X  = np.array(all_sequences, dtype=np.float32)
    ts = pd.DatetimeIndex(all_timestamps)

    total_windows = n - seq_length + 1
    print(f"   • Windows evaluated: {total_windows:>5d}  "
          f"│ NaN-skip: {n_nan_skip}  │ gap-skip: {n_gap_skip}  "
          f"│ Valid: {len(X)}")

    clean_out = np.array(all_clean, dtype=bool) if clean_series is not None else None
    return X, ts, clean_out



# ═════════════════════════════════════════════════════════════════════════════
# TEST SCRIPT - Run this file directly to test data loading
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import configparser
    
    print("\n🧪 TESTING DATA_LOADER.PY\n")
    
    # Load configuration
    config = configparser.ConfigParser()
    # Use main config.ini from project root (2 levels up)
    config_path = Path(__file__).parent.parent.parent / 'config.ini'
    config.read(config_path, encoding='utf-8-sig')
    
    # Test load_and_prepare_data()
    try:
        data_dict = load_and_prepare_data(config)
        print(f"✅ Data loaded successfully!")
        print(f"   DataFrame shape: {data_dict['df'].shape}")
        print(f"   Features: {data_dict['features']}")
        print(f"\n   First 5 rows:")
        print(data_dict['df'].head())
        
        # Test create_sequences()
        seq_length = config.getint('lstm_autoencoder', 'sequence_length')
        sequences = create_sequences(data_dict['df'], seq_length)
        print(f"\n✅ Sequences created successfully!")
        print(f"   Shape: {sequences.shape}")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
