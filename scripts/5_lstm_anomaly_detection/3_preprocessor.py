"""
PREPROCESSOR - Scale, split, and prepare data for LSTM training
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Purpose:
    1. Apply feature scaling (MinMaxScaler or StandardScaler)
    2. Split data chronologically (train/val/test)
    3. Create sequences for each split
    4. Apply clean training filters (optional)

CRITICAL: 
    - Fit scaler ONLY on training data (prevent data leakage!)
    - No shuffling! Time series must stay in order
    - Filter bad segments from training only (keep all test data)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import importlib
data_loader = importlib.import_module('2_data_loader')


def preprocess_data(data_dict, config):
    """
    Scale features, split data chronologically, create sequences
    
    🔧 IMPROVEMENT: Sequences are created on FULL scaled timeline, then split
    This ensures validation/test sequences have proper "history" from training period
    
    Args:
        data_dict: Output from load_and_prepare_data()
        config: ConfigParser object
        
    Returns:
        dict with:
        - 'X_train': training sequences (n_samples, seq_len, n_features)
        - 'X_val': validation sequences
        - 'X_test': test sequences
        - 'scaler': fitted scaler object (for inverse transform)
        - 'input_shape': tuple (seq_len, n_features)
        - 'train_indices': indices of training data in original df
        - 'val_indices': indices of validation data
        - 'test_indices': indices of test data
        - 'clean_mask': boolean mask for clean training samples
    """
    
    print(f"\n{'='*70}")
    print(f"⚙️  PREPROCESSING DATA (IMPROVED SEQUENCING)")
    print(f"{'='*70}")
    
    # Extract data
    df = data_dict['df']
    n_features = data_dict['n_features']
    
    # Get parameters from config (main config.ini structure)
    # Get sequence length from lstm_autoencoder section
    seq_length = config.getint('lstm_autoencoder', 'sequence_length')
    sampling_minutes = config.getint('lstm_autoencoder', 'sampling_interval_minutes', fallback=30)
    
    # Scaler type
    scaler_type = config.get('lstm_autoencoder', 'scaler_type', fallback='robust')
    
    print(f"   Scaler: {scaler_type.upper()}")
    print(f"   Sequence length: {seq_length} timesteps")

    # ═══════════════════════════════════════════════════════════════
    # STEP 0: OPTIONAL DATASET END-DATE CAP (regime-shift guard)
    # ═══════════════════════════════════════════════════════════════
    # Some seasons have a shoulder-season tail that is out-of-distribution
    # relative to the training regime (e.g. Summer 2025: Oct outdoor temp
    # drops from ~30°C to ~12°C, collapses cooling load → 5× higher test error).
    # Setting dataset_end_date in config caps the data BEFORE splitting so the
    # test set stays within the same thermal regime as training.
    active_season = config.get('global', 'season', fallback='').strip().lower()
    # Season-specific key takes precedence (dataset_end_date_summer / dataset_end_date_winter)
    end_date_str = config.get('lstm_autoencoder', f'dataset_end_date_{active_season}', fallback='').strip()
    if not end_date_str:  # fall back to generic key for backwards compatibility
        end_date_str = config.get('lstm_autoencoder', 'dataset_end_date', fallback='').strip()
    # Resolve ${global:year} placeholder (configparser BasicInterpolation doesn't handle cross-section refs)
    if end_date_str and '${global:year}' in end_date_str:
        year = config.get('global', 'year', fallback='').strip()
        end_date_str = end_date_str.replace('${global:year}', year)
    if end_date_str:
        end_date = pd.Timestamp(end_date_str)
        n_before = len(df)
        df = df[df.index <= end_date]
        n_after = len(df)
        print(f"\n   ✂️  DATASET CAPPED at {end_date_str}: "
              f"{n_before:,} → {n_after:,} rows "
              f"({n_before - n_after:,} shoulder-season rows excluded)")
    else:
        print(f"   ℹ️  dataset_end_date not set — using full dataset")

    # ═══════════════════════════════════════════════════════════════
    # STEP 1: DATA SPLIT (chronological, contiguous weeks)
    # ═══════════════════════════════════════════════════════════════
    # LSTM training/evaluation requires strictly contiguous time blocks.
    # This pipeline always uses chronological week-based splitting:
    # first train weeks, then validation weeks, then test weeks.

    df = df.copy()
    df['_week'] = ((df.index - df.index.min()).days // 7).astype(int)
    week_numbers = sorted(df['_week'].unique())
    n_weeks = len(week_numbers)

    train_frac = config.getfloat('lstm_autoencoder', 'train_split', fallback=0.70)
    val_frac   = config.getfloat('lstm_autoencoder', 'val_split',   fallback=0.15)
    n_train = max(1, int(n_weeks * train_frac))
    n_val   = max(1, int(n_weeks * val_frac))
    train_weeks = set(week_numbers[:n_train])
    val_weeks   = set(week_numbers[n_train:n_train + n_val])
    test_weeks  = set(week_numbers[n_train + n_val:])
    print(f"\n   📊 CHRONOLOGICAL SPLIT ({int(train_frac*100)}/{int(val_frac*100)}/{100-int(train_frac*100)-int(val_frac*100)} by week):")

    df_train_with_week = df[df['_week'].isin(train_weeks)].copy()
    df_val_with_week   = df[df['_week'].isin(val_weeks)].copy()
    df_test_with_week  = df[df['_week'].isin(test_weeks)].copy()

    df_train = df_train_with_week.drop('_week', axis=1)
    df_val   = df_val_with_week.drop('_week', axis=1)
    df_test  = df_test_with_week.drop('_week', axis=1)
    df = df.drop('_week', axis=1)

    n_total = len(df)
    print(f"   • Training:   {len(df_train):5d} rows ({len(df_train)/n_total*100:.1f}%) — {len(train_weeks)} weeks")
    print(f"     Date range: {df_train.index.min()} → {df_train.index.max()}")
    print(f"   • Validation: {len(df_val):5d} rows ({len(df_val)/n_total*100:.1f}%) — {len(val_weeks)} weeks")
    print(f"     Date range: {df_val.index.min()} → {df_val.index.max()}")
    print(f"   • Test:       {len(df_test):5d} rows ({len(df_test)/n_total*100:.1f}%) — {len(test_weeks)} weeks")
    print(f"     Date range: {df_test.index.min()} → {df_test.index.max()}")

    # ═══════════════════════════════════════════════════════════════
    # STEP 1.5: (B) OPTIONAL EWMA SMOOTHING ON SENSOR CHANNELS
    # ═══════════════════════════════════════════════════════════════
    # Applied per-split before scaling to avoid lookahead (Schein 2006).
    # Smooths continuous sensor channels; NOT applied to setpoints / modulations.
    if config.getboolean('lstm_autoencoder', 'apply_ewma_smoothing', fallback=False):
        ewma_span     = config.getint('lstm_autoencoder', 'ewma_span', fallback=3)
        ewma_cols_str = config.get('lstm_autoencoder', 'ewma_sensor_columns', fallback='')
        ewma_cols     = [c.strip() for c in ewma_cols_str.split(',') if c.strip()]

        def _apply_ewma(df_split, label):
            active = [c for c in ewma_cols if c in df_split.columns]
            if not active:
                return df_split
            df_split = df_split.copy()
            for col in active:
                df_split[col] = df_split[col].ewm(span=ewma_span, adjust=False).mean()
            print(f"      {label}: EWMA span={ewma_span} \u2192 {active}")
            return df_split

        print(f"\n   📈 STEP 1.5 \u2014 (B) EWMA SMOOTHING on sensor channels (span={ewma_span}):")
        df_train = _apply_ewma(df_train, 'Train')
        df_val   = _apply_ewma(df_val,   'Val')
        df_test  = _apply_ewma(df_test,  'Test')

    # ═══════════════════════════════════════════════════════════════
    # STEP 2: FEATURE SCALING (fit on training only!)
    # ═══════════════════════════════════════════════════════════════
    # 🔑 CRITICAL: Fit scaler ONLY on training data!
    # If we fit on all data, future information leaks into training
    
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'robust':
        # RobustScaler: Better for HVAC data with sensor spikes/outliers
        # Uses median & IQR instead of mean & std
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaler_type: {scaler_type}. Use 'minmax', 'standard', or 'robust'")
    
    # Fit scaler on training data only
    scaler.fit(df_train.values)
    
    print(f"\n   🔧 SCALING APPLIED:")
    print(f"   • Scaler fitted on training data only ({len(df_train)} rows)")
    # ═══════════════════════════════════════════════════════════════
    # VARIANCE CHECK & AUTO-DROP (Papers 16 & 17 implementation)
    # ═══════════════════════════════════════════════════════════════
    # If a feature has near-zero post-scaling variance it is a flatline
    # (e.g. fan locked at 80% all winter, frozen sensor, stuck valve).
    # Feeding it to the AE collapses the RobustScaler (IQR=0), corrupts
    # gradient updates, and inflates the threshold — exactly the "80% fan"
    # bug that caused the Winter model to produce 0% anomalies.
    # Fix: auto-drop the column from ALL splits and re-fit the scaler.
    train_scaled_vars = pd.DataFrame(
        scaler.transform(df_train.values),
        columns=df_train.columns
    ).var()

    print(f"\n   \U0001f50d FEATURE VARIANCE CHECK (post-scaling, training data):")
    zero_var_cols = []
    for col, var in train_scaled_vars.items():
        # NaN variance means RobustScaler hit IQR=0 (constant feature) → 0/0=NaN.
        # Both NaN and near-zero variance must trigger the auto-drop.
        is_flatline = pd.isna(var) or var < 0.01
        if is_flatline:
            flag = "\u26a0\ufe0f  NEAR-ZERO (AUTO-DROP)"
            zero_var_cols.append(col)
        else:
            flag = "\u2713"
        print(f"      {flag} {col}: var = {var:.6f}")

    if zero_var_cols:
        print(f"\n   \U0001f9f9 AUTO-DROPPING FLATLINE FEATURES (Papers 16/17):")
        for col in zero_var_cols:
            print(f"      - {col}  (prevents scaler collapse and inflated threshold)")

        # Drop from all three splits so shapes stay consistent
        df_train = df_train.drop(columns=zero_var_cols)
        df_val   = df_val.drop(columns=zero_var_cols)
        df_test  = df_test.drop(columns=zero_var_cols)

        # Keep data_dict and n_features in sync — model builder reads these
        data_dict['features']  = [f for f in data_dict['features'] if f not in zero_var_cols]
        data_dict['n_features'] = len(data_dict['features'])
        n_features = data_dict['n_features']

        # CRITICAL: re-fit scaler on the trimmed training set
        # (the old scaler still has statistics for the dropped columns)
        scaler.fit(df_train.values)
        print(f"      ✓  Scaler re-fitted on {n_features} remaining feature(s): "
              f"{data_dict['features']}")

    # ═══════════════════════════════════════════════════════════════
    # STEP 3: CREATE CLEAN TRAINING MASK
    # ═══════════════════════════════════════════════════════════════
    clean_mask = create_clean_training_mask(df_train, config)
    # Apply the same cleaning logic to validation for fair checkpoint selection
    # during training. Test split remains untouched by design.
    clean_mask_val = create_clean_training_mask(df_val, config)
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 4: SCALE EACH SPLIT USING SCALER FIT ON TRAIN
    # ═══════════════════════════════════════════════════════════════
    train_scaled = scaler.transform(df_train.values)
    val_scaled   = scaler.transform(df_val.values)
    test_scaled  = scaler.transform(df_test.values)

    # Plain DataFrames — no _week column needed (gap detection handles boundaries)
    df_train_scaled = pd.DataFrame(train_scaled, index=df_train.index, columns=df_train.columns)
    df_val_scaled   = pd.DataFrame(val_scaled,   index=df_val.index,   columns=df_val.columns)
    df_test_scaled  = pd.DataFrame(test_scaled,  index=df_test.index,  columns=df_test.columns)

    # ═══════════════════════════════════════════════════════════════
    # STEP 5: CREATE SEQUENCES FROM CONTIGUOUS BLOCKS
    # ═══════════════════════════════════════════════════════════════
    # Uses get_contiguous_blocks → create_sequences_from_blocks (data_loader).
    # Sequences are only created within each uninterrupted block, so no window
    # ever spans a NaN gap, a time jump, or the Christmas exclusion boundary.
    print(f"\n   🎬 CREATING SEQUENCES FROM CONTIGUOUS BLOCKS (gap-safe):")

    clean_mask_series = pd.Series(clean_mask, index=df_train.index) if clean_mask is not None else None
    clean_mask_val_series = pd.Series(clean_mask_val, index=df_val.index) if clean_mask_val is not None else None

    print(f"\n   ── Train ──")
    X_train, train_indices, clean_sequence_mask = data_loader.create_sequences_from_blocks(
        df_train_scaled, seq_length, sampling_minutes, clean_series=clean_mask_series
    )
    print(f"\n   ── Validation ──")
    X_val, val_indices, val_clean_sequence_mask = data_loader.create_sequences_from_blocks(
        df_val_scaled, seq_length, sampling_minutes, clean_series=clean_mask_val_series
    )
    print(f"\n   ── Test ──")
    X_test, test_indices, _ = data_loader.create_sequences_from_blocks(
        df_test_scaled, seq_length, sampling_minutes
    )
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 5.5: (E) MAD-BASED EXTREME WINDOW FILTER (training only)
    # ═══════════════════════════════════════════════════════════════
    # Removes top mad_exclusion_pct% of training windows by max per-feature
    # robust z-score (MAD). Implements the Wei (2019) "clean normal" principle.
    # Does NOT touch val / test sequences.
    if config.getboolean('lstm_autoencoder', 'apply_mad_training_filter', fallback=False):
        excl_pct = config.getfloat('lstm_autoencoder', 'mad_exclusion_pct', fallback=2.0)
        if len(X_train) > 0:
            # Per-feature MAD z-score averaged across the sequence
            # X_train shape: (n_windows, seq_len, n_features)
            window_means = X_train.mean(axis=1)                          # (n, n_features)
            feat_median  = np.median(window_means, axis=0)               # (n_features,)
            feat_mad     = np.median(np.abs(window_means - feat_median), axis=0) + 1e-8
            zscore_mad   = np.abs(window_means - feat_median) / feat_mad # (n, n_features)
            max_z        = zscore_mad.max(axis=1)                        # (n,) worst feature

            cutoff = np.percentile(max_z, 100.0 - excl_pct)
            extreme_mask = max_z > cutoff
            n_extreme = extreme_mask.sum()

            if n_extreme > 0:
                print(f"\n   🧹 STEP 5.5 \u2014 (E) MAD EXTREME WINDOW FILTER:")
                print(f"      Removed {n_extreme:,} training windows ({n_extreme/len(X_train)*100:.1f}%) "
                      f"above MAD z-score cutoff (top {excl_pct}%)")
                # Update clean_sequence_mask — first create if still None
                if clean_sequence_mask is None:
                    clean_sequence_mask = np.ones(len(X_train), dtype=bool)
                clean_sequence_mask &= ~extreme_mask

    # ═══════════════════════════════════════════════════════════════
    # STEP 6: APPLY CLEAN MASK TO TRAINING SEQUENCES
    # ═══════════════════════════════════════════════════════════════
    if clean_sequence_mask is None:
        clean_sequence_mask = np.ones(len(X_train), dtype=bool)

    clean_pct = (clean_sequence_mask.sum() / len(X_train) * 100) if len(X_train) > 0 else 0.0

    print(f"\n   ✅ PREPROCESSING COMPLETE!")
    print(f"   • Train sequences: {X_train.shape[0]} (clean: {clean_sequence_mask.sum()}, {clean_pct:.1f}%)")
    if val_clean_sequence_mask is not None and len(X_val) > 0:
        val_clean_pct = val_clean_sequence_mask.sum() / len(X_val) * 100
        print(f"   • Val sequences:   {X_val.shape[0]} (clean: {val_clean_sequence_mask.sum()}, {val_clean_pct:.1f}%)")
    else:
        print(f"   • Val sequences:   {X_val.shape[0]}")
    print(f"   • Test sequences:  {X_test.shape[0]}")
    print(f"   • Input shape: (seq_len={seq_length}, features={n_features})")
    print(f"   • Split strategy: CHRONOLOGICAL {int(train_frac*100)}/{int(val_frac*100)}/{100-int(train_frac*100)-int(val_frac*100)}")
    print(f"{'='*70}\n")
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'scaler': scaler,
        'input_shape': (seq_length, n_features),
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices,
        'clean_sequence_mask': clean_sequence_mask,
        'val_clean_sequence_mask': val_clean_sequence_mask,
        'df_train': df_train,
        'df_val': df_val,
        'df_test': df_test,
        'df_full': df,  # Keep full dataframe for reference
        'gap_info': data_dict.get('gap_info', {})  # Forward gap info for FP diagnostics
    }


def create_clean_training_mask(df_train, config):
    """
    Create boolean mask identifying "clean" data suitable for training
    
    🔑 CRITICAL PRINCIPLE: "Train on mostly normal data"
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Autoencoders learn what they see. If training data contains:
    - Inactive periods (fan off, no data generation)
    - Heavy interpolation (artificial data, not real sensor readings)
    - Outliers/possible faults (broken sensors, physical impossibilities)
    
    → Model learns ABNORMAL behavior as baseline
    → Thresholding becomes weak (everything looks "normal")
    
    IMPROVED FILTERS:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    1. Active operation only (fan > 0 or modulation > threshold)
    2. Exclude heavily interpolated periods
    3. Exclude out-of-physical-range values
    4. Exclude high variance (sensor glitches)
    
    Args:
        df_train: Training dataframe
        config: ConfigParser object
        
    Returns:
        clean_mask: Boolean array (True = clean, False = exclude from training)
    """
    
    def _uses_fractional_modulation(series):
        """
        Heuristic scale detector for fan/modulation channels.
        Returns True when values are likely in [0, 1], False for [0, 100].
        """
        finite = series.dropna()
        if finite.empty:
            return False
        q99_abs = float(finite.abs().quantile(0.99))
        return q99_abs <= 1.5

    def _pct_to_series_units(series, threshold_pct):
        """
        Convert a percentage threshold (e.g. 5%) to the channel's native scale.
        """
        return threshold_pct / 100.0 if _uses_fractional_modulation(series) else threshold_pct

    # Check if filtering is enabled
    enable_filtering = config.getboolean('lstm_autoencoder', 'enable_clean_training_filters', fallback=True)
    if not enable_filtering:
        print(f"   ⚠️  Clean training filters DISABLED (using all data)")
        return None
    
    print(f"\n   🧹 APPLYING ENHANCED CLEAN TRAINING FILTERS:")
    
    # Start with all True (all data is clean)
    clean_mask = np.ones(len(df_train), dtype=bool)
    original_count = len(df_train)
    
    # ───────────────────────────────────────────────────────────────
    # FILTER 1: Active operation only (exclude inactive periods)
    # ───────────────────────────────────────────────────────────────
    if config.getboolean('lstm_autoencoder', 'require_active_operation', fallback=True):
        # Detect inactive periods using fan/ventilator columns.
        # CORRECT HVAC LOGIC: system is OFF only when ALL fan channels are below
        # threshold simultaneously.  Using OR here (old logic) was too aggressive —
        # it marked a row inactive if *any* single fan was low, even if another was
        # still running (e.g., supply fan ON, return fan transitioning).
        fan_columns = [col for col in df_train.columns
                       if 'fan' in col.lower() or 'ventilat' in col.lower()]

        if fan_columns:
            fan_active_threshold_pct = config.getfloat('lstm_autoencoder', 'fan_active_threshold_pct', fallback=5.0)
            # Build active_from_fans: True where ANY fan column is >= threshold (OR).
            # Threshold is interpreted as a percentage and converted to native channel units.
            active_from_fans = np.zeros(len(df_train), dtype=bool)
            fan_scales = set()
            for col in fan_columns:
                _is_fractional = _uses_fractional_modulation(df_train[col])
                fan_scales.add('0-1' if _is_fractional else '0-100')
                col_threshold = _pct_to_series_units(df_train[col], fan_active_threshold_pct)
                active_from_fans |= (df_train[col] >= col_threshold)
            # Inactive = no fan is running at all
            inactive_mask = ~active_from_fans
        else:
            # No fan columns found — keep all rows (cannot infer activity)
            inactive_mask = np.zeros(len(df_train), dtype=bool)

        # NOTE: Modulation columns (Modulazione Ventilatore *) are the same physical
        # signals as the fan columns above in Italian HVAC CSVs, so we intentionally
        # do NOT add a second modulation pass to avoid double-exclusion.
        # delta_t check removed: near-zero ΔT is valid at steady-state (no heat demand).

        excluded = inactive_mask.sum()
        if excluded > 0:
            scale_note = ', '.join(sorted(fan_scales)) if fan_columns else 'n/a'
            print(
                f"      • Inactive operation (all fans < {fan_active_threshold_pct:.1f}% "
                f"on native scales {scale_note}): Excluded {excluded} samples "
                f"({excluded/original_count*100:.1f}%)"
            )
            clean_mask &= ~inactive_mask
    
    # ───────────────────────────────────────────────────────────────
    # FILTER 2A: Exclude long flat-line / interpolated gaps from training
    # ───────────────────────────────────────────────────────────────
    # Detect rows in NaN-run blocks >= 48 h (96 samples @ 30 min).
    # This catches any multi-day gap that was filled with linear interpolation
    # and still made it into the scaled training array.
    # NOTE: threshold is 48 h (not max_interpolation_gap_hours=4 h) to avoid
    # excluding short but valid periods of steady-state operation.
    max_gap_hours = config.getfloat('lstm_autoencoder', 'max_interpolation_gap_hours', fallback=4.0)
    sampling_minutes = config.getint('lstm_autoencoder', 'sampling_interval_minutes', fallback=30)
    # Minimum flat-run length that triggers exclusion: 48 h regardless of config gap setting
    min_exclude_samples = max(int(48 * 60 / sampling_minutes), int(max_gap_hours * 60 / sampling_minutes))
    
    long_gap_mask = np.zeros(len(df_train), dtype=bool)
    for col in df_train.columns:
        if any(k in col.lower() for k in ['hour', 'day', 'month', 'occupied', 'is_', '_week']):
            continue
        window = min_exclude_samples
        rolling_std = df_train[col].rolling(window=window, min_periods=window, center=True).std()
        flat = (rolling_std < 0.001).values
        in_run, run_len, run_start = False, 0, 0
        for idx in range(len(flat)):
            if flat[idx]:
                if not in_run:
                    in_run, run_start = True, idx
                run_len += 1
            else:
                if in_run and run_len >= min_exclude_samples:
                    long_gap_mask[run_start:idx] = True
                in_run, run_len = False, 0
        if in_run and run_len >= min_exclude_samples:
            long_gap_mask[run_start:] = True
    
    excluded_gap = long_gap_mask.sum()
    if excluded_gap > 0:
        print(f"      • Long flat-line gaps (>={min_exclude_samples} samples / >=48h): Excluded {excluded_gap} samples ({excluded_gap/original_count*100:.1f}%)")
        clean_mask &= ~long_gap_mask
    else:
        print(f"      • Long flat-line gaps: None detected")
    
    # ───────────────────────────────────────────────────────────────
    # FILTER 2B: Exclude explicitly-marked interpolated periods
    # ───────────────────────────────────────────────────────────────
    if config.getboolean('lstm_autoencoder', 'exclude_interpolated_periods', fallback=False):
        # Look for interpolation marker columns (e.g., "TM_interpolated", "TR_interpolated")
        interp_columns = [col for col in df_train.columns if 'interpolat' in col.lower() or '_mask' in col.lower()]
        
        if interp_columns:
            interp_mask = np.zeros(len(df_train), dtype=bool)
            for col in interp_columns:
                # If column is boolean/binary, True/1 = interpolated
                interp_mask |= (df_train[col] > 0)
            
            excluded = interp_mask.sum()
            if excluded > 0:
                print(f"      • Interpolated data: Excluded {excluded} samples ({excluded/original_count*100:.1f}%)")
                clean_mask &= ~interp_mask
        else:
            # Alternative: detect interpolated data by flat-line patterns
            max_flat_length = config.getint('lstm_autoencoder', 'max_flat_line_samples', fallback=6)
            
            for col in df_train.columns:
                # Skip non-sensor columns
                if any(keyword in col.lower() for keyword in ['hour', 'day', 'month', 'occupied', 'is_']):
                    continue
                
                # Detect constant values (flat lines often indicate interpolation)
                rolling_std = df_train[col].rolling(window=max_flat_length, center=True).std()
                flat_mask = rolling_std < 0.001  # Nearly constant
                
                excluded_flat = flat_mask.sum()
                if excluded_flat > 100:  # Only report significant flat periods
                    print(f"      • {col}: Excluded {excluded_flat} flat-line samples (likely interpolated)")
                    clean_mask &= ~flat_mask
    
    # ───────────────────────────────────────────────────────────────
    # FILTER 3: Exclude out-of-physical-range values
    # ───────────────────────────────────────────────────────────────
    if config.getboolean('lstm_autoencoder', 'exclude_out_of_range', fallback=True):
        # Use physical thresholds from config if available
        if config.has_section('physical_thresholds'):
            for col in df_train.columns:
                try:
                    threshold_str = config.get('physical_thresholds', col)
                    min_val, max_val = [float(x.strip()) for x in threshold_str.split(',')]
                    
                    out_of_range = (df_train[col] < min_val) | (df_train[col] > max_val)
                    excluded = out_of_range.sum()
                    if excluded > 0:
                        print(f"      • {col}: Excluded {excluded} out-of-range points")
                        clean_mask &= ~out_of_range
                except:
                    pass
        else:
            # Fallback: use IQR-based outlier detection
            for col in df_train.columns:
                # Skip non-sensor columns
                if any(keyword in col.lower() for keyword in ['hour', 'day', 'month', 'occupied', 'is_']):
                    continue
                
                Q1 = df_train[col].quantile(0.25)
                Q3 = df_train[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Extreme outliers: beyond 3*IQR
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                outliers = (df_train[col] < lower_bound) | (df_train[col] > upper_bound)
                excluded = outliers.sum()
                if excluded > 0:
                    print(f"      • {col}: Excluded {excluded} extreme outliers (>3×IQR)")
                    clean_mask &= ~outliers
    
    # ───────────────────────────────────────────────────────────────
    # FILTER 4: Exclude high variance periods (sensor glitches)
    # ───────────────────────────────────────────────────────────────
    if config.getboolean('lstm_autoencoder', 'exclude_high_variance', fallback=False):
        variance_multiplier = config.getfloat('lstm_autoencoder', 'variance_threshold_multiplier', fallback=5.0)
        
        for col in df_train.columns:
            # Skip non-sensor columns
            if any(keyword in col.lower() for keyword in ['hour', 'day', 'month', 'occupied', 'is_']):
                continue
            
            # Calculate rolling std with 6-hour window
            rolling_std = df_train[col].rolling(window=12, center=True).std()
            global_std = df_train[col].std()
            
            # Flag points with excessive local variance
            high_variance = rolling_std > (variance_multiplier * global_std)
            excluded = high_variance.sum()
            if excluded > 50:  # Only report if significant
                print(f"      • {col}: Excluded {excluded} high-variance points (sensor glitches)")
                clean_mask &= ~high_variance
    
    # ───────────────────────────────────────────────────────────────    # FILTER 5: (A) Transient masking after switch events (Schein 2006)
    # ─────────────────────────────────────────────────────────────────
    # Mask N samples after setpoint jumps or fan on/off transitions.
    if config.getboolean('lstm_autoencoder', 'mask_switch_transients', fallback=False):
        sp_thresh  = config.getfloat('lstm_autoencoder', 'switch_sp_threshold_degC', fallback=0.5)
        mod_thresh_pct = config.getfloat('lstm_autoencoder', 'switch_mod_threshold_pct', fallback=10.0)
        n_mask     = config.getint('lstm_autoencoder', 'switch_mask_steps', fallback=2)

        transient_mask = np.zeros(len(df_train), dtype=bool)

        # Detect setpoint switch events
        sp_cols = [c for c in df_train.columns
                   if 'set point' in c.lower() or 'setpoint' in c.lower() or 'compensat' in c.lower()]
        for col in sp_cols:
            diff_sp = df_train[col].diff().abs()
            switch_idx = np.where(diff_sp >= sp_thresh)[0]
            for idx in switch_idx:
                end = min(idx + n_mask + 1, len(df_train))
                transient_mask[idx:end] = True

        # Detect fan on/off transitions (modulation crosses 0 <-> >mod_thresh)
        mod_cols = [c for c in df_train.columns
                    if 'modul' in c.lower() or 'ventilat' in c.lower() or 'fan' in c.lower()]
        for col in mod_cols:
            col_thresh = _pct_to_series_units(df_train[col], mod_thresh_pct)
            mod_on  = (df_train[col] > col_thresh).astype(int)
            mod_off = (df_train[col] <= col_thresh).astype(int)
            on_transitions  = np.where(mod_on.diff().fillna(0) == 1)[0]
            off_transitions = np.where(mod_off.diff().fillna(0) == 1)[0]
            for idx in np.concatenate([on_transitions, off_transitions]):
                end = min(idx + n_mask + 1, len(df_train))
                transient_mask[idx:end] = True

        excluded_trans = transient_mask.sum()
        if excluded_trans > 0:
            print(f"      \u2022 Transient masking (N={n_mask} steps after switch): Excluded {excluded_trans} ({excluded_trans/original_count*100:.1f}%)")
            clean_mask &= ~transient_mask
        else:
            print(f"      \u2022 Transient masking: No switch events detected")

    # ─────────────────────────────────────────────────────────────────    # SUMMARY
    # ───────────────────────────────────────────────────────────────
    excluded_count = original_count - clean_mask.sum()
    clean_percentage = clean_mask.sum()/original_count*100
    
    print(f"\n      📊 ENHANCED FILTER SUMMARY:")
    print(f"      • Original samples: {original_count:,}")
    print(f"      • Excluded: {excluded_count:,} ({excluded_count/original_count*100:.1f}%)")
    print(f"      • Clean (for training): {clean_mask.sum():,} ({clean_percentage:.1f}%)")
    
    # Warn if too much data excluded
    if clean_percentage < 50:
        print(f"      ⚠️  WARNING: Less than 50% of training data is clean!")
        print(f"      Consider relaxing filter criteria in config.")
    elif clean_percentage > 95:
        print(f"      ✅ Excellent: >95% of data passes quality filters")
    
    return clean_mask


def filter_sequences_by_clean_mask(sequences, clean_mask, seq_len):
    """
    Exclude sequences that touch any "unclean" data points
    
    A sequence is clean ONLY if ALL timesteps in it are clean
    
    Args:
        sequences: Array of sequence indices
        clean_mask: Boolean mask (True = clean data)
        seq_len: Length of each sequence
        
    Returns:
        clean_sequence_mask: Boolean mask for sequences
    """
    
    if clean_mask is None:
        return np.ones(len(sequences), dtype=bool)
    
    clean_sequence_mask = np.ones(len(sequences), dtype=bool)
    
    # For each sequence, check if ALL timesteps are clean
    for i in range(len(sequences)):
        start_idx = i
        end_idx = i + seq_len
        
        # If ANY timestep in sequence is unclean, exclude entire sequence
        if not clean_mask[start_idx:end_idx].all():
            clean_sequence_mask[i] = False
    
    return clean_sequence_mask


def inverse_transform_sequences(sequences, scaler, n_features):
    """
    Convert scaled sequences back to original scale
    Useful for visualization and interpretation
    
    Args:
        sequences: Scaled sequences (n_samples, seq_len, n_features)
        scaler: Fitted scaler object
        n_features: Number of features
        
    Returns:
        Unscaled sequences in original units
    """
    
    n_samples, seq_len, _ = sequences.shape
    
    # Reshape to 2D for scaler
    sequences_2d = sequences.reshape(-1, n_features)
    
    # Inverse transform
    unscaled_2d = scaler.inverse_transform(sequences_2d)
    
    # Reshape back to 3D
    unscaled_sequences = unscaled_2d.reshape(n_samples, seq_len, n_features)
    
    return unscaled_sequences


# ═════════════════════════════════════════════════════════════════════════════
# TEST SCRIPT - Run this file directly to test preprocessing
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import configparser
    from pathlib import Path
    import importlib
    data_loader = importlib.import_module('2_data_loader')
    load_and_prepare_data = data_loader.load_and_prepare_data
    
    print("\n🧪 TESTING PREPROCESSOR.PY\n")
    
    # Load configuration
    config = configparser.ConfigParser()
    # Use main config.ini from project root (2 levels up)
    config_path = Path(__file__).parent.parent.parent / 'config.ini'
    config.read(config_path, encoding='utf-8-sig')
    
    try:
        # Load data
        data_dict = load_and_prepare_data(config)
        
        # Preprocess
        processed = preprocess_data(data_dict, config)
        
        print(f"\n✅ PREPROCESSING SUCCESSFUL!")
        print(f"\n   Shapes:")
        print(f"   • X_train: {processed['X_train'].shape}")
        print(f"   • X_val:   {processed['X_val'].shape}")
        print(f"   • X_test:  {processed['X_test'].shape}")
        print(f"   • Input shape for model: {processed['input_shape']}")
        
        # Test inverse transform
        sample_seq = processed['X_test'][:5]
        unscaled = inverse_transform_sequences(sample_seq, processed['scaler'], data_dict['n_features'])
        print(f"\n   ✓ Inverse transform works: {unscaled.shape}")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
