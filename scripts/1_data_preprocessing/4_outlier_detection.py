# ==============================================================================
# Enhanced Multi-Method Outlier Detection Script for HVAC Data
# ==============================================================================
#
# This script performs a multi-layered outlier detection analysis using:
#   1. Physical Limits: A simple min/max check for impossible values.
#   2. IQR Method: A statistical check for improbable values based on distribution.
#   3. Valid States: A precise check for discrete variables against a list of
#      allowed operational states.
#
# All methods and column assignments are controlled via the `config.ini` file.
# ==============================================================================

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import configparser
from datetime import datetime
import warnings
from pathlib import Path

# Ignore all warnings to keep the output clean
warnings.filterwarnings('ignore')

# Try to import python-docx for Word report generation
try:
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("[WARNING] python-docx not installed. Word report generation will be disabled.")
    print("To enable: pip install python-docx")

# Add parent directories to path to import config_utils
# From scripts/1_data_preprocessing/ -> scripts/ -> project_root/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Try to import config_utils, with fallback if not available
try:
    from config_utils import load_config
except ModuleNotFoundError:
    def load_config(config_path):
        """Fallback function to load config WITH interpolation support."""
        config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        config.optionxform = str  # Preserve case sensitivity
        config.read(config_path, encoding='utf-8-sig')
        return config

def read_config(config_path="config.ini"):
    """
    Read configuration settings from a file and return a ConfigParser object.
    Enhanced with robust error handling and path validation.
    """
    # Validate config file exists
    config_file = Path(config_path)
    if not config_file.exists():
        # Try parent directory (project root) - 2 levels up from scripts/1_data_preprocessing/
        config_file = Path(__file__).parent.parent.parent / 'config.ini'
    
    if not config_file.exists():
        print(f"❌ ERROR: Configuration file not found!")
        print(f"  Looking for: {config_path}")
        print(f"  Also tried: {config_file}")
        print(f"  Current directory: {Path.cwd()}")
        print(f"  Script location: {Path(__file__).parent}")
        sys.exit(1)
    
    print(f"[INFO] Using config: {config_file}")
    
    try:
        # Use the load_config helper (with or without config_utils)
        config = load_config(str(config_file))
        # Preserve case sensitivity for column names
        config.optionxform = str
        # Record config directory to resolve relative paths reliably
        try:
            config._config_dir = str(config_file.parent)
        except Exception:
            pass
        print(f"✅ Configuration loaded successfully")
        return config
    except Exception as e:
        print(f"❌ ERROR loading config file: {e}")
        sys.exit(1)

def clean_temp(series):
    """
    Clean temperature strings by removing symbols and normalizing decimal separators.
    """
    cleaned_series = series.astype(str)
    cleaned_series = cleaned_series.str.replace(' °C', '', regex=False)
    cleaned_series = cleaned_series.str.replace(' Â°C', '', regex=False)
    cleaned_series = cleaned_series.str.replace(' Ã‚Â°C', '', regex=False)
    cleaned_series = cleaned_series.str.replace(',', '.', regex=False)
    cleaned_series = cleaned_series.str.strip()
    return pd.to_numeric(cleaned_series, errors='coerce')

# --- NEW: Outlier Detection Functions ---

def detect_physical_limit_outliers(data, min_threshold, max_threshold):
    """
    Identifies outliers based on fixed physical min/max thresholds.
    """
    return (data < min_threshold) | (data > max_threshold)

def detect_iqr_outliers(data, multiplier=1.5):
    """
    Identifies outliers using the Interquartile Range (IQR) method.
    Returns a "soft flag" - these are statistical anomalies but not necessarily errors.
    
    NOTE: This method should be used cautiously:
    - NOT suitable for setpoint columns (flags legitimate control actions)
    - Best for continuous sensor data (temperatures, pressures, flows)
    - Consider using regime-based IQR for variables affected by operating conditions
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (multiplier * IQR)
    upper_bound = Q3 + (multiplier * IQR)
    
    outliers = (data < lower_bound) | (data > upper_bound)
    return outliers, lower_bound, upper_bound

def detect_iqr_outliers_by_regime(data, regime_column, multiplier=1.5, n_bins=5):
    """
    Identifies outliers using IQR method within operating regimes.
    
    This is more accurate than global IQR for variables whose normal range
    depends on operating conditions (e.g., supply temp varies with outdoor temp).
    
    Parameters:
    -----------
    data : pd.Series
        Time-series data to analyze
    regime_column : pd.Series
        Column defining regimes (e.g., outdoor temperature)
        Must have same index as data
    multiplier : float
        IQR multiplier (default: 1.5)
    n_bins : int
        Number of bins to split regime_column into (default: 5)
    
    Returns:
    --------
    pd.Series : Boolean mask where True indicates an outlier
    dict : Information about regimes and bounds
    """
    outliers = pd.Series(False, index=data.index)
    regime_info = {}
    
    # Create bins for the regime column
    try:
        # Use qcut for equal-frequency bins
        regime_bins = pd.qcut(regime_column, q=n_bins, duplicates='drop', labels=False)
    except Exception:
        # Fallback to equal-width bins if qcut fails
        regime_bins = pd.cut(regime_column, bins=n_bins, labels=False)
    
    # Apply IQR within each regime
    for regime_id in regime_bins.dropna().unique():
        regime_mask = regime_bins == regime_id
        regime_data = data[regime_mask].dropna()
        
        if len(regime_data) < 4:  # Need at least 4 points for IQR
            continue
        
        Q1 = regime_data.quantile(0.25)
        Q3 = regime_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (multiplier * IQR)
        upper_bound = Q3 + (multiplier * IQR)
        
        regime_outliers = (regime_data < lower_bound) | (regime_data > upper_bound)
        outliers.loc[regime_outliers.index] |= regime_outliers
        
        regime_info[int(regime_id)] = {
            'lower': lower_bound,
            'upper': upper_bound,
            'count': regime_mask.sum(),
            'outliers': regime_outliers.sum()
        }
    
    return outliers, regime_info

def detect_valid_states_outliers(data, valid_states, tolerance=0.0):
    """
    Identifies outliers by checking if values are in a list of valid states.

    If tolerance > 0 and data is numeric, a value is considered valid if it is
    within +/- tolerance of any configured valid state. This avoids false
    positives for floating values that are "near" discrete states.
    """
    # If non-numeric or no tolerance requested, use exact membership
    try:
        series = pd.to_numeric(data, errors='coerce')
        numeric = series.notna().any()
    except Exception:
        numeric = False

    if not numeric or tolerance <= 0:
        return ~data.isin(valid_states)

    # Numeric with tolerance: consider values close to any valid state as valid
    arr = series.to_numpy()
    states = np.array(valid_states, dtype=float)
    # For rows that are NaN, keep them as non-outliers here; other steps will handle NaNs
    nan_mask = np.isnan(arr)
    if states.size == 0:
        return pd.Series(True, index=data.index)  # if no states, mark all as outliers by convention

    diffs = np.abs(arr[:, None] - states[None, :])
    within = np.any(diffs <= float(tolerance), axis=1)
    within[nan_mask] = False  # do not mark NaNs as within states
    return pd.Series(~within, index=data.index)

def detect_spike_outliers(data, threshold=3.0, window=None):
    """
    Detects sudden spikes/jumps in time-series data using rate of change.
    
    A spike is detected when the absolute change from one timestep to the next
    exceeds threshold * rolling standard deviation of changes.
    
    Parameters:
    -----------
    data : pd.Series
        Time-series data to analyze
    threshold : float
        Number of standard deviations for spike detection (default: 3.0)
    window : int or None
        Window size for rolling std calculation (default: None = global std)
    
    Returns:
    --------
    pd.Series : Boolean mask where True indicates a spike
    float : Detection threshold used
    """
    # Calculate absolute rate of change
    diff = data.diff().abs()
    
    if window is not None:
        # Use rolling standard deviation
        std_dev = diff.rolling(window=window, min_periods=1).std()
    else:
        # Use global standard deviation
        std_dev = diff.std()
    
    # Detect spikes: changes larger than threshold * std
    if isinstance(std_dev, pd.Series):
        spike_threshold = threshold * std_dev
        spikes = diff > spike_threshold
    else:
        spike_threshold = threshold * std_dev
        spikes = diff > spike_threshold
    
    # First value can't be a spike (no previous value to compare)
    spikes.iloc[0] = False
    
    return spikes, spike_threshold if not isinstance(spike_threshold, pd.Series) else spike_threshold.median()

def detect_stuck_values(data, min_duration=10, tolerance=0.01):
    """
    Detects "stuck" or "frozen" sensor values that remain constant for too long.
    
    A value is considered stuck if it remains within tolerance for min_duration
    consecutive timesteps. This is useful for detecting sensor failures.
    
    Parameters:
    -----------
    data : pd.Series
        Time-series data to analyze
    min_duration : int
        Minimum number of consecutive identical (within tolerance) values
        to flag as stuck (default: 10)
    tolerance : float
        Tolerance for considering values as "same" (default: 0.01)
    
    Returns:
    --------
    pd.Series : Boolean mask where True indicates stuck values
    int : Number of stuck sequences detected
    """
    stuck_mask = pd.Series(False, index=data.index)
    
    # Calculate absolute differences
    diff = data.diff().abs()
    
    # Find sequences where change is within tolerance
    is_constant = diff <= tolerance
    is_constant.iloc[0] = False  # First value can't be compared
    
    # Group consecutive constant values
    # Create groups by detecting changes in the is_constant mask
    groups = (is_constant != is_constant.shift()).cumsum()
    
    # For each group of constant values, check if duration >= min_duration
    for group_id, group_data in is_constant.groupby(groups):
        if group_data.iloc[0]:  # Only process "True" groups (constant segments)
            if len(group_data) >= min_duration:
                stuck_mask.loc[group_data.index] = True
    
    # Count number of stuck sequences
    stuck_sequences = stuck_mask.groupby((stuck_mask != stuck_mask.shift()).cumsum()).size()
    num_sequences = (stuck_sequences > 0).sum() if len(stuck_sequences) > 0 else 0
    
    return stuck_mask, num_sequences

def resolve_physical_thresholds(config, column_name):
    """
    Resolve physical thresholds for a column, supporting optional seasonal overrides.

    Lookup order:
      1) [physical_thresholds_<season>] (e.g., physical_thresholds_winter)
      2) [physical_thresholds]
      3) Backward-compat: [outlier_detection] (key named exactly as column)

    Returns (min, max) as floats or None if not found/parseable.
    """
    # Determine season if available
    season = None
    try:
        season = config.get('global', 'season', fallback='').strip().lower()
    except Exception:
        season = ''

    candidate_sections = []
    if season in ('winter', 'summer'):
        candidate_sections.append(f'physical_thresholds_{season}')
    candidate_sections.append('physical_thresholds')
    # Back-compat: allow old placement in [outlier_detection]
    candidate_sections.append('outlier_detection')

    for section in candidate_sections:
        if section in config and config.has_option(section, column_name):
            try:
                raw = config.get(section, column_name)
                parts = [p.strip() for p in raw.split(',') if p.strip()]
                if len(parts) == 2:
                    return float(parts[0]), float(parts[1])
            except Exception:
                continue
    return None

# --- Data Type Diagnostics ---

def check_data_type(series, column_name):
    """
    Diagnostic function to check if data appears to be continuous or discrete.
    
    Returns:
        dict: {
            'type': 'continuous' | 'discrete' | 'unknown',
            'unique_count': int,
            'unique_ratio': float,  # ratio of unique values to total values
            'recommendation': str
        }
    """
    valid_data = pd.to_numeric(series, errors='coerce').dropna()
    
    if len(valid_data) == 0:
        return {'type': 'unknown', 'unique_count': 0, 'unique_ratio': 0, 'recommendation': 'No data'}
    
    unique_count = valid_data.nunique()
    unique_ratio = unique_count / len(valid_data)
    
    # Heuristic: if < 5% unique values, likely discrete; if > 20%, likely continuous
    if unique_ratio < 0.05 or unique_count <= 10:
        data_type = 'discrete'
        recommendation = '✅ Suitable for valid_states method'
    elif unique_ratio > 0.20:
        data_type = 'continuous'
        recommendation = '⚠️  NOT suitable for valid_states (use range/rate-of-change instead)'
    else:
        data_type = 'mixed'
        recommendation = '⚠️  Ambiguous - verify control system behavior'
    
    return {
        'type': data_type,
        'unique_count': unique_count,
        'unique_ratio': unique_ratio,
        'recommendation': recommendation
    }

def is_setpoint_column(column_name):
    """
    Heuristic to detect if a column is a setpoint (vs sensor reading).
    Setpoints should NOT use IQR method as they change in discrete steps.
    """
    setpoint_indicators = ['set point', 'setpoint', 'sp ', 'set_point', 'sp_', 'compensat']
    column_lower = column_name.lower()
    return any(indicator in column_lower for indicator in setpoint_indicators)

# --- MODIFIED: Main Analysis Function ---

def analyze_column_outliers(df, column, config):
    """
    Performs a complete, multi-method outlier analysis for a single column
    based on settings in the configuration file.
    
    Detection hierarchy:
    1. Physical Limits - Hard threshold (always flags as outlier)
    2. Spike Detection - Hard threshold (sudden jumps)
    3. Stuck Detection - Hard threshold (frozen sensors)
    4. IQR - Soft flag only (statistical anomaly, not necessarily error)
    5. Valid States - For discrete variables only
    """
    results = {}
    
    # --- Parse Configuration for this Column ---
    od_config = config['outlier_detection']
    
    # Check which methods are enabled globally and for this specific column
    use_physical = od_config.getboolean('enable_physical_limits', fallback=False)
    use_spike = od_config.getboolean('enable_spike_detection', fallback=False) and column in [c.strip() for c in od_config.get('spike_columns', '').split(',') if c.strip()]
    use_stuck = od_config.getboolean('enable_stuck_detection', fallback=False)
    use_iqr = od_config.getboolean('enable_iqr', fallback=False) and column in [c.strip() for c in od_config.get('iqr_columns', '').split(',') if c.strip()]
    use_iqr_regime = od_config.getboolean('enable_iqr_regime', fallback=False) and column in [c.strip() for c in od_config.get('iqr_regime_columns', '').split(',') if c.strip()]
    use_valid_states = od_config.getboolean('enable_valid_states', fallback=False) and column in [c.strip() for c in od_config.get('valid_states_columns', '').split(',') if c.strip()]

    # If no methods are enabled for this column, skip it.
    if not (use_physical or use_spike or use_stuck or use_iqr or use_iqr_regime or use_valid_states):
        print(f"⚠️ No detection methods enabled for column: {column}")
        return results

    valid_data = df[column].dropna()
    if len(valid_data) == 0:
        print(f"⚠️ No valid data for column {column}")
        return results

    # Initialize two outlier masks:
    # 1. Hard outliers - definite errors (physical limits, spikes, stuck)
    # 2. Soft flags - statistical anomalies (IQR)
    hard_outliers = pd.Series(False, index=df.index, dtype=bool)
    soft_flags = pd.Series(False, index=df.index, dtype=bool)
    
    # Store details about what was found
    methods_used = []
    
    # --- 1. Physical Limit Detection (HARD THRESHOLD) ---
    if use_physical:
        methods_used.append('Physical')
        try:
            # Read thresholds from seasonal section or [physical_thresholds] (with back-compat)
            thresholds = resolve_physical_thresholds(config, column)
            if thresholds is None:
                raise ValueError(
                    f"Threshold not found for {column} in seasonal or physical thresholds sections"
                )
            min_thresh, max_thresh = thresholds
            physical_outliers = detect_physical_limit_outliers(df[column], min_thresh, max_thresh)
            hard_outliers |= physical_outliers
            results['physical_count'] = physical_outliers.sum()
            results['physical_outliers_mask'] = physical_outliers  # Store the boolean mask
            results['min_threshold'] = min_thresh
            results['max_threshold'] = max_thresh
        except (configparser.NoOptionError, ValueError):
            print(
                f"  - Warning: No valid physical limits for '{column}' in physical thresholds. Skipping this method."
            )
            results['physical_count'] = 0
            results['physical_outliers_mask'] = pd.Series(False, index=df.index)

    # --- 2. Spike Detection (HARD THRESHOLD) ---
    if use_spike:
        methods_used.append('Spike')
        try:
            spike_threshold = od_config.getfloat('spike_threshold', fallback=3.0)
            spike_window = od_config.getint('spike_window', fallback=0)
            if spike_window <= 0:
                spike_window = None
            
            spike_outliers, threshold_value = detect_spike_outliers(
                valid_data, 
                threshold=spike_threshold,
                window=spike_window
            )
            # Create full-index mask
            spike_mask = pd.Series(False, index=df.index)
            spike_mask.loc[spike_outliers.index] = spike_outliers
            
            hard_outliers.loc[spike_outliers.index] |= spike_outliers
            results['spike_count'] = spike_outliers.sum()
            results['spike_outliers_mask'] = spike_mask  # Store the boolean mask
            results['spike_threshold'] = threshold_value
        except Exception as e:
            print(f"  - Warning: Spike detection failed for '{column}': {e}")
            results['spike_count'] = 0
            results['spike_outliers_mask'] = pd.Series(False, index=df.index)

    # --- 3. Stuck Value Detection (HARD THRESHOLD) ---
    if use_stuck:
        methods_used.append('Stuck')
        try:
            stuck_duration = od_config.getint('stuck_min_duration', fallback=10)
            stuck_tolerance = od_config.getfloat('stuck_tolerance', fallback=0.01)
            
            stuck_outliers, num_sequences = detect_stuck_values(
                valid_data,
                min_duration=stuck_duration,
                tolerance=stuck_tolerance
            )
            # Create full-index mask
            stuck_mask = pd.Series(False, index=df.index)
            stuck_mask.loc[stuck_outliers.index] = stuck_outliers
            
            hard_outliers.loc[stuck_outliers.index] |= stuck_outliers
            results['stuck_count'] = stuck_outliers.sum()
            results['stuck_outliers_mask'] = stuck_mask  # Store the boolean mask
            results['stuck_sequences'] = num_sequences
        except Exception as e:
            print(f"  - Warning: Stuck detection failed for '{column}': {e}")
            results['stuck_count'] = 0
            results['stuck_outliers_mask'] = pd.Series(False, index=df.index)

    # --- 4. IQR Detection (SOFT FLAG) ---
    if use_iqr:
        methods_used.append('IQR (soft)')
        
        # 🔴 CRITICAL WARNING: IQR is inappropriate for setpoint columns
        if is_setpoint_column(column):
            print(f"  ⚠️  WARNING: IQR method applied to setpoint column '{column}'!")
            print(f"      Setpoints change in discrete steps → IQR will flag control actions as outliers")
            print(f"      RECOMMENDATION: Remove '{column}' from iqr_columns in config.ini")
            print(f"      Use physical limits or spike detection instead.")
        
        multiplier = od_config.getfloat('iqr_multiplier', fallback=3.5)
        iqr_outliers, lower, upper = detect_iqr_outliers(valid_data, multiplier)
        # Create full-index mask
        iqr_mask = pd.Series(False, index=df.index)
        iqr_mask.loc[iqr_outliers.index] = iqr_outliers
        
        soft_flags.loc[iqr_outliers.index] |= iqr_outliers
        results['iqr_count'] = iqr_outliers.sum()
        results['iqr_outliers_mask'] = iqr_mask  # Store the boolean mask
        results['iqr_lower_bound'] = lower
        results['iqr_upper_bound'] = upper
        results['iqr_is_soft_flag'] = True

    # --- 5. Regime-Based IQR Detection (SOFT FLAG) ---
    if use_iqr_regime:
        methods_used.append('IQR-Regime (soft)')
        try:
            regime_column_name = od_config.get('iqr_regime_column', fallback='Outdoor Temp')
            if regime_column_name not in df.columns:
                raise ValueError(f"Regime column '{regime_column_name}' not found in dataframe")
            
            multiplier = od_config.getfloat('iqr_multiplier', fallback=3.5)
            n_bins = od_config.getint('iqr_regime_bins', fallback=5)
            
            regime_outliers, regime_info = detect_iqr_outliers_by_regime(
                valid_data,
                df[regime_column_name],
                multiplier=multiplier,
                n_bins=n_bins
            )
            # Create full-index mask
            iqr_regime_mask = pd.Series(False, index=df.index)
            iqr_regime_mask.loc[regime_outliers.index] = regime_outliers
            
            soft_flags.loc[regime_outliers.index] |= regime_outliers
            results['iqr_regime_count'] = regime_outliers.sum()
            results['iqr_regime_outliers_mask'] = iqr_regime_mask  # Store the boolean mask
            results['iqr_regime_info'] = regime_info
            results['iqr_regime_is_soft_flag'] = True
        except Exception as e:
            print(f"  - Warning: Regime-based IQR failed for '{column}': {e}")
            results['iqr_regime_count'] = 0
            results['iqr_regime_outliers_mask'] = pd.Series(False, index=df.index)

    # --- 6. Valid States Detection (for discrete variables) ---
    if use_valid_states:
        methods_used.append('Valid States')
        
        # 🔴 CRITICAL WARNING: Check if data is actually discrete
        data_check = check_data_type(df[column], column)
        if data_check['type'] == 'continuous':
            print(f"  ⚠️  CRITICAL WARNING: Valid states method applied to CONTINUOUS data!")
            print(f"      Column: '{column}'")
            print(f"      Data type: {data_check['type']}")
            print(f"      Unique values: {data_check['unique_count']:,} ({data_check['unique_ratio']:.1%} of data)")
            print(f"      ")
            print(f"      🔴 This will incorrectly flag thousands of normal values as outliers!")
            print(f"      ")
            print(f"      ✅ SOLUTIONS:")
            print(f"         1. Disable valid_states for '{column}' (remove from valid_states_columns)")
            print(f"         2. Use physical limits (0-100) instead")
            print(f"         3. Add spike detection for jump detection")
            print(f"         4. Add stuck detection for frozen value detection")
            print(f"      ")
        
        try:
            valid_states_str = od_config.get(f"{column}_valid_states")
            valid_states = [float(s.strip()) for s in valid_states_str.split(',')]
            tolerance = od_config.getfloat('valid_states_tolerance', fallback=0.05)
            valid_states_outliers = detect_valid_states_outliers(df[column], valid_states, tolerance=tolerance)
            hard_outliers |= valid_states_outliers
            results['valid_states_count'] = valid_states_outliers.sum()
            results['valid_states_outliers_mask'] = valid_states_outliers  # Store the boolean mask
            results['valid_states'] = valid_states
            results['valid_states_tolerance'] = tolerance
        except configparser.NoOptionError:
            print(f"  - Warning: No valid states defined for '{column}'. Skipping this method.")
            results['valid_states_count'] = 0
            results['valid_states_outliers_mask'] = pd.Series(False, index=df.index)

    # --- Final Calculations and Reporting ---
    # Combine hard outliers and soft flags
    combined_outliers = hard_outliers | soft_flags
    
    total_points = len(valid_data)
    hard_count = hard_outliers.sum()
    soft_count = soft_flags.sum()
    outlier_count = combined_outliers.sum()
    outlier_percentage = (outlier_count / total_points * 100) if total_points > 0 else 0
    hard_percentage = (hard_count / total_points * 100) if total_points > 0 else 0
    soft_percentage = (soft_count / total_points * 100) if total_points > 0 else 0

    # Populate the final results dictionary
    results.update({
        'outliers': combined_outliers,
        'hard_outliers': hard_outliers,
        'soft_flags': soft_flags,
        'count': outlier_count,
        'hard_count': hard_count,
        'soft_count': soft_count,
        'percentage': outlier_percentage,
        'hard_percentage': hard_percentage,
        'soft_percentage': soft_percentage,
        'total_points': total_points,
        'methods_used': ", ".join(methods_used)
    })
    
    return results

# --- Visualization and Reporting (Modified to handle new data) ---

def set_cell_border(cell, **kwargs):
    """
    Set cell borders for Word table cells.
    """
    tc = cell._element
    tcPr = tc.get_or_add_tcPr()
    
    for edge in ('top', 'left', 'bottom', 'right'):
        if edge in kwargs:
            edge_data = kwargs[edge]
            edge_el = OxmlElement(f'w:{edge}')
            edge_el.set(qn('w:val'), edge_data.get('val', 'single'))
            edge_el.set(qn('w:sz'), str(edge_data.get('sz', 12)))
            edge_el.set(qn('w:space'), '0')
            edge_el.set(qn('w:color'), edge_data.get('color', '000000'))
            tcPr.append(edge_el)

def create_individual_outlier_plots(df, all_results, plots_dir, base_name, columns_analyzed):
    """
    Creates and saves an individual time-series outlier plot for EACH column.
    Returns list of (column_name, plot_path, description) tuples for Word report.
    """
    print("Creating individual time-series plots for each column...")
    
    plot_info_list = []

    for col in columns_analyzed:
        if col not in all_results or not all_results[col]:
            print(f"  -> Skipping plot for {col} (no results).")
            continue

        fig, ax = plt.subplots(figsize=(15, 6))
        s = pd.to_numeric(df[col], errors='coerce').dropna()
        if s.empty:
            print(f"  -> No numeric data to plot for {col}")
            plt.close(fig)
            continue

        # Plot main data
        ax.plot(s.index, s.values, alpha=0.9, linewidth=0.9, color='blue', marker='.', markersize=2, label='Data')

        # Highlight hard outliers (definite errors) in RED
        hard_outlier_mask = all_results[col].get('hard_outliers')
        if hard_outlier_mask is not None:
            hard_outliers_on_valid_data = hard_outlier_mask.reindex(s.index).fillna(False)
            if hard_outliers_on_valid_data.any():
                ax.scatter(s.index[hard_outliers_on_valid_data], s[hard_outliers_on_valid_data], 
                          color='red', s=40, alpha=0.95, label='Hard Outliers (errors)', zorder=6, marker='x')
        
        # Highlight soft flags (statistical anomalies) in ORANGE
        soft_flag_mask = all_results[col].get('soft_flags')
        if soft_flag_mask is not None:
            soft_flags_on_valid_data = soft_flag_mask.reindex(s.index).fillna(False)
            if soft_flags_on_valid_data.any():
                ax.scatter(s.index[soft_flags_on_valid_data], s[soft_flags_on_valid_data], 
                          color='orange', s=25, alpha=0.7, label='Soft Flags (IQR anomalies)', zorder=5, marker='o')

        # Plot threshold lines
        if 'min_threshold' in all_results[col]:
            ax.axhline(y=all_results[col]['min_threshold'], color='green', linestyle='--', alpha=0.7, label=f"Physical Min/Max")
            ax.axhline(y=all_results[col]['max_threshold'], color='green', linestyle='--')
        
        if 'iqr_lower_bound' in all_results[col]:
            ax.axhline(y=all_results[col]['iqr_lower_bound'], color='orange', linestyle=':', alpha=0.8, label=f"IQR Bounds")
            ax.axhline(y=all_results[col]['iqr_upper_bound'], color='orange', linestyle=':')

        ax.set_title(f'Outlier Analysis for: {col} (Methods: {all_results[col]["methods_used"]})', fontsize=16)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add date range at the bottom of the plot
        date_range_text = f"Periodo: {s.index.min().strftime('%d/%m/%Y')} - {s.index.max().strftime('%d/%m/%Y')}"
        fig.text(0.5, 0.02, date_range_text, ha='center', fontsize=11, style='italic', color='gray')
        
        plt.tight_layout(rect=[0, 0.03, 1, 1])  # Make room for date range text

        plot_filename = f"{base_name}_{col}_timeseries_outliers.png"
        save_path = os.path.join(plots_dir, plot_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Generate description for Word report
        description = generate_plot_description(col, all_results[col])
        plot_info_list.append((col, save_path, description))
        
    print("✅ Individual plots created successfully.")
    return plot_info_list

def generate_plot_description(column_name, result):
    """
    Generate an Italian description for the outlier plot.
    """
    methods = result.get('methods_used', 'N/A')
    total_outliers = result.get('count', 0)
    outlier_pct = result.get('percentage', 0)
    hard_count = result.get('hard_count', 0)
    soft_count = result.get('soft_count', 0)
    hard_pct = result.get('hard_percentage', 0)
    soft_pct = result.get('soft_percentage', 0)
    total_points = result.get('total_points', 0)
    
    description = f"Questo grafico mostra l'analisi degli outlier per la variabile '{column_name}' utilizzando i metodi: {methods}.\n\n"
    
    description += f"**Statistiche:**\n"
    description += f"• Punti totali analizzati: {total_points:,}\n"
    description += f"• Outlier totali: {total_outliers:,} ({outlier_pct:.2f}%)\n"
    if hard_count > 0:
        description += f"  - Hard outliers (errori certi): {hard_count:,} ({hard_pct:.2f}%)\n"
    if soft_count > 0:
        description += f"  - Soft flags (anomalie IQR): {soft_count:,} ({soft_pct:.2f}%)\n"
    description += f"\n"
    
    # Add method-specific details
    if 'physical_count' in result and result['physical_count'] > 0:
        description += f"**Limiti Fisici (Hard):**\n"
        description += f"• Outlier fuori limiti fisici: {result['physical_count']:,}\n"
        if 'min_threshold' in result:
            description += f"• Range ammesso: {result['min_threshold']} - {result['max_threshold']}\n"
        description += f"• Questi valori superano i limiti fisicamente possibili per la variabile.\n\n"
    
    if 'spike_count' in result and result['spike_count'] > 0:
        description += f"**Spike Detection (Hard):**\n"
        description += f"• Spike rilevati: {result['spike_count']:,}\n"
        if 'spike_threshold' in result:
            description += f"• Soglia: {result['spike_threshold']:.2f}\n"
        description += f"• Salti improvvisi nel valore che indicano possibili errori di misura.\n\n"
    
    if 'stuck_count' in result and result['stuck_count'] > 0:
        description += f"**Stuck Detection (Hard):**\n"
        description += f"• Valori bloccati: {result['stuck_count']:,}\n"
        if 'stuck_sequences' in result:
            description += f"• Sequenze bloccate: {result['stuck_sequences']}\n"
        description += f"• Sensori che rimangono costanti per troppo tempo (possibile guasto).\n\n"
    
    if 'iqr_count' in result and result['iqr_count'] > 0:
        description += f"**Metodo IQR (Soft Flag):**\n"
        description += f"• Anomalie statistiche: {result['iqr_count']:,}\n"
        if 'iqr_lower_bound' in result:
            description += f"• Range IQR: {result['iqr_lower_bound']:.2f} - {result['iqr_upper_bound']:.2f}\n"
        description += f"• ⚠️ SOFT FLAG: valori inusuali ma non necessariamente errori.\n"
        description += f"• Valori che si discostano dalla distribuzione normale.\n\n"
    
    if 'iqr_regime_count' in result and result['iqr_regime_count'] > 0:
        description += f"**Metodo IQR per Regime (Soft Flag):**\n"
        description += f"• Anomalie per regime: {result['iqr_regime_count']:,}\n"
        description += f"• ⚠️ SOFT FLAG: anomalie considerando le condizioni operative.\n\n"
    
    if 'valid_states_count' in result and result['valid_states_count'] > 0:
        description += f"**Stati Validi (Hard):**\n"
        description += f"• Valori non validi: {result['valid_states_count']:,}\n"
        if 'valid_states' in result:
            states_str = ", ".join([str(s) for s in result['valid_states'][:5]])
            description += f"• Stati ammessi: {states_str}{'...' if len(result['valid_states']) > 5 else ''}\n"
        description += f"• Valori che non corrispondono agli stati operativi previsti.\n\n"
    
    # Add interpretation based on HARD outliers only
    description += f"**Interpretazione:**\n"
    if hard_pct > 10:
        description += f"🔴 Alta percentuale di errori certi ({hard_pct:.1f}%). Si raccomanda un'analisi approfondita e possibile esclusione della variabile o pulizia dei dati.\n"
    elif hard_pct > 5:
        description += f"⚠️ Presenza moderata di errori ({hard_pct:.1f}%). Verificare la causa e valutare la correzione dei valori anomali.\n"
    elif hard_pct > 0:
        description += f"✓ Bassa percentuale di errori certi ({hard_pct:.1f}%). La qualità dei dati è accettabile.\n"
    else:
        description += f"✅ Nessun errore certo rilevato. Ottima qualità dei dati.\n"
    
    if soft_pct > 0:
        description += f"\nℹ️ {soft_pct:.1f}% dei dati sono segnalati come anomalie statistiche (soft flags). Questi valori potrebbero essere corretti ma inusuali.\n"
    
    return description

def create_word_report(plot_info_list, output_dir, base_name, df):
    """
    Creates a Word report with outlier plots in the same format as missing data analysis.
    Each plot is presented in a 3-row table: Title | Image | Description
    """
    if not DOCX_AVAILABLE:
        print("⚠️  python-docx not available. Skipping Word report generation.")
        return None
    
    try:
        print(f"\n{'='*60}")
        print("GENERATING WORD REPORT - OUTLIER DETECTION")
        print(f"{'='*60}\n")
        
        doc = Document()
        
        # Set page margins
        sections = doc.sections
        for section in sections:
            section.top_margin = Inches(0.75)
            section.bottom_margin = Inches(0.75)
            section.left_margin = Inches(0.75)
            section.right_margin = Inches(0.75)
        
        # Add title and date range
        start_date = df.index.min()
        end_date = df.index.max()
        date_range_str = f"Periodo di analisi: {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}"
        
        title = doc.add_heading('Analisi Outlier - Sistema HVAC', level=0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title_format = title.runs[0].font
        title_format.size = Pt(16)
        title_format.bold = True
        title_format.color.rgb = RGBColor(0, 0, 0)
        
        date_para = doc.add_paragraph(date_range_str)
        date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        date_format = date_para.runs[0].font
        date_format.size = Pt(11)
        date_format.bold = False
        date_format.color.rgb = RGBColor(128, 128, 128)
        
        doc.add_paragraph()
        
        # Add each plot with its description
        for idx, (column_name, image_path, description) in enumerate(plot_info_list, 1):
            print(f"   Adding plot {idx}/{len(plot_info_list)}: {column_name}")
            
            # Create table with 3 rows
            table = doc.add_table(rows=3, cols=1)
            table.style = 'Table Grid'
            table.autofit = False
            table.allow_autofit = False
            
            # Set table width to page width (6.5 inches for standard margins)
            for row in table.rows:
                for cell in row.cells:
                    cell.width = Inches(6.5)
            
            # Row 1: Title (Column Name)
            cell_title = table.rows[0].cells[0]
            set_cell_border(cell_title, top={}, left={}, right={}, bottom={})
            
            # Set light blue background color
            tc = cell_title._element
            tcPr = tc.get_or_add_tcPr()
            shd = OxmlElement('w:shd')
            shd.set(qn('w:fill'), 'E6F2FF')
            tcPr.append(shd)
            
            # Set cell padding
            tcMar = OxmlElement('w:tcMar')
            for margin_name in ['top', 'bottom']:
                node = OxmlElement(f'w:{margin_name}')
                node.set(qn('w:w'), '150')
                node.set(qn('w:type'), 'dxa')
                tcMar.append(node)
            tcPr.append(tcMar)
            
            title_para = cell_title.paragraphs[0]
            title_para.text = column_name
            title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            title_format = title_para.runs[0].font
            title_format.bold = True
            title_format.size = Pt(12)
            title_format.color.rgb = RGBColor(0, 51, 102)
            cell_title.vertical_alignment = 1  # Center vertically
            
            # Row 2: Image
            cell_image = table.rows[1].cells[0]
            set_cell_border(cell_image, top={}, left={}, right={}, bottom={})
            cell_image.paragraphs[0].clear()
            
            paragraph = cell_image.paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            
            try:
                run = paragraph.add_run()
                run.add_picture(image_path, width=Inches(6.3))
            except Exception as e:
                print(f"    ✗ Error adding image: {str(e)}")
                paragraph.add_run(f"[Errore caricamento immagine]")
            
            # Row 3: Description
            cell_desc = table.rows[2].cells[0]
            set_cell_border(cell_desc, top={}, left={}, right={}, bottom={})
            
            # Set cell padding
            tc = cell_desc._element
            tcPr = tc.get_or_add_tcPr()
            tcMar = OxmlElement('w:tcMar')
            for margin_name in ['top', 'left', 'bottom', 'right']:
                node = OxmlElement(f'w:{margin_name}')
                node.set(qn('w:w'), '100')
                node.set(qn('w:type'), 'dxa')
                tcMar.append(node)
            tcPr.append(tcMar)
            
            desc_header = cell_desc.paragraphs[0]
            desc_header.text = "Descrizione:"
            desc_header.paragraph_format.space_after = Pt(3)
            desc_header_format = desc_header.runs[0].font
            desc_header_format.bold = True
            desc_header_format.size = Pt(11)
            
            desc_para = cell_desc.add_paragraph(description)
            desc_para.paragraph_format.line_spacing = 1.15
            desc_para_format = desc_para.runs[0].font
            desc_para_format.size = Pt(10)
            
            # Add spacing
            if idx < len(plot_info_list):
                doc.add_paragraph()
        
        # Save Word document
        word_path = os.path.join(output_dir, f"{base_name}_Outlier_Detection_Report.docx")
        doc.save(word_path)
        
        print(f"\n✅ Word report saved: {os.path.basename(word_path)}")
        
        print(f"{'='*60}")
        print("WORD REPORT GENERATION COMPLETED")
        print(f"{'='*60}\n")
        
        return word_path
        
    except Exception as e:
        print(f"\n❌ Error generating Word report: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_outlier_report(all_results, df, output_dir, base_name, columns_analyzed):
    """
    Creates an Excel report with multiple sheets and a summary CSV file.
    """
    # Ensure timezone-naive for Excel compatibility - create fresh copy
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        # Remove timezone by converting to UTC-naive representation
        df = df.tz_localize(None)
    
    # Also remove timezone from any datetime columns
    for col in df.select_dtypes(include=["datetimetz"]).columns:
        df[col] = df[col].dt.tz_localize(None)

    excel_path = os.path.join(output_dir, f"{base_name}_outlier_analysis_report.xlsx")
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        summary_data = []
        for col in columns_analyzed:
            if col in all_results and all_results[col]:
                result = all_results[col]
                drop_threshold = config.getfloat('outlier_detection', 'drop_threshold_percentage', fallback=10)
                # Use hard_percentage for drop recommendation (not total percentage)
                hard_pct = result.get('hard_percentage', result.get('percentage', 0))
                summary_data.append({
                    'Column': col,
                    'Methods_Used': result.get('methods_used', 'N/A'),
                    'Total_Points': result['total_points'],
                    'Total_Outliers': result['count'],
                    'Total_Percentage': result['percentage'],
                    'Hard_Outliers': result.get('hard_count', 0),
                    'Hard_Percentage': result.get('hard_percentage', 0),
                    'Soft_Flags': result.get('soft_count', 0),
                    'Soft_Percentage': result.get('soft_percentage', 0),
                    'Physical_Outliers': result.get('physical_count', 0),
                    'Spike_Outliers': result.get('spike_count', 0),
                    'Stuck_Outliers': result.get('stuck_count', 0),
                    'IQR_Flags': result.get('iqr_count', 0),
                    'IQR_Regime_Flags': result.get('iqr_regime_count', 0),
                    'ValidState_Outliers': result.get('valid_states_count', 0),
                    'Recommendation': 'DROP' if hard_pct > drop_threshold else 'KEEP'
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Create detailed outlier_flags dataframe with individual detection methods
        # This allows selective exclusion during interpolation/LSTM training
        outlier_flags = pd.DataFrame(index=df.index)
        for col in columns_analyzed:
            if col in all_results and all_results[col]:
                result = all_results[col]
                # Overall flags
                outlier_flags[f'{col}_outlier'] = result['outliers']
                outlier_flags[f'{col}_hard'] = result.get('hard_outliers', False)
                outlier_flags[f'{col}_soft'] = result.get('soft_flags', False)
                
                # Individual detection method flags (for selective exclusion)
                outlier_flags[f'{col}_physical_outlier'] = result.get('physical_outliers_mask', False)
                outlier_flags[f'{col}_spike_outlier'] = result.get('spike_outliers_mask', False)
                outlier_flags[f'{col}_stuck_outlier'] = result.get('stuck_outliers_mask', False)
                outlier_flags[f'{col}_iqr_outlier'] = result.get('iqr_outliers_mask', False)
                outlier_flags[f'{col}_iqr_regime_outlier'] = result.get('iqr_regime_outliers_mask', False)
                outlier_flags[f'{col}_valid_state_outlier'] = result.get('valid_states_outliers_mask', False)
        
        # Convert boolean to int (0/1) for easier use
        outlier_flags = outlier_flags.astype(int)
        outlier_flags.to_excel(writer, sheet_name='Outlier_Flags')
        
        # Create data_with_outliers - use already timezone-naive df
        data_with_outliers = df[[c for c in columns_analyzed if c in df.columns]].copy()
        for col in columns_analyzed:
            if col in all_results and all_results[col]:
                data_with_outliers[f'{col}_is_outlier'] = all_results[col]['outliers'].astype(int)
                data_with_outliers[f'{col}_is_hard'] = all_results[col].get('hard_outliers', False).astype(int)
                data_with_outliers[f'{col}_is_soft'] = all_results[col].get('soft_flags', False).astype(int)
        
        data_with_outliers.to_excel(writer, sheet_name='Data_with_Outliers')

    csv_path = os.path.join(output_dir, f"{base_name}_outlier_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    
    # Save standalone boolean flags CSV (for merging with interpolation/LSTM data)
    flags_csv_path = os.path.join(output_dir, f"{base_name}_outlier_flags.csv")
    outlier_flags.to_csv(flags_csv_path, index=True)
    print(f"✅ Outlier flags saved to: {os.path.basename(flags_csv_path)}")
    print(f"   → Use this file to exclude IQR outliers from interpolation/LSTM training")
    print(f"   → Columns format: {{variable}}_iqr_outlier = 0/1")
    
    return excel_path, csv_path, flags_csv_path
    
# --- Main Execution Logic (Updated) ---

def main():
    """
    The main function that orchestrates the entire outlier detection workflow.
    """
    global config # Make config globally accessible for the report function
    print("Starting Enhanced Multi-Method Outlier Detection...")
    print("=" * 60)

    # --- Configuration and Path Setup ---
    script_dir = Path(__file__).parent
    config_paths = [
        script_dir.parent.parent / 'config.ini',  # Project root: scripts/1_data_preprocessing/ -> scripts/ -> project_root/
        script_dir.parent / 'config.ini',  # Fallback: scripts/
        Path('config.ini'),  # Current directory
    ]
    
    config_path = None
    for path in config_paths:
        if path.exists():
            config_path = str(path)
            break
    
    if not config_path:
        print("❌ ERROR: config.ini not found!")
        print("Searched in:")
        for p in config_paths:
            print(f"  - {p}")
        print("\nPlease ensure config.ini exists in the project root directory.")
        sys.exit(1)

    config = read_config(config_path)
    print("✅ Configuration loaded!")

    base_folder = config.get("paths", "base_folder")
    processed_root = config.get("paths", "processed_folder")
    # Resolve relative folders using config.ini directory (not current working dir)
    cfg_dir = getattr(config, '_config_dir', None)
    if cfg_dir:
        if base_folder and not os.path.isabs(base_folder):
            base_folder = os.path.abspath(os.path.join(cfg_dir, base_folder))
        if processed_root and not os.path.isabs(processed_root):
            processed_root = os.path.abspath(os.path.join(cfg_dir, processed_root))
    input_csv = config.get("paths", "input_csv")
    merged_folder = config.get("paths", "merged_folder", fallback="merged")
    base_name = os.path.splitext(input_csv)[0]
    merged_file = os.path.join(processed_root, merged_folder, base_name, f"{base_name}_merged.csv")
    # Also support 'merged_data' folder name
    alt_merged_file = os.path.join(processed_root, "merged_data", base_name, f"{base_name}_merged.csv")
    if not os.path.exists(merged_file) and os.path.exists(alt_merged_file):
        merged_file = alt_merged_file
    
    if not os.path.exists(merged_file):
        merged_file = os.path.join(base_folder, input_csv)
        print(f"⚠️ Merged file not found, using original: {input_csv}")
    else:
        print(f"✅ Using merged data: {os.path.basename(merged_file)}")

    if "interpolated" in os.path.basename(merged_file).lower():
        print("\n❌ ERROR: INTERPOLATED DATA DETECTED! Use MERGED data for outlier detection.")
        sys.exit(1)

    if not os.path.exists(merged_file):
        print(f"❌ Data file not found: {merged_file}")
        sys.exit(1)

    # --- Data Loading and Preparation ---
    columns_to_process_str = config.get("interpolation", "columns_to_process", fallback="")
    columns_to_process = [col.strip() for col in columns_to_process_str.split(",") if col.strip()]
    if not columns_to_process:
        print("❌ No columns specified in config.ini [interpolation] section!")
        sys.exit(1)

    df = pd.read_csv(merged_file)
    time_column = config.get("data", "time_column", fallback="Time")
    if time_column in df.columns:
        df[time_column] = pd.to_datetime(df[time_column])
        df.set_index(time_column, inplace=True)
    else:
        df.index = pd.to_datetime(df.iloc[:, 0])
        df.drop(columns=df.columns[0], inplace=True)
    
    # Remove timezone from index if present (for consistency)
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
        print(f"   ✓ Removed timezone from index")
    
    # Drop rows with NaT timestamps (from DST ambiguous times) to avoid duplicate index errors
    initial_len = len(df)
    df = df[df.index.notna()]
    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"   ✓ Dropped {dropped} rows with NaT timestamps (DST ambiguous times)")

    # Drop duplicate timestamps (can remain after DST tz_localize removal)
    n_dupes = df.index.duplicated().sum()
    if n_dupes > 0:
        print(f"   [WARNING] {n_dupes} duplicate timestamp(s) after timezone removal — keeping first occurrence")
        df = df[~df.index.duplicated(keep='first')]
        print(f"   ✓ Records after deduplication: {len(df):,}")

    print(f"✅ Data loaded: {len(df)} rows, {len(df.columns)} columns")

    existing_columns = [col for col in columns_to_process if col in df.columns]
    
    print("✅ Cleaning data...")
    for col in existing_columns:
        df[col] = clean_temp(df[col])

    analyzable_cols = [col for col in existing_columns if df[col].dropna().count() > 10]
    if not analyzable_cols:
        print("❌ No columns have sufficient data!")
        sys.exit(1)
    
    print(f"✅ Analyzing: {analyzable_cols}")

    # --- Setup Output Directories ---
    script_name = "outlier_detection"
    output_dir = os.path.join(processed_root, script_name, base_name)
    plots_root = config.get("paths", "plots_folder", fallback=base_folder)
    if cfg_dir and plots_root and not os.path.isabs(plots_root):
        plots_root = os.path.abspath(os.path.join(cfg_dir, plots_root))
    plots_dir = os.path.join(plots_root, script_name, base_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # --- Core Outlier Analysis Loop ---
    print("\nAnalyzing outliers using configured methods...")
    print("-" * 40)
    
    # 🔍 PRE-ANALYSIS DIAGNOSTIC: Check data types vs methods
    print("\n🔍 DATA TYPE DIAGNOSTIC (validates method selection):")
    print("=" * 60)
    
    # Get configuration
    od_config = config['outlier_detection']
    iqr_columns = [c.strip() for c in od_config.get('iqr_columns', '').split(',') if c.strip()]
    valid_states_columns = [c.strip() for c in od_config.get('valid_states_columns', '').split(',') if c.strip()]
    
    warnings_found = False
    
    for col in analyzable_cols:
        data_check = check_data_type(df[col], col)
        is_setpoint = is_setpoint_column(col)
        
        print(f"\n{col}:")
        print(f"  Data type: {data_check['type']}")
        print(f"  Unique values: {data_check['unique_count']:,} ({data_check['unique_ratio']:.1%})")
        print(f"  Is setpoint: {'Yes' if is_setpoint else 'No'}")
        
        # Check for problematic method combinations
        if col in valid_states_columns and data_check['type'] == 'continuous':
            print(f"  🔴 ERROR: valid_states enabled for CONTINUOUS data!")
            print(f"      This will mark normal values as outliers.")
            print(f"      ACTION: Remove '{col}' from valid_states_columns in config")
            warnings_found = True
        
        if col in iqr_columns and is_setpoint:
            print(f"  ⚠️  WARNING: IQR enabled for SETPOINT column!")
            print(f"      IQR will flag legitimate control actions as outliers.")
            print(f"      RECOMMENDATION: Remove '{col}' from iqr_columns in config")
            warnings_found = True
        
        if col in valid_states_columns and data_check['type'] == 'discrete':
            print(f"  ✅ valid_states is appropriate (discrete data)")
        
        if col in iqr_columns and not is_setpoint and data_check['type'] == 'continuous':
            print(f"  ✅ IQR is appropriate (continuous sensor data)")
    
    print("\n" + "=" * 60)
    
    if warnings_found:
        print("\n⚠️  CRITICAL WARNINGS FOUND - Review configuration before proceeding!")
        print("Do you want to continue anyway? (y/n): ", end='')
        try:
            response = input().strip().lower()
            if response != 'y':
                print("\n❌ Analysis cancelled. Please fix config.ini and re-run.")
                sys.exit(0)
        except:
            # If running in non-interactive mode, continue with warning
            print("[Non-interactive mode - continuing with warnings]")
    
    print("\n" + "-" * 40)
    print("Starting outlier detection...\n")
    
    all_results = {}
    
    for col in analyzable_cols:
        print(f"\nAnalyzing: {col}")
        results = analyze_column_outliers(df, col, config)
        if results:
            all_results[col] = results
            print(f"  Methods: {results['methods_used']}")
            hard_count = results.get('hard_count', 0)
            soft_count = results.get('soft_count', 0)
            hard_pct = results.get('hard_percentage', 0)
            soft_pct = results.get('soft_percentage', 0)
            
            if hard_count > 0:
                print(f"  Hard outliers: {hard_count} ({hard_pct:.1f}%) - DEFINITE ERRORS")
            if soft_count > 0:
                print(f"  Soft flags: {soft_count} ({soft_pct:.1f}%) - Statistical anomalies")
            if hard_count == 0 and soft_count == 0:
                print(f"  ✅ No outliers detected")
            
            drop_threshold = config.getfloat('outlier_detection', 'drop_threshold_percentage', fallback=10)
            if hard_pct > drop_threshold:
                print(f"  ⚠️ RECOMMEND DROPPING (based on hard outliers)")
    
    # --- Generate Outputs ---
    if not all_results:
        print("\nNo outliers found or no analysis performed. Skipping report generation.")
    else:
        plot_info_list = []
        
        if config.getboolean('outlier_detection', 'save_outlier_reports', fallback=True):
            print("\nGenerating reports...")
            excel_path, csv_path, flags_csv_path = create_outlier_report(all_results, df, output_dir, base_name, analyzable_cols)
        
        if config.getboolean('outlier_detection', 'create_outlier_plots', fallback=True):
            print("\nGenerating visualizations...")
            plot_info_list = create_individual_outlier_plots(df, all_results, plots_dir, base_name, analyzable_cols)
        
        # Generate Word report if plots were created
        if plot_info_list and DOCX_AVAILABLE:
            print("\nGenerating Word report...")
            word_path = create_word_report(plot_info_list, plots_dir, base_name, df)
            if word_path:
                print(f"✅ Word report created: {word_path}")
        
        print("\n" + "=" * 60)
        print("ENHANCED OUTLIER DETECTION COMPLETED!")
        print("=" * 60)
        total_outliers = sum(res['count'] for res in all_results.values() if res)
        total_hard = sum(res.get('hard_count', 0) for res in all_results.values() if res)
        total_soft = sum(res.get('soft_count', 0) for res in all_results.values() if res)
        print(f"📊 SUMMARY:")
        print(f"   • Columns analyzed: {len(analyzable_cols)}")
        print(f"   • Total outliers found: {total_outliers}")
        print(f"     - Hard outliers (errors): {total_hard}")
        print(f"     - Soft flags (IQR anomalies): {total_soft}")
        print(f"\n📁 OUTPUT FILES:")
        if 'excel_path' in locals():
            print(f"   • Excel report: {excel_path}")
            print(f"   • CSV summary: {csv_path}")
            print(f"   • Outlier flags (boolean): {flags_csv_path}")
            print(f"     └─ For interpolation/LSTM: exclude {{var}}_iqr_outlier == 1")
        if config.getboolean('outlier_detection', 'create_outlier_plots', fallback=True):
            print(f"   • Plots folder: {plots_dir}")
        if 'word_path' in locals() and word_path:
            print(f"   • Word report: {word_path}")
        print("\n💡 NOTE:")
        print("   • Physical limits, spikes, and stuck values = HARD outliers (definite errors)")
        print("   • IQR flags = SOFT flags (statistical anomalies, may be correct)")
        print("   • Recommendations based on HARD outliers only")
        print("\n📋 USAGE IN INTERPOLATION/LSTM:")
        print("   1. Load outlier flags CSV")
        print("   2. Exclude IQR outliers from training: df[df['{var}_iqr_outlier'] == 0]")
        print("   3. Keep IQR outliers for analysis: df[df['{var}_iqr_outlier'] == 1]")
        print("=" * 60)

if __name__ == "__main__":
    main()
