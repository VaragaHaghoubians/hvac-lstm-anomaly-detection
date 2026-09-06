# -*- coding: utf-8 -*-
"""
Enhanced Missing Data Analysis Script for AHU Monitoring Data 
================================================================================
Clean version - Part 1: Imports and Basic Functions
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import configparser
import sys
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment
import warnings
from pathlib import Path
import glob
from scipy.stats import chi2_contingency
import json

# Try to import python-docx for Word document generation
try:
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    DOCX_AVAILABLE = True
except ImportError:
    print("[WARNING] python-docx library is not installed!")
    print("Word report generation will be disabled.")
    print("To enable it, install using: pip install python-docx")
    DOCX_AVAILABLE = False

warnings.filterwarnings('ignore')

# Add parent directories to path to import config_utils
# From scripts/1_data_preprocessing/ -> scripts/ -> project_root/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Try to import config_utils, with fallback if not available
try:
    from config_utils import load_config
except ModuleNotFoundError:
    print("[WARNING] config_utils not found, using fallback config loading")
    
    def load_config(config_path):
        """Fallback function to load config WITH interpolation support."""
        config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        config.read(config_path, encoding='utf-8-sig')
        return config

def read_config(config_path="config.ini"):
    """Read configuration file with enhanced error handling and path validation."""
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
        # Remember config.ini directory to resolve relative paths against it
        try:
            config._config_dir = str(config_file.parent)
        except Exception:
            pass
        print(f"✅ Configuration loaded successfully")
        return config
    except Exception as e:
        print(f"❌ ERROR loading config file: {e}")
        sys.exit(1)

def clean_temperature_data(series):
    """Clean temperature data by removing symbols and converting to numeric."""
    if series.empty:
        return series
    
    cleaned = series.astype(str)
    cleaned = cleaned.str.replace(' °C', '', regex=False)
    cleaned = cleaned.str.replace(' Â°C', '', regex=False)
    cleaned = cleaned.str.replace('°C', '', regex=False)
    cleaned = cleaned.str.replace('Â°C', '', regex=False)
    cleaned = cleaned.str.replace(',', '.', regex=False)
    cleaned = cleaned.str.strip()
    cleaned = cleaned.replace('nan', np.nan)
    
    numeric_series = pd.to_numeric(cleaned, errors='coerce')
    return numeric_series

def find_data_file_with_search(config, base_name):
    """Find data file with intelligent search."""
    print("\n🔍 SEARCHING FOR DATA FILE...")
    base_folder = config.get("paths", "base_folder")
    processed_folder = config.get("paths", "processed_folder")
    # Resolve relative folders against config.ini directory (not current working dir)
    cfg_dir = getattr(config, '_config_dir', None)
    if cfg_dir:
        if base_folder and not os.path.isabs(base_folder):
            base_folder = os.path.abspath(os.path.join(cfg_dir, base_folder))
        if processed_folder and not os.path.isabs(processed_folder):
            processed_folder = os.path.abspath(os.path.join(cfg_dir, processed_folder))
    input_csv = config.get("paths", "input_csv")
    merged_folder = config.get("paths", "merged_folder", fallback="merged")
    
    # Priority search paths
    search_paths = [
        os.path.join(processed_folder, merged_folder, base_name, f"{base_name}_merged.csv"),
        os.path.join(processed_folder, "merged_data", base_name, f"{base_name}_merged.csv"),
        os.path.join(processed_folder, merged_folder, f"{base_name}_merged.csv"),
        os.path.join(processed_folder, f"{base_name}_merged.csv"),
        os.path.join(base_folder, f"{base_name}_merged.csv"),
        os.path.join(base_folder, input_csv)
    ]
    
    for i, path in enumerate(search_paths, 1):
        print(f"  {i}. Checking: {os.path.relpath(path)}")
        if os.path.exists(path):
            if "merged" in os.path.basename(path).lower():
                print(f"     ✅ FOUND MERGED DATA: {path}")
                return path, "merged"
            else:
                print(f"     ⚠️  FOUND ORIGINAL DATA: {path}")
                return path, "original"
    
    # Fallback: glob search
    glob_patterns = [
        os.path.join(processed_folder, "**", f"*{base_name}*merged*.csv"),
        os.path.join(base_folder, "**", f"*{base_name}*.csv")
    ]
    
    for pattern in glob_patterns:
        matches = glob.glob(pattern, recursive=True)
        valid_matches = [m for m in matches if 'interpolated' not in m.lower()]
        if valid_matches:
            found_file = valid_matches[0]
            source_type = "merged" if "merged" in found_file.lower() else "fallback"
            print(f"     ✅ FOUND via search: {found_file}")
            return found_file, source_type
    
    return None, "not_found"

def verify_data_source(file_path, base_name):
    """Verify data source to prevent analyzing wrong files."""
    print(f"\n🔒 VERIFYING DATA SOURCE")
    print(f"File: {os.path.basename(file_path)}")
    
    filename = os.path.basename(file_path).lower()
    
    # Check for dangerous keywords
    danger_keywords = ['interpolated', 'cleaned', 'filled', 'processed']
    found_danger = [kw for kw in danger_keywords if kw in filename]
    
    if found_danger:
        print(f"❌ CRITICAL ERROR: PROCESSED DATA DETECTED!")
        print(f"Danger keywords: {found_danger}")
        print(f"This script must analyze MERGED data, not processed data.")
        print(f"\nCORRECT SEQUENCE:")
        print(f"1. Raw data → Merge with weather")
        print(f"2. Merged data → Missing data analysis (THIS SCRIPT)")
        print(f"3. → Operation analysis → Outlier detection → Interpolation")
        sys.exit(1)
    
    # Check for weather data (indicates merged data)
    try:
        sample_df = pd.read_csv(file_path, nrows=5)
        weather_indicators = ['temp', 'humid', 'weather', 'extern']
        weather_columns = []
        
        for col in sample_df.columns:
            if any(indicator in col.lower() for indicator in weather_indicators):
                weather_columns.append(col)
        
        if weather_columns:
            print(f"✅ WEATHER DATA DETECTED: {weather_columns}")
            print(f"This appears to be merged data - proceeding.")
        else:
            print(f"⚠️  NO WEATHER DATA DETECTED")
            response = input("Continue anyway? (y/N): ").strip().lower()
            if response != 'y':
                sys.exit(1)
                
    except Exception as e:
        print(f"⚠️  Could not verify file content: {e}")
    
    return True

"""
Part 2: Core Analysis Functions
"""

def find_missing_periods(df, column_name, threshold_hours):
    """Find consecutive missing data periods above threshold."""
    if column_name not in df.columns:
        print(f"⚠️  Column '{column_name}' not found")
        return []

    print(f"   Analyzing missing periods in '{column_name}'...")
    
    is_missing = df[column_name].isna()
    missing_periods = []
    
    # Group consecutive missing values
    missing_groups = (is_missing != is_missing.shift()).cumsum()
    
    for group_id, group_data in is_missing.groupby(missing_groups):
        if group_data.iloc[0]:  # Missing data group
            group_indices = group_data.index
            start_time = group_indices[0]
            end_time = group_indices[-1]
            
            # Skip if index contains NaT values from DST transitions
            if pd.isna(start_time) or pd.isna(end_time):
                continue
            
            duration = end_time - start_time
            duration_hours = duration.total_seconds() / 3600
            
            # Skip invalid durations
            if pd.isna(duration_hours) or duration_hours < 0:
                continue
            
            # Estimate actual end time
            if len(df) > 1:
                avg_time_step = df.index.to_series().diff().dt.total_seconds().mean()
                actual_end_time = end_time + pd.Timedelta(seconds=avg_time_step)
                actual_duration_hours = (actual_end_time - start_time).total_seconds() / 3600
            else:
                actual_end_time = end_time
                actual_duration_hours = duration_hours
            
            if actual_duration_hours >= threshold_hours:
                # Classify gap severity
                if actual_duration_hours < 12:
                    severity = "Minor"
                elif actual_duration_hours < 48:
                    severity = "Moderate"
                elif actual_duration_hours < 168:  # 1 week
                    severity = "Major"
                else:
                    severity = "Critical"
                
                missing_periods.append({
                    'start_time': start_time,
                    'end_time': actual_end_time,
                    'duration_hours': actual_duration_hours,
                    'duration_days': actual_duration_hours / 24,
                    'missing_records': len(group_indices),
                    'severity': severity
                })
    
    if missing_periods:
        total_gap_hours = sum(p['duration_hours'] for p in missing_periods)
        print(f"      Found {len(missing_periods)} gaps ≥{threshold_hours}h (total: {total_gap_hours:.1f}h)")
    
    return missing_periods

def create_timestamp_missing_report(df, output_dir, base_name):
    """Create detailed timestamp-level missing data report."""
    print(f"\n📍 CREATING TIMESTAMP-LEVEL MISSING REPORT")
    
    missing_details = []
    
    for col in df.columns:
        missing_mask = df[col].isna()
        missing_count = missing_mask.sum()
        
        if missing_count == 0:
            print(f"   ✅ {col}: No missing data")
            continue
            
        print(f"   📊 {col}: {missing_count} missing points")
        
        missing_timestamps = df[missing_mask].index
        
        for timestamp in missing_timestamps:
            # Handle NaT (Not-a-Time) values from DST transitions
            if pd.isna(timestamp):
                missing_details.append({
                    'Timestamp': timestamp,
                    'Column': col,
                    'Date': None,
                    'Time': None,
                    'Hour_of_Day': None,
                    'Day_of_Week': None,
                    'Is_Weekend': None,
                    'Is_Working_Hours': None
                })
            else:
                missing_details.append({
                    'Timestamp': timestamp,
                    'Column': col,
                    'Date': timestamp.date(),
                    'Time': timestamp.time(),
                    'Hour_of_Day': timestamp.hour,
                    'Day_of_Week': timestamp.strftime('%A'),
                    'Is_Weekend': timestamp.weekday() >= 5,
                    'Is_Working_Hours': 8 <= timestamp.hour <= 17
                })
    
    if not missing_details:
        print("   ✅ No missing data found")
        return []
    
    missing_details_df = pd.DataFrame(missing_details)
    
    # Save detailed report
    timestamp_file = os.path.join(output_dir, f"{base_name}_missing_timestamps.csv")
    missing_details_df.to_csv(timestamp_file, index=False)
    print(f"✅ Timestamp report: {os.path.basename(timestamp_file)}")
    
    print(f"✅ Pattern summaries created")
    
    return missing_details

def analyze_correlations(df, output_dir, base_name):
    """Analyze correlations between missing data patterns."""
    print(f"\n🔗 ANALYZING MISSING DATA CORRELATIONS")
    
    missing_matrix = df.isna().astype(int)
    
    if missing_matrix.sum().sum() == 0:
        print("   ✅ No missing data - skipping correlation analysis")
        return {'correlations': [], 'correlation_matrix': pd.DataFrame()}
    
    print(f"   📊 Analyzing correlations between {len(df.columns)} columns...")
    
    missing_corr = missing_matrix.corr()
    high_correlations = []
    
    correlation_threshold = 0.5
    
    for i in range(len(missing_corr.columns)):
        for j in range(i+1, len(missing_corr.columns)):
            corr_val = missing_corr.iloc[i, j]
            
            if abs(corr_val) > correlation_threshold:
                col1, col2 = missing_corr.columns[i], missing_corr.columns[j]
                
                if corr_val > 0.8:
                    strength = "Very Strong"
                elif corr_val > 0.6:
                    strength = "Strong"
                else:
                    strength = "Moderate"
                
                high_correlations.append({
                    'Column_1': col1,
                    'Column_2': col2,
                    'Correlation': corr_val,
                    'Strength': strength
                })
    
    if high_correlations:
        corr_df = pd.DataFrame(high_correlations)
        corr_file = os.path.join(output_dir, f"{base_name}_correlations.csv")
        corr_df.to_csv(corr_file, index=False)
        print(f"   ✅ High correlations found: {len(high_correlations)}")
        
        for corr in high_correlations[:3]:
            print(f"      • {corr['Column_1']} ↔ {corr['Column_2']}: {corr['Correlation']:.3f}")
    else:
        print(f"   ✅ No significant correlations found")
    
    return {
        'correlations': high_correlations,
        'correlation_matrix': missing_corr
    }

def identify_sensor_failures(df, output_dir, base_name):
    """Identify potential sensor failures based on missing patterns."""
    print(f"\n🔧 IDENTIFYING SENSOR FAILURE PATTERNS")
    
    failure_patterns = []
    
    LONG_GAP_THRESHOLD = 48
    HIGH_MISSING_THRESHOLD = 30
    
    for col in df.columns:
        missing_count = df[col].isna().sum()
        
        if missing_count == 0:
            continue
            
        missing_pct = (missing_count / len(df)) * 100
        
        # Find consecutive missing periods
        is_missing = df[col].isna()
        missing_groups = (is_missing != is_missing.shift()).cumsum()
        max_gap_hours = 0
        gap_count = 0
        
        for group_id, group_data in is_missing.groupby(missing_groups):
            if group_data.iloc[0]:  # Missing data group
                duration_hours = (group_data.index[-1] - group_data.index[0]).total_seconds() / 3600
                max_gap_hours = max(max_gap_hours, duration_hours)
                gap_count += 1
        
        # Classify failure type
        failure_type = 'Normal gaps'
        priority = 'Low'
        
        if max_gap_hours > LONG_GAP_THRESHOLD:
            failure_type = 'Sensor failure'
            priority = 'High'
        elif missing_pct > HIGH_MISSING_THRESHOLD:
            failure_type = 'Extensive missing data'
            priority = 'Medium'
        elif gap_count > 50:
            failure_type = 'Communication issues'
            priority = 'Medium'
        
        failure_patterns.append({
            'Column': col,
            'Failure_Type': failure_type,
            'Priority': priority,
            'Missing_Percentage': missing_pct,
            'Max_Gap_Hours': max_gap_hours,
            'Gap_Count': gap_count
        })
    
    if failure_patterns:
        failure_df = pd.DataFrame(failure_patterns)
        failure_file = os.path.join(output_dir, f"{base_name}_sensor_failures.csv")
        failure_df.to_csv(failure_file, index=False)
        
        high_priority = [p for p in failure_patterns if p['Priority'] == 'High']
        if high_priority:
            print(f"   🚨 HIGH PRIORITY SENSORS:")
            for sensor in high_priority:
                print(f"      • {sensor['Column']}: {sensor['Failure_Type']}")
    
    return failure_patterns


def detect_system_shutdowns(df, output_dir, base_name, columns_to_check,
                            min_columns_fraction=0.8, min_duration_hours=12.0):
    """
    Detect periods where the entire monitoring system was offline:
    i.e., >= min_columns_fraction of tracked columns are simultaneously NaN
    for at least min_duration_hours.

    Parameters
    ----------
    df : DataFrame
        Reindexed dataframe with datetime index.
    output_dir : str
        Directory for output CSV.
    base_name : str
        Base name for output file.
    columns_to_check : list
        Columns considered "system-critical" for the outage check.
    min_columns_fraction : float
        Fraction of columns that must be NaN simultaneously (default 0.8).
    min_duration_hours : float
        Minimum continuous outage duration to report (default 12 h).

    Returns
    -------
    list of dict  : Each dict has keys start, end, duration_hours, columns_missing, pct_columns.
    pd.Series     : Boolean series (same index as df) — True during a shutdown.
    """
    print(f"\n🔌 DETECTING SYSTEM-WIDE SHUTDOWNS")
    print(f"   Threshold: ≥{min_columns_fraction*100:.0f}% of {len(columns_to_check)} columns missing "
          f"for ≥{min_duration_hours}h")

    valid_cols = [c for c in columns_to_check if c in df.columns]
    if not valid_cols:
        print("   ⚠️  No valid columns — skipping shutdown detection")
        return [], pd.Series(False, index=df.index)

    # At each timestamp, count how many tracked columns are NaN
    missing_counts = df[valid_cols].isna().sum(axis=1)
    required_missing = int(np.ceil(min_columns_fraction * len(valid_cols)))
    is_shutdown = missing_counts >= required_missing

    # Group consecutive shutdown timestamps
    groups = (is_shutdown != is_shutdown.shift()).cumsum()
    shutdowns = []
    for _, group in is_shutdown.groupby(groups):
        if not group.iloc[0]:          # not a shutdown segment
            continue
        start = group.index[0]
        end = group.index[-1]
        duration_hours = (end - start).total_seconds() / 3600
        if duration_hours < min_duration_hours:
            continue
        n_missing = int(df.loc[start, valid_cols].isna().sum())
        shutdowns.append({
            'start': start,
            'end': end,
            'duration_hours': round(duration_hours, 1),
            'duration_days': round(duration_hours / 24, 1),
            'columns_missing': n_missing,
            'pct_columns': round(n_missing / len(valid_cols) * 100, 1),
        })

    # Build a boolean mask covering all shutdown periods
    shutdown_mask = pd.Series(False, index=df.index)
    for s in shutdowns:
        shutdown_mask.loc[s['start']:s['end']] = True

    if shutdowns:
        total_shutdown_hours = sum(s['duration_hours'] for s in shutdowns)
        total_pct = total_shutdown_hours / (len(df) * (df.index[1] - df.index[0]).total_seconds() * 3600
                                            / 3600) * 100 if len(df) > 1 else 0
        # simpler: fraction of rows flagged
        total_pct = shutdown_mask.sum() / len(df) * 100
        print(f"   Found {len(shutdowns)} shutdown period(s)")
        print(f"   Total shutdown time: {total_shutdown_hours:.1f}h "
              f"({total_pct:.1f}% of dataset)")
        for i, s in enumerate(shutdowns, 1):
            print(f"   [{i}] {s['start'].date()} → {s['end'].date()} "
                  f"({s['duration_days']}d | {s['pct_columns']}% columns missing)")

        sd_df = pd.DataFrame(shutdowns)
        sd_file = os.path.join(output_dir, f"{base_name}_system_shutdowns.csv")
        sd_df.to_csv(sd_file, index=False)
        print(f"   ✅ Saved: {os.path.basename(sd_file)}")
    else:
        print("   ✅ No system-wide shutdowns detected")

    return shutdowns, shutdown_mask


def find_gap_durations(df, column, min_threshold_hours=0):
    """
    Extract list of gap durations in hours for a specific column.
    
    Parameters:
    -----------
    df : DataFrame
        The dataframe with datetime index
    column : str
        Column name to analyze
    min_threshold_hours : float
        Minimum gap duration to include (default=0 means ALL gaps)
        Set to 0 for manager report to see full distribution
        Set to 4 for config-based filtering
    
    Returns:
    --------
    list : Gap durations in hours
    """
    if column not in df.columns:
        return []
    
    is_missing = df[column].isna()
    missing_groups = (is_missing != is_missing.shift()).cumsum()
    
    gaps = []
    for group_id, group_data in is_missing.groupby(missing_groups):
        if group_data.iloc[0]:  # This is a missing data group
            # Skip if index contains NaT values from DST transitions
            if pd.isna(group_data.index[0]) or pd.isna(group_data.index[-1]):
                continue
            
            # Calculate gap duration
            duration_hours = (group_data.index[-1] - group_data.index[0]).total_seconds() / 3600
            
            # Skip invalid durations
            if pd.isna(duration_hours) or duration_hours < 0:
                continue
            
            # Include gap if it meets minimum threshold
            if duration_hours >= min_threshold_hours:
                gaps.append(duration_hours)
    
    return gaps


def set_cell_border(cell, **kwargs):
    """
    Set cell borders for table cells in Word document.
    This creates professional-looking table borders.
    
    Args:
        cell: Word table cell object
        **kwargs: Border settings (top, bottom, left, right, etc.)
    """
    # Get the table cell properties element
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    
    # Create borders element
    tcBorders = OxmlElement('w:tcBorders')
    
    # Define border sides
    for edge in ('left', 'top', 'right', 'bottom', 'insideH', 'insideV'):
        if edge in kwargs:
            # Create border element for this edge
            edge_data = kwargs.get(edge)
            edge_el = OxmlElement(f'w:{edge}')
            edge_el.set(qn('w:val'), 'single')
            edge_el.set(qn('w:sz'), '12')  # Border size (12 = 1.5pt)
            edge_el.set(qn('w:space'), '0')
            edge_el.set(qn('w:color'), '000000')  # Black border
            tcBorders.append(edge_el)
    
    tcPr.append(tcBorders)


def create_manager_report(df, output_dir, base_name, columns_to_check, config_threshold_hours=4.0):
    """
    Create a clean manager report as a Word document with plots.
    Each plot is presented in a table with:
    - Row 1: Plot title (column name)
    - Row 2: Plot image
    - Row 3: Description section
    
    Args:
        df: DataFrame with the data
        output_dir: Directory to save the Word report
        base_name: Base name for the output file
        columns_to_check: List of columns to analyze
        config_threshold_hours: Threshold for significant gaps
        
    Returns:
        str: Path to the generated Word report (or None if failed)
    """
    print(f"\n{'='*60}")
    print("CREATING MANAGER REPORT (WORD FORMAT)")
    print(f"{'='*60}")
    
    # Check if python-docx is available
    if not DOCX_AVAILABLE:
        print("[WARNING] Cannot generate Word report - python-docx library not installed")
        print("To enable Word report generation, run: pip install python-docx")
        return None
    
    # List to store plot information (title, path, description)
    plots_info = []
    
    # =========================================================================
    # PLOT 1: MISSING DATA OVERVIEW
    # =========================================================================
    print("\n1. Generating Missing Data Overview plot...")
    
    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.4)
    
    # Top: Bar chart
    ax1 = fig.add_subplot(gs[0])
    missing_stats = []
    for col in columns_to_check:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            missing_stats.append({
                'Variable': col,
                'Missing_Pct': missing_pct,
                'Missing_Count': missing_count
            })
    
    if missing_stats:
        stats_df = pd.DataFrame(missing_stats)
        colors = ['#d73027' if x > 20 else '#fc8d59' if x > 10 else '#fee090' if x > 5 else '#91bfdb' 
                 for x in stats_df['Missing_Pct']]
        
        bars = ax1.barh(stats_df['Variable'], stats_df['Missing_Pct'], color=colors, edgecolor='black')
        ax1.set_xlabel('Dati Mancanti (%)', fontweight='bold', fontsize=11)
        date_range_str = f"{df.index.min().strftime('%d/%m/%Y')} - {df.index.max().strftime('%d/%m/%Y')}"
        ax1.set_title(f'Percentuale Dati Mancanti per Variabile\n{date_range_str}', fontweight='bold', fontsize=13, pad=15)
        ax1.grid(axis='x', alpha=0.3)
        ax1.set_xlim(0, max(stats_df['Missing_Pct'].max() * 1.1, 5))
        
        for i, (bar, pct, count) in enumerate(zip(bars, stats_df['Missing_Pct'], stats_df['Missing_Count'])):
            ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'{pct:.1f}% ({count:,} record)',
                    va='center', fontsize=9)
    
    # Bottom: Summary table
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    
    table_data = []
    for stat in missing_stats:
        col = stat['Variable']
        # Don't shorten variable names - keep full names
        all_gaps = find_gap_durations(df, col, min_threshold_hours=0)
        significant_gaps = find_gap_durations(df, col, min_threshold_hours=config_threshold_hours)
        
        table_data.append([
            col,  # Use full variable name
            f"{stat['Missing_Pct']:.1f}%",
            f"{stat['Missing_Count']:,}",
            len(all_gaps),
            len(significant_gaps),
            f"{max(all_gaps):.1f}h" if all_gaps else "0h"
        ])
    
    table = ax2.table(
        cellText=table_data,
        colLabels=['Variabile', '% Manc.', '# Manc.', 'Gap Tot.', f'Gap ≥{config_threshold_hours}h', 'Gap Max'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Set column widths - make first column wider for long variable names
    for i in range(len(table_data) + 1):
        table[(i, 0)].set_width(0.35)  # Variable name column - 35% width
        for j in range(1, 6):
            table[(i, j)].set_width(0.13)  # Other columns - 13% each
    
    # Format header row
    for i in range(6):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Left-align variable names in first column for better readability
    for i in range(1, len(table_data) + 1):
        table[(i, 0)].set_text_props(ha='left')
        table[(i, 0)].PAD = 0.05
    
    info_text = f"Dataset: {len(df):,} record | Periodo: {df.index.min().date()} - {df.index.max().date()} | Giorni: {(df.index.max() - df.index.min()).days}"
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=9, style='italic')
    
    plot1_path = os.path.join(output_dir, f"{base_name}_missing_overview.png")
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Description for plot 1
    total_records = len(df)
    total_missing = df[columns_to_check].isna().sum().sum()
    overall_missing_pct = (total_missing / (total_records * len(columns_to_check))) * 100
    
    desc1 = (
        f"Questo grafico mostra la percentuale di dati mancanti per ciascuna variabile monitorata. "
        f"Il dataset contiene {total_records:,} record totali con un tasso complessivo di dati mancanti "
        f"del {overall_missing_pct:.2f}%.\n\n"
        f"La tabella sottostante fornisce statistiche dettagliate per ogni variabile, inclusi:\n"
        f"• Percentuale e numero totale di dati mancanti\n"
        f"• Numero totale di gap nella serie temporale\n"
        f"• Numero di gap significativi (≥{config_threshold_hours}h)\n"
        f"• Durata massima del gap più lungo\n\n"
        f"Interpretazione dei colori:\n"
        f"• Blu = Buona qualità (<5% mancanti)\n"
        f"• Giallo = Qualità accettabile (5-10% mancanti)\n"
        f"• Arancione = Necessita attenzione (10-20% mancanti)\n"
        f"• Rosso = Qualità critica (>20% mancanti)"
    )
    
    # Add date range to title
    date_range = f"{df.index.min().strftime('%d/%m/%Y')} - {df.index.max().strftime('%d/%m/%Y')}"
    plot1_title = f"Panoramica Dati Mancanti ({date_range})"
    plots_info.append((plot1_title, plot1_path, desc1))
    print("   ✓ Plot 1 saved")
    
    # =========================================================================
    # PLOT 2: GAP LENGTH DISTRIBUTION
    # =========================================================================
    print("\n2. Generating Gap Length Distribution plot...")
    
    fig, ax = plt.subplots(figsize=(11, 8.5))
    
    all_gaps = []
    for col in columns_to_check:
        if col in df.columns:
            col_gaps = find_gap_durations(df, col, min_threshold_hours=0)
            all_gaps.extend(col_gaps)
    
    if all_gaps and len(all_gaps) > 0:
        max_gap = max(all_gaps)
        
        if max_gap <= 0.5:
            bins = [0, 0.1, 0.25, 0.5]
            bin_labels = ['<6min', '6-15min', '15-30min']
        elif max_gap <= 1:
            bins = [0, 0.5, 1]
            bin_labels = ['<30min', '30min-1h']
        elif max_gap <= 4:
            bins = [0, 0.5, 1, 4]
            bin_labels = ['<30min', '30min-1h', '1-4h']
        elif max_gap <= 24:
            bins = [0, 0.5, 1, 4, 24]
            bin_labels = ['<30min', '30min-1h', '1-4h', '4-24h']
        else:
            bins = [0, 0.5, 1, 4, 24, max_gap + 1]
            bin_labels = ['<30min', '30min-1h', '1-4h', '4-24h', '>24h']
        
        counts, edges, patches = ax.hist(all_gaps, bins=bins, edgecolor='black', 
                                        linewidth=1.5, alpha=0.8, color='steelblue')
        
        colors_map = ['#91bfdb', '#ffffbf', '#fee090', '#fc8d59', '#d73027']
        for i, patch in enumerate(patches):
            if i < len(colors_map):
                patch.set_facecolor(colors_map[i])
        
        for count, patch in zip(counts, patches):
            if count > 0:
                height = patch.get_height()
                ax.text(patch.get_x() + patch.get_width()/2, height,
                       f'{int(count)}\ngaps',
                       ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_xlabel('Durata Gap (ore)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Numero di Gap', fontweight='bold', fontsize=12)
        date_range_str = f"{df.index.min().strftime('%d/%m/%Y')} - {df.index.max().strftime('%d/%m/%Y')}"
        ax.set_title(f'Distribuzione Lunghezza Gap\n{date_range_str}', fontweight='bold', fontsize=14, pad=20)
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)
        
        bin_centers = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
        ax.set_xticks(bin_centers)
        ax.set_xticklabels(bin_labels, fontsize=10)
        
        safe_gaps = sum(1 for g in all_gaps if g <= 4)
        medium_gaps = sum(1 for g in all_gaps if 4 < g <= 24)
        long_gaps = sum(1 for g in all_gaps if g > 24)
        total_gaps = len(all_gaps)
        
        safe_pct = safe_gaps / total_gaps * 100 if total_gaps > 0 else 0
        medium_pct = medium_gaps / total_gaps * 100 if total_gaps > 0 else 0
        long_pct = long_gaps / total_gaps * 100 if total_gaps > 0 else 0
        
        interpretation = (
            f"VALUTAZIONE SICUREZZA INTERPOLAZIONE\n"
            f"{'='*40}\n"
            f"Sicuri (≤4h):          {safe_gaps:4d} gaps ({safe_pct:5.1f}%)\n"
            f"Rischio medio (4-24h): {medium_gaps:4d} gaps ({medium_pct:5.1f}%)\n"
            f"Alto rischio (>24h):   {long_gaps:4d} gaps ({long_pct:5.1f}%)\n"
            f"{'='*40}\n"
            f"Totale gaps:           {total_gaps:4d}\n"
            f"Max durata gap:        {max_gap:.1f}h\n"
        )
        
        if safe_pct > 90:
            recommendation = "✓ Ottimo - sicuro interpolare"
            box_color = '#d4edda'
        elif safe_pct > 70:
            recommendation = "✓ Buono - per lo più sicuro"
            box_color = '#d4edda'
        elif safe_pct > 50:
            recommendation = "⚠ Moderato - rivedere gap medi"
            box_color = '#fff3cd'
        else:
            recommendation = "⚠ Attenzione - molti gap lunghi"
            box_color = '#f8d7da'
        
        interpretation += f"\nRaccomandazione: {recommendation}"
        
        ax.text(0.98, 0.97, interpretation,
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=1', facecolor=box_color, 
                         edgecolor='black', linewidth=1.5),
                family='monospace')
    else:
        ax.text(0.5, 0.5, 'NESSUN GAP NEI DATI\nQualità: ECCELLENTE', 
               ha='center', va='center', transform=ax.transAxes, 
               fontsize=20, color='green', fontweight='bold')
        date_range_str = f"{df.index.min().strftime('%d/%m/%Y')} - {df.index.max().strftime('%d/%m/%Y')}"
        ax.set_title(f'Distribuzione Lunghezza Gap\n{date_range_str}', fontweight='bold', fontsize=14)
    
    plot2_path = os.path.join(output_dir, f"{base_name}_gap_distribution.png")
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Description for plot 2
    total_gaps = len(all_gaps)
    max_gap = max(all_gaps) if all_gaps else 0
    safe_pct = (safe_gaps / total_gaps * 100) if total_gaps > 0 else 0
    
    desc2 = (
        f"Questo istogramma mostra la distribuzione della lunghezza dei gap nei dati. "
        f"Sono stati identificati {total_gaps} gap totali, con una durata massima di {max_gap:.1f} ore.\n\n"
        f"Valutazione della sicurezza per l'interpolazione:\n"
        f"• Gap ≤4h: {safe_pct:.1f}% - Sicuri da interpolare\n"
        f"• Gap 4-24h: Rischio medio - Richiedono revisione\n"
        f"• Gap >24h: Alto rischio - Sconsigliata interpolazione diretta\n\n"
        f"La soglia configurata per i gap significativi è {config_threshold_hours} ore. "
        f"Gap più lunghi potrebbero indicare guasti ai sensori o interruzioni del sistema di acquisizione dati.\n\n"
        f"L'analisi della distribuzione aiuta a determinare la strategia di interpolazione più appropriata "
        f"e a identificare eventuali problemi sistematici nella raccolta dati."
    )
    
    # Add date range to title (reuse from plot 1)
    plot2_title = f"Distribuzione Lunghezza Gap ({date_range})"
    plots_info.append((plot2_title, plot2_path, desc2))
    print("   ✓ Plot 2 saved")
    
    # =========================================================================
    # PLOT 3: GAPS TIMELINE
    # =========================================================================
    print("\n3. Generating Gaps Timeline plot...")
    
    n_cols = len([c for c in columns_to_check if c in df.columns])
    # Make figure wider (14 inches) and taller for better visibility
    fig, axes = plt.subplots(n_cols, 1, figsize=(14, max(10, n_cols*1.8)), sharex=True)
    
    # Increase spacing between subplots for readability
    plt.subplots_adjust(hspace=0.4, left=0.08, right=0.98)
    
    if n_cols == 1:
        axes = [axes]
    
    plot_idx = 0
    for col in columns_to_check:
        if col not in df.columns:
            continue
        
        ax = axes[plot_idx]
        is_missing = df[col].isna()
        
        ax.fill_between(df.index, 0, is_missing.astype(int), 
                       color='#d73027', alpha=0.7, step='post', linewidth=0)
        
        missing_pct = (is_missing.sum() / len(df)) * 100
        gaps = find_gap_durations(df, col, min_threshold_hours=0)
        
        # Set variable name as subplot title (above the plot)
        ax.set_title(col, fontsize=10, fontweight='bold', loc='left', pad=8)
        
        ax.set_ylim(-0.05, 1.15)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Presenti', 'Mancanti'], fontsize=9)
        ax.grid(axis='x', alpha=0.3)
        ax.set_facecolor('#f0f0f0')
        
        stats_text = f'{missing_pct:.1f}% manc. | {len(gaps)} gap'
        ax.text(0.99, 0.5, stats_text, transform=ax.transAxes,
               ha='right', va='center', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                        alpha=0.8, edgecolor='gray'))
        
        plot_idx += 1
    
    axes[-1].set_xlabel('Data', fontweight='bold', fontsize=12)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    
    date_range_str = f"{df.index.min().strftime('%d/%m/%Y')} - {df.index.max().strftime('%d/%m/%Y')}"
    fig.suptitle(f'Timeline Gap Temporale (Rosso = Dati Mancanti)\n{date_range_str}', 
                fontweight='bold', fontsize=14, y=0.995)
    
    guide_text = (
        'Interpretazione: '
        'Bande rosse verticali su più variabili = downtime sistema | '
        'Patch rosse sparse = problemi sensori individuali | '
        'Linee orizzontali rosse = variabile costantemente mancante'
    )
    fig.text(0.5, 0.005, guide_text, ha='center', fontsize=9, style='italic', wrap=True)
    
    plt.tight_layout(rect=[0, 0.015, 1, 0.99])
    
    plot3_path = os.path.join(output_dir, f"{base_name}_gaps_timeline.png")
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Description for plot 3
    desc3 = (
        f"Questo grafico mostra la distribuzione temporale dei dati mancanti per ciascuna variabile nel tempo. "
        f"Le aree rosse indicano periodi di dati mancanti, mentre le aree bianche indicano dati presenti.\n\n"
        f"Interpretazione dei pattern:\n"
        f"• Bande verticali rosse su più variabili = Downtime completo del sistema di monitoraggio\n"
        f"• Patch rosse sparse = Problemi individuali dei sensori o interruzioni localizzate\n"
        f"• Linee orizzontali rosse continue = Variabile costantemente mancante (possibile sensor guasto)\n\n"
        f"Questo grafico aiuta a identificare:\n"
        f"• Pattern temporali nei malfunzionamenti del sistema\n"
        f"• Correlazione tra guasti di diversi sensori\n"
        f"• Periodi critici che richiedono manutenzione\n"
        f"• Efficacia delle strategie di monitoraggio e raccolta dati\n\n"
        f"L'analisi temporale è fondamentale per pianificare interventi di manutenzione preventiva "
        f"e migliorare l'affidabilità del sistema di acquisizione dati."
    )
    
    # Add date range to title (reuse from plot 1)
    plot3_title = f"Timeline Gap Temporale ({date_range})"
    plots_info.append((plot3_title, plot3_path, desc3))
    print("   ✓ Plot 3 saved")
    
    # =========================================================================
    # CREATE WORD DOCUMENT
    # =========================================================================
    print("\n4. Creating Word document...")
    
    try:
        doc = Document()
        
        # Set margins
        sections = doc.sections
        for section in sections:
            section.top_margin = Inches(0.5)
            section.bottom_margin = Inches(0.5)
            section.left_margin = Inches(0.75)
            section.right_margin = Inches(0.75)
        
        # Add title
        start_date = df.index.min()
        end_date = df.index.max()
        date_range_str = f"Periodo di analisi: {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}"
        
        title = doc.add_heading('Analisi Dati Mancanti - Sistema HVAC', level=0)
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
        for idx, (plot_title, image_path, description) in enumerate(plots_info, 1):
            print(f"   Adding plot {idx}/{len(plots_info)}: {plot_title}")
            
            # Create table with 3 rows
            table = doc.add_table(rows=3, cols=1)
            table.style = 'Table Grid'
            table.autofit = False
            table.allow_autofit = False
            
            # Set table width to page width (6.5 inches for standard margins)
            for row in table.rows:
                for cell in row.cells:
                    cell.width = Inches(6.5)
            
            # Row 1: Title
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
            title_para.text = plot_title
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
                run.add_picture(image_path, width=Inches(6.3))  # Smaller to fit with margins
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
            if idx < len(plots_info):
                doc.add_paragraph()
        
        # Save Word document
        word_path = os.path.join(output_dir, f"{base_name}_Missing_Data_Report.docx")
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


"""
Part 3: Reporting and Visualization Functions
"""

def create_excel_report(df, processed_dir, base_name):
    """Create comprehensive Excel report."""
    print(f"\n📊 CREATING EXCEL REPORT")
    
    # 🔧 Safety: ensure index and datetime columns are timezone-naive
    df = df.copy()  # Work on a copy to avoid modifying original
    
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
        print(f"   ✓ Removed timezone from index")
    elif df.index.dtype == "object":
        # Handle case where index is object dtype with tz-aware datetimes
        try:
            idx = pd.to_datetime(df.index, errors="coerce")
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_localize(None)
                print(f"   ✓ Converted and removed timezone from object index")
            df.index = idx
        except Exception as e:
            print(f"   ⚠️  Could not normalize index timezone: {e}")
    
    # Also strip timezone from any datetime columns
    for col in df.select_dtypes(include=["datetimetz"]).columns:
        df[col] = df[col].dt.tz_localize(None)
        print(f"   ✓ Removed timezone from column '{col}'")
    
    excel_path = os.path.join(processed_dir, f"{base_name}_Missing_Data_Report.xlsx")
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        
        # Sheet 1: Summary
        summary_data = []
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            
            summary_data.append({
                'Column': col,
                'Total_Records': len(df),
                'Missing_Count': missing_count,
                'Missing_Percentage': missing_pct,
                'Quality_Rating': ('Excellent' if missing_pct < 1 else
                                 'Good' if missing_pct < 5 else
                                 'Fair' if missing_pct < 15 else
                                 'Poor' if missing_pct < 30 else 'Critical')
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Sheet 2: Missing Data Matrix (sample)
        missing_matrix = df.isna().astype(int)
        if len(missing_matrix) > 5000:
            missing_sample = missing_matrix.iloc[:5000]
        else:
            missing_sample = missing_matrix
        
        missing_sample.to_excel(writer, sheet_name='Missing_Matrix', index=True)
        
    
    print(f"✅ Excel report: {os.path.basename(excel_path)}")
    return excel_path

def filter_seasonal_data(df, season_name):
    """Filter data by season.
    
    Parameters:
    -----------
    df : DataFrame
        Full dataframe with datetime index
    season_name : str
        Either 'Summer2025' or 'Winter2026'
    
    Returns:
    --------
    DataFrame : Filtered data for the specified season
    """
    if season_name == 'Summer2025':
        # Summer 2025: July 9 to October 15, 2025
        start_date = pd.Timestamp('2025-07-09')
        end_date = pd.Timestamp('2025-10-15')
    elif season_name == 'Winter2026':
        # Winter 2026: October 16, 2025 to end of available data
        start_date = pd.Timestamp('2025-10-16')
        end_date = df.index.max()  # Use all available data after Oct 16
    else:
        raise ValueError(f"Unknown season: {season_name}. Use 'Summer2025' or 'Winter2026'")
    
    # Filter data
    mask = (df.index >= start_date) & (df.index <= end_date)
    filtered_df = df[mask].copy()
    
    print(f"   Filtered {season_name}: {len(filtered_df)} records from {filtered_df.index.min()} to {filtered_df.index.max()}")
    return filtered_df


def create_seasonal_overview_plot(season_df, plots_dir, base_name, season_name):
    """Create overview plot for a specific season with exact dimensions.
    
    Parameters:
    -----------
    season_df : DataFrame
        Filtered data for the season
    plots_dir : str
        Output directory for plots
    base_name : str
        Base name for output files
    season_name : str
        'Summer2025' or 'Winter2026'
    """
    # Calculate missing percentage for ALL columns (including 0%)
    missing_pct = (season_df.isna().sum() / len(season_df)) * 100
    missing_pct = missing_pct.sort_values(ascending=False)  # Sort descending, keep all values including 0%
    
    # Figure size: 5368 x 3543 pixels at 300 DPI = 17.893333 x 11.81 inches
    fig_width = 5368 / 300
    fig_height = 3543 / 300
    
    # Overview plot - Use 1x2 layout with more space for the bar chart
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), gridspec_kw={'width_ratios': [1.2, 0.8]})
    date_range_str = f"{season_df.index.min().strftime('%d/%m/%Y')} - {season_df.index.max().strftime('%d/%m/%Y')}"
    
    # Determine the season label for title
    season_label = f'C1_UTA1_{season_name}'
    
    fig.suptitle(f'Missing Data Analysis: {season_label}\n{date_range_str}', fontsize=16, fontweight='bold')
    
    # Plot 1: Missing percentage by column (including 0%)
    ax1 = axes[0]
    
    # Always plot all columns, even those with 0% missing
    colors = ['#d73027' if x > 20 else '#fc8d59' if x > 10 else '#fee090' if x > 5 else '#91bfdb' 
             for x in missing_pct.values]
    missing_pct.plot(kind='barh', ax=ax1, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_title('Missing Data by Column (%)', fontweight='bold', fontsize=13)
    ax1.set_xlabel('Missing Percentage (%)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (idx, val) in enumerate(missing_pct.items()):
        ax1.text(val + 0.3, i, f'{val:.1f}%', va='center', fontsize=9)
    
    # Plot 2: Summary statistics
    ax2 = axes[1]
    ax2.axis('off')
    
    total_records = len(season_df)
    total_missing = season_df.isna().sum().sum()
    overall_missing_pct = (total_missing / (total_records * len(season_df.columns))) * 100
    
    summary_text = f"""
DATA SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Records:     {total_records:,}
Total Columns:     {len(season_df.columns)}
Time Span:         {(season_df.index.max() - season_df.index.min()).days} days
Period:            {season_df.index.min().date()} to
                   {season_df.index.max().date()}

MISSING DATA
━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Missing:     {total_missing:,}
Missing Rate:      {overall_missing_pct:.2f}%
Columns Affected:  {(season_df.isna().sum() > 0).sum()}/{len(season_df.columns)}

QUALITY RATING
━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    if overall_missing_pct < 1:
        quality = "✓ EXCELLENT"
        quality_color = '#d4edda'
    elif overall_missing_pct < 5:
        quality = "✓ GOOD"
        quality_color = '#d4edda'
    elif overall_missing_pct < 10:
        quality = "⚠ ACCEPTABLE"
        quality_color = '#fff3cd'
    else:
        quality = "⚠ NEEDS ATTENTION"
        quality_color = '#f8d7da'
    
    summary_text += f"{quality}"
    
    ax2.text(0.1, 0.9, summary_text, transform=ax2.transAxes, fontsize=11,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor=quality_color, 
                     alpha=0.9, edgecolor='black', linewidth=2))
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    overview_plot = os.path.join(plots_dir, f"{base_name}_{season_name}_overview.png")
    plt.savefig(overview_plot, dpi=300)
    plt.close()
    
    print(f"   ✅ {season_name} overview plot: {os.path.basename(overview_plot)}")
    return overview_plot


def create_visualizations(df, plots_dir, base_name, columns_to_check):
    """Create missing data visualizations."""
    print(f"\n📊 CREATING VISUALIZATIONS")
    
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create seasonal overview plots for both Summer 2025 and Winter 2026
    seasons = ['Summer2025', 'Winter2026']
    
    # First pass: collect both seasonal dataframes to find common columns
    season_dfs = {}
    for season in seasons:
        season_df = filter_seasonal_data(df, season)
        if len(season_df) > 0:
            season_dfs[season] = season_df
    
    # Identify common columns for fair comparison
    common_columns = None
    if len(season_dfs) == 2:
        summer_cols = set(season_dfs['Summer2025'].columns)
        winter_cols = set(season_dfs['Winter2026'].columns)
        common_columns = list(summer_cols & winter_cols)
        
        summer_only = summer_cols - winter_cols
        winter_only = winter_cols - summer_cols
        
        if summer_only or winter_only:
            print(f"\n   ℹ️  Column differences detected:")
            if summer_only:
                print(f"      • Only in Summer 2025: {sorted(summer_only)}")
            if winter_only:
                print(f"      • Only in Winter 2026: {sorted(winter_only)}")
            print(f"      • Common columns ({len(common_columns)}): Using these for comparison")
    
    # Second pass: create plots with common columns only
    for season in seasons:
        print(f"\n   Creating {season} analysis...")
        
        try:
            if season not in season_dfs:
                print(f"   ⚠️ No data available for {season}, skipping...")
                continue
            
            season_df = season_dfs[season]
            
            # Filter to common columns if both seasons exist
            if common_columns is not None and len(common_columns) > 0:
                season_df = season_df[sorted(common_columns)]
                print(f"      Using {len(common_columns)} common columns")
            
            # Create overview plot with exact dimensions (5368x3543 pixels)
            create_seasonal_overview_plot(season_df, plots_dir, base_name, season)
            
        except Exception as e:
            print(f"   ⚠️ Error creating {season} plot: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✅ Seasonal overview plots created for Summer 2025 and Winter 2026")
    
    # Diagnostic: Show which columns have missing data in each season
    if len(season_dfs) == 2:
        summer_missing_cols = [col for col in season_dfs['Summer2025'].columns 
                              if season_dfs['Summer2025'][col].isna().sum() > 0]
        winter_missing_cols = [col for col in season_dfs['Winter2026'].columns 
                              if season_dfs['Winter2026'][col].isna().sum() > 0]
        
        print(f"\n📊 MISSING DATA COLUMN ANALYSIS:")
        print(f"   • Summer 2025: {len(summer_missing_cols)} columns with missing data")
        print(f"   • Winter 2026: {len(winter_missing_cols)} columns with missing data")
        
        only_summer = set(summer_missing_cols) - set(winter_missing_cols)
        only_winter = set(winter_missing_cols) - set(summer_missing_cols)
        
        if only_summer:
            print(f"   • Missing data ONLY in Summer: {sorted(only_summer)}")
        if only_winter:
            print(f"   • Missing data ONLY in Winter: {sorted(only_winter)}")
        if not only_summer and not only_winter:
            print(f"   • Both seasons have missing data in the same columns")
    
    # Individual column plots for columns with missing data
    for col in df.columns:
        if df[col].isna().sum() > 0:
            # Use 1x2 layout instead of 2x2 to avoid empty plots
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            date_range_str = f"{df.index.min().strftime('%d/%m/%Y')} - {df.index.max().strftime('%d/%m/%Y')}"
            fig.suptitle(f'Missing Data Analysis: {col}\n{date_range_str}', fontsize=14, fontweight='bold')
            
            # Left plot: Weekly pattern
            ax1 = axes[0]
            weekly_pct = df[col].isna().resample("W").mean() * 100
            if not weekly_pct.empty:
                weekly_pct.plot(kind="bar", ax=ax1, color="coral", alpha=0.8)
                ax1.set_title('Weekly Missing %', fontweight='bold')
                ax1.set_ylabel('% Missing', fontweight='bold')
                ax1.set_xlabel('Week', fontweight='bold')
                ax1.tick_params(axis='x', rotation=45)
                ax1.grid(True, alpha=0.3)
            
            # Right plot: Gap duration distribution
            ax2 = axes[1]
            is_missing = df[col].isna()
            missing_groups = (is_missing != is_missing.shift()).cumsum()
            gap_durations = []
            
            for group_id, group_data in is_missing.groupby(missing_groups):
                if group_data.iloc[0]:
                    # Skip if index contains NaT values
                    if pd.isna(group_data.index[0]) or pd.isna(group_data.index[-1]):
                        continue
                    duration_hours = (group_data.index[-1] - group_data.index[0]).total_seconds() / 3600
                    # Only add finite duration values
                    if not pd.isna(duration_hours) and duration_hours >= 0:
                        gap_durations.append(duration_hours)
            
            if gap_durations:
                ax2.hist(gap_durations, bins=min(20, len(gap_durations)), 
                        color='gold', alpha=0.7, edgecolor='black')
                ax2.set_title('Gap Duration Distribution', fontweight='bold')
                ax2.set_xlabel('Gap Duration (hours)', fontweight='bold')
                ax2.set_ylabel('Frequency', fontweight='bold')
                ax2.grid(True, alpha=0.3)
                
                # Add statistics text
                max_gap = max(gap_durations)
                mean_gap = sum(gap_durations) / len(gap_durations)
                stats_text = f'Gaps: {len(gap_durations)}\nMax: {max_gap:.1f}h\nMean: {mean_gap:.1f}h'
                ax2.text(0.97, 0.97, stats_text, transform=ax2.transAxes,
                        ha='right', va='top', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            col_plot = os.path.join(plots_dir, f"{base_name}_{col.replace(' ', '_')}_analysis.png")
            plt.savefig(col_plot, dpi=300)
            plt.close()
    
    print(f"✅ Individual column plots created")
    print("\n📅 Creating combined daily pattern visualization...")
    create_combined_daily_pattern_visualization(df, plots_dir, base_name, columns_to_check)
    create_weekly_summary_heatmap(df, plots_dir, base_name, columns_to_check)
    
    print(f"✅ All visualizations created successfully")

def create_combined_daily_pattern_visualization(df, plots_dir, base_name, columns_to_check):
    """
    Create a single combined missing data pattern plot for all variables.
    Each variable gets one row on a shared time axis, making co-missing patterns
    immediately visible without producing a separate file per column.
    """
    print(f"\nCreating combined daily pattern visualization...")

    cols_with_missing = [
        col for col in columns_to_check
        if col in df.columns and df[col].isna().sum() > 0
    ]

    if not cols_with_missing:
        print("  No missing data found — skipping combined daily pattern plot")
        return

    n_vars = len(cols_with_missing)
    fig, axes = plt.subplots(n_vars, 1, figsize=(14, max(8, n_vars * 1.6)), sharex=True)
    if n_vars == 1:
        axes = [axes]

    date_range_str = (
        f"{df.index.min().strftime('%d/%m/%Y')} - "
        f"{df.index.max().strftime('%d/%m/%Y')}"
    )
    fig.suptitle(
        f'Missing Data Pattern: All Variables\n{date_range_str}',
        fontsize=15, fontweight='bold', y=0.995
    )

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='|', color='w', markerfacecolor='#91bfdb',
               markeredgecolor='#91bfdb', markersize=12, linewidth=2,
               label='Data Present'),
        Line2D([0], [0], marker='|', color='w', markerfacecolor='#d73027',
               markeredgecolor='#d73027', markersize=12, linewidth=2,
               label='Missing Data'),
    ]

    for ax_idx, (ax, col) in enumerate(zip(axes, cols_with_missing)):
        is_missing = df[col].isna().astype(int)
        present_dates = df.index[is_missing == 0]
        missing_dates = df.index[is_missing == 1]

        ax.scatter(present_dates, [0.5] * len(present_dates),
                   c='#91bfdb', marker='|', s=80, alpha=0.5, linewidths=1.5)
        ax.scatter(missing_dates, [0.5] * len(missing_dates),
                   c='#d73027', marker='|', s=80, alpha=0.9, linewidths=1.5)

        missing_pct = (is_missing.sum() / len(df)) * 100
        ax.set_ylim(0, 1)
        ax.set_yticks([0.5])
        ax.set_yticklabels([col], fontsize=9)
        ax.set_xlim(df.index.min(), df.index.max())
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_facecolor('#f5f5f5')

        stats_text = f'{is_missing.sum():,}/{len(df):,} missing ({missing_pct:.1f}%)'
        ax.text(0.99, 0.5, stats_text, transform=ax.transAxes,
                ha='right', va='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          alpha=0.9, edgecolor='gray'))

        if ax_idx == 0:
            ax.legend(handles=legend_elements, loc='upper left',
                      fontsize=8, framealpha=0.9)

    axes[-1].set_xlabel('Date', fontweight='bold', fontsize=11)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate(rotation=45)

    fig.text(
        0.5, 0.005,
        'Blue = data present | Red = missing | '
        'Columns sharing red patterns at the same dates have co-missing data',
        ha='center', fontsize=9, style='italic'
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.99])
    output_path = os.path.join(plots_dir, f"{base_name}_combined_daily_pattern.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(output_path)}")


def create_weekly_summary_heatmap(df, plots_dir, base_name, columns_to_check):
    """
    Create a heatmap showing missing data percentage for each day of week.
    Complements the daily pattern visualization with aggregate statistics.
    """
    print(f"\nCreating weekly summary heatmap...")
    
    # Prepare data
    df_with_dow = df.copy()
    df_with_dow['day_of_week'] = df_with_dow.index.dayofweek
    
    # Calculate missing percentage for each column and day of week
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    heatmap_data = []
    
    for col in columns_to_check:
        if col not in df.columns:
            continue
        
        row_data = []
        for day_idx in range(7):
            day_data = df_with_dow[df_with_dow['day_of_week'] == day_idx][col]
            if len(day_data) > 0:
                missing_pct = (day_data.isna().sum() / len(day_data)) * 100
                row_data.append(missing_pct)
            else:
                row_data.append(0)
        
        heatmap_data.append(row_data)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, max(6, len(columns_to_check) * 0.6)))
    
    heatmap_array = np.array(heatmap_data)
    im = ax.imshow(heatmap_array, cmap='YlOrRd', aspect='auto', vmin=0, vmax=20)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(7))
    ax.set_yticks(np.arange(len(columns_to_check)))
    ax.set_xticklabels(day_names, fontsize=11, fontweight='bold')
    ax.set_yticklabels(columns_to_check, fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Missing Data (%)', rotation=270, labelpad=20, fontsize=11)
    
    # Add percentage text in each cell
    for i in range(len(columns_to_check)):
        for j in range(7):
            value = heatmap_array[i, j]
            text_color = 'white' if value > 10 else 'black'
            text = ax.text(j, i, f'{value:.1f}%',
                         ha="center", va="center", color=text_color,
                         fontsize=9, fontweight='bold')
    
    date_range_str = f"{df.index.min().strftime('%d/%m/%Y')} - {df.index.max().strftime('%d/%m/%Y')}"
    ax.set_title(f'Weekly Missing Data Pattern - Summary Heatmap\n{date_range_str}', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Day of Week', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variable', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = os.path.join(plots_dir, f"{base_name}_weekly_summary_heatmap.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {os.path.basename(output_path)}")


def create_regular_timeline(df, freq='30min'):
    """
    Create a complete regular timeline by reindexing to fill missing timestamps.
    
    This is CRITICAL for interpolation:
    - Missing timestamps don't appear as NaNs in the original data (they're just absent rows)
    - Reindexing creates explicit NaNs at missing timestamps
    - Only then can interpolation methods fill the gaps properly
    
    Args:
        df (DataFrame): Data with datetime index (may have missing timestamps)
        freq (str): Target frequency (default: '30min')
        
    Returns:
        tuple: (df_reindexed, missing_timestamp_rate)
            - df_reindexed: DataFrame with complete regular timeline
            - missing_timestamp_rate: Percentage of timestamps that were missing
    """
    print(f"\n🔄 CREATING REGULAR TIMELINE (Reindexing)")
    print(f"   Original records: {len(df):,}")

    # Remove duplicate timestamps before reindexing (can arise from DST transitions)
    n_dupes = df.index.duplicated().sum()
    if n_dupes > 0:
        print(f"   [WARNING] {n_dupes} duplicate timestamp(s) found — keeping first occurrence")
        df = df[~df.index.duplicated(keep='first')]
        print(f"   Records after deduplication: {len(df):,}")

    # Create expected complete timeline
    start = df.index.min()
    end = df.index.max()
    expected_timeline = pd.date_range(start=start, end=end, freq=freq)
    expected_count = len(expected_timeline)
    
    print(f"   Expected records (with {freq} frequency): {expected_count:,}")
    
    # Reindex to create complete timeline (missing timestamps become NaNs)
    df_reindexed = df.reindex(expected_timeline)
    
    # Calculate missing timestamp rate
    missing_timestamps = expected_count - len(df)
    missing_timestamp_rate = (missing_timestamps / expected_count) * 100
    
    print(f"   Missing timestamps: {missing_timestamps:,} ({missing_timestamp_rate:.2f}%)")
    print(f"   ✓ Reindexed data now has {len(df_reindexed):,} records with explicit NaNs")
    
    # Show before/after NaN counts for key columns
    print(f"\n   NaN counts (before → after reindexing):")
    for col in df.columns[:5]:  # Show first 5 columns as example
        nans_before = df[col].isna().sum()
        nans_after = df_reindexed[col].isna().sum()
        print(f"      {col}: {nans_before:,} → {nans_after:,}")
    
    return df_reindexed, missing_timestamp_rate


def load_and_prepare_data(file_path, config):
    """Load and prepare data with error handling."""
    print(f"\n📊 LOADING DATA")
    
    # ⚠️  CRITICAL: skiprows must match your actual CSV format
    #    - If CSV has sep=; header line: use skiprows=1 (skip the sep line)
    #    - If CSV has NO sep line: use skiprows=0 (don't skip header)
    #    Default changed to 0 to prevent accidentally skipping header row
    skiprows = config.getint("data", "skiprows", fallback=0)
    time_column = config.get("data", "time_column", fallback="Time").strip()
    
    try:
        df = pd.read_csv(file_path, skiprows=skiprows)
        print(f"✅ Loaded {len(df)} records")
        
        # Handle time column
        if time_column == "" or time_column not in df.columns:
            print("⚠️  Using first column as time index")
            df.index = pd.to_datetime(df.iloc[:, 0], errors="coerce")
            df.drop(columns=df.columns[0], inplace=True)
        else:
            df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
            df.set_index(time_column, inplace=True)
        
        # Remove invalid timestamps
        initial_length = len(df)
        df = df[df.index.notna()]
        removed_rows = initial_length - len(df)
        
        if removed_rows > 0:
            print(f"⚠️  Removed {removed_rows} rows with invalid timestamps")
        
        # Remove timezone for Excel compatibility (robust approach)
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            print(f"   ✓ Removed timezone from index for Excel compatibility")
        elif df.index.dtype == "object":
            # Handle object index with tz-aware datetimes
            try:
                idx = pd.to_datetime(df.index, errors="coerce")
                if getattr(idx, "tz", None) is not None:
                    idx = idx.tz_localize(None)
                    print(f"   ✓ Converted and removed timezone from object index")
                df.index = idx
            except Exception:
                pass
        
        print(f"✅ Data period: {df.index.min()} to {df.index.max()}")
        
        # Clean numeric columns
        try:
            temp_columns = [col.strip() for col in config.get("data", "temperature_columns", fallback="").split(",") if col.strip()]
            modulation_columns = [col.strip() for col in config.get("data", "modulation_columns", fallback="").split(",") if col.strip()]
            all_numeric_columns = temp_columns + modulation_columns
            
            for col in all_numeric_columns:
                if col in df.columns:
                    df[col] = clean_temperature_data(df[col])
        except:
            print("⚠️  Skipping column cleaning")
        
        # ===================================================================
        # ADD THIS DIAGNOSTIC SECTION HERE (before return statement)
        # ===================================================================
        print("\n" + "="*60)
        print("🔍 DATA INTERVAL DIAGNOSTIC")
        print("="*60)
        
        time_diffs = df.index.to_series().diff()
        actual_mode = time_diffs.mode()[0] if len(time_diffs.mode()) > 0 else None
        
        if actual_mode:
            print(f"Most common interval in data: {actual_mode}")
        
        # Check config resampling setting
        try:
            expected_freq = config.get('resampling', 'frequency', fallback='30min')
            expected_interval = pd.Timedelta(expected_freq)
            print(f"Expected interval (config):   {expected_interval}")
            
            if actual_mode and actual_mode != expected_interval:
                print("\n⚠️  WARNING: DATA INTERVAL MISMATCH!")
                print(f"   Your actual data interval: {actual_mode}")
                print(f"   Your config expects:       {expected_interval}")
                print(f"\n   🔴 CRITICAL: This mismatch will cause incorrect missing data detection!")
                print(f"   \n   📝 NOTE: This script only DIAGNOSES the problem - it does NOT fix it.")
                print(f"   \n   ✅ SOLUTIONS (choose one):")
                print(f"      1. Update config [resampling] frequency = {actual_mode}")
                print(f"      2. Run a resampling script to standardize your data to {expected_interval}")
                print(f"      3. Ensure your raw data is sampled at consistent {expected_interval} intervals")
                print(f"   \n   ⚠️  Do NOT interpret missing data results until interval is corrected!")
        except:
            print("Could not read resampling config")
        
        # Show interval distribution
        print("\nInterval distribution (top 5 most common):")
        interval_counts = time_diffs.value_counts().head()
        for interval, count in interval_counts.items():
            pct = (count / len(time_diffs)) * 100
            print(f"  {interval}: {count} occurrences ({pct:.1f}%)")
        
        print("="*60 + "\n")
        # ===================================================================
        # END OF DIAGNOSTIC SECTION
        # ===================================================================
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)


"""
Main Function and Script Execution
"""

def analyze_missing_data():
    """Main analysis function."""
    print("🚀 ENHANCED MISSING DATA ANALYSIS")
    print("=" * 50)

    # Configuration setup - try multiple locations
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
    
    # Data file discovery
    input_csv = config.get("paths", "input_csv")
    base_name = os.path.splitext(input_csv)[0]
    
    data_file_path, source_type = find_data_file_with_search(config, base_name)
    
    if data_file_path is None:
        print("❌ ERROR: No suitable data file found!")
        print("Expected locations:")
        print("1. processed_data/merged/[base_name]/[base_name]_merged.csv")
        print("2. data/[input_csv]")
        sys.exit(1)
    
    # Verify data source
    verify_data_source(data_file_path, base_name)
    
    # Directory setup
    processed_folder = config.get("paths", "processed_folder")
    plots_folder = config.get("paths", "plots_folder")
    # Resolve output dirs relative to config.ini directory
    cfg_dir = getattr(config, '_config_dir', None)
    if cfg_dir:
        if processed_folder and not os.path.isabs(processed_folder):
            processed_folder = os.path.abspath(os.path.join(cfg_dir, processed_folder))
        if plots_folder and not os.path.isabs(plots_folder):
            plots_folder = os.path.abspath(os.path.join(cfg_dir, plots_folder))
    
    processed_dir = os.path.join(processed_folder, "missing_data_analysis", base_name)
    plots_dir = os.path.join(plots_folder, "missing_data", base_name)
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"✅ Output directories created")
    print(f"   Reports: {processed_dir}")
    print(f"   Plots: {plots_dir}")
    
    # Analysis parameters
    try:
        missing_threshold_hours = config.getfloat("missing_data", "missing_threshold_hours", fallback=4.0)
        
        # Get columns from missing_data section, fallback to data section columns
        columns_str = config.get("missing_data", "columns_to_check", fallback="")
        if not columns_str:
            # Try to get from data section
            temp_cols = config.get("data", "temperature_columns", fallback="")
            mod_cols = config.get("data", "modulation_columns", fallback="")
            columns_str = f"{temp_cols}, {mod_cols}" if temp_cols and mod_cols else temp_cols or mod_cols
        
        columns_to_check = [col.strip() for col in columns_str.split(",") if col.strip()]
        
        if not columns_to_check:
            columns_to_check = ["T Mandata"]
            
        create_plots = config.getboolean("missing_data", "create_missing_data_plots", fallback=True)
        
    except Exception as e:
        print(f"⚠️  Configuration issue: {e}")
        missing_threshold_hours = 4.0
        columns_to_check = ["T Mandata"]
        create_plots = True
    
    print(f"🎯 Analysis parameters:")
    print(f"   Gap threshold: {missing_threshold_hours} hours")
    print(f"   Columns to analyze: {columns_to_check}")
    
    # Load and prepare data
    df = load_and_prepare_data(data_file_path, config)
    
    # Verify columns exist
    available_columns = list(df.columns)
    valid_columns = [col for col in columns_to_check if col in available_columns]
    
    if not valid_columns:
        print("❌ ERROR: None of the specified columns exist!")
        print(f"Available columns: {available_columns}")
        sys.exit(1)
    
    columns_to_check = valid_columns
    print(f"✅ Valid columns for analysis: {columns_to_check}")
    
    # =========================================================================
    # CRITICAL STEP: Create regular timeline (reindex to expose missing timestamps)
    # =========================================================================
    # Get configured frequency
    try:
        resample_freq = config.get("processing", "resample_freq", fallback="30min")
    except:
        resample_freq = "30min"
    
    # Reindex to create complete timeline with explicit NaNs
    df_reindexed, missing_timestamp_rate = create_regular_timeline(df, freq=resample_freq)
    
    # Use reindexed data for all subsequent analysis
    df_original = df.copy()  # Keep original for reference
    df = df_reindexed
    
    # Main analysis (now with complete timeline)
    total_records = len(df)
    total_missing = df.isna().sum().sum()
    overall_missing_rate = (total_missing / (total_records * len(df.columns))) * 100
    
    print(f"\n📊 DATASET OVERVIEW (After Reindexing)")
    print(f"Records: {total_records:,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Missing NaN rate: {overall_missing_rate:.2f}%")
    print(f"Missing timestamp rate: {missing_timestamp_rate:.2f}%")
    
    # 1. Timestamp-level analysis
    missing_details = create_timestamp_missing_report(df, processed_dir, base_name)
    
    # 2. Correlation analysis
    correlations = analyze_correlations(df, processed_dir, base_name)
    
    # 3. Sensor failure analysis
    failure_patterns = identify_sensor_failures(df, processed_dir, base_name)

    # 3b. System-wide shutdown detection
    shutdowns, shutdown_mask = detect_system_shutdowns(
        df, processed_dir, base_name, columns_to_check
    )

    # 4. Gap analysis for specified columns
    print(f"\n🔍 DETAILED GAP ANALYSIS")
    all_missing_periods = {}
    
    for col in columns_to_check:
        missing_periods = find_missing_periods(df, col, missing_threshold_hours)
        all_missing_periods[col] = missing_periods
        
        if missing_periods:
            periods_df = pd.DataFrame(missing_periods)
            periods_file = os.path.join(processed_dir, f"{base_name}_{col.replace(' ', '_')}_gaps.csv")
            periods_df.to_csv(periods_file, index=False)
            print(f"   ✅ {col}: {len(missing_periods)} gaps saved")
        else:
            print(f"   ✅ {col}: No significant gaps")
    
    # 5. Excel report
    excel_report = create_excel_report(df, processed_dir, base_name)
    
    # 6. Visualizations
    if create_plots:
        create_visualizations(df, plots_dir, base_name, columns_to_check)
    
    # 6.5 Manager-friendly Word report (new addition)
    # This creates a Word document with plots and descriptions
    print(f"\n{'='*60}")
    manager_word = create_manager_report(
        df=df,
        output_dir=plots_dir,  # Save to plots folder instead of processed_dir
        base_name=base_name,
        columns_to_check=columns_to_check,
        config_threshold_hours=missing_threshold_hours  # Pass config threshold
    )
    print(f"{'='*60}")
    
    # 7. Summary files
    print(f"\n📁 GENERATING SUMMARY FILES")
    
    # Analysis summary
    summary_data = {
        'analysis_date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'data_file': [os.path.basename(data_file_path)],
        'source_type': [source_type], 
        'total_records': [total_records],
        'total_columns': [len(df.columns)],
        'overall_missing_nan_rate': [overall_missing_rate],  # NaN-based rate
        'missing_timestamp_rate': [missing_timestamp_rate],  # Missing timestamp rate (NEW)
        'missing_threshold_hours': [missing_threshold_hours],
        'timestamp_records': [len(missing_details)],
        'correlations_found': [len(correlations.get('correlations', []))],
        'high_priority_failures': [len([p for p in failure_patterns if p['Priority'] == 'High'])]
    }
    
    # Add per-column statistics
    for col in columns_to_check:
        col_missing = df[col].isna().sum()
        col_missing_pct = (col_missing / len(df)) * 100
        col_gaps = len(all_missing_periods.get(col, []))
        
        summary_data[f'{col}_missing_count'] = [col_missing]
        summary_data[f'{col}_missing_pct'] = [col_missing_pct]
        summary_data[f'{col}_significant_gaps'] = [col_gaps]
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(processed_dir, f"{base_name}_analysis_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    
    # Missing flags matrix (rename columns with _missing_flag suffix)
    missing_flags = df.isna().astype(int)
    missing_flags.columns = [f"{col}_missing_flag" for col in missing_flags.columns]
    flags_file = os.path.join(processed_dir, f"{base_name}_missing_flags.csv")
    missing_flags.to_csv(flags_file, index=True)
    print(f"   ✓ Missing flags saved with '_missing_flag' suffix")
    
    # Data ready for interpolation (REINDEXED with complete timeline)
    # Attach system_shutdown_flag so downstream scripts can skip/mask shutdown periods
    df_out = df.copy()
    df_out['system_shutdown_flag'] = shutdown_mask.astype(int)
    output_file = os.path.join(processed_dir, f"{base_name}_reindexed_for_interpolation.csv")
    df_out.to_csv(output_file, index=True)
    print(f"   ✓ Reindexed data saved (ready for interpolation, incl. system_shutdown_flag)")
    
    # Also save original data (without reindexing) for reference
    original_file = os.path.join(processed_dir, f"{base_name}_original_data.csv")
    df_original.to_csv(original_file, index=True)
    print(f"   ✓ Original data saved (for reference)")
    
    print(f"✅ Summary files created")
    
    # Final report
    print(f"\n" + "="*60)
    print(f"🎉 ANALYSIS COMPLETE")
    print("="*60)
    
    print(f"\n📋 RESULTS SUMMARY")
    print(f"Dataset: {total_records:,} records × {len(df.columns)} columns")
    print(f"Period: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    print(f"Missing NaN rate: {overall_missing_rate:.2f}%")
    print(f"Missing timestamp rate: {missing_timestamp_rate:.2f}%")
    
    print(f"\n🔍 COLUMN ANALYSIS")
    for col in columns_to_check:
        col_missing = df[col].isna().sum()
        col_missing_pct = (col_missing / len(df)) * 100
        col_gaps = len(all_missing_periods.get(col, []))
        print(f"   {col}: {col_missing:,} missing ({col_missing_pct:.2f}%), {col_gaps} significant gaps")
    
    print(f"\n🔗 CORRELATIONS")
    correlations_found = len(correlations.get('correlations', []))
    if correlations_found > 0:
        print(f"   High correlations: {correlations_found}")
    else:
        print(f"   No significant correlations found")
    
    print(f"\n� SYSTEM SHUTDOWNS")
    if shutdowns:
        total_sd_hours = sum(s['duration_hours'] for s in shutdowns)
        print(f"   {len(shutdowns)} shutdown period(s) detected — "
              f"{total_sd_hours:.1f}h total ({shutdown_mask.sum()/len(df)*100:.1f}% of data)")
        for s in shutdowns:
            print(f"   • {s['start'].date()} → {s['end'].date()} ({s['duration_days']}d)")
        print(f"   ⚠️  system_shutdown_flag=1 rows should NOT be interpolated")
    else:
        print(f"   ✅ No system-wide shutdowns detected")

    print(f"\n�🔧 SENSOR ISSUES")
    high_priority = len([p for p in failure_patterns if p['Priority'] == 'High'])
    medium_priority = len([p for p in failure_patterns if p['Priority'] == 'Medium'])
    
    if high_priority > 0:
        print(f"   🚨 High priority: {high_priority} sensors need immediate attention")
    if medium_priority > 0:
        print(f"   ⚠️  Medium priority: {medium_priority} sensors need monitoring")
    if high_priority == 0 and medium_priority == 0:
        print(f"   ✅ No critical sensor issues detected")
    
    print(f"\n📁 OUTPUT FILES")
    print(f"   📊 Manager Word Report: {os.path.basename(manager_word) if manager_word else 'Not created'}")
    print(f"   📋 Excel report: {os.path.basename(excel_report)}")
    print(f"   📄 Analysis summary: {os.path.basename(summary_file)}")
    print(f"   📍 Missing flags: {os.path.basename(flags_file)} (with _missing_flag suffix)")
    print(f"   🔄 Reindexed data: {os.path.basename(output_file)} (READY FOR INTERPOLATION, incl. system_shutdown_flag)")
    print(f"   📦 Original data: {os.path.basename(original_file)} (reference)")
    if shutdowns:
        shutdown_file = os.path.join(processed_dir, f"{base_name}_system_shutdowns.csv")
        print(f"   🔌 Shutdown periods: {os.path.basename(shutdown_file)}")
    
    if create_plots:
        print(f"   Visualizations: {plots_dir}")
    
    print(f"\n🔄 NEXT STEPS")
    print(f"1. ✅ Raw data → Weather merge")
    print(f"2. ✅ Missing data analysis (COMPLETED)")
    print(f"3. ✅ Timeline reindexing (COMPLETED - NaNs now explicit)")
    print(f"4. ➡️  Operation analysis")
    print(f"5. ➡️  Outlier detection")
    print(f"6. ➡️  Data interpolation (use reindexed file!)")
    
    print(f"\n💡 RECOMMENDATIONS")
    if missing_timestamp_rate > 5:
        print(f"⚠️  ATTENTION: {missing_timestamp_rate:.2f}% of timestamps missing - check data collection")
    elif missing_timestamp_rate > 1:
        print(f"ℹ️  NOTE: {missing_timestamp_rate:.2f}% of timestamps missing - acceptable for interpolation")
    else:
        print(f"✅ EXCELLENT: Only {missing_timestamp_rate:.2f}% timestamps missing")
    
    if overall_missing_rate > 20:
        print(f"🚨 CRITICAL: High NaN rate - investigate sensor issues")
    elif overall_missing_rate > 10:
        print(f"⚠️  MODERATE: NaN rate requires monitoring")
    else:
        print(f"✅ GOOD: NaN rate is acceptable")
    
    if high_priority > 0:
        print(f"🚨 IMMEDIATE: {high_priority} sensors need maintenance")

    if shutdowns:
        total_sd_pct = shutdown_mask.sum() / len(df) * 100
        print(f"\n🔌 SHUTDOWN RECOMMENDATION")
        print(f"   {len(shutdowns)} system shutdown(s) cover {total_sd_pct:.1f}% of the dataset.")
        print(f"   → Do NOT interpolate rows where system_shutdown_flag=1.")
        print(f"   → Consider excluding these periods from LSTM training,")
        print(f"     or add system_shutdown_flag as a binary input feature.")

    print("="*60)
    print("✅ ANALYSIS SUCCESSFULLY COMPLETED!")
    print("="*60)
    
    return {
        'total_records': total_records,
        'missing_nan_rate': overall_missing_rate,
        'missing_timestamp_rate': missing_timestamp_rate,
        'high_priority_sensors': high_priority,
        'files_generated': {
            'excel_report': excel_report,
            'summary_file': summary_file,
            'reindexed_file': output_file,
            'original_file': original_file
        }
    }

if __name__ == "__main__":
    """Script execution with error handling."""
    print("🌟 HVAC MISSING DATA ANALYSIS TOOL")
    print("=" * 40)
    print("Version: 2.0 Clean")
    print("Purpose: Missing data analysis for HVAC systems")
    print("=" * 40)
    
    try:
        result = analyze_missing_data()
        
        print(f"\n🎉 SUCCESS!")
        print(f"Missing NaN rate: {result['missing_nan_rate']:.2f}%")
        print(f"Missing timestamp rate: {result['missing_timestamp_rate']:.2f}%")
        if result['high_priority_sensors'] > 0:
            print(f"⚠️  {result['high_priority_sensors']} sensors need attention")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  ANALYSIS INTERRUPTED")
        print("Analysis stopped by user (Ctrl+C)")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ ANALYSIS FAILED")
        print(f"Error: {type(e).__name__}: {str(e)}")
        
        print(f"\n🛠️  TROUBLESHOOTING:")
        print("1. Check config.ini exists and has correct paths")
        print("2. Ensure merge script was run first")
        print("3. Verify column names in config match CSV file")
        print("4. Check file permissions")
        
        import traceback
        traceback.print_exc()
        sys.exit(1)
