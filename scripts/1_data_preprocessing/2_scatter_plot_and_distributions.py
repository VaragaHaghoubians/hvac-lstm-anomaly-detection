"""
SCATTER PLOTS AND DISTRIBUTIONS - HVAC System Data Exploration
===============================================================

Purpose:
--------
Generate comprehensive data exploration visualizations to:
1. DISTRIBUTIONS: Understand variable ranges, identify outliers, justify cleaning thresholds
2. SCATTER PLOTS: Reveal feature relationships, operational regimes, and data quality

Key Outputs:
------------
DISTRIBUTIONS (Histograms + Box Plots):
  • Temperature variables: TM, TR, TSat (thermodynamic validation)
  • Setpoint: SP_TM (control strategy)
  • Fans: FanMandata, FanRipresa (operational patterns)
  • Weather: OutdoorT, OutdoorRH (seasonal context)

SCATTER PLOTS (Feature-Pairs):
  • TM vs SP_TM → Control tracking quality
  • TR vs TM → Thermal relationships and operational regimes
  • OutdoorT vs TM → Seasonal influence and regime separation
  • FanMandata vs FanRipresa → System balance verification
  • FanMandata vs (TR-TM) → Fan effort vs thermal effect
  • OutdoorT vs OutdoorRH → Weather sensor validation

Usage:
------
Run after 1.merged_data.py to validate and understand the merged dataset.
Use distributions to:
  - Set cleaning thresholds (e.g., remove TM > 60°C)
  - Identify sensor failures (flat distributions, impossible values)
  - Understand seasonal differences

Use scatter plots to:
  - Validate control performance
  - Identify operational modes
  - Detect regime mixing
  - Support thesis discussion

Output:
-------
Saves to: HVAC_project/plots/distributions_and_scatter/[dataset]/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import configparser
from pathlib import Path
from datetime import datetime
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Standard Tukey whisker fence multiplier for box plot whiskers
TUKEY_FENCE = 1.5
# Number of bins for distribution histograms
HISTOGRAM_BINS = 50

# Add parent directory to path to import config utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_config(config_path):
    """Load configuration file with interpolation support."""
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(config_path, encoding='utf-8-sig')
    return config


def find_merged_data(config):
    """
    Find the most recent merged data file based on config settings.
    
    Returns:
        str: Path to merged CSV file
    """
    # Get config values
    building_id = config.get('global', 'building_id')
    ahu_unit = config.get('global', 'ahu_unit')
    season = config.get('global', 'season')
    year = config.get('global', 'year')
    
    # Construct expected filename
    base_name = f"{building_id}_{ahu_unit}_{season}{year}"
    
    # Find project root (3 levels up from this script)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # Expected merged data location
    merged_dir = project_root / 'processed_data' / 'merged' / base_name
    merged_file = merged_dir / f"{base_name}_merged.csv"
    
    if merged_file.exists():
        print(f"✓ Found merged data: {merged_file}")
        return str(merged_file)
    else:
        print(f"❌ Merged data not found at: {merged_file}")
        print(f"\nPlease run 1.merged_data.py first to create the merged dataset.")
        sys.exit(1)


def load_data(filepath):
    """
    Load merged CSV data with proper datetime parsing.
    
    Args:
        filepath (str): Path to merged CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    print(f"\n📁 Loading data from: {Path(filepath).name}")
    
    try:
        df = pd.read_csv(filepath, parse_dates=['Time'])
        print(f"   ✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
        print(f"   ✓ Date range: {df['Time'].min()} to {df['Time'].max()}")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)


def get_column_mapping():
    """
    Define column name mapping for Italian/English compatibility.
    
    Returns:
        dict: Mapping of standard names to possible column variants
    """
    return {
        'TM': ['Temperatura Mandata', 'Temp. Mandata', 'T Mandata'],
        'TR': ['Temperatura Ripresa', 'Temp. Ripresa', 'T Ripresa'],
        'T_Sat': ['Temperatura Saturazione', 'Temp. Saturazione', 'T Saturazione'],
        'T_Ext': ['Temperatura Esterna', 'Temp. Esterna'],
        'Umid_Ext': ['Umidita Esterna', 'Umid. Esterna'],
        'SP_TM': ['Set Points Temperatura Mandata', 'SP Temp. Mand.'],
        'Fan_M': ['Modulazione Ventilatore Mandata', 'Mod. V. Mandata'],
        'Fan_R': ['Modulazione Ventilatore Ripresa', 'Mod. V. Ripresa']
    }


def find_column(df, standard_name, column_mapping):
    """
    Find actual column name in dataframe from list of variants.
    
    Args:
        df (pd.DataFrame): Dataset to search
        standard_name (str): Standard column identifier (e.g., 'TM')
        column_mapping (dict): Mapping of standard names to variants
        
    Returns:
        str or None: Actual column name if found, None otherwise
    """
    variants = column_mapping.get(standard_name, [])
    for variant in variants:
        if variant in df.columns:
            return variant
    return None


def create_distribution_plots(df, output_dir, column_mapping):
    """
    Create combined histogram and box plot distributions for key variables.
    
    Args:
        df (pd.DataFrame): Dataset
        output_dir (Path): Output directory
        column_mapping (dict): Column name mapping
    """
    print("\n📊 Generating distribution plots (Histogram + Box Plot)...")
    
    # Define variables to analyze with metadata
    variables = [
        ('TM', 'Temperatura Mandata', '°C', 'Temperature'),
        ('TR', 'Temperatura Ripresa', '°C', 'Temperature'),
        ('T_Sat', 'Temperatura Saturazione', '°C', 'Temperature'),
        ('SP_TM', 'Setpoint Temperatura Mandata', '°C', 'Temperature'),
        ('Fan_M', 'Modulazione Ventilatore Mandata', '%', 'Percentage'),
        ('Fan_R', 'Modulazione Ventilatore Ripresa', '%', 'Percentage'),
        ('T_Ext', 'Temperatura Esterna', '°C', 'Temperature'),
        ('Umid_Ext', 'Umidità Esterna', '%', 'Percentage')
    ]
    
    for std_name, display_name, unit, var_type in variables:
        col = find_column(df, std_name, column_mapping)
        
        if col is None:
            print(f"   ⚠️  Skipping {display_name} (column not found)")
            continue
        
        # Get valid data
        data = df[col].dropna()
        
        if len(data) == 0:
            print(f"   ⚠️  Skipping {display_name} (no valid data)")
            continue
        
        # Create figure with 2 subplots (histogram and box plot)
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), 
                                gridspec_kw={'height_ratios': [3, 1]})
        
        # ===== HISTOGRAM =====
        ax_hist = axes[0]
        
        # Plot histogram with KDE
        n, bins, patches = ax_hist.hist(data, bins=HISTOGRAM_BINS, alpha=0.7, color='steelblue',
                                        edgecolor='black', linewidth=0.5, density=False)
        
        # Add KDE overlay
        from scipy import stats
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        kde_values = kde(x_range)
        # Scale KDE to match histogram height
        kde_scaled = kde_values * len(data) * (bins[1] - bins[0])
        ax_hist.plot(x_range, kde_scaled, 'r-', linewidth=2, label='KDE')
        
        # Calculate statistics
        mean_val = data.mean()
        median_val = data.median()
        std_val = data.std()
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        
        # Add vertical lines for mean and median
        ax_hist.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                       label=f'Media: {mean_val:.2f} {unit}', alpha=0.8)
        ax_hist.axvline(median_val, color='orange', linestyle='--', linewidth=2,
                       label=f'Mediana: {median_val:.2f} {unit}', alpha=0.8)
        
        # Add statistics text box
        textstr = '\n'.join([
            f'n = {len(data):,}',
            f'Media = {mean_val:.2f} {unit}',
            f'Mediana = {median_val:.2f} {unit}',
            f'Std = {std_val:.2f} {unit}',
            f'Min = {data.min():.2f} {unit}',
            f'Max = {data.max():.2f} {unit}',
            f'Q25 = {q25:.2f} {unit}',
            f'Q75 = {q75:.2f} {unit}'
        ])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
        ax_hist.text(0.98, 0.97, textstr, transform=ax_hist.transAxes, 
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=props, family='monospace')
        
        ax_hist.set_xlabel(f'{display_name} ({unit})', fontsize=12, fontweight='bold')
        ax_hist.set_ylabel('Frequenza', fontsize=12, fontweight='bold')
        ax_hist.set_title(f'Distribuzione: {display_name}', fontsize=14, fontweight='bold', pad=15)
        ax_hist.legend(loc='upper left', framealpha=0.9)
        ax_hist.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # ===== BOX PLOT =====
        ax_box = axes[1]
        
        # Create horizontal box plot
        bp = ax_box.boxplot([data], vert=False, widths=0.6, patch_artist=True,
                            showmeans=True, meanline=True,
                            boxprops=dict(facecolor='lightblue', edgecolor='black', linewidth=1.5),
                            whiskerprops=dict(color='black', linewidth=1.5),
                            capprops=dict(color='black', linewidth=1.5),
                            medianprops=dict(color='red', linewidth=2),
                            meanprops=dict(color='green', linewidth=2, linestyle='--'),
                            flierprops=dict(marker='o', markerfacecolor='red', markersize=4, 
                                          alpha=0.5, markeredgecolor='darkred'))
        
        ax_box.set_xlabel(f'{display_name} ({unit})', fontsize=12, fontweight='bold')
        ax_box.set_yticks([])
        ax_box.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='x')
        
        # Annotate outliers count
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - TUKEY_FENCE * iqr
        upper_fence = q3 + TUKEY_FENCE * iqr
        outliers = data[(data < lower_fence) | (data > upper_fence)]
        outlier_pct = (len(outliers) / len(data)) * 100
        
        ax_box.text(0.02, 0.5, f'Outliers: {len(outliers):,} ({outlier_pct:.2f}%)', 
                   transform=ax_box.transAxes, fontsize=10, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # Add date range subtitle
        date_min = df['Time'].min().strftime('%d/%m/%Y')
        date_max = df['Time'].max().strftime('%d/%m/%Y')
        fig.text(0.5, 0.98, f'Periodo: {date_min} - {date_max}', 
                ha='center', va='top', fontsize=10, style='italic', color='gray')
        
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        # Save plot
        output_path = output_dir / f'dist_{std_name}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ {display_name}: mean={mean_val:.2f}, outliers={len(outliers):,} ({outlier_pct:.1f}%)")


def create_scatter_plot(df, x_col, y_col, title, xlabel, ylabel, 
                       output_path, date_range=None, diagonal=False, 
                       threshold=None, computed_y=None):
    """
    Create publication-quality scatter plot with density visualization.
    
    Args:
        df (pd.DataFrame): Dataset
        x_col (str): X-axis column name
        y_col (str): Y-axis column name (or None if computed_y is provided)
        title (str): Plot title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
        output_path (str): Save location
        date_range (str, optional): Date range string to display as subtitle
        diagonal (bool): Draw diagonal reference line
        threshold (float, optional): Threshold distance from diagonal for highlighting
        computed_y (pd.Series, optional): Pre-computed y values (e.g., for TR-TM)
    """
    # Handle data preparation
    if computed_y is not None:
        # Use computed y values
        mask = df[x_col].notna() & computed_y.notna()
        x_data = df.loc[mask, x_col]
        y_data = computed_y.loc[mask]
    else:
        # Use direct column
        mask = df[x_col].notna() & df[y_col].notna()
        x_data = df.loc[mask, x_col]
        y_data = df.loc[mask, y_col]
    
    if len(x_data) == 0:
        print(f"   ⚠️  No valid data for {title}")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Density-based visualization
    if len(x_data) > 5000:
        # Use hexbin for large datasets
        hexbin = ax.hexbin(x_data, y_data, gridsize=50, cmap='Blues', 
                          mincnt=1, alpha=0.8, rasterized=True)
        cbar = plt.colorbar(hexbin, ax=ax, label='Densità punti')
    else:
        # Use scatter with transparency for smaller datasets
        ax.scatter(x_data, y_data, alpha=0.4, s=15, c='steelblue', 
                  edgecolors='none', rasterized=True)
    
    # Draw diagonal reference line if requested
    if diagonal:
        lims = [
            np.min([ax.get_xlim()[0], ax.get_ylim()[0]]),
            np.max([ax.get_xlim()[1], ax.get_ylim()[1]]),
        ]
        ax.plot(lims, lims, 'r--', alpha=0.6, linewidth=2, 
               label='Riferimento perfetto (y=x)', zorder=10)
        
        # Draw threshold bands if specified
        if threshold is not None:
            ax.fill_between(lims, 
                           [l - threshold for l in lims], 
                           [l + threshold for l in lims],
                           alpha=0.15, color='green', 
                           label=f'±{threshold}°C tolleranza', zorder=5)
    
    # Formatting
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    
    # Set title with optional date range subtitle
    if date_range:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=25)
        ax.text(0.5, 1.02, date_range, transform=ax.transAxes, 
                ha='center', va='bottom', fontsize=10, style='italic', color='gray')
    else:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add statistics text box
    corr = x_data.corr(y_data)
    
    textstr = f'n = {len(x_data):,}\n'
    textstr += f'Correlazione = {corr:.3f}\n'
    
    # Add RMSE for tracking plots
    if diagonal and threshold is not None:
        rmse = np.sqrt(np.mean((y_data - x_data)**2))
        mae = np.abs(y_data - x_data).mean()
        tracking_ok = (np.abs(y_data - x_data) <= threshold).sum()
        tracking_pct = (tracking_ok / len(x_data)) * 100
        
        textstr += f'RMSE = {rmse:.2f}°C\n'
        textstr += f'MAE = {mae:.2f}°C\n'
        textstr += f'Entro ±{threshold}°C: {tracking_pct:.1f}%'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')
    
    if diagonal:
        ax.legend(loc='lower right', framealpha=0.9, fontsize=9)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: {Path(output_path).name} (r={corr:.3f})")


def generate_scatter_plots(df, output_dir, column_mapping, config):
    """
    Generate specific feature-pair scatter plots as per thesis requirements.

    Args:
        df (pd.DataFrame): Dataset
        output_dir (Path): Output directory
        column_mapping (dict): Column name mapping
        config (ConfigParser): Project configuration
    """
    print("\n📊 Generating scatter plots (feature-pair relationships)...")

    tracking_threshold = config.getfloat('visualization', 'scatter_tracking_threshold_degC', fallback=2.0)
    fan_imbalance_threshold = config.getfloat('visualization', 'scatter_fan_imbalance_threshold_pct', fallback=5.0)

    # Calculate date range for subtitles
    date_min = df['Time'].min().strftime('%d/%m/%Y')
    date_max = df['Time'].max().strftime('%d/%m/%Y')
    date_range = f"Periodo: {date_min} - {date_max}"
    
    # Find actual column names
    cols = {}
    for std_name in column_mapping.keys():
        cols[std_name] = find_column(df, std_name, column_mapping)
    
    # Track missing columns
    missing = [name for name, col in cols.items() if col is None]
    if missing:
        print(f"\n   ℹ️  Columns not found (some plots may be skipped): {', '.join(missing)}")
    
    print("\n📍 Creating feature-pair scatter plots:")
    
    # 1. TM vs SP_TM - Control tracking quality
    if cols['TM'] and cols['SP_TM']:
        print("\n1️⃣  TM vs SP_TM (Control Tracking)")
        create_scatter_plot(
            df, cols['SP_TM'], cols['TM'],
            title='Control Tracking: Temperatura Mandata vs Setpoint',
            xlabel='Setpoint Temperatura Mandata (°C)',
            ylabel='Temperatura Mandata Effettiva (°C)',
            output_path=output_dir / 'scatter_1_TM_vs_SP_TM.png',
            date_range=date_range,
            diagonal=True,
            threshold=tracking_threshold
        )
    
    # 2. TR vs TM - Thermal relationships and operational regimes
    if cols['TR'] and cols['TM']:
        print("\n2️⃣  TR vs TM (Thermal Relationships)")
        create_scatter_plot(
            df, cols['TM'], cols['TR'],
            title='Relazione Termica: Temperatura Ripresa vs Mandata',
            xlabel='Temperatura Mandata (°C)',
            ylabel='Temperatura Ripresa (°C)',
            output_path=output_dir / 'scatter_2_TR_vs_TM.png',
            date_range=date_range,
            diagonal=False
        )
    
    # 3. OutdoorT vs TM - Seasonal influence and regime separation
    if cols['T_Ext'] and cols['TM']:
        print("\n3️⃣  OutdoorT vs TM (Seasonal Influence)")
        create_scatter_plot(
            df, cols['T_Ext'], cols['TM'],
            title='Influenza Stagionale: Temperatura Esterna vs Mandata',
            xlabel='Temperatura Esterna (°C)',
            ylabel='Temperatura Mandata (°C)',
            output_path=output_dir / 'scatter_3_OutdoorT_vs_TM.png',
            date_range=date_range,
            diagonal=False
        )
    
    # 4. FanMandata vs FanRipresa - System balance verification
    if cols['Fan_M'] and cols['Fan_R']:
        print("\n4️⃣  FanMandata vs FanRipresa (System Balance)")
        create_scatter_plot(
            df, cols['Fan_M'], cols['Fan_R'],
            title='Bilanciamento Sistema: Ventilatore Mandata vs Ripresa',
            xlabel='Modulazione Ventilatore Mandata (%)',
            ylabel='Modulazione Ventilatore Ripresa (%)',
            output_path=output_dir / 'scatter_4_FanM_vs_FanR.png',
            date_range=date_range,
            diagonal=True,
            threshold=fan_imbalance_threshold
        )
    
    # 5. FanMandata vs (TR-TM) - Fan effort vs thermal effect
    if cols['Fan_M'] and cols['TR'] and cols['TM']:
        print("\n5️⃣  FanMandata vs (TR-TM) (Effort vs Effect)")
        # Compute TR - TM (thermal rise/drop)
        thermal_diff = df[cols['TR']] - df[cols['TM']]
        
        create_scatter_plot(
            df, cols['Fan_M'], None,
            title='Sforzo Ventilatore vs Effetto Termico: Fan vs (TR−TM)',
            xlabel='Modulazione Ventilatore Mandata (%)',
            ylabel='Differenza Termica TR−TM (°C)',
            output_path=output_dir / 'scatter_5_FanM_vs_ThermalDiff.png',
            date_range=date_range,
            diagonal=False,
            computed_y=thermal_diff
        )
    
    # 6. OutdoorT vs OutdoorRH - Weather sensor validation
    if cols['T_Ext'] and cols['Umid_Ext']:
        print("\n6️⃣  OutdoorT vs OutdoorRH (Weather Sensor Check)")
        create_scatter_plot(
            df, cols['T_Ext'], cols['Umid_Ext'],
            title='Validazione Sensori Meteo: Temperatura vs Umidità Esterna',
            xlabel='Temperatura Esterna (°C)',
            ylabel='Umidità Relativa Esterna (%)',
            output_path=output_dir / 'scatter_6_OutdoorT_vs_OutdoorRH.png',
            date_range=date_range,
            diagonal=False
        )


def create_summary_report(df, output_dir, column_mapping):
    """
    Generate comprehensive analysis summary report.
    
    Args:
        df (pd.DataFrame): Dataset
        output_dir (Path): Output directory
        column_mapping (dict): Column name mapping
    """
    summary_path = output_dir / 'analysis_summary.txt'
    
    print("\n📋 Generating comprehensive analysis summary...")
    
    # Find columns
    cols = {}
    for std_name in column_mapping.keys():
        cols[std_name] = find_column(df, std_name, column_mapping)
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SCATTER PLOTS AND DISTRIBUTIONS ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {len(df):,} rows, {len(df.columns)} columns\n")
        f.write(f"Period: {df['Time'].min()} to {df['Time'].max()}\n")
        f.write("\n")
        
        # PART 1: DISTRIBUTION ANALYSIS
        f.write("=" * 80 + "\n")
        f.write("PART 1: DISTRIBUTION ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Purpose: Justify cleaning thresholds, identify outliers, understand seasonal patterns\n\n")
        
        # Analyze distributions
        dist_vars = [
            ('TM', 'Temperatura Mandata', '°C'),
            ('TR', 'Temperatura Ripresa', '°C'),
            ('T_Sat', 'Temperatura Saturazione', '°C'),
            ('SP_TM', 'Setpoint TM', '°C'),
            ('Fan_M', 'Fan Mandata', '%'),
            ('Fan_R', 'Fan Ripresa', '%'),
            ('T_Ext', 'Outdoor Temp', '°C'),
            ('Umid_Ext', 'Outdoor RH', '%')
        ]
        
        for std_name, display_name, unit in dist_vars:
            col = cols.get(std_name)
            if col and col in df.columns:
                data = df[col].dropna()
                if len(data) > 0:
                    q1 = data.quantile(0.25)
                    q3 = data.quantile(0.75)
                    iqr = q3 - q1
                    lower_fence = q1 - TUKEY_FENCE * iqr
                    upper_fence = q3 + TUKEY_FENCE * iqr
                    outliers = data[(data < lower_fence) | (data > upper_fence)]
                    outlier_pct = (len(outliers) / len(data)) * 100
                    
                    f.write(f"{display_name}:\n")
                    f.write(f"  Range: {data.min():.2f} to {data.max():.2f} {unit}\n")
                    f.write(f"  Mean ± Std: {data.mean():.2f} ± {data.std():.2f} {unit}\n")
                    f.write(f"  Median [Q1, Q3]: {data.median():.2f} [{q1:.2f}, {q3:.2f}] {unit}\n")
                    f.write(f"  Outliers: {len(outliers):,} ({outlier_pct:.2f}%)\n")
                    f.write(f"  Suggested cleaning: Remove < {lower_fence:.2f} or > {upper_fence:.2f} {unit}\n")
                    f.write("\n")
        
        # PART 2: SCATTER PLOT ANALYSIS
        f.write("=" * 80 + "\n")
        f.write("PART 2: SCATTER PLOT ANALYSIS (Feature Relationships)\n")
        f.write("=" * 80 + "\n\n")
        
        # 1. Control Tracking
        if cols['TM'] and cols['SP_TM']:
            mask = df[cols['TM']].notna() & df[cols['SP_TM']].notna()
            tm = df.loc[mask, cols['TM']]
            sp_tm = df.loc[mask, cols['SP_TM']]
            
            tracking_error = (tm - sp_tm).abs()
            mean_error = tracking_error.mean()
            max_error = tracking_error.max()
            good_tracking = (tracking_error <= 2.0).sum()
            good_tracking_pct = (good_tracking / len(tracking_error)) * 100
            
            f.write("1. CONTROL TRACKING (TM vs SP_TM):\n")
            f.write(f"   Correlation: {tm.corr(sp_tm):.3f}\n")
            f.write(f"   Mean tracking error: {mean_error:.2f}°C\n")
            f.write(f"   Max tracking error: {max_error:.2f}°C\n")
            f.write(f"   Within ±2°C: {good_tracking:,} points ({good_tracking_pct:.1f}%)\n")
            
            if good_tracking_pct > 85:
                f.write("   ✓ ASSESSMENT: Excellent control tracking\n")
            elif good_tracking_pct > 70:
                f.write("   ✓ ASSESSMENT: Good control tracking\n")
            elif good_tracking_pct > 50:
                f.write("   ⚠️  ASSESSMENT: Moderate tracking issues\n")
            else:
                f.write("   ❌ ASSESSMENT: Significant control problems\n")
            f.write("\n")
        
        # 2. Thermal Relationships
        if cols['TR'] and cols['TM']:
            mask = df[cols['TR']].notna() & df[cols['TM']].notna()
            tr = df.loc[mask, cols['TR']]
            tm = df.loc[mask, cols['TM']]
            correlation = tr.corr(tm)
            thermal_diff = tr - tm
            
            f.write("2. THERMAL RELATIONSHIPS (TR vs TM):\n")
            f.write(f"   Correlation: {correlation:.3f}\n")
            f.write(f"   Mean TR-TM: {thermal_diff.mean():.2f}°C\n")
            f.write(f"   Std TR-TM: {thermal_diff.std():.2f}°C\n")
            
            if correlation > 0.85:
                f.write("   ✓ Strong relationship - likely single operational mode dominant\n")
            elif correlation > 0.6:
                f.write("   ℹ️  Moderate relationship - multiple modes may be present\n")
            else:
                f.write("   ⚠️  Weak relationship - check for regime mixing or sensor issues\n")
            f.write("\n")
        
        # 3. Weather Influence
        if cols['T_Ext'] and cols['TM']:
            mask = df[cols['T_Ext']].notna() & df[cols['TM']].notna()
            t_ext = df.loc[mask, cols['T_Ext']]
            tm = df.loc[mask, cols['TM']]
            correlation = t_ext.corr(tm)
            
            f.write("3. SEASONAL INFLUENCE (OutdoorT vs TM):\n")
            f.write(f"   Correlation: {correlation:.3f}\n")
            f.write(f"   Outdoor range: {t_ext.min():.1f}°C to {t_ext.max():.1f}°C\n")
            f.write(f"   Supply range: {tm.min():.1f}°C to {tm.max():.1f}°C\n")
            
            if abs(correlation) > 0.7:
                f.write("   ✓ Strong outdoor influence - good outdoor reset control\n")
            elif abs(correlation) > 0.4:
                f.write("   ℹ️  Moderate outdoor influence\n")
            else:
                f.write("   ⚠️  Weak outdoor influence - fixed setpoints or poor control?\n")
            f.write("\n")
        
        # 4. System Balance
        if cols['Fan_M'] and cols['Fan_R']:
            mask = df[cols['Fan_M']].notna() & df[cols['Fan_R']].notna()
            fan_m = df.loc[mask, cols['Fan_M']]
            fan_r = df.loc[mask, cols['Fan_R']]
            correlation = fan_m.corr(fan_r)
            balance = (fan_m - fan_r).abs()
            
            f.write("4. SYSTEM BALANCE (FanM vs FanR):\n")
            f.write(f"   Correlation: {correlation:.3f}\n")
            f.write(f"   Mean difference: {(fan_m - fan_r).mean():.2f}%\n")
            f.write(f"   Mean absolute difference: {balance.mean():.2f}%\n")
            
            if correlation > 0.85:
                f.write("   ✓ Well-coordinated fan operation\n")
            elif correlation > 0.6:
                f.write("   ℹ️  Moderate coordination\n")
            else:
                f.write("   ⚠️  Poor coordination - check control strategy\n")
            f.write("\n")
        
        # 5. Fan Effort vs Effect
        if cols['Fan_M'] and cols['TR'] and cols['TM']:
            mask = (df[cols['Fan_M']].notna() & 
                   df[cols['TR']].notna() & 
                   df[cols['TM']].notna())
            fan_m = df.loc[mask, cols['Fan_M']]
            thermal_diff = df.loc[mask, cols['TR']] - df.loc[mask, cols['TM']]
            correlation = fan_m.corr(thermal_diff)
            
            f.write("5. FAN EFFORT VS THERMAL EFFECT (FanM vs TR-TM):\n")
            f.write(f"   Correlation: {correlation:.3f}\n")
            f.write(f"   Mean thermal diff: {thermal_diff.mean():.2f}°C\n")
            
            if abs(correlation) > 0.5:
                f.write("   ✓ Clear relationship between fan speed and thermal effect\n")
            else:
                f.write("   ℹ️  Weak relationship - complex control or mode-dependent behavior\n")
            f.write("\n")
        
        # 6. Weather Sensor Validation
        if cols['T_Ext'] and cols['Umid_Ext']:
            mask = df[cols['T_Ext']].notna() & df[cols['Umid_Ext']].notna()
            t_ext = df.loc[mask, cols['T_Ext']]
            rh = df.loc[mask, cols['Umid_Ext']]
            correlation = t_ext.corr(rh)
            
            f.write("6. WEATHER SENSOR VALIDATION (OutdoorT vs OutdoorRH):\n")
            f.write(f"   Correlation: {correlation:.3f}\n")
            f.write(f"   Typical inverse relationship expected (negative correlation)\n")
            
            if correlation < -0.3:
                f.write("   ✓ Sensors show expected inverse relationship\n")
            elif correlation > 0.3:
                f.write("   ⚠️  Unexpected positive correlation - check sensors\n")
            else:
                f.write("   ℹ️  Weak correlation - may indicate sensor issues or complex weather\n")
            f.write("\n")
        
        # RECOMMENDATIONS
        f.write("=" * 80 + "\n")
        f.write("RECOMMENDATIONS FOR THESIS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("DISTRIBUTION PLOTS to include:\n")
        f.write("  • TM, TR distributions → Show operational temperature ranges\n")
        f.write("  • OutdoorT distribution → Justify seasonal analysis\n")
        f.write("  • Fan distributions → Show typical modulation patterns\n\n")
        
        f.write("SCATTER PLOTS to include:\n")
        f.write("  • TM vs SP_TM → Control quality (most critical)\n")
        f.write("  • TR vs TM → Operational regimes\n")
        f.write("  • OutdoorT vs TM → Seasonal behavior\n")
        f.write("  • FanM vs (TR-TM) → System efficiency insights\n\n")
        
        f.write("Use distributions to:\n")
        f.write("  ✓ Justify outlier removal thresholds\n")
        f.write("  ✓ Support data cleaning decisions\n")
        f.write("  ✓ Show seasonal differences\n\n")
        
        f.write("Use scatter plots to:\n")
        f.write("  ✓ Validate control performance\n")
        f.write("  ✓ Identify operational modes\n")
        f.write("  ✓ Support modeling decisions (regime separation)\n")
        f.write("=" * 80 + "\n")
    
    print(f"   ✓ Saved: {summary_path.name}")


def set_cell_background(cell, color):
    """Set background color for table cell."""
    shading_elm = OxmlElement('w:shd')
    shading_elm.set(qn('w:fill'), color)
    cell._element.get_or_add_tcPr().append(shading_elm)


def add_plot_to_document(doc, plot_path, title, description):
    """
    Add a plot to the document in table format.
    
    Args:
        doc: Document object
        plot_path (Path): Path to plot image
        title (str): Plot title
        description (str): Italian description
    """
    # Create table with 3 rows (title, image, description)
    table = doc.add_table(rows=3, cols=1)
    table.style = 'Table Grid'
    
    # Row 1: Title
    title_cell = table.rows[0].cells[0]
    title_paragraph = title_cell.paragraphs[0]
    title_run = title_paragraph.add_run(title)
    title_run.font.size = Pt(14)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(0, 51, 102)  # Dark blue
    title_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_cell_background(title_cell, 'E8F4F8')  # Light blue background
    
    # Row 2: Image
    image_cell = table.rows[1].cells[0]
    image_paragraph = image_cell.paragraphs[0]
    image_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    if plot_path.exists():
        # Add image with appropriate width (6.5 inches for good readability)
        image_paragraph.add_run().add_picture(str(plot_path), width=Inches(6.5))
    else:
        error_run = image_paragraph.add_run(f"⚠️ Immagine non trovata: {plot_path.name}")
        error_run.font.italic = True
        error_run.font.color.rgb = RGBColor(255, 0, 0)  # Red
    
    # Row 3: Description
    desc_cell = table.rows[2].cells[0]
    desc_paragraph = desc_cell.paragraphs[0]
    desc_run = desc_paragraph.add_run(description)
    desc_run.font.size = Pt(11)
    desc_paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    set_cell_background(desc_cell, 'FFF9E6')  # Light yellow background
    
    # Add spacing after table
    doc.add_paragraph()


def generate_word_report(df, output_dir, config):
    """
    Generate comprehensive Word document with all plots and Italian descriptions.
    
    Args:
        df (pd.DataFrame): Dataset (for date range)
        output_dir (Path): Directory containing plots
        config (ConfigParser): Configuration object
    """
    print("\n" + "=" * 80)
    print("📝 GENERATING WORD REPORT")
    print("=" * 80)
    
    # Get configuration details
    building_id = config.get('global', 'building_id')
    ahu_unit = config.get('global', 'ahu_unit')
    season = config.get('global', 'season')
    year = config.get('global', 'year')
    
    # Get date range from data
    date_min = df['Time'].min().strftime('%d/%m/%Y')
    date_max = df['Time'].max().strftime('%d/%m/%Y')
    date_range_text = f"{date_min} - {date_max}"
    
    print(f"\n📋 Report Information:")
    print(f"   Building: {building_id} - {ahu_unit}")
    print(f"   Season: {season} {year}")
    print(f"   Period: {date_range_text}")
    
    # Create Word document
    doc = Document()
    
    # Set document margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.5)
        section.bottom_margin = Inches(0.5)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)
    
    # ===== TITLE PAGE =====
    title = doc.add_heading('Analisi Distribuzioni e Scatter Plot', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    subtitle = doc.add_heading(f'{building_id} - {ahu_unit}', level=1)
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    date_heading = doc.add_heading(f'Periodo: {date_range_text}', level=2)
    date_heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in date_heading.runs:
        run.font.color.rgb = RGBColor(220, 50, 50)
    
    doc.add_paragraph()
    
    # Add introduction
    intro = doc.add_paragraph()
    intro_text = (
        f"Questo documento presenta un'analisi completa delle distribuzioni e delle relazioni "
        f"tra variabili del sistema HVAC {ahu_unit} nell'edificio {building_id} durante la stagione "
        f"{season} {year}. L'analisi include 8 grafici di distribuzione che mostrano i pattern "
        f"operativi delle principali variabili di sistema, e 6 scatter plot che rivelano le relazioni "
        f"funzionali tra coppie di variabili chiave."
    )
    intro.add_run(intro_text)
    intro.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    
    doc.add_page_break()
    
    # ===== DISTRIBUTION PLOTS =====
    section_heading = doc.add_heading('PARTE 1: ANALISI DELLE DISTRIBUZIONI', level=1)
    for run in section_heading.runs:
        run.font.color.rgb = RGBColor(0, 102, 204)
    
    section_intro = doc.add_paragraph()
    section_intro_text = (
        "Le distribuzioni delle variabili HVAC rivelano i pattern operativi tipici, la variabilità "
        "del sistema, e la presenza di anomalie. L'istogramma mostra la frequenza delle misurazioni, "
        "la curva KDE fornisce una stima della distribuzione continua, e il box plot evidenzia "
        "quartili e outliers."
    )
    section_intro.add_run(section_intro_text)
    section_intro.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    doc.add_paragraph()
    
    print("\n📊 Adding distribution plots...")
    
    # Distribution plots with descriptions
    dist_plots = [
        ('dist_TM.png', 'Distribuzione: Temperatura Mandata',
         'Distribuzione della temperatura di mandata del sistema HVAC. L\'istogramma mostra la frequenza delle misurazioni, '
         'con la curva KDE che rappresenta la distribuzione continua. Le linee verticali indicano media (rossa) e mediana (arancione). '
         'Il box plot evidenzia i quartili e gli outliers. Questo grafico è fondamentale per giustificare le soglie di pulizia dati.'),
        
        ('dist_TR.png', 'Distribuzione: Temperatura Ripresa',
         'Distribuzione della temperatura di ripresa dall\'ambiente. Rappresenta le condizioni interne prima del trattamento. '
         'Il confronto tra media e mediana rivela l\'asimmetria della distribuzione, mentre il box plot evidenzia la variabilità operativa.'),
        
        ('dist_T_Sat.png', 'Distribuzione: Temperatura Saturazione',
         'La temperatura di saturazione è critica per il controllo dell\'umidità e la prevenzione della condensa. '
         'Una distribuzione stretta indica controllo preciso, valori anomali segnalano possibili problemi di calibrazione.'),
        
        ('dist_SP_TM.png', 'Distribuzione: Setpoint Temperatura Mandata',
         'Distribuzione dei setpoint (valori target) impostati. I setpoint mostrano tipicamente una distribuzione discreta '
         'con pochi valori dominanti, riflettendo le strategie di controllo programmate.'),
        
        ('dist_Fan_M.png', 'Distribuzione: Modulazione Ventilatore Mandata',
         'La modulazione del ventilatore di mandata (%) indica l\'intensità di funzionamento. Valori bassi = standby, '
         'valori medi = operatività normale, valori elevati = condizioni di picco. Gli outliers possono indicare emergenze.'),
        
        ('dist_Fan_R.png', 'Distribuzione: Modulazione Ventilatore Ripresa',
         'La modulazione del ventilatore di ripresa mostra l\'intensità di estrazione dell\'aria. Deve essere bilanciata '
         'con la mandata per mantenere la corretta pressurizzazione degli ambienti.'),
        
        ('dist_T_Ext.png', 'Distribuzione: Temperatura Esterna',
         'La temperatura esterna è il principale driver delle condizioni operative HVAC. La forma della distribuzione '
         'rivela pattern stagionali e severità delle condizioni climatiche.'),
        
        ('dist_Umid_Ext.png', 'Distribuzione: Umidità Esterna',
         'L\'umidità relativa esterna influenza il carico latente del sistema, specialmente in estate quando la deumidificazione '
         'è critica. Valori elevati aumentano il consumo energetico.')
    ]
    
    for i, (filename, title_text, description) in enumerate(dist_plots, 1):
        plot_path = output_dir / filename
        print(f"   {i}/8: {filename}")
        add_plot_to_document(doc, plot_path, title_text, description)
        if i < len(dist_plots):
            doc.add_page_break()
    
    doc.add_page_break()
    
    # ===== SCATTER PLOTS =====
    section_heading = doc.add_heading('PARTE 2: ANALISI SCATTER PLOT', level=1)
    for run in section_heading.runs:
        run.font.color.rgb = RGBColor(204, 102, 0)
    
    section_intro = doc.add_paragraph()
    section_intro_text = (
        "Gli scatter plot rivelano le relazioni funzionali tra coppie di variabili, permettendo di valutare "
        "la qualità del controllo, identificare regimi operativi, e validare l'integrità dei sensori."
    )
    section_intro.add_run(section_intro_text)
    section_intro.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    doc.add_paragraph()
    
    print("\n📈 Adding scatter plots...")
    
    # Scatter plots with descriptions
    scatter_plots = [
        ('scatter_1_TM_vs_SP_TM.png', 'Control Tracking: Temperatura Mandata vs Setpoint',
         'Confronto tra temperatura di mandata effettiva e setpoint programmato. La linea diagonale rappresenta il tracking perfetto. '
         'Punti vicini alla diagonale = controllo efficace. La banda di tolleranza (±2°C) definisce l\'accuratezza accettabile. '
         'Essenziale per valutare la qualità del sistema di controllo.'),
        
        ('scatter_2_TR_vs_TM.png', 'Relazione Termica: Temperatura Ripresa vs Mandata',
         'Relazione tra aria di ripresa (condizioni interne) e aria di mandata (aria trattata). '
         'In riscaldamento TM > TR, in raffreddamento TM < TR. La correlazione indica la coerenza del sistema. '
         'Cruciale per identificare e separare le modalità operative.'),
        
        ('scatter_3_OutdoorT_vs_TM.png', 'Influenza Stagionale: Temperatura Esterna vs Mandata',
         'Analisi dell\'influenza della temperatura esterna sulla temperatura di mandata. '
         'Rivela la strategia di compensazione climatica: temperature esterne basse richiedono temperature di mandata elevate. '
         'Fondamentale per comprendere l\'adattamento alle condizioni climatiche.'),
        
        ('scatter_4_FanM_vs_FanR.png', 'Bilanciamento Sistema: Ventilatore Mandata vs Ripresa',
         'Verifica del bilanciamento tra ventilatori. In sistemi bilanciati, i punti seguono la diagonale. '
         'Deviazioni sistematiche indicano strategie di pressurizzazione intenzionali. '
         'La banda di tolleranza (±5%) definisce il range operativo accettabile.'),
        
        ('scatter_5_FanM_vs_ThermalDiff.png', 'Sforzo Ventilatore vs Effetto Termico',
         'Confronto tra sforzo del ventilatore (modulazione) e effetto termico (TR−TM). '
         'Valori positivi = riscaldamento, valori negativi = raffreddamento. '
         'Utile per identificare anomalie operative e valutare l\'efficienza del trasferimento termico.'),
        
        ('scatter_6_OutdoorT_vs_OutdoorRH.png', 'Validazione Sensori Meteo: Temperatura vs Umidità',
         'Verifica la coerenza dei sensori meteorologici. Relazione fisica attesa: aria calda = RH più bassa. '
         'Outliers o pattern anomali indicano problemi di calibrazione. Essenziale per validare i dati meteorologici.')
    ]
    
    for i, (filename, title_text, description) in enumerate(scatter_plots, 1):
        plot_path = output_dir / filename
        print(f"   {i}/6: {filename}")
        add_plot_to_document(doc, plot_path, title_text, description)
        if i < len(scatter_plots):
            doc.add_page_break()
    
    # Save document
    project_root = Path(__file__).parent.parent.parent
    report_dir = project_root / 'processed_data' / 'distributions_and_scatter' / output_dir.name
    report_dir.mkdir(parents=True, exist_ok=True)
    
    output_filename = f"{output_dir.name}_Analisi_Distribuzioni_e_Scatter.docx"
    output_path = report_dir / output_filename
    
    print(f"\n💾 Saving Word document...")
    doc.save(str(output_path))
    
    print(f"\n✅ Word report generated successfully!")
    print(f"   📄 Document: {output_filename}")
    print(f"   📁 Location: {report_dir}")
    print(f"   📊 Content: 8 distributions + 6 scatter plots with Italian descriptions")


def main():
    """Main execution function."""
    print("=" * 80)
    print("SCATTER PLOTS AND DISTRIBUTIONS - HVAC Data Exploration")
    print("=" * 80)
    
    # Load configuration
    script_dir = Path(__file__).parent
    config_paths = [
        script_dir.parent.parent / 'config.ini',
        script_dir.parent / 'config.ini',
        Path('config.ini')
    ]
    
    config_path = None
    for path in config_paths:
        if path.exists():
            config_path = str(path)
            break
    
    if not config_path:
        print("❌ Error: config.ini not found!")
        sys.exit(1)
    
    config = load_config(config_path)
    
    # Find merged data file
    merged_file = find_merged_data(config)
    
    # Load data
    df = load_data(merged_file)
    
    # Setup output directory
    building_id = config.get('global', 'building_id')
    ahu_unit = config.get('global', 'ahu_unit')
    season = config.get('global', 'season')
    year = config.get('global', 'year')
    dataset_name = f"{building_id}_{ahu_unit}_{season}{year}"
    
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / 'plots' / 'distributions_and_scatter' / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir}")
    
    # Get column mapping
    column_mapping = get_column_mapping()
    
    # Part 1: Generate distribution plots
    print("\n" + "=" * 80)
    print("PART 1: DISTRIBUTION ANALYSIS")
    print("=" * 80)
    create_distribution_plots(df, output_dir, column_mapping)
    
    # Part 2: Generate scatter plots
    print("\n" + "=" * 80)
    print("PART 2: SCATTER PLOT ANALYSIS")
    print("=" * 80)
    generate_scatter_plots(df, output_dir, column_mapping, config)
    
    # Generate comprehensive summary
    create_summary_report(df, output_dir, column_mapping)
    
    # Generate Word report with all plots
    generate_word_report(df, output_dir, config)
    
    # Final summary
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"📊 Plots saved in: {output_dir}")
    print(f"📋 Summary report: analysis_summary.txt")
    print(f"📄 Word report: {dataset_name}_Analisi_Distribuzioni_e_Scatter.docx")
    print("\n📈 Generated outputs:")
    print("   • 8 distribution plots (histogram + box plot)")
    print("   • 6 scatter plots (feature-pair relationships)")
    print("   • 1 comprehensive analysis summary (TXT)")
    print("   • 1 Word document with all plots and Italian descriptions")
    print("\n💡 Use these visualizations to:")
    print("   ✓ Justify data cleaning decisions")
    print("   ✓ Validate control performance")
    print("   ✓ Identify operational regimes")
    print("   ✓ Support thesis discussion")
    print("=" * 80)


if __name__ == "__main__":
    main()
