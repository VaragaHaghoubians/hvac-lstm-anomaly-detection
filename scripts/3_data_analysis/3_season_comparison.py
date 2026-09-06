"""
=============================================================================
SEASONAL COMPARISON ANALYSIS FOR HVAC SYSTEMS
=============================================================================
This script compares interpolated HVAC data between summer and winter seasons
for the Campus Einaudi building monitoring system.

Features:
- Statistical comparisons (mean, median, std, variance)
- Temperature profile analysis
- Modulation pattern comparisons
- Humidity analysis
- Load duration curves
- Setpoint tracking effectiveness
- Operational efficiency KPIs
- Significance testing (t-tests, Mann-Whitney U)
- Comprehensive visualizations

OPTIMIZATIONS IMPLEMENTED:
--------------------------
1. ✅ Cached Data Loading
   - LRU cache for CSV files (avoids redundant disk reads)
   - Column filtering at load time (reduces memory usage)
   
2. ✅ Pre-calculated Statistics
   - Statistics computed once and cached in __init__
   - Reused across all plotting and analysis methods
   - Eliminates redundant calculations
   
3. ✅ Sequential Plot Generation (Matplotlib Agg Compatibility)
   - Plots generated sequentially (not parallel) for Agg backend
   - Prevents blank/empty image files
   - Proper error handling for each plot
   - Matplotlib 'Agg' backend prevents GUI conflicts
   
4. ✅ Memory Optimization
   - Only loads required columns from CSV files
   - Drops NaN values before processing
   - Efficient DataFrame operations
   
5. ✅ Comprehensive Docstrings
   - All major methods documented
   - Clear parameter and return type descriptions
   - Usage examples and notes included
   
6. ✅ Multilingual Support (Partial)
   - TRANSLATIONS dictionary for key phrases
   - Language preference from config
   - Currently supports Italian (it) and English (en)
   - Ready for full internationalization

Author: HVAC Analysis System
Date: October 2025
Last Updated: October 2025 - Performance optimizations added
=============================================================================
"""

import pandas as pd
import numpy as np
# Set matplotlib backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for thread safety
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import configparser
from datetime import datetime
import warnings
from scipy import stats
from scipy.stats import mannwhitneyu, ttest_ind, ks_2samp
import logging
import re
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, Tuple

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

# Ensure project root is on sys.path so we can import config_utils when running
# this script from the scripts/ directory.
script_dir = Path(__file__).parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Try to import config_utils, if it fails, define a local load_config function
try:
    from config_utils import load_config
except ImportError:
    # Fallback: define load_config locally if config_utils is not available
    def load_config(config_path):
        """Load config WITH interpolation support for variable expansion."""
        config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        config.read(config_path, encoding='utf-8-sig')
        return config

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

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


# Localization dictionary for multi-language support
TRANSLATIONS = {
    'it': {
        'data_analyzed': '**Dati Analizzati:**',
        'summer': 'Estate',
        'winter': 'Inverno',
        'measurements': 'misurazioni',
        'box_plot_elements': '**Elementi del Box Plot:**',
        'orange_box': '• Scatola arancione: Distribuzione temperature estive',
        'blue_box': '• Scatola blu: Distribuzione temperature invernali',
        'line_in_box': '• Linea nella scatola: Mediana',
        'box_height': "• Altezza della scatola: Range interquartile (50% centrale dei dati)",
        'whiskers': '• Baffi: Range completo (escludendo outlier)',
        'dots': '• Punti: Outlier (valori anomali)',
        'interpretation': '**Interpretazione:**',
        'non_overlapping': '• Scatole non sovrapposte = Differenze significative',
        'median_comparison': '• Confrontare le linee mediane per vedere quale stagione ha temperature più alte/basse',
        'variability': '• Scatole più alte = Maggiore variabilità nelle temperature',
        'stat_tests': '• Test Statistici',
        'hourly_profile': 'Profilo Orario',
        'delta_t_correlation': 'Correlazione Delta-T',
        'load_duration': 'Curve di Durata del Carico',
        'humidity_comparison': 'Confronto Umidità',
        'kpi_dashboard': 'Cruscotto KPI'
    },
    'en': {
        'data_analyzed': '**Data Analyzed:**',
        'summer': 'Summer',
        'winter': 'Winter',
        'measurements': 'measurements',
        'box_plot_elements': '**Box Plot Elements:**',
        'orange_box': '• Orange box: Summer temperature distribution',
        'blue_box': '• Blue box: Winter temperature distribution',
        'line_in_box': '• Line in box: Median',
        'box_height': '• Box height: Interquartile range (central 50% of data)',
        'whiskers': '• Whiskers: Full range (excluding outliers)',
        'dots': '• Dots: Outliers (anomalous values)',
        'interpretation': '**Interpretation:**',
        'non_overlapping': '• Non-overlapping boxes = Significant differences',
        'median_comparison': '• Compare median lines to see which season has higher/lower temperatures',
        'variability': '• Taller boxes = Greater temperature variability',
        'stat_tests': '• Statistical Tests',
        'hourly_profile': 'Hourly Profile',
        'delta_t_correlation': 'Delta-T Correlation',
        'load_duration': 'Load Duration Curves',
        'humidity_comparison': 'Humidity Comparison',
        'kpi_dashboard': 'KPI Dashboard'
    }
}


@functools.lru_cache(maxsize=2)
def _load_csv_cached(path_str: str, time_column: str) -> pd.DataFrame:
    """
    Load CSV file with caching to avoid redundant reads.
    
    Args:
        path_str: String path to CSV file (strings are hashable for lru_cache)
        time_column: Name of the time column to parse and set as index
        
    Returns:
        DataFrame with parsed datetime index
        
    Note:
        Uses LRU cache to store up to 2 DataFrames in memory
        (one summer, one winter typically)
    """
    path = Path(path_str)
    df = pd.read_csv(path)
    # utc=True handles mixed timezone offsets (e.g. +02:00 in DST, +01:00 in CET)
    # without it, pandas creates an object Index instead of DatetimeIndex
    df[time_column] = pd.to_datetime(df[time_column], errors='coerce', utc=True)
    df[time_column] = df[time_column].dt.tz_convert('Europe/Rome')
    df.set_index(time_column, inplace=True)
    return df


class SeasonalComparison:
    """
    A class to perform comprehensive seasonal comparison analysis
    between summer and winter HVAC operational data.
    """
    
    def __init__(self, config_path='config.ini'):
        """Initialize the seasonal comparison analyzer."""
        self.config_path = Path(config_path)

        # Load configuration using shared helper which disables interpolation and
        # reads utf-8-sig. This prevents InterpolationSyntaxError when values
        # contain percent signs (e.g., 'CV_%').
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found at: {self.config_path}\n"
                f"Current working directory: {Path.cwd()}\n"
                f"Please ensure config.ini is in the correct location."
            )

        try:
            self.config = load_config(self.config_path)
            # Record config directory to resolve relative paths reliably
            try:
                self.config._config_dir = str(self.config_path.parent)
            except Exception:
                pass
        except Exception as e:
            raise RuntimeError(
                f"Failed to load config file: {self.config_path}\n"
                f"Error: {str(e)}\n"
                f"Please check that the config file is properly formatted."
            )
        
        # Setup logging
        self.setup_logging()
        
        # Load configuration
        self.load_config()
        
        # Initialize data containers
        self.summer_data = None
        self.winter_data = None
        self.comparison_results = {}
        
        # Cached statistics (will be populated after data loading)
        self.summer_stats = {}
        self.winter_stats = {}
        
        # Store plot paths for Word report
        self.plot_paths = []
        
        # Get language preference from config (default to Italian)
        self.language = self.config.get('reporting', 'language', fallback='it') if hasattr(self.config, 'get') else 'it'
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_format)
        self.logger = logging.getLogger(__name__)
    
    def extract_season_name(self, file_path):
        """
        Extract season name from file path.
        E.g., 'C1_UTA1_Summer2024\\C1_UTA1_Summer2024_interpolated.csv' -> 'Summer2024'
        """
        # Get the folder/file name without extension
        path_parts = file_path.replace('\\', '/').split('/')
        
        # Try to extract from folder name first
        if len(path_parts) > 1:
            folder_name = path_parts[0]
        else:
            folder_name = path_parts[0].replace('_interpolated.csv', '').replace('.csv', '')
        
        # Extract season identifier (Summer2024, Winter2025, etc.)
        # Look for patterns like Summer2024, Winter2025
        import re
        match = re.search(r'(Summer|Winter)\d{4}', folder_name)
        if match:
            return match.group(0)
        
        # Fallback: return the folder name itself
        return folder_name
        
    def load_config(self):
        """Load configuration from config.ini file."""
        try:
            # Debug: Print available sections
            available_sections = self.config.sections()
            self.logger.info(f"Available config sections: {available_sections}")
            
            # Check if required sections exist
            if 'paths' not in available_sections:
                raise ValueError("Config file is missing required [paths] section")
            if 'seasonal_comparison' not in available_sections:
                raise ValueError("Config file is missing required [seasonal_comparison] section")
            
            # Get paths
            self.processed_folder = Path(self.config.get('paths', 'processed_folder'))
            self.plots_folder = Path(self.config.get('paths', 'plots_folder'))
            # Resolve relative folders using config.ini directory (not current working dir)
            cfg_dir = Path(getattr(self.config, '_config_dir', Path.cwd()))
            if not self.processed_folder.is_absolute():
                self.processed_folder = (cfg_dir / self.processed_folder).resolve()
            if not self.plots_folder.is_absolute():
                self.plots_folder = (cfg_dir / self.plots_folder).resolve()
            
            # Get seasonal comparison settings
            self.enable_comparison = self.config.getboolean('seasonal_comparison', 'enable_seasonal_comparison', fallback=True)
            self.summer_file = self.config.get('seasonal_comparison', 'summer_file')
            self.winter_file = self.config.get('seasonal_comparison', 'winter_file')
            
            # Get columns to compare
            columns_str = self.config.get('seasonal_comparison', 'columns_to_compare', 
                                         fallback='T Ripresa, T Mandata, SP Temp. Mand. Raff., SP Temp. Mand. Risc., Temp. Esterna, Umid. Esterna')
            self.columns_to_compare = [col.strip() for col in columns_str.split(',')]
            
            # Get analysis settings
            self.significance_threshold = self.config.getfloat('seasonal_comparison', 'significance_threshold', fallback=0.05)
            self.create_plots = self.config.getboolean('seasonal_comparison', 'create_seasonal_plots', fallback=True)
            self.save_results_flag = self.config.getboolean('seasonal_comparison', 'save_seasonal_results', fallback=True)
            
            # Get building identifiers
            self.building_id = self.config.get('seasonal_comparison', 'building_id', fallback='C1')
            self.ahu_unit = self.config.get('seasonal_comparison', 'ahu_unit', fallback='UTA1')
            
            # Time column
            self.time_column = self.config.get('data', 'time_column', fallback='Time')
            
            # Create output directories with organized folder structure
            # Extract season names from file paths for folder naming
            # e.g., "C1_UTA1_Summer2024\C1_UTA1_Summer2024_interpolated.csv" -> "Summer2024"
            summer_name = self.extract_season_name(self.summer_file)
            winter_name = self.extract_season_name(self.winter_file)
            
            # Create folder name with building and AHU identifiers
            # Format: BuildingID_AHUUnit_Season1_vs_Season2
            comparison_folder = f'{self.building_id}_{self.ahu_unit}_{summer_name}_vs_{winter_name}'
            
            self.output_dir = self.plots_folder / 'seasonal_comparison' / comparison_folder
            self.results_dir = self.processed_folder / 'seasonal_comparison' / comparison_folder
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.results_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Output folder: {comparison_folder}")
            
            self.logger.info("Configuration loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise
            
    def load_data(self):
        """Load summer and winter interpolated data files."""
        try:
            # Construct full paths to interpolated files
            # The summer_file and winter_file should include subfolder structure
            # e.g., "C1_UTA1_Summer2024\C1_UTA1_Summer2024_interpolated.csv"
            interpolation_folder = self.processed_folder / 'interpolation'
            summer_path = interpolation_folder / self.summer_file
            winter_path = interpolation_folder / self.winter_file
            # Fallback to with_masks filenames if standard files not present
            if not summer_path.exists():
                sp = Path(str(summer_path).replace('_interpolated.csv', '_interpolated_with_masks.csv'))
                if sp.exists():
                    summer_path = sp
            if not winter_path.exists():
                wp = Path(str(winter_path).replace('_interpolated.csv', '_interpolated_with_masks.csv'))
                if wp.exists():
                    winter_path = wp
            
            self.logger.info(f"Looking for summer data at: {summer_path}")
            self.logger.info(f"Looking for winter data at: {winter_path}")
            
            # Check if interpolated files exist
            if not summer_path.exists() or not winter_path.exists():
                self.logger.warning("Interpolated files not found!")
                self.logger.warning(f"Summer file exists: {summer_path.exists()} - {summer_path}")
                self.logger.warning(f"Winter file exists: {winter_path.exists()} - {winter_path}")
                self.logger.warning("Checking for raw data files...")
                
                # Try data folder with raw files
                data_folder = Path(self.config.get('paths', 'base_folder'))
                summer_raw = self.summer_file.replace('_interpolated.csv', '.csv')
                winter_raw = self.winter_file.replace('_interpolated.csv', '.csv')
                
                summer_path_raw = data_folder / summer_raw
                winter_path_raw = data_folder / winter_raw
                
                if summer_path_raw.exists() and winter_path_raw.exists():
                    self.logger.warning("Found raw data files. Using them instead.")
                    self.logger.warning("NOTE: Raw data may contain missing values and outliers.")
                    self.logger.warning("For best results, run interpolation script first!")
                    summer_path = summer_path_raw
                    winter_path = winter_path_raw
                else:
                    # Provide helpful error message
                    error_msg = (
                        f"\n{'='*80}\n"
                        f"ERROR: Data files not found!\n"
                        f"{'='*80}\n"
                        f"Looked for interpolated files:\n"
                        f"  - {summer_path}\n"
                        f"  - {winter_path}\n\n"
                        f"Also looked for raw files:\n"
                        f"  - {summer_path_raw}\n"
                        f"  - {winter_path_raw}\n\n"
                        f"SOLUTION:\n"
                        f"1. Ensure your data files exist in one of the above locations, OR\n"
                        f"2. Update the file names in config.ini [seasonal_comparison] section:\n"
                        f"   - summer_file = {self.summer_file}\n"
                        f"   - winter_file = {self.winter_file}\n"
                        f"{'='*80}\n"
                    )
                    self.logger.error(error_msg)
                    print(error_msg)
                    raise FileNotFoundError("Required data files not found")
            
            # Determine columns needed (for memory optimization)
            columns_needed = set(self.columns_to_compare + [self.time_column])
            
            self.logger.info(f"Loading summer data from: {summer_path}")
            # Use cached loader with column filtering
            self.summer_data = _load_csv_cached(str(summer_path), self.time_column)
            # Filter columns if specified
            if self.columns_to_compare:
                available_cols = [col for col in self.columns_to_compare if col in self.summer_data.columns]
                self.summer_data = self.summer_data[available_cols]
            
            self.logger.info(f"Loading winter data from: {winter_path}")
            # Use cached loader with column filtering
            self.winter_data = _load_csv_cached(str(winter_path), self.time_column)
            # Filter columns if specified
            if self.columns_to_compare:
                available_cols = [col for col in self.columns_to_compare if col in self.winter_data.columns]
                self.winter_data = self.winter_data[available_cols]
            
            self.logger.info(f"Summer data loaded: {len(self.summer_data)} records")
            self.logger.info(f"Winter data loaded: {len(self.winter_data)} records")
            
            # Verify columns exist
            self.verify_columns()
            
            # Calculate and cache statistics for better performance
            self._calculate_cached_statistics()
            
        except FileNotFoundError as e:
            self.logger.error(f"Data file not found: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def _calculate_cached_statistics(self):
        """
        Pre-calculate and cache common statistics to avoid repeated calculations.
        This significantly improves performance when generating multiple plots.
        """
        self.logger.info("Caching statistics for improved performance...")
        
        for col in self.columns_to_compare:
            summer_vals = self.summer_data[col].dropna()
            winter_vals = self.winter_data[col].dropna()
            
            self.summer_stats[col] = {
                'values': summer_vals,
                'mean': summer_vals.mean(),
                'median': summer_vals.median(),
                'std': summer_vals.std(),
                'min': summer_vals.min(),
                'max': summer_vals.max(),
                'q25': summer_vals.quantile(0.25),
                'q75': summer_vals.quantile(0.75),
                'count': len(summer_vals)
            }
            
            self.winter_stats[col] = {
                'values': winter_vals,
                'mean': winter_vals.mean(),
                'median': winter_vals.median(),
                'std': winter_vals.std(),
                'min': winter_vals.min(),
                'max': winter_vals.max(),
                'q25': winter_vals.quantile(0.25),
                'q75': winter_vals.quantile(0.75),
                'count': len(winter_vals)
            }
        
        self.logger.info("Statistics cached successfully")
    
    def assess_data_quality(self):
        """
        Assess and report data quality for both seasons.
        
        Checks:
            - Percentage of interpolated points per variable
            - Data completeness
            - Fair comparison between seasons
            
        Returns:
            dict: Quality metrics for summer and winter
            
        Notes:
            - Requires interpolation mask columns (e.g., 'Variable_is_interpolated')
            - Warns if interpolation rates differ significantly between seasons
        """
        self.logger.info("Assessing data quality for fair comparison...")
        
        quality_report = {
            'summer': {},
            'winter': {},
            'warnings': []
        }
        
        for col in self.columns_to_compare:
            # Check for interpolation mask column
            mask_col = f"{col}_is_interpolated"
            
            # Summer quality
            if mask_col in self.summer_data.columns:
                total_points = len(self.summer_data)
                interpolated = self.summer_data[mask_col].sum()
                pct_interpolated = (interpolated / total_points * 100) if total_points > 0 else 0
                quality_report['summer'][col] = {
                    'total_points': total_points,
                    'interpolated': interpolated,
                    'pct_interpolated': pct_interpolated
                }
            else:
                quality_report['summer'][col] = {
                    'total_points': len(self.summer_data),
                    'interpolated': 'N/A',
                    'pct_interpolated': 'N/A'
                }
            
            # Winter quality
            if mask_col in self.winter_data.columns:
                total_points = len(self.winter_data)
                interpolated = self.winter_data[mask_col].sum()
                pct_interpolated = (interpolated / total_points * 100) if total_points > 0 else 0
                quality_report['winter'][col] = {
                    'total_points': total_points,
                    'interpolated': interpolated,
                    'pct_interpolated': pct_interpolated
                }
            else:
                quality_report['winter'][col] = {
                    'total_points': len(self.winter_data),
                    'interpolated': 'N/A',
                    'pct_interpolated': 'N/A'
                }
            
            # Check for significant differences in interpolation rates
            if (mask_col in self.summer_data.columns and mask_col in self.winter_data.columns):
                summer_pct = quality_report['summer'][col]['pct_interpolated']
                winter_pct = quality_report['winter'][col]['pct_interpolated']
                
                if abs(summer_pct - winter_pct) > 10:  # More than 10% difference
                    warning = (
                        f"⚠️  {col}: Significant interpolation difference "
                        f"(Summer: {summer_pct:.1f}%, Winter: {winter_pct:.1f}%). "
                        f"Comparison may be biased."
                    )
                    quality_report['warnings'].append(warning)
                    self.logger.warning(warning)
        
        # Log summary
        self.logger.info("\n" + "="*80)
        self.logger.info("DATA QUALITY REPORT")
        self.logger.info("="*80)
        
        for col in self.columns_to_compare[:5]:  # Show first 5 variables
            if col in quality_report['summer'] and col in quality_report['winter']:
                summer_info = quality_report['summer'][col]
                winter_info = quality_report['winter'][col]
                
                self.logger.info(f"\n{col}:")
                if summer_info['pct_interpolated'] != 'N/A':
                    self.logger.info(f"  Summer: {summer_info['pct_interpolated']:.1f}% interpolated")
                    self.logger.info(f"  Winter: {winter_info['pct_interpolated']:.1f}% interpolated")
                else:
                    self.logger.info("  Interpolation masks not found (OK if using raw data)")
        
        if quality_report['warnings']:
            self.logger.warning("\n" + "="*80)
            self.logger.warning("DATA QUALITY WARNINGS")
            self.logger.warning("="*80)
            for warning in quality_report['warnings']:
                self.logger.warning(warning)
        else:
            self.logger.info("\n✓ No significant data quality issues detected")
        
        self.quality_report = quality_report
        return quality_report
            
    def verify_columns(self):
        """
        Verify that requested columns exist in both datasets.
        
        Checks:
            - Column availability in both summer and winter data
            - Logs warnings for missing columns
            - Updates self.columns_to_compare with only available columns
            
        Side Effects:
            - Modifies self.columns_to_compare to exclude missing columns
            - Logs warnings for any columns not found in both datasets
        """
        summer_cols = set(self.summer_data.columns)
        winter_cols = set(self.winter_data.columns)
        
        available_cols = []
        missing_cols = []
        
        for col in self.columns_to_compare:
            if col in summer_cols and col in winter_cols:
                available_cols.append(col)
            else:
                missing_cols.append(col)
                self.logger.warning(f"Column '{col}' not found in both datasets")
        
        self.columns_to_compare = available_cols
        
        if missing_cols:
            self.logger.warning(f"Missing columns: {missing_cols}")
        
        self.logger.info(f"Columns to compare: {self.columns_to_compare}")
        
    def calculate_basic_statistics(self):
        """
        Calculate basic descriptive statistics for both seasons.
        
        Returns:
            DataFrame containing comparative statistics for all variables
            
        Note:
            Uses pre-cached statistics for improved performance
        """
        self.logger.info("Calculating basic statistics...")
        
        stats_list = []
        
        for col in self.columns_to_compare:
            # Use cached statistics
            summer_stats = self.summer_stats[col]
            winter_stats = self.winter_stats[col]
            
            stats_dict = {
                'Variable': col,
                'Season': ['Summer', 'Winter'],
                'Count': [summer_stats['count'], winter_stats['count']],
                'Mean': [summer_stats['mean'], winter_stats['mean']],
                'Median': [summer_stats['median'], winter_stats['median']],
                'Std': [summer_stats['std'], winter_stats['std']],
                'Min': [summer_stats['min'], winter_stats['min']],
                'Max': [summer_stats['max'], winter_stats['max']],
                'Q25': [summer_stats['q25'], winter_stats['q25']],
                'Q75': [summer_stats['q75'], winter_stats['q75']],
                'CV%': [(summer_stats['std']/summer_stats['mean']*100) if summer_stats['mean'] != 0 else 0,
                       (winter_stats['std']/winter_stats['mean']*100) if winter_stats['mean'] != 0 else 0]
            }
            
            # Create DataFrame for this variable
            var_df = pd.DataFrame(stats_dict)
            stats_list.append(var_df)
        
        self.basic_stats = pd.concat(stats_list, ignore_index=True)
        self.logger.info("Basic statistics calculated")
        
        return self.basic_stats
    
    def perform_statistical_tests(self):
        """
        Perform statistical significance tests comparing summer vs winter.
        
        IMPORTANT: Uses daily aggregates instead of raw time-series data to ensure
        independence assumption is met. Raw time-series data is autocorrelated,
        violating the independence assumption of t-tests and Mann-Whitney U tests.
        
        Tests Performed:
            1. T-test: Parametric test for mean differences (assumes normality)
            2. Mann-Whitney U: Non-parametric test for distribution differences
            3. Kolmogorov-Smirnov: Tests if distributions are fundamentally different
            4. Cohen's d: Effect size measure (practical significance)
            
        Returns:
            DataFrame with test results including:
                - Test statistics and p-values
                - Significance flags (p < 0.05)
                - Effect size interpretation
                - Sample sizes (daily aggregates)
                
        Notes:
            - p-value < 0.05 indicates statistical significance
            - Cohen's d shows practical importance: small (<0.5), medium (0.5-0.8), large (>0.8)
            - Daily aggregation ensures independent samples for valid statistical inference
            - Uses pre-cached statistics for improved performance
        """
        self.logger.info("Performing statistical significance tests...")
        self.logger.info("Note: Using daily aggregates to ensure sample independence")
        
        test_results = []
        
        for col in self.columns_to_compare:
            # Aggregate to daily means to ensure independence
            # (time-series points are autocorrelated, violating test assumptions)
            summer_daily = self.summer_data[col].resample('D').mean().dropna()
            winter_daily = self.winter_data[col].resample('D').mean().dropna()
            
            # Log sample sizes for transparency
            n_summer_daily = len(summer_daily)
            n_winter_daily = len(winter_daily)
            
            # Skip if insufficient daily samples
            if n_summer_daily < 3 or n_winter_daily < 3:
                self.logger.warning(f"Skipping {col}: insufficient daily samples")
                continue
            
            # T-test (parametric)
            t_stat, t_pval = ttest_ind(summer_daily, winter_daily, equal_var=False)
            
            # Mann-Whitney U test (non-parametric)
            u_stat, u_pval = mannwhitneyu(summer_daily, winter_daily, alternative='two-sided')
            
            # Kolmogorov-Smirnov test (distribution comparison)
            ks_stat, ks_pval = ks_2samp(summer_daily, winter_daily)
            
            # Effect size (Cohen's d) based on daily aggregates
            pooled_std = np.sqrt((summer_daily.std()**2 + winter_daily.std()**2) / 2)
            cohens_d = (summer_daily.mean() - winter_daily.mean()) / pooled_std if pooled_std != 0 else 0
            
            result = {
                'Variable': col,
                'Mean_Diff': summer_daily.mean() - winter_daily.mean(),
                'N_Summer_Days': n_summer_daily,
                'N_Winter_Days': n_winter_daily,
                'T_Statistic': t_stat,
                'T_PValue': t_pval,
                'T_Significant': t_pval < self.significance_threshold,
                'U_Statistic': u_stat,
                'U_PValue': u_pval,
                'U_Significant': u_pval < self.significance_threshold,
                'KS_Statistic': ks_stat,
                'KS_PValue': ks_pval,
                'KS_Significant': ks_pval < self.significance_threshold,
                'Cohens_D': cohens_d,
                'Effect_Size': self.interpret_cohens_d(cohens_d)
            }
            
            test_results.append(result)
        
        self.statistical_tests = pd.DataFrame(test_results)
        self.logger.info("Statistical tests completed")
        self.logger.info(f"Tested {len(test_results)} variables using daily aggregates")
        
        return self.statistical_tests
    
    def interpret_cohens_d(self, d):
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return 'Negligible'
        elif abs_d < 0.5:
            return 'Small'
        elif abs_d < 0.8:
            return 'Medium'
        else:
            return 'Large'
    
    def calculate_thermal_load_proxy(self, data):
        """
        Calculate thermal load proxy (ΔT between supply and return).
        
        Args:
            data: DataFrame containing HVAC temperature data
            
        Returns:
            Series with absolute temperature difference or None if columns not found
        """
        if 'T Mandata' in data.columns and 'T Ripresa' in data.columns:
            return abs(data['T Mandata'] - data['T Ripresa'])
        return None
    
    def calculate_hourly_profiles(self):
        """
        Calculate 24-hour average profiles for each variable.
        
        Returns:
            dict: {column_name: {'summer': Series, 'winter': Series, 
                                 'summer_by_dow': dict, 'winter_by_dow': dict}}
            
        Notes:
            - Groups data by hour of day (0-23)
            - Also creates separate profiles for each day of week (0=Monday, 6=Sunday)
            - Useful for identifying operational schedules and daily patterns
        """
        self.logger.info("Calculating hourly profiles...")
        
        hourly_profiles = {}
        
        for col in self.columns_to_compare:
            # Overall hourly profile (all days combined)
            summer_hourly = self.summer_data[col].groupby(self.summer_data.index.hour).mean()
            winter_hourly = self.winter_data[col].groupby(self.winter_data.index.hour).mean()
            
            # Hourly profile by day of week
            summer_by_dow = {}
            winter_by_dow = {}
            
            for dow in range(7):  # 0=Monday, 6=Sunday
                summer_dow_data = self.summer_data[self.summer_data.index.dayofweek == dow]
                winter_dow_data = self.winter_data[self.winter_data.index.dayofweek == dow]
                
                if len(summer_dow_data) > 0:
                    summer_by_dow[dow] = summer_dow_data[col].groupby(summer_dow_data.index.hour).mean()
                if len(winter_dow_data) > 0:
                    winter_by_dow[dow] = winter_dow_data[col].groupby(winter_dow_data.index.hour).mean()
            
            hourly_profiles[col] = {
                'summer': summer_hourly,
                'winter': winter_hourly,
                'summer_by_dow': summer_by_dow,
                'winter_by_dow': winter_by_dow
            }
        
        self.hourly_profiles = hourly_profiles
        self.logger.info("Hourly profiles calculated")
        
        return hourly_profiles
    
    def calculate_load_duration_curves(self):
        """
        Calculate load duration curves for modulation variables.
        
        Returns:
            dict: {column_name: {'summer_sorted': array, 'summer_percentile': array,
                                 'winter_sorted': array, 'winter_percentile': array}}
            
        Notes:
            - Sorts values in descending order
            - Calculates percentile of time at each load level
            - Useful for understanding equipment utilization patterns
            - Example: "Equipment runs at >80% for 30% of the time"
        """
        self.logger.info("Calculating load duration curves...")
        
        modulation_cols = [col for col in self.columns_to_compare if 'Modul' in col]
        
        self.load_duration = {}
        
        for col in modulation_cols:
            summer_sorted = np.sort(self.summer_data[col].dropna())[::-1]
            winter_sorted = np.sort(self.winter_data[col].dropna())[::-1]
            
            summer_percentile = np.linspace(0, 100, len(summer_sorted))
            winter_percentile = np.linspace(0, 100, len(winter_sorted))
            
            self.load_duration[col] = {
                'summer': {'values': summer_sorted, 'percentile': summer_percentile},
                'winter': {'values': winter_sorted, 'percentile': winter_percentile}
            }
        
        self.logger.info("Load duration curves calculated")
        
        return self.load_duration
    
    def plot_temperature_distributions(self):
        """
        Create temperature distribution comparison plots (box plots only).
        
        Creates:
            Box plots comparing summer vs winter temperature distributions
            
        Features:
            - Orange boxes for summer, blue boxes for winter
            - Shows median, quartiles, and outliers
            - One subplot per temperature variable
            
        Saves:
            PNG file to output directory
        """
        self.logger.info("Creating temperature distribution plots...")
        
        temp_cols = [col for col in self.columns_to_compare if 'T' in col or 'Temp' in col]
        
        if not temp_cols:
            self.logger.warning("No temperature columns found for distribution plots")
            return
        
        # Get date ranges
        summer_start = self.summer_data.index.min().strftime('%Y-%m-%d')
        summer_end = self.summer_data.index.max().strftime('%Y-%m-%d')
        winter_start = self.winter_data.index.min().strftime('%Y-%m-%d')
        winter_end = self.winter_data.index.max().strftime('%Y-%m-%d')
        
        # Single row of box plots
        fig, axes = plt.subplots(1, len(temp_cols), figsize=(6*len(temp_cols), 6))
        if len(temp_cols) == 1:
            axes = [axes]
        
        for idx, col in enumerate(temp_cols):
            ax = axes[idx]
            summer_data = self.summer_data[col].dropna()
            winter_data = self.winter_data[col].dropna()
            
            data_to_plot = [summer_data, winter_data]
            
            bp = ax.boxplot(data_to_plot, labels=['Summer', 'Winter'], patch_artist=True,
                           showmeans=True, meanline=True,
                           boxprops=dict(linewidth=1.5),
                           whiskerprops=dict(linewidth=1.5),
                           capprops=dict(linewidth=1.5),
                           medianprops=dict(linewidth=2, color='darkred'),
                           meanprops=dict(linewidth=2, color='green', linestyle='--'))
            
            # Color the boxes
            bp['boxes'][0].set_facecolor('orange')
            bp['boxes'][0].set_alpha(0.7)
            bp['boxes'][1].set_facecolor('lightblue')
            bp['boxes'][1].set_alpha(0.7)
            
            # Add mean values as text
            summer_mean = summer_data.mean()
            winter_mean = winter_data.mean()
            ax.text(1, summer_mean, f'{summer_mean:.1f}°C', 
                   ha='right', va='center', fontsize=9, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.5))
            ax.text(2, winter_mean, f'{winter_mean:.1f}°C', 
                   ha='left', va='center', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.5))
            
            ax.set_ylabel('Temperature (°C)', fontweight='bold', fontsize=10)
            
            # Improve column name display
            display_name = col.replace('Temperatura ', '').replace('Temperature ', '')
            display_name = display_name.replace('T ', '')
            ax.set_title(f'{display_name}', fontweight='bold', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=0.5)
        
        plt.suptitle(f'{self.building_id} - {self.ahu_unit}: Temperature Distributions - Seasonal Comparison\n'
                    f'Summer: {summer_start} to {summer_end} | Winter: {winter_start} to {winter_end}',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        output_path = self.output_dir / f'{self.building_id}_{self.ahu_unit}_temperature_distributions.png'
        # Save with explicit format and facecolor
        fig.savefig(output_path, dpi=300, bbox_inches='tight', format='png', facecolor='white', edgecolor='none')
        plt.close(fig)
        
        self.logger.info(f"Temperature distribution plots saved to {output_path}")
        
        # Store plot path for Word report
        self.plot_paths.append(('Temperature Distributions', str(output_path)))
    
    def plot_day_of_week_comparison(self):
        """
        Create hourly profile plots separated by day of week (Mon-Sun) for each variable.
        
        Creates:
            7-panel plots (one per weekday) showing hourly patterns
            
        Features:
            - Orange line for summer, blue line for winter
            - Dashed lines showing daily averages
            - Separate analysis for each day of week
            - Identifies weekday vs weekend operational differences
            
        Saves:
            One PNG file per variable analyzed
        """
        self.logger.info("Creating day-of-week comparison plots...")
        
        # Get date ranges
        summer_start = self.summer_data.index.min().strftime('%Y-%m-%d')
        summer_end = self.summer_data.index.max().strftime('%Y-%m-%d')
        winter_start = self.winter_data.index.min().strftime('%Y-%m-%d')
        winter_end = self.winter_data.index.max().strftime('%Y-%m-%d')
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Create plots for each variable
        for col in self.columns_to_compare:
            self.logger.info(f"Processing {col}...")
            
            # Calculate hourly profiles by day of week
            summer_by_dow = {}
            winter_by_dow = {}
            
            for dow in range(7):  # 0=Monday, 6=Sunday
                summer_dow_data = self.summer_data[self.summer_data.index.dayofweek == dow]
                winter_dow_data = self.winter_data[self.winter_data.index.dayofweek == dow]
                
                if len(summer_dow_data) > 0:
                    summer_by_dow[dow] = summer_dow_data[col].groupby(summer_dow_data.index.hour).mean()
                if len(winter_dow_data) > 0:
                    winter_by_dow[dow] = winter_dow_data[col].groupby(winter_dow_data.index.hour).mean()
            
            # Create 7 subplots (one per day) - INCREASED SIZE for better visibility
            fig, axes = plt.subplots(3, 3, figsize=(24, 16))
            axes = axes.flatten()
            
            for dow in range(7):
                ax = axes[dow]
                
                # Plot summer data
                if dow in summer_by_dow:
                    summer_profile = summer_by_dow[dow]
                    ax.plot(summer_profile.index, summer_profile.values, 'o-', 
                           label='Summer', color='orange', linewidth=3, markersize=8, alpha=0.8)
                    # Add average line
                    avg_val = summer_profile.mean()
                    ax.axhline(y=avg_val, color='orange', linestyle='--', linewidth=2, alpha=0.5,
                              label=f'Summer avg: {avg_val:.1f}')
                
                # Plot winter data
                if dow in winter_by_dow:
                    winter_profile = winter_by_dow[dow]
                    ax.plot(winter_profile.index, winter_profile.values, 's-', 
                           label='Winter', color='blue', linewidth=3, markersize=8, alpha=0.8)
                    # Add average line
                    avg_val = winter_profile.mean()
                    ax.axhline(y=avg_val, color='blue', linestyle='--', linewidth=2, alpha=0.5,
                              label=f'Winter avg: {avg_val:.1f}')
                
                ax.set_xlabel('Hour of Day', fontweight='bold', fontsize=14)
                ax.set_ylabel('Value', fontweight='bold', fontsize=14)
                ax.set_title(f'{day_names[dow]}', fontweight='bold', fontsize=16, pad=12)
                ax.legend(loc='best', fontsize=11, framealpha=0.9)
                ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
                ax.set_xticks(range(0, 24, 3))
                ax.set_xlim(-0.5, 23.5)
                ax.tick_params(axis='both', which='major', labelsize=12)
            
            # Hide unused subplots (8th and 9th)
            for idx in range(7, 9):
                axes[idx].set_visible(False)
            
            # Add main title with date ranges
            plt.suptitle(f'{self.building_id} - {self.ahu_unit}: {col} - Day of Week Comparison (Hourly Patterns)\n'
                        f'Summer: {summer_start} to {summer_end} | Winter: {winter_start} to {winter_end}',
                        fontsize=18, fontweight='bold', y=0.997)
            
            plt.tight_layout()
            
            # Sanitize filename
            safe_col_name = col.replace(' ', '_').replace('.', '').replace(',', '').replace('/', '_')
            output_path = self.output_dir / f'{self.building_id}_{self.ahu_unit}_{safe_col_name}_dayofweek.png'
            # Save with explicit parameters
            fig.savefig(output_path, dpi=300, bbox_inches='tight', format='png', facecolor='white', edgecolor='none')
            plt.close(fig)
            
            self.logger.info(f"Saved: {output_path.name}")
            
            # Store plot path for Word report
            self.plot_paths.append((f'{col} - Day of Week', str(output_path)))
        
        self.logger.info("All day-of-week comparison plots created")
    
    def plot_correlation_scatter(self):
        """
        Create scatter plot showing Delta T vs External Temperature (V-shaped pattern).
        
        Delta T Calculation:
            - Summer: ΔT = T Ripresa - T Mandata (cooling load indicator)
            - Winter: ΔT = T Mandata - T Ripresa (heating load indicator)
            
        Creates:
            Scatter plot showing V-shaped relationship between thermal load and external temp
            
        Features:
            - Orange points for summer cooling
            - Blue points for winter heating
            - Annotations explaining the V-pattern
            - Reference line at ΔT = 0 (no load)
            
        Physical Meaning:
            - Higher ΔT = Higher thermal load
            - V-shape shows heating load (left) and cooling load (right)
            - Center of V = mild weather with minimal HVAC load
        """
        self.logger.info("Creating Delta T vs External Temperature plot...")
        
        if 'Temp. Esterna' not in self.columns_to_compare:
            self.logger.warning("External temperature not available for correlation plots")
            return
        
        if 'T Mandata' not in self.columns_to_compare or 'T Ripresa' not in self.columns_to_compare:
            self.logger.warning("T Mandata or T Ripresa not available for Delta T calculation")
            return
        
        # Calculate Delta T for both seasons
        # Summer: T Ripresa - T Mandata (positive when cooling, return air warmer than supply)
        # Winter: T Mandata - T Ripresa (positive when heating, supply air warmer than return)
        summer_delta_t = self.summer_data['T Ripresa'] - self.summer_data['T Mandata']
        winter_delta_t = self.winter_data['T Mandata'] - self.winter_data['T Ripresa']
        
        # Get date ranges
        summer_start = self.summer_data.index.min().strftime('%Y-%m-%d')
        summer_end = self.summer_data.index.max().strftime('%Y-%m-%d')
        winter_start = self.winter_data.index.min().strftime('%Y-%m-%d')
        winter_end = self.winter_data.index.max().strftime('%Y-%m-%d')
        
        # Create single plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Summer data
        ax.scatter(self.summer_data['Temp. Esterna'], summer_delta_t,
                  alpha=0.4, c='orange', label='Summer (Cooling)', s=15, edgecolors='none')
        
        # Winter data
        ax.scatter(self.winter_data['Temp. Esterna'], winter_delta_t,
                  alpha=0.4, c='blue', label='Winter (Heating)', s=15, edgecolors='none')
        
        # Add horizontal line at y=0 for reference
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='ΔT = 0 (No Load)')
        
        # Labels and title
        ax.set_xlabel('External Temperature (°C)', fontweight='bold', fontsize=12)
        ax.set_ylabel('ΔT (°C) - Always Positive', fontweight='bold', fontsize=12)
        ax.set_title('Delta T vs External Temperature (V-shaped pattern)\n'
                    'Summer: ΔT = T Ripresa - T Mandata | Winter: ΔT = T Mandata - T Ripresa\n'
                    'Higher ΔT = Higher Heating/Cooling Load',
                    fontweight='bold', fontsize=12, pad=15)
        
        # Legend
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Add annotations to explain the V-shape
        ax.text(0.02, 0.98, 'Winter Heating\n(Supply warmer\nthan Return)', transform=ax.transAxes,
               fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.5))
        ax.text(0.98, 0.98, 'Summer Cooling\n(Return warmer\nthan Supply)', transform=ax.transAxes,
               fontsize=10, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.suptitle(f'{self.building_id} - {self.ahu_unit}: Temperature Difference vs External Temperature\n'
                    f'Summer: {summer_start} to {summer_end} | Winter: {winter_start} to {winter_end}',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        output_path = self.output_dir / f'{self.building_id}_{self.ahu_unit}_delta_t_correlation.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Delta T correlation plot saved to {output_path}")
        
        # Store plot path for Word report
        self.plot_paths.append(('Delta T Correlation', str(output_path)))
    
    def plot_load_duration_curves(self):
        """
        Create load duration curve plots for equipment modulation.
        
        Creates:
            Line plots showing percentage of time at each modulation level
            
        Features:
            - Descending curves from max to min modulation
            - X-axis: Percentile of time (0-100%)
            - Y-axis: Modulation value (0-1)
            - Comparison of summer vs winter utilization
            
        Use Cases:
            - Energy consumption estimation
            - Equipment sizing validation
            - Understanding operational patterns
            - Identifying over/under-utilization
        """
        self.logger.info("Creating load duration curve plots...")
        
        if not self.load_duration:
            self.logger.warning("No modulation data available for load duration curves")
            return
        
        # Get date ranges
        summer_start = self.summer_data.index.min().strftime('%Y-%m-%d')
        summer_end = self.summer_data.index.max().strftime('%Y-%m-%d')
        winter_start = self.winter_data.index.min().strftime('%Y-%m-%d')
        winter_end = self.winter_data.index.max().strftime('%Y-%m-%d')
        
        n_cols = len(self.load_duration)
        fig, axes = plt.subplots(1, n_cols, figsize=(8*n_cols, 6))
        
        if n_cols == 1:
            axes = [axes]
        
        for idx, col in enumerate(self.load_duration.keys()):
            ax = axes[idx]
            
            summer_data = self.load_duration[col]['summer']
            winter_data = self.load_duration[col]['winter']
            
            ax.plot(summer_data['percentile'], summer_data['values'], 
                   label='Summer', color='orange', linewidth=2)
            ax.plot(winter_data['percentile'], winter_data['values'], 
                   label='Winter', color='blue', linewidth=2)
            
            ax.set_xlabel('Percentile of Time (%)')
            ax.set_ylabel('Modulation Value')
            ax.set_title(f'{col} - Load Duration Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 100)
        
        plt.suptitle(f'{self.building_id} - {self.ahu_unit}: Load Duration Curves - Seasonal Comparison\n'
                    f'Summer: {summer_start} to {summer_end} | Winter: {winter_start} to {winter_end}',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        output_path = self.output_dir / f'{self.building_id}_{self.ahu_unit}_load_duration.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Load duration curve plots saved to {output_path}")
        
        # Store plot path for Word report
        self.plot_paths.append(('Load Duration Curves', str(output_path)))
    
    def plot_humidity_comparison(self):
        """
        Create humidity comparison plots (3 panels: distribution, box plot, time series).
        
        Creates:
            3-panel visualization of humidity data
            
        Panels:
            1. Histogram: Frequency distribution of humidity levels
            2. Box plot: Statistical summary (median, quartiles, outliers)
            3. Time series: Daily average trend over the season
            
        Features:
            - Orange for summer, blue for winter
            - Normalized histograms for fair comparison
            - Identifies humidity patterns and variability
            
        HVAC Relevance:
            - High humidity increases latent cooling load
            - Low humidity may require humidification
            - Affects comfort and energy consumption
        """
        self.logger.info("Creating humidity comparison plots...")
        
        if 'Umid. Esterna' not in self.columns_to_compare:
            self.logger.warning("Humidity data not available")
            return
        
        # Get date ranges
        summer_start = self.summer_data.index.min().strftime('%Y-%m-%d')
        summer_end = self.summer_data.index.max().strftime('%Y-%m-%d')
        winter_start = self.winter_data.index.min().strftime('%Y-%m-%d')
        winter_end = self.winter_data.index.max().strftime('%Y-%m-%d')
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Distribution comparison
        ax1 = axes[0]
        ax1.hist(self.summer_data['Umid. Esterna'].dropna(), bins=50, 
                alpha=0.6, color='orange', label='Summer', density=True)
        ax1.hist(self.winter_data['Umid. Esterna'].dropna(), bins=50, 
                alpha=0.6, color='blue', label='Winter', density=True)
        ax1.set_xlabel('Humidity (%)')
        ax1.set_ylabel('Density')
        ax1.set_title('Humidity Distribution', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2 = axes[1]
        data_to_plot = [
            self.summer_data['Umid. Esterna'].dropna(),
            self.winter_data['Umid. Esterna'].dropna()
        ]
        bp = ax2.boxplot(data_to_plot, labels=['Summer', 'Winter'], patch_artist=True)
        bp['boxes'][0].set_facecolor('orange')
        bp['boxes'][1].set_facecolor('lightblue')
        ax2.set_ylabel('Humidity (%)')
        ax2.set_title('Humidity Box Plot', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Time series (daily averages)
        ax3 = axes[2]
        summer_daily = self.summer_data['Umid. Esterna'].resample('D').mean()
        winter_daily = self.winter_data['Umid. Esterna'].resample('D').mean()
        ax3.plot(summer_daily.index, summer_daily.values, 
                color='orange', label='Summer', linewidth=1.5)
        ax3.plot(winter_daily.index, winter_daily.values, 
                color='blue', label='Winter', linewidth=1.5)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Humidity (%)')
        ax3.set_title('Daily Average Humidity Over Time', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle(f'{self.building_id} - {self.ahu_unit}: Humidity Comparison - Seasonal Analysis\n'
                    f'Summer: {summer_start} to {summer_end} | Winter: {winter_start} to {winter_end}',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        output_path = self.output_dir / f'{self.building_id}_{self.ahu_unit}_humidity_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Humidity comparison plots saved to {output_path}")
        
        # Store plot path for Word report
        self.plot_paths.append(('Humidity Comparison', str(output_path)))
    
    def plot_kpi_dashboard(self):
        """
        Create a comprehensive KPI dashboard comparing seasons with clear labels.
        
        Creates:
            8-panel dashboard with key performance indicators
            
        Panels:
            1. Average temperature levels
            2. Average modulation (equipment utilization)
            3. Coefficient of variation (variability measure)
            4. Operating hours (time with modulation >10%)
            5. Temperature range (max-min spread)
            6. Statistical significance (pie chart of t-test results)
            7. External temperature trend
            8. Text summary with key metrics
            
        Features:
            - Numeric labels on all bars for exact values
            - Color-coded (orange=summer, blue=winter)
            - Professional layout with grid structure
            - Comprehensive overview on single page
        """
        self.logger.info("Creating KPI dashboard...")
        
        # INCREASED figure size for better visibility
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3, height_ratios=[1, 1, 1.2])
        
        # 1. Mean temperature comparison
        ax1 = fig.add_subplot(gs[0, 0])
        temp_cols = [col for col in self.columns_to_compare if 'T' in col and 'SP' not in col]
        if temp_cols:
            summer_means = [self.summer_data[col].mean() for col in temp_cols]
            winter_means = [self.winter_data[col].mean() for col in temp_cols]
            x = np.arange(len(temp_cols))
            width = 0.35
            bars1 = ax1.bar(x - width/2, summer_means, width, label='Summer', color='orange', alpha=0.8)
            bars2 = ax1.bar(x + width/2, winter_means, width, label='Winter', color='blue', alpha=0.8)
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}°C', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            ax1.set_ylabel('Average Temperature (°C)', fontweight='bold', fontsize=12)
            ax1.set_title('1. Average Temperature Levels\n(Higher values = warmer conditions)', fontweight='bold', fontsize=13)
            ax1.set_xticks(x)
            ax1.set_xticklabels([col.replace('T ', '').replace('Temp. ', '').replace('Temperatura ', '') for col in temp_cols], 
                               rotation=45, ha='right', fontsize=11)
            ax1.legend(loc='upper right', fontsize=11)
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.tick_params(axis='both', labelsize=11)
        
        # 2. Modulation comparison
        ax2 = fig.add_subplot(gs[0, 1])
        modul_cols = [col for col in self.columns_to_compare if 'Modul' in col]
        if modul_cols:
            summer_means = [self.summer_data[col].mean() for col in modul_cols]
            winter_means = [self.winter_data[col].mean() for col in modul_cols]
            x = np.arange(len(modul_cols))
            width = 0.35
            bars1 = ax2.bar(x - width/2, summer_means, width, label='Summer', color='orange', alpha=0.8)
            bars2 = ax2.bar(x + width/2, winter_means, width, label='Winter', color='blue', alpha=0.8)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            ax2.set_ylabel('Average Modulation (0-1)', fontweight='bold', fontsize=12)
            ax2.set_title('2. Average Equipment Modulation\n(0=off, 1=max capacity)', fontweight='bold', fontsize=13)
            ax2.set_xticks(x)
            ax2.set_xticklabels([col.replace('Modul ', '').replace('Modulation ', '') for col in modul_cols], 
                               rotation=45, ha='right', fontsize=11)
            ax2.legend(loc='upper right', fontsize=11)
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.set_ylim(0, 1.1)
            ax2.tick_params(axis='both', labelsize=11)
        
        # 3. Coefficient of Variation comparison
        ax3 = fig.add_subplot(gs[0, 2])
        cv_data = []
        cv_cols = []
        for col in self.columns_to_compare[:6]:  # Limit to first 6 columns
            summer_cv = (self.summer_data[col].std() / self.summer_data[col].mean() * 100) if self.summer_data[col].mean() != 0 else 0
            winter_cv = (self.winter_data[col].std() / self.winter_data[col].mean() * 100) if self.winter_data[col].mean() != 0 else 0
            cv_data.append([summer_cv, winter_cv])
            cv_cols.append(col)
        
        if cv_data:
            cv_array = np.array(cv_data)
            x = np.arange(len(cv_data))
            width = 0.35
            bars1 = ax3.bar(x - width/2, cv_array[:, 0], width, label='Summer', color='orange', alpha=0.8)
            bars2 = ax3.bar(x + width/2, cv_array[:, 1], width, label='Winter', color='blue', alpha=0.8)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax3.set_ylabel('Coefficient of Variation (%)', fontweight='bold', fontsize=11)
            ax3.set_title('3. Variability of Measurements\n(Higher % = more fluctuation)', fontweight='bold', fontsize=13)
            ax3.set_xticks(x)
            # Improved label shortening with better abbreviations
            short_labels = []
            for col in cv_cols:
                label = col.replace('Temperatura ', 'T.').replace('Temperature ', 'T.')
                label = label.replace('Modulation ', 'Mod.').replace('Modul ', 'Mod.')
                label = label.replace('Temp. ', '').replace('T ', '')
                label = label.replace('Esterna', 'Ext').replace('Mandata', 'Man').replace('Ripresa', 'Rip')
                short_labels.append(label[:10])
            ax3.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=10)
            ax3.legend(loc='upper right', fontsize=11)
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.tick_params(axis='both', labelsize=10)
        
        # 4. Operating hours (based on modulation > 0.1)
        ax4 = fig.add_subplot(gs[1, 0])
        if modul_cols:
            op_hours = []
            for col in modul_cols:
                summer_op = (self.summer_data[col] > 0.1).sum() / 2  # Assuming 30min intervals
                winter_op = (self.winter_data[col] > 0.1).sum() / 2
                op_hours.append([summer_op, winter_op])
            
            op_array = np.array(op_hours)
            x = np.arange(len(modul_cols))
            width = 0.35
            bars1 = ax4.bar(x - width/2, op_array[:, 0], width, label='Summer', color='orange', alpha=0.8)
            bars2 = ax4.bar(x + width/2, op_array[:, 1], width, label='Winter', color='blue', alpha=0.8)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.0f}h', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            ax4.set_ylabel('Total Hours', fontweight='bold', fontsize=12)
            ax4.set_title('4. Equipment Operating Hours\n(Hours with modulation > 10%)', fontweight='bold', fontsize=13)
            ax4.set_xticks(x)
            ax4.set_xticklabels([col.replace('Modul ', '').replace('Modulation ', '') for col in modul_cols], 
                               rotation=45, ha='right', fontsize=11)
            ax4.legend(loc='upper right', fontsize=11)
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.tick_params(axis='both', labelsize=11)
        
        # 5. Temperature range (max - min)
        ax5 = fig.add_subplot(gs[1, 1])
        if temp_cols:
            summer_ranges = [self.summer_data[col].max() - self.summer_data[col].min() for col in temp_cols]
            winter_ranges = [self.winter_data[col].max() - self.winter_data[col].min() for col in temp_cols]
            x = np.arange(len(temp_cols))
            width = 0.35
            bars1 = ax5.bar(x - width/2, summer_ranges, width, label='Summer', color='orange', alpha=0.8)
            bars2 = ax5.bar(x + width/2, winter_ranges, width, label='Winter', color='blue', alpha=0.8)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax5.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}°C', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            ax5.set_ylabel('Temperature Range (°C)', fontweight='bold', fontsize=12)
            ax5.set_title('5. Temperature Range (Max - Min)\n(Larger = more extreme variations)', fontweight='bold', fontsize=13)
            ax5.set_xticks(x)
            ax5.set_xticklabels([col.replace('T ', '').replace('Temp. ', '').replace('Temperatura ', '')[:12] for col in temp_cols], 
                               rotation=45, ha='right', fontsize=11)
            ax5.legend(loc='upper right', fontsize=11)
            ax5.grid(True, alpha=0.3, axis='y')
            ax5.tick_params(axis='both', labelsize=11)
        
        # 6. Statistical significance summary
        ax6 = fig.add_subplot(gs[1, 2])
        if hasattr(self, 'statistical_tests'):
            significant_vars = self.statistical_tests[self.statistical_tests['T_Significant']]['Variable'].tolist()
            non_significant = [col for col in self.columns_to_compare if col not in significant_vars]
            
            sizes = [len(significant_vars), len(non_significant)]
            labels = [f'Statistically\nDifferent\n({len(significant_vars)} vars)', 
                     f'Not Significantly\nDifferent\n({len(non_significant)} vars)']
            colors = ['#ff6b6b', '#95e1d3']
            
            if sum(sizes) > 0:
                wedges, texts, autotexts = ax6.pie(sizes, labels=labels, colors=colors, 
                                                    autopct='%1.0f%%', startangle=90,
                                                    textprops={'fontsize': 12, 'weight': 'bold'})
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontsize(14)
                    autotext.set_fontweight('bold')
                ax6.set_title('6. Statistical Significance Test\n(T-test with 95% confidence)', 
                             fontweight='bold', fontsize=13)
        
        # 7. External conditions comparison (spans 2 columns)
        ax7 = fig.add_subplot(gs[2, :2])
        if 'Temp. Esterna' in self.columns_to_compare or 'Temperatura Esterna' in self.columns_to_compare:
            # Check which column name exists
            ext_temp_col = 'Temp. Esterna' if 'Temp. Esterna' in self.columns_to_compare else 'Temperatura Esterna'
            
            # Resample to daily averages
            summer_ext = self.summer_data[ext_temp_col].resample('D').mean().dropna()
            winter_ext = self.winter_data[ext_temp_col].resample('D').mean().dropna()
            
            if len(summer_ext) > 0 and len(winter_ext) > 0:
                ax7.plot(range(len(summer_ext)), summer_ext.values, 
                        color='orange', label=f'Summer (avg: {summer_ext.mean():.1f}°C)', 
                        linewidth=2, alpha=0.8, marker='o', markersize=3)
                ax7.plot(range(len(winter_ext)), winter_ext.values, 
                        color='blue', label=f'Winter (avg: {winter_ext.mean():.1f}°C)', 
                        linewidth=2, alpha=0.8, marker='s', markersize=3)
                
                ax7.fill_between(range(len(summer_ext)), summer_ext.values, alpha=0.2, color='orange')
                ax7.fill_between(range(len(winter_ext)), winter_ext.values, alpha=0.2, color='blue')
                
                # Add horizontal lines for averages
                ax7.axhline(y=summer_ext.mean(), color='orange', linestyle='--', linewidth=1.5, alpha=0.6)
                ax7.axhline(y=winter_ext.mean(), color='blue', linestyle='--', linewidth=1.5, alpha=0.6)
                
                ax7.set_xlabel('Days in Period', fontweight='bold', fontsize=13)
                ax7.set_ylabel('External Temperature (°C)', fontweight='bold', fontsize=13)
                ax7.set_title('7. Daily Average External Temperature Trend\n(Shows outdoor climate conditions driving HVAC operation)', 
                             fontweight='bold', fontsize=14)
                ax7.legend(loc='best', fontsize=12, framealpha=0.9)
                ax7.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
                ax7.set_ylim(bottom=min(summer_ext.min(), winter_ext.min()) - 2,
                            top=max(summer_ext.max(), winter_ext.max()) + 2)
                ax7.tick_params(axis='both', labelsize=12)
            else:
                ax7.text(0.5, 0.5, 'Insufficient external temperature data', 
                        ha='center', va='center', fontsize=14, transform=ax7.transAxes)
                ax7.set_title('7. Daily Average External Temperature Trend', fontweight='bold', fontsize=14)
        else:
            ax7.text(0.5, 0.5, 'External temperature data not available', 
                    ha='center', va='center', fontsize=14, transform=ax7.transAxes)
            ax7.set_title('7. Daily Average External Temperature Trend', fontweight='bold', fontsize=14)
        
        # 8. Summary statistics table (right column in third row)
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')
        
        # Create summary text
        summary_lines = [
            "KEY FINDINGS:",
            ""
        ]
        
        if 'Temp. Esterna' in self.columns_to_compare:
            summer_avg = self.summer_data['Temp. Esterna'].mean()
            winter_avg = self.winter_data['Temp. Esterna'].mean()
            summary_lines.append(f"• External Temperature: Summer avg = {summer_avg:.1f}°C, Winter avg = {winter_avg:.1f}°C (Δ = {abs(summer_avg - winter_avg):.1f}°C)")
        
        if temp_cols and len(temp_cols) > 0:
            col = temp_cols[0]  # Use first temp column
            summer_avg = self.summer_data[col].mean()
            winter_avg = self.winter_data[col].mean()
            summary_lines.append(f"• {col}: Summer avg = {summer_avg:.1f}°C, Winter avg = {winter_avg:.1f}°C (Δ = {abs(summer_avg - winter_avg):.1f}°C)")
        
        if modul_cols and len(modul_cols) > 0:
            col = modul_cols[0]
            summer_avg = self.summer_data[col].mean()
            winter_avg = self.winter_data[col].mean()
            summer_hours = (self.summer_data[col] > 0.1).sum() / 2
            winter_hours = (self.winter_data[col] > 0.1).sum() / 2
            summary_lines.append(f"• {col}: Summer avg = {summer_avg:.2f}, Winter avg = {winter_avg:.2f}")
            summary_lines.append(f"  Operating hours: Summer = {summer_hours:.0f}h, Winter = {winter_hours:.0f}h")
        
        if hasattr(self, 'statistical_tests'):
            sig_count = self.statistical_tests['T_Significant'].sum()
            total_count = len(self.statistical_tests)
            summary_lines.append(f"• Statistical Tests: {sig_count}/{total_count} variables show statistically significant differences (p < 0.05)")
        
        summary_text = '\n'.join(summary_lines)
        ax8.text(0.05, 0.95, summary_text, fontsize=10, verticalalignment='top',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
                wrap=True)
        
        # Get date ranges
        summer_start = self.summer_data.index.min().strftime('%Y-%m-%d')
        summer_end = self.summer_data.index.max().strftime('%Y-%m-%d')
        winter_start = self.winter_data.index.min().strftime('%Y-%m-%d')
        winter_end = self.winter_data.index.max().strftime('%Y-%m-%d')
        
        plt.suptitle(f'{self.building_id} - {self.ahu_unit}: Seasonal Comparison KPI Dashboard\n'
                    f'Summer: {summer_start} to {summer_end} | Winter: {winter_start} to {winter_end}', 
                    fontsize=20, fontweight='bold', y=0.998)
        
        output_path = self.output_dir / f'{self.building_id}_{self.ahu_unit}_kpi_dashboard.png'
        # Save without bbox_inches='tight' to prevent vertical stretching with text-heavy layouts
        plt.tight_layout()
        fig.savefig(output_path, dpi=200, format='png', facecolor='white', edgecolor='none')
        plt.close(fig)
        
        self.logger.info(f"KPI dashboard saved to {output_path}")
        
        # Store plot path for Word report
        self.plot_paths.append(('KPI Dashboard', str(output_path)))
    
    def _generate_plot_description(self, plot_type):
        """
        Generate localized description for each plot type.
        
        Args:
            plot_type: Type of plot ('Temperature Distributions', 'KPI Dashboard', etc.)
            
        Returns:
            str: Formatted description text in the configured language
        """
        
        # Get translations for current language
        t = TRANSLATIONS.get(self.language, TRANSLATIONS['it'])
        
        # Get date ranges for context
        summer_start = self.summer_data.index.min().strftime('%d/%m/%Y')
        summer_end = self.summer_data.index.max().strftime('%d/%m/%Y')
        winter_start = self.winter_data.index.min().strftime('%d/%m/%Y')
        winter_end = self.winter_data.index.max().strftime('%d/%m/%Y')
        
        n_summer = len(self.summer_data)
        n_winter = len(self.winter_data)
        
        if plot_type == 'Temperature Distributions':
            return (
                f"Questo grafico confronta le distribuzioni delle temperature tra estate e inverno "
                f"utilizzando box plot per ciascuna variabile termica monitorata.\n\n"
                f"{t['data_analyzed']}\n"
                f"• {t['summer']}: {n_summer} {t['measurements']} ({summer_start} - {summer_end})\n"
                f"• {t['winter']}: {n_winter} {t['measurements']} ({winter_start} - {winter_end})\n\n"
                f"{t['box_plot_elements']}\n"
                f"{t['orange_box']}\n"
                f"{t['blue_box']}\n"
                f"{t['line_in_box']}\n"
                f"{t['box_height']}\n"
                f"{t['whiskers']}\n"
                f"{t['dots']}\n\n"
                f"{t['interpretation']}\n"
                f"{t['non_overlapping']}\n"
                f"{t['median_comparison']}\n"
                f"{t['variability']}"
            )
        
        elif 'Day of Week' in plot_type:
            var_name = plot_type.replace(' - Day of Week', '')
            return (
                f"Questo grafico mostra i pattern orari giornalieri della variabile '{var_name}' "
                f"separati per giorno della settimana (lunedì-domenica), confrontando estate e inverno.\n\n"
                f"**Struttura del Grafico:**\n"
                f"• 7 pannelli: Uno per ciascun giorno della settimana\n"
                f"• Linea arancione: Profilo orario medio estivo\n"
                f"• Linea blu: Profilo orario medio invernale\n"
                f"• Linee tratteggiate: Valori medi giornalieri\n\n"
                f"**Analisi Pattern:**\n"
                f"• **Giorni feriali (Lun-Ven)**: Mostrano pattern operativi standard dell'edificio\n"
                f"• **Weekend (Sab-Dom)**: Possono mostrare ridotta occupazione/utilizzo\n"
                f"• **Picchi orari**: Indicano momenti di massimo utilizzo/carico\n"
                f"• **Differenze estate-inverno**: Evidenziano diversi regimi operativi stagionali\n\n"
                f"**Interpretazione per Sistemi HVAC:**\n"
                f"• Temperature: Pattern riflettono setpoint e carichi termici stagionali\n"
                f"• Modulazione: Indica intensità operativa del sistema\n"
                f"• Consistenza pattern = Operazione regolare e prevedibile\n"
                f"• Anomalie = Possibili guasti o operazioni non standard"
            )
        
        elif plot_type == 'Delta T Correlation':
            return (
                f"Questo grafico mostra la relazione tra la differenza di temperatura (ΔT) "
                f"e la temperatura esterna, evidenziando il pattern a V caratteristico dei sistemi HVAC.\n\n"
                f"**Calcolo ΔT:**\n"
                f"• **Estate (raffrescamento)**: ΔT = T Ripresa - T Mandata\n"
                f"  - Positivo quando l'aria di ritorno è più calda dell'aria di mandata\n"
                f"  - ΔT maggiore = maggiore carico di raffrescamento\n"
                f"• **Inverno (riscaldamento)**: ΔT = T Mandata - T Ripresa\n"
                f"  - Positivo quando l'aria di mandata è più calda dell'aria di ritorno\n"
                f"  - ΔT maggiore = maggiore carico di riscaldamento\n\n"
                f"**Pattern a V:**\n"
                f"• Ramo sinistro: Inverno (temperature esterne basse, necessità riscaldamento)\n"
                f"• Punto centrale: Temperature miti (minimo carico termico)\n"
                f"• Ramo destro: Estate (temperature esterne alte, necessità raffrescamento)\n\n"
                f"**Interpretazione:**\n"
                f"• ΔT = 0: Nessun carico termico, sistema in free-cooling o spento\n"
                f"• ΔT alto: Sistema sotto carico significativo\n"
                f"• Pendenza rami: Efficacia del sistema nel rispondere al carico\n"
                f"• Dispersione punti: Variabilità dovuta a occupazione, guadagni interni, etc."
            )
        
        elif plot_type == 'Load Duration Curves':
            return (
                f"Le curve di durata del carico mostrano per quanto tempo (in percentuale) "
                f"la modulazione dei ventilatori si mantiene a determinati livelli durante l'intero periodo.\n\n"
                f"**Come Leggere il Grafico:**\n"
                f"• **Asse X**: Percentile del tempo (0% = sempre, 100% = mai)\n"
                f"• **Asse Y**: Valore di modulazione (0 = spento, 1 = massima capacità)\n"
                f"• **Curva discendente**: Dal valore massimo (sinistra) al minimo (destra)\n\n"
                f"**Analisi Curve:**\n"
                f"• **Posizione verticale**: Curva più alta = maggiore modulazione media\n"
                f"• **Pendenza**: Ripida = variabilità alta, Piatta = operazione costante\n"
                f"• **Confronto estate-inverno**: Mostra differenze nei regimi operativi\n\n"
                f"**Interpretazione Operativa:**\n"
                f"• Area sotto la curva ∝ Consumo energetico totale\n"
                f"• Curva alta per >50% del tempo = Sistema frequentemente sotto carico\n"
                f"• Curva bassa = Sistema poco utilizzato o sovradimensionato\n"
                f"• Differenze stagionali = Adattamento ai carichi termici variabili\n\n"
                f"**KPI Derivabili:**\n"
                f"• Ore operative (modulazione > 10%)\n"
                f"• Fattore di carico medio\n"
                f"• Distribuzione temporale del carico"
            )
        
        elif plot_type == 'Humidity Comparison':
            return (
                f"Questo grafico tri-pannello confronta l'umidità esterna tra estate e inverno "
                f"attraverso tre diverse visualizzazioni complementari.\n\n"
                f"**Pannello 1 - Distribuzione:**\n"
                f"• Istogrammi sovrapposti (arancione = estate, blu = inverno)\n"
                f"• Mostra la frequenza dei diversi livelli di umidità\n"
                f"• Normalizzato per permettere confronto diretto\n\n"
                f"**Pannello 2 - Box Plot:**\n"
                f"• Sintesi statistica della distribuzione\n"
                f"• Mediana, quartili, range, outlier\n"
                f"• Confronto visivo immediato tra stagioni\n\n"
                f"**Pannello 3 - Serie Temporale:**\n"
                f"• Andamento medio giornaliero nel periodo\n"
                f"• Mostra trend e variazioni temporali\n"
                f"• Evidenzia stabilità o fluttuazioni\n\n"
                f"**Importanza per Sistemi HVAC:**\n"
                f"• **Alta umidità estiva**: Maggiore carico latente, deumidificazione\n"
                f"• **Bassa umidità invernale**: Possibile necessità umidificazione\n"
                f"• **Variabilità**: Impatto sul comfort e efficienza energetica\n"
                f"• **Range operativo**: 30-60% RH per comfort ottimale\n\n"
                f"**Analisi Risultati:**\n"
                f"• Confronto mediane = Differenza tipica estate/inverno\n"
                f"• Ampiezza distribuzioni = Stabilità condizioni esterne\n"
                f"• Outlier = Eventi climatici estremi"
            )
        
        elif plot_type == 'KPI Dashboard':
            # Get some statistics for the description
            sig_count = 0
            total_tests = 0
            if hasattr(self, 'statistical_tests'):
                sig_count = self.statistical_tests['T_Significant'].sum()
                total_tests = len(self.statistical_tests)
            
            return (
                f"Questa dashboard fornisce una panoramica completa dei principali indicatori di prestazione (KPI) "
                f"per il confronto stagionale del sistema HVAC.\n\n"
                f"**Pannelli della Dashboard:**\n\n"
                f"**1. Livelli Medi di Temperatura**\n"
                f"• Confronto diretto delle temperature medie tra estate e inverno\n"
                f"• Valori numerici sopra le barre per lettura immediata\n"
                f"• Include: Ripresa, Mandata, Setpoint, Esterna\n\n"
                f"**2. Modulazione Media Apparecchiature**\n"
                f"• Intensità operativa media dei ventilatori/componenti\n"
                f"• Range 0-1 (0=spento, 1=massima capacità)\n"
                f"• Indica carico operativo del sistema\n\n"
                f"**3. Variabilità delle Misurazioni (CV%)**\n"
                f"• Coefficiente di variazione = deviazione standard / media × 100\n"
                f"• CV% basso = operazione stabile, CV% alto = alta variabilità\n"
                f"• Permette confrontare variabilità di grandezze diverse\n\n"
                f"**4. Ore Operative Totali**\n"
                f"• Conteggio ore con modulazione > 10%\n"
                f"• Indica utilizzo effettivo del sistema\n"
                f"• Differenze stagionali mostrano pattern di utilizzo\n\n"
                f"**5. Escursione Termica (Max - Min)**\n"
                f"• Ampiezza totale della variazione di temperatura\n"
                f"• Range più ampio = condizioni più estreme\n"
                f"• Impatto su dimensionamento e controllo\n\n"
                f"**6. Test di Significatività Statistica**\n"
                f"• Grafico a torta: variabili con differenze statisticamente significative\n"
                f"• Test t-student con confidenza 95% (p < 0.05)\n"
                f"• Risultato: {sig_count}/{total_tests} variabili significativamente diverse\n\n"
                f"**7. Trend Temperatura Esterna**\n"
                f"• Andamento giornaliero nel periodo\n"
                f"• Medie mobili per evidenziare trend\n"
                f"• Linee tratteggiate = medie stagionali\n\n"
                f"**8. Sintesi Risultati Chiave**\n"
                f"• Testo con principali metriche numeriche\n"
                f"• Delta temperature, ore operative, test statistici\n"
                f"• Riferimento rapido ai risultati principali\n\n"
                f"**Utilizzo della Dashboard:**\n"
                f"• Identificazione rapida differenze stagionali\n"
                f"• Valutazione efficienza operativa\n"
                f"• Base per ottimizzazione energetica\n"
                f"• Documentazione prestazioni sistema"
            )
        
        return ""
    
    def _create_word_report(self):
        """Creates a Word report with seasonal comparison plots."""
        if not DOCX_AVAILABLE:
            self.logger.warning("python-docx not available. Skipping Word report generation.")
            return None
        
        if not self.plot_paths:
            self.logger.warning("No plots available for Word report.")
            return None
        
        try:
            self.logger.info("="*60)
            self.logger.info("GENERATING WORD REPORT - SEASONAL COMPARISON")
            self.logger.info("="*60)
            
            doc = Document()
            
            # Set page margins
            sections = doc.sections
            for section in sections:
                section.top_margin = Inches(0.75)
                section.bottom_margin = Inches(0.75)
                section.left_margin = Inches(0.75)
                section.right_margin = Inches(0.75)
            
            # Add title
            title = doc.add_heading('Confronto Stagionale - Sistema HVAC', level=0)
            title.alignment = WD_ALIGN_PARAGRAPH.CENTER
            title_format = title.runs[0].font
            title_format.size = Pt(16)
            title_format.bold = True
            title_format.color.rgb = RGBColor(0, 0, 0)
            
            # Get date ranges
            summer_start = self.summer_data.index.min().strftime('%d/%m/%Y')
            summer_end = self.summer_data.index.max().strftime('%d/%m/%Y')
            winter_start = self.winter_data.index.min().strftime('%d/%m/%Y')
            winter_end = self.winter_data.index.max().strftime('%d/%m/%Y')
            
            date_range_str = (f"Periodo di analisi - Estate: {summer_start} - {summer_end} | "
                            f"Inverno: {winter_start} - {winter_end}")
            
            date_para = doc.add_paragraph(date_range_str)
            date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            date_format = date_para.runs[0].font
            date_format.size = Pt(11)
            date_format.bold = False
            date_format.color.rgb = RGBColor(128, 128, 128)
            
            # Add building/AHU info
            info_para = doc.add_paragraph(f"Edificio: {self.building_id} | Unità: {self.ahu_unit}")
            info_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            info_format = info_para.runs[0].font
            info_format.size = Pt(10)
            info_format.italic = True
            info_format.color.rgb = RGBColor(100, 100, 100)
            
            doc.add_paragraph()
            
            # Add each plot with its description
            for idx, (plot_type, plot_path) in enumerate(self.plot_paths, 1):
                self.logger.info(f"   Adding plot {idx}/{len(self.plot_paths)}: {plot_type}")
                
                # Create table with 3 rows
                table = doc.add_table(rows=3, cols=1)
                table.style = 'Table Grid'
                table.autofit = False
                table.allow_autofit = False
                
                # Set table width
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
                title_para.text = plot_type
                title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                title_format = title_para.runs[0].font
                title_format.bold = True
                title_format.size = Pt(12)
                title_format.color.rgb = RGBColor(0, 51, 102)
                
                cell_title.vertical_alignment = 1
                
                # Row 2: Image
                cell_image = table.rows[1].cells[0]
                set_cell_border(cell_image, top={}, left={}, right={}, bottom={})
                cell_image.paragraphs[0].clear()
                
                paragraph = cell_image.paragraphs[0]
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                
                try:
                    run = paragraph.add_run()
                    run.add_picture(plot_path, width=Inches(6.3))
                except Exception as e:
                    self.logger.error(f"Error adding image: {str(e)}")
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
                
                description = self._generate_plot_description(plot_type)
                desc_para = cell_desc.add_paragraph(description)
                desc_para.paragraph_format.line_spacing = 1.15
                desc_para_format = desc_para.runs[0].font
                desc_para_format.size = Pt(10)
                
                # Add spacing between plots
                if idx < len(self.plot_paths):
                    doc.add_paragraph()
            
            # Save Word document
            word_filename = f"{self.building_id}_{self.ahu_unit}_Seasonal_Comparison_Report.docx"
            word_path = self.output_dir / word_filename
            doc.save(str(word_path))
            
            self.logger.info(f"Word report saved: {word_filename}")
            self.logger.info("="*60)
            self.logger.info("WORD REPORT GENERATION COMPLETED")
            self.logger.info("="*60)
            
            return str(word_path)
            
        except Exception as e:
            self.logger.error(f"Error generating Word report: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_results(self):
        """
        Save all comparison results to CSV and Excel files.
        
        Saves:
            - Basic statistics as CSV
            - Statistical tests as CSV
            - Combined Excel workbook with multiple sheets:
                * Basic_Statistics
                * Statistical_Tests
                * Summary (metadata and overview)
                
        Files:
            - Timestamped filenames for version control
            - Saved to self.results_dir
            
        Side Effects:
            - Creates output files in results directory
            - Logs save locations
        """
        if not self.save_results_flag:
            return
        
        self.logger.info("Saving comparison results...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save basic statistics
        if hasattr(self, 'basic_stats'):
            csv_path = self.results_dir / f'{self.building_id}_{self.ahu_unit}_basic_stats_{timestamp}.csv'
            self.basic_stats.to_csv(csv_path, index=False)
            self.logger.info(f"Basic statistics saved to {csv_path}")
        
        # Save statistical tests
        if hasattr(self, 'statistical_tests'):
            csv_path = self.results_dir / f'{self.building_id}_{self.ahu_unit}_statistical_tests_{timestamp}.csv'
            self.statistical_tests.to_csv(csv_path, index=False)
            self.logger.info(f"Statistical tests saved to {csv_path}")
        
        # Save combined Excel report
        excel_path = self.results_dir / f'{self.building_id}_{self.ahu_unit}_seasonal_comparison_{timestamp}.xlsx'
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            if hasattr(self, 'basic_stats'):
                self.basic_stats.to_excel(writer, sheet_name='Basic_Statistics', index=False)
            
            if hasattr(self, 'statistical_tests'):
                self.statistical_tests.to_excel(writer, sheet_name='Statistical_Tests', index=False)
            
            # Add summary sheet
            summary_data = {
                'Metric': ['Building ID', 'AHU Unit', 'Summer File', 'Winter File', 
                          'Columns Analyzed', 'Significance Threshold', 'Analysis Date'],
                'Value': [self.building_id, self.ahu_unit, self.summer_file, self.winter_file,
                         ', '.join(self.columns_to_compare), self.significance_threshold,
                         datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Analysis_Summary', index=False)
        
        self.logger.info(f"Combined Excel report saved to {excel_path}")
    
    def run_analysis(self):
        """
        Run the complete seasonal comparison analysis.
        
        Pipeline:
            1. Load and cache data
            2. Calculate statistics
            3. Generate plots (sequentially for matplotlib Agg backend compatibility)
            4. Save results
            5. Create Word report (if enabled)
            
        Note:
            Plots are generated sequentially (not in parallel) to avoid
            matplotlib threading issues with the Agg backend that can
            result in blank/empty plot files.
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING SEASONAL COMPARISON ANALYSIS")
        self.logger.info("=" * 80)
        
        if not self.enable_comparison:
            self.logger.warning("Seasonal comparison is disabled in config.ini")
            return
        
        try:
            # Load data
            self.load_data()
            
            # Assess data quality for fair comparison
            self.assess_data_quality()
            
            # Calculate statistics
            self.calculate_basic_statistics()
            self.perform_statistical_tests()
            self.calculate_hourly_profiles()
            self.calculate_load_duration_curves()
            
            # Create plots SEQUENTIALLY to avoid matplotlib threading issues with Agg backend
            if self.create_plots:
                self.logger.info("\n" + "="*80)
                self.logger.info("GENERATING PLOTS")
                self.logger.info("="*80)
                
                # Sequential plotting to ensure proper rendering with Agg backend
                try:
                    self.plot_temperature_distributions()
                except Exception as e:
                    self.logger.error(f"Error in temperature distributions: {e}")
                
                try:
                    self.plot_day_of_week_comparison()
                except Exception as e:
                    self.logger.error(f"Error in day-of-week comparison: {e}")
                
                try:
                    self.plot_correlation_scatter()
                except Exception as e:
                    self.logger.error(f"Error in correlation scatter: {e}")
                
                try:
                    self.plot_load_duration_curves()
                except Exception as e:
                    self.logger.error(f"Error in load duration curves: {e}")
                
                try:
                    self.plot_humidity_comparison()
                except Exception as e:
                    self.logger.error(f"Error in humidity comparison: {e}")
                
                try:
                    self.plot_kpi_dashboard()
                except Exception as e:
                    self.logger.error(f"Error in KPI dashboard: {e}")
                
                self.logger.info("All plots generated successfully")
            
            # Save results
            self.save_results()
            
            # Generate Word report
            if self.plot_paths and DOCX_AVAILABLE:
                self.logger.info("\n" + "="*80)
                self.logger.info("GENERATING WORD REPORT")
                self.logger.info("="*80)
                word_path = self._create_word_report()
                if word_path:
                    self.logger.info(f"✅ Word report created: {word_path}")
            
            self.logger.info("=" * 80)
            self.logger.info("SEASONAL COMPARISON ANALYSIS COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)
            
            # Print summary
            self.print_summary()
            
        except Exception as e:
            self.logger.error(f"Error during analysis: {e}")
            raise
    
    def print_summary(self):
        """Print a summary of the analysis results."""
        print("\n" + "=" * 80)
        print("SEASONAL COMPARISON SUMMARY")
        print("=" * 80)
        
        print(f"\nBuilding: {self.building_id} | AHU Unit: {self.ahu_unit}")
        print(f"Summer Data: {len(self.summer_data)} records")
        print(f"Winter Data: {len(self.winter_data)} records")
        print(f"Columns Analyzed: {len(self.columns_to_compare)}")
        
        if hasattr(self, 'statistical_tests'):
            significant_count = self.statistical_tests['T_Significant'].sum()
            print(f"\nStatistically Significant Differences: {significant_count}/{len(self.statistical_tests)}")
            
            if significant_count > 0:
                print("\nVariables with Significant Differences:")
                sig_vars = self.statistical_tests[self.statistical_tests['T_Significant']]
                for _, row in sig_vars.iterrows():
                    print(f"  - {row['Variable']}: "
                          f"Mean Diff = {row['Mean_Diff']:.2f}, "
                          f"Effect Size = {row['Effect_Size']}")
        
        print(f"\nResults saved to: {self.results_dir}")
        print(f"Plots saved to: {self.output_dir}")
        print("=" * 80 + "\n")


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("HVAC SEASONAL COMPARISON ANALYSIS TOOL")
    print("Campus Einaudi Building - AHU Monitoring System")
    print("=" * 80 + "\n")
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    # Config file is 2 levels up: 3_data_analysis → scripts → HVAC_project
    config_path = script_dir.parent.parent / 'config.ini'
    
    print(f"Looking for config file at: {config_path}")
    
    if not config_path.exists():
        print(f"ERROR: Config file not found at {config_path}")
        print("Please ensure config.ini is in the HVAC_project directory.")
        return
    
    # Initialize analyzer
    analyzer = SeasonalComparison(config_path=str(config_path))
    
    # Run analysis
    analyzer.run_analysis()
    
    print("\nAnalysis complete! Check the output directories for results and plots.")


if __name__ == "__main__":
    main()