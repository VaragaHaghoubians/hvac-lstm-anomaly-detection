"""
HVAC Correlation Matrix Analysis Script (MAIN RESULT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 PURPOSE: Analyze how variables correlate AT THE SAME TIME
   "When variable A is high now, is variable B also high now?"

🎯 ANSWERS YOUR MANAGER'S QUESTION:
   "Create features and find correlation between them (Pearson/Spearman)"

� STATISTICAL RIGOR (THESIS-GRADE):
   ✓ Temporal aggregation to ensure independent samples
   ✓ Autocorrelation in raw time-series violates correlation test assumptions
   ✓ Default: Aggregate to hourly means before computing correlations
   ✓ Makes p-values valid and results academically defensible

📈 OUTPUTS:
   ✓ Correlation heatmaps (Pearson + Spearman)
   ✓ Lists of strongest positive/negative correlations
   ✓ Multicollinearity detection (redundant features)
   ✓ Word report with findings

💡 USE THIS FOR:
   - Feature selection (which variables to keep/remove)
   - Understanding physical relationships
   - Explaining results to stakeholders

🔧 CONTROL: Set enable_correlation_matrix = true in config.ini [correlation_matrix]
   Set apply_temporal_aggregation = true (recommended for thesis)
   Set aggregation_frequency = 1H or D (hourly or daily)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import configparser
from datetime import datetime
import warnings
import sys

try:
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
except ImportError:
    Document = None

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')


class CorrelationMatrixAnalyzer:
    """Analyzes correlation matrices for HVAC data"""
    
    def __init__(self, config_path='config.ini'):
        """
        Resolves the config path from multiple candidate locations so the
        class can be run from any working directory without hardcoding paths.

        Args:
            config_path (str): Relative or absolute path to config.ini.
        """
        # Allow inline comments in config values
        self.config = configparser.ConfigParser(inline_comment_prefixes=(';', '#'))
        self.plot_records = []
        self.analysis_start = None
        self.analysis_end = None
        self.analysis_period_text = ""
        
        # Handle path resolution
        config_file = Path(config_path)
        
        if not config_file.is_absolute():
            if config_file.exists():
                config_file = config_file.resolve()
            else:
                script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
                parent_config = script_dir.parent / config_path
                
                if parent_config.exists():
                    config_file = parent_config
                elif not config_file.exists():
                    project_root = Path.cwd()
                    while project_root != project_root.parent:
                        test_config = project_root / config_path
                        if test_config.exists():
                            config_file = test_config
                            break
                        project_root = project_root.parent
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        print(f"Reading config from: {config_file}")
        self.config.read(config_file, encoding='utf-8-sig')
        self.config_dir = config_file.parent
        
        self.load_settings()
        
    def load_settings(self):
        """
        Centralises all config reading into self.* attributes so the rest
        of the class can access settings without repeated config.get() calls.
        """
        # Global settings
        self.building_id = self.config.get('global', 'building_id')
        self.ahu_unit = self.config.get('global', 'ahu_unit')
        self.season = self.config.get('global', 'season')
        self.year = self.config.get('global', 'year')
        
        # Timezone
        self.timezone = self.config.get('global', 'timezone', fallback='Europe/Rome')
        
        # Correlation matrix settings
        corr_section = 'correlation_matrix'
        self.enable_corr = self.config.getboolean(corr_section, 'enable_correlation_matrix', fallback=True)
        
        # Correlation types
        self.calc_pearson = self.config.getboolean(corr_section, 'calculate_pearson', fallback=True)
        self.calc_spearman = self.config.getboolean(corr_section, 'calculate_spearman', fallback=True)
        self.calc_kendall = self.config.getboolean(corr_section, 'calculate_kendall', fallback=False)
        
        # Visualization settings
        self.create_heatmaps = self.config.getboolean(corr_section, 'create_correlation_heatmaps', fallback=True)
        self.show_upper_triangle = self.config.getboolean(corr_section, 'show_upper_triangle_only', fallback=True)
        self.highlight_threshold = self.config.getfloat(corr_section, 'highlight_threshold', fallback=0.7)
        self.highlight_color = self.config.get(corr_section, 'highlight_color', fallback='lime')
        self.colormap = self.config.get(corr_section, 'heatmap_colormap', fallback='coolwarm')
        self.annotate = self.config.getboolean(corr_section, 'annotate_values', fallback=True)
        self.annotation_fontsize = self.config.getint(corr_section, 'annotation_fontsize', fallback=9)
        
        # Analysis settings
        self.identify_strong = self.config.getboolean(corr_section, 'identify_strong_correlations', fallback=True)
        self.strong_threshold = self.config.getfloat(corr_section, 'strong_correlation_threshold', fallback=0.7)
        self.identify_weak = self.config.getboolean(corr_section, 'identify_weak_correlations', fallback=True)
        self.weak_threshold = self.config.getfloat(corr_section, 'weak_correlation_threshold', fallback=0.3)
        self.detect_multicollinearity = self.config.getboolean(corr_section, 'detect_multicollinearity', fallback=True)
        self.multicollinearity_threshold = self.config.getfloat(corr_section, 'multicollinearity_threshold', fallback=0.9)
        
        # Report settings
        self.generate_report = self.config.getboolean(corr_section, 'generate_correlation_report', fallback=True)
        
        # Temporal aggregation settings (for statistical independence)
        # Time-series data has autocorrelation, which violates independence assumption
        # Aggregating to hourly/daily before computing correlations makes p-values valid
        self.apply_temporal_aggregation = self.config.getboolean(corr_section, 'apply_temporal_aggregation', fallback=True)
        self.aggregation_frequency = self.config.get(corr_section, 'aggregation_frequency', fallback='1H')  # '1H' or 'D'
        self.aggregation_method = self.config.get(corr_section, 'aggregation_method', fallback='mean')  # 'mean', 'median'
        self.filter_steady_state = self.config.getboolean(corr_section, 'filter_steady_state', fallback=False)
        self.steady_state_column = self.config.get(corr_section, 'steady_state_column', fallback='is_steady_state')
        self.top_n = self.config.getint(corr_section, 'top_n_correlations', fallback=10)
        self.report_positive = self.config.getboolean(corr_section, 'report_positive_correlations', fallback=True)
        self.report_negative = self.config.getboolean(corr_section, 'report_negative_correlations', fallback=True)
        
        # Output settings
        self.save_csv = self.config.getboolean(corr_section, 'save_correlation_matrices_csv', fallback=True)
        self.save_top_csv = self.config.getboolean(corr_section, 'save_top_correlations_csv', fallback=True)
        self.create_word = self.config.getboolean(corr_section, 'create_word_report', fallback=True)
        
        # Figure settings
        self.fig_width = self.config.getfloat(corr_section, 'heatmap_figure_width', fallback=18)
        self.fig_height = self.config.getfloat(corr_section, 'heatmap_figure_height', fallback=16)
        self.font_scale = self.config.getfloat(corr_section, 'heatmap_font_scale', fallback=1.0)
        
        # Create base name
        self.base_name = f"{self.building_id}_{self.ahu_unit}_{self.season}{self.year}"
        
        # Paths
        processed_folder = self._resolve_path(self.config.get('paths', 'processed_folder'))
        plots_folder = self._resolve_path(self.config.get('paths', 'plots_folder', fallback='plots'))
        
        # Output directories - separate folder for correlation matrix analysis
        self.output_dir = processed_folder / 'correlation_matrix_analysis' / self.base_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.plots_dir = plots_folder / 'correlation_matrix_analysis' / self.base_name
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        self.plot_format = self.config.get(corr_section, 'correlation_plot_format', fallback='png')
        self.plot_dpi = self.config.getint(corr_section, 'correlation_plot_dpi', fallback=300)
        
        # === NEW PRODUCTION ENHANCEMENTS ===
        
        # Column selection from config
        cols_str = self.config.get(corr_section, 'correlation_columns', fallback='')
        self.selected_columns = [c.strip() for c in cols_str.split(',') if c.strip()] if cols_str.strip() else []
        
        # Data quality thresholds
        self.min_coverage = self.config.getfloat(corr_section, 'min_data_coverage', fallback=0.7)
        self.drop_zero_variance = self.config.getboolean(corr_section, 'drop_zero_variance', fallback=True)
        
        # Kendall speed control
        self.max_rows_for_kendall = self.config.getint(corr_section, 'max_rows_for_kendall', fallback=5000)
        
        # Statistical significance
        self.compute_pvalues = self.config.getboolean(corr_section, 'compute_pvalues', fallback=False)
        self.fdr_correction = self.config.getboolean(corr_section, 'fdr_correction', fallback=False)
        self.significance_level = self.config.getfloat(corr_section, 'significance_level', fallback=0.05)
        
        # Set seaborn font scale
        sns.set(font_scale=self.font_scale)
        
        # File paths - Find full features data file
        features_filename = f"{self.base_name}_features.csv"
        feature_candidates = [
            processed_folder / 'feature_engineering' / self.base_name / features_filename,
            processed_folder / 'interpolation' / self.base_name / features_filename,
            processed_folder / 'interpolation' / features_filename,
        ]
        
        # Fallback to interpolated
        interp_filename = f"{self.base_name}_interpolated.csv"
        interp_candidates = [
            processed_folder / 'interpolation' / self.base_name / interp_filename,
            processed_folder / 'interpolation' / interp_filename,
        ]
        
        all_candidates = feature_candidates + interp_candidates
        
        self.data_file = all_candidates[0]
        for candidate in all_candidates:
            if candidate.exists():
                self.data_file = candidate
                break
        
    def _resolve_path(self, path_str):
        """
        Converts relative paths to absolute using the config file's location
        as the base, so paths in config.ini work regardless of the caller's
        working directory.

        Args:
            path_str (str): Relative or absolute path string from config.

        Returns:
            Path: Absolute path resolved relative to the config directory.
        """
        path = Path(path_str)
        if not path.is_absolute():
            path = self.config_dir / path
        return path
    
    def _format_date(self, value):
        """
        Normalises pandas Timestamps and Python date objects to a single
        display format for the report, since their APIs differ.

        Args:
            value: A pd.Timestamp, datetime.date, or None.

        Returns:
            str: Date formatted as dd/mm/yyyy, or "N/D" if value is None.
        """
        if value is None:
            return "N/D"
        if isinstance(value, pd.Timestamp):
            value = value.to_pydatetime()
        if hasattr(value, 'strftime'):
            return value.strftime('%d/%m/%Y')
        return str(value)
    
    def _update_analysis_period(self, df):
        """
        Keeps the analysis period metadata in sync after the dataset is loaded
        or filtered, so all report sections show the correct date range.

        Args:
            df (pd.DataFrame): The current working dataset.
        """
        if df is None or df.empty:
            self.analysis_period_text = "Periodo di analisi: Dati non disponibili"
            return
        
        self.analysis_start = df.index.min()
        self.analysis_end = df.index.max()
        self.analysis_period_text = f"Periodo di analisi: {self._format_date(self.analysis_start)} - {self._format_date(self.analysis_end)}"
    
    def load_data(self):
        """
        Loads the best available dataset (feature-engineered file preferred,
        falls back to interpolated) and applies column selection, timezone
        normalisation, and coverage filtering before analysis.

        Returns:
            pd.DataFrame: Cleaned dataset ready for correlation analysis.
        """
        print(f"Loading data from: {self.data_file}")
        
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        df = pd.read_csv(self.data_file)
        
        # Parse time column
        time_col = self.config.get('data', 'time_column', fallback='Time')
        if time_col not in df.columns:
            raise ValueError(f"Time column '{time_col}' not found")
        
        # Parse datetime with UTC handling for timezone-aware strings
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        df.set_index(time_col, inplace=True)
        
        print(f"Loaded {len(df)} records from {df.index.min()} to {df.index.max()}")
        
        # Timezone handling (Critical for consistency with time-derived features)
        if df.index.tz is None:
            df.index = df.index.tz_localize(self.timezone, ambiguous='NaT', nonexistent='shift_forward')
            df = df[~df.index.isna()]
            print(f"  ✓ Localized to {self.timezone}")
        else:
            df.index = df.index.tz_convert(self.timezone)
            print(f"  ✓ Converted to {self.timezone}")
        
        self._update_analysis_period(df)
        return df
    
    def _compute_pvalue_matrix(self, df, method='pearson'):
        """
        Compute p-value matrix for correlations.
        
        Args:
            df: DataFrame with numeric data
            method: 'pearson' or 'spearman'
            
        Returns:
            DataFrame with p-values
        """
        n = len(df.columns)
        pvalues = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    pvalues[i, j] = 0.0  # Diagonal is always significant
                else:
                    col_i = df.iloc[:, i].dropna()
                    col_j = df.iloc[:, j].dropna()
                    
                    # Find common non-null indices
                    common_idx = col_i.index.intersection(col_j.index)
                    if len(common_idx) < 3:
                        pvalues[i, j] = pvalues[j, i] = 1.0
                        continue
                    
                    x = col_i.loc[common_idx]
                    y = col_j.loc[common_idx]
                    
                    try:
                        if method == 'pearson':
                            _, p = pearsonr(x, y)
                        elif method == 'spearman':
                            _, p = spearmanr(x, y)
                        else:
                            p = 1.0
                        
                        pvalues[i, j] = pvalues[j, i] = p
                    except:
                        pvalues[i, j] = pvalues[j, i] = 1.0
        
        return pd.DataFrame(pvalues, index=df.columns, columns=df.columns)
    
    def _apply_fdr_correction(self, pvalue_matrix):
        """
        Apply Benjamini-Hochberg FDR correction to p-values.
        
        Args:
            pvalue_matrix: DataFrame with p-values
            
        Returns:
            DataFrame with corrected p-values
        """
        # Extract upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(pvalue_matrix, dtype=bool), k=1)
        pvals_upper = pvalue_matrix.where(mask).values.flatten()
        pvals_upper = pvals_upper[~np.isnan(pvals_upper)]
        
        # Apply FDR correction
        from scipy.stats import false_discovery_control
        try:
            corrected = false_discovery_control(pvals_upper, method='bh')
        except:
            # Fallback to manual BH correction
            n = len(pvals_upper)
            sorted_idx = np.argsort(pvals_upper)
            sorted_p = pvals_upper[sorted_idx]
            
            # BH correction
            corrected_sorted = sorted_p * n / (np.arange(n) + 1)
            corrected_sorted = np.minimum.accumulate(corrected_sorted[::-1])[::-1]
            
            corrected = np.empty(n)
            corrected[sorted_idx] = corrected_sorted
        
        # Reconstruct matrix
        corrected_matrix = pvalue_matrix.copy()
        idx = 0
        for i in range(len(pvalue_matrix)):
            for j in range(i + 1, len(pvalue_matrix)):
                corrected_matrix.iloc[i, j] = corrected[idx]
                corrected_matrix.iloc[j, i] = corrected[idx]
                idx += 1
        
        return corrected_matrix
    
    def _apply_temporal_aggregation(self, df):
        """
        Apply temporal aggregation to address autocorrelation in time-series data.
        
        STATISTICAL RATIONALE:
        - Raw time-series data violates independence assumption of correlation tests
        - Consecutive measurements are autocorrelated (e.g., temp at 10:00 predicts temp at 10:30)
        - This makes p-values artificially significant ("too significant")
        - Aggregating to hourly/daily creates approximately independent samples
        - Makes correlation results statistically defensible for thesis/publication
        
        Args:
            df: DataFrame with DatetimeIndex
            
        Returns:
            Aggregated DataFrame with independent temporal samples
        """
        if not self.apply_temporal_aggregation:
            return df
        
        print(f"\n  Applying temporal aggregation for statistical independence...")
        print(f"    Rationale: Time-series autocorrelation violates independence assumption")
        print(f"    Frequency: {self.aggregation_frequency}")
        print(f"    Method: {self.aggregation_method}")
        
        original_rows = len(df)
        
        # Filter to steady-state if requested and column exists
        if self.filter_steady_state and self.steady_state_column in df.columns:
            df = df[df[self.steady_state_column] == 1].copy()
            print(f"    Filtered to steady-state periods: {len(df)} / {original_rows} records")
        
        # Select numeric columns for aggregation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Apply aggregation based on method
        if self.aggregation_method == 'median':
            df_agg = df[numeric_cols].resample(self.aggregation_frequency).median()
        else:  # default to mean
            df_agg = df[numeric_cols].resample(self.aggregation_frequency).mean()
        
        # Drop rows with all NaN (empty time bins)
        df_agg = df_agg.dropna(how='all')
        
        aggregated_rows = len(df_agg)
        print(f"    ✓ Aggregated: {original_rows} raw samples → {aggregated_rows} independent samples")
        print(f"    ✓ Effective sample size ratio: {aggregated_rows/original_rows:.1%}")
        
        if aggregated_rows < 30:
            print(f"    ⚠️  Warning: Only {aggregated_rows} independent samples. Consider longer period or coarser aggregation.")
        
        return df_agg
    
    def compute_correlation_matrices(self, df):
        """
        Compute Pearson, Spearman, and Kendall correlations with data hygiene.

        Applies temporal aggregation before computing to satisfy the statistical
        independence assumption — raw 30-min time-series data is autocorrelated,
        which invalidates p-values and makes results indefensible in the thesis.

        Args:
            df (pd.DataFrame): Dataset returned by load_data().

        Returns:
            dict: Method name → correlation DataFrame (e.g. {'pearson': df, ...}).
        """
        print("\nComputing correlation matrices...")
        
        # === TEMPORAL AGGREGATION (for statistical independence) ===
        df = self._apply_temporal_aggregation(df)
        
        # Select only numeric columns
        num_df = df.select_dtypes(include=[np.number]).copy()
        print(f"  Starting with {len(num_df.columns)} numeric variables")
        
        # === DATA HYGIENE ===
        
        # 1. Filter to selected columns from config (if specified)
        if self.selected_columns:
            available = [c for c in self.selected_columns if c in num_df.columns]
            missing = [c for c in self.selected_columns if c not in num_df.columns]
            if available:
                num_df = num_df[available]
                print(f"  ✓ Filtered to {len(available)} selected columns from config")
                if missing:
                    print(f"  ℹ Skipped {len(missing)} missing columns: {missing[:3]}{'...' if len(missing) > 3 else ''}")
            else:
                print(f"  ⚠ Warning: None of the selected columns found, using all numeric columns")
        
        # 2. Drop columns with insufficient coverage
        min_nonnull = int(self.min_coverage * len(num_df))
        coverage = num_df.notna().sum()
        low_coverage = coverage[coverage < min_nonnull]
        if len(low_coverage) > 0:
            num_df = num_df.loc[:, coverage >= min_nonnull]
            print(f"  ✓ Dropped {len(low_coverage)} columns with < {self.min_coverage*100:.0f}% coverage")
        
        # 3. Drop zero-variance columns (constants)
        if self.drop_zero_variance:
            std = num_df.std(numeric_only=True)
            zero_var = std[std == 0].index.tolist()
            if len(zero_var) > 0:
                num_df = num_df.drop(columns=zero_var)
                print(f"  ✓ Dropped {len(zero_var)} zero-variance columns: {zero_var}")
        
        print(f"  → Final dataset: {len(num_df.columns)} variables × {len(num_df)} records")
        
        # Store cleaned column list
        self.final_columns = num_df.columns.tolist()
        
        results = {}
        pvalue_results = {}
        
        # === COMPUTE CORRELATIONS ===
        
        if self.calc_pearson:
            print("  Computing Pearson correlation (linear relationships)...")
            results['pearson'] = num_df.corr(method='pearson')
            
            if self.compute_pvalues:
                print("    → Computing p-values...")
                pvalue_results['pearson'] = self._compute_pvalue_matrix(num_df, method='pearson')
                
                if self.fdr_correction:
                    print("    → Applying FDR correction...")
                    pvalue_results['pearson'] = self._apply_fdr_correction(pvalue_results['pearson'])
        
        if self.calc_spearman:
            print("  Computing Spearman correlation (monotonic relationships)...")
            results['spearman'] = num_df.corr(method='spearman')
            
            if self.compute_pvalues:
                print("    → Computing p-values...")
                pvalue_results['spearman'] = self._compute_pvalue_matrix(num_df, method='spearman')
                
                if self.fdr_correction:
                    print("    → Applying FDR correction...")
                    pvalue_results['spearman'] = self._apply_fdr_correction(pvalue_results['spearman'])
        
        if self.calc_kendall:
            # Check dataset size for Kendall (O(n²) complexity)
            if len(num_df) > self.max_rows_for_kendall:
                print(f"  ⚠ Skipping Kendall: Dataset has {len(num_df)} rows (max: {self.max_rows_for_kendall})")
                print(f"    → Increase max_rows_for_kendall in config or sample data to enable")
            else:
                print("  Computing Kendall Tau correlation (robust to outliers)...")
                results['kendall'] = num_df.corr(method='kendall')
        
        # Store p-values for later use
        self.pvalue_matrices = pvalue_results if pvalue_results else None
        
        return results
    
    def plot_correlation_heatmap(self, corr_matrix, method_name):
        """
        Visualises the correlation structure as a colour-coded heatmap with the
        upper triangle masked to avoid redundancy, and highlights cells above
        the strong correlation threshold for quick visual identification.

        Args:
            corr_matrix (pd.DataFrame): Square correlation matrix.
            method_name (str): Label for the plot title (e.g. "pearson").
        """
        # Calculate dynamic figure size based on number of variables
        n_vars = len(corr_matrix)
        
        # Adaptive sizing: minimum 0.5 inches per variable, with reasonable bounds
        dynamic_width = max(self.fig_width, n_vars * 0.5)
        dynamic_height = max(self.fig_height, n_vars * 0.5)
        
        # Cap at reasonable maximum to avoid excessive memory usage
        dynamic_width = min(dynamic_width, 30)
        dynamic_height = min(dynamic_height, 30)
        
        fig, ax = plt.subplots(figsize=(dynamic_width, dynamic_height))
        
        # Create mask for upper triangle (if enabled)
        mask = None
        if self.show_upper_triangle:
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        # Adaptive annotation font size based on number of variables
        if n_vars > 50:
            annot_size = max(4, self.annotation_fontsize - 4)
            show_annot = False  # Disable annotations for very large matrices
            print(f"    → {n_vars} variables detected: annotations disabled for readability")
        elif n_vars > 30:
            annot_size = max(5, self.annotation_fontsize - 2)
            show_annot = self.annotate
            if show_annot:
                print(f"    → {n_vars} variables detected: using reduced annotation size ({annot_size}pt)")
        else:
            annot_size = self.annotation_fontsize
            show_annot = self.annotate
        
        # Adaptive tick label size
        if n_vars > 40:
            tick_fontsize = 7
        elif n_vars > 25:
            tick_fontsize = 9
        else:
            tick_fontsize = 11
        
        # Create heatmap
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=show_annot,
            fmt='.2f',
            cmap=self.colormap,
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5 if n_vars < 50 else 0.1,
            cbar_kws={'label': 'Coefficiente di Correlazione', 'shrink': 0.8},
            annot_kws={'fontsize': annot_size},
            ax=ax
        )
        
        # Set tick label sizes for better readability
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=tick_fontsize, rotation=90, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=tick_fontsize, rotation=0)
        
        # Highlight strong correlations (if enabled)
        if self.identify_strong and self.strong_threshold > 0:
            for i in range(len(corr_matrix)):
                for j in range(i):
                    if abs(corr_matrix.iloc[i, j]) >= self.strong_threshold:
                        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, 
                                                   edgecolor=self.highlight_color, lw=3))
        
        method_titles = {
            'pearson': 'Correlazione di Pearson (Relazioni Lineari)',
            'spearman': 'Correlazione di Spearman (Relazioni Monotoniche)',
            'kendall': 'Correlazione di Kendall Tau (Robusta agli Outlier)'
        }
        
        # Create title with date range
        main_title = method_titles.get(method_name, method_name)
        title_with_date = f"{main_title}\n{self.analysis_period_text}"
        
        # Adaptive title font size
        title_fontsize = 16 if n_vars < 30 else 14
        ax.set_title(title_with_date, 
                    fontsize=title_fontsize, fontweight='bold', color='#003366', pad=20)
        
        # Use tight_layout with padding to prevent label cutoff
        plt.tight_layout(pad=2.0)
        
        # Save plot with bbox_inches='tight' to ensure all labels are included
        filename = f"correlation_heatmap_{method_name}_{self.base_name}.{self.plot_format}"
        filepath = self.plots_dir / filename
        plt.savefig(filepath, dpi=self.plot_dpi, bbox_inches='tight', pad_inches=0.3)
        print(f"  Saved {method_name} heatmap: {filepath}")
        
        # Add to plot records
        descriptions = {
            'pearson': f'Matrice di correlazione di Pearson che mostra le relazioni lineari tra tutte le variabili. Valori vicini a +1 indicano forte correlazione positiva, valori vicini a -1 indicano forte correlazione negativa. Le celle evidenziate in {self.highlight_color} mostrano correlazioni con |r| ≥ {self.strong_threshold}.',
            'spearman': 'Matrice di correlazione di Spearman che cattura relazioni monotoniche non lineari. Più robusta agli outlier rispetto a Pearson. Utile per identificare relazioni dove una variabile aumenta (o diminuisce) costantemente con l\'altra, anche se non linearmente.',
            'kendall': 'Matrice di correlazione di Kendall Tau, particolarmente robusta agli outlier e adatta per dataset più piccoli. Misura la concordanza ordinale tra variabili.'
        }
        
        self.plot_records.append({
            'title': method_titles.get(method_name, method_name),
            'path': filepath,
            'description': descriptions.get(method_name, f'Matrice di correlazione {method_name}.')
        })
        
        plt.close()
    
    def _is_tautological_pair(self, var1, var2):
        """
        Detect tautological correlations (derived features that are mathematically related).
        
        Args:
            var1, var2: Variable names
            
        Returns:
            bool: True if pair is tautological
        """
        # Normalize names for comparison
        v1_lower = var1.lower()
        v2_lower = var2.lower()
        
        # Define tautology patterns
        tautology_patterns = [
            # Delta-T variants (all derived from same temperatures)
            ('delta_t', 'delta_t_signed'),
            ('delta_t', 'delta_t_heating'),
            ('delta_t', 'delta_t_cooling'),
            ('delta_t_signed', 'delta_t_heating'),
            ('delta_t_signed', 'delta_t_cooling'),
            
            # Thermal load decompositions (cooling + heating ≈ thermal_load)
            ('thermal_load', 'cooling_load'),
            ('thermal_load', 'heating_load'),
            ('thermal_load', 'thermal_power'),
            
            # Thermal power variants
            ('thermal_power', 'thermal_power_signed'),
            
            # Temperature lag features (lag_1, lag_2, etc. of same variable)
            # This catches "Temperatura Esterna_lag_1" vs "Temperatura Esterna_lag_2"
        ]
        
        # Check bidirectional patterns
        for pattern1, pattern2 in tautology_patterns:
            if (pattern1 in v1_lower and pattern2 in v2_lower) or \
               (pattern2 in v1_lower and pattern1 in v2_lower):
                return True
        
        # Check lag features of same base variable
        # E.g., "Temperatura Esterna" vs "Temperatura Esterna_lag_1"
        base1 = v1_lower.replace('_lag_', '_').split('_lag')[0]
        base2 = v2_lower.replace('_lag_', '_').split('_lag')[0]
        
        if base1 == base2 and ('lag' in v1_lower or 'lag' in v2_lower):
            return True
        
        # Check rolling averages of same variable
        # E.g., "Temperatura Esterna" vs "Temperatura Esterna_rolling_mean_24"
        if 'rolling' in v1_lower or 'rolling' in v2_lower:
            base1_no_roll = v1_lower.split('_rolling')[0]
            base2_no_roll = v2_lower.split('_rolling')[0]
            if base1_no_roll == base2_no_roll:
                return True
        
        # Check rate of change vs base variable
        if 'rate_of_change' in v1_lower or 'rate_of_change' in v2_lower:
            base1_no_roc = v1_lower.replace('_rate_of_change', '')
            base2_no_roc = v2_lower.replace('_rate_of_change', '')
            if base1_no_roc == base2_no_roc:
                return True
        
        return False
    
    def find_top_correlations(self, corr_matrix, method_name):
        """
        Extracts the strongest variable pairs from the full correlation matrix
        for the report, filtering out tautological pairs that would inflate
        results with mathematically trivial relationships.

        Args:
            corr_matrix (pd.DataFrame): Square correlation matrix.
            method_name (str): Correlation method label used for significance lookup.

        Returns:
            tuple: (top_positive DataFrame, top_negative DataFrame, all_pairs DataFrame).
        """
        # Get upper triangle (exclude diagonal)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        upper_tri = corr_matrix.where(mask)
        
        # Flatten and sort
        corr_pairs = []
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                if not np.isnan(upper_tri.iloc[i, j]):
                    corr_pairs.append({
                        'Variable_1': corr_matrix.index[i],
                        'Variable_2': corr_matrix.columns[j],
                        'Correlation': upper_tri.iloc[i, j],
                        'Abs_Correlation': abs(upper_tri.iloc[i, j])
                    })
        
        df_corr = pd.DataFrame(corr_pairs)
        
        # Filter out tautological pairs
        df_corr['is_tautology'] = df_corr.apply(
            lambda row: self._is_tautological_pair(row['Variable_1'], row['Variable_2']), 
            axis=1
        )
        
        n_tautologies = df_corr['is_tautology'].sum()
        if n_tautologies > 0:
            print(f"    → Filtered {n_tautologies} tautological pairs")
        
        # Filter by statistical significance (if p-values computed)
        if self.compute_pvalues and self.pvalue_matrices and method_name in self.pvalue_matrices:
            pval_matrix = self.pvalue_matrices[method_name]
            
            df_corr['p_value'] = df_corr.apply(
                lambda row: pval_matrix.loc[row['Variable_1'], row['Variable_2']],
                axis=1
            )
            
            n_before_sig = len(df_corr[~df_corr['is_tautology']])
            df_corr['is_significant'] = df_corr['p_value'] <= self.significance_level
            n_significant = df_corr['is_significant'].sum()
            
            print(f"    → {n_significant}/{n_before_sig} non-tautological pairs are significant (α={self.significance_level})")
            
            # Apply filters
            df_corr_filtered = df_corr[~df_corr['is_tautology'] & df_corr['is_significant']].copy()
        else:
            # No p-values, just filter tautologies
            df_corr_filtered = df_corr[~df_corr['is_tautology']].copy()
        
        df_corr_filtered = df_corr_filtered.sort_values('Abs_Correlation', ascending=False)
        
        # Get top positive and negative from filtered data
        top_positive = df_corr_filtered.nlargest(self.top_n, 'Correlation')
        top_negative = df_corr_filtered.nsmallest(self.top_n, 'Correlation')
        
        return top_positive, top_negative, df_corr
    
    def generate_correlation_report(self, results):
        """
        Prints a structured console summary of findings so results can be
        reviewed and discussed without opening the Word document.

        Args:
            results (dict): Output from compute_correlation_matrices().
        """
        print("\n" + "="*70)
        print("CORRELATION ANALYSIS FINDINGS")
        print("="*70)
        
        all_findings = {}
        
        for method_name, corr_matrix in results.items():
            print(f"\n{method_name.upper()} CORRELATION:")
            
            top_pos, top_neg, all_corr = self.find_top_correlations(corr_matrix, method_name)
            
            print(f"\n  Top {self.top_n} Positive Correlations:")
            for idx, row in top_pos.head(self.top_n).iterrows():
                print(f"    {row['Variable_1']:30s} ↔ {row['Variable_2']:30s}  r = {row['Correlation']:6.3f}")
            
            print(f"\n  Top {self.top_n} Negative Correlations:")
            for idx, row in top_neg.head(self.top_n).iterrows():
                print(f"    {row['Variable_1']:30s} ↔ {row['Variable_2']:30s}  r = {row['Correlation']:6.3f}")
            
            # Identify strong correlations (if enabled)
            if self.identify_strong:
                strong_corr = all_corr[all_corr['Abs_Correlation'] >= self.strong_threshold]
                print(f"\n  Strong correlations (|r| ≥ {self.strong_threshold}): {len(strong_corr)}")
                
                # Check for multicollinearity
                if self.detect_multicollinearity:
                    multicoll = all_corr[all_corr['Abs_Correlation'] >= self.multicollinearity_threshold]
                    if len(multicoll) > 0:
                        print(f"  ⚠️  Multicollinearity detected (|r| ≥ {self.multicollinearity_threshold}): {len(multicoll)} pairs")
            
            # Save to CSV (if enabled)
            if self.save_csv:
                csv_file = self.output_dir / f"correlation_{method_name}_{self.base_name}.csv"
                corr_matrix.to_csv(csv_file)
                print(f"  Saved correlation matrix: {csv_file}")
            
            # Save top correlations (if enabled)
            if self.save_top_csv:
                top_file = self.output_dir / f"top_correlations_{method_name}_{self.base_name}.csv"
                all_corr.to_csv(top_file, index=False)
                print(f"  Saved top correlations: {top_file}")
            
            all_findings[method_name] = {
                'top_positive': top_pos,
                'top_negative': top_neg,
                'strong_correlations': strong_corr,
                'all_correlations': all_corr
            }
        
        return all_findings
    
    def identify_weak_correlations(self, corr_matrix):
        """
        Flags variables that are poorly correlated with everything else in the
        dataset, which may indicate uninformative or redundant features worth
        removing before modelling.

        Args:
            corr_matrix (pd.DataFrame): Square correlation matrix.

        Returns:
            pd.Series or None: Mean absolute correlation per weak variable,
            sorted ascending. None if weak correlation identification is disabled.
        """
        if not self.identify_weak:
            return None
        
        # Calculate mean absolute correlation for each variable (EXCLUDING diagonal)
        # Diagonal is always 1.0 and inflates the mean
        mask = ~np.eye(len(corr_matrix), dtype=bool)
        mean_abs_corr = (corr_matrix.abs().where(mask)).sum() / mask.sum(axis=0)
        weak_vars = mean_abs_corr[mean_abs_corr < self.weak_threshold].sort_values()
        
        if len(weak_vars) > 0:
            print(f"\n  Variables with weak correlations (mean |r| < {self.weak_threshold}, excluding diagonal):")
            for var, mean_corr in weak_vars.items():
                print(f"    {var:30s}  mean |r| = {mean_corr:.3f}")
            print(f"\n  → Consider removing these variables for dimensionality reduction")
            
            # Export weak variables to CSV
            weak_df = pd.DataFrame({
                'variable': weak_vars.index,
                'mean_abs_correlation': weak_vars.values
            })
            weak_file = self.output_dir / f"weak_variables_{self.base_name}.csv"
            weak_df.to_csv(weak_file, index=False)
            print(f"  ✓ Saved weak variables list: {weak_file}")
        else:
            print(f"\n  ✓ No weak variables found (all have mean |r| ≥ {self.weak_threshold})")
        
        return weak_vars
    
    def create_word_report(self, results, findings):
        """
        Assembles all heatmaps, top correlation tables, and statistical findings
        into a structured Word document for inclusion in the thesis.

        Args:
            results (dict): Output from compute_correlation_matrices().
            findings (dict): Output from generate_correlation_report().
        """
        if not self.create_word:
            print("\nWord report creation disabled in config")
            return
        
        if Document is None:
            print("\npython-docx not available: skipping Word document")
            return
        
        print("\nGenerating Word report...")
        
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement
        
        doc = Document()
        
        # Main title
        title = doc.add_heading('Analisi Matrice di Correlazione - Sistema HVAC', level=0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title_format = title.runs[0].font
        title_format.size = Pt(16)
        title_format.bold = True
        title_format.color.rgb = RGBColor(0, 0, 0)
        
        # Date range subtitle
        subtitle = doc.add_paragraph(f'Periodo di analisi: {self.analysis_period_text}')
        subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
        subtitle_format = subtitle.runs[0].font
        subtitle_format.size = Pt(11)
        subtitle_format.bold = False
        subtitle_format.color.rgb = RGBColor(128, 128, 128)
        
        # Add methodology note about temporal aggregation
        if self.apply_temporal_aggregation:
            agg_text = f"Metodologia: Aggregazione temporale a {self.aggregation_frequency} ({self.aggregation_method}) per garantire l'indipendenza statistica dei campioni"
            method_note = doc.add_paragraph(agg_text)
            method_note.alignment = WD_ALIGN_PARAGRAPH.CENTER
            method_format = method_note.runs[0].font
            method_format.size = Pt(9)
            method_format.italic = True
            method_format.color.rgb = RGBColor(100, 100, 100)
        
        doc.add_paragraph()  # Spacing
        
        # Add each correlation heatmap in a table
        for plot_info in self.plot_records:
            # Create 3-row table
            table = doc.add_table(rows=3, cols=1)
            table.style = 'Table Grid'
            
            # Row 1: Plot title (centered, bold, dark blue)
            title_cell = table.rows[0].cells[0]
            title_para = title_cell.paragraphs[0]
            title_para.text = plot_info['title']
            title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            title_run = title_para.runs[0]
            title_run.bold = True
            title_run.font.size = Pt(12)
            title_run.font.color.rgb = RGBColor(0, 51, 102)
            
            # Set cell background color (light blue)
            shading_elm = OxmlElement('w:shd')
            shading_elm.set(qn('w:fill'), 'E6F2FF')
            title_cell._element.get_or_add_tcPr().append(shading_elm)
            
            # Row 2: Plot image
            image_cell = table.rows[1].cells[0]
            image_para = image_cell.paragraphs[0]
            image_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            if plot_info['path'].exists():
                run = image_para.add_run()
                run.add_picture(str(plot_info['path']), width=Inches(6))
            
            # Row 3: Description
            desc_cell = table.rows[2].cells[0]
            desc_para = desc_cell.paragraphs[0]
            desc_para.text = f"Descrizione:\n{plot_info['description']}"
            desc_run = desc_para.runs[0]
            desc_run.font.size = Pt(10)
            
            # Add spacing after table
            doc.add_paragraph()
        
        # Save document
        doc_file = self.plots_dir / f"correlation_matrix_report_{self.base_name}.docx"
        doc.save(str(doc_file))
        print(f"✓ Word report saved: {doc_file}")
    
    def run_analysis(self):
        """
        Orchestrates the full analysis pipeline — load, compute, plot, report —
        in one call so the script entry point stays a simple one-liner.
        """
        if not self.enable_corr:
            print("Correlation analysis is disabled in config")
            return
        
        print("\n" + "="*70)
        print("CORRELATION MATRIX ANALYSIS")
        print("="*70)
        print(f"Building: {self.building_id}")
        print(f"AHU Unit: {self.ahu_unit}")
        print(f"Season: {self.season} {self.year}")
        if self.apply_temporal_aggregation:
            print(f"Temporal Aggregation: {self.aggregation_frequency} ({self.aggregation_method})")
            print("  → Ensures statistical independence (thesis-grade rigor)")
        print("="*70)
        
        # Load data
        try:
            df = self.load_data()
        except Exception as e:
            print(f"\nFailed to load data: {e}")
            return
        
        # Compute correlation matrices
        results = self.compute_correlation_matrices(df)
        
        if not results:
            print("\nNo correlation matrices computed")
            return
        
        # Create heatmaps
        if self.create_heatmaps:
            print("\nCreating correlation heatmaps...")
            for method_name, corr_matrix in results.items():
                self.plot_correlation_heatmap(corr_matrix, method_name)
        
        # Generate findings report
        if self.generate_report:
            findings = self.generate_correlation_report(results)
            
            # Identify weak correlations (using Pearson as reference)
            if 'pearson' in results:
                self.identify_weak_correlations(results['pearson'])
            
            # Create Word report
            self.create_word_report(results, findings)
        
        print("\n" + "="*70)
        print("CORRELATION MATRIX ANALYSIS COMPLETED")
        print("="*70)
        print(f"\nOutput locations:")
        print(f"  CSV files: {self.output_dir}")
        print(f"  Plots & Report: {self.plots_dir}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.ini'
    
    print(f"HVAC Correlation Matrix Analysis")
    print(f"=" * 70)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {Path.cwd()}")
    print(f"=" * 70)
    
    try:
        analyzer = CorrelationMatrixAnalyzer(config_path)
        analyzer.run_analysis()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
