"""
HVAC Feature Importance & Selection Analysis
Identifies which features are useful and which can be removed
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import configparser
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import spearmanr
import sys

try:
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
except ImportError:
    Document = None

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')


class FeatureImportanceAnalyzer:
    """Analyzes feature importance and suggests which features to keep/remove"""
    
    def __init__(self, config_path='config.ini'):
        """Initialize analyzer"""
        self.config = configparser.ConfigParser(inline_comment_prefixes=(';', '#'))
        
        # Handle path resolution
        config_file = Path(config_path)
        
        # Try multiple locations
        search_paths = []
        
        if not config_file.is_absolute():
            # 1. Current working directory
            search_paths.append(Path.cwd() / config_path)
            
            # 2. Script directory
            script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
            search_paths.append(script_dir / config_path)
            
            # 3. Parent directory of script
            search_paths.append(script_dir.parent / config_path)
            
            # 4. Two levels up (common for nested script folders)
            search_paths.append(script_dir.parent.parent / config_path)
        else:
            search_paths.append(config_file)
        
        # Find the first existing config file
        config_file = None
        for path in search_paths:
            if path.exists():
                config_file = path
                break
        
        if config_file is None:
            print(f"\n✗ ERROR: Config file '{config_path}' not found!")
            print(f"\nSearched in the following locations:")
            for i, path in enumerate(search_paths, 1):
                print(f"  {i}. {path}")
            print(f"\nPlease ensure config.ini exists in one of these locations.")
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        print(f"✓ Reading config from: {config_file}")
        self.config.read(config_file, encoding='utf-8')
        self.config_dir = config_file.parent
        
        self.load_settings()
        
    def load_settings(self):
        """Load settings from config"""
        self.building_id = self.config.get('global', 'building_id')
        self.ahu_unit = self.config.get('global', 'ahu_unit')
        self.season = self.config.get('global', 'season')
        self.year = self.config.get('global', 'year')
        
        # Get date range - try [data] section first, then [global], then infer from data
        self.start_date = self.config.get('data', 'start_date', fallback=None)
        self.end_date = self.config.get('data', 'end_date', fallback=None)
        
        if not self.start_date or not self.start_date.strip():
            self.start_date = self.config.get('global', 'start_date', fallback=None)
        if not self.end_date or not self.end_date.strip():
            self.end_date = self.config.get('global', 'end_date', fallback=None)
        
        # Paths
        processed_folder = self._resolve_path(self.config.get('paths', 'processed_folder'))
        plots_folder = self._resolve_path(self.config.get('paths', 'plots_folder', fallback='plots'))
        
        self.base_name = f"{self.building_id}_{self.ahu_unit}_{self.season}{self.year}"
        
        # Try to find features file
        candidates = [
            processed_folder / 'feature_engineering' / self.base_name / f"{self.base_name}_features.csv",
            processed_folder / 'interpolation' / self.base_name / f"{self.base_name}_features.csv",
            processed_folder / 'interpolation' / f"{self.base_name}_features.csv",
        ]
        
        self.features_file = candidates[0]
        for candidate in candidates:
            if candidate.exists():
                self.features_file = candidate
                break
        
        # Output directories
        self.output_dir = processed_folder / 'feature_analysis' / self.base_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plots directory - create subfolder for this analysis
        self.plots_dir = plots_folder / 'feature_analysis' / self.base_name
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
    def _resolve_path(self, path_str):
        """Resolve path relative to config directory"""
        path = Path(path_str)
        if not path.is_absolute():
            path = self.config_dir / path
        return path
    
    def load_data(self):
        """Load feature-engineered data with timezone handling"""
        print(f"Loading data from: {self.features_file}")
        
        if not self.features_file.exists():
            raise FileNotFoundError(f"Features file not found: {self.features_file}")
        
        df = pd.read_csv(self.features_file)
        time_col = self.config.get('data', 'time_column', fallback='Time')
        
        if time_col in df.columns:
            # Parse timestamps - try multiple formats and handle errors
            try:
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                
                # Check if parsing was successful
                if df[time_col].isna().all():
                    print(f"  ⚠ Warning: Could not parse time column '{time_col}', skipping time indexing")
                    # Remove the unparseable column and continue without time index
                    df = df.drop(columns=[time_col])
                    if not self.start_date or not self.start_date.strip():
                        self.start_date = 'N/A'
                    if not self.end_date or not self.end_date.strip():
                        self.end_date = 'N/A'
                else:
                    # Localize to Europe/Rome for consistency with feature engineering
                    timezone = self.config.get('global', 'timezone', fallback='Europe/Rome')
                    
                    # Check if already has timezone
                    if df[time_col].dt.tz is None:
                        # Naive timestamps - assume UTC and convert to local
                        df[time_col] = df[time_col].dt.tz_localize('UTC').dt.tz_convert(timezone)
                    else:
                        # Already has timezone - just convert
                        df[time_col] = df[time_col].dt.tz_convert(timezone)
                    
                    df.set_index(time_col, inplace=True)
                    
                    # ALWAYS infer actual date range from DataFrame index
                    # This is more reliable than config values
                    self.start_date = df.index.min().strftime('%Y-%m-%d')
                    self.end_date = df.index.max().strftime('%Y-%m-%d')
                    
            except Exception as e:
                print(f"  ⚠ Warning: Error processing time column: {e}")
                print(f"  Continuing without time indexing...")
                # Fallback if no time column - try config
                if not self.start_date or not self.start_date.strip():
                    self.start_date = 'N/A'
                if not self.end_date or not self.end_date.strip():
                    self.end_date = 'N/A'
        else:
            # Fallback if no time column - try config
            if not self.start_date or not self.start_date.strip():
                self.start_date = 'N/A'
            if not self.end_date or not self.end_date.strip():
                self.end_date = 'N/A'
        
        print(f"Loaded {len(df)} records with {len(df.columns)} features")
        print(f"Date range: {self.start_date} to {self.end_date}")
        return df
    
    def identify_feature_categories(self, df):
        """Categorize features into groups"""
        categories = {
            'Original Measurements': [],
            'Temperature Features': [],
            'Time Features': [],
            'Occupancy Features': [],
            'Rolling Averages': [],
            'Control Features': [],
            'Thermal Load Features': [],
            'Interaction Features': [],
            'Rate of Change Features': [],
            'Advanced Control Features': [],
            'Cyclic Encoding': [],
            'Lag Features': []
        }
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Original measurements (no derived feature keywords)
            if not any(keyword in col_lower for keyword in [
                'delta', 'rolling', 'hour', 'day', 'weekend', 'occupied', 
                'error', 'efficiency', 'load', 'power', 'rate', 'sin', 'cos',
                'interaction', 'inertia', 'vacant', 'cumulative', 'squared',
                'tolerance', 'abs', 'lag', 'progress', 'change'
            ]):
                categories['Original Measurements'].append(col)
            
            # Temperature features
            elif 'delta_t' in col_lower:
                categories['Temperature Features'].append(col)
            
            # Time features
            elif any(kw in col_lower for kw in ['hour_of', 'day_of_week', 'month', 'progress']) and 'sin' not in col_lower and 'cos' not in col_lower:
                categories['Time Features'].append(col)
            
            # Cyclic encoding
            elif 'sin' in col_lower or 'cos' in col_lower:
                categories['Cyclic Encoding'].append(col)
            
            # Occupancy
            elif 'occupied' in col_lower or 'weekend' in col_lower:
                categories['Occupancy Features'].append(col)
            
            # Rolling averages
            elif 'rolling' in col_lower:
                categories['Rolling Averages'].append(col)
            
            # Control features
            elif any(kw in col_lower for kw in ['setpoint_error', 'efficiency_proxy']) and 'rate' not in col_lower and 'cumulative' not in col_lower:
                categories['Control Features'].append(col)
            
            # Advanced control
            elif any(kw in col_lower for kw in ['cumulative', 'tolerance', 'squared', 'abs_setpoint']):
                categories['Advanced Control Features'].append(col)
            
            # Thermal load
            elif any(kw in col_lower for kw in ['thermal_load', 'thermal_power', 'cooling_load', 'heating_load']):
                categories['Thermal Load Features'].append(col)
            
            # Interactions
            elif any(kw in col_lower for kw in ['temp_occupied', 'temp_vacant', 'temp_weekend', 'temp_hour', 'temp_inertia', 'load_occupied']):
                categories['Interaction Features'].append(col)
            
            # Rate of change
            elif '_rate' in col_lower or '_change' in col_lower:
                categories['Rate of Change Features'].append(col)
            
            # Lag features
            elif 'lag' in col_lower:
                categories['Lag Features'].append(col)
        
        # Remove empty categories
        categories = {k: v for k, v in categories.items() if v}
        return categories
    
    def calculate_missing_variance(self, df):
        """Calculate missing data percentage and variance for each feature"""
        stats = []
        
        for col in df.columns:
            missing_pct = (df[col].isna().sum() / len(df)) * 100
            
            if df[col].dtype in ['float64', 'int64']:
                variance = df[col].var()
                std = df[col].std()
                mean = df[col].mean()
                cv = (std / mean * 100) if mean != 0 else 0
            else:
                variance = np.nan
                std = np.nan
                mean = np.nan
                cv = np.nan
            
            stats.append({
                'feature': col,
                'missing_pct': missing_pct,
                'variance': variance,
                'std': std,
                'mean': mean,
                'cv': cv
            })
        
        return pd.DataFrame(stats)
    
    def calculate_correlation_with_targets(self, df):
        """Calculate correlation with key target variables"""
        # Define potential target variables
        target_candidates = [
            'Modulazione Mandata',
            'Modulazione Ripresa', 
            'thermal_load',
            'thermal_power',
            'cooling_load',
            'heating_load',
            'Temperatura Mandata',
            'setpoint_error'
        ]
        
        # Find which targets exist
        targets = [t for t in target_candidates if t in df.columns]
        
        if not targets:
            print("Warning: No target variables found for correlation analysis")
            return None
        
        print(f"\nCalculating correlations with targets: {targets}")
        
        correlations = []
        
        for col in df.columns:
            if col in targets:
                continue
            
            if df[col].dtype not in ['float64', 'int64']:
                continue
            
            row = {'feature': col}
            
            for target in targets:
                # Calculate Spearman correlation (handles non-linear relationships)
                valid_data = df[[col, target]].dropna()
                
                if len(valid_data) > 10:
                    corr, pval = spearmanr(valid_data[col], valid_data[target])
                    row[f'{target}_corr'] = corr
                    row[f'{target}_pval'] = pval
                else:
                    row[f'{target}_corr'] = np.nan
                    row[f'{target}_pval'] = np.nan
            
            # Calculate max absolute correlation across all targets
            corr_cols = [c for c in row.keys() if c.endswith('_corr')]
            if corr_cols:
                row['max_abs_corr'] = max([abs(row[c]) for c in corr_cols if not pd.isna(row[c])], default=0)
                # Find which target has the highest correlation
                best_corr = 0
                best_target = 'N/A'
                for corr_col in corr_cols:
                    if not pd.isna(row[corr_col]) and abs(row[corr_col]) > abs(best_corr):
                        best_corr = row[corr_col]
                        best_target = corr_col.replace('_corr', '')
                row['best_target'] = best_target
            else:
                row['max_abs_corr'] = 0
                row['best_target'] = 'N/A'
            
            correlations.append(row)
        
        return pd.DataFrame(correlations)
    
    def calculate_mutual_information(self, df, target_col='thermal_load'):
        """Calculate mutual information scores (measures non-linear dependencies)"""
        if target_col not in df.columns:
            # Try alternative targets
            alternatives = ['Modulazione Mandata', 'thermal_power', 'Temperatura Mandata']
            target_col = next((t for t in alternatives if t in df.columns), None)
            
            if target_col is None:
                print("Warning: No suitable target for mutual information")
                return None
        
        print(f"\nCalculating mutual information with target: {target_col}")
        
        # Get numeric features
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        feature_cols = [c for c in numeric_cols if c != target_col]
        
        # Prepare data
        X = df[feature_cols].fillna(df[feature_cols].median())
        y = df[target_col].fillna(df[target_col].median())
        
        # Calculate mutual information
        mi_scores = mutual_info_regression(X, y, random_state=42)
        
        mi_df = pd.DataFrame({
            'feature': feature_cols,
            'mutual_info': mi_scores
        }).sort_values('mutual_info', ascending=False)
        
        return mi_df
    
    def calculate_feature_importance_rf(self, df, target_col='thermal_load'):
        """Calculate feature importance using Random Forest"""
        if target_col not in df.columns:
            alternatives = ['Modulazione Mandata', 'thermal_power', 'Temperatura Mandata']
            target_col = next((t for t in alternatives if t in df.columns), None)
            
            if target_col is None:
                print("Warning: No suitable target for Random Forest")
                return None
        
        print(f"\nCalculating Random Forest importance with target: {target_col}")
        
        # Get numeric features
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        feature_cols = [c for c in numeric_cols if c != target_col]
        
        # Prepare data
        X = df[feature_cols].fillna(df[feature_cols].median())
        y = df[target_col].fillna(df[target_col].median())
        
        # Train Random Forest
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Get feature importances
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'rf_importance': rf.feature_importances_
        }).sort_values('rf_importance', ascending=False)
        
        return importance_df
    
    def create_comprehensive_report(self, df, categories, stats_df, corr_df, mi_df, rf_df):
        """Create comprehensive feature analysis report"""
        print("\n" + "="*70)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*70)
        
        # Identify target columns used in analysis
        target_candidates = [
            'Modulazione Mandata', 'Modulazione Ripresa', 
            'thermal_load', 'thermal_power', 'cooling_load', 'heating_load',
            'Temperatura Mandata', 'setpoint_error'
        ]
        targets_in_data = [t for t in target_candidates if t in df.columns]
        
        # Merge all metrics
        report = stats_df.copy()
        
        if corr_df is not None:
            report = report.merge(corr_df[['feature', 'max_abs_corr', 'best_target']], on='feature', how='left')
        
        if mi_df is not None:
            report = report.merge(mi_df[['feature', 'mutual_info']], on='feature', how='left')
        
        if rf_df is not None:
            report = report.merge(rf_df[['feature', 'rf_importance']], on='feature', how='left')
        
        # CRITICAL FIX: Exclude target columns from recommendations
        # These should not be evaluated as features since they are prediction targets
        if targets_in_data:
            print(f"\nExcluding target columns from feature recommendations: {', '.join(targets_in_data)}")
            report = report[~report['feature'].isin(targets_in_data)].copy()
        else:
            print("\nNo target columns found to exclude")
        
        # Calculate composite score
        report['composite_score'] = 0
        
        if 'max_abs_corr' in report.columns:
            report['composite_score'] += report['max_abs_corr'].fillna(0) * 0.3
        
        if 'mutual_info' in report.columns:
            mi_normalized = report['mutual_info'].fillna(0) / report['mutual_info'].max() if report['mutual_info'].max() > 0 else 0
            report['composite_score'] += mi_normalized * 0.35
        
        if 'rf_importance' in report.columns:
            report['composite_score'] += report['rf_importance'].fillna(0) * 0.35
        
        # Add recommendation
        def get_recommendation(row):
            # Remove if high missing data
            if row['missing_pct'] > 50:
                return 'REMOVE - High Missing Data'
            
            # Remove if zero variance
            if pd.notna(row['variance']) and row['variance'] < 1e-10:
                return 'REMOVE - No Variation'
            
            # Keep if high importance
            if row['composite_score'] > 0.5:
                return 'KEEP - High Importance'
            
            # Consider if medium importance
            if row['composite_score'] > 0.2:
                return 'CONSIDER - Medium Importance'
            
            # Remove if low importance
            return 'REMOVE - Low Importance'
        
        report['recommendation'] = report.apply(get_recommendation, axis=1)
        
        # Sort by composite score
        report = report.sort_values('composite_score', ascending=False)
        
        # Save full report
        report_file = self.output_dir / f"{self.base_name}_feature_importance_report.csv"
        report.to_csv(report_file, index=False)
        print(f"\nSaved detailed report: {report_file}")
        
        # Save selected features list (KEEP + CONSIDER for downstream use)
        keep_features = report[report['recommendation'].str.contains('KEEP', na=False)]['feature'].tolist()
        consider_features = report[report['recommendation'].str.contains('CONSIDER', na=False)]['feature'].tolist()
        
        # Export KEEP features only (high confidence)
        keep_file = self.output_dir / f"{self.base_name}_selected_features_keep.csv"
        pd.DataFrame({'feature': keep_features}).to_csv(keep_file, index=False)
        print(f"Saved KEEP features list: {keep_file} ({len(keep_features)} features)")
        
        # Export KEEP + CONSIDER (more inclusive for experimentation)
        all_selected = keep_features + consider_features
        selected_file = self.output_dir / f"{self.base_name}_selected_features.csv"
        pd.DataFrame({'feature': all_selected}).to_csv(selected_file, index=False)
        print(f"Saved KEEP + CONSIDER features list: {selected_file} ({len(all_selected)} features)")
        
        # Print summary
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)
        
        print(f"\nTotal features analyzed: {len(report)}")
        print(f"Target columns excluded from analysis: {', '.join(targets_in_data)}")
        print(f"\nRecommendations:")
        for rec, count in report['recommendation'].value_counts().items():
            print(f"  {rec}: {count}")
        
        # Show top features
        print("\n" + "="*70)
        print("TOP 15 MOST IMPORTANT FEATURES")
        print("="*70)
        
        top_15 = report.head(15)[['feature', 'composite_score', 'max_abs_corr', 'mutual_info', 'rf_importance', 'recommendation']]
        print(top_15.to_string(index=False))
        
        # Show features to remove
        print("\n" + "="*70)
        print("FEATURES RECOMMENDED FOR REMOVAL")
        print("="*70)
        
        to_remove = report[report['recommendation'].str.contains('REMOVE')]
        if len(to_remove) > 0:
            print(f"\n{len(to_remove)} features recommended for removal:")
            for idx, row in to_remove.iterrows():
                reason = row['recommendation'].split(' - ')[1] if ' - ' in row['recommendation'] else 'Low Importance'
                print(f"  • {row['feature']}: {reason}")
        else:
            print("\nNo features recommended for removal")
        
        return report, targets_in_data
    
    def create_visualizations(self, report, categories):
        """Create visualization plots"""
        print("\nCreating visualizations...")
        
        # Date range string
        date_range = f"Periodo: {self.start_date} - {self.end_date}"
        
        # 1. Feature Importance Bar Chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        top_30 = report.head(30).sort_values('composite_score')
        colors = ['green' if 'KEEP' in r else 'orange' if 'CONSIDER' in r else 'red' 
                  for r in top_30['recommendation']]
        
        ax.barh(range(len(top_30)), top_30['composite_score'], color=colors, alpha=0.7)
        ax.set_yticks(range(len(top_30)))
        ax.set_yticklabels(top_30['feature'], fontsize=9)
        ax.set_xlabel('Composite Importance Score', fontsize=12)
        ax.set_title(f'Top 30 Features by Importance\n{self.building_id} - {self.ahu_unit} - {self.season} {self.year}\n{date_range}',
                     fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='KEEP'),
            Patch(facecolor='orange', alpha=0.7, label='CONSIDER'),
            Patch(facecolor='red', alpha=0.7, label='REMOVE')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plot_file = self.plots_dir / f"{self.base_name}_feature_importance.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {plot_file}")
        plt.close()
        
        # 2. Feature Categories Distribution
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 2a. Distribution by category
        category_counts = {cat: len(features) for cat, features in categories.items()}
        ax = axes[0]
        ax.bar(range(len(category_counts)), list(category_counts.values()), alpha=0.7)
        ax.set_xticks(range(len(category_counts)))
        ax.set_xticklabels(list(category_counts.keys()), rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Number of Features')
        ax.set_title('Features by Category', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # 2b. Variance distribution
        ax = axes[1]
        variance_log = np.log10(report['variance'].replace(0, np.nan).dropna())
        ax.hist(variance_log, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax.set_xlabel('Log10(Variance)')
        ax.set_ylabel('Number of Features')
        ax.set_title('Feature Variance Distribution', fontweight='bold')
        ax.grid(alpha=0.3)
        
        # 2c. Recommendation pie chart
        ax = axes[2]
        rec_counts = report['recommendation'].value_counts()
        colors_pie = ['green' if 'KEEP' in r else 'orange' if 'CONSIDER' in r else 'red' for r in rec_counts.index]
        ax.pie(rec_counts, labels=rec_counts.index, autopct='%1.1f%%', colors=colors_pie, startangle=90)
        ax.set_title('Feature Recommendations', fontweight='bold')
        
        plt.suptitle(f'Feature Analysis Summary - {self.building_id} {self.ahu_unit} {self.season} {self.year}\n{date_range}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_file = self.plots_dir / f"{self.base_name}_feature_analysis_summary.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {plot_file}")
        plt.close()
    
    def create_word_report(self, report, categories, targets_excluded=None):
        """Create Word document with recommendations in Italian"""
        if Document is None:
            print("\n! python-docx not installed, skipping Word report")
            return
        
        print("\nGenerating Word report with recommendations...")
        
        doc = Document()
        
        # Extract script name without number
        script_name = "Analisi Importanza delle Feature"
        
        # Main Title (centered, bold)
        title = doc.add_heading(script_name, 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Date range (centered, below title)
        date_para = doc.add_paragraph()
        date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        date_run = date_para.add_run(f"Periodo di analisi: {self.start_date} - {self.end_date}")
        date_run.font.size = Pt(12)
        
        doc.add_paragraph()  # Spacing
        
        # Plot 1: Feature Importance Bar Chart
        plot1_path = self.plots_dir / f"{self.base_name}_feature_importance.png"
        if plot1_path.exists():
            # Title in table
            table = doc.add_table(rows=3, cols=1)
            table.style = 'Light Grid Accent 1'
            
            # Row 1: Title
            title_cell = table.rows[0].cells[0]
            title_para = title_cell.paragraphs[0]
            title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            title_run = title_para.add_run("Top 30 Feature per Importanza")
            title_run.bold = True
            title_run.font.size = Pt(14)
            title_run.font.color.rgb = RGBColor(0, 51, 102)  # Dark blue
            
            # Row 2: Plot image
            img_cell = table.rows[1].cells[0]
            img_para = img_cell.paragraphs[0]
            img_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            img_para.add_run().add_picture(str(plot1_path), width=Inches(6.0))
            
            # Row 3: Description
            desc_cell = table.rows[2].cells[0]
            desc_para = desc_cell.paragraphs[0]
            desc_title = desc_para.add_run("Descrizione:\n")
            desc_title.bold = True
            desc_para.add_run(
                "Questo grafico mostra le 30 feature più importanti ordinate per punteggio composito. "
                "Le barre verdi indicano feature da MANTENERE (alta importanza), "
                "le barre arancioni indicano feature da CONSIDERARE (importanza media), "
                "e le barre rosse indicano feature da RIMUOVERE (bassa importanza)."
            )
            
            doc.add_paragraph()  # Spacing
        
        # Plot 2: Feature Analysis Summary
        plot2_path = self.plots_dir / f"{self.base_name}_feature_analysis_summary.png"
        if plot2_path.exists():
            # Title in table
            table = doc.add_table(rows=3, cols=1)
            table.style = 'Light Grid Accent 1'
            
            # Row 1: Title
            title_cell = table.rows[0].cells[0]
            title_para = title_cell.paragraphs[0]
            title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            title_run = title_para.add_run("Riepilogo Analisi delle Feature")
            title_run.bold = True
            title_run.font.size = Pt(14)
            title_run.font.color.rgb = RGBColor(0, 51, 102)  # Dark blue
            
            # Row 2: Plot image
            img_cell = table.rows[1].cells[0]
            img_para = img_cell.paragraphs[0]
            img_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            img_para.add_run().add_picture(str(plot2_path), width=Inches(6.0))
            
            # Row 3: Description
            desc_cell = table.rows[2].cells[0]
            desc_para = desc_cell.paragraphs[0]
            desc_title = desc_para.add_run("Descrizione:\n")
            desc_title.bold = True
            desc_para.add_run(
                "Questa figura contiene quattro sottografici che forniscono una visione completa dell'analisi: "
                "distribuzione delle feature per categoria, distribuzione dei dati mancanti, "
                "distribuzione della varianza delle feature, e un grafico a torta con le raccomandazioni finali."
            )
            
            doc.add_paragraph()  # Spacing
        
        # Riepilogo Esecutivo
        doc.add_heading('Riepilogo Esecutivo', 1)
        
        keep_count = len(report[report['recommendation'].str.contains('KEEP')])
        consider_count = len(report[report['recommendation'].str.contains('CONSIDER')])
        remove_count = len(report[report['recommendation'].str.contains('REMOVE')])
        
        summary_para = doc.add_paragraph()
        summary_para.add_run(f"• Feature da MANTENERE: {keep_count}\n").bold = True
        summary_para.add_run(f"• Feature da CONSIDERARE: {consider_count}\n")
        summary_para.add_run(f"• Feature da RIMUOVERE: {remove_count}\n")
        
        # Add note about excluded targets
        if targets_excluded:
            doc.add_paragraph()
            note_para = doc.add_paragraph()
            note_run = note_para.add_run("Nota: ")
            note_run.bold = True
            note_para.add_run(f"Le seguenti colonne target sono state escluse dall'analisi: {', '.join(targets_excluded)}")
        
        doc.add_paragraph()
        
        # Top 15 Features
        doc.add_heading('Top 15 Feature Più Importanti', 1)
        
        top_15 = report.head(15)
        for i, (idx, row) in enumerate(top_15.iterrows(), 1):
            doc.add_heading(f"{i}. {row['feature']}", 2)
            
            stats_para = doc.add_paragraph()
            stats_para.add_run(f"Punteggio Importanza: {row['composite_score']:.3f}\n")
            stats_para.add_run(f"Raccomandazione: ").bold = True
            
            # Translate recommendation
            rec_text = row['recommendation']
            if 'KEEP' in rec_text:
                rec_translated = 'MANTENERE - Alta Importanza'
                color = RGBColor(0, 128, 0)
            elif 'CONSIDER' in rec_text:
                rec_translated = 'CONSIDERARE - Importanza Media'
                color = RGBColor(255, 140, 0)
            else:
                rec_translated = 'RIMUOVERE - Bassa Importanza'
                color = RGBColor(255, 0, 0)
            
            rec_run = stats_para.add_run(f"{rec_translated}\n")
            rec_run.font.color.rgb = color
            
            if pd.notna(row.get('max_abs_corr')):
                stats_para.add_run(f"Correlazione Massima: {row['max_abs_corr']:.3f}\n")
            
            doc.add_paragraph()
        
        # Features da Rimuovere
        doc.add_heading('Feature Raccomandate per la Rimozione', 1)
        
        to_remove = report[report['recommendation'].str.contains('REMOVE')]
        if len(to_remove) > 0:
            for idx, row in to_remove.iterrows():
                para = doc.add_paragraph(style='List Bullet')
                para.add_run(f"{row['feature']}: ").bold = True
                
                # Translate reason
                reason = row['recommendation'].split(' - ')[1] if ' - ' in row['recommendation'] else 'Low Importance'
                if 'High Missing Data' in reason:
                    reason = 'Molti Dati Mancanti'
                elif 'No Variation' in reason:
                    reason = 'Nessuna Variazione'
                elif 'Low Importance' in reason:
                    reason = 'Bassa Importanza'
                
                para.add_run(reason)
        else:
            doc.add_paragraph("Nessuna feature raccomandata per la rimozione.")
        
        # Save document
        doc_file = self.plots_dir / f"{self.base_name}_feature_recommendations.docx"
        doc.save(str(doc_file))
        print(f"✓ Saved Word report: {doc_file}")
    
    def run_analysis(self):
        """Run complete feature importance analysis"""
        print("\n" + "="*70)
        print("FEATURE IMPORTANCE & SELECTION ANALYSIS")
        print("="*70)
        print(f"Building: {self.building_id}")
        print(f"AHU Unit: {self.ahu_unit}")
        print(f"Season: {self.season} {self.year}")
        print("="*70)
        
        # Load data
        df = self.load_data()
        
        # Identify categories
        print("\nCategorizing features...")
        categories = self.identify_feature_categories(df)
        for cat, features in categories.items():
            print(f"  {cat}: {len(features)} features")
        
        # Calculate statistics
        print("\nCalculating feature statistics...")
        stats_df = self.calculate_missing_variance(df)
        
        # Calculate correlations
        corr_df = self.calculate_correlation_with_targets(df)
        
        # Calculate mutual information
        mi_df = self.calculate_mutual_information(df)
        
        # Calculate Random Forest importance
        rf_df = self.calculate_feature_importance_rf(df)
        
        # Create comprehensive report
        report, targets_excluded = self.create_comprehensive_report(df, categories, stats_df, corr_df, mi_df, rf_df)
        
        # Create visualizations
        self.create_visualizations(report, categories)
        
        # Create Word report
        self.create_word_report(report, categories, targets_excluded)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETED")
        print("="*70)
        print(f"\nCSV report saved to: {self.output_dir}")
        print(f"Plots and Word report saved to: {self.plots_dir}")
        
        return report, categories


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.ini'
    
    print(f"Feature Importance & Selection Analysis")
    print(f"=" * 70)
    
    try:
        analyzer = FeatureImportanceAnalyzer(config_path)
        report, categories = analyzer.run_analysis()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
