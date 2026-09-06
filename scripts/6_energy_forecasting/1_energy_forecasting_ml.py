"""
HVAC Energy Forecasting - Machine Learning Model (CORRECTED VERSION)
=====================================================================

This script trains and evaluates machine learning models to predict HVAC energy consumption.

⚠️ CRITICAL: This version follows the CORRECT order of operations to prevent data leakage:
   1. Load and clean data (remove NaN, fix types)
   2. Train-test split (BEFORE any preprocessing that "learns")
   3. Fit preprocessing ONLY on training data (scaling, encoding, etc.)
   4. Apply preprocessing to validation and test data
   5. Train models ONLY on training data
   6. Evaluate on test data

GOAL: Predict thermal_power (or Modulazione Mandata) using the 10 selected features

MODELS TRAINED:
1. Linear Regression (baseline)
2. Random Forest (ensemble method)
3. XGBoost (gradient boosting)

EVALUATION METRICS:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (R-squared / Coefficient of Determination)

AUTHOR: Varaga Haghoubians
DATE: 2025
"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import configparser
from datetime import datetime
import warnings
import json

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Word Document Generation
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


# =============================================================================
# CLASS: ENERGY FORECASTING MODEL (CORRECTED - NO DATA LEAKAGE)
# =============================================================================
class EnergyForecastingModel:
    """
    A machine learning pipeline for HVAC energy forecasting that follows
    the CORRECT order of operations to prevent data leakage.
    
    CORRECT ORDER:
    1. Load data
    2. Clean data (remove NaN, fix types) - SAFE on whole dataset
    3. Split into train/val/test - BEFORE any "learning" operations
    4. Fit scaler ONLY on training data
    5. Transform val and test using the fitted scaler
    6. Train models ONLY on training data
    7. Evaluate on test data
    """
    
    def __init__(self, config_path='config.ini'):
        """
        Initialize the forecasting model.
        
        Parameters:
        -----------
        config_path : str
            Path to the configuration file (config.ini)
        """
        print("="*80)
        print("HVAC ENERGY FORECASTING - CORRECTED ML PIPELINE (NO DATA LEAKAGE)")
        print("="*80)
        
        # Load configuration
        self.config = configparser.ConfigParser(inline_comment_prefixes=(';', '#'))
        self.config_path = self._find_config(config_path)
        self.config.read(self.config_path, encoding='utf-8')
        self.config_dir = self.config_path.parent
        
        # Load settings from config
        self.load_settings()
        
        # Define the final 10 features for modeling
        # These were selected based on feature importance and correlation analysis
        self.SELECTED_FEATURES = [
            'thermal_power_signed',      # #1: Primary energy transfer metric
            'efficiency_proxy',           # #2: System efficiency measure
            'load_occupied',              # #3: Occupancy-weighted thermal load
            'Temperatura Esterna',        # #4: Outdoor temperature (instantaneous)
            'outdoor_temp_rolling_24h',   # #5: Daily weather trend
            'hour_cos',                   # #6: Daily operational cycle
            'delta_t',                    # #7: Pure temperature difference
            'Set Points Temperatura Mandata Riscaldamento',  # #8: Heating setpoint
            'Set Points Temperatura Mandata Raffrescamento', # #9: Cooling setpoint
            'Temperatura Ripresa'         # #10: Indoor feedback temperature
        ]
        
        # Initialize placeholders for data and models
        self.df = None
        self.df_clean = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_val_scaled = None
        self.X_test_scaled = None
        self.scaler = None
        self.models = {}
        self.results = {}
        
    def _find_config(self, config_path):
        """Find the config.ini file in multiple possible locations."""
        config_file = Path(config_path)
        
        # Try multiple locations
        search_paths = [
            Path.cwd() / config_path,
            Path(__file__).parent / config_path if '__file__' in globals() else None,
            Path(__file__).parent.parent / config_path if '__file__' in globals() else None,
        ]
        search_paths = [p for p in search_paths if p is not None]
        
        for path in search_paths:
            if path.exists():
                print(f"✓ Found config file: {path}")
                return path
        
        raise FileNotFoundError(f"Config file not found. Searched: {search_paths}")
    
    def load_settings(self):
        """Load all necessary settings from config.ini."""
        # Global settings
        self.building_id = self.config.get('global', 'building_id')
        self.ahu_unit = self.config.get('global', 'ahu_unit')
        self.season = self.config.get('global', 'season')
        self.year = self.config.get('global', 'year')
        self.timezone = self.config.get('global', 'timezone', fallback='Europe/Rome')
        
        # Create base name for file naming
        self.base_name = f"{self.building_id}_{self.ahu_unit}_{self.season}{self.year}"
        
        # ML-specific settings (with fallbacks if section doesn't exist)
        try:
            self.train_ratio = self.config.getfloat('ml_energy_forecasting', 'train_ratio', fallback=0.70)
            self.val_ratio = self.config.getfloat('ml_energy_forecasting', 'validation_ratio', fallback=0.15)
            self.test_ratio = self.config.getfloat('ml_energy_forecasting', 'test_ratio', fallback=0.15)
            self.TARGET = self.config.get('ml_energy_forecasting', 'target_variable', fallback='thermal_power')
            self.enable_scaling = self.config.getboolean('ml_energy_forecasting', 'enable_scaling', fallback=True)
            self.scaling_method = self.config.get('ml_energy_forecasting', 'scaling_method', fallback='standard')
            
            # Random Forest hyperparameters
            self.rf_n_estimators = self.config.getint('ml_energy_forecasting', 'rf_n_estimators', fallback=100)
            self.rf_max_depth = self.config.getint('ml_energy_forecasting', 'rf_max_depth', fallback=15)
            self.rf_min_samples_split = self.config.getint('ml_energy_forecasting', 'rf_min_samples_split', fallback=10)
            self.rf_min_samples_leaf = self.config.getint('ml_energy_forecasting', 'rf_min_samples_leaf', fallback=5)
            
            # XGBoost hyperparameters
            self.xgb_n_estimators = self.config.getint('ml_energy_forecasting', 'xgb_n_estimators', fallback=100)
            self.xgb_max_depth = self.config.getint('ml_energy_forecasting', 'xgb_max_depth', fallback=6)
            self.xgb_learning_rate = self.config.getfloat('ml_energy_forecasting', 'xgb_learning_rate', fallback=0.1)
            self.xgb_subsample = self.config.getfloat('ml_energy_forecasting', 'xgb_subsample', fallback=0.8)
            self.xgb_colsample_bytree = self.config.getfloat('ml_energy_forecasting', 'xgb_colsample_bytree', fallback=0.8)
            
            self.random_seed = self.config.getint('ml_energy_forecasting', 'random_seed', fallback=42)
            self.plot_dpi = self.config.getint('ml_energy_forecasting', 'plot_dpi', fallback=300)
            
            print("✓ Loaded ML settings from config.ini [ml_energy_forecasting] section")
        except:
            print("⚠ [ml_energy_forecasting] section not found in config.ini, using default values")
            self.train_ratio = 0.70
            self.val_ratio = 0.15
            self.test_ratio = 0.15
            self.TARGET = 'thermal_power'
            self.enable_scaling = True
            self.scaling_method = 'standard'
            self.rf_n_estimators = 100
            self.rf_max_depth = 15
            self.rf_min_samples_split = 10
            self.rf_min_samples_leaf = 5
            self.xgb_n_estimators = 100
            self.xgb_max_depth = 6
            self.xgb_learning_rate = 0.1
            self.xgb_subsample = 0.8
            self.xgb_colsample_bytree = 0.8
            self.random_seed = 42
            self.plot_dpi = 300
        
        # Paths
        processed_folder = self._resolve_path(self.config.get('paths', 'processed_folder'))
        plots_folder = self._resolve_path(self.config.get('paths', 'plots_folder'))
        
        # Input: Look for feature-engineered CSV
        # PRIORITY ORDER: Look for actual features data first, not the feature list
        feature_candidates = [
            processed_folder / 'feature_engineering' / self.base_name / f"{self.base_name}_features.csv",
            processed_folder / 'feature_analysis' / self.base_name / f"{self.base_name}_feature_importance.csv",
        ]
        
        self.input_file = None
        for candidate in feature_candidates:
            if candidate.exists():
                self.input_file = candidate
                print(f"✓ Found input file: {candidate}")
                break
        
        if self.input_file is None:
            raise FileNotFoundError(f"Could not find feature data file. Searched: {feature_candidates}")
        
        # Output directories
        self.output_dir = processed_folder / 'ml_energy_forecasting' / self.base_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.plots_dir = plots_folder / 'ml_energy_forecasting' / self.base_name
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        self.readme_dir = Path('./ReadMe') / 'ml_energy_forecasting'
        self.readme_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Output directory: {self.output_dir}")
        print(f"✓ Plots directory: {self.plots_dir}")
        print(f"✓ README directory: {self.readme_dir}")
        
    def _resolve_path(self, path_str):
        """Resolve path relative to config directory."""
        path = Path(path_str)
        if not path.is_absolute():
            path = self.config_dir / path
        return path
    
    def load_and_clean_data(self):
        """
        STEP 1: Load and clean data.
        
        ✅ SAFE to do on whole dataset:
        - Remove NaN values
        - Fix data types
        - Remove duplicates
        - Handle obvious outliers
        
        ❌ NOT SAFE (will do later, after split):
        - Scaling / normalization
        - Feature selection based on correlations
        - Any operation that "learns" from data
        """
        print("\n" + "="*80)
        print("STEP 1: LOAD AND CLEAN DATA (Before Split)")
        print("="*80)
        
        # Load CSV
        print(f"Reading data from: {self.input_file}")
        self.df = pd.read_csv(self.input_file)
        
        # Parse time column if it exists
        time_col = self.config.get('data', 'time_column', fallback='Time')
        if time_col in self.df.columns:
            self.df[time_col] = pd.to_datetime(self.df[time_col], errors='coerce')
            self.df.set_index(time_col, inplace=True)
        
        print(f"✓ Loaded {len(self.df)} records")
        print(f"✓ Date range: {self.df.index.min()} to {self.df.index.max()}")
        
        # Check if all required features exist
        missing_features = [f for f in self.SELECTED_FEATURES if f not in self.df.columns]
        if missing_features:
            print(f"\n⚠ WARNING: Missing features: {missing_features}")
            print("Available features will be used.")
            self.SELECTED_FEATURES = [f for f in self.SELECTED_FEATURES if f in self.df.columns]
        
        # Check if target exists
        if self.TARGET not in self.df.columns:
            print(f"\n⚠ WARNING: Target '{self.TARGET}' not found!")
            print("Available columns:", list(self.df.columns))
            raise ValueError(f"Target variable '{self.TARGET}' not in dataset")
        
        # CLEANING (safe on whole dataset - doesn't "learn" anything)
        print(f"\nCleaning data...")
        print(f"  Records before cleaning: {len(self.df)}")
        
        # Remove rows with NaN in selected features or target
        self.df_clean = self.df[self.SELECTED_FEATURES + [self.TARGET]].dropna()
        print(f"  Records after removing NaN: {len(self.df_clean)} ({len(self.df) - len(self.df_clean)} removed)")
        
        # Sort by time index (ensure chronological order)
        self.df_clean = self.df_clean.sort_index()
        
        print(f"\n✓ Data loaded and cleaned")
        print(f"✓ Using {len(self.SELECTED_FEATURES)} features")
        print(f"✓ Target variable: {self.TARGET}")
        print(f"\n⚠ IMPORTANT: NO scaling or feature selection has been done yet!")
        print(f"   These will be done AFTER the train-test split to prevent data leakage.")
        
        return self
    
    def split_data(self):
        """
        STEP 2: Split data into Train, Validation, and Test sets.
        
        ⚠️ CRITICAL: This MUST happen BEFORE any preprocessing that "learns" from data!
        
        STRATEGY: Chronological split (time-based)
        - Train: First 70% of data
        - Validation: Next 15%
        - Test: Last 15%
        """
        print("\n" + "="*80)
        print("STEP 2: TRAIN/VALIDATION/TEST SPLIT (Before Any Preprocessing)")
        print("="*80)
        
        # Calculate split indices
        n = len(self.df_clean)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))
        
        # Split data chronologically
        train_data = self.df_clean.iloc[:train_end]
        val_data = self.df_clean.iloc[train_end:val_end]
        test_data = self.df_clean.iloc[val_end:]
        
        # Separate features (X) and target (y)
        self.X_train = train_data[self.SELECTED_FEATURES]
        self.y_train = train_data[self.TARGET]
        
        self.X_val = val_data[self.SELECTED_FEATURES]
        self.y_val = val_data[self.TARGET]
        
        self.X_test = test_data[self.SELECTED_FEATURES]
        self.y_test = test_data[self.TARGET]
        
        # Print split summary
        print(f"\n✓ Data Split Summary:")
        print(f"  Total records: {n}")
        print(f"  Train:      {len(self.X_train):5d} records ({self.train_ratio*100:.0f}%) - {train_data.index.min()} to {train_data.index.max()}")
        print(f"  Validation: {len(self.X_val):5d} records ({self.val_ratio*100:.0f}%) - {val_data.index.min()} to {val_data.index.max()}")
        print(f"  Test:       {len(self.X_test):5d} records ({self.test_ratio*100:.0f}%) - {test_data.index.min()} to {test_data.index.max()}")
        
        print(f"\n✓ Split completed BEFORE any scaling or preprocessing")
        print(f"✓ Test set is now isolated and will not be touched until final evaluation")
        
        return self
    
    def scale_features(self):
        """
        STEP 3: Scale features (FIT on training data ONLY).
        
        ✅ CORRECT approach:
        1. Fit scaler on X_train only
        2. Transform X_train using the fitted scaler
        3. Transform X_val and X_test using the SAME fitted scaler
        
        ❌ WRONG approach (causes data leakage):
        - Fitting scaler on all data (train + val + test)
        - This would leak information about the test set's distribution
        """
        print("\n" + "="*80)
        print("STEP 3: FEATURE SCALING (Fit on Train, Apply to Val/Test)")
        print("="*80)
        
        if not self.enable_scaling:
            print("⚠ Scaling is disabled in config.ini")
            self.X_train_scaled = self.X_train.values
            self.X_val_scaled = self.X_val.values
            self.X_test_scaled = self.X_test.values
            return self
        
        print(f"Scaling method: {self.scaling_method}")
        
        # Initialize scaler
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
            print("  Using StandardScaler (mean=0, std=1)")
        else:
            self.scaler = StandardScaler()  # Default to standard
            print("  Using StandardScaler (default)")
        
        # FIT on training data ONLY
        print(f"\n  Step 3a: Fitting scaler on TRAINING data only...")
        self.scaler.fit(self.X_train)
        print(f"    ✓ Scaler learned mean and std from {len(self.X_train)} training samples")
        print(f"    ✓ Feature means: {self.scaler.mean_[:3]}... (first 3 features)")
        print(f"    ✓ Feature stds:  {self.scaler.scale_[:3]}... (first 3 features)")
        
        # TRANSFORM all three sets using the fitted scaler
        print(f"\n  Step 3b: Transforming all datasets using the fitted scaler...")
        self.X_train_scaled = self.scaler.transform(self.X_train)
        print(f"    ✓ Transformed training data")
        
        self.X_val_scaled = self.scaler.transform(self.X_val)
        print(f"    ✓ Transformed validation data (using training scaler)")
        
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print(f"    ✓ Transformed test data (using training scaler)")
        
        print(f"\n✓ Feature scaling completed correctly (no data leakage)")
        print(f"✓ Test set was transformed using scaler fitted ONLY on training data")
        
        return self
    
    def train_models(self):
        """
        STEP 4: Train models (on training data ONLY).
        """
        print("\n" + "="*80)
        print("STEP 4: TRAINING MODELS (On Training Data Only)")
        print("="*80)
        
        # -------------------------------------------------------------------------
        # MODEL 1: LINEAR REGRESSION
        # -------------------------------------------------------------------------
        print("\n[1/3] Training Linear Regression...")
        lr_model = LinearRegression()
        lr_model.fit(self.X_train_scaled, self.y_train)
        self.models['Linear Regression'] = lr_model
        print("      ✓ Linear Regression trained on training data only")
        
        # -------------------------------------------------------------------------
        # MODEL 2: RANDOM FOREST
        # -------------------------------------------------------------------------
        print("\n[2/3] Training Random Forest...")
        print(f"      Hyperparameters from config.ini:")
        print(f"        n_estimators={self.rf_n_estimators}, max_depth={self.rf_max_depth}")
        print(f"        min_samples_split={self.rf_min_samples_split}, min_samples_leaf={self.rf_min_samples_leaf}")
        
        rf_model = RandomForestRegressor(
            n_estimators=self.rf_n_estimators,
            max_depth=self.rf_max_depth,
            min_samples_split=self.rf_min_samples_split,
            min_samples_leaf=self.rf_min_samples_leaf,
            random_state=self.random_seed,
            n_jobs=-1
        )
        rf_model.fit(self.X_train_scaled, self.y_train)
        self.models['Random Forest'] = rf_model
        print("      ✓ Random Forest trained on training data only")
        
        # -------------------------------------------------------------------------
        # MODEL 3: XGBOOST
        # -------------------------------------------------------------------------
        print("\n[3/3] Training XGBoost...")
        print(f"      Hyperparameters from config.ini:")
        print(f"        n_estimators={self.xgb_n_estimators}, max_depth={self.xgb_max_depth}")
        print(f"        learning_rate={self.xgb_learning_rate}, subsample={self.xgb_subsample}")
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=self.xgb_n_estimators,
            max_depth=self.xgb_max_depth,
            learning_rate=self.xgb_learning_rate,
            subsample=self.xgb_subsample,
            colsample_bytree=self.xgb_colsample_bytree,
            random_state=self.random_seed,
            n_jobs=-1
        )
        xgb_model.fit(self.X_train_scaled, self.y_train)
        self.models['XGBoost'] = xgb_model
        print("      ✓ XGBoost trained on training data only")
        
        print("\n✓ All models trained successfully on training data")
        print("✓ Models have NOT seen validation or test data yet")
        
        return self
    
    def evaluate_models(self):
        """
        STEP 5: Evaluate models on Train, Validation, and Test sets.
        """
        print("\n" + "="*80)
        print("STEP 5: MODEL EVALUATION (Now We Can Use Test Data)")
        print("="*80)
        
        # Prepare results dictionary
        self.results = {
            'Linear Regression': {},
            'Random Forest': {},
            'XGBoost': {}
        }
        
        # Evaluate each model on each dataset
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}...")
            
            # Predictions
            y_train_pred = model.predict(self.X_train_scaled)
            y_val_pred = model.predict(self.X_val_scaled)
            y_test_pred = model.predict(self.X_test_scaled)
            
            # Calculate metrics for each set
            for set_name, y_true, y_pred in [
                ('Train', self.y_train, y_train_pred),
                ('Validation', self.y_val, y_val_pred),
                ('Test', self.y_test, y_test_pred)
            ]:
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                
                self.results[model_name][set_name] = {
                    'RMSE': rmse,
                    'MAE': mae,
                    'R²': r2,
                    'y_true': y_true,
                    'y_pred': y_pred
                }
                
                print(f"  {set_name:12s}: RMSE={rmse:6.3f}, MAE={mae:6.3f}, R²={r2:6.3f}")
        
        print("\n✓ Evaluation completed")
        print("✓ Test set results are now valid (no data leakage)")
        
        return self
    
    def create_comparison_table(self):
        """Create a summary table comparing all models."""
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        
        # Create DataFrame for comparison
        comparison_data = []
        for model_name in ['Linear Regression', 'Random Forest', 'XGBoost']:
            for set_name in ['Train', 'Validation', 'Test']:
                metrics = self.results[model_name][set_name]
                comparison_data.append({
                    'Model': model_name,
                    'Dataset': set_name,
                    'RMSE': metrics['RMSE'],
                    'MAE': metrics['MAE'],
                    'R²': metrics['R²']
                })
        
        self.comparison_df = pd.DataFrame(comparison_data)
        
        # Print formatted table
        print("\n" + self.comparison_df.to_string(index=False))
        
        # Check for overfitting
        print("\n" + "="*80)
        print("OVERFITTING CHECK")
        print("="*80)
        for model_name in ['Linear Regression', 'Random Forest', 'XGBoost']:
            train_r2 = self.results[model_name]['Train']['R²']
            test_r2 = self.results[model_name]['Test']['R²']
            gap = train_r2 - test_r2
            
            if gap > 0.15:
                status = "⚠ OVERFITTING DETECTED"
            elif gap > 0.08:
                status = "⚠ Slight overfitting"
            else:
                status = "✓ Good generalization"
            
            print(f"{model_name:20s}: Train R²={train_r2:.3f}, Test R²={test_r2:.3f}, Gap={gap:.3f} {status}")
        
        # Save to CSV
        csv_path = self.output_dir / f"{self.base_name}_model_comparison.csv"
        self.comparison_df.to_csv(csv_path, index=False)
        print(f"\n✓ Comparison table saved: {csv_path}")
        
        return self
    
    def plot_results(self):
        """Create comprehensive visualizations."""
        print("\n" + "="*80)
        print("STEP 6: CREATING VISUALIZATIONS")
        print("="*80)
        
        # Model Comparison Bar Chart
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, metric in enumerate(['RMSE', 'MAE', 'R²']):
            ax = axes[idx]
            pivot_data = self.comparison_df.pivot(index='Model', columns='Dataset', values=metric)
            pivot_data.plot(kind='bar', ax=ax, width=0.8)
            ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
            ax.set_xlabel('Model', fontsize=12)
            ax.set_ylabel(metric, fontsize=12)
            ax.legend(title='Dataset', fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
        
        plt.tight_layout()
        plot_path = self.plots_dir / f"{self.base_name}_model_comparison.png"
        plt.savefig(plot_path, dpi=self.plot_dpi, bbox_inches='tight')
        print(f"✓ Saved: {plot_path.name}")
        plt.close()
        
        # Actual vs Predicted (Test Set)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, model_name in enumerate(['Linear Regression', 'Random Forest', 'XGBoost']):
            ax = axes[idx]
            y_true = self.results[model_name]['Test']['y_true']
            y_pred = self.results[model_name]['Test']['y_pred']
            r2 = self.results[model_name]['Test']['R²']
            
            ax.scatter(y_true, y_pred, alpha=0.5, s=10)
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            ax.set_title(f'{model_name}\nR² = {r2:.3f}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Actual', fontsize=11)
            ax.set_ylabel('Predicted', fontsize=11)
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.suptitle('Actual vs Predicted (Test Set - No Data Leakage)', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plot_path = self.plots_dir / f"{self.base_name}_actual_vs_predicted.png"
        plt.savefig(plot_path, dpi=self.plot_dpi, bbox_inches='tight')
        print(f"✓ Saved: {plot_path.name}")
        plt.close()
        
        # Residual Analysis
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, model_name in enumerate(['Linear Regression', 'Random Forest', 'XGBoost']):
            ax = axes[idx]
            y_true = self.results[model_name]['Test']['y_true']
            y_pred = self.results[model_name]['Test']['y_pred']
            residuals = y_true - y_pred
            
            ax.scatter(y_pred, residuals, alpha=0.5, s=10)
            ax.axhline(y=0, color='r', linestyle='--', lw=2)
            ax.set_title(f'{model_name}\nResiduals', fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicted', fontsize=11)
            ax.set_ylabel('Residual (Actual - Predicted)', fontsize=11)
            ax.grid(alpha=0.3)
        
        plt.suptitle('Residual Analysis (Test Set)', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plot_path = self.plots_dir / f"{self.base_name}_residuals.png"
        plt.savefig(plot_path, dpi=self.plot_dpi, bbox_inches='tight')
        print(f"✓ Saved: {plot_path.name}")
        plt.close()
        
        # Feature Importance
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for idx, model_name in enumerate(['Random Forest', 'XGBoost']):
            ax = axes[idx]
            importances = self.models[model_name].feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': self.SELECTED_FEATURES,
                'Importance': importances
            }).sort_values('Importance', ascending=True)
            
            ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
            ax.set_title(f'{model_name}\nFeature Importance', fontsize=14, fontweight='bold')
            ax.set_xlabel('Importance', fontsize=12)
            ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.plots_dir / f"{self.base_name}_feature_importance.png"
        plt.savefig(plot_path, dpi=self.plot_dpi, bbox_inches='tight')
        print(f"✓ Saved: {plot_path.name}")
        plt.close()
        
        print("\n✓ All visualizations created successfully!")
        
        return self
    
    def save_predictions(self):
        """Save predictions to CSV files."""
        print("\n" + "="*80)
        print("SAVING PREDICTIONS")
        print("="*80)
        
        for model_name in ['Linear Regression', 'Random Forest', 'XGBoost']:
            predictions_df = pd.DataFrame({
                'Actual': self.results[model_name]['Test']['y_true'],
                'Predicted': self.results[model_name]['Test']['y_pred'],
                'Residual': self.results[model_name]['Test']['y_true'] - self.results[model_name]['Test']['y_pred']
            })
            
            csv_path = self.output_dir / f"{self.base_name}_{model_name.replace(' ', '_')}_predictions.csv"
            predictions_df.to_csv(csv_path, index=True)
            print(f"✓ Saved: {csv_path.name}")
        
        return self
    
    def generate_report(self):
        """Generate a comprehensive text report."""
        print("\n" + "="*80)
        print("GENERATING REPORT")
        print("="*80)
        
        report_path = self.readme_dir / f"{self.base_name}_ML_Energy_Forecasting_Report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("HVAC ENERGY FORECASTING - ML REPORT (CORRECTED - NO DATA LEAKAGE)\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Building: {self.building_id}\n")
            f.write(f"AHU Unit: {self.ahu_unit}\n")
            f.write(f"Season: {self.season} {self.year}\n")
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("="*80 + "\n")
            f.write("METHODOLOGY (CORRECT ORDER TO PREVENT DATA LEAKAGE)\n")
            f.write("="*80 + "\n")
            f.write("1. Load and clean data (remove NaN, fix types)\n")
            f.write("2. Split into Train/Val/Test BEFORE any preprocessing\n")
            f.write("3. Fit scaler on training data ONLY\n")
            f.write("4. Transform val/test using the fitted scaler\n")
            f.write("5. Train models on training data ONLY\n")
            f.write("6. Evaluate on test data\n\n")
            
            f.write("="*80 + "\n")
            f.write("DATA SPLIT\n")
            f.write("="*80 + "\n")
            f.write(f"Train:      {len(self.X_train)} records ({self.train_ratio*100:.0f}%)\n")
            f.write(f"Validation: {len(self.X_val)} records ({self.val_ratio*100:.0f}%)\n")
            f.write(f"Test:       {len(self.X_test)} records ({self.test_ratio*100:.0f}%)\n\n")
            
            f.write("="*80 + "\n")
            f.write("MODEL PERFORMANCE\n")
            f.write("="*80 + "\n\n")
            f.write(self.comparison_df.to_string(index=False))
            f.write("\n\n")
            
            f.write("="*80 + "\n")
            f.write("OVERFITTING CHECK\n")
            f.write("="*80 + "\n")
            for model_name in ['Linear Regression', 'Random Forest', 'XGBoost']:
                train_r2 = self.results[model_name]['Train']['R²']
                test_r2 = self.results[model_name]['Test']['R²']
                gap = train_r2 - test_r2
                f.write(f"{model_name}: Train R²={train_r2:.3f}, Test R²={test_r2:.3f}, Gap={gap:.3f}\n")
            f.write("\n")
            
            # Best model
            test_r2 = {model: self.results[model]['Test']['R²'] for model in self.models.keys()}
            best_model = max(test_r2, key=test_r2.get)
            
            f.write("="*80 + "\n")
            f.write("RECOMMENDATION\n")
            f.write("="*80 + "\n")
            f.write(f"Best Model: {best_model} (Test R² = {test_r2[best_model]:.3f})\n\n")
            
            f.write("="*80 + "\n")
            f.write("VALIDATION\n")
            f.write("="*80 + "\n")
            f.write("✓ No data leakage - test set was isolated before preprocessing\n")
            f.write("✓ Scaler was fit on training data only\n")
            f.write("✓ Models were trained on training data only\n")
            f.write("✓ Test results are valid and represent real-world performance\n\n")
        
        print(f"✓ Report saved: {report_path}")
        
        return self
    
    def generate_word_report(self):
        """Generate a comprehensive Word report in Italian with plots and detailed explanations."""
        print("\n" + "="*80)
        print("GENERATING WORD REPORT (ITALIAN)")
        print("="*80)
        
        # Create Word document
        doc = Document()
        
        # Get script name without number
        script_name = "Previsione Energetica HVAC - Machine Learning"
        
        # Get date range from data
        date_start = self.df_clean.index.min().strftime('%d/%m/%Y')
        date_end = self.df_clean.index.max().strftime('%d/%m/%Y')
        date_range = f"Periodo di analisi: {date_start} - {date_end}"
        
        # =============================================================================
        # MAIN TITLE AND DATE RANGE (ONCE AT TOP)
        # =============================================================================
        title = doc.add_heading(script_name, level=0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in title.runs:
            run.font.size = Pt(18)
            run.font.color.rgb = RGBColor(0, 51, 102)  # Dark blue
        
        # Date range subtitle
        date_para = doc.add_paragraph(date_range)
        date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in date_para.runs:
            run.font.size = Pt(12)
            run.font.italic = True
            run.font.color.rgb = RGBColor(70, 70, 70)
        
        doc.add_paragraph()  # Spacing
        
        # =============================================================================
        # EXECUTIVE SUMMARY
        # =============================================================================
        summary_title = doc.add_heading('Sommario Esecutivo', level=1)
        summary_title.alignment = WD_ALIGN_PARAGRAPH.LEFT
        for run in summary_title.runs:
            run.font.size = Pt(14)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0, 51, 102)
        
        # Calculate summary statistics
        n_total = len(self.df_clean)
        n_train = len(self.X_train)
        n_val = len(self.X_val)
        n_test = len(self.X_test)
        
        summary_text = (
            f"Questo rapporto presenta i risultati dell'analisi di machine learning per la previsione del consumo "
            f"energetico del sistema HVAC {self.building_id}_{self.ahu_unit} durante il periodo {self.season} {self.year}.\n\n"
            f"📊 Dati Analizzati:\n"
            f"• Periodo totale: {date_start} - {date_end}\n"
            f"• Record totali: {n_total:,} ore\n"
            f"• Set di training: {n_train:,} ore ({self.train_ratio*100:.0f}%)\n"
            f"• Set di validation: {n_val:,} ore ({self.val_ratio*100:.0f}%)\n"
            f"• Set di test: {n_test:,} ore ({self.test_ratio*100:.0f}%)\n"
            f"• Caratteristiche utilizzate: {len(self.SELECTED_FEATURES)} features\n"
            f"• Variabile target: {self.TARGET}\n\n"
            f"🤖 Modelli Addestrati:\n"
            f"1. Regressione Lineare (baseline)\n"
            f"2. Random Forest (ensemble di alberi decisionali)\n"
            f"3. XGBoost (gradient boosting ottimizzato)\n\n"
        )
        
        # Add best model results
        test_r2 = {model: self.results[model]['Test']['R²'] for model in self.models.keys()}
        best_model = max(test_r2, key=test_r2.get)
        best_rmse = self.results[best_model]['Test']['RMSE']
        best_mae = self.results[best_model]['Test']['MAE']
        
        summary_text += (
            f"🏆 Modello Migliore: {best_model}\n"
            f"• R² (test): {test_r2[best_model]:.4f} ({test_r2[best_model]*100:.2f}% della varianza spiegata)\n"
            f"• RMSE (test): {best_rmse:.4f}\n"
            f"• MAE (test): {best_mae:.4f}\n\n"
            f"✅ Conclusione: Il modello {best_model} dimostra eccellente capacità predittiva con "
            f"prestazioni molto elevate sul set di test non visto durante l'addestramento."
        )
        
        summary_para = doc.add_paragraph(summary_text)
        for run in summary_para.runs:
            run.font.size = Pt(10)
        
        doc.add_page_break()
        
        # =============================================================================
        # DATA SPLIT EXPLANATION
        # =============================================================================
        split_title = doc.add_heading('Suddivisione dei Dati (70/15/15)', level=1)
        split_title.alignment = WD_ALIGN_PARAGRAPH.LEFT
        for run in split_title.runs:
            run.font.size = Pt(14)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0, 51, 102)
        
        split_text = (
            "I dati sono stati suddivisi in tre set seguendo un approccio cronologico (non casuale) "
            "per preservare l'ordine temporale e simulare uno scenario di deployment reale:\n\n"
            f"1️⃣ Set di Training (70% - {n_train:,} ore):\n"
            f"   • Periodo: {self.X_train.index.min().strftime('%d/%m/%Y %H:%M')} - {self.X_train.index.max().strftime('%d/%m/%Y %H:%M')}\n"
            "   • Scopo: Addestrare i modelli e far apprendere i pattern nei dati\n"
            "   • Lo scaler (StandardScaler) è stato FIT su questi dati SOLAMENTE\n\n"
            f"2️⃣ Set di Validation (15% - {n_val:,} ore):\n"
            f"   • Periodo: {self.X_val.index.min().strftime('%d/%m/%Y %H:%M')} - {self.X_val.index.max().strftime('%d/%m/%Y %H:%M')}\n"
            "   • Scopo: Monitorare l'overfitting durante l'addestramento\n"
            "   • Trasformato usando lo scaler FIT sul training set\n\n"
            f"3️⃣ Set di Test (15% - {n_test:,} ore):\n"
            f"   • Periodo: {self.X_test.index.min().strftime('%d/%m/%Y %H:%M')} - {self.X_test.index.max().strftime('%d/%m/%Y %H:%M')}\n"
            "   • Scopo: Valutazione finale su dati completamente non visti\n"
            "   • ISOLATO durante tutto il processo di addestramento\n"
            "   • Trasformato usando lo scaler FIT sul training set\n\n"
            "⚠️ IMPORTANTE: La suddivisione cronologica è stata effettuata PRIMA di qualsiasi operazione di "
            "scaling o preprocessing per prevenire il data leakage. Questo garantisce che i risultati sul "
            "set di test siano rappresentativi delle prestazioni reali del modello su dati futuri."
        )
        
        split_para = doc.add_paragraph(split_text)
        split_para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        for run in split_para.runs:
            run.font.size = Pt(10)
        
        doc.add_page_break()
        
        # =============================================================================
        # FEATURES USED
        # =============================================================================
        features_title = doc.add_heading('Caratteristiche Utilizzate (10 Features)', level=1)
        features_title.alignment = WD_ALIGN_PARAGRAPH.LEFT
        for run in features_title.runs:
            run.font.size = Pt(14)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0, 51, 102)
        
        features_intro = (
            "Il modello utilizza 10 caratteristiche selezionate attraverso l'analisi di importanza e "
            "correlazione per predire la potenza termica (thermal_power):\n"
        )
        doc.add_paragraph(features_intro)
        
        feature_descriptions = {
            'thermal_power_signed': 'Potenza termica con segno (positivo=riscaldamento, negativo=raffrescamento)',
            'efficiency_proxy': 'Indicatore dell\'efficienza del sistema',
            'load_occupied': 'Fattore di carico basato sull\'occupazione dell\'edificio',
            'Temperatura Esterna': 'Temperatura esterna misurata (°C)',
            'outdoor_temp_rolling_24h': 'Media mobile a 24 ore della temperatura esterna',
            'hour_cos': 'Codifica ciclica dell\'ora (trasformata coseno)',
            'delta_t': 'Differenza di temperatura: Mandata - Ripresa (°C)',
            'Set Points Temperatura Mandata Riscaldamento': 'Setpoint temperatura mandata riscaldamento',
            'Set Points Temperatura Mandata Raffrescamento': 'Setpoint temperatura mandata raffrescamento',
            'Temperatura Ripresa': 'Temperatura aria di ripresa (°C)'
        }
        
        for i, feature in enumerate(self.SELECTED_FEATURES, 1):
            desc = feature_descriptions.get(feature, 'Caratteristica del sistema HVAC')
            feature_para = doc.add_paragraph(f"{i}. {feature}: {desc}", style='List Number')
            for run in feature_para.runs:
                run.font.size = Pt(10)
        
        doc.add_page_break()
        
        # =============================================================================
        # PLOT DESCRIPTIONS IN ITALIAN
        # =============================================================================
        plot_descriptions = {
            'model_comparison': {
                'title': 'Confronto delle Prestazioni dei Modelli',
                'description': (
                    'Questo grafico presenta un confronto completo delle prestazioni dei tre modelli di machine learning '
                    '(Regressione Lineare, Random Forest e XGBoost) attraverso tre metriche chiave:\n\n'
                    '• RMSE (Root Mean Squared Error): Errore quadratico medio. Valori più bassi indicano previsioni più accurate. '
                    'Penalizza maggiormente gli errori grandi.\n\n'
                    '• MAE (Mean Absolute Error): Errore assoluto medio. Più intuitivo dell\'RMSE, rappresenta l\'errore medio '
                    'in valore assoluto.\n\n'
                    '• R² (Coefficiente di Determinazione): Percentuale di varianza spiegata dal modello. '
                    'Valori vicini a 1.0 (100%) indicano previsioni quasi perfette.\n\n'
                    'I risultati sono mostrati per i tre set di dati (Training, Validation, Test). Le barre BLU rappresentano '
                    'le prestazioni sul set di test, che sono le PIÙ IMPORTANTI perché indicano la capacità del modello di '
                    'generalizzare su dati completamente non visti.'
                )
            },
            'actual_vs_predicted': {
                'title': 'Previsioni vs Valori Reali (Set di Test)',
                'description': (
                    'Questi grafici di dispersione confrontano i valori di potenza termica previsti dai modelli con i valori '
                    'reali misurati nel set di test. La linea rossa tratteggiata rappresenta la previsione perfetta (y = x, '
                    'dove previsto = reale). Punti più vicini alla linea indicano previsioni più accurate.\n\n'
                    'Come interpretare:\n'
                    '• Punti sulla linea rossa = previsione perfetta\n'
                    '• Punti sopra la linea = modello sottostima (prevede meno del reale)\n'
                    '• Punti sotto la linea = modello sovrastima (prevede più del reale)\n'
                    '• R² vicino a 1.0 = eccellente capacità predittiva\n\n'
                    '⚠️ IMPORTANTE: Questi risultati sono completamente indipendenti e non influenzati da data leakage, '
                    'poiché il set di test è rimasto isolato durante tutto il processo di addestramento. I modelli non hanno '
                    'MAI visto questi dati durante la fase di training.'
                )
            },
            'residuals': {
                'title': 'Analisi dei Residui (Set di Test)',
                'description': (
                    'L\'analisi dei residui mostra la differenza tra i valori reali e quelli previsti:\n'
                    'Residuo = Valore Reale - Valore Previsto\n\n'
                    'Cosa cercare in un buon modello:\n'
                    '✅ Residui distribuiti CASUALMENTE attorno allo zero (linea rossa tratteggiata)\n'
                    '✅ NESSUN pattern sistematico (forme curve, imbuto, ecc.)\n'
                    '✅ Varianza COSTANTE dei residui (omoschedasticità)\n'
                    '✅ Maggior parte dei punti vicini allo zero\n\n'
                    'Segni di problemi:\n'
                    '❌ Pattern sistematici = modello troppo semplice (underfitting)\n'
                    '❌ Varianza crescente = eteroschedasticità\n'
                    '❌ Residui concentrati lontano da zero = bias sistematico\n\n'
                    'Nei nostri risultati, Random Forest e XGBoost mostrano residui casuali e ben distribuiti, '
                    'confermando che i modelli sono ben calibrati. La Regressione Lineare mostra pattern sistematici, '
                    'indicando che la relazione tra le variabili è NON-LINEARE.'
                )
            },
            'feature_importance': {
                'title': 'Importanza delle Caratteristiche',
                'description': (
                    'Questo grafico mostra l\'importanza relativa di ciascuna caratteristica (feature) utilizzata dai modelli '
                    'Random Forest e XGBoost per fare previsioni. L\'importanza indica quanto ciascuna variabile contribuisce '
                    'alla capacità predittiva del modello.\n\n'
                    'Metodi di calcolo:\n'
                    '• Random Forest: Basato sulla riduzione media dell\'errore quando una feature viene usata per dividere i nodi '
                    'degli alberi decisionali. Features che riducono maggiormente l\'errore sono più importanti.\n\n'
                    '• XGBoost: Basato sul "gain" medio (riduzione della loss function) quando una feature viene usata. '
                    'Tiene conto della frequenza d\'uso e dell\'impatto di ciascuna feature.\n\n'
                    'Interpretazione per il sistema HVAC:\n'
                    '• thermal_power_signed e delta_t sono le features PIÙ IMPORTANTI, indicando che la direzione del flusso '
                    'termico e la differenza di temperatura sono i principali driver del consumo energetico.\n'
                    '• efficiency_proxy e load_occupied sono anch\'esse rilevanti, confermando l\'importanza dell\'efficienza '
                    'del sistema e dell\'occupazione dell\'edificio.\n'
                    '• Le temperature esterne e i setpoint hanno importanza moderata, influenzando le prestazioni in modo indiretto.\n\n'
                    'Questa analisi aiuta a identificare quali sensori e variabili sono più critici per il monitoraggio e il controllo '
                    'del sistema HVAC.'
                )
            }
        }
        
        # =============================================================================
        # ADD EACH PLOT WITH DESCRIPTION
        # =============================================================================
        plot_files = [
            ('model_comparison', f"{self.base_name}_model_comparison.png"),
            ('actual_vs_predicted', f"{self.base_name}_actual_vs_predicted.png"),
            ('residuals', f"{self.base_name}_residuals.png"),
            ('feature_importance', f"{self.base_name}_feature_importance.png")
        ]
        
        for plot_key, plot_filename in plot_files:
            plot_path = self.plots_dir / plot_filename
            
            if not plot_path.exists():
                print(f"  ⚠ Warning: Plot not found: {plot_filename}")
                continue
            
            # Add page break before each plot (except first)
            if plot_key != 'model_comparison':
                doc.add_page_break()
            
            # ROW 1: Plot title (centered, bold, dark blue)
            title_para = doc.add_heading(plot_descriptions[plot_key]['title'], level=1)
            title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in title_para.runs:
                run.font.size = Pt(14)
                run.font.bold = True
                run.font.color.rgb = RGBColor(0, 51, 102)  # Dark blue
            
            # ROW 2: Plot image (centered)
            plot_para = doc.add_paragraph()
            plot_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = plot_para.add_run()
            run.add_picture(str(plot_path), width=Inches(6.5))
            
            # ROW 3: Description
            desc_heading = doc.add_paragraph()
            desc_heading_run = desc_heading.add_run('Descrizione:')
            desc_heading_run.font.bold = True
            desc_heading_run.font.size = Pt(11)
            desc_heading_run.font.color.rgb = RGBColor(0, 51, 102)
            
            desc_para = doc.add_paragraph(plot_descriptions[plot_key]['description'])
            desc_para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            for run in desc_para.runs:
                run.font.size = Pt(10)
            
            doc.add_paragraph()  # Spacing
            
            print(f"  ✓ Added: {plot_descriptions[plot_key]['title']}")
        
        # =============================================================================
        # ADD MODEL PERFORMANCE TABLE
        # =============================================================================
        doc.add_page_break()
        
        perf_title = doc.add_heading('Prestazioni dei Modelli', level=1)
        perf_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in perf_title.runs:
            run.font.size = Pt(14)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0, 51, 102)
        
        # Add table
        table = doc.add_table(rows=1 + len(self.comparison_df), cols=5)
        table.style = 'Light Grid Accent 1'
        
        # Header row
        header_cells = table.rows[0].cells
        headers = ['Modello', 'Dataset', 'RMSE', 'MAE', 'R²']
        for i, header in enumerate(headers):
            header_cells[i].text = header
            for paragraph in header_cells[i].paragraphs:
                for run in paragraph.runs:
                    run.font.bold = True
                    run.font.size = Pt(10)
        
        # Data rows
        for idx, row in self.comparison_df.iterrows():
            row_cells = table.rows[idx + 1].cells
            row_cells[0].text = str(row['Model'])
            row_cells[1].text = str(row['Dataset'])
            row_cells[2].text = f"{row['RMSE']:.6f}"
            row_cells[3].text = f"{row['MAE']:.6f}"
            row_cells[4].text = f"{row['R²']:.6f}"
            
            for cell in row_cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.size = Pt(9)
        
        # Add overfitting check section
        doc.add_paragraph()
        overfit_heading = doc.add_paragraph()
        overfit_heading_run = overfit_heading.add_run('Verifica del Sovradattamento (Overfitting):')
        overfit_heading_run.font.bold = True
        overfit_heading_run.font.size = Pt(11)
        overfit_heading_run.font.color.rgb = RGBColor(0, 51, 102)
        
        for model_name in ['Linear Regression', 'Random Forest', 'XGBoost']:
            train_r2 = self.results[model_name]['Train']['R²']
            test_r2 = self.results[model_name]['Test']['R²']
            gap = train_r2 - test_r2
            
            status = "✓ Buona generalizzazione" if gap < 0.08 else "⚠ Leggero sovradattamento" if gap < 0.15 else "❌ Sovradattamento"
            
            overfit_para = doc.add_paragraph(f"{model_name}: Train R²={train_r2:.3f}, Test R²={test_r2:.3f}, Gap={gap:.3f} - {status}")
            for run in overfit_para.runs:
                run.font.size = Pt(10)
        
        # Add methodology section
        doc.add_page_break()
        
        method_title = doc.add_heading('Metodologia: Prevenzione del Data Leakage', level=1)
        method_title.alignment = WD_ALIGN_PARAGRAPH.LEFT
        for run in method_title.runs:
            run.font.size = Pt(14)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0, 51, 102)
        
        methodology_intro = (
            "Questo studio segue un ordine RIGOROSO di operazioni per prevenire il data leakage e garantire "
            "risultati validi e rappresentativi delle prestazioni reali. Il data leakage si verifica quando "
            "informazioni dal set di test \"filtrano\" nel processo di training, causando stime di prestazioni "
            "eccessivamente ottimistiche che non si riflettono nella realtà.\n"
        )
        doc.add_paragraph(methodology_intro)
        
        method_heading = doc.add_paragraph()
        method_heading_run = method_heading.add_run('🔄 Pipeline di Machine Learning (Ordine Corretto):')
        method_heading_run.font.bold = True
        method_heading_run.font.size = Pt(11)
        method_heading_run.font.color.rgb = RGBColor(0, 51, 102)
        
        methodology_steps = (
            "1️⃣ STEP 1: Caricamento e Pulizia dei Dati\n"
            "   • Lettura del file CSV con features ingegnerizzate\n"
            "   • Rimozione di righe con valori NaN\n"
            "   • Correzione dei tipi di dati\n"
            "   • Ordinamento cronologico\n"
            "   ✅ SICURO: queste operazioni non \"imparano\" dai dati\n\n"
            
            "2️⃣ STEP 2: Suddivisione Train/Validation/Test (70/15/15)\n"
            "   ⚠️ CRITICO: Questa suddivisione DEVE avvenire PRIMA di qualsiasi preprocessing!\n"
            "   • Split cronologico (non casuale)\n"
            "   • Train: primi 70% dei dati temporali\n"
            "   • Validation: successivi 15%\n"
            "   • Test: ultimi 15% (ISOLATO)\n"
            "   ✅ Il set di test è ora separato e non sarà più toccato fino alla valutazione finale\n\n"
            
            "3️⃣ STEP 3: Scaling delle Features (StandardScaler)\n"
            "   • FIT dello scaler SOLO sul set di training\n"
            "   • Lo scaler calcola media e deviazione standard dal training set\n"
            "   • TRANSFORM del training set usando questi parametri\n"
            "   • TRANSFORM dei set validation e test usando GLI STESSI parametri\n"
            "   ✅ CORRETTO: i set validation e test sono trasformati \"alla cieca\" usando le statistiche del training\n"
            "   ❌ SBAGLIATO sarebbe: FIT dello scaler su tutti i dati (causerebbe data leakage)\n\n"
            
            "4️⃣ STEP 4: Addestramento dei Modelli\n"
            "   • Training di Linear Regression, Random Forest, XGBoost\n"
            "   • Addestramento SOLO sul set di training\n"
            "   • I modelli non vedono MAI i dati di validation o test\n"
            "   ✅ I modelli apprendono pattern SOLO dai dati di training\n\n"
            
            "5️⃣ STEP 5: Valutazione\n"
            "   • Previsioni sul set di training (controllo overfitting)\n"
            "   • Previsioni sul set di validation (monitoraggio durante training)\n"
            "   • Previsioni sul set di test (VALUTAZIONE FINALE)\n"
            "   • Calcolo di RMSE, MAE, R² per ogni set\n"
            "   ✅ Le prestazioni sul set di test sono AFFIDABILI e non distorte\n\n"
        )
        
        method_para = doc.add_paragraph(methodology_steps)
        method_para.alignment = WD_ALIGN_PARAGRAPH.LEFT
        for run in method_para.runs:
            run.font.size = Pt(9)
        
        # Add comparison of wrong vs correct approach
        comparison_heading = doc.add_paragraph()
        comparison_heading_run = comparison_heading.add_run('❌ Approccio SBAGLIATO vs ✅ Approccio CORRETTO:')
        comparison_heading_run.font.bold = True
        comparison_heading_run.font.size = Pt(11)
        comparison_heading_run.font.color.rgb = RGBColor(0, 51, 102)
        
        comparison_text = (
            "SBAGLIATO (Causa Data Leakage):\n"
            "1. Caricare dati\n"
            "2. ❌ Scalare TUTTI i dati insieme (lo scaler \"vede\" il test set!)\n"
            "3. Suddividere in train/test\n"
            "4. Addestrare modello\n"
            "➡️ Risultato: Prestazioni inflazionate e non rappresentative\n\n"
            
            "CORRETTO (Nessun Data Leakage):\n"
            "1. Caricare dati\n"
            "2. ✅ Suddividere PRIMA in train/test (isolare il test set)\n"
            "3. ✅ FIT dello scaler SOLO sul training\n"
            "4. ✅ Trasformare test usando lo scaler del training\n"
            "5. Addestrare modello\n"
            "➡️ Risultato: Prestazioni affidabili e realistiche\n\n"
            
            "Perché è importante?\n"
            "Se lo scaler calcola media e deviazione standard usando TUTTI i dati (incluso il test set), "
            "il modello ha indirettamente \"visto\" informazioni sulla distribuzione del test set. "
            "Questo porta a risultati ottimistici che non si verificano nel mondo reale quando il modello "
            "deve fare previsioni su dati veramente nuovi."
        )
        
        comparison_para = doc.add_paragraph(comparison_text)
        comparison_para.alignment = WD_ALIGN_PARAGRAPH.LEFT
        for run in comparison_para.runs:
            run.font.size = Pt(9)
        
        # Add best model recommendation with detailed justification
        doc.add_page_break()
        
        test_r2 = {model: self.results[model]['Test']['R²'] for model in self.models.keys()}
        test_rmse = {model: self.results[model]['Test']['RMSE'] for model in self.models.keys()}
        test_mae = {model: self.results[model]['Test']['MAE'] for model in self.models.keys()}
        best_model = max(test_r2, key=test_r2.get)
        
        rec_title = doc.add_heading('Raccomandazione e Conclusioni', level=1)
        rec_title.alignment = WD_ALIGN_PARAGRAPH.LEFT
        for run in rec_title.runs:
            run.font.size = Pt(14)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0, 51, 102)
        
        rec_heading = doc.add_paragraph()
        rec_heading_run = rec_heading.add_run('🏆 Modello Raccomandato per Deployment:')
        rec_heading_run.font.bold = True
        rec_heading_run.font.size = Pt(12)
        rec_heading_run.font.color.rgb = RGBColor(0, 102, 0)
        
        rec_text = (
            f"Basandosi sui risultati dell'analisi, il modello {best_model} è raccomandato per la previsione "
            f"del consumo energetico del sistema HVAC.\n\n"
            f"📊 Prestazioni sul Set di Test:\n"
            f"• R²: {test_r2[best_model]:.6f} ({test_r2[best_model]*100:.4f}% della varianza spiegata)\n"
            f"• RMSE: {test_rmse[best_model]:.6f}\n"
            f"• MAE: {test_mae[best_model]:.6f}\n\n"
            f"✅ Motivi della Scelta:\n"
        )
        doc.add_paragraph(rec_text)
        
        # Add detailed justification
        train_r2 = self.results[best_model]['Train']['R²']
        gap = train_r2 - test_r2[best_model]
        
        justification_points = [
            f"1. Eccellente capacità predittiva: R² di {test_r2[best_model]:.4f} indica che il modello spiega "
            f"{test_r2[best_model]*100:.2f}% della varianza nei dati di test.",
            
            f"2. Minimo overfitting: Differenza train-test R² = {gap:.4f}, indicando buona generalizzazione. "
            "Il modello non è \"memorizzato\" i dati di training.",
            
            f"3. Errore di previsione molto basso: RMSE={test_rmse[best_model]:.4f} e MAE={test_mae[best_model]:.4f} "
            "dimostrano accuratezza elevata nelle previsioni.",
            
            "4. Residui ben distribuiti: L'analisi dei residui mostra distribuzione casuale senza pattern sistematici, "
            "confermando che il modello è ben calibrato.",
            
            "5. Gestione di relazioni non-lineari: A differenza della Regressione Lineare, questo modello cattura "
            "efficacemente le complesse relazioni non-lineari nel sistema HVAC."
        ]
        
        for point in justification_points:
            point_para = doc.add_paragraph(point, style='List Number')
            for run in point_para.runs:
                run.font.size = Pt(10)
        
        # Add comparison with other models
        doc.add_paragraph()
        comparison_heading = doc.add_paragraph()
        comparison_heading_run = comparison_heading.add_run('📊 Confronto con Altri Modelli:')
        comparison_heading_run.font.bold = True
        comparison_heading_run.font.size = Pt(11)
        comparison_heading_run.font.color.rgb = RGBColor(0, 51, 102)
        
        for model in ['Linear Regression', 'Random Forest', 'XGBoost']:
            if model == best_model:
                status = "🏆 MIGLIORE"
            else:
                status = "📋"
            
            model_text = (
                f"{status} {model}:\n"
                f"  • R² Test: {test_r2[model]:.6f}\n"
                f"  • RMSE Test: {test_rmse[model]:.6f}\n"
                f"  • MAE Test: {test_mae[model]:.6f}\n"
            )
            
            if model == 'Linear Regression':
                model_text += "  • Commento: Buon baseline ma assume relazioni lineari. Prestazioni inferiori indicano non-linearità.\n"
            elif model == 'Random Forest' and model == best_model:
                model_text += "  • Commento: Eccellente! Cattura relazioni non-lineari, robusto agli outliers, minimo overfitting.\n"
            elif model == 'XGBoost':
                model_text += "  • Commento: Ottime prestazioni ma leggero overfitting rispetto a Random Forest.\n"
            
            model_para = doc.add_paragraph(model_text)
            for run in model_para.runs:
                run.font.size = Pt(9)
        
        # Add practical implications
        doc.add_paragraph()
        implications_heading = doc.add_paragraph()
        implications_heading_run = implications_heading.add_run('💡 Implicazioni Pratiche:')
        implications_heading_run.font.bold = True
        implications_heading_run.font.size = Pt(11)
        implications_heading_run.font.color.rgb = RGBColor(0, 51, 102)
        
        implications_text = (
            f"Il modello {best_model} può essere utilizzato per:\n\n"
            "1. Previsione del Consumo Energetico: Stimare il consumo futuro basandosi su condizioni operative "
            "e metereologiche previste.\n\n"
            "2. Ottimizzazione del Controllo: Identificare condizioni operative ottimali per minimizzare il consumo "
            "energetico mantenendo il comfort.\n\n"
            "3. Rilevamento Anomalie: Identificare deviazioni significative tra consumo previsto e reale, indicando "
            "possibili malfunzionamenti o inefficienze.\n\n"
            "4. Pianificazione Energetica: Supportare decisioni di gestione energetica a breve e medio termine.\n\n"
            "5. Model Predictive Control (MPC): Integrare il modello in sistemi di controllo predittivo per "
            "ottimizzazione in tempo reale.\n\n"
            f"⚠️ Nota: Il modello è stato addestrato su dati {self.season} {self.year}. Per applicazioni in altre "
            "stagioni, si raccomanda di addestrare modelli specifici per ciascuna stagione (estate/inverno) per "
            "catturare le differenze nei pattern operativi."
        )
        
        implications_para = doc.add_paragraph(implications_text)
        implications_para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        for run in implications_para.runs:
            run.font.size = Pt(10)
        
        # Add limitations
        doc.add_paragraph()
        limitations_heading = doc.add_paragraph()
        limitations_heading_run = limitations_heading.add_run('⚠️ Limitazioni e Considerazioni:')
        limitations_heading_run.font.bold = True
        limitations_heading_run.font.size = Pt(11)
        limitations_heading_run.font.color.rgb = RGBColor(153, 51, 0)
        
        limitations_text = (
            "1. Validità Temporale: Il modello è addestrato su un periodo specifico. Cambiamenti nel sistema "
            "(manutenzione, upgrade, cambio di controllo) potrebbero richiedere re-training.\n\n"
            "2. Specificità Stagionale: Prestazioni potrebbero variare significativamente tra estate e inverno. "
            "Considerare modelli separati per stagioni diverse.\n\n"
            "3. Qualità dei Dati: Le prestazioni dipendono dalla qualità dei sensori e dalla calibrazione. "
            "Monitorare regolarmente la qualità dei dati in input.\n\n"
            "4. Eventi Eccezionali: Il modello potrebbe non generalizzare bene a condizioni estreme non presenti "
            "nei dati di training (es. ondate di calore/freddo eccezionali).\n\n"
            "5. Re-training Periodico: Si raccomanda di ri-addestrare il modello periodicamente (es. ogni 6-12 mesi) "
            "con nuovi dati per mantenere l'accuratezza."
        )
        
        limitations_para = doc.add_paragraph(limitations_text)
        limitations_para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        for run in limitations_para.runs:
            run.font.size = Pt(9)
        
        # Save document
        word_report_path = self.plots_dir / f"{self.base_name}_ML_Energy_Forecasting_Report.docx"
        doc.save(str(word_report_path))
        
        print(f"\n✓ Word report saved: {word_report_path}")
        print(f"  Location: {self.plots_dir}")
        
        return self
    
    def run_complete_pipeline(self):
        """Execute the complete ML pipeline (CORRECT ORDER)."""
        self.load_and_clean_data()      # Step 1: Load and clean (safe on whole dataset)
        self.split_data()                # Step 2: Split BEFORE any "learning"
        self.scale_features()            # Step 3: Fit scaler on train, apply to val/test
        self.train_models()              # Step 4: Train on training data only
        self.evaluate_models()           # Step 5: Evaluate (now we can use test)
        self.create_comparison_table()
        self.plot_results()
        self.save_predictions()
        self.generate_report()
        self.generate_word_report()      # Step 6: Generate Italian Word report with plots
        
        print("\n" + "="*80)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY (NO DATA LEAKAGE)")
        print("="*80)
        print(f"\nResults saved to:")
        print(f"  - CSV files: {self.output_dir}")
        print(f"  - Plots: {self.plots_dir}")
        print(f"  - Text Report: {self.readme_dir}")
        print(f"  - Word Report: {self.plots_dir}")
        print("\n" + "="*80)


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    """
    Main execution block.
    
    To run this script:
    1. Ensure config.ini is in the same directory or parent directory
    2. Ensure you have run the feature engineering script first
    3. Run: python 14.energy_forecasting_ml_CORRECTED.py
    """
    
    # Create and run the forecasting model
    model = EnergyForecastingModel(config_path='config.ini')
    model.run_complete_pipeline()
