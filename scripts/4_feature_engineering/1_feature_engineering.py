"""
HVAC Feature Engineering Script
Creates derived features from interpolated HVAC data based on config.ini settings
"""

import pandas as pd
import numpy as np
from pathlib import Path
import configparser
from datetime import datetime
import warnings
import sys
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
except ImportError:
    Document = None

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')


class FeatureEngineer:
    """Creates derived features for HVAC analysis"""
    
    def __init__(self, config_path='config.ini'):
        """Initialize with config file path"""
        # Allow inline comments in config values (e.g. "1  ; comment")
        self.config = configparser.ConfigParser(inline_comment_prefixes=(';', '#'))
        
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
        """Load settings from config file"""
        # Global settings
        self.building_id = self.config.get('global', 'building_id')
        self.ahu_unit = self.config.get('global', 'ahu_unit')
        self.season = self.config.get('global', 'season')
        self.year = self.config.get('global', 'year')
        
        # Feature engineering settings
        fe_section = 'feature_engineering'
        self.enable_fe = self.config.getboolean(fe_section, 'enable_feature_engineering', fallback=True)
        
        # --- START OF AUTOMATED SEASON SELECTION ---
        # 1. Get the current season from the config (global setting)
        current_season = self.season.strip().lower()
        
        # 2. Check if feature_set is 'custom' to enable automated selection
        feature_set = self.config.get(fe_section, 'feature_set', fallback='all').strip().lower()
        
        if feature_set == 'custom':
            # 3. Automatically pick the correct feature list
            if 'summer' in current_season:
                print(f"🌞 Detecting Summer Season: Loading Summer feature set...")
                feature_str = self.config.get(fe_section, 'custom_features_summer')
            else:
                print(f"❄️ Detecting Winter Season: Loading Winter feature set...")
                feature_str = self.config.get(fe_section, 'custom_features_winter')
                
            # 4. Convert comma-separated string to a list
            self.custom_features = [f.strip() for f in feature_str.split(',')]
            print(f"   Selected features: {', '.join(self.custom_features)}")
        else:
            self.custom_features = []
        # --- END OF AUTOMATED SEASON SELECTION ---
        
        # Temperature features - delta_t_signed is always created when FE is enabled
        # delta_t_signed = TM - TR (Supply - Return)
        # Sign convention: positive=heating (supply warmer), negative=cooling (supply cooler)
        # Note: Differs from standard HVAC convention (TR - TM) but maintained for consistency
        # (Legacy heating/cooling flags removed - single delta_t_signed covers both modes)
        
        # Time features
        self.create_hour = self.config.getboolean(fe_section, 'create_hour_of_day', fallback=False)  # Changed: redundant with cyclic
        self.create_dow = self.config.getboolean(fe_section, 'create_day_of_week', fallback=False)  # Changed: redundant with cyclic
        self.create_weekend = self.config.getboolean(fe_section, 'create_is_weekend', fallback=False)  # Changed: low importance
        self.create_season_progress = self.config.getboolean(fe_section, 'create_season_progress', fallback=False)  # Changed: zero variance in single season
        self.create_cyclic_time = self.config.getboolean(fe_section, 'create_cyclic_time_encoding', fallback=True)
        
        # Cyclic time feature control (new granular options)
        self.create_cyclic_hour = self.config.getboolean(fe_section, 'create_cyclic_hour', fallback=True)
        self.create_cyclic_dow = self.config.getboolean(fe_section, 'create_cyclic_day_of_week', fallback=True)
        self.create_cyclic_month = self.config.getboolean(fe_section, 'create_cyclic_month', fallback=False)  # Low variance in seasonal
        self.create_cyclic_doy = self.config.getboolean(fe_section, 'create_cyclic_day_of_year', fallback=False)  # Low variance in seasonal
        
        # Occupancy features
        self.create_occupancy = self.config.getboolean(fe_section, 'create_is_occupied', fallback=True)
        self.occ_weekdays = self.config.get(fe_section, 'occupancy_weekdays', fallback='Mon-Fri')
        self.occ_weekend_sat = self.config.getboolean(fe_section, 'occupancy_weekend_sat', fallback=True)
        self.occ_start = self.config.get(fe_section, 'occupancy_start_time', fallback='06:00')
        self.occ_end = self.config.get(fe_section, 'occupancy_end_time', fallback='18:00')
        self.occ_sat_end = self.config.get(fe_section, 'occupancy_sat_end_time', fallback='14:00')
        self.create_startup_shutdown = self.config.getboolean(fe_section, 'create_startup_shutdown_flags', fallback=True)
        self.startup_hours = self.config.getfloat(fe_section, 'startup_hours', fallback=1.0)
        self.shutdown_hours = self.config.getfloat(fe_section, 'shutdown_hours', fallback=1.0)
        
        # Holiday support
        self.holiday_csv = self.config.get(fe_section, 'holiday_csv', fallback='')
        
        # Modulation-based occupancy detection
        self.enable_modulation_occupancy = self.config.getboolean(fe_section, 'enable_modulation_based_occupancy', fallback=True)
        self.modulation_threshold = self.config.getfloat(fe_section, 'modulation_threshold_active', fallback=0.3)
        self.use_supply_mod = self.config.getboolean(fe_section, 'use_supply_modulation', fallback=True)
        self.use_return_mod = self.config.getboolean(fe_section, 'use_return_modulation', fallback=True)
        self.combine_logic = self.config.get(fe_section, 'combine_logic', fallback='OR').upper()
        
        # Rolling features
        self.create_rolling_3h = self.config.getboolean(fe_section, 'create_outdoor_temp_rolling_3h', fallback=True)
        self.create_rolling_24h = self.config.getboolean(fe_section, 'create_outdoor_temp_rolling_24h', fallback=True)
        
        # Control features
        self.create_setpoint_error = self.config.getboolean(fe_section, 'create_setpoint_error', fallback=True)
        self.create_efficiency = self.config.getboolean(fe_section, 'create_efficiency_proxy', fallback=True)
        self.create_advanced_control = self.config.getboolean(fe_section, 'create_advanced_control_features', fallback=False)  # Changed: low importance
        self.setpoint_tolerance = self.config.getfloat(fe_section, 'setpoint_tolerance_degC', fallback=0.5)
        
        # Thermal load features
        self.create_thermal_load = self.config.getboolean(fe_section, 'create_thermal_load', fallback=True)
        
        # Interaction features - granular control
        self.create_interactions = self.config.getboolean(fe_section, 'create_interaction_features', fallback=True)
        self.create_temp_occupancy_interactions = self.config.getboolean(fe_section, 'create_temp_occupancy_interactions', fallback=False)  # Low importance
        self.create_temp_time_interactions = self.config.getboolean(fe_section, 'create_temp_time_interactions', fallback=False)  # Low importance
        self.create_temp_weekend_interaction = self.config.getboolean(fe_section, 'create_temp_weekend_interaction', fallback=False)  # Low importance
        self.create_temp_inertia = self.config.getboolean(fe_section, 'create_temp_inertia', fallback=False)  # Low importance
        self.create_thermal_power = self.config.getboolean(fe_section, 'create_thermal_power', fallback=True)  # Keep - physical relationship
        self.create_load_occupied = self.config.getboolean(fe_section, 'create_load_occupied', fallback=True)  # Keep - meaningful
        
        # Rate of change features
        self.create_rate_of_change = self.config.getboolean(fe_section, 'create_rate_of_change_features', fallback=False)  # Changed: low importance for tree models
        
        # Lag features
        self.create_lags = self.config.getboolean(fe_section, 'create_lag_features', fallback=False)
        if self.create_lags:
            lag_vars_str = self.config.get(fe_section, 'lag_variables', fallback='Temperatura Esterna,temp')
            self.lag_variables = [v.strip() for v in lag_vars_str.split(',')]
            lag_steps_str = self.config.get(fe_section, 'lag_steps', fallback='1, 2, 3, 6, 12')
            self.lag_steps = [int(s.strip()) for s in lag_steps_str.split(',')]
        
        # Output settings
        self.save_csv = self.config.getboolean(fe_section, 'save_features_csv', fallback=True)
        self.save_full_precision = self.config.getboolean(fe_section, 'save_full_precision', fallback=True)
        self.output_suffix = self.config.get(fe_section, 'features_output_suffix', fallback='_features')
        
        # Paths
        processed_folder = self._resolve_path(self.config.get('paths', 'processed_folder'))
        plots_folder = self._resolve_path(self.config.get('paths', 'plots_folder', fallback='plots'))
        interp_folder = Path(self.config.get('paths', 'interpolation_folder'))
        
        self.base_name = f"{self.building_id}_{self.ahu_unit}_{self.season}{self.year}"
        default_filename = f"{self.base_name}_interpolated.csv"
        
        # Try multiple locations for input
        candidates = [
            processed_folder / interp_folder / self.base_name / default_filename,
            processed_folder / interp_folder / default_filename,
            processed_folder / 'interpolation' / self.base_name / default_filename,
            processed_folder / 'interpolation' / default_filename
        ]
        
        self.input_file = candidates[0]
        for candidate in candidates:
            if candidate.exists():
                self.input_file = candidate
                break
        
        # Output directory - create subfolder structure
        self.output_dir = processed_folder / 'feature_engineering' / self.base_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get seasonal dates
        self.season_start, self.season_end = self._get_season_dates()
        
    def _resolve_path(self, path_str):
        """Resolve path relative to config directory"""
        path = Path(path_str)
        if not path.is_absolute():
            path = self.config_dir / path
        return path
    
    def _get_season_dates(self):
        """Get season start/end dates from config"""
        presets = 'seasonal_presets'
        season_lower = self.season.lower()
        
        if season_lower == 'summer':
            start_key = f'summer_start_{self.year}'
            end_key = f'summer_end_{self.year}'
        else:  # winter
            start_key = f'winter_start_{self.year}'
            end_year = int(self.year) + 1
            end_key = f'winter_end_{end_year}'
        
        start_date = self.config.get(presets, start_key, fallback=None)
        end_date = self.config.get(presets, end_key, fallback=None)
        
        if start_date:
            start_date = pd.to_datetime(start_date)
        if end_date:
            end_date = pd.to_datetime(end_date)
            
        return start_date, end_date
    
    def load_data(self):
        """Load interpolated data with timezone handling and frequency validation"""
        print(f"Loading data from: {self.input_file}")
        
        if not self.input_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.input_file}")
        
        df = pd.read_csv(self.input_file)
        
        # Parse time column
        time_col = self.config.get('data', 'time_column', fallback='Time')
        if time_col not in df.columns:
            raise ValueError(f"Time column '{time_col}' not found")
        
        # Parse timestamps robustly:
        # utc=True collapses mixed-offset strings (+02:00 / +01:00 from DST fold)
        # into a uniform UTC DatetimeIndex, avoiding a plain object Index whose
        # elements have no .tz attribute. tz_convert then maps to local time.
        parsed = pd.to_datetime(df[time_col], errors='coerce', utc=True)
        if parsed.isna().all():
            # CSV was saved without offsets (tz-naive) — fall back to naive parse
            parsed = pd.to_datetime(df[time_col], errors='coerce')
        df[time_col] = parsed
        df = df.dropna(subset=[time_col])
        df.set_index(time_col, inplace=True)

        # Timezone handling: normalise to Europe/Rome
        timezone = self.config.get('global', 'timezone', fallback='Europe/Rome')
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors='coerce')
        if getattr(df.index, 'tz', None) is not None:
            df.index = df.index.tz_convert(timezone)
            print(f"  ✓ Converted to {timezone}")
        else:
            df.index = df.index.tz_localize(timezone, ambiguous='infer', nonexistent='shift_forward')
            print(f"  ✓ Localized to {timezone}")
        
        # Validate frequency
        inferred_freq = pd.infer_freq(df.index[:100])  # Check first 100 records
        expected_freq = self.config.get('resampling', 'frequency', fallback='30min')
        
        if inferred_freq:
            print(f"  ℹ Detected frequency: {inferred_freq}")
            if inferred_freq != expected_freq:
                print(f"  ⚠ Warning: Expected {expected_freq}, got {inferred_freq}")
                print(f"    Rolling window sizes will be adjusted accordingly")
        else:
            print(f"  ⚠ Warning: Could not infer frequency, assuming {expected_freq}")
        
        # Store frequency for rolling window calculations
        self.data_freq = inferred_freq or expected_freq
        
        # Apply date range filter if specified
        start_date = self.config.get('data', 'start_date', fallback=None)
        end_date = self.config.get('data', 'end_date', fallback=None)
        
        if start_date and start_date.strip():
            start_date = pd.to_datetime(start_date).tz_localize(timezone)
            df = df[df.index >= start_date]
            print(f"  ✓ Filtered to start_date >= {start_date}")
        
        if end_date and end_date.strip():
            end_date = pd.to_datetime(end_date).tz_localize(timezone)
            df = df[df.index <= end_date]
            print(f"  ✓ Filtered to end_date <= {end_date}")
        
        print(f"Loaded {len(df)} records from {df.index.min()} to {df.index.max()}")
        return df
    
    def _check_columns(self, df, required_cols, feature_name):
        """Helper to check if required columns exist"""
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"  ⚠ Skipped {feature_name}: Missing columns {missing}")
            return False
        return True
    
    def _find_column(self, df, possible_names):
        """
        Find column in dataframe from list of possible names.
        
        Args:
            df: DataFrame to search
            possible_names: List of possible column names (e.g., ['Temperatura Esterna', 'temp'])
            
        Returns:
            str: Actual column name found, or None if not found
        """
        for name in possible_names:
            if name in df.columns:
                return name
        return None
    
    def create_temperature_features(self, df):
        """
        Create temperature difference features - only delta_t_signed.
        
        Sign Convention:
            delta_t_signed = TM - TR (Supply Temperature - Return Temperature)
            
            Note: This differs from the common HVAC convention of TR - TM.
            With this sign convention:
            - Positive values: Supply is warmer than return (heating mode)
            - Negative values: Supply is cooler than return (cooling mode)
            - Magnitude: Absolute temperature change applied by the AHU
        """
        print("\nCreating temperature features...")
        
        required_temp_cols = ['Temperatura Ripresa', 'Temperatura Mandata']
        
        # Create delta_t_signed with explicit sign convention
        # Using TM - TR (not standard TR - TM) for consistency with heating/cooling interpretation
        if self._check_columns(df, required_temp_cols, 'delta_t_signed'):
            df['delta_t_signed'] = df['Temperatura Mandata'] - df['Temperatura Ripresa']
            print("  ✓ Created delta_t_signed = TM - TR")
            print("    Sign convention: positive=heating, negative=cooling")
            print("    (Supply warmer than return = positive value)")
        
        return df
    
    def create_time_features(self, df):
        """Create time-based features"""
        print("\nCreating time-based features...")
        
        if self.create_hour:
            df['hour_of_day'] = df.index.hour
            print("  ✓ Created hour_of_day (0-23)")
        
        if self.create_dow:
            df['day_of_week'] = df.index.dayofweek
            print("  ✓ Created day_of_week (0=Mon, 6=Sun)")
        
        if self.create_weekend:
            df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
            print("  ✓ Created is_weekend (Sat/Sun = 1)")
        
        if self.create_season_progress and self.season_start and self.season_end:
            # Ensure season_start/season_end have the same tz-awareness as the index
            idx_tz = df.index.tz
            start = self.season_start
            end = self.season_end
            
            if idx_tz is not None:
                # index is tz-aware -> localize or convert season bounds to that tz
                if start.tzinfo is None:
                    start = start.tz_localize(idx_tz)
                else:
                    start = start.tz_convert(idx_tz)
                if end.tzinfo is None:
                    end = end.tz_localize(idx_tz)
                else:
                    end = end.tz_convert(idx_tz)
            else:
                # index is tz-naive -> remove tz from season bounds if present
                if start.tzinfo is not None:
                    start = start.tz_localize(None)
                if end.tzinfo is not None:
                    end = end.tz_localize(None)
            
            total_days = (end - start).days
            if total_days <= 0:
                print("  ! Warning: season start/end invalid or zero-length; skipping season_progress")
            else:
                df['season_progress'] = (df.index - start).days / total_days
                df['season_progress'] = df['season_progress'].clip(0, 1)
                print(f"  ✓ Created season_progress (0.0-1.0 over {total_days} days)")
        
        return df
    
    def create_cyclic_time_features(self, df):
        """Create sin/cos encoded time features for ML models
        
        Critical for ML: Encodes cyclical nature of time so hour 23 and 0 are close.
        Without this, linear models treat hour 23 and 0 as 23 units apart!
        
        NOTE: Granular control added based on feature importance analysis.
        - Month features disabled by default (zero variance in seasonal analysis)
        - Day-of-year features disabled by default (low value in seasonal analysis)
        """
        if not self.create_cyclic_time:
            return df
        
        print("\nCreating cyclic time features (sin/cos encoding)...")
        
        # Hour of day (24-hour cycle) - HIGH IMPORTANCE
        if self.create_cyclic_hour:
            hour = df.index.hour
            df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
            print("  ✓ Created hour_sin, hour_cos (24-hour cycle)")
        
        # Day of week (7-day cycle) - MEDIUM/HIGH IMPORTANCE
        if self.create_cyclic_dow:
            dow = df.index.dayofweek
            df['day_of_week_sin'] = np.sin(2 * np.pi * dow / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * dow / 7)
            print("  ✓ Created day_of_week_sin, day_of_week_cos (7-day cycle)")
        
        # Month of year (12-month cycle) - LOW IMPORTANCE in seasonal analysis
        if self.create_cyclic_month:
            month = df.index.month
            df['month_sin'] = np.sin(2 * np.pi * month / 12)
            df['month_cos'] = np.cos(2 * np.pi * month / 12)
            print("  ✓ Created month_sin, month_cos (12-month cycle)")
            print("  ⚠️  WARNING: Month features have low variance in seasonal analysis")
        
        # Day of year (365-day cycle) - LOW IMPORTANCE in seasonal analysis
        if self.create_cyclic_doy:
            day_of_year = df.index.dayofyear
            df['day_of_year_sin'] = np.sin(2 * np.pi * day_of_year / 365)
            df['day_of_year_cos'] = np.cos(2 * np.pi * day_of_year / 365)
            print("  ✓ Created day_of_year_sin, day_of_year_cos (365-day cycle)")
            print("  ⚠️  WARNING: Day-of-year features have low importance in seasonal analysis")
        
        return df
    
    def create_occupancy_feature(self, df):
        """Create occupancy indicator with precise time handling and custom weekday support"""
        if not self.create_occupancy:
            return df
        
        print("\nCreating occupancy feature...")
        
        # Parse occupancy times with full HH:MM precision
        def parse_time(time_str):
            """Parse HH:MM to (hour, minute) tuple"""
            parts = time_str.split(':')
            return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        
        # Parse weekday string (e.g., "Mon-Fri", "Mon-Sat", "Mon,Wed,Fri")
        def parse_weekdays(weekdays_str):
            """Parse weekday string to list of day indices (0=Mon, 6=Sun)"""
            day_map = {
                'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4, 'sat': 5, 'sun': 6,
                'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                'friday': 4, 'saturday': 5, 'sunday': 6
            }
            
            weekdays_str = weekdays_str.lower().strip()
            
            # Handle range format (e.g., "mon-fri")
            if '-' in weekdays_str:
                parts = weekdays_str.split('-')
                if len(parts) == 2:
                    start_day = day_map.get(parts[0].strip())
                    end_day = day_map.get(parts[1].strip())
                    if start_day is not None and end_day is not None:
                        return list(range(start_day, end_day + 1))
            
            # Handle comma-separated format (e.g., "mon,wed,fri")
            if ',' in weekdays_str:
                days = []
                for day in weekdays_str.split(','):
                    day_idx = day_map.get(day.strip())
                    if day_idx is not None:
                        days.append(day_idx)
                return days
            
            # Single day
            day_idx = day_map.get(weekdays_str)
            return [day_idx] if day_idx is not None else [0, 1, 2, 3, 4]  # Default Mon-Fri
        
        start_h, start_m = parse_time(self.occ_start)
        end_h, end_m = parse_time(self.occ_end)
        sat_end_h, sat_end_m = parse_time(self.occ_sat_end)
        
        # Convert to total minutes for precise comparison
        start_minutes = start_h * 60 + start_m
        end_minutes = end_h * 60 + end_m
        sat_end_minutes = sat_end_h * 60 + sat_end_m
        
        # Parse custom weekday pattern
        occupancy_days = parse_weekdays(self.occ_weekdays)
        
        # Initialize as not occupied
        df['is_occupied'] = 0
        
        # Calculate minutes since midnight for each timestamp
        time_minutes = df.index.hour * 60 + df.index.minute
        
        # Apply occupancy for configured weekdays
        for day_idx in occupancy_days:
            day_mask = df.index.dayofweek == day_idx
            time_mask = (time_minutes >= start_minutes) & (time_minutes < end_minutes)
            df.loc[day_mask & time_mask, 'is_occupied'] = 1
        
        # Saturday (if enabled)
        if self.occ_weekend_sat:
            saturday_mask = df.index.dayofweek == 5
            saturday_time_mask = (time_minutes >= start_minutes) & (time_minutes < sat_end_minutes)
            df.loc[saturday_mask & saturday_time_mask, 'is_occupied'] = 1
        
        occupied_pct = (df['is_occupied'].sum() / len(df)) * 100
        print(f"  ✓ Created is_occupied ({occupied_pct:.1f}% occupied, {self.occ_start}-{self.occ_end})")
        print(f"    Applied to days: {self.occ_weekdays}")
        
        # Create startup/shutdown flags for transient analysis
        if self.create_startup_shutdown:
            self._create_startup_shutdown_flags(df)
        
        # Load holidays if specified
        if self.holiday_csv and self.holiday_csv.strip():
            self._apply_holiday_mask(df)
        
        return df
    
    def _create_startup_shutdown_flags(self, df):
        """Create startup and shutdown flags around occupancy transitions"""
        print("\nCreating startup/shutdown flags...")
        
        # Calculate window sizes based on data frequency
        try:
            freq_td = pd.Timedelta(self.data_freq)
            startup_periods = int((self.startup_hours * 3600) / freq_td.total_seconds())
            shutdown_periods = int((self.shutdown_hours * 3600) / freq_td.total_seconds())
        except:
            # Fallback: assume 30min frequency
            startup_periods = int(self.startup_hours * 2)
            shutdown_periods = int(self.shutdown_hours * 2)
        
        # Detect transitions
        df['is_startup'] = 0
        df['is_shutdown'] = 0
        
        occupancy_changes = df['is_occupied'].diff()
        
        # Startup: periods around 0 -> 1 transition
        startup_transitions = df.index[occupancy_changes == 1]
        for trans_time in startup_transitions:
            start_window = trans_time - pd.Timedelta(hours=self.startup_hours/2)
            end_window = trans_time + pd.Timedelta(hours=self.startup_hours/2)
            mask = (df.index >= start_window) & (df.index <= end_window)
            df.loc[mask, 'is_startup'] = 1
        
        # Shutdown: periods around 1 -> 0 transition
        shutdown_transitions = df.index[occupancy_changes == -1]
        for trans_time in shutdown_transitions:
            start_window = trans_time - pd.Timedelta(hours=self.shutdown_hours/2)
            end_window = trans_time + pd.Timedelta(hours=self.shutdown_hours/2)
            mask = (df.index >= start_window) & (df.index <= end_window)
            df.loc[mask, 'is_shutdown'] = 1
        
        startup_pct = (df['is_startup'].sum() / len(df)) * 100
        shutdown_pct = (df['is_shutdown'].sum() / len(df)) * 100
        print(f"  ✓ Created is_startup ({startup_pct:.1f}% of data, ±{self.startup_hours}h window)")
        print(f"  ✓ Created is_shutdown ({shutdown_pct:.1f}% of data, ±{self.shutdown_hours}h window)")
    
    def _apply_holiday_mask(self, df):
        """Apply holiday mask from CSV file"""
        try:
            holiday_file = self._resolve_path(Path(self.holiday_csv))
            if not holiday_file.exists():
                print(f"  ⚠ Holiday file not found: {holiday_file}")
                return
            
            holidays = pd.read_csv(holiday_file, parse_dates=['date'])
            holiday_dates = pd.to_datetime(holidays['date']).dt.date
            
            df['is_holiday'] = df.index.date.isin(holiday_dates).astype(int)
            
            # Override occupancy on holidays
            df.loc[df['is_holiday'] == 1, 'is_occupied'] = 0
            
            holiday_count = df['is_holiday'].sum()
            print(f"  ✓ Created is_holiday ({holiday_count} holiday records)")
            print(f"  ✓ Occupancy overridden on holidays")
            
        except Exception as e:
            print(f"  ⚠ Could not load holidays: {e}")
        
        return df
    
    def create_modulation_based_occupancy(self, df):
        """Create occupancy detection based on fan modulation levels
        
        This is a DATA-DRIVEN approach that detects actual operation mode:
        - High modulation (>= threshold) = Active operation (working hours)
        - Low modulation (< threshold) = Minimal operation (nights/weekends)
        
        More accurate than time-based schedules because it reflects real usage patterns.
        """
        if not self.enable_modulation_occupancy:
            return df
        
        print("\nCreating modulation-based occupancy detection...")
        
        # Check available modulation columns
        supply_col = 'Modulazione Ventilatore Mandata'
        return_col = 'Modulazione Ripresa'
        
        has_supply = supply_col in df.columns and self.use_supply_mod
        has_return = return_col in df.columns and self.use_return_mod
        
        if not has_supply and not has_return:
            print(f"  ⚠ Skipped: No modulation columns available")
            return df
        
        # Initialize the feature
        df['is_active_operation'] = 0
        
        # Build condition based on available columns and combine logic
        if has_supply and has_return:
            supply_active = df[supply_col] >= self.modulation_threshold
            return_active = df[return_col] >= self.modulation_threshold
            
            if self.combine_logic == 'OR':
                df['is_active_operation'] = (supply_active | return_active).astype(int)
                logic_desc = "either supply OR return"
            else:  # AND
                df['is_active_operation'] = (supply_active & return_active).astype(int)
                logic_desc = "both supply AND return"
            
            used_cols = f"{supply_col} and {return_col}"
        elif has_supply:
            df['is_active_operation'] = (df[supply_col] >= self.modulation_threshold).astype(int)
            logic_desc = "supply only"
            used_cols = supply_col
        else:  # has_return
            df['is_active_operation'] = (df[return_col] >= self.modulation_threshold).astype(int)
            logic_desc = "return only"
            used_cols = return_col
        
        # Calculate statistics
        active_pct = (df['is_active_operation'].sum() / len(df)) * 100
        
        print(f"  ✓ Created is_active_operation:")
        print(f"    Threshold: modulation >= {self.modulation_threshold}")
        print(f"    Using: {used_cols} ({logic_desc})")
        print(f"    Active operation: {active_pct:.1f}% of time")
        print(f"    Minimal operation: {100-active_pct:.1f}% of time")
        
        return df
    
    def create_rolling_features(self, df):
        """Create rolling average features with frequency-aware window sizing"""
        # Find external temperature column (try both Italian and English names)
        ext_temp_col = self._find_column(df, ['Temperatura Esterna', 'temp'])
        
        if ext_temp_col is None:
            print("\n⚠ Warning: Outdoor temperature column not found, skipping rolling features")
            return df
        
        print(f"\nCreating rolling average features (using column '{ext_temp_col}')...")
        
        # Calculate window sizes based on actual frequency
        try:
            freq_td = pd.Timedelta(self.data_freq)
            window_3h = int(pd.Timedelta('3H') / freq_td)
            window_24h = int(pd.Timedelta('24H') / freq_td)
            print(f"  ℹ Using frequency {self.data_freq}: 3h={window_3h} steps, 24h={window_24h} steps")
        except Exception as e:
            print(f"  ⚠ Warning: Could not compute window from frequency, using defaults (6, 48)")
            window_3h = 6
            window_24h = 48
        
        if self.create_rolling_3h:
            df['outdoor_temp_rolling_3h'] = df[ext_temp_col].rolling(
                window=window_3h, min_periods=1, center=False
            ).mean()
            print(f"  ✓ Created outdoor_temp_rolling_3h (window={window_3h})")
        
        if self.create_rolling_24h:
            df['outdoor_temp_rolling_24h'] = df[ext_temp_col].rolling(
                window=window_24h, min_periods=1, center=False
            ).mean()
            print(f"  ✓ Created outdoor_temp_rolling_24h (window={window_24h})")
        
        return df
    
    def create_control_features(self, df):
        """Create control-related features - only setpoint_error"""
        print("\nCreating control features...")
        
        # Setpoint error with flexible column matching
        if self.create_setpoint_error:
            if 'Temperatura Mandata' in df.columns:
                # Get all available setpoint-related columns
                setpoint_like = [col for col in df.columns if 'setpoint' in col.lower() or 'set point' in col.lower()]
                
                if setpoint_like:
                    print(f"  ℹ Available setpoint columns: {setpoint_like}")
                
                # Determine active setpoint based on season - match against available columns
                setpoint_col = None
                
                if self.season.lower() == 'summer':
                    # Look for cooling/raffrescamento keywords in available columns
                    cooling_keywords = ['raffrescamento', 'raffreddamento', 'cooling']
                    for col in setpoint_like:
                        col_lower = col.lower()
                        if any(keyword in col_lower for keyword in cooling_keywords):
                            setpoint_col = col
                            break
                    
                elif self.season.lower() == 'winter':
                    # Priority 1: compensated setpoint (Set Points Temperatura Compensata)
                    # This is the winter-specific heating setpoint that varies with outdoor conditions.
                    compensata_keywords = ['compensat']
                    for col in setpoint_like:
                        if any(k in col.lower() for k in compensata_keywords):
                            setpoint_col = col
                            break
                    # Priority 2: explicit heating/riscaldamento keywords
                    if setpoint_col is None:
                        heating_keywords = ['riscaldamento', 'heating']
                        for col in setpoint_like:
                            if any(k in col.lower() for k in heating_keywords):
                                setpoint_col = col
                                break
                    # Priority 3: 'mandata' setpoint (supply temp SP)
                    if setpoint_col is None:
                        for col in setpoint_like:
                            if 'mandata' in col.lower():
                                setpoint_col = col
                                break
                    # Priority 4: first available setpoint column
                    if setpoint_col is None and setpoint_like:
                        setpoint_col = setpoint_like[0]
                
                if setpoint_col:
                    active_setpoint = df[setpoint_col]
                    df['setpoint_error'] = df['Temperatura Mandata'] - active_setpoint
                    print(f"  ✓ Created setpoint_error ({self.season}, using '{setpoint_col}')")
                    # (C) Deadband: zero out tiny errors within sensor noise tolerance
                    sp_deadband = self.config.getfloat('feature_engineering', 'setpoint_error_deadband_degC', fallback=0.0)
                    if sp_deadband > 0.0:
                        n_zeroed = (df['setpoint_error'].abs() <= sp_deadband).sum()
                        df['setpoint_error'] = df['setpoint_error'].where(df['setpoint_error'].abs() > sp_deadband, 0.0)
                        print(f"  ✓ Deadband ±{sp_deadband}°C applied: {n_zeroed:,} values zeroed ({n_zeroed/len(df)*100:.1f}%)")
                else:
                    print(f"  ⚠ Skipped setpoint_error: No matching setpoint column found for {self.season} season")
            else:
                print("  ⚠ Skipped setpoint_error: Temperatura Mandata not found")
        
        return df
    
    def create_advanced_control_features(self, df):
        """Create advanced control features for MPC and control analysis
        
        Critical for MPC: Provides PID-like features (proportional, integral, derivative)
        for better control performance analysis and optimization.
        """
        if not self.create_advanced_control:
            return df
        
        print("\nCreating advanced control features...")
        
        if 'setpoint_error' in df.columns:
            # Cumulative error (integral term) - tracks persistent errors
            df['cumulative_setpoint_error'] = df['setpoint_error'].cumsum()
            print("  ✓ Created cumulative_setpoint_error (integral term)")
            
            # Error rate (derivative term) - tracks how fast error is changing
            df['setpoint_error_rate'] = df['setpoint_error'].diff()
            print("  ✓ Created setpoint_error_rate (derivative term)")
            
            # Time out of tolerance (control quality metric)
            df['out_of_tolerance'] = (np.abs(df['setpoint_error']) > self.setpoint_tolerance).astype(int)
            df['time_out_of_tolerance'] = df['out_of_tolerance'].cumsum()
            print(f"  ✓ Created time_out_of_tolerance (tolerance = {self.setpoint_tolerance}°C)")
            
            # Absolute error (for MAE-like metrics)
            df['abs_setpoint_error'] = np.abs(df['setpoint_error'])
            print("  ✓ Created abs_setpoint_error")
            
            # Squared error (for MSE-like metrics)
            df['squared_setpoint_error'] = df['setpoint_error'] ** 2
            print("  ✓ Created squared_setpoint_error")
        else:
            print("  ⚠ Skipped advanced control features: setpoint_error not available")
        
        return df
    
    def create_thermal_load_features(self, df):
        """
        Create thermal_power_signed feature only.
        
        Formula: thermal_power_signed = delta_t_signed × fan_modulation
        
        This represents a proxy for thermal power delivery:
        - Positive: Heating power (supply warmer than return)
        - Negative: Cooling power (supply cooler than return)
        - Magnitude: Proportional to temperature change and airflow
        
        Note: Uses TM - TR sign convention (see create_temperature_features)
        """
        if not self.create_thermal_power:
            return df
        
        print("\nCreating thermal power feature...")
        
        # Only create thermal_power_signed (signed thermal power)
        if 'delta_t_signed' in df.columns and 'Modulazione Ventilatore Mandata' in df.columns:
            df['thermal_power_signed'] = df['delta_t_signed'] * df['Modulazione Ventilatore Mandata']
            print("  ✓ Created thermal_power_signed = delta_t_signed × fan_modulation")
            print("    (directional: positive=heating power, negative=cooling power)")
        else:
            print("  ⚠ Skipped thermal_power_signed: Missing delta_t_signed or Modulazione Ventilatore Mandata")
        
        return df
    
    def create_differencing_features(self, df):
        """(D) First-difference features for hunting / oscillation detection (Salsbury 2014).

        Creates Δcol = col(t) − col(t-1) for each configured column.
        Useful as LSTM inputs to detect unstable control loops.
        NOTE: these features are written to the output CSV but are NOT added to the
        LSTM feature lists automatically — add them to features_winter / features_summer
        in config.ini [lstm_autoencoder] if you want the model to use them.
        """
        if not self.config.getboolean('feature_engineering', 'create_differencing_features', fallback=False):
            return df

        print("\nCreating differencing features (Salsbury 2014)...")
        diff_cols_str = self.config.get('feature_engineering', 'differencing_columns', fallback='')
        diff_cols = [c.strip() for c in diff_cols_str.split(',') if c.strip()]

        for col in diff_cols:
            if col not in df.columns:
                print(f"  ⚠ Skipped diff for '{col}': column not found")
                continue
            safe_name = col.replace(' ', '_').replace('.', '').replace('/', '_')
            feat_name = f"d_{safe_name}"
            df[feat_name] = df[col].diff().fillna(0.0)
            print(f"  ✓ Created {feat_name} = Δ{col}")

        return df

    def create_interaction_features(self, df):
        """Create interaction features - DISABLED (all interactions removed)"""
        if not self.create_interactions:
            return df
        
        print("\nInteraction features disabled - keeping only essential features")
        
        return df
    
    def create_rate_of_change_features(self, df):
        """Create rate of change (derivative) features
        
        Critical for MPC: Optimization algorithms need derivatives to understand
        how fast variables are changing. Essential for predictive control.
        """
        if not self.create_rate_of_change:
            return df
        
        print("\nCreating rate of change features...")
        
        # Temperature rates (°C per time step)
        # Find external temp column with flexible naming
        ext_temp_col = self._find_column(df, ['Temperatura Esterna', 'temp'])
        temp_cols = [ext_temp_col, 'Temperatura Ripresa', 'Temperatura Mandata'] if ext_temp_col else ['Temperatura Ripresa', 'Temperatura Mandata']
        
        for col in temp_cols:
            if col and col in df.columns:
                rate_col = f'{col}_rate'
                df[rate_col] = df[col].diff()
                
                # Also create absolute rate (speed of change regardless of direction)
                abs_rate_col = f'{col}_abs_rate'
                df[abs_rate_col] = np.abs(df[rate_col])
                
                print(f"  ✓ Created {rate_col}, {abs_rate_col}")
        
        # Fan speed change rate (control action speed)
        if 'Modulazione Ventilatore Mandata' in df.columns:
            df['fan_speed_change'] = df['Modulazione Ventilatore Mandata'].diff()
            df['fan_speed_abs_change'] = np.abs(df['fan_speed_change'])
            print("  ✓ Created fan_speed_change, fan_speed_abs_change")
        
        if 'Modulazione Ventilatore Ripresa' in df.columns:
            df['return_fan_change'] = df['Modulazione Ventilatore Ripresa'].diff()
            df['return_fan_abs_change'] = np.abs(df['return_fan_change'])
            print("  ✓ Created return_fan_change, return_fan_abs_change")
        
        # Thermal load rate (how fast energy demand is changing)
        if 'thermal_load' in df.columns:
            df['thermal_load_rate'] = df['thermal_load'].diff()
            df['thermal_load_abs_rate'] = np.abs(df['thermal_load_rate'])
            print("  ✓ Created thermal_load_rate, thermal_load_abs_rate")
        
        # Setpoint error rate (already created in advanced_control_features, but check)
        if 'setpoint_error' in df.columns and 'setpoint_error_rate' not in df.columns:
            df['setpoint_error_rate'] = df['setpoint_error'].diff()
            print("  ✓ Created setpoint_error_rate")
        
        return df
    
    def create_lag_features(self, df):
        """Create lagged features for causal/time-series analysis"""
        if not self.create_lags:
            return df
        
        print("\nCreating lag features...")
        
        for var in self.lag_variables:
            # Try to find the column if not directly present (flexible naming)
            actual_col = var if var in df.columns else self._find_column(df, [var])
            
            if actual_col is None or actual_col not in df.columns:
                print(f"  ⚠ Skipped lags for {var}: Column not found")
                continue
            
            for lag in self.lag_steps:
                lag_col_name = f"{actual_col}_lag_{lag}"
                df[lag_col_name] = df[actual_col].shift(lag)
                
            print(f"  ✓ Created {len(self.lag_steps)} lags for {actual_col}: {self.lag_steps}")
        
        # Report impact
        if self.create_lags:
            n_lag_features = len(self.lag_variables) * len(self.lag_steps)
            print(f"  ℹ Total lag features created: {n_lag_features}")
            print(f"  ⚠ Note: First {max(self.lag_steps)} rows will have NaN values")
        
        return df
    
    def generate_summary(self, df_original, df_enhanced):
        """Generate summary of created features"""
        print("\n" + "="*70)
        print("FEATURE ENGINEERING SUMMARY")
        print("="*70)
        
        original_cols = set(df_original.columns)
        new_cols = set(df_enhanced.columns) - original_cols
        
        print(f"\nOriginal columns: {len(original_cols)}")
        print(f"New features created: {len(new_cols)}")
        print(f"Total columns: {len(df_enhanced.columns)}")
        
        if new_cols:
            print("\nNew features:")
            for col in sorted(new_cols):
                non_null = df_enhanced[col].notna().sum()
                pct = (non_null / len(df_enhanced)) * 100
                print(f"  • {col:30s} ({non_null:6d} values, {pct:5.1f}%)")
        
        print("\nData quality:")
        print(f"  Total records: {len(df_enhanced)}")
        print(f"  Date range: {df_enhanced.index.min()} to {df_enhanced.index.max()}")
        print(f"  Missing values: {df_enhanced.isnull().sum().sum()}")
        
        return new_cols
    
    def save_enhanced_data(self, df):
        """Save enhanced dataset with both full precision and rounded versions"""
        if not self.save_csv:
            return
        
        output_filename = f"{self.base_name}{self.output_suffix}.csv"
        output_path = self.output_dir / output_filename
        
        # Save full precision version (if enabled)
        if self.save_full_precision:
            df.to_csv(output_path)
            print(f"\n✓ Saved full precision dataset: {output_path}")
            print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")
        
        # Also save rounded version for human review
        rounded_filename = f"{self.base_name}{self.output_suffix}_rounded.csv"
        rounded_path = self.output_dir / rounded_filename
        
        df_rounded = df.copy()
        numeric_cols = df_rounded.select_dtypes(include=[np.number]).columns
        df_rounded[numeric_cols] = df_rounded[numeric_cols].round(2)
        
        df_rounded.to_csv(rounded_path)
        print(f"✓ Saved rounded dataset (2 decimals): {rounded_path}")
        print(f"  Size: {rounded_path.stat().st_size / 1024:.1f} KB")
        
        # If only rounded version requested, make it the primary file
        if not self.save_full_precision:
            rounded_path.replace(output_path)
            print(f"  ℹ Using rounded version as primary output")
    
    def generate_feature_summary(self, df, new_features):
        """Generate summary information for new features (no plots)"""
        print("\nGenerating feature summary...")
        
        feature_records = []
        
        for feature in sorted(new_features):
            if feature not in df.columns or df[feature].isna().all():
                continue
            
            try:
                data = df[feature].dropna()
                description = self._generate_feature_description(feature, data)
                
                feature_records.append({
                    'feature': feature,
                    'description': description,
                    'mean': data.mean(),
                    'std': data.std(),
                    'min': data.min(),
                    'max': data.max(),
                    'count': len(data)
                })
                
                print(f"  ✓ Processed {feature}")
                
            except Exception as e:
                print(f"  ! Warning: Could not process {feature}: {e}")
                continue
        
        return feature_records
    
    def _generate_feature_description(self, feature_name, data):
        """Generate Italian description text for a feature"""
        descriptions = {
            'delta_t_signed': f"Differenza di temperatura con segno (TM - TR): temperatura di mandata meno temperatura di ripresa. "
                             f"Media: {data.mean():.2f}°C. "
                             f"Valori positivi = riscaldamento (mandata più calda della ripresa), "
                             f"valori negativi = raffrescamento (mandata più fredda della ripresa). "
                             f"Nota: convenzione TM - TR, diversa dalla convenzione HVAC standard TR - TM.",
            'delta_t_cooling': f"Differenza di temperatura tra aria di ripresa e aria di mandata in modalità raffrescamento. "
                             f"ΔT medio: {data.mean():.2f}°C. Valori più elevati indicano una migliore efficienza di rimozione del calore.",
            'delta_t_heating': f"Differenza di temperatura tra aria di mandata e aria di ripresa in modalità riscaldamento. "
                             f"ΔT medio: {data.mean():.2f}°C. Valori più elevati indicano una migliore efficienza di distribuzione del calore.",
            'delta_t': f"Differenza di temperatura assoluta tra aria di ripresa e aria di mandata. "
                      f"Media: {data.mean():.2f}°C. Indica l'efficacia termica dell'UTA.",
            'hour_of_day': f"Ora del giorno (0-23). La distribuzione mostra i pattern operativi durante la giornata. "
                          f"I dati coprono {data.nunique()} ore distinte.",
            'day_of_week': f"Giorno della settimana (0=Lunedì, 6=Domenica). Mostra i pattern operativi settimanali. "
                          f"{data.nunique()} giorni unici rappresentati.",
            'is_weekend': f"Indicatore di fine settimana (1=fine settimana, 0=giorno feriale). "
                         f"{(data.sum()/len(data)*100):.1f}% dei punti dati sono nei fine settimana.",
            'is_occupied': f"Stato di occupazione basato sul programma dell'edificio. "
                          f"{(data.sum()/len(data)*100):.1f}% dei periodi di tempo sono durante le ore di occupazione. "
                          f"Giorni feriali: 6:00-18:00, Sabato: 6:00-14:00, Domenica: non occupato.",
            'season_progress': f"Progresso normalizzato nella stagione (0.0 a 1.0). "
                             f"Permette di tracciare le tendenze stagionali e i cambiamenti nel tempo.",
            'outdoor_temp_rolling_3h': f"Media mobile a 3 ore della temperatura esterna. "
                                      f"Media: {data.mean():.2f}°C. Attenua le variazioni a breve termine.",
            'outdoor_temp_rolling_24h': f"Media mobile a 24 ore della temperatura esterna. "
                                       f"Media: {data.mean():.2f}°C. Cattura i pattern di temperatura giornalieri.",
            'setpoint_error': f"Differenza tra temperatura di mandata effettiva e setpoint. "
                            f"Errore medio: {data.mean():.2f}°C. Valori positivi indicano sovra-temperatura, "
                            f"valori negativi indicano sotto-temperatura rispetto al target.",
            'efficiency_proxy': f"Metrica proxy per l'efficienza del sistema (|ΔT| / modulazione). "
                              f"Media: {data.mean():.3f}. Valori più elevati suggeriscono prestazioni migliori "
                              f"per unità di modulazione del ventilatore."
        }
        
        return descriptions.get(feature_name, 
                              f"Feature: {feature_name}. Media: {data.mean():.2f}, "
                              f"Dev. Std: {data.std():.2f}, Range: [{data.min():.2f}, {data.max():.2f}]")
    
    def create_word_report(self, df, feature_records):
        """Create Word document report with feature descriptions"""
        if Document is None:
            print("\n! python-docx not installed, skipping Word report generation")
            print("  Install with: pip install python-docx")
            return
        
        if not feature_records or len(feature_records) == 0:
            print("\n⚠ No features to report, skipping Word document")
            return
        
        print("\nGenerating Word report...")
        
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement
        
        doc = Document()
        
        # Main title
        title = doc.add_heading('Feature Engineering', level=1)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Date range subtitle
        date_min = df.index.min().strftime('%Y-%m-%d')
        date_max = df.index.max().strftime('%Y-%m-%d')
        subtitle = doc.add_paragraph(f'Periodo di analisi: {date_min} - {date_max}')
        subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
        subtitle_format = subtitle.runs[0].font
        subtitle_format.size = Pt(11)
        subtitle_format.color.rgb = RGBColor(102, 102, 102)
        
        doc.add_paragraph()  # Spacing
        
        # Create a table for each feature
        for idx, record in enumerate(feature_records, 1):
            # Create table: 2 rows, 1 column (no image row)
            table = doc.add_table(rows=2, cols=1)
            table.style = 'Table Grid'
            
            # Row 1: Feature name (centered, bold, dark blue)
            title_cell = table.rows[0].cells[0]
            title_para = title_cell.paragraphs[0]
            title_para.text = record['feature']
            title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            title_run = title_para.runs[0]
            title_run.font.bold = True
            title_run.font.size = Pt(12)
            title_run.font.color.rgb = RGBColor(0, 51, 102)
            
            # Set cell background color (light blue)
            shading_elm = OxmlElement('w:shd')
            shading_elm.set(qn('w:fill'), 'E6F2FF')
            title_cell._element.get_or_add_tcPr().append(shading_elm)
            
            # Row 2: Description and statistics
            desc_cell = table.rows[1].cells[0]
            desc_para = desc_cell.paragraphs[0]
            
            # Add description
            desc_text = f"Descrizione:\n{record['description']}\n\n"
            desc_text += f"Statistiche:\n"
            desc_text += f"  • Media: {record['mean']:.2f}\n"
            desc_text += f"  • Deviazione Standard: {record['std']:.2f}\n"
            desc_text += f"  • Minimo: {record['min']:.2f}\n"
            desc_text += f"  • Massimo: {record['max']:.2f}\n"
            desc_text += f"  • Conteggio: {record['count']:,} valori"
            
            desc_para.text = desc_text
            desc_run = desc_para.runs[0]
            desc_run.font.size = Pt(10)
            
            doc.add_paragraph()  # Spacing between tables
        
        # Save report in output directory (no plots folder needed)
        report_filename = f"{self.base_name}_feature_engineering_report.docx"
        report_path = self.output_dir / report_filename
        doc.save(str(report_path))
        
        print(f"✓ Word report saved: {report_path}")
    
    def run_engineering(self):
        """Run complete feature engineering pipeline"""
        if not self.enable_fe:
            print("\n" + "="*70)
            print("FEATURE ENGINEERING DISABLED - STARTING SIMPLE")
            print("="*70)
            print("\n🎯 STEP 1: Use original measurements only")
            print("\n📊 Your 8 original features:")
            print("   • Temperatura Ripresa")
            print("   • Temperatura Mandata") 
            print("   • Set Points Temperatura Mandata Raffrescamento")
            print("   • Set Points Temperatura Mandata Riscaldamento")
            print("   • Modulazione Ripresa")
            print("   • Modulazione Mandata")
            print("   • Temperatura Esterna (or 'temp' from weather merge)")
            print("   • Umidita Esterna (or 'rh' from weather merge)")
            print("\n✅ Cleaned data location:")
            print(f"   {self.input_file}")
            print("\n🔬 Next steps for unsupervised learning:")
            print("   1. Load the cleaned data")
            print("   2. StandardScaler (normalize features)")
            print("   3. PCA or k-means clustering")
            print("   4. Visualize: PCA scatter (PC1 vs PC2), cluster centers")
            print("   5. Check: Do clusters make physical sense?")
            print("\n💡 If Step 1 results are UNCLEAR:")
            print("   → Edit config.ini:")
            print("     enable_feature_engineering = true")
            print("   → Enable ONLY simple features (e.g., delta_t_signed)")
            print("   → Re-run this script")
            print("   → Compare PCA/clustering results")
            print("\n📖 Strategy: Add features selectively, not all at once!")
            print("="*70)
            return
        
        print("\n" + "="*70)
        print("HVAC FEATURE ENGINEERING")
        print("="*70)
        print(f"Building: {self.building_id}")
        print(f"AHU Unit: {self.ahu_unit}")
        print(f"Season: {self.season} {self.year}")
        print("="*70)
        
        # Load data
        try:
            df = self.load_data()
            df_original = df.copy()
        except Exception as e:
            print(f"\nFailed to load data: {e}")
            return
        
        # Create features - ORDER MATTERS (some features depend on others)
        df = self.create_temperature_features(df)              # Creates delta_t_signed
        df = self.create_time_features(df)                      # Creates giorno_settimana, is_weekday
        df = self.create_cyclic_time_features(df)               # Disabled
        df = self.create_occupancy_feature(df)                  # Disabled
        df = self.create_modulation_based_occupancy(df)         # Creates is_active_operation (data-driven)
        df = self.create_rolling_features(df)                   # Creates rolling averages
        df = self.create_control_features(df)                   # Creates setpoint_error (+ deadband)
        df = self.create_advanced_control_features(df)          # Disabled
        df = self.create_thermal_load_features(df)              # Creates thermal_power_signed
        df = self.create_differencing_features(df)              # (D) dT_mandata, dMod_mandata
        df = self.create_interaction_features(df)               # Disabled
        df = self.create_rate_of_change_features(df)            # Disabled
        df = self.create_lag_features(df)                       # Disabled
        
        # Generate summary
        new_features = self.generate_summary(df_original, df)
        
        # Ensure index has proper name before saving (for compatibility with analyzer)
        time_col = self.config.get('data', 'time_column', fallback='Time')
        if df.index.name != time_col:
            df.index.name = time_col
            print(f"\n✓ Set index name to '{time_col}' for compatibility")
        
        # Save results
        self.save_enhanced_data(df)
        
        # Generate feature summary and report
        feature_records = self.generate_feature_summary(df, new_features)
        self.create_word_report(df, feature_records)
        
        print("\n" + "="*70)
        print("FEATURE ENGINEERING COMPLETED")
        print("="*70)
        print(f"\nOutput location:")
        print(f"  CSV & Report: {self.output_dir}")
        
        return df, new_features


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.ini'
    
    print(f"HVAC Feature Engineering")
    print(f"=" * 70)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {Path.cwd()}")
    print(f"=" * 70)
    
    try:
        engineer = FeatureEngineer(config_path)
        engineer.run_engineering()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)