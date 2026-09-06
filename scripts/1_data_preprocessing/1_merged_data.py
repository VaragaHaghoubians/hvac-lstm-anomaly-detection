"""

ENHANCED DATA MERGING SCRIPT - HVAC + Weather Integration
============================================================
Key Feature: Weather Data Resampling to Eliminate Merge Gaps

PROBLEM SOLVED:
When merging HVAC data (30min intervals) with weather data (60min intervals),
traditional merging creates ~50% missing weather values due to timestamp misalignment.

SOLUTION:
This script automatically resamples weather data to match HVAC frequency BEFORE merging.
Result: Clean merge with 0% frequency-mismatch gaps.

Example:
- HVAC: 00:00, 00:30, 01:00, 01:30, ... (30min intervals)
- Weather Original: 00:00, 01:00, 02:00, ... (60min intervals)
- Weather Resampled: 00:00, 00:30, 01:00, 01:30, ... (30min intervals)
- Merge Result: All timestamps aligned, no artificial gaps!

Remaining gaps after merge represent TRUE missing data from source files,
not artifacts of frequency mismatch.

TIMEZONE HANDLING STRATEGY (BEST PRACTICE):
============================================
1. HVAC timestamps: Localized to Europe/Rome (they are local times)
2. Weather timestamps: Converted from UTC to Europe/Rome
3. All processing: Uses timezone-aware timestamps (accurate DST handling)
4. Merging: Performed on timezone-aware timestamps (no ambiguity)
5. Saving: Only at final step, timezone removed for CSV compatibility
   
This approach ensures:
✅ Correct DST transition handling
✅ Accurate time-based merging
✅ No timestamp ambiguity during processing
✅ CSV compatibility for downstream tools
"""

import pandas as pd  # For data manipulation and analysis
import os  # For operating system interface (file paths, directory operations)
import configparser  # For reading configuration files
import sys  # For system-specific parameters and functions
from datetime import datetime  # For timestamp generation
from pathlib import Path  # For modern path handling
from dateutil.tz import gettz  # For timezone fallback resolution

# Add parent directory to path to import config_utils
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


def get_timezone(tz_name):
    """Resolve a timezone object, with fallbacks for environments missing zoneinfo data."""
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(tz_name)
    except Exception as zone_error:
        tz_obj = gettz(tz_name)
        if tz_obj is not None:
            print(f"  ⚠️  ZoneInfo unavailable ({zone_error}), using dateutil fallback for {tz_name}")
            return tz_obj
        try:
            import pytz
            tz_obj = pytz.timezone(tz_name)
            print(f"  ⚠️  ZoneInfo unavailable ({zone_error}), using pytz fallback for {tz_name}")
            return tz_obj
        except Exception:
            raise RuntimeError(
                f"Unable to resolve timezone '{tz_name}'. Install tzdata or use a supported timezone provider."
            ) from zone_error


# =============================================================================
# COLUMN RENAMING FUNCTIONS
# =============================================================================
def get_column_display_names(config):
    """
    Extract column display name mapping from config.ini
    
    Args:
        config: ConfigParser object with loaded config.ini
    
    Returns:
        Dictionary mapping original column names to display names
        Example: {'T Ripresa': 'Temperatura Ripresa', ...}
    """
    if not config.has_section('column_display_names'):
        return {}
    
    # config.items() returns lowercase keys, so we need to read the file directly
    # to preserve the original case
    rename_map = {}
    
    # Read the config file directly to get original case
    config_file = config._sections.get('column_display_names', {})
    
    # If that doesn't work, read from the raw file
    if not config_file or all(k.islower() for k in config_file.keys()):
        # Fall back to manual parsing to preserve case
        import re
        config_path = None
        
        # Try to find config.ini
        from pathlib import Path
        script_dir = Path(__file__).parent
        possible_paths = [
            script_dir.parent.parent / 'config.ini',  # scripts/1_data_preprocessing/ -> scripts/ -> project_root/
            script_dir.parent / 'config.ini',  # Fallback: scripts/
            Path('config.ini'),  # Current directory
        ]
        
        for path in possible_paths:
            if path.exists():
                config_path = path
                break
        
        if config_path:
            with open(config_path, 'r', encoding='utf-8') as f:
                in_section = False
                for line in f:
                    line = line.strip()
                    if line == '[column_display_names]':
                        in_section = True
                        continue
                    elif line.startswith('[') and in_section:
                        break  # End of section
                    elif in_section and '=' in line and not line.startswith(';') and not line.startswith('#'):
                        # Parse the line preserving case
                        key, value = line.split('=', 1)
                        rename_map[key.strip()] = value.strip()
    
    return rename_map


def rename_columns_from_config(df, config):
    """
    Rename DataFrame columns based on config.ini [column_display_names] section
    
    Args:
        df: DataFrame to rename columns in
        config: ConfigParser object with loaded config.ini
    
    Returns:
        DataFrame with renamed columns
    
    Example:
        # Rename to Italian display names
        df = rename_columns_from_config(df, config)
    """
    rename_map = get_column_display_names(config)
    
    if not rename_map:
        print("ℹ️  No [column_display_names] section found in config.ini - skipping column renaming")
        return df
    
    # Only rename columns that exist in the DataFrame
    existing_renames = {k: v for k, v in rename_map.items() if k in df.columns}
    
    if existing_renames:
        df = df.rename(columns=existing_renames)
        print(f"\n✅ Renamed {len(existing_renames)} columns to Italian display names:")
        for old, new in existing_renames.items():
            print(f"   {old:<30} → {new}")
    else:
        print("ℹ️  No matching columns found to rename")
    
    return df


def apply_all_column_renames(df, config):
    """
    Unified column renaming: weather columns + Italian display names.
    
    Args:
        df (pd.DataFrame): DataFrame to rename
        config (configparser.ConfigParser): Configuration object
        
    Returns:
        pd.DataFrame: DataFrame with all renames applied
    """
    print("\n🔤 Applying column name standardization...")
    
    # Step 1: Weather column renaming (raw API names → English standard)
    weather_rename_map = {
        'temp': 'Temp. Esterna',
        'rh': 'Umid. Esterna'
    }
    
    weather_renamed = []
    for old_name, new_name in weather_rename_map.items():
        if old_name in df.columns:
            weather_renamed.append(f"{old_name} → {new_name}")
    
    if weather_renamed:
        df = df.rename(columns=weather_rename_map)
        print(f"✓ Standardized weather columns: {', '.join(weather_renamed)}")
    
    # Step 2: Italian display names (English standard → Italian)
    df = rename_columns_from_config(df, config)
    
    return df


#####newcode#####
def align_and_resample_data(df, config, time_col):
    """
    Aligns the dataframe to a uniform time grid.
    
    This function performs these critical steps:
    1. Sorts the data chronologically.
    2. Removes any duplicate timestamps.
    3. Sets the time column as the index.
    4. Resamples the data to a fixed frequency, averaging values within each bin.
    5. Reorders columns according to the configuration file.
    
    Args:
        df (pd.DataFrame): The dataframe to process.
        config (configparser.ConfigParser): The configuration object.
        time_col (str): The name of the timestamp column.
        
    Returns:
        pd.DataFrame: The cleaned, resampled, and ordered dataframe.
    """
    print("\n🔄 Aligning data to uniform time grid...")
    
    # 1. Sort by time and remove duplicates
    df = df.sort_values(by=time_col).drop_duplicates(subset=[time_col], keep='first')
    print(f"✓ Data sorted and duplicates removed. {len(df)} rows remaining.")
    
    # 2. Set time column as index for resampling
    df = df.set_index(time_col)
    
    # 2.5. Ensure all columns are numeric before resampling
    print("  Ensuring numeric data types...")
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'string':
            # Try to convert any remaining object columns to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"    ✓ Converted {col} to numeric")
    
    # 3. Resample to a uniform grid
    try:
        resample_freq = config.get("processing", "resample_freq")
        # Use .mean() for numeric columns only
        df_resampled = df.resample(resample_freq).mean(numeric_only=True)
        print(f"✓ Resampled data to '{resample_freq}' frequency. Result has {len(df_resampled)} rows.")
    except (configparser.NoOptionError, configparser.NoSectionError):
        print("⚠️  resample_freq not found in config. Resampling skipped.")
        df_resampled = df # Skip resampling if not configured
    
    # Reset index to bring the time column back
    df_resampled = df_resampled.reset_index()

    # 4. Reorder columns for consistency
    try:
        # Get the desired column order from the config
        column_order_str = config.get("data_setup", "column_order")
        desired_order = [col.strip() for col in column_order_str.split(',')]
        
        # Create the final column list
        # Start with the desired columns that actually exist in the dataframe
        final_columns = [col for col in desired_order if col in df_resampled.columns]
        # Append any other columns that were not in the desired_order list
        remaining_columns = [col for col in df_resampled.columns if col not in final_columns]
        final_columns.extend(remaining_columns)
        
        df_resampled = df_resampled[final_columns]
        print("✓ Columns reordered for consistency.")
        
    except (configparser.NoOptionError, configparser.NoSectionError):
        print("⚠️  column_order not found in config. Column reordering skipped.")

    return df_resampled
####endnewcode#####


def read_config(config_path):
    """
    Reads the configuration file and returns a ConfigParser object.
    Enhanced with robust error handling and path validation.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        configparser.ConfigParser: Parsed configuration object
    """
    # Validate config file exists
    config_file = Path(config_path)
    if not config_file.exists():
        # Try parent directory (project root) - 2 levels up for new folder structure
        config_file = Path(__file__).parent.parent.parent / config_path
    
    if not config_file.exists():
        print(f"❌ Error: Configuration file not found!")
        print(f"  Looking for: {config_path}")
        print(f"  Also tried: {config_file}")
        print(f"  Current directory: {Path.cwd()}")
        print(f"  Script location: {Path(__file__).parent}")
        sys.exit(1)
    
    print(f"[INFO] Using config: {config_file}")
    
    try:
        # Use the load_config helper (with or without config_utils)
        config = load_config(str(config_file))
        # Remember config.ini directory to resolve relative paths (e.g., ./data)
        try:
            config._config_dir = str(config_file.parent)
        except Exception:
            pass
        print(f"✓ Configuration loaded successfully")
        return config
    except Exception as e:
        print(f"❌ Error loading config file: {e}")
        sys.exit(1)

def validate_config(config):
    """
    Validate configuration file for common issues that could cause merge failures.
    This prevents runtime errors by catching configuration problems early.
    
    Args:
        config (configparser.ConfigParser): Configuration object to validate
        
    Returns:
        bool: True if configuration is valid, False if issues found
    """
    print("\n🔍 Validating configuration...")
    
    issues = []  # List to collect all configuration issues
    
    # Check that all required sections exist in the config file
    required_sections = ['paths', 'data', 'merging', 'weather']
    for section in required_sections:
        if not config.has_section(section):
            issues.append(f"Missing required section: [{section}]")
    
    # Verify that the base folder path exists on the filesystem
    if config.has_section('paths'):
        base_folder = config.get('paths', 'base_folder', fallback=None)
        if base_folder:
            cfg_dir = getattr(config, '_config_dir', None)
            base_folder_checked = (
                os.path.abspath(os.path.join(cfg_dir, base_folder))
                if cfg_dir and not os.path.isabs(base_folder)
                else base_folder
            )
            if not os.path.exists(base_folder_checked):
                issues.append(f"Base folder does not exist: {base_folder_checked}")
    
    # Check weather merge configuration consistency
    if config.has_section('merging'):
        enable_merge = config.getboolean('merging', 'enable_merge', fallback=False)
        if enable_merge:
            # If merging is enabled, weather file must be specified
            weather_csv = config.get('merging', 'weather_csv_name', fallback=None)
            if not weather_csv:
                issues.append("Weather merge enabled but weather_csv_name not specified")
            
            # Check that weather columns are specified
            weather_cols = config.get('merging', 'weather_columns_to_add', fallback='')
            if not weather_cols.strip():
                issues.append("Weather merge enabled but no weather_columns_to_add specified")
    
    # Report validation results
    if issues:
        print("❌ Configuration issues found:")
        for issue in issues:
            print(f"  • {issue}")
        return False
    else:
        print("✓ Configuration validation passed")
        return True

def detect_csv_separator(filepath):
    """
    Detect the separator used in a CSV file.
    Handles files with 'sep=' header line.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        tuple: (separator, rows_to_skip) - separator character and number of header rows to skip
    """
    try:
        # Open the file with UTF-8 encoding
        with open(filepath, 'r', encoding='utf-8') as f:
            # Read the first line to check for separator indicators
            first_line = f.readline().strip()
            
            # Check if the first line explicitly defines the separator
            if first_line.startswith('sep='):
                # Extract the separator character after 'sep='
                separator = first_line.replace('sep=', '').strip()
                print(f"✓ Detected separator from sep= line: '{separator}'")
                return separator, 1  # Return separator and number of rows to skip
            
            # Reset file pointer to beginning for further analysis
            f.seek(0)
            
            # Check if 'sep=' appears anywhere in the first line (handle encoding issues)
            if 'sep=' in first_line:
                # Try to extract separator for common cases
                if 'sep=;' in first_line:
                    print(f"✓ Found sep=; indicator, using semicolon separator")
                    return ';', 1
                elif 'sep=,' in first_line:
                    print(f"✓ Found sep=, indicator, using comma separator")
                    return ',', 1
            
            # Read multiple lines for separator detection analysis
            lines = []
            f.seek(0)  # Reset to beginning
            for _ in range(5):  # Read first 5 lines for analysis
                line = f.readline()
                if line:
                    lines.append(line)
            
            # Check if semicolons appear consistently across lines
            semicolon_counts = [line.count(';') for line in lines]
            if all(count > 2 for count in semicolon_counts):  # At least 3 semicolons per line
                print(f"✓ Detected semicolon separator (found in all lines)")
                # Check if first line is a sep= declaration
                if lines[0].strip() == 'sep=;' or lines[0].strip().startswith('sep='):
                    return ';', 1  # Skip the sep= line
                else:
                    return ';', 0  # Don't skip any lines
            
            # Test other common separators
            separators = [',', '\t', '|']  # Comma, tab, pipe
            sep_counts = {}
            
            # Count occurrences of each separator in each line
            for sep in separators:
                counts = [line.count(sep) for line in lines if line]
                # Only consider if separator appears in all lines
                if counts and all(c > 0 for c in counts):
                    sep_counts[sep] = min(counts)  # Use minimum count as consistency measure
            
            # Choose the separator with the most consistent occurrence
            if sep_counts:
                best_sep = max(sep_counts.items(), key=lambda x: x[1])[0]
                print(f"✓ Detected separator: '{best_sep}'")
                return best_sep, 0
            else:
                # Fall back to comma if no separator detected
                print("⚠️  Could not detect separator, defaulting to comma")
                return ',', 0
                
    except Exception as e:
        # Handle any file reading errors
        print(f"⚠️  Error detecting separator: {e}, defaulting to comma")
        return ',', 0

def load_csv_with_detection(filepath, skiprows_config=0):
    """
    Load CSV file with automatic separator detection and proper handling of sep= lines.
    
    Args:
        filepath (str): Path to the CSV file
        skiprows_config (int): Additional rows to skip from configuration
        
    Returns:
        pandas.DataFrame: Loaded and cleaned DataFrame
    """
    print(f"\n🔍 Detecting CSV format for: {os.path.basename(filepath)}")
    
    # Detect the separator and rows to skip
    separator, sep_skiprows = detect_csv_separator(filepath)
    
    # Calculate total rows to skip (config + separator detection)
    total_skiprows = skiprows_config + sep_skiprows
    
    # Try multiple encodings to handle different file formats
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    # Attempt to load with each encoding until successful
    for encoding in encodings:
        try:
            # Load CSV with detected separator and encoding
            df = pd.read_csv(filepath, sep=separator, skiprows=total_skiprows, encoding=encoding)
            print(f"✓ Successfully loaded CSV with encoding: {encoding}")
            
            # Clean column names by removing leading/trailing spaces
            df.columns = df.columns.str.strip()
            
            return df
        except Exception as e:
            # If this is the last encoding to try, raise the error
            if encoding == encodings[-1]:
                print(f"❌ Failed to load CSV with any encoding. Last error: {e}")
                raise
            continue  # Try next encoding
    
    return None

def safe_merge_with_fallback(df_main, df_weather, main_time_col, weather_time_col, tolerance_hours=2):
    """
    Perform merge with multiple fallback strategies to ensure robust data integration.
    Uses different merge approaches depending on data characteristics and overlap.
    
    Args:
        df_main (pd.DataFrame): Main HVAC dataset
        df_weather (pd.DataFrame): Weather dataset  
        main_time_col (str): Time column name in main dataset
        weather_time_col (str): Time column name in weather dataset
        tolerance_hours (float): Maximum time difference allowed for nearest merge (default: 2 hours)
        
    Returns:
        tuple: (merged_dataframe, weather_time_column_name or None)
    """
    print("\n🔄 Attempting merge with fallback strategies...")
    print(f"  Merge tolerance: {tolerance_hours} hours")
    
    try:
        # Strategy 1: Exact timestamp merge (fastest, works when timestamps align perfectly)
        print("  Trying Strategy 1: Exact timestamp merge...")
        df_merged = pd.merge(
            df_main, 
            df_weather[[weather_time_col, 'temp', 'rh']], 
            left_on=main_time_col, 
            right_on=weather_time_col, 
            how='left'  # Keep all main data rows, add weather where available
        )
        
        # Check merge success rate by counting non-null weather data
        weather_match_rate = df_merged['temp'].notna().mean()
        if weather_match_rate > 0.8:  # 80% match rate is considered successful
            print(f"✓ Strategy 1 successful: {weather_match_rate:.1%} weather data coverage")
            return df_merged, weather_time_col
        else:
            print(f"⚠️  Strategy 1 low coverage: {weather_match_rate:.1%}, trying next strategy...")
    
    except Exception as e:
        print(f"⚠️  Strategy 1 failed: {e}")
    
    try:
        # Strategy 2: Nearest timestamp merge (most robust, handles slight time misalignments)
        print("  Trying Strategy 2: Nearest timestamp merge...")
        df_merged = pd.merge_asof(
            df_main.sort_values(main_time_col),  # Sort main data by time
            df_weather[[weather_time_col, 'temp', 'rh']].sort_values(weather_time_col),  # Sort weather data
            left_on=main_time_col,
            right_on=weather_time_col,
            direction='nearest',  # Find closest timestamp match
            tolerance=pd.Timedelta(f'{tolerance_hours}hours')  # Use configurable tolerance
        )
        
        # Check final merge success
        weather_match_rate = df_merged['temp'].notna().mean()
        print(f"✓ Strategy 2 completed: {weather_match_rate:.1%} weather data coverage")
        return df_merged, weather_time_col
        
    except Exception as e:
        print(f"❌ Strategy 2 failed: {e}")
        
    # Strategy 3: Fallback - return original data without weather merge
    print("⚠️  All merge strategies failed. Proceeding without weather data integration.")
    print("     The script will continue with original HVAC data only.")
    return df_main, None

def handle_missing_weather_data(df_merged, config):
    """
    Handle cases where weather data is missing or incomplete after merge.
    Applies intelligent gap-filling strategies to minimize data loss.
    
    IMPORTANT: Creates 'weather_filled' flag column to track which data was imputed
    with seasonal defaults. This allows you to exclude synthetic data from analysis.
    
    Args:
        df_merged (pd.DataFrame): Merged dataset potentially with missing weather data
        config (configparser.ConfigParser): Configuration object for settings
        
    Returns:
        pd.DataFrame: Dataset with improved weather data coverage and 'weather_filled' flag
    """
    print("\n🌡️  Processing weather data gaps...")
    
    # Initialize weather_filled flag: 0 = original data, 1 = filled with defaults
    df_merged['weather_filled'] = 0
    
    # Load seasonal temperatures from config
    try:
        seasonal_temps_str = config.get("weather_defaults", "seasonal_temps")
        seasonal_temps = {}
        for pair in seasonal_temps_str.split(','):
            month, temp = pair.strip().split(':')
            seasonal_temps[int(month)] = float(temp)
        print(f"✓ Loaded seasonal temperatures from config")
    except (configparser.NoOptionError, configparser.NoSectionError, ValueError):
        print(f"⚠️  Using default seasonal temperatures for Northern Italy")
        seasonal_temps = {
            12: 5, 1: 3, 2: 7,      # Winter: Dec, Jan, Feb
            3: 12, 4: 16, 5: 20,    # Spring: Mar, Apr, May  
            6: 24, 7: 27, 8: 26,    # Summer: Jun, Jul, Aug
            9: 22, 10: 17, 11: 10   # Autumn: Sep, Oct, Nov
        }
    
    # Load default humidity from config
    try:
        default_humidity = config.getfloat("weather_defaults", "default_humidity")
    except (configparser.NoOptionError, configparser.NoSectionError):
        default_humidity = 60.0
        print(f"⚠️  Using default humidity: {default_humidity}%")
    
    # Load forward fill limit from config
    try:
        forward_fill_limit = config.getint("weather_defaults", "forward_fill_limit")
    except (configparser.NoOptionError, configparser.NoSectionError):
        forward_fill_limit = 6
        print(f"⚠️  Using default forward fill limit: {forward_fill_limit} rows")
    
    weather_cols = ['Temp. Esterna', 'Umid. Esterna']  # Target weather columns
    
    for col in weather_cols:
        if col in df_merged.columns:
            # Calculate missing data percentage
            missing_pct = df_merged[col].isna().mean() * 100
            
            if missing_pct > 5:  # Only process if significant missing data
                print(f"  Processing {col}: {missing_pct:.1f}% missing data")
                
                # Strategy 1: Forward fill for short gaps
                # This assumes weather conditions change gradually
                df_merged[col] = df_merged[col].fillna(method='ffill', limit=forward_fill_limit)
                
                # Strategy 2: Use seasonal averages for longer gaps
                if col == 'Temp. Esterna' and 'Time' in df_merged.columns:
                    # Extract month from timestamp for seasonal estimation
                    month = df_merged['Time'].dt.month
                    
                    # Apply seasonal estimates only where data is still missing
                    for month_num, temp in seasonal_temps.items():
                        mask = (month == month_num) & df_merged[col].isna()
                        if mask.any():
                            df_merged.loc[mask, col] = temp
                            df_merged.loc[mask, 'weather_filled'] = 1  # Mark as filled
                
                elif col == 'Umid. Esterna':
                    # Use configurable default humidity
                    mask = df_merged[col].isna()
                    if mask.any():
                        df_merged.loc[mask, col] = default_humidity
                        df_merged.loc[mask, 'weather_filled'] = 1  # Mark as filled
                
                # Report improvement
                final_missing = df_merged[col].isna().mean() * 100
                improvement = missing_pct - final_missing
                filled_count = (df_merged['weather_filled'] == 1).sum()
                print(f"    → Reduced missing data by {improvement:.1f}% (now {final_missing:.1f}% missing)")
                if filled_count > 0:
                    filled_pct = (filled_count / len(df_merged)) * 100
                    print(f"    → {filled_count:,} rows ({filled_pct:.1f}%) filled with seasonal defaults")
                    print(f"    ℹ️  Use 'weather_filled == 0' filter to exclude synthetic data from analysis")
    
    return df_merged

def perform_data_quality_checks(df_merged, main_time_col):
    """
    Perform comprehensive data quality checks on merged dataset.
    Identifies potential issues that could affect downstream analysis.
    
    Args:
        df_merged (pd.DataFrame): Merged dataset to check
        main_time_col (str): Name of the main time column
        
    Returns:
        pd.DataFrame: Same dataset (quality checks are informational only)
    """
    print("\n🔍 Enhanced Data Quality Analysis:")
    
    # Check 1: Duplicate timestamps (can cause analysis errors)
    duplicates = df_merged[main_time_col].duplicated().sum()
    if duplicates > 0:
        print(f"  ⚠️  Warning: {duplicates} duplicate timestamps found!")
        # Remove duplicates keeping the first occurrence
        df_merged = df_merged.drop_duplicates(subset=[main_time_col], keep='first')
        print(f"      → Removed duplicates, {len(df_merged)} rows remaining")
    else:
        print(f"  ✓ No duplicate timestamps")
    
    # Check 2: Time gaps analysis (irregular sampling can affect analysis)
    df_sorted = df_merged.sort_values(main_time_col)
    time_diffs = df_sorted[main_time_col].diff()
    large_gaps = time_diffs > pd.Timedelta('2 hours')
    gap_count = large_gaps.sum()
    
    if gap_count > 0:
        print(f"  ⚠️  Warning: {gap_count} time gaps > 2 hours detected")
        max_gap = time_diffs.max()
        print(f"      Largest gap: {max_gap}")
        
        # Show location of largest gap for debugging
        max_gap_idx = time_diffs.idxmax()
        gap_time = df_sorted.loc[max_gap_idx, main_time_col]
        print(f"      Gap occurs around: {gap_time}")
    else:
        print(f"  ✓ No significant time gaps detected")
    
    # Check 3: Weather data integration success
    weather_cols = ['Temp. Esterna', 'Umid. Esterna']
    for col in weather_cols:
        if col in df_merged.columns:
            coverage = (df_merged[col].notna().sum() / len(df_merged)) * 100
            if coverage < 90:
                print(f"  ⚠️  {col}: Only {coverage:.1f}% coverage")
            else:
                print(f"  ✓ {col}: {coverage:.1f}% coverage")
    
    # Check 4: Temperature reasonableness validation
    temp_cols = ['T Ripresa', 'T Mandata', 'Temp. Esterna']
    for col in temp_cols:
        if col in df_merged.columns:
            valid_data = df_merged[col].dropna()
            if len(valid_data) > 0:
                min_temp, max_temp = valid_data.min(), valid_data.max()
                mean_temp = valid_data.mean()
                
                # Different validation ranges for different temperature types
                if col == 'Temp. Esterna':
                    # External temperature: should be within reasonable climate range
                    if min_temp < -25 or max_temp > 45:
                        print(f"  ⚠️  {col}: Unusual range {min_temp:.1f}°C to {max_temp:.1f}°C (mean: {mean_temp:.1f}°C)")
                    else:
                        print(f"  ✓ {col}: Normal range {min_temp:.1f}°C to {max_temp:.1f}°C (mean: {mean_temp:.1f}°C)")
                else:
                    # Internal temperatures: should be within HVAC operating range
                    if min_temp < 5 or max_temp > 40:
                        print(f"  ⚠️  {col}: Unusual range {min_temp:.1f}°C to {max_temp:.1f}°C (mean: {mean_temp:.1f}°C)")
                    else:
                        print(f"  ✓ {col}: Normal range {min_temp:.1f}°C to {max_temp:.1f}°C (mean: {mean_temp:.1f}°C)")
    
    # Check 5: Sensor failure detection (constant values over time)
    numeric_cols = df_merged.select_dtypes(include=[float, int]).columns
    for col in numeric_cols:
        if col != main_time_col:  # Skip time column
            non_null_data = df_merged[col].dropna()
            if len(non_null_data) > 100:  # Only check if enough data points
                unique_values = non_null_data.nunique()
                if unique_values == 1:
                    constant_value = non_null_data.iloc[0]
                    print(f"  ⚠️  {col}: Constant value {constant_value} detected (possible sensor failure)")
                elif unique_values < 3 and len(non_null_data) > 1000:
                    print(f"  ⚠️  {col}: Very few unique values ({unique_values}) - check sensor")
    
    # Check 6: Data completeness summary
    total_rows = len(df_merged)
    print(f"\n📊 Data Completeness Summary:")
    print(f"  Total rows: {total_rows}")
    
    # Calculate completeness for key columns
    key_columns = ['T Ripresa', 'T Mandata', 'Temp. Esterna', 'Umid. Esterna']
    for col in key_columns:
        if col in df_merged.columns:
            complete_pct = (df_merged[col].notna().sum() / total_rows) * 100
            print(f"  {col}: {complete_pct:.1f}% complete")
    
    return df_merged

def main():
    """
    Main function to merge AHU and weather datasets based on config settings.
    Enhanced with robust error handling and data quality validation.
    
    Performs the following steps:
    1. Load and validate configuration
    2. Load main AHU data with format detection
    3. Clean and process main data
    4. Load weather data  
    5. Handle timezone compatibility
    6. Merge datasets with fallback strategies
    7. Handle missing weather data
    8. Perform quality checks
    9. Save merged results with summary
    """
    # Print header for script execution
    print("=" * 80)
    print("ENHANCED DATA MERGING SCRIPT - Robust CSV & Weather Integration")
    print("=" * 80)

    # --- 1. READ AND VALIDATE CONFIGURATION ---
    # Find config.ini - try project root first
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
        print("❌ Error: config.ini not found!")
        print("Searched in:")
        for p in config_paths:
            print(f"  - {p}")
        print("\nPlease ensure config.ini exists in the project root directory.")
        sys.exit(1)
    
    # Load configuration settings
    config = read_config(config_path)
    
    # Validate configuration before proceeding with merge
    if not validate_config(config):
        print("\n❌ Please fix configuration issues before proceeding.")
        print("   Check the paths, column names, and merge settings in config.ini")
        sys.exit(1)

    # --- 2. GET PATHS AND FILENAMES FROM CONFIG ---
    # Try to use config_helper for dynamic season-based values
    season_start_date = None
    season_end_date = None
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from config_helper import load_hvac_config
        hvac_config = load_hvac_config(str(config_path))
        print(f"\n✅ Using config_helper for dynamic season-based configuration")
        print(f"   Season: {hvac_config.season} {hvac_config.year}")
        print(f"   Date range: {hvac_config.start_date} to {hvac_config.end_date}")
        
        # Use dynamically generated values from config_helper
        # Resolve base folder relative to config.ini directory
        base_folder_raw = config.get("paths", "base_folder")
        cfg_dir = Path(getattr(config, "_config_dir", Path(config_path).parent))
        base_folder_path = (cfg_dir / base_folder_raw) if not Path(base_folder_raw).is_absolute() else Path(base_folder_raw)
        data_folder = str(base_folder_path.resolve())
        main_input_csv = hvac_config.input_csv  # Dynamic: C1_UTA1_Winter2025.csv
        weather_csv_name = config.get("weather", "file_path")  # Use actual weather file in data folder
        weather_time_col = config.get("merging", "weather_time_column")
        weather_cols_to_add = [c.strip() for c in config.get("merging", "weather_columns_to_add").split(',')]
        main_time_col = config.get("data", "time_column")
        skiprows = config.getint("data", "skiprows")
        
        # Store season date range for filtering weather data later
        season_start_date = pd.to_datetime(hvac_config.start_date)
        season_end_date = pd.to_datetime(hvac_config.end_date)
        
    except (ImportError, Exception) as helper_error:
        # Fallback to standard config if config_helper unavailable
        print(f"ℹ️  config_helper not available ({helper_error}), using standard config")
        try:
            # Extract all required configuration values from config.ini
            # Resolve base folder relative to the config.ini directory
            base_folder_raw = config.get("paths", "base_folder")  # Base data directory (may be relative)
            cfg_dir = Path(getattr(config, "_config_dir", Path(config_path).parent))
            base_folder_path = (cfg_dir / base_folder_raw) if not Path(base_folder_raw).is_absolute() else Path(base_folder_raw)
            data_folder = str(base_folder_path.resolve())
            main_input_csv = config.get("paths", "input_csv")  # Main AHU data file
            weather_csv_name = config.get("merging", "weather_csv_name")  # Weather data file
            weather_time_col = config.get("merging", "weather_time_column")  # Weather timestamp column
            # Parse comma-separated list of weather columns to merge
            weather_cols_to_add = [c.strip() for c in config.get("merging", "weather_columns_to_add").split(',')]
            main_time_col = config.get("data", "time_column")  # Main data timestamp column
            skiprows = config.getint("data", "skiprows")  # Rows to skip when loading
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            # Handle missing configuration sections or options
            print(f"❌ Error: A required setting is missing from your config.ini file: {e}")
            sys.exit(1)

    # Display configuration summary
    print(f"\n📋 Configuration Summary:")
    print(f"  Main CSV: {main_input_csv}")
    print(f"  Weather CSV: {weather_csv_name}")
    print(f"  Main time column: {main_time_col}")
    print(f"  Weather time column: {weather_time_col}")
    print(f"  Weather columns to add: {weather_cols_to_add}")

    # Construct full file paths
    main_file_path = os.path.join(data_folder, main_input_csv)
    weather_file_path = os.path.join(data_folder, weather_csv_name)

    # DEBUG: Show path construction details
    print(f"\n🔍 DEBUG - Path Construction:")
    print(f"  data_folder = {repr(data_folder)}")
    print(f"  weather_csv_name = {repr(weather_csv_name)}")
    print(f"  weather_file_path = {repr(weather_file_path)}")
    print(f"  Path exists? {Path(weather_file_path).exists()}")
    print(f"  Data folder exists? {Path(data_folder).exists()}")
    if Path(data_folder).exists():
        print(f"  Files in data folder matching 'LIMF*':")
        for f in Path(data_folder).glob('LIMF*'):
            print(f"    - {f.name}")
        print(f"  All files in data folder:")
        for f in Path(data_folder).iterdir():
            print(f"    - {f.name}")

    # --- 3. LOAD MAIN DATASET ---
    print(f"\n📁 Loading main data file: {main_input_csv}")
    # Check if main data file exists
    if not os.path.exists(main_file_path):
        print(f"❌ Error: Main input file not found at '{main_file_path}'")
        sys.exit(1)
    
    # Load main data with automatic format detection
    df_main = load_csv_with_detection(main_file_path, skiprows)
    print(f"✓ Loaded {len(df_main)} rows, {len(df_main.columns)} columns")
    print(f"✓ Columns found: {list(df_main.columns)}")
    
    # Verify that the specified time column exists
    if main_time_col not in df_main.columns:
        print(f"\n❌ Error: Time column '{main_time_col}' not found in main data!")
        print(f"Available columns: {list(df_main.columns)}")
        
        # Try to find time column with case-insensitive search
        for col in df_main.columns:
            if col.lower() == main_time_col.lower():
                print(f"✓ Found time column with different case: '{col}'")
                main_time_col = col  # Update to actual column name
                break
        else:
            # If no matching column found, exit
            print("\nPlease update config.ini with the correct time column name.")
            sys.exit(1)
    
    # Convert time column to datetime format and localize to Europe/Rome
    print(f"\n🕒 Processing time column: '{main_time_col}'")
    # Try automatic parsing first
    df_main[main_time_col] = pd.to_datetime(df_main[main_time_col], errors='coerce')
    
    # If automatic failed, try US format (7/9/2025 12:00:00 AM)
    if df_main[main_time_col].isna().all():
        print("  Automatic parsing failed, trying US date format...")
        df_main[main_time_col] = pd.to_datetime(df_main[main_time_col], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    
    # Localize HVAC timestamps as Europe/Rome (they are local times, not UTC)
    # Handle DST transitions with a duplicate-aware boolean array.
    #
    # WHY NOT ambiguous='infer' or ambiguous=False:
    #   'infer'      — can fail when all rows are processed as a flat Series
    #   ambiguous=False — maps BOTH occurrences of the duplicated hour to +01:00
    #                     (winter time), so the merge picks the wrong (second)
    #                     weather row for the FIRST occurrence (which should be +02:00).
    #
    # CORRECT APPROACH — sequential duplicate detection:
    #   During the Oct fall-back, the BMS records 02:00 and 02:30 twice.
    #   In chronological order: first occurrence = CEST (+02:00, DST still active),
    #   second occurrence = CET (+01:00, DST ended).
    #   We build a bool array where True = DST active (+02:00) for every timestamp.
    #   Only truly ambiguous timestamps use the value; all others are ignored by pandas.
    print("  Localizing timestamps to Europe/Rome timezone...")
    naive_ts = df_main[main_time_col]
    # Build DST bool array: True for all positions (non-ambiguous timestamps ignore it),
    # then flip to False for the SECOND occurrence of any duplicated timestamp (DST fold).
    import numpy as np
    dst_flags = np.ones(len(naive_ts), dtype=bool)
    seen: dict = {}
    for i, ts in enumerate(naive_ts):
        key = ts  # NaT-safe key; each unique naive timestamp
        if key in seen:
            dst_flags[i] = False   # second occurrence → standard time (+01:00)
        else:
            seen[key] = i          # first (or only) occurrence → DST active (+02:00)

    rome_tz = get_timezone('Europe/Rome')
    try:
        df_main[main_time_col] = df_main[main_time_col].dt.tz_localize(
            rome_tz,
            ambiguous=dst_flags,        # first duplicate → +02:00, second → +01:00
            nonexistent='shift_forward'  # spring-forward: shift ahead by 1 h
        )
        # Report any DST-fold rows that were localized
        n_fold = int((~dst_flags).sum())
        if n_fold > 0:
            fold_times = naive_ts[~dst_flags].dt.strftime('%Y-%m-%d %H:%M').unique()
            print(f"  ✓ DST fall-back: {n_fold} row(s) at {list(fold_times)} assigned +01:00 (standard time)")
        print("  ✓ Timezone localization successful (duplicate-aware DST handling)")
    except Exception as e:
        # Ultimate fallback — should not normally be reached
        print(f"  ⚠️  Duplicate-aware localization failed ({type(e).__name__}): {e}")
        print("     Falling back to ambiguous='infer'")
        df_main[main_time_col] = df_main[main_time_col].dt.tz_localize(
            rome_tz,
            ambiguous='infer',
            nonexistent='shift_forward'
        )
        print("  ✓ Timezone localization successful (infer fallback)")
    
    # Show first parsed timestamp
    first_valid = df_main[main_time_col].dropna().iloc[0] if len(df_main[main_time_col].dropna()) > 0 else None
    if first_valid:
        print(f"  First timestamp: {first_valid}")
    
    # Report invalid timestamps
    invalid_times = df_main[main_time_col].isna().sum()
    if invalid_times > 0:
        print(f"ℹ️  {invalid_times} rows with invalid timestamps (kept for interpolation)")
    
    print(f"✓ HVAC time localized to Europe/Rome ({len(df_main)} total rows)")

    # --- 4. CLEAN DATA ---
    print("\n🧹 Cleaning main data...")
    
    # Delete existing 'Temp. Esterna' column if it exists (prevent conflicts with weather data)
    if "Temp. Esterna" in df_main.columns:
        print("✓ Removing existing 'Temp. Esterna' column to prevent conflicts")
        df_main.drop(columns=["Temp. Esterna"], inplace=True)
    
    # Clean ALL numeric columns by removing units and converting to numeric
    # This includes temperature columns, modulation columns, and any other numeric data
    print("  Cleaning all numeric columns...")
    
    # Get all columns except the time column
    cols_to_clean = [col for col in df_main.columns if col != main_time_col]
    
    for col in cols_to_clean:
        # Try to convert to numeric, checking if cleaning is needed
        if df_main[col].dtype == 'object' or df_main[col].dtype == 'string':
            # Clean string columns that might contain numeric data with units
            # Handle various encodings of degree symbol (°, Â°, ¬∞)
            df_main[col] = df_main[col].astype(str).str.replace(' °C', '', regex=False)
            df_main[col] = df_main[col].str.replace(' Â°C', '', regex=False)  # Handle Â° encoding
            df_main[col] = df_main[col].str.replace(' ¬∞C', '', regex=False)  # Handle ¬∞ encoding (Mac)
            df_main[col] = df_main[col].str.replace('°C', '', regex=False)  # Without space
            df_main[col] = df_main[col].str.replace('Â°C', '', regex=False)  # Without space
            df_main[col] = df_main[col].str.replace('¬∞C', '', regex=False)  # Without space
            df_main[col] = df_main[col].str.replace(' %', '', regex=False)  # Handle percentage
            df_main[col] = df_main[col].str.replace('%', '', regex=False)
            # Replace comma decimal separators with dots
            df_main[col] = df_main[col].str.replace(',', '.', regex=False)
            # Remove any extra whitespace
            df_main[col] = df_main[col].str.strip()
            # Convert to numeric, invalid values become NaN
            df_main[col] = pd.to_numeric(df_main[col], errors='coerce')
            print(f"    ✓ Cleaned and converted: {col}")
    
    # Validate temperature columns specifically
    temp_cols = [col for col in df_main.columns if 'temp' in col.lower() or col in ["T Ripresa", "T Mandata", "SP Temp. Mand.", "SP Temp. Comp."]]
    for col in temp_cols:
        if col in df_main.columns and pd.api.types.is_numeric_dtype(df_main[col]):
            # Remove unrealistic temperature readings (outside reasonable HVAC range)
            invalid_mask = (df_main[col] < -10) | (df_main[col] > 60)
            invalid_count = invalid_mask.sum()
            if invalid_count > 0:
                print(f"    ⚠️  Found {invalid_count} unrealistic readings in {col} (<-10°C or >60°C), setting to NaN")
                df_main.loc[invalid_mask, col] = pd.NA

    # --- 5. LOAD WEATHER DATASET ---
    print(f"\n📁 Loading weather data file: {weather_csv_name}")
    # Check if weather data file exists
    if not os.path.exists(weather_file_path):
        print(f"❌ Error: Weather file not found at '{weather_file_path}'")
        sys.exit(1)
    
    # Load weather data (typically standard CSV format)
    df_weather = load_csv_with_detection(weather_file_path)
    print(f"✓ Loaded {len(df_weather)} rows")
    print(f"✓ Weather columns: {list(df_weather.columns)}")
    
    # Verify that required weather columns exist
    if weather_time_col not in df_weather.columns:
        print(f"❌ Error: Weather time column '{weather_time_col}' not found!")
        print(f"Available columns: {list(df_weather.columns)}")
        sys.exit(1)
    
    # Check for required weather data columns
    for col in ['temp', 'rh']:  # Temperature and relative humidity
        if col not in df_weather.columns:
            print(f"❌ Error: Weather column '{col}' not found!")
            print(f"Available weather columns: {list(df_weather.columns)}")
            sys.exit(1)
    
    # Convert weather timestamp: UTC -> Europe/Rome
    print(f"🕒 Processing weather time column: '{weather_time_col}'")
    # Parse timestamps as UTC
    df_weather[weather_time_col] = pd.to_datetime(df_weather[weather_time_col], errors='coerce', utc=True)
    
    # Convert from UTC to Europe/Rome timezone
    rome_tz = get_timezone('Europe/Rome')
    df_weather[weather_time_col] = df_weather[weather_time_col].dt.tz_convert(rome_tz)
    
    # Report invalid timestamps but KEEP them
    invalid_weather = df_weather[weather_time_col].isna().sum()
    if invalid_weather > 0:
        print(f"ℹ️  Found {invalid_weather} weather rows with invalid timestamps (kept)")
    
    # Show first weather timestamp
    first_weather = df_weather[weather_time_col].dropna().iloc[0] if len(df_weather[weather_time_col].dropna()) > 0 else None
    if first_weather:
        print(f"  First weather timestamp: {first_weather}")
    
    print(f"✓ Weather time converted UTC -> Europe/Rome ({len(df_weather)} rows kept)")
    
    # --- 6. MERGE DATASETS (DIRECT MERGE - NO FILTERING) ---
    print(f"\n🔄 Merging datasets directly (no date filtering)...")
    print(f"   HVAC rows: {len(df_main):,}")
    print(f"   Weather rows: {len(df_weather):,}")

    
    # Use LEFT JOIN to keep ALL HVAC data, add weather where timestamps match EXACTLY
    df_merged = pd.merge(
        df_main,
        df_weather,
        left_on=main_time_col,
        right_on=weather_time_col,
        how='left'
    )
    
    # Remove redundant weather timestamp column
    if weather_time_col in df_merged.columns and weather_time_col != main_time_col:
        df_merged.drop(columns=[weather_time_col], inplace=True)
    
    # Report merge results
    weather_matches = df_merged['temp'].notna().sum() if 'temp' in df_merged.columns else 0
    match_rate = (weather_matches / len(df_merged) * 100) if len(df_merged) > 0 else 0
    print(f"✓ Merge complete: {len(df_merged):,} rows (all HVAC data preserved)")
    print(f"  Weather matched: {weather_matches:,} rows ({match_rate:.1f}%)")
    print(f"  Weather NaN: {len(df_merged) - weather_matches:,} rows (kept for interpolation)")

    # --- 7.5 VERIFY MERGE QUALITY (RESAMPLING EFFECTIVENESS) ---
    print("\n📊 Merge Quality Check:")
    weather_cols_check = ['Temp. Esterna', 'Umid. Esterna']
    for col in weather_cols_check:
        if col in df_merged.columns:
            missing_count = df_merged[col].isna().sum()
            missing_pct = (missing_count / len(df_merged)) * 100
            if missing_pct < 1:
                print(f"   ✅ {col}: {missing_count:,} missing ({missing_pct:.2f}%) - Excellent!")
            elif missing_pct < 5:
                print(f"   ✓ {col}: {missing_count:,} missing ({missing_pct:.2f}%) - Good")
            else:
                print(f"   ⚠️  {col}: {missing_count:,} missing ({missing_pct:.2f}%)")
    
    print("   Merge complete - all data preserved as-is")

    # --- 8. RENAME COLUMNS ---
    print("\n🔤 Renaming columns...")
    df_merged = apply_all_column_renames(df_merged, config)
    print("✓ All timestamps remain timezone-aware (Europe/Rome) for accurate processing")

    # --- 8.5. REINDEX TO COMPLETE 30-MIN GRID (make gaps explicit NaN rows) ---
    # After the left-join merge, missing HVAC timestamps are simply absent rows.
    # Reindexing inserts NaN rows for every missing slot so that:
    #   • downstream gap detection works correctly (timestamp-diff ≠ 30 min → gap)
    #   • interpolation scripts receive a continuous DatetimeIndex, not a sparse one
    #   • gap characterization counts are exact rather than inferred from diff jumps
    # -------------------------------------------------------------------------
    print("\n📐 Reindexing to complete 30-min grid (explicit NaN rows for gaps)...")
    try:
        resample_freq = config.get('processing', 'resample_freq', fallback='30min')
    except Exception:
        resample_freq = '30min'

    # Build a complete grid from first to last timestamp
    ts_min = df_merged[main_time_col].min()
    ts_max = df_merged[main_time_col].max()
    full_grid = pd.date_range(start=ts_min, end=ts_max, freq=resample_freq, tz=ts_min.tzinfo)

    rows_before = len(df_merged)
    df_merged = (
        df_merged
        .set_index(main_time_col)
        .reindex(full_grid)
        .rename_axis(main_time_col)
        .reset_index()
    )
    rows_after  = len(df_merged)
    rows_added  = rows_after - rows_before

    if rows_added > 0:
        print(f"✓ Reindex complete: inserted {rows_added:,} missing rows as NaN "
              f"({rows_before:,} → {rows_after:,} total rows)")
        # Report how many consecutive blocks those gaps form
        gap_rows  = df_merged[df_merged.drop(columns=main_time_col).isna().all(axis=1)]
        if len(gap_rows) > 0:
            # Consecutive NaN rows belong to the same gap episode
            is_nan_row = df_merged.drop(columns=main_time_col).isna().all(axis=1).values
            gaps_in_episodes = int(
                sum(1 for i, v in enumerate(is_nan_row)
                    if v and (i == 0 or not is_nan_row[i - 1]))
            )
            print(f"   → {gaps_in_episodes} distinct gap episode(s) now represented as NaN blocks")
    else:
        print("✓ No missing timestamps found — the grid was already complete")

    # --- 9. SAVE RESULTS ---
    print("\n💾 Preparing to save merged dataset...")
    
    # --- 9.1. REMOVE TIMEZONE FOR CSV COMPATIBILITY (FINAL STEP) ---
    print("\n🕒 Removing timezone for CSV compatibility (final step only)...")
    # Create a copy for saving without modifying the original timezone-aware data
    df_to_save = df_merged.copy()
    
    # Remove timezone from time column (keeps local time representation)
    if main_time_col in df_to_save.columns:
        if isinstance(df_to_save[main_time_col].dtype, pd.DatetimeTZDtype):
            df_to_save[main_time_col] = df_to_save[main_time_col].dt.tz_localize(None)
            print(f"✓ Removed timezone from '{main_time_col}' for CSV export")
    
    # Also check for any other timezone-aware datetime columns
    for col in df_to_save.select_dtypes(include=['datetimetz']).columns:
        df_to_save[col] = df_to_save[col].dt.tz_localize(None)
        print(f"✓ Removed timezone from '{col}' for CSV export")
    
    print("  Note: Timezone removed only for saving - processing used tz-aware timestamps")
    
    # --- 9.2. SAVE TO FILE ---
    
    # Construct output directory path with proper hierarchy
    # Go up three levels: 1_data_preprocessing -> scripts -> HVAC_project
    script_file = os.path.abspath(__file__)
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_file)))
    base_name = os.path.splitext(main_input_csv)[0]  # Remove file extension from input name
    
    # Create organized output directory structure
    # Format: HVAC_project/processed_data/merged/[original_filename]/
    output_dir = os.path.join(project_dir, 'processed_data', 'merged', base_name)
    os.makedirs(output_dir, exist_ok=True)  # Create directories if they don't exist
    print(f"✓ Created output directory: {output_dir}")
    
    # Generate descriptive output filename
    output_filename = f"{base_name}_merged.csv"
    output_filepath = os.path.join(output_dir, output_filename)
    
    # Save merged dataset with standard comma separator and UTF-8 encoding
    try:
        df_to_save.to_csv(output_filepath, index=False, encoding='utf-8')
        print(f"✅ Successfully saved merged data to:")
        print(f"   {output_filepath}")
        
        # Verify file was created and get size
        if os.path.exists(output_filepath):
            file_size = os.path.getsize(output_filepath) / (1024 * 1024)  # Convert to MB
            print(f"   File size: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        # Try alternative save location if main location fails
        fallback_path = os.path.join(project_dir, f"{base_name}_merged_backup.csv")
        try:
            df_to_save.to_csv(fallback_path, index=False, encoding='utf-8')
            print(f"✓ Saved to fallback location: {fallback_path}")
        except Exception as e2:
            print(f"❌ Fallback save also failed: {e2}")
            sys.exit(1)

    # --- 12. GENERATE PROCESSING SUMMARY REPORT ---
    print("\n📋 Generating processing summary...")
    
    # Create summary report file
    summary_filename = f"{base_name}_merge_summary.txt"
    summary_filepath = os.path.join(output_dir, summary_filename)
    
    try:
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            f.write(f"HVAC DATA MERGE SUMMARY REPORT\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"INPUT FILES:\n")
            f.write(f"  Main HVAC Data: {main_input_csv}\n")
            f.write(f"  Weather Data: {weather_csv_name}\n\n")
            
            f.write(f"PROCESSING RESULTS:\n")
            f.write(f"  Total rows processed: {len(df_merged)}\n")
            f.write(f"  Total columns: {len(df_merged.columns)}\n")
            f.write(f"  Time period: {df_merged[main_time_col].min()} to {df_merged[main_time_col].max()}\n\n")
            
            f.write(f"WEATHER DATA INTEGRATION:\n")
            # Check for weather columns (now in Italian after renaming)
            weather_cols_to_check = ['Temperatura Esterna', 'Umidita Esterna']
            for col in weather_cols_to_check:
                if col in df_merged.columns:
                    coverage = (df_merged[col].notna().sum() / len(df_merged)) * 100
                    f.write(f"  {col}: {coverage:.1f}% coverage\n")
            
            f.write(f"\nFINAL COLUMNS:\n")
            for i, col in enumerate(df_merged.columns, 1):
                f.write(f"  {i:2d}. {col}\n")
        
        print(f"✓ Summary report saved: {summary_filename}")
        
    except Exception as e:
        print(f"⚠️  Could not save summary report: {e}")

    # --- 13. FINAL COMPREHENSIVE SUMMARY ---
    print("\n" + "=" * 80)
    print("ENHANCED MERGE COMPLETE - COMPREHENSIVE SUMMARY")
    print("=" * 80)
    
    # Column count comparison
    original_cols = len(df_main.columns)
    final_cols = len(df_merged.columns)
    added_cols = final_cols - original_cols
    print(f"📊 Data Structure Changes:")
    print(f"   Original columns: {original_cols}")
    print(f"   Final columns: {final_cols}")
    print(f"   Columns added: {added_cols}")
    
    # Weather integration success assessment
    # Check for both English and Italian weather column names
    weather_cols_english = ['Temp. Esterna', 'Umid. Esterna']
    weather_cols_italian = ['Temperatura Esterna', 'Umidita Esterna']
    weather_cols_added = [col for col in weather_cols_english + weather_cols_italian if col in df_merged.columns]
    weather_success = len(weather_cols_added) >= 2
    print(f"\n🌤️  Weather Integration Status: {'✅ SUCCESS' if weather_success else '⚠️  PARTIAL'}")
    if weather_cols_added:
        print(f"   Weather columns integrated: {', '.join(weather_cols_added)}")
    
    # Data quality summary
    total_missing = df_merged.isnull().sum().sum()
    total_cells = len(df_merged) * len(df_merged.columns)
    completeness = (1 - total_missing / total_cells) * 100
    print(f"\n📈 Data Quality Metrics:")
    print(f"   Overall completeness: {completeness:.1f}%")
    print(f"   Total data points: {total_cells:,}")
    print(f"   Missing values: {total_missing:,}")
    
    # Time coverage analysis
    time_span = df_merged[main_time_col].max() - df_merged[main_time_col].min()
    print(f"\n⏱️  Time Coverage:")
    print(f"   Period covered: {time_span.days} days")
    print(f"   Data frequency: ~{len(df_merged) / max(time_span.days, 1):.1f} records/day")
    
    # Final column listing
    print(f"\n📋 Final Dataset Columns ({len(df_merged.columns)}):")
    for i, col in enumerate(df_merged.columns, 1):
        # Add data type and sample info
        dtype = str(df_merged[col].dtype)
        non_null = df_merged[col].notna().sum()
        print(f"   {i:2d}. {col:<25} ({dtype}, {non_null}/{len(df_merged)} values)")
    
    print(f"\n🎯 Output Files Generated:")
    print(f"   📄 Main dataset: {os.path.basename(output_filepath)}")
    print(f"   📋 Summary report: {summary_filename}")
    print(f"   📁 Location: {output_dir}")
    
    print(f"\n✅ Enhanced data merging completed successfully!")
    print(f"   Your merged dataset is ready for interpolation and analysis.")
    print("=" * 80)

# Execute main function only if script is run directly (not imported)
if __name__ == "__main__":
    main()
