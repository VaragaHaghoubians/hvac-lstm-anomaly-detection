"""
HVAC Data Carpet Plot Generator
==============================

This script creates carpet plots (heatmaps) for HVAC data analysis.
A carpet plot shows the hourly patterns across all days in your dataset,
making it easy to identify operational patterns, peak usage times, and anomalies.

Features:
- Combined carpet plots with time series: 2D heatmaps and linear time series in one image
- Temporal plots: Traditional time series to help understand data transformation
- Daily pattern plots: 24-hour cycle analysis with statistical insights
- Comprehensive error handling and user feedback
- Configurable output settings and date filtering

Author: HVAC Analysis System
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import configparser
import os
from datetime import datetime, timedelta
import warnings
import traceback
from pathlib import Path
import sys

warnings.filterwarnings('ignore')

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

# Add parent directories to path to import config_utils
# From scripts/2_data_visualization/ -> scripts/ -> project_root/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Try to import config_utils, with fallback if not available (quiet)
try:
    from config_utils import load_config
except ModuleNotFoundError:
    def load_config(config_path):
        """Fallback function to load config WITH interpolation support."""
        config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        config.read(config_path, encoding='utf-8-sig')
        return config

class HVACCarpetPlotGenerator:
    """
    A comprehensive class for generating carpet plots from HVAC interpolated data.
    
    This class handles:
    - Reading interpolated HVAC data
    - Processing timestamps and creating hourly/daily grids
    - Generating combined carpet plots with time series
    - Creating temporal plots to help understand data transformation
    - Creating 24-hour daily pattern analysis
    - Saving plots with organized directory structure
    """
    
    def __init__(self, config_path):
        """
        Initialize the carpet plot generator with configuration settings.
        
        Args:
            config_path (str): Path to the configuration file
        """
        # Validate config file exists
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        print(f"[INFO] Using config: {config_file}")
        
        # Use the load_config helper (with or without config_utils)
        self.config = load_config(str(config_file))
        # Record config directory to resolve relative paths (e.g., ./processed_data)
        try:
            self.config._config_dir = str(config_file.parent)
        except Exception:
            pass
        
        # Extract configuration parameters
        self.setup_paths()
        self.setup_data_parameters()
        self.setup_carpet_plot_parameters()
        
        print("HVAC Carpet Plot Generator initialized successfully!")
        print(f"Input file: {self.input_file_path}")
        print(f"Output directory: {self.output_plots_dir}")
    
    def setup_paths(self):
        """
        Set up all the file paths based on configuration settings.
        This method constructs the complete directory structure for input and output files.
        """
        # Get base paths from config
        self.base_folder = self.config.get('paths', 'base_folder')
        self.processed_folder = self.config.get('paths', 'processed_folder')
        self.plots_folder = self.config.get('paths', 'plots_folder')

        # Resolve relative folders against config.ini directory (not current working dir)
        cfg_dir = getattr(self.config, '_config_dir', None)
        if cfg_dir:
            if self.base_folder and not os.path.isabs(self.base_folder):
                self.base_folder = os.path.abspath(os.path.join(cfg_dir, self.base_folder))
            if self.processed_folder and not os.path.isabs(self.processed_folder):
                self.processed_folder = os.path.abspath(os.path.join(cfg_dir, self.processed_folder))
            if self.plots_folder and not os.path.isabs(self.plots_folder):
                self.plots_folder = os.path.abspath(os.path.join(cfg_dir, self.plots_folder))
        
        # Get input CSV name (without extension for folder naming)
        self.input_csv = self.config.get('paths', 'input_csv')
        self.csv_name_without_ext = os.path.splitext(self.input_csv)[0]
        
        # Construct the interpolated file path
        # Format: processed_data/interpolation/csv_name/csv_name_interpolated.csv
        self.input_file_path = os.path.join(
            self.processed_folder,
            'interpolation',
            self.csv_name_without_ext,
            f"{self.csv_name_without_ext}_interpolated.csv"
        )
        # Also support the alternative filename with masks if the main one is missing
        alt_input_file = os.path.join(
            self.processed_folder,
            'interpolation',
            self.csv_name_without_ext,
            f"{self.csv_name_without_ext}_interpolated_with_masks.csv"
        )
        if not os.path.exists(self.input_file_path) and os.path.exists(alt_input_file):
            self.input_file_path = alt_input_file
        
        # Construct output directory for carpet plots
        # Format: plots/carpet_plot/csv_name/
        self.output_plots_dir = os.path.join(
            self.plots_folder,
            'carpet_plot',
            self.csv_name_without_ext
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_plots_dir, exist_ok=True)
    
    def setup_data_parameters(self):
        """
        Extract data-related parameters from the configuration file.
        These parameters define how to read and process the timestamp data.
        """
        self.time_column = self.config.get('data', 'time_column')
        
        # Get all columns that we want to create carpet plots for
        # This includes temperature, modulation, and weather columns
        temp_cols = [col.strip() for col in self.config.get('data', 'temperature_columns').split(',')]
        mod_cols = [col.strip() for col in self.config.get('data', 'modulation_columns').split(',')]
        
        # Combine all columns for processing
        self.columns_to_plot = temp_cols + mod_cols
        
        print(f"Columns to process: {self.columns_to_plot}")
    
    def setup_carpet_plot_parameters(self):
        """
        Extract carpet plot specific parameters from configuration.
        These control the appearance and behavior of the plots.
        """
        # Plot creation settings
        self.enable_carpet_plot = self.config.getboolean('carpet_plot', 'create_carpet_plot')
        self.create_combined_plots = self.config.getboolean('carpet_plot', 'create_combined_plots', fallback=True)
        
        # Date range settings (optional - can be empty for full range)
        try:
            start_date_str = self.config.get('carpet_plot', 'start_date')
            if start_date_str:
                self.start_date = pd.to_datetime(start_date_str)
            else:
                self.start_date = None
        except:
            self.start_date = None
            
        try:
            end_date_str = self.config.get('carpet_plot', 'end_date')
            if end_date_str:
                self.end_date = pd.to_datetime(end_date_str)
            else:
                self.end_date = None
        except:
            self.end_date = None
        
        # Plot appearance settings
        self.plot_title_template = self.config.get('carpet_plot', 'plot_title', fallback='Hourly Profile for {column}')
        self.colormap = self.config.get('carpet_plot', 'colormap', fallback='YlOrRd')
        
        # Plot quality settings
        self.plot_dpi = self.config.getint('visualization', 'plot_dpi', fallback=300)
        self.plot_format = self.config.get('visualization', 'plot_format', fallback='png')
        
        # Word report generation settings
        self.create_word_report = self.config.getboolean('carpet_plot', 'create_word_report', fallback=True)
    
    def is_setpoint_or_command_column(self, column_name):
        """
        Determine if a column represents a setpoint or command (step-like variable).
        
        Step-like variables should use median or last aggregation instead of mean,
        since they represent control signals that change in discrete steps.
        
        Args:
            column_name (str): Name of the column to check
            
        Returns:
            bool: True if column is a setpoint or command, False otherwise
        """
        # Common patterns for setpoints and commands in HVAC systems
        setpoint_patterns = [
            'setpoint', 'sp', '_set', 'set_', 'target',  # Temperature/pressure setpoints
            'command', 'cmd', 'control',                  # Control commands
            'fan', 'pump', 'valve',                       # Equipment commands (often discrete)
            'mode', 'status', 'state',                    # Operating modes/states
            'enable', 'disable', 'on', 'off'             # Binary control signals
        ]
        
        # Check if any pattern appears in the column name (case-insensitive)
        column_lower = column_name.lower()
        for pattern in setpoint_patterns:
            if pattern in column_lower:
                return True
        
        return False
    
    def get_aggregation_method(self, column_name, data_series):
        """
        Determine the appropriate aggregation method for a column.
        
        Args:
            column_name (str): Name of the column
            data_series (pandas.Series): Data to analyze
            
        Returns:
            str: Aggregation method ('median', 'last', or 'mean')
        """
        # Check if it's a setpoint or command column
        if self.is_setpoint_or_command_column(column_name):
            # For step-like variables, use median (robust to outliers)
            # or 'last' if you want the most recent command value
            return 'median'
        else:
            # For continuous sensor readings, use mean
            return 'mean'
    
    def load_and_prepare_data(self):
        """
        Load the interpolated HVAC data and prepare it for carpet plot generation.
        
        This method:
        1. Loads the CSV file
        2. Processes timestamps
        3. Filters data by date range if specified
        4. Validates that required columns exist
        
        Returns:
            pandas.DataFrame: Prepared data ready for plotting
        """
        print(f"\nLoading data from: {self.input_file_path}")
        
        # Check if input file exists
        if not os.path.exists(self.input_file_path):
            raise FileNotFoundError(f"Interpolated data file not found: {self.input_file_path}")
        
        # Load the data
        try:
            df = pd.read_csv(self.input_file_path)
            print(f"Data loaded successfully! Shape: {df.shape}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
        
        # Convert timestamp column to datetime
        # Use utc=True to handle mixed-offset strings (+02:00/+01:00 from DST transitions)
        # which would otherwise produce an object-dtype column and break .dt accessor.
        try:
            parsed = pd.to_datetime(df[self.time_column], utc=True, errors='coerce')
            if parsed.isna().all():
                # Fallback: tz-naive CSV (no offset strings)
                parsed = pd.to_datetime(df[self.time_column], errors='coerce')
            elif parsed.dt.tz is not None:
                parsed = parsed.dt.tz_convert('Europe/Rome')
            df[self.time_column] = parsed
            print(f"Timestamp column '{self.time_column}' converted to datetime")
        except Exception as e:
            raise Exception(f"Error converting timestamp column: {str(e)}")
        
        # Sort by timestamp to ensure proper order
        df = df.sort_values(self.time_column).reset_index(drop=True)
        
        # Apply date filtering if specified in config
        if self.start_date or self.end_date:
            original_length = len(df)
            
            # Check if the datetime column is timezone-aware
            if df[self.time_column].dt.tz is not None:
                print(f"  Detected timezone-aware data: {df[self.time_column].dt.tz}")
                # Localize start_date and end_date to match the data's timezone
                if self.start_date:
                    self.start_date = pd.Timestamp(self.start_date).tz_localize(df[self.time_column].dt.tz)
                if self.end_date:
                    self.end_date = pd.Timestamp(self.end_date).tz_localize(df[self.time_column].dt.tz)
            
            if self.start_date:
                df = df[df[self.time_column] >= self.start_date]
                print(f"Applied start date filter: {self.start_date}")
            
            if self.end_date:
                df = df[df[self.time_column] <= self.end_date]
                print(f"Applied end date filter: {self.end_date}")
            
            print(f"Data filtered: {original_length} -> {len(df)} rows")
        
        # Convert timezone-aware timestamps to naive for plotting compatibility
        if df[self.time_column].dt.tz is not None:
            print(f"  Converting timezone-aware timestamps to naive (local time representation)")
            df[self.time_column] = df[self.time_column].dt.tz_localize(None)
        
        # Validate that we have data after filtering
        if len(df) == 0:
            raise Exception("No data remaining after date filtering!")
        
        # Check which columns are available for plotting
        available_columns = []
        missing_columns = []
        
        for col in self.columns_to_plot:
            if col in df.columns:
                available_columns.append(col)
            else:
                missing_columns.append(col)
        
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")
        
        if not available_columns:
            raise Exception("None of the specified columns found in the data!")
        
        self.available_columns = available_columns
        print(f"Available columns for plotting: {available_columns}")
        
        # Display data range information
        print(f"Data time range: {df[self.time_column].min()} to {df[self.time_column].max()}")
        
        return df
    
    def create_hourly_grid(self, df, column):
        """
        Transform the time series data into a 2D grid for carpet plot visualization.
        
        This is the core transformation that converts your linear time series into
        a matrix where:
        - Rows represent hours of the day (0-23, displayed as 1-24)
        - Columns represent days in your dataset (each day gets one column)
        - Values are the actual measurements for that specific hour on that specific day
        
        Args:
            df (pandas.DataFrame): Input data with timestamps
            column (str): Column name to create grid for
            
        Returns:
            tuple: (grid_data, date_range, hour_range) for plotting
        """
        print(f"\n{'='*50}")
        print(f"CREATING HOURLY GRID FOR: {column}")
        print(f"{'='*50}")
        
        # Create copies to avoid modifying original data
        data = df.copy()
        
        # Extract date and hour components from timestamps
        print("Step 1: Extracting date and hour information from timestamps...")
        data['date'] = data[self.time_column].dt.date  # Extract just the date part
        data['hour'] = data[self.time_column].dt.hour  # Extract just the hour part (0-23)
        
        print(f"   - Example timestamp: {data[self.time_column].iloc[0]}")
        print(f"   - Extracted date: {data['date'].iloc[0]}")
        print(f"   - Extracted hour: {data['hour'].iloc[0]}")
        
        # Create the date range (X-axis of our grid)
        print("\nStep 2: Creating complete date range for the grid...")
        date_range = pd.date_range(
            start=data['date'].min(),    # First date in our data
            end=data['date'].max(),      # Last date in our data
            freq='D'                     # 'D' means daily frequency
        ).date
        
        # Create the hour range (Y-axis of our grid)
        hour_range = np.arange(24)  # Creates [0, 1, 2, ..., 23]
        
        # Display grid dimensions and structure
        print(f"\n📊 GRID STRUCTURE INFORMATION:")
        print(f"   🗓️  Date range: {len(date_range)} days")
        print(f"       └─ From: {date_range[0]} (first day)")
        print(f"       └─ To:   {date_range[-1]} (last day)")
        print(f"   🕐 Hour range: {len(hour_range)} hours per day")
        print(f"   📏 Total grid size: {len(hour_range)} rows × {len(date_range)} columns")
        print(f"       └─ That's {24 * len(date_range):,} individual cells in our grid!")
        
        # Initialize the empty grid
        print(f"\nStep 3: Initializing empty grid...")
        grid_data = np.full((24, len(date_range)), np.nan)
        print(f"   - Created empty grid with shape: {grid_data.shape}")
        
        # Fill the grid with actual data
        print(f"\nStep 4: Filling grid with actual {column} measurements...")
        
        # Determine appropriate aggregation method based on column type
        agg_method = self.get_aggregation_method(column, data[column])
        print(f"   - Using '{agg_method}' aggregation for {column}")
        print(f"     (Step-like variables use median; continuous sensors use mean)")
        
        # Apply the selected aggregation method
        grouped = data.groupby(['date', 'hour'])[column].agg(agg_method)
        
        print(f"   - Found {len(grouped)} unique date-hour combinations in your data")
        
        # Fill the grid systematically
        filled_cells = 0
        for (date, hour), value in grouped.items():
            # Find the position of this date in our date range
            date_idx = np.where(date_range == date)[0]
            if len(date_idx) > 0:
                # Place the value in the correct position: [hour, day_index]
                grid_data[hour, date_idx[0]] = value
                filled_cells += 1
        
        print(f"   - Successfully filled {filled_cells} cells with data")
        
        # Calculate and display statistics
        total_possible_points = 24 * len(date_range)
        actual_data_points = np.sum(~np.isnan(grid_data))  # Count non-NaN values
        coverage_percent = (actual_data_points / total_possible_points) * 100
        
        print(f"\n📈 DATA COVERAGE STATISTICS:")
        print(f"   ✅ Cells with data: {actual_data_points:,}")
        print(f"   📊 Total possible cells: {total_possible_points:,}")
        print(f"   🎯 Coverage percentage: {coverage_percent:.1f}%")
        
        print(f"\n{'='*50}")
        print(f"GRID CREATION COMPLETED SUCCESSFULLY!")
        print(f"{'='*50}")
        
        return grid_data, date_range, hour_range
    
    def create_combined_carpet_plot(self, df, grid_data, date_range, hour_range, column):
        """
        Create a combined plot with both carpet plot and time series in one image.
        
        This method creates a single image containing:
        1. Top subplot: Carpet plot (2D heatmap) - "Hourly HVAC Profile"
        2. Bottom subplot: Complete time series
        
        Args:
            df (pandas.DataFrame): Input data with timestamps
            grid_data (numpy.array): 2D array of values for carpet plot
            date_range (pandas.DatetimeIndex): Range of dates for x-axis
            hour_range (numpy.array): Range of hours for y-axis
            column (str): Column name being plotted
        """
        print(f"\n🔥📈 Creating combined carpet plot + time series for: {column}")
        
        try:
            # Set up the figure with two subplots (top and bottom)
            fig_width = max(16, len(date_range) * 0.08)
            fig_height = 12

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, fig_height), dpi=self.plot_dpi)

            # =================================================================
            # TOP SUBPLOT: CARPET PLOT (2D HEATMAP) - "Hourly HVAC Profile"
            # =================================================================
            print("   🔥 Creating carpet plot (top subplot)...")

            # Check for any NaN or infinite values that might cause issues
            if np.all(np.isnan(grid_data)):
                print(f"   ⚠️  Warning: All data is NaN for column '{column}'. Creating empty carpet plot.")
                ax1.text(0.5, 0.5, f'No Data Available\nfor {column}',
                         horizontalalignment='center', verticalalignment='center',
                         transform=ax1.transAxes, fontsize=16)
            else:
                # Create the carpet plot heatmap
                # Flip the data vertically so hour 1 appears at the bottom
                im = ax1.imshow(
                    np.flipud(grid_data),  # Flip so hour 1 is at bottom
                    cmap=self.colormap,
                    aspect='auto',
                    interpolation='nearest',
                    extent=[0, len(date_range), 0, 24]  # Set the coordinate system
                )

                # Add colorbar with proper labeling
                cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
                cbar.set_label(f'{column} Values', rotation=270, labelpad=20)

            # Set up x-axis (dates) for carpet plot
            n_dates = len(date_range)
            if n_dates <= 30:
                tick_interval = max(1, n_dates // 10)
            else:
                tick_interval = max(7, n_dates // 20)

            tick_positions = np.arange(0, n_dates, tick_interval)
            tick_labels = [date_range[i].strftime('%m-%d') for i in tick_positions]

            ax1.set_xticks(tick_positions)
            ax1.set_xticklabels(tick_labels, rotation=45, ha='right')
            ax1.set_xlabel('Date (MM-DD)')

            # Set up y-axis (hours) for carpet plot
            ax1.set_yticks(np.arange(0.5, 24.5, 1))  # Position ticks in center of each hour
            ax1.set_yticklabels(np.arange(1, 25))    # Label as 1-24
            ax1.set_ylabel('Hour of Day')

            # Add title for carpet plot (improved readability)
            ax1.set_title(f'Hourly HVAC Profile - {column}', fontsize=22, fontweight='bold', color='navy', pad=28)

            # Add grid for better readability
            ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

            # =================================================================
            # BOTTOM SUBPLOT: COMPLETE TIME SERIES
            # =================================================================
            print("   📈 Creating complete time series (bottom subplot)...")

            # Plot the complete time series
            ax2.plot(df[self.time_column], df[column],
                     linewidth=0.8, alpha=0.8, color='blue', label='Complete Time Series')

            # =================================================================
            # ENHANCED: HIGHLIGHT GAPS ("BUCCHI") IN TIME SERIES
            # =================================================================
            print("   🔴 Detecting and highlighting gaps ('bucchi') in the time series...")
            
            # Calculate the time difference between consecutive data points
            time_diffs = df[self.time_column].diff()
            
            # Define a threshold for what constitutes a "gap"
            gap_threshold = timedelta(hours=2)
            
            # Find the data points that appear immediately *after* a gap
            gap_mask = time_diffs > gap_threshold
            points_after_gap = df[gap_mask]
            
            if not points_after_gap.empty:
                # Get indices and corresponding data points before the gap
                indices_after = points_after_gap.index
                indices_before = indices_after - 1
                points_before_gap = df.loc[indices_before]
                
                # Plot highly visible markers at gap boundaries
                # RED CIRCLES for gap start
                ax2.plot(points_before_gap[self.time_column], 
                         points_before_gap[column],
                         'o', 
                         markerfacecolor='red', 
                         markeredgecolor='black',
                         markersize=10, 
                         markeredgewidth=1.5,
                         label='Gap Start', 
                         zorder=10)
                
                # ORANGE TRIANGLES for gap end/data resumption
                ax2.plot(points_after_gap[self.time_column], 
                         points_after_gap[column],
                         '^', 
                         markerfacecolor='orange',
                         markeredgecolor='black',
                         markersize=10,
                         markeredgewidth=1.5,
                         label='Gap End (Data Resumes)', 
                         zorder=10)
                
                # Add shaded regions for the gap period
                for _, row_before in points_before_gap.iterrows():
                    start_time = row_before[self.time_column]
                    # Find the corresponding point after the gap
                    end_time = points_after_gap[points_after_gap.index == row_before.name + 1][self.time_column].iloc[0]
                    ax2.axvspan(start_time, end_time, 
                                alpha=0.2, 
                                color='red', 
                                zorder=1)

                print(f"      ✓ Found and marked {len(points_after_gap)} gaps in the data.")
            else:
                print("      - No significant gaps found in the time series data.")
            # =================================================================

            # Add vertical lines to show monthly boundaries for better time reference
            start_time = df[self.time_column].min()
            end_time = df[self.time_column].max()

            # Create monthly markers for better time reference
            monthly_markers = pd.date_range(
                start=start_time.normalize().replace(day=1),  # Start at first of month
                end=end_time.normalize(),
                freq='MS'  # Month start frequency
            )

            # Add vertical lines for each month
            for month_marker in monthly_markers:
                if start_time <= month_marker <= end_time:
                    ax2.axvline(x=month_marker, color='gray', alpha=0.4,
                                linestyle='--', linewidth=1)

                    # Add month labels at the top
                    month_label = month_marker.strftime('%B\n%Y')
                    ax2.text(month_marker, ax2.get_ylim()[1] * 0.95, month_label,
                             ha='center', va='top', fontsize=10, fontweight='bold',
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            # Set up time series plot formatting
            ax2.set_title(f'Complete Time Series - {column}', fontsize=16, fontweight='bold')
            ax2.set_xlabel('Date and Time')
            ax2.set_ylabel(f'{column} Values')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper right')


            # Remove x-axis tick labels for time series
            ax2.tick_params(axis='x', labelbottom=False)


            # Add concise one-line explanation beneath the figure (to avoid overlap with date labels)
            days_span = (end_time - start_time).days
            explanation_line = f'📊 {days_span} days of data — carpet plot shows daily cycles as columns; gray dashed lines = month starts'
            plt.tight_layout()
            plt.subplots_adjust(top=0.92, bottom=0.12)
            fig.text(0.5, 0.04, explanation_line, ha='center', va='center', fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.9))

            # =================================================================
            # OVERALL FORMATTING AND SAVE
            # =================================================================
            # Add main title for the entire figure (improved readability)
            fig.suptitle(f'HVAC Analysis: Carpet Plot + Time Series - {column}',
                         fontsize=26, fontweight='heavy', color='darkred', y=0.98)

            # Improve layout to prevent overlap
            # Adjust layout so carpet plot is directly under the main title
            plt.tight_layout(rect=[0, 0, 1, 0.90])
            plt.subplots_adjust(top=0.90)  # More space for title, carpet plot starts just below

            # Save the combined plot in the main carpet plot directory
            filename = f"{column.replace(' ', '_').replace('.', '_')}_combined_plot.{self.plot_format}"
            output_path = os.path.join(self.output_plots_dir, filename)

            plt.savefig(
                output_path,
                dpi=self.plot_dpi,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none'
            )

            print(f"   ✅ Combined plot saved: {output_path}")

            # Print combined plot statistics
            print(f"\n   📊 COMBINED PLOT STATISTICS:")
            print(f"   🔥 Carpet plot dimensions: {grid_data.shape[0]} hours × {grid_data.shape[1]} days")
            print(f"   📈 Time series data points: {len(df):,}")
            print(f"   📅 Date range: {days_span} days")
            print(f"   🎯 This combined view shows both the big picture and detailed patterns!")

            # Close the figure to free memory
            plt.close(fig)

        except Exception as e:
            print(f"   ❌ Error creating combined plot for '{column}': {str(e)}")
            plt.close('all')
            raise
    
    def set_cell_border(self, cell, **kwargs):
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
    
    def get_plot_description(self, column_name):
        """
        Generate a description for a specific plot/column.
        This creates explanatory text about what the plot shows.
        
        Args:
            column_name (str): Name of the column/variable
            
        Returns:
            str: Description text for the plot
        """
        # Create contextual descriptions based on column name
        description = f"Il grafico mostra l'analisi del profilo orario per '{column_name}'. "
        
        # Add specific descriptions based on variable type
        if 'temp' in column_name.lower() or 'temperatura' in column_name.lower():
            description += (
                "La parte superiore (carpet plot) visualizza la distribuzione oraria dei valori di temperatura "
                "attraverso tutti i giorni del periodo analizzato. Ogni colonna rappresenta un giorno, "
                "e ogni riga rappresenta un'ora del giorno (1-24). "
                "La scala di colori indica l'intensità della temperatura misurata. "
                "\n\n"
                "La parte inferiore mostra la serie temporale completa dei dati, "
                "permettendo di identificare trend, pattern stagionali e eventuali anomalie. "
                "I cerchi rossi indicano l'inizio di interruzioni nei dati (gaps), "
                "mentre i triangoli arancioni indicano la ripresa della raccolta dati dopo un'interruzione."
            )
        elif 'mod' in column_name.lower() or 'modulazione' in column_name.lower():
            description += (
                "La parte superiore (carpet plot) visualizza la distribuzione oraria della modulazione "
                "attraverso tutti i giorni del periodo analizzato. Ogni colonna rappresenta un giorno, "
                "e ogni riga rappresenta un'ora del giorno (1-24). "
                "La scala di colori indica il livello di modulazione del sistema. "
                "\n\n"
                "La parte inferiore mostra la serie temporale completa dei dati di modulazione, "
                "permettendo di analizzare i pattern operativi del sistema HVAC nel tempo. "
                "I cerchi rossi indicano l'inizio di interruzioni nei dati (gaps), "
                "mentre i triangoli arancioni indicano la ripresa della raccolta dati dopo un'interruzione."
            )
        else:
            description += (
                "La parte superiore (carpet plot) visualizza la distribuzione oraria dei valori "
                "attraverso tutti i giorni del periodo analizzato. Ogni colonna rappresenta un giorno, "
                "e ogni riga rappresenta un'ora del giorno (1-24). "
                "\n\n"
                "La parte inferiore mostra la serie temporale completa dei dati. "
                "I cerchi rossi indicano l'inizio di interruzioni nei dati (gaps), "
                "mentre i triangoli arancioni indicano la ripresa della raccolta dati dopo un'interruzione."
            )
        
        return description
    
    def get_plot_impacts(self, column_name):
        """
        Generate impact assessment text for a specific plot/column.
        This creates text explaining the implications and impacts of the data.
        
        Args:
            column_name (str): Name of the column/variable
            
        Returns:
            str: Impact assessment text for the plot
        """
        # Create contextual impact assessments based on column name
        impacts = ""
        
        if 'temp' in column_name.lower() or 'temperatura' in column_name.lower():
            impacts = (
                "Valori di temperatura non adeguati comportano un aumento dei consumi energetici "
                "e una riduzione del comfort percepito dagli occupanti. "
                "L'analisi dei pattern orari permette di identificare:\n"
                "• Periodi di sovra-riscaldamento o sotto-riscaldamento\n"
                "• Opportunità di ottimizzazione dei setpoint\n"
                "• Necessità di aggiustamenti nella programmazione oraria\n"
                "• Anomalie nel funzionamento del sistema di controllo"
            )
        elif 'mod' in column_name.lower() or 'modulazione' in column_name.lower():
            impacts = (
                "L'analisi della modulazione fornisce informazioni cruciali sull'efficienza operativa del sistema:\n"
                "• Pattern di modulazione elevata indicano alta richiesta energetica\n"
                "• Modulazione costante al massimo può indicare problemi di dimensionamento\n"
                "• Variazioni rapide possono segnalare instabilità nel controllo\n"
                "• Pattern regolari indicano un funzionamento efficiente del sistema"
            )
        else:
            impacts = (
                "L'analisi di questo parametro permette di:\n"
                "• Identificare pattern operativi e anomalie\n"
                "• Ottimizzare le strategie di controllo\n"
                "• Migliorare l'efficienza energetica complessiva\n"
                "• Garantire il comfort degli occupanti"
            )
        
        return impacts
    
    def generate_word_report(self, df, plot_images):
        """
        Generate a professional Word document report with all plots and descriptions.
        This creates a report similar to the format shown in the user's example image.
        
        The report includes:
        - Main title and date range at the top
        - For each plot: a table with description, image, and impact assessment
        
        Args:
            df (pandas.DataFrame): The data used for plotting (to extract date range)
            plot_images (list): List of tuples (column_name, image_path) for plots to include
        """
        # Check if python-docx is available
        if not DOCX_AVAILABLE:
            print("\n[WARNING] Cannot generate Word report - python-docx library not installed")
            print("To enable Word report generation, run: pip install python-docx")
            return None
        
        if not self.create_word_report:
            print("\n[INFO] Word report generation is disabled in configuration")
            return None
        
        print("\n" + "="*60)
        print("GENERATING WORD REPORT")
        print("="*60)
        
        try:
            # Create a new Word document
            doc = Document()
            
            # Set document margins (narrower for more space)
            sections = doc.sections
            for section in sections:
                section.top_margin = Inches(0.5)
                section.bottom_margin = Inches(0.5)
                section.left_margin = Inches(0.75)
                section.right_margin = Inches(0.75)
            
            # Get date range from data
            start_date = df[self.time_column].min()
            end_date = df[self.time_column].max()
            
            # Format date range string
            if start_date and end_date:
                date_range_str = f"Periodo di analisi: {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}"
            else:
                date_range_str = "Periodo di analisi: Non disponibile"
            
            # Add main title (similar to the image you provided)
            title = doc.add_heading('Analisi HVAC - Adeguamento Range Operativa di Temperatura', level=0)
            title.alignment = WD_ALIGN_PARAGRAPH.CENTER
            title_format = title.runs[0].font
            title_format.size = Pt(16)
            title_format.bold = True
            title_format.color.rgb = RGBColor(0, 0, 0)
            
            # Add date range below title
            date_para = doc.add_paragraph(date_range_str)
            date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            date_format = date_para.runs[0].font
            date_format.size = Pt(11)
            date_format.bold = False
            date_format.color.rgb = RGBColor(128, 128, 128)
            
            # Add a blank line for spacing
            doc.add_paragraph()
            
            if not plot_images:
                print("\n[WARNING] No plot images provided for Word report")
                return None
            
            # Process each plot and create a table for it
            for idx, (column_name, image_path) in enumerate(plot_images, 1):
                print(f"\nProcessing plot {idx}/{len(plot_images)}: {column_name}")
                
                # Create a table with 3 rows and 1 column
                # Row 1: Plot title (column name)
                # Row 2: Plot image
                # Row 3: "Descrizione:" header + description text
                table = doc.add_table(rows=3, cols=1)
                table.style = 'Table Grid'
                
                # Set table width to full page width
                table.autofit = False
                table.allow_autofit = False
                
                # ROW 1: Plot title section
                cell_title = table.rows[0].cells[0]
                self.set_cell_border(cell_title, top={}, left={}, right={}, bottom={})
                
                # Set light blue background color
                tc = cell_title._element
                tcPr = tc.get_or_add_tcPr()
                shd = OxmlElement('w:shd')
                shd.set(qn('w:fill'), 'E6F2FF')
                tcPr.append(shd)
                
                # Add plot title (column name)
                title_para = cell_title.paragraphs[0]
                title_para.text = column_name
                title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                title_format = title_para.runs[0].font
                title_format.bold = True
                title_format.size = Pt(12)
                title_format.color.rgb = RGBColor(0, 51, 102)  # Dark blue color
                
                # ROW 2: Plot image
                cell_image = table.rows[1].cells[0]
                self.set_cell_border(cell_image, top={}, left={}, right={}, bottom={})
                
                # Clear default paragraph and add image
                cell_image.paragraphs[0].clear()
                
                # Add the plot image (centered)
                paragraph = cell_image.paragraphs[0]
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                
                try:
                    # Insert image with appropriate width (fit to page width with margins)
                    run = paragraph.add_run()
                    run.add_picture(image_path, width=Inches(7.0))
                    print(f"  ✓ Image added successfully")
                except Exception as e:
                    print(f"  ✗ Error adding image: {str(e)}")
                    paragraph.add_run(f"[Error loading image: {os.path.basename(image_path)}]")
                
                # ROW 3: Description section
                cell_desc = table.rows[2].cells[0]
                self.set_cell_border(cell_desc, top={}, left={}, right={}, bottom={})
                
                # Add "Descrizione:" header
                desc_header = cell_desc.paragraphs[0]
                desc_header.text = "Descrizione:"
                desc_header_format = desc_header.runs[0].font
                desc_header_format.bold = True
                desc_header_format.size = Pt(11)
                
                # Add description text
                description_text = self.get_plot_description(column_name)
                desc_para = cell_desc.add_paragraph(description_text)
                desc_para_format = desc_para.runs[0].font
                desc_para_format.size = Pt(10)
                
                # Add spacing between plots (but not after the last one)
                if idx < len(plot_images):
                    doc.add_paragraph()  # Blank line between tables
            
            # Save the Word document
            output_filename = f"Carpet_plot_Report_{self.csv_name_without_ext}.docx"
            output_path = os.path.join(self.output_plots_dir, output_filename)
            
            doc.save(output_path)
            print(f"\n{'='*60}")
            print(f"✅ SUCCESS! Word report saved to:")
            print(f"{output_path}")
            print(f"{'='*60}")
            return output_path
            
        except Exception as e:
            print(f"\n❌ Error generating Word report: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_all_carpet_plots(self):
        """
        Main method to generate carpet plots for all specified columns.
        
        This method orchestrates the entire process:
        1. Loads and prepares the data
        2. Creates combined plots (carpet plot + time series) for each column
        3. Creates temporal plots (if enabled) to help understand the data
        4. Provides summary information about the process
        """
        if not self.enable_carpet_plot:
            print("Carpet plot generation is disabled in configuration.")
            return
        
        print("=" * 60)
        print("HVAC CARPET PLOT GENERATION STARTED")
        print("=" * 60)
        
        try:
            # Load and prepare the data
            df = self.load_and_prepare_data()
            
            print(f"\nGenerating plots for {len(self.available_columns)} columns...")
            if self.create_combined_plots:
                print("🔥📈 Combined plots (Carpet + Time Series): ENABLED")
            
            successful_combined_plots = 0
            failed_plots = []
            plot_images = []  # Track successfully created plots for Word report
            
            # Generate plots for each available column
            for column in self.available_columns:
                try:
                    print(f"\n{'-' * 40}")
                    print(f"Processing column: {column}")
                    
                    # Check if column has any valid data
                    valid_data_count = df[column].notna().sum()
                    if valid_data_count == 0:
                        print(f"Warning: Column '{column}' has no valid data. Skipping.")
                        continue
                    
                    # STEP 1: Create the hourly grid for carpet plot
                    grid_data, date_range, hour_range = self.create_hourly_grid(df, column)
                    
                    # STEP 2: Generate the combined plot (Carpet + Time Series)
                    if self.create_combined_plots:
                        print("🔥📈 Creating combined plot (Carpet + Time Series)...")
                        self.create_combined_carpet_plot(df, grid_data, date_range, hour_range, column)
                        successful_combined_plots += 1
                        
                        # Track the plot image path for Word report
                        filename = f"{column.replace(' ', '_').replace('.', '_')}_combined_plot.{self.plot_format}"
                        image_path = os.path.join(self.output_plots_dir, filename)
                        if os.path.exists(image_path):
                            plot_images.append((column, image_path))
                    
                except Exception as e:
                    error_msg = f"Error processing column '{column}': {str(e)}"
                    print(f"ERROR: {error_msg}")
                    failed_plots.append((column, error_msg))
            
            # Print summary
            print("\n" + "=" * 60)
            print("CARPET PLOT GENERATION SUMMARY")
            print("=" * 60)
            print(f"Successfully generated:")
            if self.create_combined_plots:
                print(f"  🔥📈 Combined plots: {successful_combined_plots}")
            print(f"Failed: {len(failed_plots)} plots")
            
            if failed_plots:
                print("\nFailed columns:")
                for column, error in failed_plots:
                    print(f"  - {column}: {error}")
            
            if successful_combined_plots > 0:
                print(f"\nPlots saved to: {self.output_plots_dir}")
            
            # STEP 3: Generate Word report if enabled and plots were created successfully
            if plot_images and self.create_word_report and DOCX_AVAILABLE:
                print("\n" + "="*60)
                print("GENERATING WORD REPORT WITH ALL PLOTS")
                print("="*60)
                word_report_path = self.generate_word_report(df, plot_images)
                if word_report_path:
                    print(f"\n📄 Word report includes {len(plot_images)} plots")
                    print(f"📂 Report location: {word_report_path}")
            elif not DOCX_AVAILABLE and self.create_word_report:
                print("\n[INFO] Word report generation skipped - python-docx not installed")
                print("To enable: pip install python-docx")
            
            print("\n" + "="*60)
            print("Process completed!")
            print("="*60)
            
        except Exception as e:
            print(f"CRITICAL ERROR: {str(e)}")
            raise


def find_config_file():
    """
    Intelligently locate the config.ini file by searching in multiple possible locations.
    Enhanced with robust path validation.
    
    Returns:
        str: Path to the config.ini file if found
        
    Raises:
        FileNotFoundError: If config.ini cannot be found in any expected location
    """
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # List of possible locations for config.ini (using Path objects)
    possible_locations = [
        script_dir.parent.parent / 'config.ini',  # Project root (2 levels up: 2_data_visualization -> scripts -> HVAC_project)
        Path.cwd() / 'config.ini',  # Current working directory
        script_dir / 'config.ini',  # Same directory as script
    ]
    
    print("Searching for config.ini file...")
    
    for i, config_path in enumerate(possible_locations, 1):
        config_path = config_path.resolve()  # Convert to absolute path
        print(f"  {i}. Checking: {config_path}")
        
        if config_path.exists():
            print(f"  ✓ Found config.ini at: {config_path}")
            return str(config_path)
        else:
            print(f"  ✗ Not found")
    
    # If we get here, config.ini wasn't found anywhere
    print("\n" + "="*60)
    print("ERROR: config.ini file not found!")
    print("="*60)
    print("The script searched in these locations:")
    for i, path in enumerate(possible_locations, 1):
        print(f"  {i}. {path.resolve()}")
    
    print("\nTo fix this issue:")
    print("1. Make sure config.ini exists in your HVAC_project folder (project root)")
    print("2. Or copy config.ini to the same folder as this script")
    print("3. Or run this script from the folder containing config.ini")
    print(f"\n   Script location: {script_dir}")
    print(f"   Current directory: {Path.cwd()}")
    
    raise FileNotFoundError("config.ini not found in any expected location")


def main():
    """
    Main function to run the carpet plot generation.
    """
    print("="*60)
    print("HVAC CARPET PLOT GENERATOR")
    print("="*60)
    print(f"Script location: {os.path.abspath(__file__)}")
    print(f"Working directory: {os.getcwd()}")
    print()
    
    try:
        # Step 1: Find the configuration file
        config_path = find_config_file()
        print()
        
        # Step 2: Create the carpet plot generator
        print("Initializing carpet plot generator...")
        generator = HVACCarpetPlotGenerator(config_path)
        print()
        
        # Step 3: Generate all carpet plots
        generator.generate_all_carpet_plots()
        
    except FileNotFoundError as e:
        print(f"\nFILE NOT FOUND ERROR:")
        print(str(e))
        print("\nPlease follow the instructions above to resolve this issue.")
        
    except configparser.Error as e:
        print(f"\nCONFIGURATION FILE ERROR:")
        print(f"There's an issue with your config.ini file: {str(e)}")
        print("\nPlease check that:")
        print("1. The config.ini file is properly formatted")
        print("2. All required sections exist (like [paths], [data], [carpet_plot])")
        print("3. There are no syntax errors in the file")
        
    except Exception as e:
        print(f"\nUNEXPECTED ERROR:")
        print(f"An unexpected error occurred: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print("\nThis might be due to:")
        print("1. Missing or incorrect file paths in config.ini")
        print("2. Issues with the input data file")
        print("3. Permission problems with creating output directories")
        
        # For debugging purposes, show the full error traceback
        import traceback
        print("\nDETAILED ERROR INFORMATION:")
        print("-" * 40)
        traceback.print_exc()


if __name__ == "__main__":
    main()
