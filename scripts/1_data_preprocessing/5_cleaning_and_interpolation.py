"""
Data Cleaning and Interpolation Script for AHU Monitoring Data

PAPER 9 COMPLIANT - Proper Long Gap Handling:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This script follows best practices from HVAC monitoring literature:
• ALWAYS reindex to full 30-min grid (missing timestamps → NaN rows)
• Fill ONLY short gaps (<4h configurable) with appropriate methods
• Leave long gaps (≥4h) as NaN (unreliable periods kept missing)
• Maintain full traceability with interpolation masks
• Keep consistent time grid for LSTM, plots, and season comparison

KEY PRINCIPLE (Paper 9): "Unreliable periods shouldn't be treated as continuous"

This script performs comprehensive data cleaning and interpolation:
1. Loads merged CSV data (from step 1)
2. REINDEXES to complete 30-min grid (CRITICAL for time series)
3. Cleans temperature values by removing units and formatting
4. Converts modulation values to proper numeric format
5. Identifies long gaps (>max_gap_hours) - reports but does NOT delete rows
6. Applies method-specific interpolation with strict gap limits:
   - Temperatures/Humidity: time-based linear (smooth transitions)
   - Setpoints/Controls: forward-fill (step-like behavior)
7. Generates masks showing which points were interpolated
8. Saves TWO versions: with masks (traceability) and without (clean)
9. Creates visualizations comparing original vs interpolated data
10. Generates comprehensive reports for documentation

OUTPUT FILES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• *_interpolated_with_masks.csv - Full dataset with *_is_interpolated columns
• *_interpolated.csv - Clean dataset (no mask columns)
• *_long_gaps_report.csv - Documentation of gaps NOT filled
• *_interpolation_summary.json - Process metadata and statistics

Author: AHU Analysis Pipeline - Paper 9 Compliant Version
"""

# =================== IMPORT SECTION ===================
import pandas as pd                  # For data manipulation and analysis
import matplotlib.pyplot as plt      # For creating visualizations
import os                            # For file and directory operations
import configparser                  # For reading configuration settings
import sys                           # For system-specific parameters and functions
import numpy as np                   # For numerical operations
from datetime import datetime        # For timestamp operations
import time                          # For timestamp suffixes when files are locked
from scipy import stats              # For statistical operations (mode detection)

# Timezone resolution: try zoneinfo (Python 3.9+) with tzdata fallback, then pytz
try:
    from zoneinfo import ZoneInfo
    ROME_TZ = ZoneInfo('Europe/Rome')
except Exception:
    try:
        import pytz
        ROME_TZ = pytz.timezone('Europe/Rome')
    except ImportError:
        print("[WARNING] Neither 'tzdata' nor 'pytz' is installed. Install one:")
        print("  pip install tzdata   (recommended)")
        print("  pip install pytz     (alternative)")
        ROME_TZ = None

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

def generate_interpolation_description(column_name, df_original, df_interp, method_type):
    """
    Generate an Italian description for the interpolation plot.
    """
    total_points = len(df_original)
    original_missing = df_original[column_name].isna().sum()
    original_valid = total_points - original_missing
    
    # Check if mask column exists
    mask_col_name = f"{column_name.replace(' ', '_').replace('.', '')}_is_interpolated"
    if mask_col_name in df_interp.columns:
        interpolated_points = df_interp[mask_col_name].sum()
    else:
        # Fallback calculation
        interpolated_points = original_missing - df_interp[column_name].isna().sum()
    
    missing_pct = (original_missing / total_points * 100) if total_points > 0 else 0
    interp_pct = (interpolated_points / total_points * 100) if total_points > 0 else 0
    
    description = f"Questo grafico mostra il confronto tra i dati originali e quelli interpolati per la variabile '{column_name}'.\n\n"
    
    description += f"**Statistiche:**\n"
    description += f"• Punti totali nel dataset: {total_points:,}\n"
    description += f"• Punti validi originali: {original_valid:,} ({100-missing_pct:.1f}%)\n"
    description += f"• Punti mancanti originali: {original_missing:,} ({missing_pct:.1f}%)\n"
    description += f"• Punti interpolati: {interpolated_points:,} ({interp_pct:.1f}%)\n\n"
    
    description += f"**Metodo di Interpolazione:**\n"
    if method_type == 'linear':
        description += f"• Interpolazione lineare temporale (time-based)\n"
        description += f"• Questo metodo considera gli intervalli di tempo irregolari e interpola i valori mancanti creando una transizione lineare tra i punti validi adiacenti.\n"
        description += f"• Appropriato per variabili sensore che cambiano gradualmente nel tempo.\n\n"
    elif method_type == 'forward_fill':
        description += f"• Forward-fill (propagazione in avanti)\n"
        description += f"• Questo metodo mantiene l'ultimo valore valido fino al prossimo cambiamento.\n"
        description += f"• Appropriato per setpoint e valori di controllo che rimangono costanti tra le modifiche.\n\n"
    else:
        description += f"• Metodo: {method_type}\n\n"
    
    description += f"**Legenda del Grafico:**\n"
    description += f"• Punti blu: Dati originali misurati dal sensore\n"
    description += f"• Linea rossa: Serie temporale completa dopo interpolazione\n"
    description += f"• Punti verdi: Valori specifici che sono stati interpolati\n"
    description += f"• Marcatori X rossi: Limiti di gap rimossi (gap > soglia configurata)\n\n"
    
    description += f"**Interpretazione:**\n"
    if missing_pct < 5:
        description += f"✓ Eccellente completezza dei dati ({100-missing_pct:.1f}% validi). L'interpolazione ha riempito solo piccole lacune isolate.\n"
    elif missing_pct < 15:
        description += f"✓ Buona qualità dei dati ({100-missing_pct:.1f}% validi). L'interpolazione ha migliorato significativamente la continuità della serie temporale.\n"
    elif missing_pct < 30:
        description += f"⚠️ Qualità moderata dei dati ({missing_pct:.1f}% mancanti). L'interpolazione ha riempito lacune sostanziali - verificare l'affidabilità per analisi critiche.\n"
    else:
        description += f"⚠️ Alta percentuale di dati mancanti ({missing_pct:.1f}%). L'interpolazione è stata estensiva - usare con cautela per analisi quantitative.\n"
    
    return description

def create_word_report(plot_info_list, plot_folder, base_name, df_original):
    """
    Creates a Word report with interpolation plots in the same format as missing data analysis.
    Each plot is presented in a 3-row table: Title | Image | Description
    """
    if not DOCX_AVAILABLE:
        print("⚠️  python-docx not available. Skipping Word report generation.")
        return None
    
    try:
        print(f"\n{'='*60}")
        print("GENERATING WORD REPORT - DATA INTERPOLATION")
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
        start_date = df_original.index.min()
        end_date = df_original.index.max()
        date_range_str = f"Periodo di analisi: {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}"
        
        title = doc.add_heading('Interpolazione Dati - Sistema HVAC', level=0)
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
        word_path = os.path.join(plot_folder, f"{base_name}_Interpolation_Report.docx")
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

def safe_overwrite(path, df):
    """
    Safely saves a DataFrame to CSV, handling file locks by adding timestamps.

    This function prevents script crashes when output files are open in Excel
    or other applications that lock the file.

    Args:
        path (str): Target file path for saving the DataFrame
        df (pd.DataFrame): DataFrame to save

    Returns:
        None

    Side Effects:
        - Creates CSV file at specified path
        - If path is locked, creates timestamped version
        - Prints status messages to console
    """
    try:
        # Attempt to save DataFrame to the specified path
        df.to_csv(path, index=False)
        print(f"✓ Successfully saved to: {path}")
    except PermissionError:
        # Handle case where file is locked (e.g., open in Excel)
        print("⚠ File is locked, saving with a timestamp suffix instead.")

        # Split filename into base and extension
        base, ext = os.path.splitext(path)

        # Create timestamp suffix in format: _YYYYMMDD-HHMMSS
        ts_suffix = time.strftime("_%Y%m%d-%H%M%S")

        # Construct new filename with timestamp
        new_path = f"{base}{ts_suffix}{ext}"

        # Save with the new timestamped filename
        df.to_csv(new_path, index=False)
        print(f"✓ Saved as {new_path}")

def read_config(config_path="config.ini"):
    """
    Reads the configuration file and returns a ConfigParser object.

    This function handles configuration file loading with proper error handling
    for missing files and provides absolute path information for debugging.

    Args:
        config_path (str): Path to the configuration file. Defaults to "config.ini".

    Returns:
        configparser.ConfigParser: ConfigParser object containing configuration settings.

    Exits:
        If the configuration file is not found.
    """
    # Create ConfigParser instance with ExtendedInterpolation for variable expansion
    config = configparser.ConfigParser(
        allow_no_value=True,
        interpolation=configparser.ExtendedInterpolation()
    )

    # Check if configuration file exists
    if not os.path.exists(config_path):
        print(f"Error: Configuration file '{config_path}' not found.")
        print(f"Looking in: {os.path.abspath(config_path)}")
        sys.exit(1)

    # Read the configuration file
    config.read(config_path, encoding='utf-8-sig')
    # Record the directory of config.ini to resolve relative paths (e.g., ./data)
    try:
        config._config_dir = os.path.dirname(os.path.abspath(config_path))
    except Exception:
        pass
    return config

def clean_temp(series):
    """
    Cleans temperature values by removing units and converting to numeric format.

    This function handles various temperature unit formats commonly found in
    AHU monitoring data and ensures consistent numeric representation.

    Args:
        series (pandas.Series): Series containing temperature values with units

    Returns:
        pandas.Series: Series with cleaned numeric temperature values.

    Processing Steps:
        1. Convert all values to strings for consistent processing
        2. Remove ' °C' temperature unit symbols
        3. Remove ' Â°C' malformed temperature unit symbols (encoding issues)
        4. Replace European decimal commas with dots
        5. Strip whitespace from all values
        6. Convert to numeric, setting invalid values to NaN
    """
    return pd.to_numeric(
        series.astype(str)                           # Convert to string for text operations
        .str.replace(' °C', '', regex=False)         # Remove standard temperature unit
        .str.replace(' Â°C', '', regex=False)        # Remove malformed temperature unit
        .str.replace(',', '.', regex=False)          # European decimal comma to dot
        .str.strip(),                                # Remove leading/trailing whitespace
        errors='coerce'                              # Convert invalid values to NaN
    )

def clean_modulation(series):
    """
    Cleans modulation values by converting them to proper numeric format.

    Modulation values are typically percentages (0-100) representing
    the operating level of AHU components like fans or dampers.

    Args:
        series (pandas.Series): Series containing modulation values.

    Returns:
        pandas.Series: Series with cleaned numeric modulation values.

    Note:
        Uses 'coerce' error handling to convert invalid values to NaN
        rather than raising exceptions.
    """
    return pd.to_numeric(series, errors='coerce')

def get_display_name(column_name):
    """
    Converts technical column names to user-friendly display names for plots.

    This function maps internal column names (often from weather APIs or
    technical systems) to more readable names for visualization purposes.

    Args:
        column_name (str): Original technical column name

    Returns:
        str: User-friendly display name for plots and reports

    Mappings:
        - 'temp' → 'Temp. Esterna' (External Temperature)
        - 'rh' → 'Umid. Esterna' (External Humidity)
        - Other names remain unchanged
    """
    if column_name == 'temp':
        return 'Temp. Esterna'
    if column_name == 'rh':
        return 'Umid. Esterna'
    return column_name

def identify_gaps_across_columns(df, columns_to_check, max_gap_seconds):
    """
    CRITICAL FUNCTION: Identifies time gaps that exceed the threshold in ANY of the specified columns.

    This is the core implementation of the cross-column gap handling requirement.
    When ANY column has a gap larger than max_gap_hours, the boundary rows of that gap
    must be removed from ALL columns to maintain data consistency.

    Args:
        df (pd.DataFrame): DataFrame with datetime index
        columns_to_check (list): List of columns to check for gaps
        max_gap_seconds (float): Maximum allowed gap in seconds

    Returns:
        tuple: (rows_to_remove, column_gap_info)
            - rows_to_remove (pd.Series): Boolean series where True indicates rows to remove
            - column_gap_info (dict): Dictionary containing gap information for each column

    ENHANCED Cross-Column Logic:
        1. Check EACH column individually for missing data periods
        2. For ANY gap > max_gap_hours in ANY column:
           - Mark boundary rows for removal from ALL columns
        3. Also check for large time jumps in the datetime index itself
        4. Return comprehensive removal mask and detailed gap information
    """
    print("\n" + "=" * 60)
    print("ENHANCED: Checking ALL columns for cross-column gap handling")
    print("=" * 60)

    # Calculate time differences between consecutive rows to detect time jumps
    time_diffs = df.index.to_series().diff().dt.total_seconds()

    # Initialize boolean mask for rows to remove (starts with all False)
    rows_to_remove = pd.Series(False, index=df.index)

    # Dictionary to track gaps found in each column for reporting
    column_gap_info = {}

    # =================== CORE CROSS-COLUMN GAP DETECTION ===================
    # Check each column individually for missing data periods
    for col in columns_to_check:
        # Skip columns that don't exist in the DataFrame
        if col not in df.columns:
            print(f"⚠️ Column '{col}' not found in data, skipping...")
            continue

        # Create boolean mask where True = missing value for this column
        col_missing = df[col].isna()

        # Find transitions into and out of missing periods using diff()
        # diff() on boolean series: False→True = 1, True→False = -1, same→same = 0
        missing_changes = col_missing.astype(int).diff()

        # Identify start points of missing periods (transition from valid to missing)
        gap_starts = df.index[missing_changes == 1]

        # Identify end points of missing periods (transition from missing to valid)
        gap_ends = df.index[missing_changes == -1]

        # =================== HANDLE EDGE CASES ===================
        # If first value is missing, add start of dataset as gap start
        if col_missing.iloc[0]:
            gap_starts = gap_starts.insert(0, df.index[0])

        # If last value is missing, add end of dataset as gap end
        if col_missing.iloc[-1]:
            gap_ends = gap_ends.append(pd.Index([df.index[-1]]))

        # Ensure we have matching pairs of starts and ends
        min_len = min(len(gap_starts), len(gap_ends))
        gap_starts = gap_starts[:min_len]
        gap_ends = gap_ends[:min_len]

        # =================== PROCESS EACH GAP IN THIS COLUMN ===================
        gaps_found = []
        for start, end in zip(gap_starts, gap_ends):
            # Calculate gap duration in seconds
            gap_duration = (end - start).total_seconds()

            # CRITICAL: If ANY column has a gap > max_gap_hours, mark for removal
            if gap_duration > max_gap_seconds:
                # Store gap information for reporting
                gaps_found.append({
                    'start': start,
                    'end': end,
                    'duration_hours': gap_duration / 3600
                })

                # =================== MARK BOUNDARY ROWS FOR REMOVAL ===================
                # This is the key cross-column logic: remove boundaries from ALL columns
                #
                # ⚠️  DESIGN DECISION: Aggressive Boundary Removal
                # We remove 4 rows per gap:
                #   1. Row BEFORE gap starts
                #   2. FIRST row of gap
                #   3. LAST row of gap
                #   4. Row AFTER gap ends
                #
                # Rationale: Prevents interpolation across uncertain boundaries where
                # sensor behavior may be transitioning or unstable.
                #
                # Trade-off: May remove more data than strictly necessary. If your dataset
                # has many small gaps, this can be aggressive. Consider adjusting if needed.
                # The removal count is prominently reported in the summary.
                # =============================================================================

                # Find DataFrame row indices for gap boundaries
                start_idx = df.index.get_loc(start) if start in df.index else None
                end_idx = df.index.get_loc(end) if end in df.index else None

                # Mark the row BEFORE the gap starts (if it exists)
                if start_idx is not None and start_idx > 0:
                    rows_to_remove.iloc[start_idx - 1] = True

                # Mark the FIRST row of the gap
                if start_idx is not None:
                    rows_to_remove.iloc[start_idx] = True

                # Mark the LAST row of the gap
                if end_idx is not None:
                    rows_to_remove.iloc[end_idx] = True

                # Mark the row AFTER the gap ends (if it exists)
                if end_idx is not None and end_idx < len(df) - 1:
                    rows_to_remove.iloc[end_idx + 1] = True

        # Store gap information for this column if any gaps were found
        if gaps_found:
            column_gap_info[col] = gaps_found
            print(f"  {col}: Found {len(gaps_found)} gaps > {max_gap_seconds/3600:.1f} hours")

            # Display first 3 gaps for debugging
            for gap in gaps_found[:3]:
                print(f"    - Gap from {gap['start']} to {gap['end']} ({gap['duration_hours']:.1f} hours)")
            if len(gaps_found) > 3:
                print(f"    ... and {len(gaps_found) - 3} more gaps")

    # =================== CHECK FOR LARGE TIME JUMPS IN INDEX ===================
    # Also detect large jumps in the datetime index itself (not just missing values)
    large_time_jumps = time_diffs > max_gap_seconds
    if large_time_jumps.any():
        print(f"\n  Time index: Found {large_time_jumps.sum()} jumps > {max_gap_seconds/3600:.1f} hours")

        # Mark rows around large time jumps for removal
        rows_to_remove |= large_time_jumps                    # Mark row after the jump
        rows_to_remove |= large_time_jumps.shift(-1).fillna(False)  # Mark row before the jump

    # =================== SUMMARY REPORTING ===================
    total_gaps = len(column_gap_info)
    total_rows_to_remove = rows_to_remove.sum()

    print(f"\nSummary:")
    print(f"  - Columns with large gaps: {total_gaps}")
    print(f"  - Total boundary rows to remove: {total_rows_to_remove}")
    print("=" * 60)

    return rows_to_remove, column_gap_info

def safe_plot_data(x_data, y_data, plot_type='line', **kwargs):
    """
    Safely plots data by handling NaN values and ensuring proper data types.

    This function prevents plotting errors that can occur with missing data,
    mixed data types, or invalid values in time series data.

    Args:
        x_data: X-axis data (typically datetime index)
        y_data: Y-axis data (sensor values)
        plot_type (str): Type of plot ('line' or 'scatter')
        **kwargs: Additional arguments passed to matplotlib.pyplot.plot()

    Returns:
        matplotlib plot object or None if plotting fails

    Safety Features:
        - Converts y_data to numeric, handling non-numeric values
        - Filters out NaN values from both x and y data
        - Handles empty datasets gracefully
        - Provides error reporting without crashing
    """
    try:
        # Convert y_data to numeric, setting invalid values to NaN
        y_clean = pd.to_numeric(y_data, errors='coerce')

        # Create mask for valid data points (neither x nor y is NaN)
        valid_mask = ~(pd.isna(y_clean) | pd.isna(x_data))

        # Only plot if we have valid data points
        if valid_mask.any():
            # Extract valid data points only
            x_valid = x_data[valid_mask]
            y_valid = y_clean[valid_mask]

            # Create appropriate plot type
            if plot_type == 'scatter':
                return plt.plot(x_valid, y_valid, 'o', **kwargs)
            else:
                return plt.plot(x_valid, y_valid, **kwargs)
        else:
            print(f"Warning: No valid data points to plot")
            return None

    except Exception as e:
        print(f"Error in safe_plot_data: {e}")
        return None

def interpolate_and_plot():
    """
    Main orchestration function for the complete data cleaning and interpolation process.

    This function implements the enhanced cross-column gap handling as specified:
    1. Configuration and file path setup
    2. Data loading and preprocessing
    3. ENHANCED: Cross-column gap identification and removal
    4. Column-specific data cleaning (temperature units, modulation values)
    5. Intelligent interpolation method selection
    6. Comprehensive visualization creation
    7. Detailed reporting and file output

    ENHANCED: Now properly handles gaps across ALL columns as requested.
    The key requirement: if ANY column has gaps > max_gap_hours,
    those boundary rows are removed from ALL columns before interpolation.
    """
    print("Starting Enhanced Data Cleaning and Interpolation Process...")
    print("=" * 60)

    # =================== CONFIGURATION SETUP ===================
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script directory: {script_dir}")

    # Look for config.ini - 2 levels up from scripts/1_data_preprocessing/
    parent_dir = os.path.dirname(script_dir)  # scripts/
    project_root = os.path.dirname(parent_dir)  # project root
    config_path = os.path.join(project_root, "config.ini")
    
    if not os.path.exists(config_path):
        # Fallback: try parent directory only (scripts/)
        config_path = os.path.join(parent_dir, "config.ini")
        print(f"Config not found in project root, checking scripts/: {config_path}")

    print(f"Reading configuration from: {os.path.abspath(config_path)}")
    config = read_config(config_path)
    print("✓ Configuration loaded successfully!")

    # =================== FILE PATH SETUP ===================
    # Extract key paths from configuration
    base_folder = config.get("paths", "base_folder")              # Main data directory
    processed_root = config.get("paths", "processed_folder")      # Processed data root
    input_csv = config.get("paths", "input_csv")                  # Target CSV filename
    merged_folder = config.get("paths", "merged_folder", fallback="merged")  # Merged data subfolder

    print(f"Base folder: {base_folder}")
    print(f"Requested CSV: {input_csv}")

    # Extract base filename without extension for consistent naming
    base_name = os.path.splitext(input_csv)[0]

    # Resolve folders relative to config.ini directory (not current working dir)
    cfg_dir = getattr(config, '_config_dir', None)
    if cfg_dir:
        if base_folder and not os.path.isabs(base_folder):
            base_folder = os.path.abspath(os.path.join(cfg_dir, base_folder))
        if processed_root and not os.path.isabs(processed_root):
            processed_root = os.path.abspath(os.path.join(cfg_dir, processed_root))

    # =================== LOOK FOR REINDEXED FILE (STEP 3) FIRST ===================
    # CRITICAL FIX: For interpolation to work correctly, we need the REINDEXED file
    # which has a regular time grid with NaNs for missing timestamps.
    # The merged file has irregular timestamps, so gaps aren't visible as NaNs.
    
    reindex_file_path = os.path.join(processed_root, "reindex_for_interp", base_name, f"{base_name}_reindexed_for_interpolation.csv")
    merged_file_path = os.path.join(processed_root, merged_folder, base_name, f"{base_name}_merged.csv")
    alt_merged_file_path = os.path.join(processed_root, "merged_data", base_name, f"{base_name}_merged.csv")

    # =================== INPUT FILE SELECTION LOGIC ===================
    # Priority: reindexed file > merged file > raw file
    if os.path.exists(reindex_file_path):
        print(f"✅ Using REINDEXED file (step 3 output): {reindex_file_path}")
        print(f"   This file has a regular time grid with NaNs for missing timestamps.")
        file_path = reindex_file_path
        skiprows_effective = 0
        needs_reindex = False
    elif os.path.exists(merged_file_path):
        print(f"⚠️  Using MERGED file (will reindex on the fly): {merged_file_path}")
        print(f"   Recommendation: Run reindexing script (step 3) first for best results.")
        file_path = merged_file_path
        skiprows_effective = 0
        needs_reindex = True
    elif os.path.exists(alt_merged_file_path):
        print(f"⚠️  Using MERGED file (alternate path, will reindex): {alt_merged_file_path}")
        print(f"   Recommendation: Run reindexing script (step 3) first for best results.")
        file_path = alt_merged_file_path
        skiprows_effective = 0
        needs_reindex = True
    else:
        print(f"⚠️  No merged or reindexed file found – falling back to RAW file.")
        file_path = os.path.join(base_folder, input_csv)
        skiprows_effective = config.getint("data", "skiprows", fallback=0)
        needs_reindex = True

    print(f"Full file path chosen: {file_path}")

    # =================== CREATE OUTPUT DIRECTORY STRUCTURE ===================
    # Create organized directory structure: processed_data/interpolation/base_name/
    processed_data_dir = os.path.join(processed_root, "interpolation", base_name)
    os.makedirs(processed_data_dir, exist_ok=True)
    print(f"✓ Created processed data directory: {processed_data_dir}")

    # Verify input file exists before proceeding
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    print(f"✓ File found! Processing: {file_path}")

    # =================== DATA LOADING ===================
    print("\nLoading and validating data...")

    print(f"Reading CSV with skiprows = {skiprows_effective}")
    df = pd.read_csv(file_path, skiprows=skiprows_effective)
    print(f"✓ Data loaded: {len(df)} rows, {len(df.columns)} columns")
    print(f"Available columns: {list(df.columns)}")

    # =================== TIME COLUMN PROCESSING ===================
    # Get time column name from configuration
    time_column = config.get("data", "time_column", fallback="Time")

    # Verify time column exists
    if time_column not in df.columns:
        print(f"Error: Time column '{time_column}' not found in data!")
        sys.exit(1)

    print(f"Converting '{time_column}' column to datetime...")
    # Convert time column to datetime and set as index
    df[time_column] = pd.to_datetime(df[time_column])
    df.set_index(time_column, inplace=True)

    # Drop rows with NaT timestamps (from DST ambiguous times in merge script)
    initial_len = len(df)
    df = df[df.index.notna()]
    dropped_nat = initial_len - len(df)
    if dropped_nat > 0:
        print(f"   ✓ Dropped {dropped_nat} rows with NaT timestamps (DST ambiguous times from merge)")

    # Ensure timezone-aware index using Europe/Rome (timestamps are local Rome time,
    # NOT UTC — localizing as UTC shifts the 30-min grid by +1/+2 hours, creating ghost NaN rows)
    if df.index.tz is None:
        if ROME_TZ is None:
            print("[WARNING] No timezone library available — skipping tz_localize. Install tzdata or pytz.")
        else:
            try:
                df.index = df.index.tz_localize(ROME_TZ, ambiguous='infer', nonexistent='shift_forward')
            except Exception:
                df.index = df.index.tz_localize(ROME_TZ, ambiguous='NaT', nonexistent='shift_forward')
                df = df[df.index.notna()]

    print("✓ Time column processed and set as index")

    # Calculate and display time span for data quality assessment
    time_span = df.index.max() - df.index.min()
    time_span_days = time_span.total_seconds() / (24 * 3600)
    print(f"Data time span: {time_span_days:.2f} days ({df.index.min()} to {df.index.max()})")

    # =================== CRITICAL FIX: ALWAYS REINDEX TO FULL 30-MIN GRID ===================
    # 🔑 This is ESSENTIAL for time series analysis, especially LSTM
    # Missing timestamps become NaN rows (not missing rows)
    print("\n" + "="*60)
    print("REINDEXING TO FULL 30-MIN GRID (CRITICAL FOR TIME SERIES)")
    print("="*60)
    
    rows_before_reindex = len(df)
    
    # Create complete 30-minute time grid from start to end
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq='30min',
        tz=df.index.tz  # Preserve timezone
    )
    
    expected_rows = len(full_index)
    missing_timestamps = expected_rows - rows_before_reindex
    
    print(f"\nBefore reindexing:")
    print(f"  • Actual rows: {rows_before_reindex}")
    print(f"  • Expected rows (30-min grid): {expected_rows}")
    print(f"  • Missing timestamps: {missing_timestamps} ({missing_timestamps/expected_rows*100:.1f}%)")
    
    if missing_timestamps > 0:
        print(f"\n⚠️  Dataset has MISSING TIMESTAMPS (gaps in time grid)!")
        print(f"   These will become NaN rows after reindexing.")
        print(f"   Interpolation will then fill short gaps (configurable threshold).")
    else:
        print(f"\n✅ Dataset already has complete 30-min grid (no missing timestamps).")
    
    # Reindex to full grid - missing timestamps become rows with NaN values
    df = df.reindex(full_index)
    df.index.name = 'Time'
    
    print(f"\n✓ Reindexing complete: {len(df)} rows (full 30-min grid)")
    print("="*60)

    # =================== COLUMN CONFIGURATION ===================
    # Get columns to process from configuration
    columns_to_process_str = config.get("interpolation", "columns_to_process", fallback="")
    columns_to_process = [col.strip() for col in columns_to_process_str.split(",") if col.strip()]

    # Fallback logic if no columns specified in config
    if not columns_to_process:
        try:
            # Try to get temperature and modulation columns from data section
            temp_cols_str = config.get("data", "temperature_columns", fallback="")
            mod_cols_str = config.get("data", "modulation_columns", fallback="")
            temp_cols = [col.strip() for col in temp_cols_str.split(",")]
            mod_cols = [col.strip() for col in mod_cols_str.split(",")]
            columns_to_process = temp_cols + mod_cols
        except configparser.NoOptionError:
            # Last resort: use all columns
            columns_to_process = df.columns.tolist()
            print("Warning: Using all columns due to missing config")

    # Filter to only existing columns (handles config errors gracefully)
    columns_to_process = [col for col in columns_to_process if col in df.columns]
    print(f"Columns to process: {columns_to_process}")

    # =================== COLUMN CLASSIFICATION ===================
    # Separate columns by type for appropriate cleaning methods

    # Temperature columns: contain temperature-related keywords
    temp_cols = [col for col in columns_to_process
                if any(keyword in col.lower() for keyword in ['temp', 't ripresa', 't mandata', 'sp', 'esterna'])]

    # Modulation columns: contain modulation-related keywords
    mod_cols = [col for col in columns_to_process if 'modul' in col.lower()]

    print(f"Temperature columns: {temp_cols}")
    print(f"Modulation columns: {mod_cols}")

    # =================== DATA CLEANING ===================
    print("\nCleaning data...")
    # Keep original data for comparison and visualization
    df_original = df.copy()

    print("Cleaning temperature columns...")
    # Apply temperature-specific cleaning (remove units, handle decimal commas)
    for col in temp_cols:
        if col in df.columns:
            df[col] = clean_temp(df[col])

    print("Cleaning modulation columns...")
    # Apply modulation-specific cleaning (ensure numeric format)
    for col in mod_cols:
        if col in df.columns:
            df[col] = clean_modulation(df[col])

    # =================== LOAD HARD OUTLIER FLAGS FROM STEP 4 ===================
    # Hard outliers (physical limits, spikes, stuck sensors) must be NaN'd BEFORE
    # interpolation. If left as-is, an impossible value (e.g. 999°C) would anchor
    # the interpolation and corrupt the surrounding data.
    # Soft flags (IQR anomalies) are intentionally kept — the LSTM should detect them.
    outlier_flags_path = os.path.join(
        processed_root, "outlier_detection", base_name, f"{base_name}_outlier_flags.csv"
    )
    hard_outlier_mask = {}   # col -> boolean Series (True = hard outlier, set to NaN)

    if os.path.exists(outlier_flags_path):
        print(f"\n{'='*60}")
        print("LOADING HARD OUTLIER FLAGS (STEP 4)")
        print(f"{'='*60}")
        try:
            flags_df = pd.read_csv(outlier_flags_path, index_col=0, parse_dates=True)
            # Align timezone with df
            if flags_df.index.tz is None and df_cleaned.index.tz is not None:
                flags_df.index = flags_df.index.tz_localize(df_cleaned.index.tz)
            elif flags_df.index.tz is not None and df_cleaned.index.tz is None:
                flags_df.index = flags_df.index.tz_localize(None)

            total_hard_nulled = 0
            for col in columns_to_process:
                hard_col = f"{col}_is_hard"
                if hard_col not in flags_df.columns:
                    continue
                # Align index (some rows may not be in flags due to step-3 reindexing)
                flags_aligned = flags_df[hard_col].reindex(df_cleaned.index, fill_value=0)
                mask = flags_aligned.astype(bool)
                if mask.any():
                    hard_outlier_mask[col] = mask
                    df_cleaned.loc[mask, col] = np.nan
                    total_hard_nulled += int(mask.sum())
                    print(f"   ✓ {col}: {int(mask.sum())} hard outlier(s) set to NaN (will be interpolated over)")

            if total_hard_nulled == 0:
                print("   ✓ No hard outliers found in flags — data already clean")
            else:
                print(f"\n   Total hard outlier values NaN'd: {total_hard_nulled}")
                print(f"   Soft flags (IQR) left untouched — the LSTM will detect them")
            print(f"{'='*60}")
        except Exception as e:
            print(f"   ⚠️  Could not load outlier flags: {e} — proceeding without hard-outlier masking")
    else:
        print(f"\nℹ️  No outlier flags file found at expected path:")
        print(f"   {outlier_flags_path}")
        print(f"   Run step 4 (outlier_detection.py) first for best results.")

    # =================== GAP IDENTIFICATION (REPORT ONLY — NO ROW DELETION) ===================
    # Solution 1: keep the full 30-min grid and leave long gaps as NaN.
    # The interpolation step below uses limit=limit_steps, so it naturally stops
    # filling beyond max_gap_hours. Long-gap rows are preserved in the output CSV
    # with NaN values. LSTM sequence creation then discards any window that
    # contains a NaN, which is the correct place to handle missing data.

    max_gap_hours = config.getint("interpolation", "max_gap_hours", fallback=4)
    gap_seconds = max_gap_hours * 3600  # Convert to seconds

    print(f"\n" + "="*60)
    print(f"GAP IDENTIFICATION (>={max_gap_hours}h) — config max_gap_hours={max_gap_hours}")
    print("="*60)
    print(f"   • Short gaps (<{max_gap_hours}h): Will be interpolated")
    print(f"   • Long gaps (≥{max_gap_hours}h): Kept as NaN in output (NOT deleted)")
    print(f"   ℹ️  Full 30-min grid is preserved — downstream tools drop NaN windows")

    # Identify which rows sit inside long-gap NaN runs (for reporting only)
    rows_to_remove, gap_info = identify_gaps_across_columns(df, columns_to_process, gap_seconds)

    sampling_interval_min = config.getint("data", "sampling_interval_minutes", fallback=30)
    limit_steps = int((max_gap_hours * 60) / sampling_interval_min)  # e.g. 4h / 30min = 8 steps

    # Count long-gap rows for reporting (but do NOT delete them)
    long_gap_mask = pd.Series(False, index=df.index)
    for col in columns_to_process:
        if col not in df.columns:
            continue
        is_nan = df[col].isna()
        run_id = (~is_nan).cumsum()
        nan_run_lengths = is_nan.groupby(run_id).transform('sum')
        long_gap_mask |= (is_nan & (nan_run_lengths >= limit_steps))

    combined_long_gap = rows_to_remove | long_gap_mask
    total_long_gaps = sum(len(gaps) for gaps in gap_info.values())
    n_long_gap_rows = int(combined_long_gap.sum())

    if total_long_gaps > 0 or n_long_gap_rows > 0:
        print(f"\n⚠️  Found {total_long_gaps} long gap(s) (≥{max_gap_hours}h).")
        for col, gaps in gap_info.items():
            print(f"   • {col}: {len(gaps)} gap(s)")
            for gap in gaps[:3]:
                print(f"     - {gap['start']} to {gap['end']} ({gap['duration_hours']:.1f}h)")
        print(f"\n   {n_long_gap_rows} rows will remain as NaN in the output (grid preserved)")

        # =================== SAVE GAP INFORMATION REPORT ===================
        gap_report_file = os.path.join(processed_data_dir, f"{base_name}_long_gaps_report.csv")
        gap_report_data = []
        for col, gaps in gap_info.items():
            for gap in gaps:
                gap_report_data.append({
                    'column': col,
                    'gap_start': gap['start'],
                    'gap_end': gap['end'],
                    'duration_hours': gap['duration_hours']
                })
        if gap_report_data:
            pd.DataFrame(gap_report_data).to_csv(gap_report_file, index=False)
            print(f"✓ Long gaps report saved: {gap_report_file}")
    else:
        print(f"\n✅ No gaps longer than {max_gap_hours} hours found.")

    # Keep ALL rows — full 30-min grid is preserved
    df_cleaned = df.copy()

    print(f"\n✓ Dataset (full grid, no rows deleted): {len(df_cleaned)} rows")
    print("="*60)

    # =================== PRE-INTERPOLATION MISSING VALUE REPORT ===================
    print("\n" + "=" * 60)
    print("Pre-Interpolation Missing Value Report (After Gap Removal)")
    print("=" * 60)

    total_rows = len(df_cleaned)

    # Check each column for remaining missing values after gap removal
    for col in columns_to_process:
        missing_count = df_cleaned[col].isna().sum()
        if missing_count > 0:
            missing_percentage = (missing_count / total_rows) * 100
            print(f"  - Column '{get_display_name(col)}': {missing_count} missing values ({missing_percentage:.2f}%)")

    # Report if no missing values remain
    if df_cleaned[columns_to_process].isna().sum().sum() == 0:
        print("✓ No missing values detected in target columns before interpolation.")
    print("=" * 60)

    # =================== DATA INTERPOLATION ===================
    # Apply intelligent interpolation based on column type
    print("\nApplying interpolation...")
    df_interp = df_cleaned.copy()

    # =================== INTERPOLATION METHOD SELECTION ===================
    # CRITICAL FIX: Properly detect setpoints and control signals for forward-fill
    # This matches paper-consistent practices for AHU monitoring data
    
    print("\n" + "="*60)
    print("SELECTING INTERPOLATION METHODS (PAPER-CONSISTENT)")
    print("="*60)
    
    # Initialize lists
    ffill_cols = []
    linear_cols = []
    
    # Classify each column based on its type
    for col in columns_to_process:
        if col not in df_interp.columns:
            continue
            
        col_lower = col.lower()
        
        # SETPOINTS: Forward-fill (they should stay constant until changed)
        # Detect: "set point", "setpoint", "sp ", "set pt", "compensat" (compensated setpoint)
        is_setpoint = any(keyword in col_lower for keyword in [
            'set point', 'setpoint', 'set pt', 'compensat'
        ]) or (col_lower.startswith('sp ') or ' sp ' in col_lower)
        
        # CONTROL SIGNALS: Forward-fill (modulation, fan speed, valve position)
        # Detect: "modul", "fan", "ventil", "valve", "valv", "damper"
        is_control = any(keyword in col_lower for keyword in [
            'modul', 'fan', 'ventil', 'valve', 'valv', 'damper'
        ])
        
        # Classify
        if is_setpoint or is_control:
            ffill_cols.append(col)
        else:
            # Default: linear interpolation for sensor readings
            linear_cols.append(col)
    
    # Display classification results
    print("\n📊 INTERPOLATION METHOD ASSIGNMENT:")
    print("\n✅ Forward-Fill (setpoints & control signals):")
    if ffill_cols:
        for col in ffill_cols:
            print(f"   • {col}")
    else:
        print("   (none)")
    
    print("\n📈 Linear/Time Interpolation (sensor readings):")
    if linear_cols:
        for col in linear_cols:
            print(f"   • {col}")
    else:
        print("   (none)")
    
    print("\n" + "="*60)
    
    # =================== STORE PRE-INTERPOLATION STATE ===================
    # Store which values are NaN BEFORE interpolation (for mask creation).
    # Also record long-gap positions so we can RESTORE them to NaN after
    # interpolation — guaranteeing the strict rule: any gap ≥ max_gap_hours
    # remains entirely NaN, regardless of fill direction.
    print("\nStoring pre-interpolation state for mask creation...")
    pre_interpolation_na_state = {}
    long_gap_per_col = {}          # per-column long-gap boolean Series
    for col in columns_to_process:
        if col in df_cleaned.columns:
            is_nan = df_cleaned[col].isna()
            pre_interpolation_na_state[col] = is_nan.copy()
            na_count = is_nan.sum()
            if na_count > 0:
                print(f"  {col}: {na_count} NaN values to interpolate")
            # Identify runs that are >= limit_steps long (= long gaps)
            # limit_steps may not be computed yet at this point, so we use
            # the config values directly (same formula as below).
            _si = config.getint("data", "sampling_interval_minutes", fallback=30)
            _mgh = config.getint("interpolation", "max_gap_hours", fallback=4)
            _lim = int((_mgh * 60) / _si)
            run_id = (~is_nan).cumsum()
            run_len = is_nan.groupby(run_id).transform('sum')
            long_gap_per_col[col] = is_nan & (run_len >= _lim)


    # =================== APPLY INTERPOLATION METHODS ===================
    print("\n" + "="*60)
    print("PAPER-CONSISTENT INTERPOLATION (Follows Paper 9 Approach)")
    print("="*60)
    print(f"\n🔑 KEY PRINCIPLE: Fill ONLY short gaps, keep long gaps as NaN")
    print(f"   • Short gaps (<{max_gap_hours}h): Interpolated (reliable recovery)")
    print(f"   • Long gaps (≥{max_gap_hours}h): Left as NaN (unreliable periods)")
    print(f"   • Maintains data traceability with masks")
    
    # Calculate limit_steps based on actual sampling interval
    time_diffs = df_interp.index.to_series().diff()
    median_interval = time_diffs.median()
    
    if pd.notna(median_interval):
        median_hours = median_interval.total_seconds() / 3600
        limit_steps = int(max_gap_hours / median_hours) if median_hours > 0 else int(max_gap_hours * 4)
        print(f"\nSampling interval: {median_interval}")
        print(f"Max gap to fill: {max_gap_hours}h = {limit_steps} consecutive steps")
        print(f"   → Gaps ≥{limit_steps+1} steps will remain as NaN ✅")
    else:
        limit_steps = int(max_gap_hours * 4)
        print(f"\n⚠️  Using fallback limit: {limit_steps} steps")
    
    # LINEAR/TIME INTERPOLATION for sensor readings
    if linear_cols:
        print(f"\n📈 LINEAR/TIME Interpolation ({len(linear_cols)} continuous measurements):")
        print(f"   ✓ Method: time-based (accounts for irregular intervals)")
        print(f"   ✓ Limit: {limit_steps} consecutive NaNs (≈{max_gap_hours}h)")
        print(f"   ✓ Direction: both (forward and backward)")
        print(f"   ✓ Applies to: Temperatures, Humidity, Rolling averages")
        
        for col in linear_cols:
            before_na = df_interp[col].isna().sum()
            df_interp[col] = df_interp[col].interpolate(
                method='time',
                limit=limit_steps,
                limit_direction='both'
            )
            after_na = df_interp[col].isna().sum()
            filled = before_na - after_na
            if filled > 0:
                print(f"   ✓ {col}: filled {filled} points")
    
    # FORWARD-FILL INTERPOLATION for setpoints and control signals
    if ffill_cols:
        print(f"\n⏩ FORWARD-FILL ({len(ffill_cols)} setpoints & control signals):")
        print(f"   ✓ Method: ffill (maintains constant values)")
        print(f"   ✓ Limit: {limit_steps} consecutive NaNs (≈{max_gap_hours}h)")
        print(f"   ✓ Rationale: Controllers hold setpoints; modulation is step-like")
        print(f"   ✓ Applies to: Setpoints, Fan modulation, Valve positions")
        
        for col in ffill_cols:
            before_na = df_interp[col].isna().sum()
            # Apply forward-fill with limit, then back-fill for leading NaNs
            df_interp[col] = df_interp[col].ffill(limit=limit_steps)
            df_interp[col] = df_interp[col].bfill(limit=limit_steps)
            after_na = df_interp[col].isna().sum()
            filled = before_na - after_na
            if filled > 0:
                print(f"   ✓ {col}: filled {filled} points")
    
    # =================== RE-APPLY LONG-GAP MASK (STRICT OPTION A) ===================
    # Interpolation with limit_direction='both' + bfill can fill from both sides,
    # effectively bridging gaps that are exactly ~2× limit_steps long.
    # We restore those positions to NaN so the guarantee holds:
    #   "any gap ≥ max_gap_hours is entirely NaN in the output".
    print(f"\n🔒 STRICT GAP GUARD: restoring long gaps (≥{max_gap_hours}h) to NaN...")
    total_restored = 0
    for col in columns_to_process:
        if col not in df_interp.columns:
            continue
        mask = long_gap_per_col.get(col)
        if mask is None or not mask.any():
            continue
        # Re-index mask to df_interp (grid may differ from df_cleaned in edge cases)
        mask_aligned = mask.reindex(df_interp.index, fill_value=False)
        n_restore = int(mask_aligned.sum())
        if n_restore > 0:
            df_interp.loc[mask_aligned, col] = np.nan
            total_restored += n_restore
            print(f"   ↩ {col}: {n_restore} long-gap rows restored to NaN")
    if total_restored == 0:
        print("   ✓ No long-gap rows needed restoring (all gaps were short)")
    else:
        print(f"   ✓ Total restored: {total_restored} values — long gaps are strictly NaN")

    # Drop rows that are still NaN after interpolation (long-gap rows ≥ max_gap_hours)
    long_gap_rows = df_interp[columns_to_process].isna().any(axis=1)
    n_dropped_long = int(long_gap_rows.sum())
    if n_dropped_long > 0:
        df_interp = df_interp[~long_gap_rows]
        print(f"\n🗑️  Dropped {n_dropped_long} long-gap rows (≥{max_gap_hours}h) from dataset")
        print(f"   Dataset reduced: {len(df_interp)} rows remaining")
    else:
        print(f"\n✅ No long-gap rows to drop (all gaps were short)")

    print("\n" + "="*60)
    print("\n✅ INTERPOLATION COMPLETE:")
    print(f"   • Short gaps (<{max_gap_hours}h): Filled with appropriate method")
    print(f"   • Long gaps (≥{max_gap_hours}h): {n_dropped_long} rows deleted from dataset")
    print(f"   • Full traceability: Masks show which points were interpolated")
    print(f"   • Dataset ready for: Visualization, ML, and quality-controlled analysis")

    # =================== CREATE INTERPOLATION MASKS (AFTER INTERPOLATION) ===================
    # 🔴 CRITICAL FIX: Create masks AFTER interpolation completes
    # Compare pre-interpolation NaN state with post-interpolation state
    print("\nCreating interpolation masks...")
    for col in columns_to_process:
        if col not in df_interp.columns:
            continue
            
        # Define the name for the new mask column
        mask_col_name = f"{col.replace(' ', '_').replace('.', '')}_is_interpolated"
        
        # ✅ CORRECTED: Mask = (was NaN before) AND (has value after interpolation)
        # This correctly identifies points that were interpolated
        was_na_before = pre_interpolation_na_state.get(col, pd.Series(False, index=df_interp.index))
        has_value_now = df_interp[col].notna()
        interpolated_mask = was_na_before & has_value_now
        
        # Add the mask as a new column to the final DataFrame
        df_interp[mask_col_name] = interpolated_mask
        
        # Report how many points were actually interpolated
        interpolated_count = interpolated_mask.sum()
        if interpolated_count > 0:
            print(f"  ✓ {col}: {interpolated_count} points interpolated (mask created)")
        else:
            print(f"  ℹ️  {col}: No points interpolated (mask all False)")

    # =================== POST-INTERPOLATION REPORT ===================
    print("\n" + "="*60)
    print("POST-INTERPOLATION GAP ANALYSIS")
    print("="*60)
    remaining_missing = df_interp[columns_to_process].isna().sum()
    any_remaining = remaining_missing.sum() > 0

    if any_remaining:
        print("\n⚠️  Unexpected NaN values remaining after interpolation and row deletion:")
        for col in columns_to_process:
            missing_count = remaining_missing[col]
            if missing_count > 0:
                total_points = len(df_interp)
                pct = (missing_count / total_points * 100)
                print(f"  • {get_display_name(col)}: {missing_count} NaN ({pct:.1f}%)")
    else:
        print(f"  ✓ No NaN values remain — long-gap rows were deleted, short gaps filled")
        print(f"  ✓ Dataset is fully clean: {len(df_interp)} rows")
    print("="*60)

    # =================== SAVE INTERPOLATED DATA ===================
    print("\nSaving interpolated data with masks...")
    
    # Round all numeric columns to 2 decimal places
    df_interp_rounded = df_interp.copy()
    for col in df_interp_rounded.columns:
        if df_interp_rounded[col].dtype in ['float64', 'float32']:
            df_interp_rounded[col] = df_interp_rounded[col].round(2)
    
    output_file = os.path.join(processed_data_dir, f"{base_name}_interpolated_with_masks.csv")
    safe_overwrite(output_file, df_interp_rounded.reset_index())
    print(f"✓ Saved interpolated data to: {output_file}")
    
    # Also save a version without the mask columns (simpler format)
    print("Saving simplified interpolated data (without mask columns)...")
    output_file_simple = os.path.join(processed_data_dir, f"{base_name}_interpolated.csv")
    df_interp_simple = df_interp_rounded[columns_to_process].copy()  # Only include original columns (already rounded)
    safe_overwrite(output_file_simple, df_interp_simple.reset_index())
    print(f"✓ Saved simplified interpolated data to: {output_file_simple}")

    # =================== SAVE INTERPOLATION PROCESS SUMMARY ===================
    # Create comprehensive summary of the interpolation process for documentation
    
    # Calculate how many points were actually filled for each column
    points_filled = {}
    for col in columns_to_process:
        if col in df_cleaned.columns and col in df_interp.columns:
            mask_col = f"{col.replace(' ', '_').replace('.', '')}_is_interpolated"
            if mask_col in df_interp.columns:
                points_filled[col] = int(df_interp[mask_col].sum())
    
    summary_data = {
        'process_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'input_file': file_path,
        'output_file': output_file,
        'original_rows': len(df_original),
        'rows_after_gap_removal': len(df_cleaned),
        'rows_removed': len(df_original) - len(df_cleaned),
        'max_gap_hours': max_gap_hours,
        'interpolation_limit_steps': limit_steps,
        'sampling_interval_detected': str(median_interval) if pd.notna(median_interval) else 'unknown',
        'columns_with_gaps': list(gap_info.keys()),
        'total_gaps_found': sum(len(gaps) for gaps in gap_info.values()),
        'interpolation_methods': {
            'linear_time': linear_cols,
            'forward_fill': ffill_cols
        },
        'points_filled_per_column': points_filled
    }

    # Save summary as JSON file for documentation and troubleshooting
    summary_file = os.path.join(processed_data_dir, f"{base_name}_interpolation_summary.json")
    import json
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    print(f"✓ Saved interpolation summary to: {summary_file}")

    # =================== CREATE VISUALIZATIONS ===================
    # Generate comprehensive plots if enabled in configuration
    create_plots = config.getboolean("visualization", "create_plots", fallback=True)
    if create_plots:
        print("\nCreating visualizations...")

        # Setup plot directory structure
        plots_dir = config.get("visualization", "plots_dir", fallback=os.path.join(base_folder, "plots"))
        # Resolve plots_dir relative to config.ini directory when not absolute
        cfg_dir = getattr(config, '_config_dir', None)
        if cfg_dir and plots_dir and not os.path.isabs(plots_dir):
            plots_dir = os.path.abspath(os.path.join(cfg_dir, plots_dir))
        plot_folder = os.path.join(plots_dir, "interpolation", base_name)
        os.makedirs(plot_folder, exist_ok=True)
        print(f"✓ Created plot directory: {plot_folder}")

        # =================== INDIVIDUAL VARIABLE PLOTS ===================
        print("Creating individual variable plots...")
        plot_info_list = []  # Store plot information for Word report
        
        for column in columns_to_process:
            display_name = get_display_name(column)
            print(f"  Creating plot for: {display_name}")

            try:
                # Create figure with appropriate size for detailed visualization
                plt.figure(figsize=(14, 8))

                # Plot original data as scatter points to show actual measurements
                safe_plot_data(df_original.index, df_original[column],
                             plot_type='scatter', markersize=2, alpha=0.6,
                             color='blue', label='Original Data')

                # Plot interpolated data as continuous line to show smooth interpolation
                safe_plot_data(df_interp.index, df_interp[column],
                             plot_type='line', linewidth=1.5, color='red',
                             alpha=0.8, label='Interpolated Data')

                # Highlight specifically interpolated points (where original was NaN but interpolated has value)
                mask_col_name = f"{column.replace(' ', '_').replace('.', '')}_is_interpolated"
                if mask_col_name in df_interp.columns and df_interp[mask_col_name].any():
                    interpolated_mask = df_interp[mask_col_name]
                    safe_plot_data(df_interp.index[interpolated_mask],
                                 df_interp[column][interpolated_mask],
                                 plot_type='scatter', markersize=3, color='green',
                                 alpha=0.7, label='Interpolated Points')

                # Mark long-gap boundary regions on the plot for reference
                if rows_to_remove.any():
                    gap_regions = df_original.index[rows_to_remove.values]
                    if len(gap_regions) > 0:
                        # Plot gap boundaries at bottom of y-axis range
                        y_min, _ = plt.ylim()
                        plt.scatter(gap_regions, [y_min] * len(gap_regions),
                                  color='red', marker='x', s=50, alpha=0.8,
                                  label=f'Long Gap Boundaries (≥{max_gap_hours}h, kept as NaN)')

                # Configure plot appearance and labels
                date_start = df_interp.index.min().strftime('%Y-%m-%d')
                date_end = df_interp.index.max().strftime('%Y-%m-%d')
                plt.title(
                    f'{display_name} - Original vs Interpolated Data\n'
                    f'Short gaps (<{max_gap_hours}h) filled | Long gaps (≥{max_gap_hours}h) kept as NaN'
                    f'    [{date_start} → {date_end}]',
                    fontsize=13, fontweight='bold'
                )
                plt.xlabel('Time', fontsize=12)

                # Set appropriate y-axis label based on column type
                if any(keyword in column.lower() for keyword in ['temp', 't ripresa', 't mandata', 'sp']):
                    plt.ylabel('Temperature (°C)', fontsize=12)
                elif 'modul' in column.lower():
                    plt.ylabel('Modulation (%)', fontsize=12)
                elif 'rh' in column.lower() or 'umid' in column.lower():
                    plt.ylabel('Humidity (%)', fontsize=12)
                else:
                    plt.ylabel('Value', fontsize=12)

                # Add legend and grid for better readability
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                # Save plot with safe filename (remove problematic characters)
                safe_filename = column.replace('.', '').replace(' ', '_').replace('/', '_')
                plot_file = os.path.join(plot_folder, f"{safe_filename}_interpolation.png")
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()  # Close figure to free memory
                
                # Determine method type for description
                method_type = 'forward_fill' if column in ffill_cols else 'linear'
                
                # Generate description and add to plot info list
                description = generate_interpolation_description(column, df_original, df_interp, method_type)
                plot_info_list.append((display_name, plot_file, description))

            except Exception as e:
                print(f"    ❌ Error creating plot for {column}: {e}")
                plt.close()  # Ensure figure is closed even on error

        # =================== GAP ANALYSIS PLOT (FIXED) ===================
        # Create comprehensive gap visualization if gaps were found
        if gap_info:
            print("Creating FIXED gap analysis plot...")
            try:
                # --- FIX 1: Use Object-Oriented plotting for better control ---
                fig, ax = plt.subplots(figsize=(15, 8))
                
                # --- FIX 2: Prepare data for plotting ---
                y_labels = list(gap_info.keys())
                y_pos = np.arange(len(y_labels))
                
                all_gap_starts = []
                all_gap_ends = []

                colors = plt.cm.tab10(np.linspace(0, 1, len(y_labels)))

                for i, (col, gaps) in enumerate(gap_info.items()):
                    for gap in gaps:
                        # --- FIX 3: Convert duration to Timedelta for correct plotting ---
                        # This ensures the bar width is scaled correctly on the datetime axis.
                        gap_width = pd.to_timedelta(gap['duration_hours'], unit='h')
                        
                        # Use ax.barh for plotting on the axes object
                        ax.barh(y=y_pos[i], width=gap_width, left=gap['start'], height=0.6,
                                color=colors[i], alpha=0.7, edgecolor='black')
                        
                        all_gap_starts.append(gap['start'])
                        all_gap_ends.append(gap['end'])

                # --- FIX 4: Set Y-axis labels correctly ---
                # This properly aligns labels with the bars.
                ax.set_yticks(y_pos)
                ax.set_yticklabels(y_labels)
                ax.invert_yaxis()  # labels read top-to-bottom

                # --- FIX 5: Automatically zoom to the relevant time period ---
                # This makes the gaps visible by not showing the full empty timeline.
                if all_gap_starts and all_gap_ends:
                    min_time = min(all_gap_starts) - pd.Timedelta(days=1)
                    max_time = max(all_gap_ends) + pd.Timedelta(days=1)
                    ax.set_xlim(min_time, max_time)
                
                # Configure gap analysis plot appearance
                ax.set_xlabel('Time', fontsize=12)
                ax.set_ylabel('Column', fontsize=12)
                ax.set_title(f'Data Gaps > {max_gap_hours} hours by Column',
                             fontsize=14, fontweight='bold')
                
                # Improve date formatting on the x-axis
                import matplotlib.dates as mdates
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                fig.autofmt_xdate() # Rotates dates for better fit

                ax.grid(True, alpha=0.3, axis='x')
                fig.tight_layout()

                # Save gap analysis plot
                gap_plot_file = os.path.join(plot_folder, f"{base_name}_gap_analysis.png")
                plt.savefig(gap_plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Saved FIXED gap analysis plot: {gap_plot_file}")

            except Exception as e:
                print(f"  ❌ Error creating gap analysis plot: {e}")
                plt.close()

        # =================== GENERATE WORD REPORT ===================
        # Create Word report if plots were created and python-docx is available
        if plot_info_list and DOCX_AVAILABLE:
            print("\nGenerating Word report...")
            word_path = create_word_report(plot_info_list, plot_folder, base_name, df_original)
            if word_path:
                print(f"✅ Word report created: {word_path}")

        print(f"✓ All plots saved to: {plot_folder}")
    else:
        print("Skipping plot creation (disabled in configuration)")

    # =================== FINAL SUMMARY ===================
    print("\n" + "=" * 60)
    print("✅ DATA CLEANING AND INTERPOLATION COMPLETE!")
    print("=" * 60)
    print(f"Processed file: {file_path}")
    print(f"Output data saved to: {processed_data_dir}")
    print(f"\n🔑 KEY IMPROVEMENTS (Thesis-Ready Dataset):")
    print(f"   ✓ Full 30-min grid preserved ({len(df)} rows, no rows deleted)")
    print(f"   ✓ Long gaps (≥{max_gap_hours}h): {n_long_gap_rows} rows kept as NaN")
    print(f"   ✓ Short gaps (<{max_gap_hours}h) interpolated")
    print(f"   ✓ Final dataset: {len(df_interp)} rows")

    # =================== GAP HANDLING SUMMARY ===================
    print(f"\nGap handling summary:")
    print(f"  - Maximum gap threshold (from config): {max_gap_hours} hours")
    print(f"  - Columns checked: {len(columns_to_process)}")
    print(f"  - Columns with long gaps: {len(gap_info)}")
    print(f"  - Total long gaps found: {sum(len(gaps) for gaps in gap_info.values())}")
    print(f"  - Rows kept as NaN (long gaps, grid preserved): {n_long_gap_rows}")
    print(f"  - Final dataset: {len(df_interp)} rows")

    # =================== INTERPOLATION SUMMARY ===================
    print(f"\nInterpolation summary:")
    print(f"  - Linear/time interpolation: {len(linear_cols)} columns")
    print(f"    Columns: {[get_display_name(c) for c in linear_cols]}")
    print(f"  - Forward-fill interpolation: {len(ffill_cols)} columns")
    print(f"    Columns: {[get_display_name(c) for c in ffill_cols]}")

    # =================== FINAL OUTPUT LOCATIONS ===================
    if create_plots:
        print(f"\nVisualizations saved to: {plot_folder}")
        if 'word_path' in locals() and word_path:
            print(f"Word report saved to: {word_path}")

    print(f"\n📄 Key output files (Paper 9 Compliant):")
    print(f"  1. Interpolated data WITH masks: {output_file}")
    print(f"     → Use for traceability (shows which points were filled)")
    print(f"  2. Interpolated data CLEAN: {output_file_simple}")
    print(f"     → Use for ML/visualization (no mask columns)")
    print(f"  3. Process summary: {summary_file}")
    print(f"     → Metadata: methods used, points filled, etc.")
    if gap_info:
        print(f"  4. Long gaps report: {gap_report_file}")
        print(f"     → Documentation of gaps NOT filled (>={max_gap_hours}h)")
    
    print(f"\n💡 USAGE NOTES:")
    print(f"  • Both datasets have SAME row count (complete 30-min grid)")
    print(f"  • Long gaps remain as NaN (use masks to identify interpolated vs original)")
    print(f"  • For quality-critical analysis, exclude NaN periods or use masks")

    print("\n" + "=" * 60)
    print("✅ SUCCESS: Thesis-Ready Time Series Dataset Created!")
    print("=" * 60)
    print("📊 Dataset Properties:")
    print(f"   • Complete 30-min time grid (no missing timestamps)")
    print(f"   • Paper-consistent interpolation (time-based for sensors, ffill for controls)")
    print(f"   • Long gaps preserved as NaN (quality control - Paper 9 approach)")
    print(f"   • Full traceability maintained (interpolation masks included)")
    print(f"   • Ready for: LSTM sequences, carpet plots, heatmaps, season comparison")
    print("\n📁 Output Files:")
    print(f"   • With masks: {os.path.basename(output_file)}")
    print(f"   • Clean version: {os.path.basename(output_file_simple)}")
    if gap_info:
        print(f"   • Long gaps report: {os.path.basename(gap_report_file)}")
    print(f"   • Process summary: {os.path.basename(summary_file)}")
    print("=" * 60)

# =================== ENTRY POINT ===================
if __name__ == "__main__":
    """
    Script entry point with comprehensive error handling.

    This ensures that any errors during processing are caught and reported
    clearly, making debugging easier and preventing silent failures.
    """
    try:
        # Execute main interpolation process
        interpolate_and_plot()

    except Exception as e:
        # Comprehensive error reporting
        print(f"\n❌ Error occurred during interpolation:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")

        # Print full traceback for debugging
        import traceback
        traceback.print_exc()

        # Exit with error code
        sys.exit(1)
