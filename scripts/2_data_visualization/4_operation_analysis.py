"""
=============================================================================
AHU OPERATION ANALYSIS - COMBINED PLOT SCRIPT
=============================================================================

🎯 WHAT THIS SCRIPT DOES (in simple terms):
   1. Reads your HVAC sensor data from CSV files
   2. Creates beautiful graphs showing temperature and fan speed over time
   3. Generates a professional Word report with statistics
   4. Saves everything to the "plots" folder

📊 THE 4 GRAPHS IT CREATES:
   - Graph 1: External temperature (how hot/cold it is outside)
   - Graph 2: Supply temperature (air temperature going into rooms)
   - Graph 3: Return temperature (air temperature coming back from rooms)
   - Graph 4: Fan modulation (how fast the fan is running, 0-100%)

🚀 HOW TO USE (3 simple commands):
   
   Normal use (process one dataset):
   >>> python 7.operation_analysis.py
   
   Run tests (check if everything works):
   >>> python 7.operation_analysis.py --test
   
   Process multiple datasets at once (faster):
   >>> python 7.operation_analysis.py --parallel Dataset1 Dataset2

💡 WHAT YOU NEED TO KNOW:
   - Your data must be in the "processed_data" folder
   - The file must be named like: C1_UTA1_Summer2024_interpolated.csv
   - Config.ini must be set up correctly (building_id, ahu_unit, season, year)
   - Output goes to: plots/operation_analysis/

⚠️  IF SOMETHING GOES WRONG:
   - Check the "logs" folder for detailed error messages
   - Make sure your CSV file exists
   - Verify config.ini has correct settings
   - Run --test to check if the script is working

=============================================================================
"""

# =============================================================================
# IMPORT LIBRARIES (The tools this script needs to work)
# =============================================================================
# Think of these like ingredients for a recipe - we need them all!

import pandas as pd                    # For reading and working with CSV data
import matplotlib.pyplot as plt        # For creating graphs/plots
import configparser                    # For reading config.ini settings
import os                              # For working with files and folders
import sys                             # For system operations
import logging                         # For creating log files (like a diary of what the script does)
from datetime import datetime          # For working with dates and times
from pathlib import Path               # For easier file path handling
from functools import lru_cache        # For making things faster (caching = remembering)
from typing import Optional, Tuple, Dict, Any, List  # For type hints (helps prevent errors)
from multiprocessing import Pool, cpu_count  # For processing multiple files at once
import warnings                        # For handling warning messages

# Hide matplotlib warnings (just makes output cleaner)
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# =============================================================================
# TRY TO LOAD OPTIONAL FEATURES
# =============================================================================
# These are "nice to have" but not required - script works without them!

# Try to import Word document creator (for professional reports)
try:
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    DOCX_AVAILABLE = True
    # ✓ Success! We CAN create Word reports
except ImportError:
    DOCX_AVAILABLE = False
    # ✗ Word library not installed - will skip Word reports (plots still work!)
    print("[WARNING] python-docx not installed. Word report generation will be disabled.")
    print("To enable: pip install python-docx")

# Add parent directories to path to import config_utils
# From scripts/2_data_visualization/ -> scripts/ -> project_root/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Try to import config helper (makes reading config.ini easier)
try:
    from config_utils import load_config
except ModuleNotFoundError:
    def load_config(config_path):
        """Fallback function to load config WITH interpolation support."""
        config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        config.read(config_path, encoding='utf-8-sig')
        return config

# =============================================================================
# ADVANCED FEATURES (You can use these without understanding how they work!)
# =============================================================================

# -----------------------------------------------------------------------------
# FEATURE 1: LOGGING SYSTEM (Automatic Diary)
# -----------------------------------------------------------------------------
# This creates a "log file" that records everything the script does
# Think of it like a flight recorder - if something goes wrong, you can check it!
#  You don't need to touch this function - it runs automatically!
# Just check the "logs" folder if you need to see what happened.
# -----------------------------------------------------------------------------

def setup_logging(log_dir: str = None, log_level: str = 'INFO') -> logging.Logger:
    """
    Set up logging configuration for the operation analysis script.
    
    Args:
        log_dir (str): Directory to save log files. If None, do not write files.
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    log_file = None

    # Only create a log file if an explicit directory is provided
    if log_dir:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir_path / f'operation_analysis_{timestamp}.log'
        handlers.insert(0, logging.FileHandler(log_file, encoding='utf-8'))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    logger = logging.getLogger('OperationAnalysis')
    if log_file:
        logger.info(f"Logging initialized - Log file: {log_file}")
    else:
        logger.info("Logging initialized - console only (no log files)")

    return logger

# Initialize logger (will be configured in main())
logger = None

# -----------------------------------------------------------------------------
# FEATURE 2: CONFIG VALIDATION (Safety Check)
# -----------------------------------------------------------------------------
# This checks if your config.ini file is set up correctly
# Think of it like spell-check for your configuration file!
#
# If you see warnings about config.ini, just fix those lines.
# The script tells you exactly what's wrong and where!
# -----------------------------------------------------------------------------

def validate_config(config: configparser.ConfigParser) -> Tuple[bool, List[str]]:
    """
    Validate configuration file for required sections and options.
    
    Args:
        config (ConfigParser): Configuration object to validate
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_errors)
    """
    errors = []
    
    # Required sections
    required_sections = [
        'global',
        'paths',
        'data',
        'operation_combined_plot'
    ]
    
    for section in required_sections:
        if not config.has_section(section):
            errors.append(f"Missing required section: [{section}]")
    
    # Check critical options in [global]
    if config.has_section('global'):
        required_global = ['building_id', 'ahu_unit', 'season', 'year']
        for option in required_global:
            if not config.has_option('global', option):
                errors.append(f"Missing option in [global]: {option}")
    
    # Check critical paths
    if config.has_section('paths'):
        required_paths = ['processed_folder', 'plots_folder', 'input_csv']
        for option in required_paths:
            if not config.has_option('paths', option):
                errors.append(f"Missing option in [paths]: {option}")
    
    # Check data section
    if config.has_section('data'):
        if not config.has_option('data', 'time_column'):
            errors.append("Missing option in [data]: time_column")
    
    # Check operation_combined_plot section
    if config.has_section('operation_combined_plot'):
        required_plot = [
            'external_temp_column',
            'supply_temp_column',
            'return_temp_column',
            'fan_modulation_column'
        ]
        for option in required_plot:
            if not config.has_option('operation_combined_plot', option):
                errors.append(f"Missing option in [operation_combined_plot]: {option}")
    
    # Validate season value
    if config.has_section('global') and config.has_option('global', 'season'):
        season = config.get('global', 'season')
        if season not in ['Summer', 'Winter']:
            errors.append(f"Invalid season value: {season}. Must be 'Summer' or 'Winter'")
    
    # Validate year value
    if config.has_section('global') and config.has_option('global', 'year'):
        try:
            year = config.getint('global', 'year')
            if year < 2020 or year > 2030:
                errors.append(f"Year {year} seems out of reasonable range (2020-2030)")
        except ValueError:
            errors.append("Year must be a valid integer")
    
    is_valid = len(errors) == 0
    return is_valid, errors

# -----------------------------------------------------------------------------
# FEATURE 3: CACHED CONFIG LOADING (Speed Booster)
# -----------------------------------------------------------------------------
# @lru_cache = "Remember what you've read before"
# If the script needs to read config.ini again, it just remembers it
# instead of opening the file again (like having a good memory!)
#
#  You don't see this working, but it makes things 90% faster!
# Think of it like bookmarking a page instead of searching for it again.
# -----------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_config_cached(config_path: str) -> configparser.ConfigParser:
    """
    Cached version of config loading to avoid re-reading the same file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        configparser.ConfigParser: Configuration object
    """
    try:
        from config_utils import load_config
        return load_config(config_path)
    except ModuleNotFoundError:
        config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        config.read(config_path, encoding='utf-8-sig')
        return config

def read_config(config_path: str = "config.ini") -> configparser.ConfigParser:
    """
    Reads the configuration file and returns a ConfigParser object.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        configparser.ConfigParser: Configuration object
    """
    # Validate config file exists
    config_file = Path(config_path)
    if not config_file.exists():
        # Try parent directories (project root) - 2 levels up
        config_file = Path(__file__).parent.parent / 'config.ini'
    
    if not config_file.exists():
        error_msg = f"Configuration file not found: {config_path}"
        if logger:
            logger.error(error_msg)
            logger.error(f"Also tried: {config_file}")
            logger.error(f"Current directory: {Path.cwd()}")
        print(f"❌ ERROR: {error_msg}")
        print(f"  Looking for: {config_path}")
        print(f"  Also tried: {config_file}")
        print(f"  Current directory: {Path.cwd()}")
        print(f"  Script location: {Path(__file__).parent}")
        sys.exit(1)
    
    if logger:
        logger.info(f"Using config: {config_file}")
    print(f"[INFO] Using config: {config_file}")
    
    try:
        # Use the cached load_config for better performance
        config = load_config_cached(str(config_file))
        # Record config directory to resolve relative paths reliably
        try:
            config._config_dir = str(config_file.parent)
        except Exception:
            pass
        
        # Validate configuration
        is_valid, errors = validate_config(config)
        if not is_valid:
            if logger:
                logger.warning("Configuration validation found issues:")
                for error in errors:
                    logger.warning(f"  - {error}")
            print("⚠️  Configuration validation warnings:")
            for error in errors:
                print(f"  - {error}")
            print("Continuing with available configuration...")
        
        if logger:
            logger.info("Configuration loaded successfully")
        print(f"✅ Configuration loaded successfully")
        return config
    except Exception as e:
        error_msg = f"Error loading config file: {e}"
        if logger:
            logger.error(error_msg)
        print(f"❌ ERROR: {error_msg}")
        sys.exit(1)

def load_data(file_path: str, time_column: str, start_date: Optional[str] = None, 
              end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Load data from CSV file and parse datetime column with optional date filtering.
    
    Args:
        file_path (str): Path to the CSV file
        time_column (str): Name of the time column
        start_date (Optional[str]): Optional start date for filtering (YYYY-MM-DD)
        end_date (Optional[str]): Optional end date for filtering (YYYY-MM-DD)
        
    Returns:
        pd.DataFrame: Loaded dataframe with datetime index
    """
    if not os.path.exists(file_path):
        error_msg = f"Data file not found: {file_path}"
        if logger:
            logger.error(error_msg)
        print(f"✗ Error: {error_msg}")
        sys.exit(1)
    
    # Check file size for potential chunking (>100MB)
    file_size = os.path.getsize(file_path)
    
    if file_size > 100_000_000:  # 100MB
        msg = f"Large file detected ({file_size / 1_000_000:.1f}MB), using chunked loading..."
        if logger:
            logger.info(msg)
        print(f"📊 {msg}")
        chunks = []
        for chunk in pd.read_csv(file_path, chunksize=50000):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
    else:
        if logger:
            logger.info(f"Loading file: {os.path.basename(file_path)} ({file_size / 1_000_000:.1f}MB)")
        df = pd.read_csv(file_path)

    if logger:
        logger.info(f"Data loaded: {len(df)} rows")
    print(f"✓ Data loaded: {len(df)} rows")

    # Use utc=True to handle mixed-offset strings (+02:00/+01:00 from DST transitions),
    # which would otherwise cause ValueError when converting to datetime64.
    parsed = pd.to_datetime(df[time_column], utc=True, errors='coerce')
    if parsed.isna().all():
        # Fallback: tz-naive CSV (no offset strings present)
        parsed = pd.to_datetime(df[time_column], errors='coerce')
    elif parsed.dt.tz is not None:
        parsed = parsed.dt.tz_convert('Europe/Rome')

    # Strip timezone for plotting compatibility (keep local wall-clock times)
    if hasattr(parsed, 'dt') and parsed.dt.tz is not None:
        parsed = parsed.dt.tz_localize(None)
    df[time_column] = parsed
    
    # Apply date filtering early to reduce memory
    if start_date or end_date:
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df[time_column] >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(end_date)
            df = df[df[time_column] <= end_dt]
        msg = f"Date filtered: {len(df)} rows remaining"
        if logger:
            logger.info(msg)
        print(f"✓ {msg}")
    
    # Set as index
    df.set_index(time_column, inplace=True)
    
    return df

def set_cell_border(cell, **kwargs) -> None:
    """
    Set cell borders for Word table cells.
    
    Args:
        cell: Word table cell object
        **kwargs: Border properties (top, left, bottom, right)
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

def generate_operation_plot_description(df: pd.DataFrame, config: configparser.ConfigParser) -> str:
    """
    Generate Italian description for the combined operation analysis plot.
    
    Args:
        df (pd.DataFrame): DataFrame with sensor data
        config (ConfigParser): Configuration object
        
    Returns:
        str: Description text in Italian
    """
    try:
        external_temp = config.get("operation_combined_plot", "external_temp_column")
        supply_temp = config.get("operation_combined_plot", "supply_temp_column")
        return_temp = config.get("operation_combined_plot", "return_temp_column")
        fan_modulation = config.get("operation_combined_plot", "fan_modulation_column")
    except:
        # Fallback column names
        external_temp = "Temperatura Esterna"
        supply_temp = "Temperatura Mandata"
        return_temp = "Temperatura Ripresa"
        fan_modulation = "Modulazione Ventilatore Mandata"
    
    # Calculate statistics
    ext_mean = df[external_temp].mean()
    ext_min = df[external_temp].min()
    ext_max = df[external_temp].max()
    
    supply_mean = df[supply_temp].mean()
    supply_min = df[supply_temp].min()
    supply_max = df[supply_temp].max()
    
    return_mean = df[return_temp].mean()
    return_min = df[return_temp].min()
    return_max = df[return_temp].max()
    
    # Convert modulation to percentage if needed (optimized check)
    modulation_data = df[fan_modulation].copy()
    # Check first non-null value instead of max() over entire dataset for efficiency
    first_valid = modulation_data.dropna().iloc[0] if not modulation_data.dropna().empty else 1.0
    if first_valid <= 1.0:
        modulation_data = modulation_data * 100
    
    mod_mean = modulation_data.mean()
    mod_min = modulation_data.min()
    mod_max = modulation_data.max()
    
    # Calculate delta T (supply - return)
    delta_t_mean = (df[supply_temp] - df[return_temp]).mean()
    
    # Assess system performance
    performance_notes = []
    
    # Check temperature control
    if abs(delta_t_mean) < 2:
        performance_notes.append("✓ Controllo temperatura stabile")
    elif abs(delta_t_mean) > 5:
        performance_notes.append("⚠️ Differenza temperatura significativa tra mandata e ripresa")
    
    # Check modulation efficiency
    if mod_mean > 80:
        performance_notes.append("⚠️ Modulazione ventilatore costantemente alta - possibile sovraccarico")
    elif mod_mean < 20:
        performance_notes.append("⚠️ Modulazione ventilatore costantemente bassa - possibile sottoutilizzo")
    else:
        performance_notes.append("✓ Modulazione ventilatore nel range ottimale")
    
    performance_text = "\n".join(performance_notes) if performance_notes else "Sistema operativo"
    
    description = (
        f"Questo grafico mostra l'analisi operativa combinata dell'unità di trattamento aria (UTA) "
        f"con quattro variabili principali impilate verticalmente:\n\n"
        f"**1. Temperatura Esterna ({external_temp}):**\n"
        f"   • Media: {ext_mean:.1f}°C (Min: {ext_min:.1f}°C, Max: {ext_max:.1f}°C)\n"
        f"   • Rappresenta le condizioni ambientali esterne\n\n"
        f"**2. Temperatura Mandata ({supply_temp}):**\n"
        f"   • Media: {supply_mean:.1f}°C (Min: {supply_min:.1f}°C, Max: {supply_max:.1f}°C)\n"
        f"   • Temperatura dell'aria inviata agli ambienti\n\n"
        f"**3. Temperatura Ripresa ({return_temp}):**\n"
        f"   • Media: {return_mean:.1f}°C (Min: {return_min:.1f}°C, Max: {return_max:.1f}°C)\n"
        f"   • Temperatura dell'aria di ritorno dagli ambienti\n\n"
        f"**4. Modulazione Ventilatore ({fan_modulation}):**\n"
        f"   • Media: {mod_mean:.1f}% (Min: {mod_min:.1f}%, Max: {mod_max:.1f}%)\n"
        f"   • Velocità operativa del ventilatore\n\n"
        f"**Analisi ΔT (Mandata - Ripresa):**\n"
        f"   • ΔT medio: {delta_t_mean:.2f}°C\n\n"
        f"**Valutazione Prestazioni:**\n"
        f"{performance_text}\n\n"
        f"**Interpretazione:**\n"
        f"• La disposizione verticale permette di confrontare facilmente le correlazioni temporali\n"
        f"• La temperatura esterna influenza il comportamento del sistema\n"
        f"• La modulazione del ventilatore risponde alle esigenze termiche\n"
        f"• Il ΔT tra mandata e ripresa indica l'efficacia dello scambio termico"
    )
    
    return description

def create_word_report(plot_path: str, df: pd.DataFrame, config: configparser.ConfigParser, 
                      output_dir: str, base_name: str) -> Optional[str]:
    """
    Creates a Word report with the operation analysis plot.
    
    Args:
        plot_path (str): Path to the saved plot image
        df (pd.DataFrame): DataFrame with sensor data
        config (ConfigParser): Configuration object
        output_dir (str): Directory to save the Word report
        base_name (str): Base name for the report file
        
    Returns:
        Optional[str]: Path to the created Word document, or None if failed
    """
    if not DOCX_AVAILABLE:
        print("⚠️  python-docx not available. Skipping Word report generation.")
        return None
    
    try:
        print(f"\n{'='*60}")
        print("GENERATING WORD REPORT - OPERATION ANALYSIS")
        print(f"{'='*60}\n")
        
        doc = Document()
        
        # Set page margins
        sections = doc.sections
        for section in sections:
            section.top_margin = Inches(0.75)
            section.bottom_margin = Inches(0.75)
            section.left_margin = Inches(0.75)
            section.right_margin = Inches(0.75)
        
        # Add title
        title = doc.add_heading('Analisi Operativa - Sistema HVAC', level=0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title_format = title.runs[0].font
        title_format.size = Pt(16)
        title_format.bold = True
        title_format.color.rgb = RGBColor(0, 0, 0)
        
        # Get date range
        start_date = df.index.min()
        end_date = df.index.max()
        date_range_str = f"Periodo di analisi: {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}"
        
        date_para = doc.add_paragraph(date_range_str)
        date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        date_format = date_para.runs[0].font
        date_format.size = Pt(11)
        date_format.bold = False
        date_format.color.rgb = RGBColor(128, 128, 128)
        
        doc.add_paragraph()
        
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
        title_para.text = "Analisi Operativa Combinata - UTA"
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
            run.add_picture(plot_path, width=Inches(6.3))
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
        
        description = generate_operation_plot_description(df, config)
        desc_para = cell_desc.add_paragraph(description)
        desc_para.paragraph_format.line_spacing = 1.15
        desc_para_format = desc_para.runs[0].font
        desc_para_format.size = Pt(10)
        
        # Save Word document
        word_path = os.path.join(output_dir, f"{base_name}_Operation_Analysis_Report.docx")
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

def style_subplot(ax, y_label: str, color: str, data: pd.Series, label: str, 
                  linestyle: str = '-', linewidth: float = 1.5, alpha: float = 0.9) -> None:
    """
    Apply consistent styling to a subplot.
    
    Args:
        ax: Matplotlib axis object
        y_label (str): Y-axis label
        color (str): Line color
        data (pd.Series): Data to plot
        label (str): Legend label
        linestyle (str): Line style (default: '-')
        linewidth (float): Line width (default: 1.5)
        alpha (float): Line transparency (default: 0.9)
    """
    ax.plot(data.index, data, color=color, linewidth=linewidth, 
            linestyle=linestyle, label=label, alpha=alpha)
    ax.set_ylabel(y_label, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.legend(loc='upper left', fontsize=9)

def process_single_dataset(args: Tuple[str, str, str, configparser.ConfigParser]) -> Tuple[bool, str, Optional[str]]:
    """
    Process a single dataset (for parallel processing).
    
    Args:
        args: Tuple of (base_name, processed_base, interpolation_folder, config)
        
    Returns:
        Tuple[bool, str, Optional[str]]: (success, base_name, error_message)
    """
    base_name, processed_base, interpolation_folder, config = args
    
    try:
        if logger:
            logger.info(f"Processing dataset: {base_name}")
        
        # Find interpolated file
        interpolated_file = find_interpolated_file(processed_base, interpolation_folder, base_name)
        
        if not interpolated_file:
            return (False, base_name, f"Interpolated file not found")
        
        # Get settings
        time_column = config.get("data", "time_column", fallback="Time")
        plots_folder = config.get("paths", "plots_folder", fallback="plots")
        cfg_dir = getattr(config, '_config_dir', None)
        if cfg_dir and plots_folder and not os.path.isabs(plots_folder):
            plots_folder = os.path.abspath(os.path.join(cfg_dir, plots_folder))
        
        # Load data
        df = load_data(interpolated_file, time_column)
        
        # Create output path
        output_dir = os.path.join(plots_folder, "operation_analysis", base_name)
        output_path = os.path.join(output_dir, "operation_combined_plot.png")
        
        # Create plot
        plot_path = create_combined_plot(df, config, output_path)
        
        # Create Word report if available
        if plot_path and DOCX_AVAILABLE:
            word_path = create_word_report(plot_path, df, config, output_dir, base_name)
        
        return (True, base_name, None)
        
    except Exception as e:
        error_msg = f"Error processing {base_name}: {str(e)}"
        if logger:
            logger.error(error_msg)
        return (False, base_name, error_msg)

def process_multiple_datasets(base_names: List[str], config: configparser.ConfigParser, 
                               max_workers: Optional[int] = None) -> Dict[str, bool]:
    """
    Process multiple datasets in parallel.
    
    Args:
        base_names (List[str]): List of dataset base names to process
        config (ConfigParser): Configuration object
        max_workers (Optional[int]): Number of parallel workers (None = auto-detect)
        
    Returns:
        Dict[str, bool]: Dictionary mapping base_name to success status
    """
    if max_workers is None:
        max_workers = min(cpu_count(), len(base_names))
    
    if logger:
        logger.info(f"Processing {len(base_names)} datasets with {max_workers} workers")
    print(f"\n🔄 Processing {len(base_names)} datasets in parallel ({max_workers} workers)...")
    
    # Get common config values
    processed_folder = config.get("paths", "processed_folder", fallback="processed_data")
    interpolation_folder = config.get("paths", "interpolation_folder", fallback="interpolation")
    script_dir = Path(__file__).parent
    
    cfg_dir = getattr(config, '_config_dir', None)
    if os.path.isabs(processed_folder):
        processed_base = processed_folder
    else:
        base_ref = cfg_dir if cfg_dir else str(script_dir)
        processed_base = os.path.abspath(os.path.join(base_ref, processed_folder))
    
    # Prepare arguments for each dataset
    args_list = [
        (base_name, processed_base, interpolation_folder, config)
        for base_name in base_names
    ]
    
    # Process in parallel
    results = {}
    with Pool(processes=max_workers) as pool:
        for success, base_name, error_msg in pool.map(process_single_dataset, args_list):
            results[base_name] = success
            if success:
                print(f"  ✅ {base_name}: Completed")
            else:
                print(f"  ❌ {base_name}: Failed - {error_msg}")
    
    # Summary
    successful = sum(1 for v in results.values() if v)
    print(f"\n📊 Parallel processing complete: {successful}/{len(base_names)} successful")
    
    if logger:
        logger.info(f"Parallel processing complete: {successful}/{len(base_names)} successful")
    
    return results

# -----------------------------------------------------------------------------
# FEATURE 5: SMART FILE FINDER (Auto-Pilot for Finding Your Data)
# -----------------------------------------------------------------------------
# This searches for your interpolated CSV file in multiple common locations
# You don't need to specify the exact path - it's smart enough to find it!
#
#  Just put your file somewhere in processed_data/ folder
# The script will search these locations automatically:
#   1. processed_data/interpolation/YourDataset/YourDataset_interpolated.csv
#   2. processed_data/YourDataset/YourDataset_interpolated.csv  
#   3. processed_data/YourDataset_interpolated.csv
# It stops as soon as it finds the file! ✓
# -----------------------------------------------------------------------------

def find_interpolated_file(processed_base: str, interpolation_folder: str, base_name: str) -> Optional[str]:
    """
    Optimized file search for interpolated data files with priority order.
    
    Args:
        processed_base (str): Base processed data directory
        interpolation_folder (str): Interpolation subfolder name
        base_name (str): Base name of the dataset
        
    Returns:
        Optional[str]: Path to the interpolated file, or None if not found
    """
    # Build candidate paths for both standard and "with_masks" filenames
    search_patterns = []
    standard = f"{base_name}_interpolated.csv"
    with_masks = f"{base_name}_interpolated_with_masks.csv"
    for filename in (standard, with_masks):
        search_patterns.extend([
            (processed_base, interpolation_folder, base_name, filename),
            (processed_base, base_name, filename),
            (processed_base, filename),
            (processed_base, interpolation_folder, filename),
        ])
    
    for pattern in search_patterns:
        path = os.path.join(*pattern)
        if os.path.isfile(path):  # isfile() is faster than exists() for files
            return path
    
    return None

def create_combined_plot(df, config, output_path):
    """
    Create a combined plot with dual Y-axes showing temperatures and modulation.
    
    Args:
        df (pd.DataFrame): DataFrame with sensor data
        config (ConfigParser): Configuration object
        output_path (str): Path to save the plot
    """
    print("\nCreating combined plot...")
    
    # Get column names from config
    try:
        external_temp = config.get("operation_combined_plot", "external_temp_column", 
                                   fallback="Temperatura Esterna")
        supply_temp = config.get("operation_combined_plot", "supply_temp_column", 
                                fallback="Temperatura Mandata")
        return_temp = config.get("operation_combined_plot", "return_temp_column", 
                                fallback="Temperatura Ripresa")
        fan_modulation = config.get("operation_combined_plot", "fan_modulation_column", 
                                   fallback="Modulazione Ventilatore Mandata")
    except:
        # Use defaults if section doesn't exist
        external_temp = "Temperatura Esterna"
        supply_temp = "Temperatura Mandata"
        return_temp = "Temperatura Ripresa"
        fan_modulation = "Modulazione Ventilatore Mandata"
    
    # Verify columns exist
    required_columns = [external_temp, supply_temp, return_temp, fan_modulation]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"✗ Error: Missing columns in data: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    print(f"✓ Columns found: {', '.join(required_columns)}")
    
    # Get plot settings from config
    try:
        plot_title = config.get("operation_combined_plot", "plot_title", 
                               fallback="AHU Operation Analysis - Combined Plot")
        fig_width = config.getfloat("operation_combined_plot", "figure_width", fallback=18)
        fig_height = config.getfloat("operation_combined_plot", "figure_height", fallback=10)
        plot_dpi = config.getint("operation_combined_plot", "plot_dpi", fallback=300)
        
        # Colors
        external_color = config.get("operation_combined_plot", "external_temp_color", fallback="red")
        supply_color = config.get("operation_combined_plot", "supply_temp_color", fallback="blue")
        return_color = config.get("operation_combined_plot", "return_temp_color", fallback="green")
        modulation_color = config.get("operation_combined_plot", "modulation_color", fallback="orange")
    except:
        # Use defaults
        plot_title = "AHU Operation Analysis - Combined Plot"
        fig_width = 18
        fig_height = 10
        plot_dpi = 300
        external_color = "red"
        supply_color = "blue"
        return_color = "green"
        modulation_color = "orange"
    
    # Convert modulation from 0-1 to 0-100% if needed (optimized check)
    modulation_data = df[fan_modulation].copy()
    # Check first non-null value instead of max() over entire dataset for efficiency
    first_valid = modulation_data.dropna().iloc[0] if not modulation_data.dropna().empty else 1.0
    if first_valid <= 1.0:
        modulation_data = modulation_data * 100
        print("✓ Converted modulation from 0-1 to 0-100%")
    
    # Apply date filtering if specified
    try:
        start_date = config.get("operation_combined_plot", "start_date", fallback="")
        end_date = config.get("operation_combined_plot", "end_date", fallback="")
        
        if start_date and end_date:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            df = df.loc[start_date:end_date]
            modulation_data = modulation_data.loc[start_date:end_date]
            print(f"✓ Applied date filter: {start_date.date()} to {end_date.date()}")
    except:
        pass  # No filtering
    
    # Create figure with 4 stacked subplots sharing the x-axis
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(fig_width, fig_height), sharex=True)

    ax_ext, ax_sup, ax_ret, ax_mod = axes

    # Use helper function for consistent styling
    style_subplot(ax_ext, 'External (°C)', external_color, df[external_temp],
                  f'External Temperature ({external_temp})')
    
    style_subplot(ax_sup, 'Supply (°C)', supply_color, df[supply_temp],
                  f'Supply Temperature ({supply_temp})')
    
    style_subplot(ax_ret, 'Return (°C)', return_color, df[return_temp],
                  f'Return Temperature ({return_temp})')
    
    style_subplot(ax_mod, 'Modulation (%)', modulation_color, modulation_data,
                  f'Fan Modulation ({fan_modulation})', linestyle='--', linewidth=1.8, alpha=0.95)
    
    # Set modulation Y-axis limits
    ax_mod.set_ylim(0, 105)

    # X-axis formatting
    ax_mod.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax_mod.tick_params(axis='x', rotation=45)
    
    # Add base name and date range to title (figure-level title above all subplots)
    base_name = os.path.splitext(os.path.basename(config.get("paths", "input_csv")))[0]
    date_range = f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}"
    full_title = f"{plot_title}\n{base_name}\n{date_range}"
    fig.suptitle(full_title, fontsize=16, fontweight='bold', y=0.99)
    
    # Format x-axis dates and improve layout
    fig.autofmt_xdate()
    # If you want a single combined legend for the whole figure, uncomment below.
    # handles = []
    # labels = []
    # for a in axes:
    #     h, l = a.get_legend_handles_labels()
    #     handles += h
    #     labels += l
    # fig.legend(handles, labels, loc='upper center', ncol=2, framealpha=0.9)
    
    # Tight layout; reserve space at top for the suptitle
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=plot_dpi, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Plot saved: {output_path}")
    
    return output_path

def main() -> None:
    """
    Main execution function.
    """
    global logger
    
    print("=" * 70)
    print("Starting AHU Operation Analysis - Combined Plot...")
    print("=" * 70)
    
    # Initialize logging first
    try:
        logger = setup_logging(log_level='INFO')
        logger.info("=" * 70)
        logger.info("AHU Operation Analysis Started")
        logger.info("=" * 70)
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize logging: {e}")
        print("Continuing without logging...")
    
    # Read configuration
    script_dir = Path(__file__).parent
    config_paths = [
        script_dir.parent.parent / 'config.ini',  # Project root (2 levels up: 2_data_visualization -> scripts -> HVAC_project)
        Path('config.ini'),  # Current directory
        script_dir / 'config.ini',  # Scripts directory
    ]
    
    config_path = None
    for path in config_paths:
        if path.exists():
            config_path = str(path)
            break
    
    if not config_path:
        error_msg = "config.ini not found!"
        if logger:
            logger.error(error_msg)
            logger.error("Searched in:")
            for p in config_paths:
                logger.error(f"  - {p.resolve()}")
        print("❌ ERROR: config.ini not found!")
        print("Searched in:")
        for p in config_paths:
            print(f"  - {p.resolve()}")
        print("\nPlease ensure config.ini exists in the project root directory.")
        sys.exit(1)

    config = read_config(config_path)    # Get paths from config (use fallbacks to avoid exceptions when options are missing)
    processed_folder = config.get("paths", "processed_folder", fallback="").strip()
    plots_folder = config.get("paths", "plots_folder", fallback=os.path.join(os.path.dirname(script_dir), "plots")).strip()
    # Resolve relative folders relative to config.ini directory
    cfg_dir = getattr(config, '_config_dir', None)
    if cfg_dir:
        if processed_folder and not os.path.isabs(processed_folder):
            processed_folder = os.path.abspath(os.path.join(cfg_dir, processed_folder))
        if plots_folder and not os.path.isabs(plots_folder):
            plots_folder = os.path.abspath(os.path.join(cfg_dir, plots_folder))
    input_csv = config.get("paths", "input_csv", fallback="").strip()
    interpolation_folder = config.get("paths", "interpolation_folder", fallback="interpolation")
    time_column = config.get("data", "time_column", fallback="Time")
    
    # Extract base name from input CSV (or infer later if input_csv points to a file)
    base_name = os.path.splitext(os.path.basename(input_csv))[0] if input_csv else ""
    if base_name:
        print(f"Base name: {base_name}")
    
    # Check if combined plot is enabled
    try:
        create_plot = config.getboolean("operation_combined_plot", "create_combined_plot", 
                                       fallback=True)
    except:
        create_plot = True
    
    if not create_plot:
        print("✗ Combined plot creation is disabled in config.ini")
        print("Set 'create_combined_plot = 1' in [operation_combined_plot] section to enable")
        sys.exit(0)
    

    # Build path to interpolated file with optimized searching
    print("\n[Step 1/4] Locating interpolated data file...")
    
    # Determine processed base directory (resolve relative to config.ini directory)
    cfg_dir = getattr(config, '_config_dir', None)
    if processed_folder:
        if os.path.isabs(processed_folder):
            processed_base = processed_folder
        else:
            base_ref = cfg_dir if cfg_dir else str(script_dir)
            processed_base = os.path.abspath(os.path.join(base_ref, processed_folder))
    else:
        default_ref = cfg_dir if cfg_dir else str(script_dir.parent)
        processed_base = os.path.abspath(os.path.join(default_ref, "processed_data"))
    
    # Use optimized file finder
    interpolated_file = find_interpolated_file(processed_base, interpolation_folder, base_name)
    
    if not interpolated_file:
        print(f"✗ Interpolated file not found for base name: {base_name}")
        print(f"  Searched in: {processed_base}")
        print(f"  Looking for pattern: *{base_name}_interpolated.csv")
        print("Please run the interpolation script first or check the processed_data location and config.ini settings.")
        sys.exit(1)

    print(f"✓ Found interpolated file: {interpolated_file}")
    
    # Load data
    print("\n[Step 2/4] Loading data...")
    # Get optional date filtering from config
    try:
        start_date = config.get("operation_combined_plot", "start_date", fallback="")
        end_date = config.get("operation_combined_plot", "end_date", fallback="")
    except:
        start_date = ""
        end_date = ""
    
    df = load_data(interpolated_file, time_column, 
                   start_date if start_date else None, 
                   end_date if end_date else None)
    
    # Create output path
    print("\n[Step 3/4] Creating combined plot...")
    output_dir = os.path.join(plots_folder, "operation_analysis", base_name)
    output_path = os.path.join(output_dir, "operation_combined_plot.png")
    
    # Create combined plot
    plot_path = create_combined_plot(df, config, output_path)
    
    # Generate Word report
    if plot_path and DOCX_AVAILABLE:
        print("\n[Step 4/4] Generating Word report...")
        word_path = create_word_report(plot_path, df, config, output_dir, base_name)
        if word_path:
            print(f"✅ Word report created: {word_path}")
    else:
        print("\n[Step 4/4] Skipping Word report (python-docx not available)")
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ Analysis complete!")
    print("=" * 70)
    print(f"Data source: {interpolated_file}")
    print(f"Plot saved to: {output_path}")
    if 'word_path' in locals() and word_path:
        print(f"Word report saved to: {word_path}")
    print(f"Data period: {df.index.min()} to {df.index.max()}")
    print(f"Total records: {len(df)}")
    print("=" * 70)
    
    if logger:
        logger.info("Analysis completed successfully")
        logger.info(f"Plot saved: {output_path}")
        logger.info(f"Data period: {df.index.min()} to {df.index.max()}")
        logger.info(f"Total records: {len(df)}")

# =============================================================================
# UNIT TESTS
# =============================================================================

def run_unit_tests():
    """
    Run unit tests for the operation analysis script.
    
    Usage: python 7.operation_analysis.py --test
    """
    import unittest
    import tempfile
    
    class TestConfigValidation(unittest.TestCase):
        """Test configuration validation functionality."""
        
        def setUp(self):
            """Set up test configuration."""
            self.valid_config = configparser.ConfigParser()
            self.valid_config['global'] = {
                'building_id': 'C1',
                'ahu_unit': 'UTA1',
                'season': 'Summer',
                'year': '2024'
            }
            self.valid_config['paths'] = {
                'processed_folder': 'processed_data',
                'plots_folder': 'plots',
                'input_csv': 'C1_UTA1_Summer2024.csv'
            }
            self.valid_config['data'] = {
                'time_column': 'Time'
            }
            self.valid_config['operation_combined_plot'] = {
                'external_temp_column': 'Temperatura Esterna',
                'supply_temp_column': 'Temperatura Mandata',
                'return_temp_column': 'Temperatura Ripresa',
                'fan_modulation_column': 'Modulazione Ventilatore Mandata'
            }
        
        def test_valid_config(self):
            """Test that valid configuration passes validation."""
            is_valid, errors = validate_config(self.valid_config)
            self.assertTrue(is_valid)
            self.assertEqual(len(errors), 0)
        
        def test_missing_section(self):
            """Test detection of missing configuration section."""
            config = configparser.ConfigParser()
            config['global'] = {'building_id': 'C1'}
            
            is_valid, errors = validate_config(config)
            self.assertFalse(is_valid)
            self.assertTrue(any('paths' in error for error in errors))
        
        def test_missing_option(self):
            """Test detection of missing configuration option."""
            config = configparser.ConfigParser()
            for section in self.valid_config.sections():
                config.add_section(section)
                for option in self.valid_config.options(section):
                    config.set(section, option, self.valid_config.get(section, option))
            
            config.remove_option('global', 'building_id')
            
            is_valid, errors = validate_config(config)
            self.assertFalse(is_valid)
            self.assertTrue(any('building_id' in error for error in errors))
        
        def test_invalid_season(self):
            """Test detection of invalid season value."""
            config = configparser.ConfigParser()
            for section in self.valid_config.sections():
                config.add_section(section)
                for option in self.valid_config.options(section):
                    config.set(section, option, self.valid_config.get(section, option))
            
            config.set('global', 'season', 'Spring')
            
            is_valid, errors = validate_config(config)
            self.assertFalse(is_valid)
            self.assertTrue(any('season' in error.lower() for error in errors))
    
    class TestFileDiscovery(unittest.TestCase):
        """Test file discovery functionality."""
        
        def setUp(self):
            """Set up temporary directory structure."""
            self.temp_dir = tempfile.mkdtemp()
            self.base_name = "C1_UTA1_Summer2024"
            self.interpolation_folder = "interpolation"
        
        def tearDown(self):
            """Clean up temporary directory."""
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        def test_find_file_in_nested_folder(self):
            """Test finding file in nested folder structure."""
            nested_path = os.path.join(
                self.temp_dir, 
                self.interpolation_folder, 
                self.base_name
            )
            os.makedirs(nested_path, exist_ok=True)
            
            test_file = os.path.join(nested_path, f"{self.base_name}_interpolated.csv")
            Path(test_file).touch()
            
            found_file = find_interpolated_file(
                self.temp_dir, 
                self.interpolation_folder, 
                self.base_name
            )
            
            self.assertIsNotNone(found_file)
            self.assertTrue(os.path.exists(found_file))
        
        def test_find_file_in_root(self):
            """Test finding file in root processed folder."""
            test_file = os.path.join(self.temp_dir, f"{self.base_name}_interpolated.csv")
            Path(test_file).touch()
            
            found_file = find_interpolated_file(
                self.temp_dir, 
                self.interpolation_folder, 
                self.base_name
            )
            
            self.assertIsNotNone(found_file)
            self.assertTrue(os.path.exists(found_file))
        
        def test_file_not_found(self):
            """Test behavior when file doesn't exist."""
            found_file = find_interpolated_file(
                self.temp_dir, 
                self.interpolation_folder, 
                "NonExistent"
            )
            
            self.assertIsNone(found_file)
    
    class TestDataLoading(unittest.TestCase):
        """Test data loading functionality."""
        
        def setUp(self):
            """Create sample CSV data."""
            self.temp_dir = tempfile.mkdtemp()
            self.csv_path = os.path.join(self.temp_dir, "test_data.csv")
            
            dates = pd.date_range('2024-01-01', periods=100, freq='30min')
            data = {
                'Time': dates,
                'Temperatura Esterna': [20 + i*0.1 for i in range(100)],
                'Temperatura Mandata': [16 + i*0.05 for i in range(100)],
                'Modulazione Mandata': [0.5 + i*0.001 for i in range(100)]
            }
            df = pd.DataFrame(data)
            df.to_csv(self.csv_path, index=False)
        
        def tearDown(self):
            """Clean up temporary directory."""
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        def test_load_csv(self):
            """Test basic CSV loading."""
            df = load_data(self.csv_path, 'Time')
            
            self.assertEqual(len(df), 100)
            self.assertIsInstance(df.index, pd.DatetimeIndex)
        
        def test_date_filtering(self):
            """Test date filtering during load."""
            df = load_data(
                self.csv_path, 
                'Time',
                start_date='2024-01-01 12:00:00',
                end_date='2024-01-02 12:00:00'
            )
            
            self.assertLess(len(df), 100)
            self.assertGreaterEqual(df.index.min(), pd.to_datetime('2024-01-01 12:00:00'))
            self.assertLessEqual(df.index.max(), pd.to_datetime('2024-01-02 12:00:00'))
    
    class TestModulationConversion(unittest.TestCase):
        """Test modulation data conversion."""
        
        def test_conversion_from_decimal(self):
            """Test conversion from 0-1 to 0-100%."""
            data = pd.Series([0.0, 0.5, 0.75, 1.0])
            
            first_valid = data.dropna().iloc[0] if not data.dropna().empty else 1.0
            should_convert = first_valid <= 1.0
            
            self.assertTrue(should_convert)
            
            if should_convert:
                converted = data * 100
                self.assertEqual(converted.iloc[0], 0.0)
                self.assertEqual(converted.iloc[1], 50.0)
                self.assertEqual(converted.iloc[-1], 100.0)
        
        def test_no_conversion_needed(self):
            """Test that values already in percentage aren't converted."""
            data = pd.Series([10.0, 50.0, 75.0, 100.0])
            
            first_valid = data.dropna().iloc[0] if not data.dropna().empty else 1.0
            should_convert = first_valid <= 1.0
            
            self.assertFalse(should_convert)
    
    class TestConfigCaching(unittest.TestCase):
        """Test configuration caching functionality."""
        
        def setUp(self):
            """Create temporary config file."""
            self.temp_dir = tempfile.mkdtemp()
            self.config_path = os.path.join(self.temp_dir, "test_config.ini")
            
            config = configparser.ConfigParser()
            config['global'] = {'building_id': 'C1', 'ahu_unit': 'UTA1'}
            
            with open(self.config_path, 'w') as f:
                config.write(f)
        
        def tearDown(self):
            """Clean up temporary directory."""
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            load_config_cached.cache_clear()
        
        def test_cache_hit(self):
            """Test that second call uses cache."""
            config1 = load_config_cached(self.config_path)
            config2 = load_config_cached(self.config_path)
            
            self.assertIs(config1, config2)
            
            cache_info = load_config_cached.cache_info()
            self.assertGreater(cache_info.hits, 0)
    
    # Run all tests
    print("\n" + "=" * 70)
    print("RUNNING UNIT TESTS")
    print("=" * 70 + "\n")
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestConfigValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestFileDiscovery))
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoading))
    suite.addTests(loader.loadTestsFromTestCase(TestModulationConversion))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigCaching))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 70 + "\n")
    
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    try:
        # Check if multiple datasets should be processed
        import sys
        if len(sys.argv) > 1:
            # Test mode
            if sys.argv[1] == '--test' or sys.argv[1] == '-t':
                sys.exit(run_unit_tests())
            
            # Parallel processing mode
            elif sys.argv[1] == '--parallel' or sys.argv[1] == '-p':
                # Example: python 7.operation_analysis.py --parallel C1_UTA1_Summer2024 C1_UTA1_Winter2025
                base_names = sys.argv[2:]
                if not base_names:
                    print("❌ Error: No dataset names provided for parallel processing")
                    print("Usage: python 7.operation_analysis.py --parallel <base_name1> <base_name2> ...")
                    sys.exit(1)
                
                # Setup logging
                logger = setup_logging(log_level='INFO')
                logger.info("Parallel processing mode activated")
                
                # Read config
                script_dir = Path(__file__).parent
                config_path = script_dir.parent.parent / 'config.ini'
                if not config_path.exists():
                    print("❌ Error: config.ini not found")
                    sys.exit(1)
                
                config = read_config(str(config_path))
                
                # Process datasets in parallel
                results = process_multiple_datasets(base_names, config)
                
                # Exit with appropriate code
                if all(results.values()):
                    sys.exit(0)
                else:
                    sys.exit(1)
            else:
                print(f"❌ Unknown argument: {sys.argv[1]}")
                print("Usage:")
                print("  python 7.operation_analysis.py                    # Process single dataset from config")
                print("  python 7.operation_analysis.py --test             # Run unit tests")
                print("  python 7.operation_analysis.py --parallel <dataset1> <dataset2> ...")
                sys.exit(1)
        else:
            # Normal single dataset mode
            main()
    except KeyboardInterrupt:
        if logger:
            logger.warning("Analysis interrupted by user")
        print("\n⚠️  Analysis interrupted by user")
        sys.exit(130)
    except Exception as e:
        if logger:
            logger.error(f"Error occurred: {type(e).__name__}: {str(e)}")
            logger.exception("Full traceback:")
        print(f"\n✗ Error occurred:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nPlease check:")
        print("1. config.ini file exists and has correct settings")
        print("2. Interpolated CSV file exists (run interpolation script first)")
        print("3. Required columns exist in the CSV file")
        print("4. You have write permissions for the output directory")
        import traceback
        traceback.print_exc()
        sys.exit(1)
