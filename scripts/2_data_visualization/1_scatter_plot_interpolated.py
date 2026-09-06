"""
SCATTER PLOTS AND DISTRIBUTIONS - HVAC System Data Exploration (INTERPOLATED DATA)
====================================================================================

Purpose:
--------
Same as 2_scatter_plot_and_distributions.py but reads from the INTERPOLATED data folder
instead of the merged data folder.

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
Run after interpolation step. Reads from:
  processed_data/interpolation/[dataset]/[dataset]_interpolated.csv

Output:
-------
Saves to: HVAC_project/plots/distributions_and_scatter/[dataset]_interpolated/
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
from scipy import stats
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_config(config_path):
    """Load configuration file with interpolation support."""
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(config_path, encoding='utf-8-sig')
    return config


def find_interpolated_data(config):
    """
    Find the most recent interpolated data file based on config settings.

    Returns:
        str: Path to interpolated CSV file
    """
    building_id = config.get('global', 'building_id')
    ahu_unit = config.get('global', 'ahu_unit')
    season = config.get('global', 'season')
    year = config.get('global', 'year')

    base_name = f"{building_id}_{ahu_unit}_{season}{year}"

    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    # ── KEY CHANGE: point to interpolation folder ──
    interp_dir = project_root / 'processed_data' / 'interpolation' / base_name
    interp_file = interp_dir / f"{base_name}_interpolated.csv"

    if interp_file.exists():
        print(f"✓ Found interpolated data: {interp_file}")
        return str(interp_file), base_name
    else:
        # Try alternative filename patterns
        patterns = list(interp_dir.glob("*interpolat*.csv")) if interp_dir.exists() else []
        if patterns:
            chosen = patterns[0]
            print(f"✓ Found interpolated data (alt pattern): {chosen}")
            return str(chosen), base_name
        print(f"❌ Interpolated data not found at: {interp_file}")
        print(f"\nExpected location: {interp_dir}")
        print("Please run the interpolation step first.")
        sys.exit(1)


def load_data(filepath):
    """Load interpolated CSV data with proper datetime parsing."""
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
    """Define column name mapping for Italian/English compatibility."""
    return {
        'TM':       ['Temperatura Mandata', 'Temp. Mandata', 'T Mandata'],
        'TR':       ['Temperatura Ripresa', 'Temp. Ripresa', 'T Ripresa'],
        'T_Sat':    ['Temperatura Saturazione', 'Temp. Saturazione', 'T Saturazione'],
        'T_Ext':    ['Temperatura Esterna', 'Temp. Esterna'],
        'Umid_Ext': ['Umidita Esterna', 'Umid. Esterna'],
        'SP_TM':    ['Set Points Temperatura Mandata', 'SP Temp. Mand.'],
        'Fan_M':    ['Modulazione Ventilatore Mandata', 'Mod. V. Mandata'],
        'Fan_R':    ['Modulazione Ventilatore Ripresa', 'Mod. V. Ripresa']
    }


def find_column(df, standard_name, column_mapping):
    """Find actual column name in dataframe from list of variants."""
    variants = column_mapping.get(standard_name, [])
    for variant in variants:
        if variant in df.columns:
            return variant
    return None


# ─────────────────────────────────────────────────────────────────────────────
# DISTRIBUTION PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def create_distribution_plots(df, output_dir, column_mapping):
    """Create combined histogram and box plot distributions for key variables."""
    print("\n📊 Generating distribution plots (Histogram + Box Plot)...")

    variables = [
        ('TM',       'Temperatura Mandata',           '°C', 'Temperature'),
        ('TR',       'Temperatura Ripresa',            '°C', 'Temperature'),
        ('T_Sat',    'Temperatura Saturazione',        '°C', 'Temperature'),
        ('SP_TM',    'Setpoint Temperatura Mandata',   '°C', 'Temperature'),
        ('Fan_M',    'Modulazione Ventilatore Mandata','%',  'Percentage'),
        ('Fan_R',    'Modulazione Ventilatore Ripresa','%',  'Percentage'),
        ('T_Ext',    'Temperatura Esterna',            '°C', 'Temperature'),
        ('Umid_Ext', 'Umidità Esterna',                '%',  'Percentage')
    ]

    for std_name, display_name, unit, var_type in variables:
        col = find_column(df, std_name, column_mapping)

        if col is None:
            print(f"   ⚠️  Skipping {display_name} (column not found)")
            continue

        data = df[col].dropna()

        if len(data) == 0:
            print(f"   ⚠️  Skipping {display_name} (no valid data)")
            continue

        fig, axes = plt.subplots(2, 1, figsize=(12, 10),
                                 gridspec_kw={'height_ratios': [3, 1]})

        # ── Histogram ──
        ax_hist = axes[0]
        n, bins, patches = ax_hist.hist(data, bins=50, alpha=0.7, color='steelblue',
                                        edgecolor='black', linewidth=0.5, density=False)
        # KDE can fail when data has near-zero variance (e.g. constant setpoint).
        try:
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            kde_values = kde(x_range)
            kde_scaled = kde_values * len(data) * (bins[1] - bins[0])
            ax_hist.plot(x_range, kde_scaled, 'r-', linewidth=2, label='KDE')
        except np.linalg.LinAlgError:
            # Near-constant data (e.g. fixed setpoint) — skip KDE overlay
            ax_hist.axvline(data.iloc[0], color='r', linewidth=2,
                            linestyle='--', label=f'Valore costante: {data.iloc[0]:.2f}')

        mean_val   = data.mean()
        median_val = data.median()
        std_val    = data.std()
        q25        = data.quantile(0.25)
        q75        = data.quantile(0.75)

        ax_hist.axvline(mean_val,   color='red',    linestyle='--', linewidth=2,
                        label=f'Media: {mean_val:.2f} {unit}',   alpha=0.8)
        ax_hist.axvline(median_val, color='orange', linestyle='--', linewidth=2,
                        label=f'Mediana: {median_val:.2f} {unit}', alpha=0.8)

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
        ax_hist.set_title(f'Distribuzione (Interpolated): {display_name}',
                          fontsize=14, fontweight='bold', pad=15)
        ax_hist.legend(loc='upper left', framealpha=0.9)
        ax_hist.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # ── Box plot ──
        ax_box = axes[1]
        ax_box.boxplot([data], vert=False, widths=0.6, patch_artist=True,
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

        q1  = data.quantile(0.25)
        q3  = data.quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        outliers    = data[(data < lower_fence) | (data > upper_fence)]
        outlier_pct = (len(outliers) / len(data)) * 100

        ax_box.text(0.02, 0.5, f'Outliers: {len(outliers):,} ({outlier_pct:.2f}%)',
                    transform=ax_box.transAxes, fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        date_min = df['Time'].min().strftime('%d/%m/%Y')
        date_max = df['Time'].max().strftime('%d/%m/%Y')
        fig.text(0.5, 0.98, f'Periodo: {date_min} - {date_max}',
                 ha='center', va='top', fontsize=10, style='italic', color='gray')

        plt.tight_layout(rect=[0, 0, 1, 0.98])

        output_path = output_dir / f'dist_{std_name}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   ✓ {display_name}: mean={mean_val:.2f}, outliers={len(outliers):,} ({outlier_pct:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# SCATTER PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def create_scatter_plot(df, x_col, y_col, title, xlabel, ylabel,
                        output_path, date_range=None, diagonal=False,
                        threshold=None, computed_y=None):
    """Create publication-quality scatter plot with density visualization."""
    if computed_y is not None:
        mask   = df[x_col].notna() & computed_y.notna()
        x_data = df.loc[mask, x_col]
        y_data = computed_y.loc[mask]
    else:
        mask   = df[x_col].notna() & df[y_col].notna()
        x_data = df.loc[mask, x_col]
        y_data = df.loc[mask, y_col]

    if len(x_data) == 0:
        print(f"   ⚠️  No valid data for {title}")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    if len(x_data) > 5000:
        hexbin = ax.hexbin(x_data, y_data, gridsize=50, cmap='Blues',
                           mincnt=1, alpha=0.8, rasterized=True)
        plt.colorbar(hexbin, ax=ax, label='Densità punti')
    else:
        ax.scatter(x_data, y_data, alpha=0.4, s=15, c='steelblue',
                   edgecolors='none', rasterized=True)

    if diagonal:
        lims = [
            np.min([ax.get_xlim()[0], ax.get_ylim()[0]]),
            np.max([ax.get_xlim()[1], ax.get_ylim()[1]]),
        ]
        ax.plot(lims, lims, 'r--', alpha=0.6, linewidth=2,
                label='Riferimento perfetto (y=x)', zorder=10)
        if threshold is not None:
            ax.fill_between(lims,
                            [l - threshold for l in lims],
                            [l + threshold for l in lims],
                            alpha=0.15, color='green',
                            label=f'±{threshold}°C tolleranza', zorder=5)

    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')

    if date_range:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=25)
        ax.text(0.5, 1.02, date_range, transform=ax.transAxes,
                ha='center', va='bottom', fontsize=10, style='italic', color='gray')
    else:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    corr    = x_data.corr(y_data)
    textstr = f'n = {len(x_data):,}\nCorrelazione = {corr:.3f}\n'

    if diagonal and threshold is not None:
        rmse        = np.sqrt(np.mean((y_data - x_data) ** 2))
        mae         = np.abs(y_data - x_data).mean()
        tracking_ok  = (np.abs(y_data - x_data) <= threshold).sum()
        tracking_pct = (tracking_ok / len(x_data)) * 100
        textstr += f'RMSE = {rmse:.2f}°C\nMAE = {mae:.2f}°C\nEntro ±{threshold}°C: {tracking_pct:.1f}%'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')

    if diagonal:
        ax.legend(loc='lower right', framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Saved: {Path(output_path).name} (r={corr:.3f})")


def generate_scatter_plots(df, output_dir, column_mapping):
    """Generate specific feature-pair scatter plots."""
    print("\n📊 Generating scatter plots (feature-pair relationships)...")

    date_min   = df['Time'].min().strftime('%d/%m/%Y')
    date_max   = df['Time'].max().strftime('%d/%m/%Y')
    date_range = f"Periodo: {date_min} - {date_max}"

    cols = {std: find_column(df, std, column_mapping) for std in column_mapping}

    missing = [name for name, col in cols.items() if col is None]
    if missing:
        print(f"\n   ℹ️  Columns not found (some plots may be skipped): {', '.join(missing)}")

    print("\n📍 Creating feature-pair scatter plots:")

    # 1. TM vs SP_TM
    if cols['TM'] and cols['SP_TM']:
        print("\n1️⃣  TM vs SP_TM (Control Tracking)")
        create_scatter_plot(
            df, cols['SP_TM'], cols['TM'],
            title='Control Tracking: Temperatura Mandata vs Setpoint (Interpolated)',
            xlabel='Setpoint Temperatura Mandata (°C)',
            ylabel='Temperatura Mandata Effettiva (°C)',
            output_path=output_dir / 'scatter_1_TM_vs_SP_TM.png',
            date_range=date_range, diagonal=True, threshold=2.0
        )

    # 2. TR vs TM
    if cols['TR'] and cols['TM']:
        print("\n2️⃣  TR vs TM (Thermal Relationships)")
        create_scatter_plot(
            df, cols['TM'], cols['TR'],
            title='Relazione Termica: Temperatura Ripresa vs Mandata (Interpolated)',
            xlabel='Temperatura Mandata (°C)',
            ylabel='Temperatura Ripresa (°C)',
            output_path=output_dir / 'scatter_2_TR_vs_TM.png',
            date_range=date_range, diagonal=False
        )

    # 3. OutdoorT vs TM
    if cols['T_Ext'] and cols['TM']:
        print("\n3️⃣  OutdoorT vs TM (Seasonal Influence)")
        create_scatter_plot(
            df, cols['T_Ext'], cols['TM'],
            title='Influenza Stagionale: Temperatura Esterna vs Mandata (Interpolated)',
            xlabel='Temperatura Esterna (°C)',
            ylabel='Temperatura Mandata (°C)',
            output_path=output_dir / 'scatter_3_OutdoorT_vs_TM.png',
            date_range=date_range, diagonal=False
        )

    # 4. FanMandata vs FanRipresa
    if cols['Fan_M'] and cols['Fan_R']:
        print("\n4️⃣  FanMandata vs FanRipresa (System Balance)")
        create_scatter_plot(
            df, cols['Fan_M'], cols['Fan_R'],
            title='Bilanciamento Sistema: Ventilatore Mandata vs Ripresa (Interpolated)',
            xlabel='Modulazione Ventilatore Mandata (%)',
            ylabel='Modulazione Ventilatore Ripresa (%)',
            output_path=output_dir / 'scatter_4_FanM_vs_FanR.png',
            date_range=date_range, diagonal=True, threshold=5.0
        )

    # 5. FanMandata vs (TR - TM)
    if cols['Fan_M'] and cols['TR'] and cols['TM']:
        print("\n5️⃣  FanMandata vs (TR-TM) (Fan Effort vs Thermal Effect)")
        thermal_diff = df[cols['TR']] - df[cols['TM']]
        create_scatter_plot(
            df, cols['Fan_M'], None,
            title='Sforzo Ventilatore vs Effetto Termico (Interpolated)',
            xlabel='Modulazione Ventilatore Mandata (%)',
            ylabel='Differenza Termica TR − TM (°C)',
            output_path=output_dir / 'scatter_5_FanM_vs_ThermalDiff.png',
            date_range=date_range, diagonal=False,
            computed_y=thermal_diff
        )

    # 6. OutdoorT vs OutdoorRH
    if cols['T_Ext'] and cols['Umid_Ext']:
        print("\n6️⃣  OutdoorT vs OutdoorRH (Weather Sensor Validation)")
        create_scatter_plot(
            df, cols['T_Ext'], cols['Umid_Ext'],
            title='Validazione Sensori Meteo: Temperatura vs Umidità Esterna (Interpolated)',
            xlabel='Temperatura Esterna (°C)',
            ylabel='Umidità Relativa Esterna (%)',
            output_path=output_dir / 'scatter_6_OutdoorT_vs_OutdoorRH.png',
            date_range=date_range, diagonal=False
        )


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def create_summary_report(df, output_dir, column_mapping):
    """Generate a text summary report of the interpolated dataset."""
    print("\n📋 Generating summary report...")

    report_lines = [
        "=" * 80,
        "HVAC DATA EXPLORATION SUMMARY - INTERPOLATED DATA",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset:   {output_dir.name}",
        f"Source:    interpolation folder",
        "",
        f"Dataset Overview",
        f"  Rows:       {len(df):,}",
        f"  Columns:    {len(df.columns)}",
        f"  Date start: {df['Time'].min()}",
        f"  Date end:   {df['Time'].max()}",
        "",
        "Variable Statistics",
        "-" * 80,
    ]

    variables = [
        ('TM',       'Temperatura Mandata',           '°C'),
        ('TR',       'Temperatura Ripresa',            '°C'),
        ('T_Sat',    'Temperatura Saturazione',        '°C'),
        ('SP_TM',    'Setpoint Temperatura Mandata',   '°C'),
        ('Fan_M',    'Modulazione Ventilatore Mandata','%'),
        ('Fan_R',    'Modulazione Ventilatore Ripresa','%'),
        ('T_Ext',    'Temperatura Esterna',            '°C'),
        ('Umid_Ext', 'Umidità Esterna',                '%'),
    ]

    for std_name, display_name, unit in variables:
        col = find_column(df, std_name, column_mapping)
        if col is None:
            report_lines.append(f"\n{display_name}: NOT FOUND")
            continue
        data = df[col].dropna()
        if len(data) == 0:
            report_lines.append(f"\n{display_name}: NO VALID DATA")
            continue

        q1  = data.quantile(0.25)
        q3  = data.quantile(0.75)
        iqr = q3 - q1
        outliers = data[(data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)]

        report_lines += [
            f"\n{display_name} ({unit})",
            f"  n={len(data):,}  mean={data.mean():.2f}  std={data.std():.2f}  "
            f"min={data.min():.2f}  max={data.max():.2f}",
            f"  Q25={q1:.2f}  Q75={q3:.2f}  IQR={iqr:.2f}  "
            f"outliers={len(outliers):,} ({(len(outliers)/len(data))*100:.1f}%)",
        ]

    report_lines += ["", "=" * 80, "END OF REPORT", "=" * 80]

    report_path = output_dir / 'analysis_summary_interpolated.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"   ✓ Summary saved: {report_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# WORD REPORT (optional – mirrors original script's generate_word_report)
# ─────────────────────────────────────────────────────────────────────────────

def add_plot_to_document(doc, plot_path, title_text, description):
    """Add a plot image with title and description to a Word document."""
    heading = doc.add_heading(title_text, level=2)
    for run in heading.runs:
        run.font.color.rgb = RGBColor(0, 70, 127)

    if plot_path.exists():
        doc.add_picture(str(plot_path), width=Inches(6.0))
        last_para = doc.paragraphs[-1]
        last_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    else:
        p = doc.add_paragraph()
        p.add_run(f"[Plot not found: {plot_path.name}]").italic = True

    desc_para = doc.add_paragraph()
    desc_run  = desc_para.add_run(description)
    desc_run.font.size = Pt(10)
    desc_para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    doc.add_paragraph()


def generate_word_report(df, output_dir, config):
    """Generate a Word document containing all plots with Italian descriptions."""
    print("\n📄 Generating Word report...")

    doc = Document()

    # Title page
    title = doc.add_heading('Analisi Distribuzioni e Scatter Plot', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    subtitle = doc.add_paragraph('HVAC System Data Exploration — Dati Interpolati')
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER

    building_id = config.get('global', 'building_id')
    ahu_unit    = config.get('global', 'ahu_unit')
    season      = config.get('global', 'season')
    year        = config.get('global', 'year')
    dataset_name = f"{building_id}_{ahu_unit}_{season}{year}"

    meta = doc.add_paragraph(
        f"Dataset: {dataset_name}\n"
        f"Source: interpolation folder\n"
        f"Period: {df['Time'].min().strftime('%d/%m/%Y')} – {df['Time'].max().strftime('%d/%m/%Y')}\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_page_break()

    # ── Distribution plots ──
    section_heading = doc.add_heading('PARTE 1: DISTRIBUZIONI DELLE VARIABILI', level=1)
    for run in section_heading.runs:
        run.font.color.rgb = RGBColor(0, 102, 204)

    dist_plots = [
        ('dist_TM.png',       'Distribuzione: Temperatura Mandata',
         'La distribuzione della temperatura di mandata rivela i pattern operativi del sistema HVAC. '
         'Una distribuzione bimodale può indicare due modalità operative distinte (riscaldamento/raffreddamento).'),
        ('dist_TR.png',       'Distribuzione: Temperatura Ripresa',
         'La temperatura di ripresa rappresenta le condizioni interne dell\'edificio. '
         'Valori anomali possono indicare malfunzionamenti o condizioni operative non standard.'),
        ('dist_T_Sat.png',    'Distribuzione: Temperatura Saturazione',
         'La temperatura di saturazione è critica per il controllo dell\'umidità. '
         'Una distribuzione stretta indica controllo preciso.'),
        ('dist_SP_TM.png',    'Distribuzione: Setpoint Temperatura Mandata',
         'I setpoint mostrano tipicamente una distribuzione discreta con pochi valori dominanti, '
         'riflettendo le strategie di controllo programmate.'),
        ('dist_Fan_M.png',    'Distribuzione: Modulazione Ventilatore Mandata',
         'La modulazione del ventilatore indica l\'intensità di funzionamento. '
         'Valori bassi = standby, valori elevati = condizioni di picco.'),
        ('dist_Fan_R.png',    'Distribuzione: Modulazione Ventilatore Ripresa',
         'Il ventilatore di ripresa deve essere bilanciato con quello di mandata per mantenere '
         'la corretta pressurizzazione degli ambienti.'),
        ('dist_T_Ext.png',    'Distribuzione: Temperatura Esterna',
         'La temperatura esterna è il principale driver delle condizioni operative HVAC. '
         'La forma della distribuzione rivela pattern stagionali.'),
        ('dist_Umid_Ext.png', 'Distribuzione: Umidità Esterna',
         'L\'umidità relativa esterna influenza il carico latente del sistema. '
         'Valori elevati aumentano il consumo energetico.'),
    ]

    for i, (filename, title_text, description) in enumerate(dist_plots, 1):
        plot_path = output_dir / filename
        print(f"   {i}/8: {filename}")
        add_plot_to_document(doc, plot_path, title_text, description)
        if i < len(dist_plots):
            doc.add_page_break()

    doc.add_page_break()

    # ── Scatter plots ──
    section_heading = doc.add_heading('PARTE 2: ANALISI SCATTER PLOT', level=1)
    for run in section_heading.runs:
        run.font.color.rgb = RGBColor(204, 102, 0)

    scatter_plots = [
        ('scatter_1_TM_vs_SP_TM.png',
         'Control Tracking: Temperatura Mandata vs Setpoint',
         'Confronto tra temperatura di mandata effettiva e setpoint programmato. '
         'Punti vicini alla diagonale = controllo efficace. '
         'La banda di tolleranza (±2°C) definisce l\'accuratezza accettabile.'),
        ('scatter_2_TR_vs_TM.png',
         'Relazione Termica: Temperatura Ripresa vs Mandata',
         'In riscaldamento TM > TR, in raffreddamento TM < TR. '
         'La correlazione indica la coerenza del sistema.'),
        ('scatter_3_OutdoorT_vs_TM.png',
         'Influenza Stagionale: Temperatura Esterna vs Mandata',
         'Temperature esterne basse richiedono temperature di mandata elevate. '
         'Fondamentale per comprendere l\'adattamento alle condizioni climatiche.'),
        ('scatter_4_FanM_vs_FanR.png',
         'Bilanciamento Sistema: Ventilatore Mandata vs Ripresa',
         'In sistemi bilanciati, i punti seguono la diagonale. '
         'La banda di tolleranza (±5%) definisce il range operativo accettabile.'),
        ('scatter_5_FanM_vs_ThermalDiff.png',
         'Sforzo Ventilatore vs Effetto Termico',
         'Valori positivi = riscaldamento, valori negativi = raffreddamento. '
         'Utile per identificare anomalie operative e valutare l\'efficienza del trasferimento termico.'),
        ('scatter_6_OutdoorT_vs_OutdoorRH.png',
         'Validazione Sensori Meteo: Temperatura vs Umidità',
         'Relazione fisica attesa: aria calda = RH più bassa. '
         'Outliers o pattern anomali indicano problemi di calibrazione.'),
    ]

    for i, (filename, title_text, description) in enumerate(scatter_plots, 1):
        plot_path = output_dir / filename
        print(f"   {i}/6: {filename}")
        add_plot_to_document(doc, plot_path, title_text, description)
        if i < len(scatter_plots):
            doc.add_page_break()

    # Save document
    project_root = Path(__file__).parent.parent.parent
    report_dir   = project_root / 'processed_data' / 'distributions_and_scatter' / output_dir.name
    report_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"{output_dir.name}_Analisi_Distribuzioni_e_Scatter.docx"
    output_path     = report_dir / output_filename

    print(f"\n💾 Saving Word document...")
    doc.save(str(output_path))
    print(f"   ✅ Saved: {output_filename}")
    print(f"   📁 Location: {report_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("SCATTER PLOTS AND DISTRIBUTIONS - HVAC Data Exploration (INTERPOLATED)")
    print("=" * 80)

    # Load config
    script_dir   = Path(__file__).parent
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

    # ── KEY CHANGE: load from interpolation folder ──
    interp_file, base_name = find_interpolated_data(config)
    df = load_data(interp_file)

    # Setup output directory (tagged with _interpolated)
    project_root = Path(__file__).parent.parent.parent
    dataset_name = f"{base_name}_interpolated"
    output_dir   = project_root / 'plots' / 'distributions_and_scatter' / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Output directory: {output_dir}")

    column_mapping = get_column_mapping()

    print("\n" + "=" * 80)
    print("PART 1: DISTRIBUTION ANALYSIS")
    print("=" * 80)
    create_distribution_plots(df, output_dir, column_mapping)

    print("\n" + "=" * 80)
    print("PART 2: SCATTER PLOT ANALYSIS")
    print("=" * 80)
    generate_scatter_plots(df, output_dir, column_mapping)

    create_summary_report(df, output_dir, column_mapping)
    generate_word_report(df, output_dir, config)

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"📊 Plots saved in: {output_dir}")
    print(f"📋 Summary report: analysis_summary_interpolated.txt")
    print(f"📄 Word report:    {dataset_name}_Analisi_Distribuzioni_e_Scatter.docx")
    print("\n📈 Generated outputs:")
    print("   • 8 distribution plots (histogram + box plot)")
    print("   • 6 scatter plots (feature-pair relationships)")
    print("   • 1 comprehensive analysis summary (TXT)")
    print("   • 1 Word document with all plots and Italian descriptions")
    print("=" * 80)


if __name__ == "__main__":
    main()
