"""
energy_manager_report.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generatore di Report per il Responsabile dell'Energia — Campus Luigi Einaudi, UniTO
Edificio C1 · UTA1 · Estate 2025 (9 luglio – 14 ottobre)

Produce:
  GRAFICI (14 plots salvati come PNG 300 dpi)
    Plot  1 — Condizioni esterne (Temperatura + Umidità)
    Plot  2 — Modulazione ventilatori (intero periodo)
    Plot  3 — Ore di funzionamento giornaliere
    Plot  4 — Temperatura Mandata vs Setpoint (media giornaliera)
    Plot  5 — Zoom settimana "normale" (raw 30 min)
    Plot  6 — Zoom settimana "critica" (raw 30 min)
    Plot  7 — Errore dal setpoint (media giornaliera |errore|)
    Plot  8 — Distribuzione dell'errore (istogramma, solo AHU ON)
    Plot  9 — Profilo medio giornaliero Feriale vs Weekend
    Plot 10 — Heatmap Errore di controllo (giorno × ora)
    Plot 11 — Heatmap Modulazione Mandata (giorno × ora)
    Plot 12 — Scatter: T. Esterna vs T. Mandata (AHU ON)
    Plot 13 — Scatter: Modulazione vs |Errore| (AHU ON)
    Plot 14 — Scatter: T. Ripresa vs T. Mandata (AHU ON)

  TABELLE (3 CSV + preview a schermo)
    Tabella 1 — Dizionario dati (colonna, descrizione, unità)
    Tabella 2 — KPI settimanali (ore ON, MAE, P95, modulazione media)
    Tabella 3 — Top 10 giorni peggiori (max errore, ore ON, T. esterna)

Come usare:
    cd scripts/6_energy_manager_report
    python energy_manager_report.py

Output:
    plots/energy_manager_report/C1_UTA1_Summer2025/
    processed_data/energy_manager_report/C1_UTA1_Summer2025/
"""

# ─── Imports ──────────────────────────────────────────────────────────────────
import configparser
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns

try:
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

warnings.filterwarnings('ignore')

# ─── Paths & Config ───────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent          # HVAC_project/

config = configparser.ConfigParser()
config.read(PROJECT_ROOT / 'config.ini', encoding='utf-8-sig')

BUILDING  = config.get('global', 'building_id', fallback='C1')
AHU       = config.get('global', 'ahu_unit',    fallback='UTA1')
SEASON    = config.get('global', 'season',      fallback='Summer')
YEAR      = config.get('global', 'year',        fallback='2025')
FOLDER_ID = f"{BUILDING}_{AHU}_{SEASON}{YEAR}"

# Input: use the interpolated (clean) CSV produced by script 4 / 5
interp_dir  = PROJECT_ROOT / 'processed_data' / 'interpolation'
csv_path    = interp_dir / FOLDER_ID / f"{FOLDER_ID}_interpolated.csv"

# Outputs
out_plots  = PROJECT_ROOT / 'plots'        / 'energy_manager_report' / FOLDER_ID
out_tables = PROJECT_ROOT / 'processed_data' / 'energy_manager_report' / FOLDER_ID
out_plots.mkdir(parents=True, exist_ok=True)
out_tables.mkdir(parents=True, exist_ok=True)

# ─── Column names (Italian, from config) ──────────────────────────────────────
COL_TIME   = 'Time'
COL_TM     = 'Temperatura Mandata'        # supply temperature  °C
COL_TR     = 'Temperatura Ripresa'        # return temperature  °C
COL_TS     = 'Temperatura Saturazione'    # coil saturation     °C
COL_SP_M   = 'Set Points Temperatura Mandata'      # setpoint 1
COL_SP_C   = 'Set Points Temperatura Compensata'   # setpoint 2 (compensated)
COL_TEXT   = 'Temperatura Esterna'        # outdoor temperature
COL_UMH    = 'Umidita Esterna'            # outdoor humidity
COL_MOD_M  = 'Modulazione Ventilatore Mandata'     # supply fan  %
COL_MOD_R  = 'Modulazione Ventilatore Ripresa'     # return fan  %

ON_THRESHOLD = 5.0     # % — AHU considered ON when Mandata fan > this value

# ─── Plot style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        11,
    'axes.titlesize':   13,
    'axes.labelsize':   11,
    'axes.titleweight': 'bold',
    'figure.dpi':       100,
    'axes.grid':        True,
    'grid.alpha':       0.35,
    'lines.linewidth':  1.4,
})

PALETTE = {
    'mandata'  : '#d62728',   # red
    'ripresa'  : '#1f77b4',   # blue
    'satura'   : '#9467bd',   # purple
    'sp_m'     : '#2ca02c',   # green  (setpoint linea continua)
    'sp_c'     : '#ff7f0e',   # orange (setpoint compensato)
    'text_temp': '#8c564b',   # brown  (outdoor)
    'humidity' : '#17becf',   # cyan
    'mod_m'    : '#e377c2',   # pink
    'mod_r'    : '#7f7f7f',   # grey
    'error'    : '#d62728',   # red
    'on_shade' : '#d0f0d0',   # light green
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data() -> pd.DataFrame:
    """Load the interpolated CSV and return a clean DataFrame with DatetimeIndex."""
    print(f"\n📂 Caricamento dati: {csv_path}")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"\n❌ File non trovato: {csv_path}\n"
            f"   Assicurati di aver eseguito prima lo script di interpolazione (4/5).\n"
            f"   Percorso atteso: {csv_path}"
        )

    df = pd.read_csv(csv_path, encoding='utf-8-sig', low_memory=False)

    # Parse timestamp — ISO format (YYYY-MM-DD HH:MM:SS+TZ), dayfirst=False is critical
    # to avoid month/day swap on ambiguous dates (e.g. 2025-08-01 → 2025-01-08 with dayfirst=True)
    time_col = COL_TIME if COL_TIME in df.columns else df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], dayfirst=False, utc=False, errors='coerce')
    # Strip timezone → naive local timestamps so slicing/resampling works cleanly
    if hasattr(df[time_col], 'dt') and df[time_col].dt.tz is not None:
        df[time_col] = df[time_col].dt.tz_localize(None)
    df = df.dropna(subset=[time_col]).set_index(time_col).sort_index()
    df.index.name = 'Timestamp'

    # Numeric conversion (handles "," decimal separator and any residual object dtypes)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(',', '.', regex=False), errors='coerce'
            )
        else:
            # Force proper float dtype for any non-standard numeric columns
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # AHU ON/OFF flag
    if COL_MOD_M in df.columns:
        df['AHU_ON'] = df[COL_MOD_M] > ON_THRESHOLD
    else:
        df['AHU_ON'] = True    # fallback: assume always on

    # Derived columns
    if COL_TM in df.columns and COL_SP_M in df.columns:
        df['Errore_SP_Mandata']    = df[COL_TM] - df[COL_SP_M]
        df['Errore_SP_Compensata'] = df[COL_TM] - df[COL_SP_C] if COL_SP_C in df.columns else np.nan
        df['Abs_Errore']           = df['Errore_SP_Mandata'].abs()

    # Final: ensure ALL numeric columns are strict float64 so matplotlib never chokes
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = df[num_cols].astype('float64')

    print(f"   ✓  Righe totali: {len(df):,}")
    print(f"   ✓  Periodo:      {df.index[0].date()} → {df.index[-1].date()}")
    print(f"   ✓  AHU ON:       {df['AHU_ON'].sum():,} campioni "
          f"({df['AHU_ON'].mean()*100:.1f}% del periodo)")
    return df


def _week_row_count(df: pd.DataFrame, monday: pd.Timestamp) -> int:
    """Number of raw rows in the 7-day window starting on `monday`."""
    return len(df.loc[monday: monday + pd.Timedelta(days=7)])


def _good_week(df: pd.DataFrame, monday: pd.Timestamp, min_rows: int) -> bool:
    """True if the week has enough data AND is not too close to dataset boundaries."""
    data_end = df.index[-1]
    data_start = df.index[0]
    # Must start at least 7 days from either end to have a full week available
    if monday < data_start + pd.Timedelta(days=7):
        return False
    if monday + pd.Timedelta(days=7) > data_end:
        return False
    return _week_row_count(df, monday) >= min_rows


def choose_critical_week(df: pd.DataFrame) -> pd.Timestamp:
    """Return Monday of the week with highest mean absolute error,
    requiring ≥5 days of actual data and not at dataset boundaries."""
    samples_per_day = max(1, round(len(df) / max(1, (df.index[-1] - df.index[0]).days)))
    MIN_ROWS = samples_per_day * 5  # ≥ 5 days of data

    weekly = df[df['AHU_ON']]['Abs_Errore'].resample('W-MON').mean().dropna()
    for ts in weekly.sort_values(ascending=False).index:
        monday = ts - pd.Timedelta(days=ts.weekday())
        if _good_week(df, monday, MIN_ROWS):
            return monday
    # Fallback: relax to 1 day minimum
    for ts in weekly.sort_values(ascending=False).index:
        monday = ts - pd.Timedelta(days=ts.weekday())
        if _week_row_count(df, monday) >= samples_per_day:
            return monday
    return weekly.index[0] - pd.Timedelta(days=weekly.index[0].weekday())


def choose_normal_week(df: pd.DataFrame, critical_start: pd.Timestamp) -> pd.Timestamp:
    """Return Monday of the week with lowest mean absolute error,
    ≥14 days from critical week, not at boundaries, with ≥5 days of data."""
    samples_per_day = max(1, round(len(df) / max(1, (df.index[-1] - df.index[0]).days)))
    MIN_ROWS = samples_per_day * 5  # ≥ 5 days of data

    weekly = df[df['AHU_ON']]['Abs_Errore'].resample('W-MON').mean().dropna()
    for ts in weekly.sort_values(ascending=True).index:
        monday = ts - pd.Timedelta(days=ts.weekday())
        if abs((monday - critical_start).days) > 14 and _good_week(df, monday, MIN_ROWS):
            return monday
    # Relax distance to 7 days
    for ts in weekly.sort_values(ascending=True).index:
        monday = ts - pd.Timedelta(days=ts.weekday())
        if monday != critical_start and _good_week(df, monday, MIN_ROWS):
            return monday
    # Last resort: relax to 1 day minimum, any week with data
    for ts in weekly.sort_values(ascending=True).index:
        monday = ts - pd.Timedelta(days=ts.weekday())
        if monday != critical_start and _week_row_count(df, monday) >= samples_per_day:
            return monday
    return weekly.index[0] - pd.Timedelta(days=weekly.index[0].weekday())


# =============================================================================
# HELPER UTILITIES
# =============================================================================

def _to_f64(s) -> np.ndarray:
    """Nuclear float64 conversion: handles pandas nullable Float64/Int64,
    masked arrays, object dtype, pd.NA, None, etc."""
    vals = s.to_list() if hasattr(s, 'to_list') else list(s)
    result = []
    for v in vals:
        try:
            if v is pd.NA or v is None:
                result.append(np.nan)
            else:
                result.append(float(v))
        except (TypeError, ValueError):
            result.append(np.nan)
    return np.array(result, dtype=np.float64)


def _save(fig: plt.Figure, name: str) -> None:
    path = out_plots / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   💾 Salvato: {path.name}")


def _date_fmt(ax, freq: str = 'week') -> None:
    """Apply readable date formatting to x-axis."""
    if freq == 'week':
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    elif freq == 'day':
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=9)


def _add_weekend_bands(ax, df: pd.DataFrame) -> None:
    """Shade Saturday–Sunday with light grey."""
    dates = pd.date_range(df.index[0], df.index[-1], freq='D')
    for d in dates:
        if d.weekday() >= 5:   # Sat=5, Sun=6
            ax.axvspan(d, d + pd.Timedelta(days=1), color='#cccccc', alpha=0.25, linewidth=0)


# =============================================================================
# PLOTS
# =============================================================================

def plot01_outdoor(df: pd.DataFrame) -> None:
    """Plot 1 — Condizioni esterne: Temperatura e Umidità."""
    fig, ax1 = plt.subplots(figsize=(14, 4))
    ax1.set_title(
        "Grafico 1 — Condizioni Esterne durante il Periodo di Analisi\n"
        f"Campus Luigi Einaudi · {BUILDING} · {AHU} · Estate {YEAR}",
    )

    if COL_TEXT in df.columns:
        ax1.plot(df.index, df[COL_TEXT], color=PALETTE['text_temp'], label='Temperatura Esterna (°C)')
        ax1.set_ylabel('Temperatura Esterna (°C)', color=PALETTE['text_temp'])
        ax1.tick_params(axis='y', labelcolor=PALETTE['text_temp'])

    if COL_UMH in df.columns:
        ax2 = ax1.twinx()
        ax2.fill_between(df.index, _to_f64(df[COL_UMH]), alpha=0.18, color=PALETTE['humidity'])
        ax2.plot(df.index, df[COL_UMH], color=PALETTE['humidity'], alpha=0.7,
                 linewidth=0.8, label='Umidità Relativa (%)')
        ax2.set_ylabel('Umidità Relativa (%)', color=PALETTE['humidity'])
        ax2.tick_params(axis='y', labelcolor=PALETTE['humidity'])
        ax2.set_ylim(0, 110)

    _add_weekend_bands(ax1, df)
    _date_fmt(ax1)
    ax1.set_xlabel('')
    lines1, labels1 = ax1.get_legend_handles_labels()
    if COL_UMH in df.columns:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
    else:
        ax1.legend(loc='upper right', fontsize=9)

    fig.tight_layout()
    _save(fig, 'plot01_condizioni_esterne')


def plot02_fan_modulation(df: pd.DataFrame) -> None:
    """Plot 2 — Modulazione ventilatori (intero periodo)."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_title(
        "Grafico 2 — Modulazione Ventilatori (Intero Periodo)\n"
        f"Edificio C1 · UTA1 · Estate {YEAR}",
    )

    if COL_MOD_M in df.columns:
        ax.fill_between(df.index, _to_f64(df[COL_MOD_M]), alpha=0.3, color=PALETTE['mod_m'])
        ax.plot(df.index, df[COL_MOD_M], color=PALETTE['mod_m'],
                linewidth=0.7, label='Ventilatore Mandata (%)')
    if COL_MOD_R in df.columns:
        ax.plot(df.index, df[COL_MOD_R], color=PALETTE['mod_r'],
                linewidth=0.7, alpha=0.7, label='Ventilatore Ripresa (%)')

    ax.axhline(ON_THRESHOLD, color='black', linestyle='--', linewidth=1,
               label=f'Soglia ON = {ON_THRESHOLD:.0f}%')
    ax.set_ylim(-2, 105)
    ax.set_ylabel('Modulazione (%)')
    _add_weekend_bands(ax, df)
    _date_fmt(ax)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, 'plot02_modulazione_ventilatori')


def plot03_daily_hours(df: pd.DataFrame) -> None:
    """Plot 3 — Ore di funzionamento giornaliere."""
    daily_on = (
        df['AHU_ON'].resample('D').sum() * 0.5  # 30-min samples → hours
    )
    daily_on.name = 'Ore_ON'
    is_weekend = daily_on.index.weekday >= 5

    colors = ['#a0c4ff' if wd else '#4e9af1' for wd in is_weekend]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_title(
        "Grafico 3 — Ore di Funzionamento Giornaliere dell'UTA\n"
        f"Azzurro chiaro = weekend/festivi · Estate {YEAR}",
    )
    ax.bar(daily_on.index, daily_on.values, color=colors, width=0.85, edgecolor='none')
    ax.axhline(daily_on.mean(), color='red', linestyle='--', linewidth=1.2,
               label=f'Media = {daily_on.mean():.1f} h/giorno')
    ax.set_ylabel('Ore di funzionamento (h)')
    ax.set_ylim(0, 25)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(4))
    _date_fmt(ax)
    ax.legend(fontsize=9)

    # Custom legend patch
    from matplotlib.patches import Patch
    handles = [
        Patch(color='#4e9af1', label='Giorno feriale'),
        Patch(color='#a0c4ff', label='Weekend'),
        ax.get_legend_handles_labels()[0][0],
    ]
    labels = ['Giorno feriale', 'Weekend', f'Media = {daily_on.mean():.1f} h/giorno']
    ax.legend(handles, labels, fontsize=9)

    fig.tight_layout()
    _save(fig, 'plot03_ore_funzionamento_giornaliere')


def plot04_mandata_vs_setpoint_daily(df: pd.DataFrame) -> None:
    """Plot 4 — T. Mandata vs Setpoints (medie giornaliere)."""
    daily = df[[COL_TM, COL_SP_M, COL_SP_C]].resample('D').median()
    daily_q25 = df[COL_TM].resample('D').quantile(0.25)
    daily_q75 = df[COL_TM].resample('D').quantile(0.75)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_title(
        "Grafico 4 — Temperatura di Mandata vs Setpoint (Mediana Giornaliera)\n"
        f"Banda = intervallo 25°–75° percentile · Estate {YEAR}",
    )

    ax.fill_between(daily.index, _to_f64(daily_q25), _to_f64(daily_q75), alpha=0.15,
                    color=PALETTE['mandata'], label='Band. 25–75% Mandata')
    ax.plot(daily.index, daily[COL_TM], color=PALETTE['mandata'],
            linewidth=1.8, label='T. Mandata (mediana giorn.)')
    if COL_SP_M in daily.columns:
        ax.plot(daily.index, daily[COL_SP_M], color=PALETTE['sp_m'],
                linestyle='--', linewidth=1.4, label='SP Mandata')
    if COL_SP_C in daily.columns:
        ax.plot(daily.index, daily[COL_SP_C], color=PALETTE['sp_c'],
                linestyle=':', linewidth=1.4, label='SP Compensata')

    ax.set_ylabel('Temperatura (°C)')
    _add_weekend_bands(ax, df)
    _date_fmt(ax)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, 'plot04_mandata_vs_setpoint_giornaliero')


def _plot_week_zoom(df: pd.DataFrame, week_start: pd.Timestamp, label: str, tag: str) -> None:
    """Shared function for Plot 5 & 6 (raw 30-min, 7 days)."""
    end = week_start + pd.Timedelta(days=7)
    sub = df.loc[week_start:end].copy()

    if sub.empty:
        print(f"   ⚠️  Nessun dato nella settimana {week_start.date()} — plot saltato")
        return

    # Ensure all columns are proper float64 for matplotlib
    for _c in sub.columns:
        if _c != 'AHU_ON':
            sub[_c] = pd.to_numeric(sub[_c], errors='coerce').astype(float)

    # Pre-convert index to numpy datetime64 so matplotlib's unit converter
    # always receives a type it can handle, even for very sparse slices
    x = sub.index.to_numpy().astype('datetime64[ns]')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle(
        f"Grafico {tag} — Settimana {label}: "
        f"{week_start.strftime('%d %b')} – {(week_start + pd.Timedelta(days=6)).strftime('%d %b %Y')} (dati ogni 30 min)",
        fontweight='bold', fontsize=12,
    )

    # Top panel: temperatures
    ax1.plot(x, _to_f64(sub[COL_TM]), color=PALETTE['mandata'],
             linewidth=1.2, label='T. Mandata')
    if COL_SP_M in sub.columns:
        ax1.plot(x, _to_f64(sub[COL_SP_M]), color=PALETTE['sp_m'],
                 linestyle='--', linewidth=1.2, label='SP Mandata')
    if COL_SP_C in sub.columns:
        ax1.plot(x, _to_f64(sub[COL_SP_C]), color=PALETTE['sp_c'],
                 linestyle=':', linewidth=1.2, label='SP Compensata')
    if COL_TR in sub.columns:
        ax1.plot(x, _to_f64(sub[COL_TR]), color=PALETTE['ripresa'],
                 linewidth=0.9, alpha=0.7, label='T. Ripresa')
    ax1.set_ylabel('Temperatura (°C)')
    ax1.legend(fontsize=8, loc='lower right')

    # Bottom panel: fan modulation
    if COL_MOD_M in sub.columns:
        y_mod = _to_f64(sub[COL_MOD_M])
        ax2.fill_between(x, y_mod, alpha=0.4, color=PALETTE['mod_m'])
        ax2.plot(x, y_mod, color=PALETTE['mod_m'],
                 linewidth=0.9, label='Modulaz. Mandata (%)')
    ax2.set_ylabel('Modulaz. (%)')
    ax2.set_ylim(-2, 105)
    ax2.legend(fontsize=8)

    _date_fmt(ax2, freq='day')
    _add_weekend_bands(ax1, sub)
    _add_weekend_bands(ax2, sub)

    fig.tight_layout()
    fname = 'plot05_settimana_normale' if '5' in tag else 'plot06_settimana_critica'
    _save(fig, fname)


def plot05_normal_week(df: pd.DataFrame, week_start: pd.Timestamp) -> None:
    _plot_week_zoom(df, week_start, label='Normale', tag='5')


def plot06_critical_week(df: pd.DataFrame, week_start: pd.Timestamp) -> None:
    _plot_week_zoom(df, week_start, label='Critica', tag='6')


def plot07_daily_error(df: pd.DataFrame) -> None:
    """Plot 7 — Errore medio giornaliero assoluto dal setpoint."""
    on_data = df[df['AHU_ON']].copy()
    daily_ae = on_data['Abs_Errore'].resample('D').mean()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_title(
        f"Grafico 7 — Errore Medio Giornaliero |T.Mandata – SP.Mandata| (solo AHU ON)\n"
        f"Estate {YEAR}",
    )
    ax.bar(daily_ae.index, daily_ae.values, color=PALETTE['error'],
           alpha=0.7, width=0.85, edgecolor='none')
    ax.axhline(daily_ae.mean(), color='black', linestyle='--', linewidth=1.2,
               label=f'Media stagionale = {daily_ae.mean():.2f} °C')
    ax.axhline(1.0, color='orange', linestyle=':', linewidth=1.2,
               label='Soglia accettabile = 1.0 °C')
    ax.set_ylabel('Errore medio assoluto (°C)')
    _add_weekend_bands(ax, df)
    _date_fmt(ax)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, 'plot07_errore_giornaliero')


def plot08_error_distribution(df: pd.DataFrame) -> None:
    """Plot 8 — Istogramma dell'errore di controllo (AHU ON)."""
    on_data = df[df['AHU_ON']]['Errore_SP_Mandata'].dropna()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title(
        f"Grafico 8 — Distribuzione dell'Errore di Controllo\n"
        f"(T.Mandata – SP.Mandata, solo AHU ON) · Estate {YEAR}",
    )
    ax.hist(on_data, bins=80, color=PALETTE['error'], alpha=0.75, edgecolor='white',
            linewidth=0.3)
    ax.axvline(0, color='black', linewidth=1.4, linestyle='--', label='Errore zero')
    ax.axvline(on_data.mean(), color='orange', linewidth=1.4,
               label=f'Media = {on_data.mean():.2f} °C')
    pct_within_1 = ((on_data.abs() <= 1.0).sum() / len(on_data)) * 100
    ax.set_xlabel('Errore (°C)')
    ax.set_ylabel('Numero di campioni')
    ax.legend(fontsize=9)
    ax.text(0.98, 0.95, f'{pct_within_1:.1f}% dei campioni\nentro ±1 °C',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='grey'))
    fig.tight_layout()
    _save(fig, 'plot08_distribuzione_errore')


def plot09_weekday_vs_weekend(df: pd.DataFrame) -> None:
    """Plot 9 — Profilo medio 24h: Feriale vs Weekend."""
    df2 = df.copy()
    df2['ora'] = df2.index.hour + df2.index.minute / 60
    df2['tipo'] = df2.index.weekday.map(lambda d: 'Weekend' if d >= 5 else 'Feriale')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), sharey=False)
    fig.suptitle(
        f"Grafico 9 — Profilo Medio Giornaliero: Feriale vs Weekend · Estate {YEAR}",
        fontweight='bold', fontsize=12,
    )

    # T. Mandata vs SP
    for tipo, color, ls in [('Feriale', '#d62728', '-'), ('Weekend', '#1f77b4', '--')]:
        sub = df2[df2['tipo'] == tipo]
        avg_tm = sub.groupby('ora', observed=True)[COL_TM].mean()
        avg_sp = sub.groupby('ora', observed=True)[COL_SP_M].mean()
        ax1.plot(avg_tm.index, avg_tm.values, color=color, ls=ls, label=f'T.Mandata {tipo}')
        ax1.plot(avg_sp.index, avg_sp.values, color=color, ls=':', alpha=0.6,
                 label=f'SP {tipo}')
    ax1.set_title('Temperatura Mandata vs Setpoint')
    ax1.set_xlabel('Ora del giorno')
    ax1.set_ylabel('Temperatura (°C)')
    ax1.set_xticks(range(0, 25, 3))
    ax1.legend(fontsize=8)

    # Fan modulation
    for tipo, color, ls in [('Feriale', PALETTE['mod_m'], '-'), ('Weekend', PALETTE['mod_r'], '--')]:
        sub = df2[df2['tipo'] == tipo]
        avg_mod = sub.groupby('ora', observed=True)[COL_MOD_M].mean()
        ax2.fill_between(avg_mod.index, _to_f64(avg_mod), alpha=0.2, color=color)
        ax2.plot(avg_mod.index, avg_mod.values, color=color, ls=ls,
                 label=f'Modulaz. Mandata — {tipo}')
    ax2.set_title('Modulazione Ventilatore Mandata')
    ax2.set_xlabel('Ora del giorno')
    ax2.set_ylabel('Modulazione (%)')
    ax2.set_ylim(-2, 105)
    ax2.set_xticks(range(0, 25, 3))
    ax2.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, 'plot09_profilo_feriale_vs_weekend')


def plot10_heatmap_error(df: pd.DataFrame) -> None:
    """Plot 10 — Heatmap errore di controllo (giorno × ora)."""
    df2 = df[df['AHU_ON']].copy()
    df2['data'] = df2.index.normalize()
    df2['ora']  = df2.index.hour

    pivot = df2.pivot_table(
        index='ora', columns='data', values='Errore_SP_Mandata',
        aggfunc='mean'
    )
    pivot.columns = [c.strftime('%d %b') for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(max(14, len(pivot.columns) * 0.18), 6))
    ax.set_title(
        f"Grafico 10 — Heatmap Errore di Controllo (T.Mandata – SP) per ora e giorno\n"
        f"Blu = AHU troppo fredda  Rosso = AHU troppo calda · Estate {YEAR}",
    )
    vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 0.1)
    sns.heatmap(
        pivot, ax=ax, cmap='RdBu_r', center=0, vmin=-vmax, vmax=vmax,
        linewidths=0, cbar_kws={'label': 'Errore (°C)', 'shrink': 0.7},
        xticklabels=max(1, len(pivot.columns) // 20),
    )
    ax.set_xlabel('Data')
    ax.set_ylabel('Ora del giorno')
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, 'plot10_heatmap_errore')


def plot11_heatmap_modulation(df: pd.DataFrame) -> None:
    """Plot 11 — Heatmap modulazione Mandata (giorno × ora)."""
    df2 = df.copy()
    df2['data'] = df2.index.normalize()
    df2['ora']  = df2.index.hour

    pivot = df2.pivot_table(
        index='ora', columns='data', values=COL_MOD_M,
        aggfunc='mean'
    )
    pivot.columns = [c.strftime('%d %b') for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(max(14, len(pivot.columns) * 0.18), 6))
    ax.set_title(
        f"Grafico 11 — Heatmap Modulazione Ventilatore Mandata (%) per ora e giorno\n"
        f"Bianco = OFF · Viola scuro = funzionamento al massimo · Estate {YEAR}",
    )
    sns.heatmap(
        pivot, ax=ax, cmap='Purples', vmin=0, vmax=100,
        linewidths=0, cbar_kws={'label': 'Modulazione (%)', 'shrink': 0.7},
        xticklabels=max(1, len(pivot.columns) // 20),
    )
    ax.set_xlabel('Data')
    ax.set_ylabel('Ora del giorno')
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, 'plot11_heatmap_modulazione')


def plot12_scatter_text_vs_mandata(df: pd.DataFrame) -> None:
    """Plot 12 — Scatter: T. Esterna vs T. Mandata (AHU ON)."""
    on = df[df['AHU_ON']]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_title(
        f"Grafico 12 — T. Esterna vs T. Mandata\n(solo AHU ON) · Estate {YEAR}",
    )
    sc = ax.scatter(on[COL_TEXT], on[COL_TM], alpha=0.06, s=3,
                    c=on.index.dayofyear, cmap='plasma')
    plt.colorbar(sc, ax=ax, label='Giorno dell\'anno')

    # Trend line
    valid = on[[COL_TEXT, COL_TM]].dropna()
    if len(valid) > 10:
        z = np.polyfit(valid[COL_TEXT], valid[COL_TM], 1)
        p = np.poly1d(z)
        xs = np.linspace(valid[COL_TEXT].min(), valid[COL_TEXT].max(), 100)
        ax.plot(xs, p(xs), 'r--', linewidth=1.5, label=f'Tendenza (slope={z[0]:.2f})')
        ax.legend(fontsize=9)

    ax.set_xlabel('Temperatura Esterna (°C)')
    ax.set_ylabel('Temperatura Mandata (°C)')
    fig.tight_layout()
    _save(fig, 'plot12_scatter_text_mandata')


def plot13_scatter_mod_vs_error(df: pd.DataFrame) -> None:
    """Plot 13 — Scatter: Modulazione Mandata vs |Errore|."""
    on = df[df['AHU_ON']].dropna(subset=[COL_MOD_M, 'Abs_Errore'])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_title(
        f"Grafico 13 — Modulazione vs |Errore di Controllo|\n(solo AHU ON) · Estate {YEAR}",
    )
    ax.scatter(on[COL_MOD_M], on['Abs_Errore'], alpha=0.05, s=3,
               color=PALETTE['mod_m'])
    ax.set_xlabel('Modulazione Ventilatore Mandata (%)')
    ax.set_ylabel('|Errore| (°C)')
    ax.set_xlim(-2, 102)

    # Binned trend
    bins = pd.cut(on[COL_MOD_M], bins=20)
    trend = on.groupby(bins, observed=True)['Abs_Errore'].median()
    bin_centers = [iv.mid for iv in trend.index]
    ax.plot(bin_centers, trend.values, 'r-o', markersize=4, linewidth=1.5,
            label='Mediana per bin')
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, 'plot13_scatter_modulazione_errore')


def plot14_scatter_ripresa_vs_mandata(df: pd.DataFrame) -> None:
    """Plot 14 — Scatter: T. Ripresa vs T. Mandata."""
    on = df[df['AHU_ON']].dropna(subset=[COL_TR, COL_TM])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_title(
        f"Grafico 14 — T. Ripresa vs T. Mandata\n(solo AHU ON) · Estate {YEAR}",
    )
    sc = ax.scatter(on[COL_TR], on[COL_TM], alpha=0.06, s=3,
                    c=on.index.dayofyear, cmap='viridis')
    plt.colorbar(sc, ax=ax, label='Giorno dell\'anno')

    ax.set_xlabel('Temperatura Ripresa (°C)')
    ax.set_ylabel('Temperatura Mandata (°C)')

    if COL_TS in on.columns:
        ax2 = None   # Saturazione shown as note only
        sat_mean = on[COL_TS].mean()
        ax.axhline(sat_mean, color=PALETTE['satura'], linestyle='--',
                   linewidth=1.2, label=f'T.Saturazione media = {sat_mean:.1f} °C')
        ax.legend(fontsize=9)

    fig.tight_layout()
    _save(fig, 'plot14_scatter_ripresa_mandata')


# =============================================================================
# TABLES
# =============================================================================

def table01_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """Tabella 1 — Dizionario dati."""
    rows = [
        (COL_TM,   'Temperatura dell\'aria di mandata (uscita UTA verso le zone)',              '°C'),
        (COL_TR,   'Temperatura dell\'aria di ripresa (ritorno dalle zone all\'UTA)',            '°C'),
        (COL_TS,   'Temperatura di saturazione della batteria di scambio',                      '°C'),
        (COL_SP_M, 'Setpoint di temperatura di mandata impostato dall\'operatore',              '°C'),
        (COL_SP_C, 'Setpoint compensato calcolato in base alla temperatura esterna',            '°C'),
        (COL_TEXT, 'Temperatura esterna (stazione meteo)',                                      '°C'),
        (COL_UMH,  'Umidità relativa esterna (stazione meteo)',                                 '%'),
        (COL_MOD_M,'Segnale di modulazione del ventilatore di mandata (0=OFF, 100=massimo)',    '%'),
        (COL_MOD_R,'Segnale di modulazione del ventilatore di ripresa',                         '%'),
    ]
    # Filter to columns actually present
    rows_present = [(c, d, u) for c, d, u in rows if c in df.columns]
    t = pd.DataFrame(rows_present, columns=['Variabile', 'Descrizione', 'Unità'])
    # Add basic stats
    t['Media'] = t['Variabile'].map(
        lambda c: f"{df[c].mean():.1f}" if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) else '—'
    )
    t['Min'] = t['Variabile'].map(
        lambda c: f"{df[c].min():.1f}" if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) else '—'
    )
    t['Max'] = t['Variabile'].map(
        lambda c: f"{df[c].max():.1f}" if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) else '—'
    )
    path = out_tables / 'tabella01_dizionario_dati.csv'
    t.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"   💾 Salvata: {path.name}")
    return t


def table02_weekly_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Tabella 2 — KPI settimanali."""
    rows = []
    weeks = pd.date_range(df.index[0].normalize(), df.index[-1].normalize(), freq='W-MON')
    for w_start in weeks:
        w_end = w_start + pd.Timedelta(days=6, hours=23, minutes=30)
        sub   = df.loc[w_start:w_end]
        on    = sub[sub['AHU_ON']]
        if len(on) == 0:
            continue

        n_wkd = sum(1 for d in pd.date_range(w_start, w_end, freq='D') if d.weekday() < 5)
        n_wke = 7 - n_wkd

        on_wkd = sub[sub['AHU_ON'] & (sub.index.weekday < 5)]
        on_wke = sub[sub['AHU_ON'] & (sub.index.weekday >= 5)]

        mae_val  = on['Abs_Errore'].mean() if 'Abs_Errore' in on.columns else np.nan
        p95_val  = on['Abs_Errore'].quantile(0.95) if 'Abs_Errore' in on.columns else np.nan
        mod_mean = on[COL_MOD_M].mean() if COL_MOD_M in on.columns else np.nan
        t_ext    = sub[COL_TEXT].mean() if COL_TEXT in sub.columns else np.nan

        rows.append({
            'Settimana (lunedì)': w_start.strftime('%d %b %Y'),
            'Ore ON feriali (h)': round(len(on_wkd) * 0.5, 1),
            'Ore ON weekend (h)': round(len(on_wke) * 0.5, 1),
            'Ore ON totali (h)' : round(len(on) * 0.5, 1),
            'MAE Errore (°C)'   : round(mae_val, 2) if not np.isnan(mae_val) else '—',
            'P95 |Errore| (°C)' : round(p95_val, 2) if not np.isnan(p95_val) else '—',
            'Modulaz. media (%)': round(mod_mean, 1) if not np.isnan(mod_mean) else '—',
            'T.Esterna media (°C)': round(t_ext, 1) if not np.isnan(t_ext) else '—',
        })

    t = pd.DataFrame(rows)
    path = out_tables / 'tabella02_kpi_settimanali.csv'
    t.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"   💾 Salvata: {path.name}")
    return t


def table03_worst_days(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Tabella 3 — Top N giorni con il maggiore errore medio."""
    on = df[df['AHU_ON']].copy()
    daily = on.resample('D').agg(
        Ore_ON    = ('AHU_ON',    lambda x: round(x.sum() * 0.5, 1)),
        MAE       = ('Abs_Errore', 'mean'),
        Max_Errore= ('Abs_Errore', 'max'),
        ModMean   = (COL_MOD_M,  'mean'),
        T_Est_med = (COL_TEXT,   'mean'),
    ).dropna(subset=['MAE'])

    daily['Giorno'] = daily.index.strftime('%A, %d %b %Y')
    daily['Giorno'] = daily['Giorno'].str.replace(
        'Monday',    'Lunedì').str.replace('Tuesday',   'Martedì') \
        .str.replace('Wednesday', 'Mercoledì').str.replace('Thursday',  'Giovedì') \
        .str.replace('Friday',    'Venerdì').str.replace('Saturday',   'Sabato') \
        .str.replace('Sunday',    'Domenica')

    top = daily.nlargest(n, 'MAE')[[
        'Giorno', 'Ore_ON', 'MAE', 'Max_Errore', 'ModMean', 'T_Est_med'
    ]].round(2)
    top.columns = [
        'Giorno', 'Ore ON (h)', 'MAE Errore (°C)', 'Max |Errore| (°C)',
        'Modulaz. media (%)', 'T.Esterna media (°C)'
    ]
    top = top.reset_index(drop=True)
    top.index += 1
    path = out_tables / 'tabella03_giorni_peggiori.csv'
    top.to_csv(path, encoding='utf-8-sig')
    print(f"   💾 Salvata: {path.name}")
    return top


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("  REPORT RESPONSABILE ENERGIA — Campus Luigi Einaudi · UniTO")
    print(f"  {BUILDING} · {AHU} · {SEASON} {YEAR}")
    print("=" * 70)

    # ── Load ─────────────────────────────────────────────────────────────────
    df = load_data()

    # ── Identify representative weeks ────────────────────────────────────────
    crit_week   = choose_critical_week(df)
    normal_week = choose_normal_week(df, crit_week)
    print(f"\n   📅 Settimana normale selezionata:  {normal_week.strftime('%d %b %Y')} (errore più basso)")
    print(f"   📅 Settimana critica selezionata:  {crit_week.strftime('%d %b %Y')}  (errore più alto)")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n📊 Generazione grafici...")
    plot01_outdoor(df)
    plot02_fan_modulation(df)
    plot03_daily_hours(df)
    plot04_mandata_vs_setpoint_daily(df)
    plot05_normal_week(df, normal_week)
    plot06_critical_week(df, crit_week)
    plot07_daily_error(df)
    # plot08_error_distribution skipped — rimosso su richiesta (troppo tecnico)
    plot09_weekday_vs_weekend(df)
    plot10_heatmap_error(df)
    plot11_heatmap_modulation(df)
    plot12_scatter_text_vs_mandata(df)
    plot13_scatter_mod_vs_error(df)
    plot14_scatter_ripresa_vs_mandata(df)

    # ── Tables ────────────────────────────────────────────────────────────────
    print("\n📋 Generazione tabelle...")
    t1 = table01_data_dictionary(df)
    t2 = table02_weekly_kpis(df)
    t3 = table03_worst_days(df)

    # ── Print previews ────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("TABELLA 1 — Dizionario Dati")
    print("─" * 70)
    print(t1[['Variabile', 'Descrizione', 'Unità', 'Media']].to_string(index=False))

    print("\n" + "─" * 70)
    print("TABELLA 2 — KPI Settimanali")
    print("─" * 70)
    print(t2.to_string(index=False))

    print("\n" + "─" * 70)
    print(f"TABELLA 3 — Top {len(t3)} Giorni Peggiori")
    print("─" * 70)
    print(t3.to_string())

    # ── Summary stats ─────────────────────────────────────────────────────────
    total_on_h  = df['AHU_ON'].sum() * 0.5
    total_days  = (df.index[-1] - df.index[0]).days + 1
    avg_on_h    = total_on_h / total_days
    mae_overall = df[df['AHU_ON']]['Abs_Errore'].mean()
    pct_1deg    = (df[df['AHU_ON']]['Abs_Errore'] <= 1.0).mean() * 100
    wke_on_h    = df[df['AHU_ON'] & (df.index.weekday >= 5)].shape[0] * 0.5
    wkd_on_h    = df[df['AHU_ON'] & (df.index.weekday < 5)].shape[0] * 0.5

    print("\n" + "=" * 70)
    print("  📈 STATISTICHE RIASSUNTIVE STAGIONALI")
    print("=" * 70)
    print(f"  Periodo analizzato:          {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Giorni totali:               {total_days}")
    print(f"  Ore totali di funzionamento: {total_on_h:,.0f} h "
          f"(media {avg_on_h:.1f} h/giorno)")
    print(f"  Ore ON feriale vs weekend:   {wkd_on_h:,.0f} h feriale | "
          f"{wke_on_h:,.0f} h weekend")
    print(f"  MAE stagionale:              {mae_overall:.2f} °C"
          f"  ({pct_1deg:.1f}% degli intervalli ON entro ±1 °C)")
    print(f"\n  Output grafici:  {out_plots}")
    print(f"  Output tabelle:  {out_tables}")
    print("=" * 70 + "\n")

    # ── Word report ───────────────────────────────────────────────────────────
    docx_path = generate_word_report(
        df          = df,
        t1          = t1,
        t2          = t2,
        t3          = t3,
        normal_week = normal_week,
        crit_week   = crit_week,
    )
    if docx_path:
        print(f"\n  📄 Report Word:  {docx_path}\n")

    # ── Plot catalogue (saved into plots/ folder) ─────────────────────────────
    cat_path = generate_plot_catalogue()
    if cat_path:
        print(f"\n  📚 Catalogo Grafici: {cat_path}\n")


# =============================================================================
# WORD REPORT GENERATOR
# =============================================================================

# Italian titles & descriptions for all 14 plots
_PLOT_META = [
    ('plot01_condizioni_esterne',
     'Grafico 1 — Condizioni Meteorologiche Esterne',
    'Mostra come cambiano temperatura e umidità esterna nel periodo analizzato. '
    'Aiuta a capire se i giorni più caldi coincidono con i giorni in cui l\'impianto fatica di più.'),

    ('plot02_modulazione_ventilatori',
     'Grafico 2 — Modulazione dei Ventilatori (Intero Periodo)',
        'Mostra quanto lavorano i ventilatori di mandata e ripresa (0–100%). '
        'Serve per vedere subito periodi di spegnimento, notti, weekend e possibili comportamenti anomali.'),

    ('plot03_ore_funzionamento_giornaliere',
     'Grafico 3 — Ore di Funzionamento Giornaliere dell\u2019UTA',
     'Quante ore al giorno l\u2019UTA risulta accesa. '
     'I weekend (azzurro chiaro) dovrebbero avere meno ore dei feriali; se sono uguali, l\u2019impianto pu\u00f2 essere ottimizzato.'),

    ('plot04_mandata_vs_setpoint_giornaliero',
     'Grafico 4 — Temperatura di Mandata vs Setpoint (Mediana Giornaliera)',
     'Confronto tra la temperatura effettiva di mandata e il setpoint, giorno per giorno. '
     'Se le curve si sovrappongono, la regolazione funziona bene; se si allontanano, c\u2019\u00e8 un problema di controllo.'),

    ('plot05_settimana_normale',
     'Grafico 5 — Zoom Settimana "Normale" (dati ogni 30 minuti)',
        'È un esempio di settimana in cui l\'impianto ha lavorato bene. '
        'Usala come riferimento visivo per confrontare la settimana critica.'),

    ('plot06_settimana_critica',
     'Grafico 6 — Zoom Settimana "Critica" (dati ogni 30 minuti)',
        'Mostra la settimana con maggiori difficoltà di controllo. '
        'Confrontata con la settimana normale, evidenzia subito dove il sistema si discosta dal comportamento atteso.'),

    ('plot07_errore_giornaliero',
     'Grafico 7 — Errore Medio Giornaliero di Controllo |T.Mandata – SP|',
        'Indica giorno per giorno quanto la mandata è lontana dal setpoint. '
        'Quando la linea supera la soglia, quel giorno va discusso perché il controllo è stato debole.'),

    ('plot09_profilo_feriale_vs_weekend',
     'Grafico 9 — Profilo Medio Giornaliero: Feriale vs Weekend',
     'Come varia la temperatura ora per ora in un giorno feriale rispetto al weekend. '
     'Se i profili sono quasi identici, l’impianto non è programmato diversamente nei giorni festivi e potrebbe consumare più del dovuto.'),

    ('plot10_heatmap_errore',
     'Grafico 10 — Heatmap Errore di Controllo (Ora × Giorno)',
        'Mappa rapida per capire in quali ore e in quali giorni il controllo è stato peggiore. '
        'Rosso = mandata troppo alta rispetto al setpoint; blu = mandata troppo bassa.'),

    ('plot11_heatmap_modulazione',
     'Grafico 11 — Heatmap Modulazione Ventilatore Mandata (Ora × Giorno)',
        'Mostra a colpo d\'occhio quando il ventilatore è spento o vicino al massimo. '
        'È utile per verificare se gli orari reali di funzionamento sono coerenti con quelli attesi.'),

    ('plot12_scatter_text_mandata',
     'Grafico 12 — Correlazione: Temperatura Esterna vs Temperatura di Mandata',
     'Come cambia la temperatura di mandata al variare della temperatura esterna. '
     'La retta rossa mostra la tendenza: se inclinata verso il basso, l’UTA abbassa la mandata quando fa più caldo fuori.'),

    ('plot13_scatter_modulazione_errore',
     'Grafico 13 — Correlazione: Modulazione vs Errore di Controllo',
        'Confronta lo sforzo del ventilatore con l\'errore di controllo. '
        'Se aumentando la modulazione l\'errore non scende, è un segnale da approfondire con il team tecnico.'),

    ('plot14_scatter_ripresa_mandata',
     'Grafico 14 — Correlazione: Temperatura di Ripresa vs Temperatura di Mandata',
     'Relazione tra la temperatura dell\u2019aria in ingresso all\u2019UTA (ripresa) e quella in uscita (mandata). '
     'Una nuvola di punti coerente indica che l\u2019impianto risponde correttamente al carico termico dell\u2019edificio.'),
]


def _set_cell_bg(cell, hex_color: str) -> None:
    """Set background colour of a Word table cell."""
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement('w:shd')
    shd.set(qn('w:val'),   'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'),  hex_color)
    tcPr.append(shd)


def _df_to_word_table(doc: 'Document', df: pd.DataFrame,
                      header_color: str = '1F4E79') -> None:
    """Render a pandas DataFrame as a styled Word table (appended to doc)."""
    rows, cols = df.shape
    tbl = doc.add_table(rows=rows + 1, cols=cols)
    tbl.style = 'Table Grid'

    # Header row
    hdr_cells = tbl.rows[0].cells
    for i, col_name in enumerate(df.columns):
        cell = hdr_cells[i]
        cell.text = str(col_name)
        _set_cell_bg(cell, header_color)
        para = cell.paragraphs[0]
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = para.runs[0]
        run.font.bold  = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        run.font.size  = Pt(8)

    # Data rows
    for r_idx, row in enumerate(df.itertuples(index=False)):
        row_cells = tbl.rows[r_idx + 1].cells
        fill = 'D9E1F2' if r_idx % 2 == 0 else 'FFFFFF'
        for c_idx, value in enumerate(row):
            cell = row_cells[c_idx]
            cell.text = str(value)
            _set_cell_bg(cell, fill)
            para = cell.paragraphs[0]
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run  = para.runs[0]
            run.font.size = Pt(8)


# =============================================================================
# PLOT CATALOGUE  (saved to plots/ folder, not processed_data/)
# =============================================================================

# Extended metadata for each plot: (filename, title, source_columns, why_created, what_to_look_for)
_PLOT_CATALOGUE_META = [
    (
        'plot01_condizioni_esterne',
        'Grafico 1 — Condizioni Meteorologiche Esterne',
        'Temperatura_Esterna, Umidita_Esterna',
        (
            'Mostra la SERIE TEMPORALE CONTINUA di temperatura esterna (°C) e umidità '
            'relativa (%) per l\'intero periodo, con bande grigie sui weekend. '
            'Pipeline: il carpet plot mostra lo stesso dato come mappa 2D ora×giorno. '
            'Questo grafico lo riporta su un asse temporale lineare, più leggibile per '
            'il responsabile che vuole capire "com\'era il clima in agosto?".'
        ),
        (
            'Picchi di temperatura esterna > 35°C coincidono con i giorni critici? '
            'Il sistema lavora di più (alta modulazione) nelle settimane più calde? '
            'Periodi di bassa umidità combinati con alta temperatura indicano '
            'il massimo carico sul sistema di raffrescamento.'
        ),
    ),
    (
        'plot02_modulazione_ventilatori',
        'Grafico 2 — Modulazione dei Ventilatori (Intero Periodo)',
        'Modulazione_Ventilatore_Mandata, Modulazione_Ventilatore_Ripresa',
        (
            'SERIE TEMPORALE di entrambe le modulazioni (mandata + ripresa) su un unico '
            'grafico, con soglia di accensione (5%) evidenziata. '
            'Pipeline: il carpet plot di modulazione mostra solo mandata in formato 2D. '
            'Questo aggiunge la ripresa, mostra entrambe su asse lineare, e '
            'identifica automaticamente i periodi di fermo impianto.'
        ),
        (
            'I due segnali devono seguirsi abbastanza fedelmente (stessa variazione). '
            'Divergenze persistenti (mandata alta, ripresa bassa o viceversa) possono '
            'indicare problemi ai ventilatori. '
            'Periodi a 0% = impianto spento: corrispondono a weekend/ferie o guasti?'
        ),
    ),
    (
        'plot03_ore_funzionamento_giornaliere',
        'Grafico 3 — Ore di Funzionamento Giornaliere dell\'UTA',
        'AHU_ON (derivato da Modulazione_Ventilatore_Mandata > 5%)',
        (
            'METRICA COMPLETAMENTE NUOVA — non prodotta dalla pipeline. '
            'Calcola le ore/giorno in cui l\'UTA è accesa e le visualizza come '
            'istogramma giornaliero con la media stagionale. '
            'Risponde alla domanda operativa: "Quante ore lavora l\'impianto ogni giorno?" '
            'Nessun grafico della pipeline risponde a questa domanda direttamente.'
        ),
        (
            'La media stagionale è vicina alle 12-14h/giorno? '
            'I weekend hanno meno ore feriali (corretto) o le stesse (impianto non programmato)? '
            'Giorni con sole 2-4h di funzionamento = guasto o spegnimento manuale?'
        ),
    ),
    (
        'plot04_mandata_vs_setpoint_giornaliero',
        'Grafico 4 — Temperatura di Mandata vs Setpoint (Mediana Giornaliera)',
        'Temperatura_Mandata, SP_Temperatura_Mandata, SP_Compensata',
        (
            'AGGREGAZIONE GIORNALIERA con banda di variabilità (25°–75° percentile). '
            'Pipeline: lo scatter TM vs SP_TM mostra punti istantanei (30 min) senza '
            'struttura temporale. Questo grafico mostra come le mediane si evolvono '
            'nel tempo, con la banda di incertezza — molto più leggibile per il manager. '
            'Risponde a: "La temperatura di mandata segue il setpoint giorno per giorno?"'
        ),
        (
            'La linea rossa (mandata) deve restare vicina ai setpoint. '
            'Una banda larga indica giornate molto variabili. '
            'Deriva sistematica in alto = surriscaldamento cronico; in basso = '
            'raffreddamento eccessivo. Picchi isolati = eventi anomali.'
        ),
    ),
    (
        'plot05_settimana_normale',
        'Grafico 5 — Zoom Settimana "Normale" (30 min)',
        'Temperatura_Mandata, Temperatura_Ripresa, SP_Temperatura_Mandata, Modulazione_Ventilatore_Mandata',
        (
            'ZOOM A PIENA RISOLUZIONE sulla settimana con MAE più basso — non esiste '
            'nella pipeline. Mostra pannello superiore (temperature + setpoint) e '
            'pannello inferiore (modulazione ventilatore) a 30-min per l\'intera settimana. '
            'Serve come RIFERIMENTO DI CONFRONTO per il grafico 6 (settimana critica). '
            'Mostra al manager come si comporta l\'impianto quando funziona bene.'
        ),
        (
            'Il ciclo giornaliero è regolare? La temperatura di mandata oscilla intorno '
            'al setpoint? La modulazione sale e scende in modo ordinato? '
            'Qualsiasi deviazione rispetto a questo grafico nella settimana critica '
            'indica un comportamento anomalo.'
        ),
    ),
    (
        'plot06_settimana_critica',
        'Grafico 6 — Zoom Settimana "Critica" (30 min)',
        'Temperatura_Mandata, Temperatura_Ripresa, SP_Temperatura_Mandata, Modulazione_Ventilatore_Mandata',
        (
            'ZOOM A PIENA RISOLUZIONE sulla settimana con MAE più alto, selezionata '
            'automaticamente dall\'algoritmo. Non esiste nella pipeline. '
            'Il confronto diretto con il Grafico 5 (stessa struttura) permette di '
            'vedere le differenze tra una settimana buona e una critica. '
            'È lo strumento diagnostico principale per il responsabile.'
        ),
        (
            'Dove si verificano gli scostamenti dal setpoint? In quali ore del giorno? '
            'La modulazione è coerente con la temperatura? '
            'Ci sono oscillazioni (hunting) o salti bruschi? '
            'Confronta sempre con il Grafico 5 per capire quanto è anomala.'
        ),
    ),
    (
        'plot07_errore_giornaliero',
        'Grafico 7 — MAE Giornaliero |T.Mandata – SP| (KPI Principale)',
        'Abs_Errore = |Temperatura_Mandata − SP_Temperatura_Mandata|, filtrato su AHU_ON',
        (
            'IL KPI PRINCIPALE del report — COMPLETAMENTE NUOVO rispetto alla pipeline. '
            'Calcola il Mean Absolute Error giornaliero e lo visualizza come serie '
            'temporale con soglia di accettabilità (1°C in arancione). '
            'La pipeline ha dist_TM (distribuzione della temperatura grezza) e scatter TM vs SP, '
            'ma nessun grafico mostra il MAE che evolve giorno per giorno. '
            'Questo è il grafico più importante per la tesi.'
        ),
        (
            'Quanti giorni superano la soglia 1°C? Sono concentrati in agosto (caldo)? '
            'C\'è una deriva stagionale (MAE peggiora nelle settimane calde)? '
            'Giorni isolati con MAE >3°C = eventi anomali da investigare. '
            'Confronta con Grafico 1 (temperatura esterna) per trovare correlazioni.'
        ),
    ),
    (
        'plot09_profilo_feriale_vs_weekend',
        'Grafico 9 — Profilo Medio Giornaliero: Feriale vs Weekend',
        'Ore del giorno vs Temperatura_Mandata, SP_Temperatura_Mandata, Modulazione_Ventilatore_Mandata',
        (
            'ANALISI FERIALE/WEEKEND — COMPLETAMENTE NUOVA rispetto alla pipeline. '
            'Calcola il profilo medio delle 24 ore separatamente per giorni feriali '
            'e per weekend, mostrando temperatura + modulazione in due pannelli. '
            'La pipeline non ha nessun grafico simile. '
            'Risponde a: "L\'impianto è programmato diversamente nei weekend?"'
        ),
        (
            'Se i profili feriale e weekend sono identici: l\'impianto non è programmato '
            'su fasce orarie diverse (possibile spreco energetico). '
            'Differenze negli orari di accensione/spegnimento mostrano la programmazione. '
            'Confronta con la politica oraria dichiarata dal responsabile.'
        ),
    ),
    (
        'plot10_heatmap_errore',
        'Grafico 10 — Heatmap Errore di Controllo (Ora × Giorno)',
        'Errore_Controllo = T.Mandata − SP, aggregato per ora del giorno × giorno del periodo',
        (
            'HEATMAP DELL\'ERRORE — diversa dalla pipeline. '
            'Pipeline: carpet plot di T.Mandata (temperatura grezza). '
            'Questo mostra l\'ERRORE (blu = troppo freddo, rosso = troppo caldo), '
            'che è l\'informazione diagnostica più utile. '
            'È lo strumento più efficace per trovare pattern anomali: '
            '"Il surriscaldamento avviene sempre alle 14:00-16:00?" = problema di picco termico.'
        ),
        (
            'Ci sono strisce verticali rosse = giornate sistematicamente calde? '
            'Strisce orizzontali = ore della giornata sempre anomale? '
            'Blocchi di colore omogeneo = periodi di comportamento anomalo prolungato. '
            'Il bianco indica UTA spenta.'
        ),
    ),
    (
        'plot11_heatmap_modulazione',
        'Grafico 11 — Heatmap Modulazione Ventilatore Mandata (Ora × Giorno)',
        'Modulazione_Ventilatore_Mandata, aggregata per ora × giorno',
        (
            'HEATMAP DELLA MODULAZIONE — simile al carpet plot della pipeline ma con '
            'orientamento asse invertito (giorni su asse X, ore su Y) ottimizzato '
            'per la lettura degli orari di accensione. '
            'Pipeline: carpet_plot/Modulazione_Ventilatore_Mandata mostra la stessa variabile. '
            'La differenza: questo è formattato per il report (A4, leggenda chiara, '
            'titolo in italiano) e allineato agli altri grafici del report.'
        ),
        (
            'Quando inizia il funzionamento ogni giorno (transizione da bianco a colore)? '
            'Ci sono notti con modulazione non nulla (impianto dimenticato acceso)? '
            'Periodi di colore molto scuro = massima potenza = periodo critico.'
        ),
    ),
    (
        'plot12_scatter_text_mandata',
        'Grafico 12 — Correlazione: Temperatura Esterna vs Temperatura di Mandata',
        'Temperatura_Esterna vs Temperatura_Mandata, filtrato su AHU_ON',
        (
            'RELAZIONE CONDIZIONI ESTERNE → RISPOSTA UTA — NON nella pipeline. '
            'Pipeline: scatter TM vs SP_TM (confronto con setpoint). '
            'Questo mostra la relazione tra temperatura esterna e temperatura di mandata: '
            'un sistema con compensazione climatica correttamente configurata mostra '
            'correlazione NEGATIVA (più caldo fuori → mandata più fredda). '
            'Risponde a: "L\'impianto si adatta alle condizioni climatiche esterne?"'
        ),
        (
            'Pendenza della retta di regressione: negativa (compensazione attiva) o piatta/positiva? '
            'Alta dispersione dei punti = il sistema non risponde consistentemente alle condizioni esterne. '
            'Cluster separati = comportamenti diversi in fasi stagionali diverse.'
        ),
    ),
    (
        'plot13_scatter_modulazione_errore',
        'Grafico 13 — Correlazione: Modulazione vs Errore di Controllo',
        'Modulazione_Ventilatore_Mandata vs |Errore_Controllo|, filtrato su AHU_ON',
        (
            'ANALISI DEL PUNTO DI LAVORO — COMPLETAMENTE NUOVA. Non ha equivalenti nella pipeline. '
            'Mostra se il ventilatore che lavora di più riduce effettivamente l\'errore. '
            'Idealmente: alta modulazione → basso errore (il sistema compensa il carico). '
            'Se la correlazione è positiva (più modulazione = più errore), indica un problema '
            'alla batteria di scambio, alla valvola di regolazione, o al sensore.'
        ),
        (
            'La mediana dell\'errore per fascia di modulazione (linea rossa) deve essere '
            'decrescente (più modulazione = meno errore). '
            'Se la mediana è crescente o piatta: problema meccanico o di taratura. '
            'Alta dispersione a qualsiasi modulazione = sistema instabile.'
        ),
    ),
    (
        'plot14_scatter_ripresa_mandata',
        'Grafico 14 — Correlazione: Temperatura di Ripresa vs Temperatura di Mandata',
        'Temperatura_Ripresa vs Temperatura_Mandata, filtrato su AHU_ON',
        (
            'BILANCIO TERMICO IMPIANTO — non nella pipeline. '
            'La differenza (T.Mandata − T.Ripresa) è il salto termico effettivo dell\'UTA: '
            'indica quanta energia termica viene trasferita all\'aria. '
            'La linea tratteggiata mostra il punto di saturazione medio della batteria. '
            'Risponde a: "L\'UTA sta trasferendo energia efficacemente?'
            ' C\'è un salto termico adeguato?"'
        ),
        (
            'Il salto termico medio (differenza tra intercetta della nuvola e la bisettrice) '
            'deve essere relativamente costante. '
            'Riduzione progressiva del salto termico nel tempo = fouling (sporcizia) sulla '
            'batteria di scambio. Punti sulla bisettrice (T.mandata ≈ T.ripresa) = '
            'UTA non sta raffrescando nonostante la modulazione.'
        ),
    ),
]


def generate_plot_catalogue() -> Path:
    """
    Create a standalone Word document in the plots/energy_manager_report/ folder
    (NOT processed_data/) that serves as a PLOT CATALOGUE for the 14 custom diagnostic
    plots generated by this script.

    Each plot gets a full card:
      • Coloured header with plot number and title
      • The PNG image (full width)
      • A 3-row info table:
          SOURCE    — which columns / derived metrics are used
          WHY       — what this plot adds vs the existing pipeline plots
          LOOK FOR  — concrete things to check when reading this plot

    The intro page explains WHY these 14 plots exist alongside the pipeline plots.
    """
    if not DOCX_AVAILABLE:
        return None

    print("\n📚 Generazione catalogo grafici...")

    doc = Document()
    for sec in doc.sections:
        sec.top_margin    = Cm(1.8)
        sec.bottom_margin = Cm(1.8)
        sec.left_margin   = Cm(2.2)
        sec.right_margin  = Cm(2.2)

    C_WHITE = RGBColor(0xFF, 0xFF, 0xFF)
    C_NAVY  = RGBColor(0x1F, 0x4E, 0x79)
    C_GREY  = RGBColor(0x44, 0x44, 0x44)
    C_LGREY = RGBColor(0x88, 0x88, 0x88)
    C_GREEN = RGBColor(0x37, 0x86, 0x5F)
    C_AMBER = RGBColor(0xBF, 0x86, 0x00)

    # ── cycle of header colours for cards ────────────────────────────────────
    HEADER_COLORS = [
        '1F4E79', '1E5F6A', '2D6A4F', '5C3317',
        '6B3FA0', '8B0000', '17375E', '1A5276',
        '196F3D', '7D6608', '633974', '922B21',
        '1B4F72', '145A32',
    ]

    def _bg(cell, hex_color):
        tc   = cell._tc
        tcPr = tc.get_or_add_tcPr()
        shd  = OxmlElement('w:shd')
        shd.set(qn('w:val'),   'clear')
        shd.set(qn('w:color'), 'auto')
        shd.set(qn('w:fill'),  hex_color)
        tcPr.append(shd)

    def _pad(cell, top='80', left='120', bot='80', right='120'):
        tc   = cell._tc
        tcPr = tc.get_or_add_tcPr()
        tcMar = OxmlElement('w:tcMar')
        for side, val in (('top', top), ('left', left), ('bottom', bot), ('right', right)):
            m = OxmlElement(f'w:{side}')
            m.set(qn('w:w'),    val)
            m.set(qn('w:type'), 'dxa')
            tcMar.append(m)
        tcPr.append(tcMar)

    def _border(cell, color='2E74B5', sz='8'):
        tc   = cell._tc
        tcPr = tc.get_or_add_tcPr()
        tcBd = OxmlElement('w:tcBorders')
        for side in ('top', 'left', 'bottom', 'right'):
            el = OxmlElement(f'w:{side}')
            el.set(qn('w:val'),   'single')
            el.set(qn('w:sz'),    sz)
            el.set(qn('w:color'), color)
            tcBd.append(el)
        tcPr.append(tcBd)

    def para_run(p, text, bold=False, italic=False, size=10,
                 color: RGBColor = None, newline=False):
        r = p.add_run(('\n' if newline else '') + text)
        r.font.bold      = bold
        r.font.italic    = italic
        r.font.size      = Pt(size)
        if color:
            r.font.color.rgb = color
        return r

    # ══════════════════════════════════════════════════════════════════════════
    # INTRO PAGE
    # ══════════════════════════════════════════════════════════════════════════

    # Title
    tp = doc.add_paragraph()
    tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = tp.add_run('CATALOGO DEI GRAFICI DIAGNOSTICI')
    r.font.bold = True; r.font.size = Pt(20); r.font.color.rgb = C_NAVY

    sp = doc.add_paragraph()
    sp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = sp.add_run(f'14 Grafici Operativi — {AHU}, Edificio {BUILDING} · Estate {YEAR}')
    r2.font.size = Pt(12); r2.font.color.rgb = RGBColor(0x2E, 0x74, 0xB5)

    dp = doc.add_paragraph()
    dp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r3 = dp.add_run(
        f'Varaga Haghoubians — MSc Data Science, UniTO\n'
        f'Generato il {datetime.now().strftime("%d/%m/%Y %H:%M")}  ·  '
        f'Script: energy_manager_report.py'
    )
    r3.font.size = Pt(9); r3.font.italic = True; r3.font.color.rgb = C_LGREY

    doc.add_paragraph('')

    # ── "Perché questi grafici esistono?" explanation box ────────────────────
    why_tbl = doc.add_table(rows=1, cols=1)
    why_tbl.style = 'Table Grid'
    wc = why_tbl.rows[0].cells[0]
    _bg(wc, 'EBF3FB')
    _border(wc, color='1F4E79', sz='18')
    _pad(wc, '100', '140', '100', '140')

    wp = wc.paragraphs[0]
    para_run(wp, 'Perché questi 14 grafici esistono se ho già i grafici dalla pipeline?',
             bold=True, size=11, color=C_NAVY)
    para_run(wp, '\n\n', size=4)

    explanation = [
        ('📁  La pipeline (Script 1–13)', 'C0392B',
         ' produce grafici ESPLORATIVI e STATISTICI: carpet plot, distribuzioni, '
         'box plot, scatter per ogni variabile. Sono grafici di analisi statistica, '
         'ottimi per capire i dati ma non progettati per un report operativo.'),
        ('📊  Questi 14 grafici', '1F4E79',
         ' sono stati creati SPECIFICATAMENTE per il responsabile dell\'energia: '
         'calcolano KPI operativi (MAE giornaliero, ore di funzionamento, profili feriali), '
         'mostrano confronti settimanali a piena risoluzione, e usano aggregazioni '
         'temporali (giornaliere, orarie medie) che la pipeline non produce.'),
        ('🆕  Grafici completamente nuovi', '2D6A4F',
         ' senza equivalenti nella pipeline: '
         'Grafico 3 (ore funzionamento/giorno), '
         'Grafico 7 (MAE giornaliero), '
         'Grafico 9 (feriale vs weekend), '
         'Grafico 13 (modulazione vs errore).'),
        ('🔄  Grafici che rielaborano dati della pipeline', '8B5E3C',
         ' con un\'angolazione diversa: '
         'Grafico 10 mostra l\'ERRORE in heatmap (non la temperatura grezza); '
         'Grafico 8 mostra la distribuzione dell\'ERRORE (non della temperatura); '
         'Grafici 5 e 6 zoomano su settimane specifiche a 30 min. '),
    ]

    for label, lbl_hex, text in explanation:
        doc.add_paragraph('')
        item_tbl = doc.add_table(rows=1, cols=1)
        item_tbl.style = 'Table Grid'
        ic = item_tbl.rows[0].cells[0]
        _bg(ic, 'F8FBFF')
        _border(ic, color=lbl_hex, sz='12')
        _pad(ic, '60', '100', '60', '100')
        p = ic.paragraphs[0]
        para_run(p, label, bold=True, size=10,
                 color=RGBColor(int(lbl_hex[0:2], 16),
                                int(lbl_hex[2:4], 16),
                                int(lbl_hex[4:6], 16)))
        para_run(p, text, size=10, color=C_GREY)

    doc.add_paragraph('')

    # ── Table of contents strip ───────────────────────────────────────────────
    toc = doc.add_table(rows=1, cols=7)
    toc.style = 'Table Grid'
    toc.alignment = WD_TABLE_ALIGNMENT.CENTER
    toc_colors = ['1F4E79', '1E5F6A', '2D6A4F', '5C3317', '6B3FA0', '8B0000', '17375E']
    toc_groups = [
        ('1–3', 'Condizioni\ne Funzionamento'),
        ('4–6', 'Temperatura\nvs Setpoint'),
        ('7–8', 'KPI\nErrore'),
        ('9', 'Profili\nOrari'),
        ('10–11', 'Heatmap'),
        ('12–14', 'Correlazioni\nVariabili'),
        ('📄', 'Questo\nDocument'),
    ]
    for i, (nums, lbl) in enumerate(toc_groups):
        c = toc.rows[0].cells[i]
        _bg(c, toc_colors[i % len(toc_colors)])
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_n = p.add_run(nums + '\n')
        r_n.font.bold = True; r_n.font.size = Pt(11); r_n.font.color.rgb = C_WHITE
        r_l = p.add_run(lbl)
        r_l.font.size = Pt(7); r_l.font.color.rgb = RGBColor(0xCC, 0xDD, 0xFF)
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after  = Pt(4)

    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════════
    # ONE CARD PER PLOT
    # ══════════════════════════════════════════════════════════════════════════
    row_labels   = ['📥  Dati di input',   '🔎  Perché questo grafico',  '👁  Cosa osservare']
    row_label_bg = ['1F4E79',               '1A5276',                     '145A32']
    row_body_bg  = ['EAF2FB',               'EBF5FB',                     'EAFAF1']

    for idx, (fname, title, source, why, look) in enumerate(_PLOT_CATALOGUE_META):
        hdr_color = HEADER_COLORS[idx % len(HEADER_COLORS)]
        img_path  = out_plots / f'{fname}.png'

        # ── Header row ────────────────────────────────────────────────────────
        hdr_tbl = doc.add_table(rows=1, cols=1)
        hdr_tbl.style = 'Table Grid'
        hdr_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        hc = hdr_tbl.rows[0].cells[0]
        _bg(hc, hdr_color)
        _border(hc, color=hdr_color)
        _pad(hc, '80', '140', '80', '140')
        p = hc.paragraphs[0]
        nr = p.add_run(f'Grafico {idx + 1:02d}   ')
        nr.font.bold = True; nr.font.size = Pt(9)
        nr.font.color.rgb = RGBColor(0xFF, 0xFF, 0xAA)
        tr2 = p.add_run(title.split('—', 1)[-1].strip() if '—' in title else title)
        tr2.font.bold = True; tr2.font.size = Pt(11); tr2.font.color.rgb = C_WHITE

        # ── Image ─────────────────────────────────────────────────────────────
        img_tbl = doc.add_table(rows=1, cols=1)
        img_tbl.style = 'Table Grid'
        img_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        ic2 = img_tbl.rows[0].cells[0]
        _bg(ic2, 'FFFFFF')
        _border(ic2, color='CCCCCC', sz='4')
        ip = ic2.paragraphs[0]
        ip.alignment = WD_ALIGN_PARAGRAPH.CENTER
        ip.paragraph_format.space_before = Pt(6)
        ip.paragraph_format.space_after  = Pt(6)
        if img_path.exists():
            ip.add_run().add_picture(str(img_path), width=Inches(6.2))
        else:
            r_miss = ip.add_run(
                f'[Immagine non trovata: {img_path.name}]\n'
                f'Esegui prima lo script per generare i grafici.'
            )
            r_miss.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)
            r_miss.font.italic    = True

        # ── Info table (3 rows: source / why / look) ─────────────────────────
        info_tbl = doc.add_table(rows=3, cols=2)
        info_tbl.style = 'Table Grid'
        info_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER

        # Fix label column width narrow
        for row_i, (rlabel, rbg, body_bg, body_text) in enumerate(zip(
            row_labels, row_label_bg, row_body_bg,
            [source, why, look]
        )):
            # Label cell
            lc = info_tbl.rows[row_i].cells[0]
            _bg(lc, rbg)
            _border(lc, color=rbg, sz='6')
            _pad(lc, '60', '100', '60', '100')
            lp = lc.paragraphs[0]
            lp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            lr2 = lp.add_run(rlabel)
            lr2.font.bold       = True
            lr2.font.size       = Pt(9)
            lr2.font.color.rgb  = C_WHITE

            # Body cell
            bc = info_tbl.rows[row_i].cells[1]
            _bg(bc, body_bg)
            _border(bc, color='BBCCE0', sz='4')
            _pad(bc, '60', '120', '60', '120')
            bp = bc.paragraphs[0]
            br = bp.add_run(body_text)
            br.font.size      = Pt(9)
            br.font.italic    = (row_i == 1)   # italics for the "why"
            br.font.color.rgb = C_GREY

        # Set label column width
        for row in info_tbl.rows:
            row.cells[0].width = Cm(3.5)

        doc.add_paragraph('')   # gap between cards
        doc.add_paragraph('')

        # Page break every 1 plot (each card is large)
        if idx < len(_PLOT_CATALOGUE_META) - 1:
            doc.add_page_break()

    # ── Save ──────────────────────────────────────────────────────────────────
    cat_path = out_plots / f'PlotCatalogue_{FOLDER_ID}.docx'
    try:
        doc.save(cat_path)
    except PermissionError:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        cat_path = out_plots / f'PlotCatalogue_{FOLDER_ID}_{ts}.docx'
        doc.save(cat_path)
        print('   ⚠️  File catalogo in uso: creato file con timestamp.')
    print(f"   ✅ Catalogo grafici salvato: {cat_path}")
    return cat_path


def generate_word_report(
    df: pd.DataFrame,
    t1: pd.DataFrame,
    t2: pd.DataFrame,
    t3: pd.DataFrame,
    normal_week: pd.Timestamp,
    crit_week: pd.Timestamp,
) -> Path:
    """
    Build a richly-formatted Word report (inspired by Methodological Report style):
      • Cover page — university header, title, KPI strip, author/recipient boxes
      • Section 1 — Context & objectives
      • Section 2 — Monitored variables table
      • Section 3 — Season overview (pipeline plots as photo cards)
      • Section 4 — Detailed diagnostic plots (2-column cards)
      • Section 5 — Weekly KPI table with colour-coded performance
      • Section 6 — Top-10 worst days
      • Section 7 — Structured feedback form (question + writing space per item)
    """
    if not DOCX_AVAILABLE:
        print("\n⚠️  python-docx non installato — report Word saltato.")
        print("   Installa con: pip install python-docx")
        return None

    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    print("\n📄 Generazione report Word (stile colorato)...")

    doc = Document()

    # ── Page margins ─────────────────────────────────────────────────────────
    for sec in doc.sections:
        sec.top_margin    = Cm(2)
        sec.bottom_margin = Cm(2)
        sec.left_margin   = Cm(2.5)
        sec.right_margin  = Cm(2.5)

    # ── Colour palette ───────────────────────────────────────────────────────
    C_NAVY   = RGBColor(0x1F, 0x4E, 0x79)   # dark navy heading
    C_TEAL   = RGBColor(0x1E, 0x7E, 0x7E)   # teal accent
    C_STEEL  = RGBColor(0x2E, 0x74, 0xB5)   # medium blue
    C_GREEN  = RGBColor(0x37, 0x86, 0x5F)   # dark green
    C_AMBER  = RGBColor(0xBF, 0x86, 0x00)   # amber
    C_RED    = RGBColor(0xC0, 0x00, 0x00)   # alert red
    C_GREY   = RGBColor(0x44, 0x44, 0x44)   # body text grey
    C_LGREY  = RGBColor(0x88, 0x88, 0x88)   # light grey / captions
    C_WHITE  = RGBColor(0xFF, 0xFF, 0xFF)

    # ── Low-level XML helpers ─────────────────────────────────────────────────
    def _add_horiz_rule(para):
        """Insert a thin horizontal rule below a paragraph via pPr/pBdr."""
        pPr  = para._p.get_or_add_pPr()
        pBdr = OxmlElement('w:pBdr')
        bot  = OxmlElement('w:bottom')
        bot.set(qn('w:val'),   'single')
        bot.set(qn('w:sz'),    '4')
        bot.set(qn('w:space'), '1')
        bot.set(qn('w:color'), '2E74B5')
        pBdr.append(bot)
        pPr.append(pBdr)

    def _cell_border(cell, color_hex='2E74B5', size='12'):
        """Set all four borders of a table cell."""
        tc   = cell._tc
        tcPr = tc.get_or_add_tcPr()
        tcBd = OxmlElement('w:tcBorders')
        for side in ('top', 'left', 'bottom', 'right'):
            el = OxmlElement(f'w:{side}')
            el.set(qn('w:val'),   'single')
            el.set(qn('w:sz'),    size)
            el.set(qn('w:color'), color_hex)
            tcBd.append(el)
        tcPr.append(tcBd)

    def _no_border_table(tbl):
        """Remove all visible borders from a table."""
        tbl_el = tbl._tbl
        tblPr  = tbl_el.find(qn('w:tblPr'))
        if tblPr is None:
            tblPr = OxmlElement('w:tblPr')
            tbl_el.insert(0, tblPr)
        tblBd = OxmlElement('w:tblBorders')
        for side in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
            el = OxmlElement(f'w:{side}')
            el.set(qn('w:val'), 'none')
            tblBd.append(el)
        tblPr.append(tblBd)

    # ── High-level helpers ───────────────────────────────────────────────────
    def heading(text: str, level: int = 1, color: RGBColor = None, rule: bool = True) -> None:
        p = doc.add_heading(text, level=level)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        clr = color or C_NAVY
        for run in p.runs:
            run.font.color.rgb = clr
        if rule and level == 1:
            _add_horiz_rule(p)

    def subheading(text: str, color: RGBColor = None) -> None:
        heading(text, level=2, color=color or C_STEEL, rule=False)

    def body(text: str, italic: bool = False, bold: bool = False,
             color: RGBColor = None, size: int = 10) -> None:
        p   = doc.add_paragraph(text)
        run = p.runs[0] if p.runs else p.add_run(text)
        run.font.size   = Pt(size)
        run.font.italic = italic
        run.font.bold   = bold
        if color:
            run.font.color.rgb = color
        p.paragraph_format.space_after  = Pt(4)
        p.paragraph_format.space_before = Pt(0)

    def caption(text: str) -> None:
        p   = doc.add_paragraph(text)
        run = p.runs[0] if p.runs else p.add_run(text)
        run.font.size   = Pt(9)
        run.font.italic = True
        run.font.color.rgb = C_LGREY
        p.paragraph_format.space_after  = Pt(8)
        p.paragraph_format.space_before = Pt(2)

    def gap(n: int = 1) -> None:
        for _ in range(n):
            p = doc.add_paragraph('')
            p.paragraph_format.space_after  = Pt(2)
            p.paragraph_format.space_before = Pt(0)

    def callout_box(text: str, bg: str = 'EBF3FB', border: str = '2E74B5',
                    label: str = None, label_color: RGBColor = None,
                    text_color: RGBColor = None, text_bold: bool = False) -> None:
        """Single-cell coloured callout box (like the blue/red summary boxes in the screenshots)."""
        tbl = doc.add_table(rows=1, cols=1)
        tbl.style = 'Table Grid'
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = tbl.rows[0].cells[0]
        _set_cell_bg(cell, bg)
        _cell_border(cell, color_hex=border, size='18')
        # Remove outer table border (we use cell border only)
        p = cell.paragraphs[0]
        if label:
            lr = p.add_run(label + '  ')
            lr.font.bold = True
            lr.font.size = Pt(10)
            lr.font.color.rgb = label_color or C_NAVY
        tr = p.add_run(text)
        tr.font.size  = Pt(10)
        tr.font.bold  = text_bold
        tr.font.color.rgb = text_color or C_GREY
        p.paragraph_format.space_after  = Pt(4)
        p.paragraph_format.space_before = Pt(4)
        # inner cell padding via tcMar
        tc   = cell._tc
        tcPr = tc.get_or_add_tcPr()
        tcMar = OxmlElement('w:tcMar')
        for side, val in (('top', '80'), ('left', '120'), ('bottom', '80'), ('right', '120')):
            m = OxmlElement(f'w:{side}')
            m.set(qn('w:w'),    val)
            m.set(qn('w:type'), 'dxa')
            tcMar.append(m)
        tcPr.append(tcMar)
        gap()

    def photo_card(img_path: Path, title: str, description: str,
                   header_bg: str = '1F4E79', header_text_color: RGBColor = None,
                   desc_bg: str = 'F0F4FA', width_inches: float = 6.0) -> None:
        """3-row card: coloured header | centred image | light-grey description."""
        tbl = doc.add_table(rows=3, cols=1)
        tbl.style = 'Table Grid'
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER

        # Row 0 — header strip
        hdr_cell = tbl.rows[0].cells[0]
        _set_cell_bg(hdr_cell, header_bg)
        _cell_border(hdr_cell, color_hex=header_bg)
        p = hdr_cell.paragraphs[0]
        r = p.add_run(title)
        r.font.bold      = True
        r.font.size      = Pt(10)
        r.font.color.rgb = header_text_color or C_WHITE
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after  = Pt(4)
        tc = hdr_cell._tc
        tcPr = tc.get_or_add_tcPr()
        tcMar = OxmlElement('w:tcMar')
        for side, val in (('top', '60'), ('left', '120'), ('bottom', '60'), ('right', '120')):
            m = OxmlElement(f'w:{side}'); m.set(qn('w:w'), val); m.set(qn('w:type'), 'dxa')
            tcMar.append(m)
        tcPr.append(tcMar)

        # Row 1 — image
        img_cell = tbl.rows[1].cells[0]
        _set_cell_bg(img_cell, 'FFFFFF')
        _cell_border(img_cell, color_hex='CCCCCC', size='4')
        ip = img_cell.paragraphs[0]
        ip.alignment = WD_ALIGN_PARAGRAPH.CENTER
        ip.paragraph_format.space_before = Pt(6)
        ip.paragraph_format.space_after  = Pt(6)
        if img_path.exists():
            ip.add_run().add_picture(str(img_path), width=Inches(width_inches))
        else:
            r2 = ip.add_run(f'[Immagine non disponibile: {img_path.name}]')
            r2.font.color.rgb = C_RED
            r2.font.italic    = True

        # Row 2 — description
        desc_cell = tbl.rows[2].cells[0]
        _set_cell_bg(desc_cell, desc_bg)
        _cell_border(desc_cell, color_hex='BBCCE0', size='4')
        dp = desc_cell.paragraphs[0]
        dr = dp.add_run(description)
        dr.font.size      = Pt(9)
        dr.font.italic    = True
        dr.font.color.rgb = C_GREY
        dp.paragraph_format.space_before = Pt(4)
        dp.paragraph_format.space_after  = Pt(4)
        tc2 = desc_cell._tc
        tcPr2 = tc2.get_or_add_tcPr()
        tcMar2 = OxmlElement('w:tcMar')
        for side, val in (('top', '60'), ('left', '120'), ('bottom', '60'), ('right', '120')):
            m = OxmlElement(f'w:{side}'); m.set(qn('w:w'), val); m.set(qn('w:type'), 'dxa')
            tcMar2.append(m)
        tcPr2.append(tcMar2)

        gap()

    def plot_card_2col(left_meta, right_meta) -> None:
        """Side-by-side 2-column plot card with header strip + image + description."""
        tbl = doc.add_table(rows=3, cols=2)
        tbl.style = 'Table Grid'
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.columns[0].width = Cm(8.5)
        tbl.columns[1].width = Cm(8.5)

        for col_idx, meta in enumerate([left_meta, right_meta]):
            # header row
            hdr = tbl.rows[0].cells[col_idx]
            _set_cell_bg(hdr, '1F4E79' if col_idx == 0 else '2E5F8A')
            p = hdr.paragraphs[0]
            lbl = meta[1] if meta else ''
            r = p.add_run(lbl)
            r.font.bold      = True
            r.font.size      = Pt(9)
            r.font.color.rgb = C_WHITE
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after  = Pt(3)
            tc = hdr._tc
            tcPr = tc.get_or_add_tcPr()
            tcMar = OxmlElement('w:tcMar')
            for side, val in (('top', '40'), ('left', '80'), ('bottom', '40'), ('right', '80')):
                m = OxmlElement(f'w:{side}'); m.set(qn('w:w'), val); m.set(qn('w:type'), 'dxa')
                tcMar.append(m)
            tcPr.append(tcMar)

            # image row
            img_c = tbl.rows[1].cells[col_idx]
            _set_cell_bg(img_c, 'FFFFFF')
            ip = img_c.paragraphs[0]
            ip.alignment = WD_ALIGN_PARAGRAPH.CENTER
            ip.paragraph_format.space_before = Pt(4)
            ip.paragraph_format.space_after  = Pt(4)
            if meta:
                fname = meta[0]
                img_path = out_plots / f'{fname}.png'
                if img_path.exists():
                    ip.add_run().add_picture(str(img_path), width=Inches(3.0))
                else:
                    ip.add_run(f'[{img_path.name}]').font.color.rgb = C_RED

            # description row
            desc_c = tbl.rows[2].cells[col_idx]
            _set_cell_bg(desc_c, 'EBF3FB' if col_idx == 0 else 'EDF5F0')
            dp = desc_c.paragraphs[0]
            dp.paragraph_format.space_before = Pt(3)
            dp.paragraph_format.space_after  = Pt(3)
            tc2 = desc_c._tc
            tcPr2 = tc2.get_or_add_tcPr()
            tcMar2 = OxmlElement('w:tcMar')
            for side, val in (('top', '40'), ('left', '80'), ('bottom', '40'), ('right', '80')):
                m = OxmlElement(f'w:{side}'); m.set(qn('w:w'), val); m.set(qn('w:type'), 'dxa')
                tcMar2.append(m)
            tcPr2.append(tcMar2)
            if meta:
                dr = dp.add_run(meta[2])
                dr.font.size      = Pt(8)
                dr.font.italic    = True
                dr.font.color.rgb = C_GREY

        gap()

    def feedback_block(question: str, q_number: int,
                        hint: str = '', lines: int = 4) -> None:
        """Coloured question header + bordered white writing space."""
        # Question banner
        colors = ['1F4E79', '1E5F6A', '2D6A4F', '8B5E3C', '6B3FA0', '8B0000']
        bg = colors[(q_number - 1) % len(colors)]
        tbl_q = doc.add_table(rows=1, cols=1)
        tbl_q.style = 'Table Grid'
        cell_q = tbl_q.rows[0].cells[0]
        _set_cell_bg(cell_q, bg)
        _cell_border(cell_q, color_hex=bg)
        p = cell_q.paragraphs[0]
        nr = p.add_run(f'{q_number}.  ')
        nr.font.bold      = True
        nr.font.size      = Pt(10)
        nr.font.color.rgb = RGBColor(0xFF, 0xFF, 0xCC)
        qr = p.add_run(question)
        qr.font.bold      = True
        qr.font.size      = Pt(10)
        qr.font.color.rgb = C_WHITE
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after  = Pt(4)
        tc = cell_q._tc
        tcPr = tc.get_or_add_tcPr()
        tcMar = OxmlElement('w:tcMar')
        for side, val in (('top', '60'), ('left', '120'), ('bottom', '60'), ('right', '120')):
            m = OxmlElement(f'w:{side}'); m.set(qn('w:w'), val); m.set(qn('w:type'), 'dxa')
            tcMar.append(m)
        tcPr.append(tcMar)

        # Writing space
        tbl_w = doc.add_table(rows=1, cols=1)
        tbl_w.style = 'Table Grid'
        cell_w = tbl_w.rows[0].cells[0]
        _set_cell_bg(cell_w, 'FAFCFF')
        _cell_border(cell_w, color_hex='BBCCE0', size='6')
        if hint:
            hp = cell_w.paragraphs[0]
            hr = hp.add_run(f'Suggerimento: {hint}')
            hr.font.size      = Pt(9)
            hr.font.italic    = True
            hr.font.color.rgb = RGBColor(0xAA, 0xBB, 0xCC)
            hp.paragraph_format.space_before = Pt(4)

        for i in range(lines):
            lp = cell_w.add_paragraph()
            lr = lp.add_run('_' * 95)
            lr.font.size      = Pt(9)
            lr.font.color.rgb = RGBColor(0xCC, 0xDD, 0xEE)
            lp.paragraph_format.space_after  = Pt(1)
            lp.paragraph_format.space_before = Pt(8 if i == 0 else 4)
        end_p = cell_w.add_paragraph()
        end_p.paragraph_format.space_before = Pt(4)
        end_p.paragraph_format.space_after  = Pt(4)

        gap()

    # ── Summary stats ─────────────────────────────────────────────────────────
    total_days  = (df.index[-1] - df.index[0]).days + 1
    total_on_h  = df['AHU_ON'].sum() * 0.5
    avg_on_h    = total_on_h / total_days
    mae_overall = df[df['AHU_ON']]['Abs_Errore'].mean()
    pct_in_band = (df[df['AHU_ON']]['Abs_Errore'] <= 0.5).mean() * 100
    pct_1deg    = (df[df['AHU_ON']]['Abs_Errore'] <= 1.0).mean() * 100
    mae_bias    = df[df['AHU_ON']]['Errore_SP_Mandata'].mean()
    wkd_on_h    = df[df['AHU_ON'] & (df.index.weekday < 5)].shape[0] * 0.5
    wke_on_h    = df[df['AHU_ON'] & (df.index.weekday >= 5)].shape[0] * 0.5
    t_ext_mean  = df[COL_TEXT].mean() if COL_TEXT in df.columns else float('nan')
    t_ext_max   = df[COL_TEXT].max()  if COL_TEXT in df.columns else float('nan')

    # Performance: green ≥75%, amber 50–74%, red <50%
    if pct_in_band >= 75:
        perf_label, perf_bg, perf_border = 'BUONO', '2D6A4F', '1B4332'
        perf_text_color = RGBColor(0xD8, 0xF3, 0xDC)
    elif pct_in_band >= 50:
        perf_label, perf_bg, perf_border = 'ACCETTABILE', '8B5E3C', '5C3317'
        perf_text_color = RGBColor(0xFF, 0xF3, 0xCD)
    else:
        perf_label, perf_bg, perf_border = 'INSUFFICIENTE', 'C00000', '7B0000'
        perf_text_color = RGBColor(0xFF, 0xE0, 0xE0)

    # ════════════════════════════════════════════════════════════════════════
    # COVER PAGE
    # ════════════════════════════════════════════════════════════════════════

    # Top university name bar
    gap(2)
    uni_p = doc.add_paragraph()
    uni_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    ur = uni_p.add_run('UNIVERSITÀ DEGLI STUDI DI TORINO')
    ur.font.bold      = True
    ur.font.size      = Pt(14)
    ur.font.color.rgb = C_NAVY
    campus_p = doc.add_paragraph()
    campus_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cr = campus_p.add_run(f'Campus Luigi Einaudi — Edificio {BUILDING} — {AHU}')
    cr.font.size      = Pt(11)
    cr.font.color.rgb = C_GREY
    _add_horiz_rule(campus_p)
    gap(2)

    # Main title
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tr = title_p.add_run('REPORT PRESTAZIONALE IMPIANTO AHU')
    tr.font.bold      = True
    tr.font.size      = Pt(22)
    tr.font.color.rgb = C_NAVY

    sub_p = doc.add_paragraph()
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sr = sub_p.add_run(f'Analisi delle Prestazioni Operative — {AHU}, Edificio {BUILDING}')
    sr.font.bold      = True
    sr.font.size      = Pt(13)
    sr.font.color.rgb = C_STEEL

    date_p = doc.add_paragraph()
    date_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    dr_ = date_p.add_run(
        f'Stagione Estiva {YEAR}  |  '
        f'{df.index[0].strftime("%d %B")} – {df.index[-1].strftime("%d %B %Y")}'
    )
    dr_.font.size      = Pt(11)
    dr_.font.color.rgb = C_GREY
    _add_horiz_rule(date_p)
    gap(2)

    # Author / Recipient — 2-column table (no borders)
    info_tbl = doc.add_table(rows=1, cols=2)
    _no_border_table(info_tbl)
    info_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER

    def _info_cell(cell, label, name, details_lines):
        _set_cell_bg(cell, 'F5F8FD')
        _cell_border(cell, color_hex='BBCCE0', size='6')
        p = cell.paragraphs[0]
        lr = p.add_run(label + '\n')
        lr.font.size      = Pt(9)
        lr.font.italic    = True
        lr.font.color.rgb = C_LGREY
        nr = p.add_run(name + '\n')
        nr.font.bold      = True
        nr.font.size      = Pt(11)
        nr.font.color.rgb = C_NAVY
        for detail in details_lines:
            dr2 = p.add_run(detail + '\n')
            dr2.font.size      = Pt(9)
            dr2.font.color.rgb = C_GREY
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after  = Pt(6)
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        tcMar = OxmlElement('w:tcMar')
        for side, val in (('top', '80'), ('left', '120'), ('bottom', '80'), ('right', '120')):
            m = OxmlElement(f'w:{side}'); m.set(qn('w:w'), val); m.set(qn('w:type'), 'dxa')
            tcMar.append(m)
        tcPr.append(tcMar)

    _info_cell(info_tbl.rows[0].cells[0], 'Autore',
               'Varaga Haghoubians',
               ['MSc Data Science — Stochastics and Data Science',
                f'In collaborazione con Eurix',
                f'Tesi di laurea magistrale, UniTO'])
    _info_cell(info_tbl.rows[0].cells[1], 'Destinatario',
               'Responsabile dell\'Energia — Eurix',
               [f'Versione: Bozza in costruzione progressiva',
                f'{datetime.now().strftime("%B %Y")}'])
    gap(2)

    # Coloured KPI strip (5 cells)
    kpi_data_cover = [
        ('Giorni analizzati',     f'{total_days}',                '1F4E79'),
        ('Ore funzionamento',     f'{total_on_h:,.0f} h',         '1E5F6A'),
        ('MAE Controllo',         f'{mae_overall:.2f} °C',        '2D6A4F' if mae_overall < 1.0 else ('BF8600' if mae_overall < 2.0 else 'C00000')),
        ('Tempo in banda ±0.5°C', f'{pct_in_band:.1f}%',         '2D6A4F' if pct_in_band >= 75 else ('BF8600' if pct_in_band >= 50 else 'C00000')),
        ('T. esterna media',      f'{t_ext_mean:.1f} °C',         '5C3317'),
    ]
    kpi_strip = doc.add_table(rows=1, cols=5)
    _no_border_table(kpi_strip)
    kpi_strip.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, (lbl, val, bg) in enumerate(kpi_data_cover):
        cell = kpi_strip.rows[0].cells[i]
        _set_cell_bg(cell, bg)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        vr = p.add_run(val + '\n')
        vr.font.bold      = True
        vr.font.size      = Pt(13)
        vr.font.color.rgb = C_WHITE
        lr2 = p.add_run(lbl)
        lr2.font.size      = Pt(8)
        lr2.font.color.rgb = RGBColor(0xCC, 0xDD, 0xFF)
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after  = Pt(6)
    gap(2)

    # Performance verdict
    callout_box(
        f'Valutazione complessiva del controllo: {perf_label}.  '
        f'Il sistema mantiene la temperatura entro ±0.5°C dal setpoint nel {pct_in_band:.1f}% dei campioni '
        f'(soglia minima accettabile: 75%).  '
        f'MAE medio = {mae_overall:.2f}°C, bias = {mae_bias:+.2f}°C.',
        bg=perf_bg, border=perf_border,
        text_color=perf_text_color, text_bold=True
    )

    # Key messages for non-technical readers
    key_msg_1 = (
        f'Comfort: solo il {pct_in_band:.1f}% del tempo la mandata resta molto vicina al target (±0.5°C). '
        f'Obiettivo consigliato: almeno 75%.'
    )
    key_msg_2 = (
        f'Stabilità: errore medio stagionale {mae_overall:.2f}°C; '
        f'la settimana più critica inizia il {crit_week.strftime("%d %b %Y")}. '
        f'Priorità: analizzare cause operative di quella settimana.'
    )
    key_msg_3 = (
        f'Energia: ore di funzionamento {total_on_h:,.0f}h '
        f'({avg_on_h:.1f}h/giorno). Confrontare feriali ({wkd_on_h:,.0f}h) e weekend ({wke_on_h:,.0f}h) '
        f'per verificare margini di riduzione.'
    )
    callout_box(
        f'Messaggi chiave per la discussione con il manager:\n'
        f'• {key_msg_1}\n'
        f'• {key_msg_2}\n'
        f'• {key_msg_3}',
        bg='EAF2FB', border='2E74B5',
        label='🎯  In 60 secondi:', label_color=C_NAVY
    )

    tagline_p = doc.add_paragraph()
    tagline_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tgr = tagline_p.add_run(
        'Questo documento presenta i risultati dell\'analisi condotta sui dati BMS, '
        'con grafici, KPI settimanali e una sezione dedicata al feedback del responsabile dell\'energia.'
    )
    tgr.font.size      = Pt(9)
    tgr.font.italic    = True
    tgr.font.color.rgb = C_LGREY

    doc.add_page_break()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 1 — CONTESTO E OBIETTIVI
    # ════════════════════════════════════════════════════════════════════════
    heading('1. Contesto e Obiettivi dell\'Analisi')
    body(
        f'Il presente documento riassume l\'analisi delle prestazioni dell\'Unità di '
        f'Trattamento Aria (UTA) denominata {AHU}, installata nell\'edificio {BUILDING} '
        f'del Campus Luigi Einaudi dell\'Università degli Studi di Torino. '
        f'Il periodo di analisi copre la stagione estiva {YEAR}, dal '
        f'{df.index[0].strftime("%d %B")} al {df.index[-1].strftime("%d %B %Y")} '
        f'({total_days} giorni totali), con dati acquisiti ogni 30 minuti dal sistema BMS.'
    )
    body(
        'L\'analisi è condotta nell\'ambito di una tesi di laurea magistrale in '
        'Data Science (curricolo Stochastics and Data Science) presso UniTO, '
        'in collaborazione con Eurix. '
        'L\'obiettivo è valutare la qualità del controllo del sistema, identificare '
        'inefficienze operative e fornire al responsabile dell\'energia informazioni '
        'concrete per ottimizzare la gestione dell\'impianto.'
    )

    # Key findings summary box
    callout_box(
        f'Risultati principali: MAE = {mae_overall:.2f}°C · '
        f'Tempo in banda ±0.5°C = {pct_in_band:.1f}% · '
        f'Tempo in banda ±1.0°C = {pct_1deg:.1f}% · '
        f'Bias sistematico = {mae_bias:+.2f}°C · '
        f'Ore totali di funzionamento = {total_on_h:,.0f}h su {total_days} giorni.',
        bg='EBF3FB', border='2E74B5',
        label='ℹ  Riepilogo:', label_color=C_NAVY
    )

    subheading('1.1 Indice di qualità del controllo')
    body(
        'In questa sezione usiamo indicatori semplici: quanto spesso la mandata resta vicina al target '
        'e quali giorni sono stati più difficili da controllare. '
        'I dettagli tecnici restano disponibili nel catalogo grafici, ma qui il focus è decisionale.'
    )

    subheading('1.2 Struttura del documento')
    body(
        'Il report è organizzato in 7 sezioni: '
        '(1) Contesto e obiettivi; '
        '(2) Dizionario variabili con statistiche; '
        '(3) Panoramica visiva della stagione (carpet plot, scatter e boxplot); '
        '(4) Analisi grafica dettagliata (14 grafici operativi); '
        '(5) KPI settimanali; '
        '(6) Giorni critici; '
        '(7) Sezione feedback per il responsabile dell\'energia.'
    )

    doc.add_page_break()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 2 — DIZIONARIO DATI
    # ════════════════════════════════════════════════════════════════════════
    heading('2. Variabili Monitorate — Dizionario Dati')
    body(
        'Le 9 variabili acquisite dal sistema BMS sono descritte nella tabella seguente. '
        'Temperature in °C, modulazioni in %. I valori statistici si riferiscono all\'intero '
        f'periodo ({df.index[0].strftime("%d/%m/%Y")} – {df.index[-1].strftime("%d/%m/%Y")}).'
    )
    _df_to_word_table(doc, t1[['Variabile', 'Descrizione', 'Unità', 'Media', 'Min', 'Max']])
    caption('Tabella 1 — Dizionario delle variabili monitorate con statistiche descrittive sull\'intero periodo di analisi.')

    doc.add_page_break()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 3 — PANORAMICA DEL PERIODO
    # ════════════════════════════════════════════════════════════════════════
    heading('3. Panoramica del Periodo di Analisi')
    body(
        'Questa sezione mostra solo grafici di lettura immediata, utili per una discussione rapida con il manager. '
        'Ogni grafico ha una breve spiegazione in linguaggio non tecnico (massimo due frasi).'
    )
    gap()

    plots_root = PROJECT_ROOT / 'plots'

    # Header colors cycling for variety
    _overview_headers = ['1F4E79', '1E5F6A', '17375E', '2D6A4F', '5C3317', '6B3FA0']
    _PIPELINE_OVERVIEW = [
        (
            f'carpet_plot/{FOLDER_ID}/Temperatura_Mandata_combined_plot.png',
            'Carpet Plot — Temperatura di Mandata (intero periodo)',
            'Ogni riga rappresenta un giorno, ogni colonna un\'ora della giornata. '
            'I toni caldi indicano temperature elevate, quelli freddi temperature basse. '
            'Permette di identificare pattern stagionali, anomalie giornaliere e '
            'variazioni settimanali del controllo.',
        ),
        (
            f'carpet_plot/{FOLDER_ID}/Modulazione_Ventilatore_Mandata_combined_plot.png',
            'Carpet Plot — Modulazione Ventilatore di Mandata',
            'La modulazione del ventilatore di mandata visualizzata come carpet plot. '
            'I periodi di bassa modulazione (toni chiari) corrispondono a notti e weekend; '
            'quelli ad alta modulazione (toni scuri) alle ore di piena operatività dell\'edificio. '
            'Utile per verificare la coerenza dei profili orari programmati.',
        ),
        (
            f'carpet_plot/{FOLDER_ID}/Temperatura_Esterna_combined_plot.png',
            'Carpet Plot — Temperatura Esterna',
            'Andamento della temperatura esterna nel corso della stagione. '
            'Consente di correlare le prestazioni dell\'UTA con le condizioni climatiche '
            'esterne e di individuare i periodi di carico termico più elevato — '
            'tipicamente agosto per le temperature massime.',
        ),
        (
            f'distributions_and_scatter/{FOLDER_ID}_interpolated/scatter_1_TM_vs_SP_TM.png',
            'Scatter — Temperatura di Mandata vs Setpoint',
            'Ogni punto è una misura reale: asse X = setpoint, asse Y = temperatura di mandata. '
            'La linea diagonale è solo la linea di riferimento ideale (mandata = setpoint), non una curva climatica.',
        ),
        (
            f'descriptive_statistics_analysis/{FOLDER_ID}/box_plots_{FOLDER_ID}.png',
            'Box Plot — Tutte le Variabili Monitorate',
            'Confronta in modo semplice la variabilità delle principali variabili dell\'impianto. '
            'Le scatole più alte indicano variabili più instabili e quindi più sensibili da monitorare.',
        ),
    ]

    for idx, (p_path, p_title, p_desc) in enumerate(_PIPELINE_OVERVIEW):
        hdr_col = _overview_headers[idx % len(_overview_headers)]
        photo_card(
            img_path=plots_root / p_path,
            title=p_title,
            description=p_desc,
            header_bg=hdr_col,
            width_inches=6.0,
        )

    doc.add_page_break()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 4 — RIFERIMENTO AI GRAFICI DIAGNOSTICI
    # ════════════════════════════════════════════════════════════════════════
    heading('4. Grafici Diagnostici — Riferimento al Catalogo')
    body(
        'Per mantenere questo report semplice, i dettagli tecnici dei 14 grafici sono in appendice (catalogo grafici). '
        'Qui usiamo solo i messaggi utili alla decisione: cosa sta funzionando, cosa va verificato, e quali priorità aprire in riunione.'
    )
    callout_box(
        f'Focus riunione: confrontare la settimana "normale" ({normal_week.strftime("%d %b %Y")}) '
        f'con la settimana "critica" ({crit_week.strftime("%d %b %Y")}) e definire 2–3 azioni pratiche.\n\n'
        f'Se servono approfondimenti tecnici, consultare il catalogo grafici in appendice.',
        bg='EBF3FB', border='2E74B5',
        label='🧭  Come usare questo report:', label_color=C_NAVY
    )
    body(
        'Indicazione pratica: aprire il catalogo solo se emerge una domanda specifica '
        '(ad esempio su una singola anomalia o su una fascia oraria particolare).'
    )
    gap()

    doc.add_page_break()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 5 — KPI SETTIMANALI
    # ════════════════════════════════════════════════════════════════════════
    heading('5. Indicatori di Prestazione Settimanali (KPI)')
    body(
        'Questa tabella confronta le settimane tra loro con una scala colori semplice: verde meglio, giallo intermedio, arancione peggio. '
        'Per una lettura più immediata giorno-per-giorno, usa anche la Sezione 6 (Top 10 giorni critici).'
    )

    # Colour-code the KPI table rows by MAE value
    try:
        mae_vals = t2['MAE Errore (°C)'].replace('—', np.nan).astype(float)
        mae_q33  = mae_vals.quantile(0.33)
        mae_q66  = mae_vals.quantile(0.66)
    except Exception:
        mae_q33, mae_q66 = 0.8, 1.5

    tbl_kpi = doc.add_table(rows=len(t2) + 1, cols=len(t2.columns))
    tbl_kpi.style = 'Table Grid'
    # Header row
    for j, col_name in enumerate(t2.columns):
        cell = tbl_kpi.rows[0].cells[j]
        _set_cell_bg(cell, '1F4E79')
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(str(col_name))
        r.font.bold      = True
        r.font.size      = Pt(9)
        r.font.color.rgb = C_WHITE
    # Data rows
    for i, row_data in enumerate(t2.itertuples(index=False)):
        try:
            mae_v = float(str(list(row_data)[t2.columns.get_loc('MAE Errore (°C)')]).replace('—', 'nan'))
        except Exception:
            mae_v = np.nan
        if np.isnan(mae_v):
            row_bg = 'F2F2F2'
        elif mae_v <= mae_q33:
            row_bg = 'E2EFDA'   # green — best
        elif mae_v <= mae_q66:
            row_bg = 'FFF2CC'   # amber — medium
        else:
            row_bg = 'FCE4D6'   # red-pink — worst
        for j, val in enumerate(row_data):
            cell = tbl_kpi.rows[i + 1].cells[j]
            _set_cell_bg(cell, row_bg)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(str(val))
            r.font.size = Pt(9)

    caption(
        f'Tabella 2 — KPI settimanali per le {len(t2)} settimane del periodo '
        f'({df.index[0].strftime("%d %b")}–{df.index[-1].strftime("%d %b %Y")}). '
        f'Verde = settimane migliori; giallo = intermedie; arancione = settimane da approfondire.'
    )

    best_week  = t2.loc[t2['MAE Errore (°C)'].replace('—', np.nan).astype(float).idxmin()]
    worst_week = t2.loc[t2['MAE Errore (°C)'].replace('—', np.nan).astype(float).idxmax()]

    callout_box(
        f'Settimana migliore: {best_week["Settimana (lunedì)"]} '
        f'(MAE = {best_week["MAE Errore (°C)"]} °C).  '
        f'Settimana più critica: {worst_week["Settimana (lunedì)"]} '
        f'(MAE = {worst_week["MAE Errore (°C)"]} °C).',
        bg='E2EFDA', border='2D6A4F',
        label='✅  Riepilogo settimanale:', label_color=C_GREEN
    )

    doc.add_page_break()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 6 — I 10 GIORNI PEGGIORI
    # ════════════════════════════════════════════════════════════════════════
    heading('6. I 10 Giorni con il Maggiore Errore di Controllo')
    body(
        'La tabella seguente elenca i 10 giorni in cui la qualità del controllo '
        'della temperatura di mandata è stata peggiore (MAE giornaliero più alto). '
        'Questi giorni rappresentano le situazioni più critiche e meritano '
        'un\'analisi approfondita: potrebbero essere correlati a condizioni meteo '
        'estreme, guasti temporanei, o problemi di taratura del controllore.'
    )

    callout_box(
        'I giorni elencati sono calcolati solo sui periodi con UTA accesa (AHU_ON=1). '
        'Un MAE elevato in una singola giornata può indicare un evento anomalo discreto '
        'oppure una deriva del controllore che si accumula nel tempo.',
        bg='FFF2CC', border='BF8600',
        label='⚠  Nota interpretativa:', label_color=C_AMBER
    )

    _df_to_word_table(doc, t3.reset_index().rename(columns={'index': '#'}),
                      header_color='C00000')
    caption('Tabella 3 — I 10 giorni con il MAE più elevato (solo periodi con UTA in funzione).')
    gap()

    doc.add_page_break()

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 7 — FEEDBACK DEL RESPONSABILE DELL'ENERGIA
    # ════════════════════════════════════════════════════════════════════════
    heading('7. Note e Feedback del Responsabile dell\'Energia')
    body(
        'Questa sezione è riservata alle osservazioni del responsabile dell\'energia. '
        'Per ciascuna domanda è disponibile uno spazio di scrittura. '
        'Il feedback verrà incorporato nella versione finale della tesi.',
        italic=True
    )
    gap()

    feedback_questions = [
        (
            'Rilevanza degli indicatori KPI rispetto alla gestione operativa quotidiana',
            'Gli indicatori MAE, P95 e "tempo in banda" sono utili? Mancano metriche importanti?'
        ),
        (
            'Interpretazione delle anomalie — I 10 giorni critici (Sezione 6)',
            'Riconosce questi giorni come problematici? Ci sono cause note (guasti, manutenzione)?'
        ),
        (
            'Qualità del controllo: il sistema soddisfa le aspettative operative?',
            f'MAE medio = {mae_overall:.2f}°C, tempo in banda ±0.5°C = {pct_in_band:.1f}%. È accettabile?'
        ),
        (
            'Periodi di fermo impianto noti (manutenzione, ferie, chiusure edificio)',
            'Indicare date approssimative di fermo pianificato o non pianificato nel periodo analizzato.'
        ),
        (
            'Dati aggiuntivi utili per analisi future',
            'Es: portate d\'aria, consumi energia elettrica, misure di CO₂, occupazione edificio.'
        ),
        (
            'Possibili azioni correttive identificate',
            'Es: revisione setpoint, riprogrammazione orari, taratura sensori, manutenzione valvole.'
        ),
        (
            'Autorizzazione all\'utilizzo dei dati nella tesi di laurea',
            'I dati e i grafici possono essere inclusi nella tesi (con possibile anonimizzazione)?'
        ),
    ]

    for q_num, (question, hint) in enumerate(feedback_questions, start=1):
        feedback_block(question, q_num, hint=hint, lines=4)

    # ── Footer page ───────────────────────────────────────────────────────────
    doc.add_page_break()
    gap(3)
    fp = doc.add_paragraph()
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _add_horiz_rule(fp)
    footer_txt = doc.add_paragraph()
    footer_txt.alignment = WD_ALIGN_PARAGRAPH.CENTER
    ftr = footer_txt.add_run(
        f'Università degli Studi di Torino — Campus Luigi Einaudi — Edificio {BUILDING} · {AHU}\n'
        f'Varaga Haghoubians — MSc Data Science (Stochastics and Data Science) — UniTO'
    )
    ftr.font.size      = Pt(9)
    ftr.font.italic    = True
    ftr.font.color.rgb = C_LGREY

    # ── Save ─────────────────────────────────────────────────────────────────
    docx_path = out_tables / f'Report_EnergyManager_{FOLDER_ID}.docx'
    try:
        doc.save(docx_path)
    except PermissionError:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        docx_path = out_tables / f'Report_EnergyManager_{FOLDER_ID}_{ts}.docx'
        doc.save(docx_path)
        print('   ⚠️  File report in uso: creato file con timestamp.')
    print(f"   ✅ Report Word salvato: {docx_path}")
    return docx_path


if __name__ == '__main__':
    main()
