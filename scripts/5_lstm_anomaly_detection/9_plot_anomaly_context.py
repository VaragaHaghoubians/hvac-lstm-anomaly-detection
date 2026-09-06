"""
plot_anomaly_context.py
=======================
Plots ±N timestamps (default ±10 = ±5 hours) around each detected anomaly.

For each flagged window shows:
  Panel 1 — raw sensor values (TM, TR, TS) in physical units (°C)
  Panel 2 — reconstruction error vs dynamic threshold, anomaly marker
             + per-feature errors (one line per feature)

Usage
-----
Edit the four paths at the top of the CONFIGURATION block, then run:

    python plot_anomaly_context.py

The script auto-discovers the most recent anomaly_results_*.csv in the
processed_data folder — no manual filename needed.

Output
------
Saved to:  plots/lstm_autoencoder/{building}_{unit}_{season}{year}/anomaly_context/
  anomaly_context_01_2026-02-12T14-30-00.png
  anomaly_context_02_2026-02-13T08-00-00.png
  ...
"""

from pathlib import Path
import configparser
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# =============================================================================
# CONFIGURATION — edit these paths if needed
# =============================================================================
CONFIG_PATH = Path(__file__).parent.parent.parent / "config.ini"

# Set to "" to auto-select the most recent results CSV
RESULTS_CSV = ""

# Set to "" to auto-select the most recent interpolated feature CSV
FEATURE_CSV = ""

# Context window: ±N timestamps around each anomaly  (96 × 30 min = ±48 hours)
N_CONTEXT = 144

# Show (and save) a figure for every flagged window?
# Set to False to show only windows where error is among the top 10 % of anomalies
SHOW_ALL_ANOMALIES = True
# If True, crop both panels to the common timestamp overlap between
# feature data (top panel) and results/error data (bottom panel).
# This removes large blank zones when one source has missing blocks.
ALIGN_TO_COMMON_OVERLAP = True
# =============================================================================

def load_config():
    cfg = configparser.ConfigParser(
        interpolation=configparser.BasicInterpolation(),
        inline_comment_prefixes=(';', '#'),
    )
    try:
        read_ok = cfg.read(CONFIG_PATH, encoding='utf-8-sig')
    except configparser.Error as e:
        raise RuntimeError(
            f"Failed to parse config file: {CONFIG_PATH}\n"
            "Hint: ensure config.ini is a valid INI and encoded UTF-8 BOM/UTF-8.\n"
            f"Parser error: {e}"
        ) from e
    if not read_ok:
        raise FileNotFoundError(f"Configuration file not found or unreadable: {CONFIG_PATH}")

    for section in ("global", "paths", "lstm_autoencoder"):
        if not cfg.has_section(section):
            raise RuntimeError(
                f"Missing required section [{section}] in {CONFIG_PATH}\n"
                "Hint: ensure config.ini is a valid INI and encoded UTF-8 BOM/UTF-8."
            )
    return cfg


def resolve_csv(cfg, kind: str) -> Path:
    """Auto-resolve results or feature CSV from config paths."""
    building = cfg.get("global", "building_id")
    ahu      = cfg.get("global", "ahu_unit")
    season   = cfg.get("global", "season")
    year     = cfg.get("global", "year")
    folder   = f"{building}_{ahu}_{season}{year}"
    root     = Path(CONFIG_PATH).parent

    if kind == "results":
        base = root / cfg.get("paths", "processed_folder", fallback="./processed_data").lstrip("./")
        search_dir = base / "lstm_autoencoder" / folder
        candidates = sorted(search_dir.glob("anomaly_results_*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No anomaly_results_*.csv found in {search_dir}")
        return candidates[-1]   # most recent

    elif kind == "features":
        base = root / cfg.get("paths", "processed_folder", fallback="./processed_data").lstrip("./")
        # Feature engineering folder convention
        for candidate in [
            base / folder / f"{folder}_features.csv",
            base / "feature_engineering" / folder / f"{folder}_features.csv",
        ]:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Feature CSV not found. "
            f"Try setting FEATURE_CSV manually in this script."
        )


def load_data(cfg):
    results_path = Path(RESULTS_CSV) if RESULTS_CSV else resolve_csv(cfg, "results")
    feature_path = Path(FEATURE_CSV) if FEATURE_CSV else resolve_csv(cfg, "features")

    print(f"   Results CSV : {results_path}")
    print(f"   Feature CSV : {feature_path}")

    res  = pd.read_csv(results_path,  parse_dates=["Timestamp"])
    feat = pd.read_csv(feature_path,  parse_dates=["Time"])
    feat = feat.rename(columns={"Time": "Timestamp"})

    res  = res.sort_values("Timestamp").reset_index(drop=True)
    feat = feat.sort_values("Timestamp").reset_index(drop=True)
    return res, feat


def get_feature_names(cfg):
    season = cfg.get("global", "season", fallback="Summer").lower()
    key    = f"features_{season}"
    raw    = cfg.get("lstm_autoencoder", key,
                     fallback="Temperatura Mandata,Temperatura Ripresa,Temperatura Saturazione")
    return [f.strip() for f in raw.split(",")]


def get_context_windows_from_config(cfg, default_n=N_CONTEXT):
    """
    Resolve ±N windows from config hours, fallback to default_n.
    """
    sampling_minutes = cfg.getint("lstm_autoencoder", "sampling_interval_minutes", fallback=30)
    h_before = cfg.getint("lstm_autoencoder", "context_hours_before", fallback=72)
    h_after = cfg.getint("lstm_autoencoder", "context_hours_after", fallback=72)
    if h_before == h_after:
        n = int((h_before * 60) / sampling_minutes)
        return max(1, n)
    return default_n


def safe_filename(ts) -> str:
    return str(ts).replace(":", "-").replace(" ", "T")[:22]


_ITALIAN_DAYS = ["Lunedì", "Martedì", "Mercoledì", "Giovedì", "Venerdì", "Sabato", "Domenica"]


def plot_context(i, ts, res, feat, feature_names, out_dir, n=N_CONTEXT):
    """Draw and save the two-panel context figure for one anomaly timestamp."""

    # ── locate the anomaly row in res ──────────────────────────────────────
    res_idx = res.index[res["Timestamp"] == ts]
    if len(res_idx) == 0:
        print(f"   [skip] {ts} not found in results CSV")
        return
    # IMPORTANT: do NOT slice by row index here.
    # With stratified-weekly splits, results CSV contains disjoint test weeks;
    # row slicing can jump across weeks and create a fake "continuous" context.
    # Slice by timestamp window instead.
    res_idx = res_idx[0]
    # Use the median step in results (fallback to 30 min) to avoid hardcoding.
    # This also makes context windows robust to missing blocks / irregular sampling.
    diffs = res["Timestamp"].diff().dropna()
    if not diffs.empty:
        freq = diffs.median()
    else:
        freq = pd.Timedelta(minutes=30)
    # n=144 at 30-min sampling -> ±72h
    half_window = freq * n
    window_start = pd.Timestamp(ts) - half_window
    window_end = pd.Timestamp(ts) + half_window
    rwin = res[
        (res["Timestamp"] >= window_start)
        & (res["Timestamp"] <= window_end)
    ].copy()
    if rwin.empty:
        print(f"   [skip] no result rows within ±{half_window} of {ts}")
        return

    # ── locate the feature rows around ts ──────────────────────────────────
    # IMPORTANT: slice by timestamp window (not by row index).
    # Row-based slicing breaks whenever the feature file has gaps / disjoint blocks.
    fwin = feat[
        (feat["Timestamp"] >= window_start)
        & (feat["Timestamp"] <= window_end)
    ].copy()

    # ── figure ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    day_name = _ITALIAN_DAYS[pd.Timestamp(ts).weekday()]
    fig.suptitle(
        f"Anomaly context #{i:02d}   |   {day_name} {ts}   (±{n} windows ≈ ±{(half_window / pd.Timedelta(hours=1)):.0f}h)",
        fontsize=13, fontweight="bold"
    )

    # Panel 1 — raw sensor values
    ax = axes[0]
    colors = ["steelblue", "tomato", "seagreen"]
    for feat_name, color in zip(feature_names, colors):
        if feat_name in fwin.columns:
            ax.plot(fwin["Timestamp"], fwin[feat_name], label=feat_name,
                    color=color, linewidth=1.5)
    ax.axvline(ts, color="black", linestyle="--", linewidth=1.5, label="Anomaly")
    ax.set_ylabel("°C", fontsize=11)
    ax.set_title("Sensor values (physical units)", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.tick_params(axis="x", rotation=25, labelbottom=True)
    ax.grid(True, alpha=0.3)

    # Panel 2 — reconstruction error + threshold + per-feature errors
    ax = axes[1]

    # overall reconstruction error (blue)
    ax.plot(rwin["Timestamp"], rwin["Reconstruction_Error"],
            color="steelblue", linewidth=1.2, alpha=0.8, label="Reconstruction Error")

    # dynamic threshold — plot as a moving line if it varies per step, else axhline
    if "Threshold" in rwin.columns:
        thr_vals = rwin["Threshold"].values
        if thr_vals.std() > 1e-9:   # EMA mode: per-step values differ
            ax.plot(rwin["Timestamp"], thr_vals, color="black", linestyle="--",
                    linewidth=1.5, label="EMA Threshold")
        else:                        # static mode: flat line
            ax.axhline(thr_vals[0], color="black", linestyle="--",
                       linewidth=1.5, label=f"Threshold = {thr_vals[0]:.4f}")

    # per-feature errors (dashed, smaller)
    feat_colors = ["#1f77b4", "#d62728", "#2ca02c"]
    for feat_name, fc in zip(feature_names, feat_colors):
        col = f"{feat_name}_Error"
        if col in rwin.columns:
            ax.plot(rwin["Timestamp"], rwin[col], linestyle=":",
                    linewidth=1.0, color=fc, alpha=0.7,
                    label=f"{feat_name} error")

    # anomaly marker
    anom_rows = rwin[rwin["Is_Anomaly"] == True]
    if not anom_rows.empty:
        ax.scatter(anom_rows["Timestamp"], anom_rows["Reconstruction_Error"],
                   color="red", s=100, zorder=5, marker="o",
                   edgecolors="darkred", linewidths=0.8, label="Anomaly")

    # highlight the central flagged window
    ax.axvline(ts, color="black", linestyle="--", linewidth=1.5)

    ax.set_ylabel("MAE", fontsize=11)
    ax.set_title("Reconstruction error vs threshold", fontsize=11)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, alpha=0.3)

    # Force identical x-range on both panels.
    # By default, use common overlap to avoid blank areas where only one panel has data.
    if ALIGN_TO_COMMON_OVERLAP and (not fwin.empty) and (not rwin.empty):
        common_start = max(fwin["Timestamp"].min(), rwin["Timestamp"].min())
        common_end = min(fwin["Timestamp"].max(), rwin["Timestamp"].max())
        if common_start < common_end:
            axes[1].set_xlim(common_start, common_end)
        else:
            axes[1].set_xlim(window_start, window_end)
    else:
        axes[1].set_xlim(window_start, window_end)

    plt.tight_layout()

    fname = out_dir / f"anomaly_context_{i:02d}_{safe_filename(ts)}.png"
    plt.savefig(fname, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"   ✓ Saved: {fname.name}")


def main():
    print("=" * 60)
    print("  ANOMALY CONTEXT PLOTTER")
    print("=" * 60)

    cfg = load_config()
    res, feat = load_data(cfg)
    feature_names = get_feature_names(cfg)
    n_context = get_context_windows_from_config(cfg, default_n=N_CONTEXT)

    # Output directory
    building = cfg.get("global", "building_id")
    ahu      = cfg.get("global", "ahu_unit")
    season   = cfg.get("global", "season")
    year     = cfg.get("global", "year")
    plots_root = Path(CONFIG_PATH).parent / cfg.get("paths", "plots_folder",
                                                     fallback="./plots").lstrip("./")
    out_dir = plots_root / "lstm_autoencoder" / f"{building}_{ahu}_{season}{year}" / "anomaly_context"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clear stale plots from previous runs so numbering stays consistent
    for old_file in sorted(out_dir.glob("anomaly_context_*.png")):
        old_file.unlink()

    # Get anomaly timestamps (after persistence filter)
    anom_ts = res.loc[res["Is_Anomaly"] == True, "Timestamp"].unique()
    print(f"\n   Found {len(anom_ts)} anomaly timestamps")

    if len(anom_ts) == 0:
        print("   No anomalies to plot. Done.")
        return

    # Optional: filter to only the most extreme anomalies
    if not SHOW_ALL_ANOMALIES:
        top_cutoff = res["Reconstruction_Error"].quantile(0.90)
        anom_ts = res.loc[
            (res["Is_Anomaly"] == True) & (res["Reconstruction_Error"] >= top_cutoff),
            "Timestamp"
        ].unique()
        print(f"   After top-10% filter: {len(anom_ts)} anomalies to plot")

    for i, ts in enumerate(sorted(anom_ts), start=1):
        plot_context(i, ts, res, feat, feature_names, out_dir, n=n_context)

    print(f"\n   All context plots saved to:\n   {out_dir}")


def run_from_dataframes(df_results, feat_df, feature_names, out_dir, n=N_CONTEXT,
                          show_all=SHOW_ALL_ANOMALIES, config=None):
    """
    Entry point for 1_main.py — uses already-loaded DataFrames instead of
    re-reading from disk.

    Parameters
    ----------
    df_results     : pd.DataFrame  — output of save_results_to_csv()
                     must have columns: Timestamp, Reconstruction_Error,
                     Is_Anomaly, Threshold, {feature}_Error, ...
    feat_df        : pd.DataFrame  — raw (unscaled) feature values
                     must have a 'Timestamp' column (or DatetimeIndex reset)
    feature_names  : list[str]     — e.g. ['Temperatura Mandata', ...]
    out_dir        : pathlib.Path  — where to save the PNG files
    n              : int           — ±n windows around each anomaly
    show_all       : bool          — if False, only top-10 % errors
    config         : ConfigParser  — optional, resolve n from context_hours_* keys
    """
    if config is not None:
        n = get_context_windows_from_config(config, default_n=n)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clear stale plots from previous runs so numbering stays consistent
    for old_file in sorted(out_dir.glob("anomaly_context_*.png")):
        old_file.unlink()

    # Normalise feature dataframe — ensure 'Timestamp' column exists
    feat = feat_df.copy()
    if "Timestamp" not in feat.columns:
        if feat.index.dtype == "datetime64[ns]" or hasattr(feat.index, "year"):
            feat = feat.reset_index().rename(columns={feat.index.name or "index": "Timestamp"})
        else:
            raise ValueError("feat_df must have a 'Timestamp' column or DatetimeIndex")

    res = df_results.copy()
    res = res.sort_values("Timestamp").reset_index(drop=True)
    feat = feat.sort_values("Timestamp").reset_index(drop=True)

    anom_ts = res.loc[res["Is_Anomaly"] == True, "Timestamp"].unique()
    print(f"   Found {len(anom_ts)} anomaly timestamps")

    if len(anom_ts) == 0:
        print("   No anomalies to plot. Skipping context plots.")
        return

    if not show_all:
        cutoff = res["Reconstruction_Error"].quantile(0.90)
        anom_ts = res.loc[
            (res["Is_Anomaly"] == True) & (res["Reconstruction_Error"] >= cutoff),
            "Timestamp"
        ].unique()
        print(f"   After top-10 % filter: {len(anom_ts)} anomalies to plot")

    for i, ts in enumerate(sorted(anom_ts), start=1):
        plot_context(i, ts, res, feat, feature_names, out_dir, n=n)

    print(f"   Context plots saved to: {out_dir}")


if __name__ == "__main__":
    main()

