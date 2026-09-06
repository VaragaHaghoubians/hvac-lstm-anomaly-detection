"""
RESTORE VERSION - LSTM Model Version Manager
=============================================================================

Usage:
  python restore_version.py                    # list all saved versions
  python restore_version.py --restore 3        # restore version #3 from the list
  python restore_version.py --restore v2_ewma  # restore by tag name (partial match)
  python restore_version.py --describe 3 "Good result, RMSE 0.63"

How it works:
  Every run of 1_main.py saves a timestamped copy of the trained model to:
      models/versions/<timestamp>_<tag>/model.keras
      models/versions/<timestamp>_<tag>/metadata.json

  The ACTIVE model (used by the next run) is:
      models/best_autoencoder_<building>_<ahu>_<season><year>.keras

  Restoring a version copies that version's model.keras back to the active path.
  The current active model is backed up first (as 'model_before_restore.keras').
=============================================================================
"""

import argparse
import configparser
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_project_root():
    """Walk up from this script's directory until we find config.ini."""
    d = Path(__file__).resolve()
    for _ in range(6):
        d = d.parent
        if (d / 'config.ini').exists():
            return d
    raise FileNotFoundError("config.ini not found in any parent directory.")


def get_paths(config):
    base_folder = config.get('paths', 'plots_folder', fallback='./plots')
    building_id = config.get('global', 'building_id', fallback='C1')
    ahu_unit    = config.get('global', 'ahu_unit',     fallback='UTA1')
    season      = config.get('global', 'season',       fallback='Summer')
    year        = config.get('global', 'year',         fallback='2025')

    root         = find_project_root()
    models_dir   = root / base_folder.replace('./', '') / 'lstm_autoencoder' / 'models'
    versions_dir = models_dir / 'versions'
    active_name  = f'best_autoencoder_{building_id}_{ahu_unit}_{season}{year}.keras'
    active_path  = models_dir / active_name
    return models_dir, versions_dir, active_path


def load_versions(versions_dir: Path):
    """
    Return list of dicts sorted by timestamp (oldest first).
    Scans version sub-folders and reads metadata.json from each.
    """
    if not versions_dir.exists():
        return []

    versions = []
    for d in sorted(versions_dir.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / 'metadata.json'
        model_path = d / 'model.keras'
        if not model_path.exists():
            continue  # incomplete version, skip

        meta = {}
        if meta_path.exists():
            try:
                with open(meta_path, encoding='utf-8') as fh:
                    meta = json.load(fh)
            except Exception:
                pass

        versions.append({
            'dir':       d,
            'name':      d.name,
            'tag':       meta.get('tag', d.name),
            'timestamp': meta.get('timestamp', ''),
            'meta':      meta,
        })

    return versions


def fmt_ts(ts: str) -> str:
    """20260226_164730 -> 2026-02-26 16:47"""
    try:
        dt = datetime.strptime(ts, '%Y%m%d_%H%M%S')
        return dt.strftime('%Y-%m-%d %H:%M')
    except Exception:
        return ts[:16]


def print_table(versions, active_path: Path):
    """Print a formatted table of all versions."""
    if not versions:
        print("  No versions saved yet. Run 1_main.py to create one.")
        return

    # Determine active model name by reading its source if possible
    active_name = active_path.name if active_path.exists() else '(none)'

    print()
    print(f"  {'#':<4} {'Tag':<30} {'Date':<17} {'Val Loss':<10} {'RMSE(C)':<9} {'MAE(C)':<8} {'Anom%':<7} {'Description'}")
    print(f"  {'-'*4} {'-'*30} {'-'*17} {'-'*10} {'-'*9} {'-'*8} {'-'*7} {'-'*30}")

    for i, v in enumerate(versions, start=1):
        m        = v['meta']
        training = m.get('training', {})
        ev       = m.get('eval', {})
        desc     = m.get('description', '')

        val_loss = training.get('best_val_loss', '')
        rmse     = ev.get('rmse_overall_degC', '')
        mae      = ev.get('mae_overall_degC', '')
        anom     = ev.get('anomaly_rate_pct', '')

        val_str  = f"{val_loss:.5f}" if isinstance(val_loss, float) else '-'
        rmse_str = f"{rmse:.4f}"     if isinstance(rmse, float)     else '-'
        mae_str  = f"{mae:.4f}"      if isinstance(mae, float)      else '-'
        anom_str = f"{anom:.1f}%"    if isinstance(anom, float)     else '-'

        marker = ' <-- ACTIVE' if _is_active(v, active_path) else ''
        print(f"  {i:<4} {v['tag']:<30} {fmt_ts(v['timestamp']):<17} "
              f"{val_str:<10} {rmse_str:<9} {mae_str:<8} {anom_str:<7} {desc}{marker}")

    print()


def _is_active(v: dict, active_path: Path) -> bool:
    """Check if this version is the currently active model (best-guess by mtime)."""
    try:
        version_model = v['dir'] / 'model.keras'
        if not active_path.exists() or not version_model.exists():
            return False
        # Compare file size as a lightweight check (mtime can differ after copy)
        return active_path.stat().st_size == version_model.stat().st_size
    except Exception:
        return False


def find_version(versions, selector: str):
    """
    Find a version by integer index (1-based) or partial tag match.
    Returns the version dict or None.
    """
    # Try integer index
    try:
        idx = int(selector) - 1
        if 0 <= idx < len(versions):
            return versions[idx]
        print(f"  Error: index {selector} out of range (1-{len(versions)}).")
        return None
    except ValueError:
        pass

    # Try partial tag match
    matches = [v for v in versions if selector.lower() in v['tag'].lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"  Error: '{selector}' matches {len(matches)} versions. Be more specific:")
        for v in matches:
            print(f"    - {v['tag']}")
        return None
    print(f"  Error: no version matches '{selector}'.")
    return None


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_list(versions, active_path):
    print("\n  === Saved Model Versions ===")
    print_table(versions, active_path)


def cmd_restore(versions, active_path, selector: str):
    v = find_version(versions, selector)
    if v is None:
        sys.exit(1)

    src = v['dir'] / 'model.keras'
    if not src.exists():
        print(f"  Error: model.keras missing in {v['name']}.")
        sys.exit(1)

    # Back up the current active model first
    if active_path.exists():
        backup_path = active_path.parent / 'model_before_restore.keras'
        shutil.copy2(active_path, backup_path)
        print(f"  Current active model backed up -> model_before_restore.keras")

    shutil.copy2(src, active_path)
    print(f"\n  Restored: {v['name']}")
    print(f"  Active model is now: {active_path.name}")
    meta = v['meta']
    ev   = meta.get('eval', {})
    tr   = meta.get('training', {})
    print(f"\n  Version details:")
    print(f"    Tag          : {v['tag']}")
    print(f"    Date         : {fmt_ts(v['timestamp'])}")
    print(f"    Best val loss: {tr.get('best_val_loss', '-')}")
    print(f"    RMSE         : {ev.get('rmse_overall_degC', '-')} deg C")
    print(f"    MAE          : {ev.get('mae_overall_degC',  '-')} deg C")
    print(f"    Anomaly rate : {ev.get('anomaly_rate_pct',  '-')} %")
    cfg = meta.get('config', {})
    print(f"    Key config   : encoder={cfg.get('encoder_units')}, "
          f"bottleneck={cfg.get('bottleneck_units')}, "
          f"lr={cfg.get('learning_rate')}, "
          f"ewma={cfg.get('apply_ewma_smoothing')}")
    print()


def cmd_describe(versions, selector: str, description: str):
    v = find_version(versions, selector)
    if v is None:
        sys.exit(1)
    meta_path = v['dir'] / 'metadata.json'
    try:
        with open(meta_path, encoding='utf-8') as fh:
            meta = json.load(fh)
        meta['description'] = description
        with open(meta_path, 'w', encoding='utf-8') as fh:
            json.dump(meta, fh, indent=2)
        print(f"  Description saved for version '{v['name']}'.")
    except Exception as e:
        print(f"  Error updating metadata: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='LSTM Model Version Manager',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--restore',  metavar='VERSION',
                        help='Restore a version by number or tag')
    parser.add_argument('--describe', nargs=2, metavar=('VERSION', 'TEXT'),
                        help='Add a description to a version')
    args = parser.parse_args()

    # Load config
    root = find_project_root()
    config = configparser.ConfigParser()
    config.read(root / 'config.ini', encoding='utf-8-sig')  # utf-8-sig strips BOM (\ufeff)

    models_dir, versions_dir, active_path = get_paths(config)
    versions = load_versions(versions_dir)

    if args.restore:
        cmd_restore(versions, active_path, args.restore)
    elif args.describe:
        cmd_describe(versions, args.describe[0], args.describe[1])
    else:
        cmd_list(versions, active_path)


if __name__ == '__main__':
    main()
