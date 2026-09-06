"""
Synthetic sample-data generator for the HVAC analysis pipeline.

The real Building Management System (BMS) data used in the thesis / Eurix
collaboration is confidential and is NOT included in this repository.
This script generates physically plausible synthetic data with the exact
same file format, column layout, and quirks as the real BMS export
(UTF-8 BOM, Excel `sep=,` hint line, values suffixed with ` °C`,
irregular gaps, sensor spikes), so every script in the pipeline runs
out of the box.

Generated files (written next to this script):
  - C1_UTA1_Summer2025.csv                 AHU sensor data, cooling season
  - C1_UTA1_Winter2026.csv                 AHU sensor data, heating season
  - LIMF_from_2025-07-01_to_2026-02-18.csv Weather station data (30-min)

Injected events (so the analysis has something to find):
  - Random multi-hour data gaps in each season
  - Occasional sensor spike outliers (~85 °C glitches)
  - A 36-hour heating-coil fault around 2026-02-12 (supply temperature
    collapses while fan modulation stays high) — mirrors the kind of
    critical anomaly the LSTM autoencoder flagged on the real data.

Usage:
    python generate_sample_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
RNG = np.random.default_rng(SEED)
OUT_DIR = Path(__file__).resolve().parent

SUMMER_START, SUMMER_END = "2025-07-09 00:00", "2025-10-14 23:30"
WINTER_START, WINTER_END = "2025-10-15 00:00", "2026-02-18 23:30"
WEATHER_START, WEATHER_END = "2025-07-01 00:00", "2026-02-18 23:30"


def ar1_noise(n, sigma, phi=0.9):
    """Smooth autocorrelated noise (AR(1)) — sensor readings drift, they don't jitter."""
    out = np.zeros(n)
    eps = RNG.normal(0, sigma, n)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + eps[i]
    return out


def outdoor_temperature(index):
    """Seasonal + diurnal outdoor temperature for Turin, Italy."""
    day_of_year = index.dayofyear.to_numpy()
    hour = index.hour.to_numpy() + index.minute.to_numpy() / 60.0
    seasonal = 13.5 + 11.5 * np.sin(2 * np.pi * (day_of_year - 105) / 365.0)
    diurnal = 4.5 * np.sin(2 * np.pi * (hour - 9) / 24.0)
    return seasonal + diurnal + ar1_noise(len(index), 0.6)


def occupancy_mask(index):
    """Fan runs Mon-Fri 06:00-20:00, Sat 07:00-14:00, off Sundays."""
    dow = index.dayofweek.to_numpy()
    hour = index.hour.to_numpy()
    weekday = (dow < 5) & (hour >= 6) & (hour < 20)
    saturday = (dow == 5) & (hour >= 7) & (hour < 14)
    return weekday | saturday


def build_season(index, mode, t_out):
    """Simulate one season of AHU operation. mode: 'cooling' or 'heating'."""
    n = len(index)
    occupied = occupancy_mask(index)

    if mode == "cooling":
        sp = np.full(n, 21.0)
        # Weather-compensated setpoint: slightly higher when very hot outside
        sp_comp = sp + np.clip((t_out - 28.0) * 0.15, 0, 1.5)
        # Supply air tracks the compensated setpoint when the fan is on
        supply = sp_comp + ar1_noise(n, 0.35) + 0.04 * np.clip(t_out - 26, 0, None)
        # Return air follows building load and outdoor conditions
        ret = 23.5 + 0.18 * (t_out - 25.0) + 0.8 * occupied + ar1_noise(n, 0.30)
        # Cooling-coil saturation temperature sits below supply, coupled to outdoor
        sat = supply - 4.5 - 0.10 * (t_out - 25.0) + ar1_noise(n, 0.45)
        mod_supply_on = 55 + 18 * np.clip((t_out - 24.0) / 8.0, 0, 1)
    else:
        sp = np.full(n, 28.0)
        sp_comp = sp + np.clip((8.0 - t_out) * 0.20, 0, 3.0)
        # Heating supply oscillates with coil modulation cycles (~3 h period)
        cycle = 2.8 * np.sin(2 * np.pi * np.arange(n) / 6.0)
        supply = sp_comp + cycle + ar1_noise(n, 0.40)
        ret = 20.8 + 0.9 * occupied + ar1_noise(n, 0.25)
        sat = supply + 2.0 + ar1_noise(n, 0.50)
        mod_supply_on = 60 + 20 * np.clip((10.0 - t_out) / 12.0, 0, 1)

    # When the AHU is off, temperatures drift toward the return/ambient level
    drift_target = ret + 0.3 * (t_out - ret)
    supply = np.where(occupied, supply, drift_target + ar1_noise(n, 0.25))
    sat = np.where(occupied, sat, drift_target + ar1_noise(n, 0.30))

    mod_supply = np.where(occupied, mod_supply_on + RNG.normal(0, 3, n), 0)
    mod_return = np.where(occupied, mod_supply - 10 + RNG.normal(0, 2, n), 0)
    mod_supply = np.clip(np.round(mod_supply), 0, 100).astype(int)
    mod_return = np.clip(np.round(mod_return), 0, 100).astype(int)

    df = pd.DataFrame(
        {
            "Time": index.strftime("%Y-%m-%d %H:%M:%S"),
            "Temp. Mandata": supply,
            "Temp. Ripresa": ret,
            "Temp. Saturazione": sat,
            "Temp. Esterna": t_out,
            "SP Temp. Mand.": sp,
            "SP Temp. Mand. Comp.": sp_comp,
            "Mod. V. Mandata": mod_supply,
            "Mod. V. Ripresa": mod_return,
        }
    )
    return df


def inject_faults(df, n_gaps, n_spikes, coil_fault=None):
    """Add realistic data-quality problems and (optionally) a physical fault."""
    # Heating-coil fault: supply temperature collapses while the fan keeps running
    if coil_fault is not None:
        start, hours = coil_fault
        i0 = df.index[df["Time"] >= start][0]
        i1 = min(i0 + hours * 2, len(df) - 1)  # 30-min steps
        idx = np.arange(i0, i1)
        df.loc[idx, "Temp. Mandata"] = 16.0 + ar1_noise(len(idx), 0.5)
        df.loc[idx, "Temp. Saturazione"] = 14.0 + ar1_noise(len(idx), 0.6)

    # Sensor spike outliers (electrical glitches)
    spike_rows = RNG.choice(len(df), size=n_spikes, replace=False)
    spike_cols = RNG.choice(
        ["Temp. Mandata", "Temp. Ripresa", "Temp. Saturazione"], size=n_spikes
    )
    for row, col in zip(spike_rows, spike_cols):
        df.loc[row, col] = float(RNG.choice([85.0, -20.0, 99.9]))

    # Multi-hour transmission gaps (rows simply missing from the export)
    drop = []
    for _ in range(n_gaps):
        g0 = int(RNG.integers(0, len(df) - 30))
        drop.extend(range(g0, g0 + int(RNG.integers(4, 25))))
    return df.drop(index=[i for i in set(drop) if i < len(df)]).reset_index(drop=True)


def write_bms_csv(df, path):
    """Write in the exact BMS export format: BOM, sep hint, quoted header, ` °C` units."""
    temp_cols = ["Temp. Mandata", "Temp. Ripresa", "Temp. Saturazione", "Temp. Esterna"]
    out = df.copy()
    for col in temp_cols:
        out[col] = out[col].round(1).map(lambda v: f"{v:.1f} °C")
    out["SP Temp. Mand."] = out["SP Temp. Mand."].map(lambda v: f"{v:.0f} °C")
    out["SP Temp. Mand. Comp."] = out["SP Temp. Mand. Comp."].round(1).map(
        lambda v: f"{v:.1f} °C"
    )
    header = ",".join(f'"{c}"' for c in out.columns)
    with open(path, "w", encoding="utf-8-sig", newline="\n") as f:
        f.write("sep=,\n")
        f.write(header + "\n")
        for row in out.itertuples(index=False):
            f.write(",".join(str(v) for v in row) + "\n")
    print(f"  wrote {path.name}  ({len(out):,} rows)")


def write_weather_csv(path):
    """Weather station export: local-time timestamps with explicit UTC offset."""
    index = pd.date_range(WEATHER_START, WEATHER_END, freq="30min")
    t_out = outdoor_temperature(index)
    rh = np.clip(70 - 1.8 * (t_out - 15) + ar1_noise(len(index), 3.0), 20, 100)
    # Europe/Rome: CEST (+02:00) until 2025-10-26 03:00, then CET (+01:00)
    dst_end = pd.Timestamp("2025-10-26 03:00:00")
    offsets = np.where(index < dst_end, "+02:00", "+01:00")
    df = pd.DataFrame(
        {
            "timestamp": [
                f"{ts.strftime('%Y-%m-%d %H:%M:%S')}{off}"
                for ts, off in zip(index, offsets)
            ],
            "temp": np.round(t_out).astype(int),
            "rh": np.round(rh).astype(int),
        }
    )
    df.to_csv(path, index=False, lineterminator="\n")
    print(f"  wrote {path.name}  ({len(df):,} rows)")


def main():
    print("Generating synthetic HVAC sample data (seed=42)...")

    summer_idx = pd.date_range(SUMMER_START, SUMMER_END, freq="30min")
    summer = build_season(summer_idx, "cooling", outdoor_temperature(summer_idx))
    summer = inject_faults(summer, n_gaps=5, n_spikes=8)
    write_bms_csv(summer, OUT_DIR / "C1_UTA1_Summer2025.csv")

    winter_idx = pd.date_range(WINTER_START, WINTER_END, freq="30min")
    winter = build_season(winter_idx, "heating", outdoor_temperature(winter_idx))
    winter = inject_faults(
        winter, n_gaps=6, n_spikes=8, coil_fault=("2026-02-12 06:00:00", 36)
    )
    write_bms_csv(winter, OUT_DIR / "C1_UTA1_Winter2026.csv")

    write_weather_csv(OUT_DIR / "LIMF_from_2025-07-01_to_2026-02-18.csv")
    print("Done. The pipeline can now be run end to end on this sample data.")


if __name__ == "__main__":
    main()
