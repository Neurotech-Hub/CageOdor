#!/usr/bin/env python3
"""Pretty raw time-series plots for BME688 cage-odor logs.

Keeps every reading row. Marker rows (label_change, profile_change, etc.)
are ignored — nothing is trimmed after a label swap.

By default, gas resistance and humidity are averaged across all sensors
in 1 s bins. Pass --ref-sensor N to plot one logical_id instead.

One scatter series per heater profile (face = heater_step via jet;
edge = protocol). Label spans are drawn as background bands; humidity
on the right y-axis.

Usage:
    python analysis/pretty_raw_plots.py path/to/bme688_log_YYYYMMDD_HHMMSS.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

LABELS = ("UNKNOWN", "CLEAN", "SOILED")
LABEL_COLORS = {
    "UNKNOWN": "#9e9e9e",
    "CLEAN": "#4c78a8",
    "SOILED": "#f58518",
}
LABEL_DISPLAY = {
    "UNKNOWN": "UNKNOWN (air)",
    "CLEAN": "CLEAN",
    "SOILED": "SOILED",
}
# Distinct edge colors so protocols stay readable under jet-filled markers.
PROFILE_LINE_COLORS = (
    "#1b1b1b",
    "#2a6f97",
    "#6a4c93",
    "#bc4749",
)
HUMIDITY_COLOR = "#000000"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pretty raw BME688 time series (no swap-window trimming)."
    )
    p.add_argument("csv", type=Path, help="Path to bme688_log_*.csv")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("analysis/out"),
        help="Output directory (default: analysis/out)",
    )
    p.add_argument(
        "--ref-sensor",
        type=int,
        default=None,
        help="Plot one logical_id only (default: mean across all sensors)",
    )
    p.add_argument(
        "--gap-sec",
        type=float,
        default=60.0,
        help="Break a profile line when samples are farther apart than this (default: 60)",
    )
    p.add_argument(
        "--bin-ms",
        type=float,
        default=1000.0,
        help="When averaging sensors, time-bin width in ms (default: 1000)",
    )
    p.add_argument(
        "--rh-smooth-sec",
        type=float,
        default=100.0,
        help="Centered rolling-mean window for humidity in seconds (default: 100)",
    )
    return p.parse_args()


def load_readings(path: Path) -> pd.DataFrame:
    """Load CSV and keep reading rows only (any non-empty marker is ignored)."""
    df = pd.read_csv(path, comment="#")
    needed = {
        "millis",
        "label_name",
        "marker",
        "logical_id",
        "profile_id",
        "profile_name",
        "heater_step",
        "gas_resistance_ohm",
        "humidity_pct",
    }
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"CSV is missing columns: {sorted(missing)}")

    marker = df["marker"]
    is_marker = marker.notna() & marker.astype(str).str.strip().ne("") & marker.astype(
        str
    ).str.lower().ne("nan")
    data = df.loc[~is_marker].copy()

    for c in (
        "millis",
        "logical_id",
        "profile_id",
        "heater_step",
        "gas_resistance_ohm",
        "humidity_pct",
    ):
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data["label_name"] = data["label_name"].astype(str).str.strip().str.upper()
    data["profile_name"] = data["profile_name"].astype(str).str.strip()

    data = data[
        data["label_name"].isin(LABELS)
        & data["millis"].notna()
        & data["logical_id"].notna()
        & data["heater_step"].between(0, 9)
        & data["gas_resistance_ohm"].gt(0)
    ].copy()
    data["logical_id"] = data["logical_id"].astype(int)
    data["heater_step"] = data["heater_step"].astype(int)
    data["log_r"] = np.log10(data["gas_resistance_ohm"])
    return data.sort_values("millis")


def label_spans(data: pd.DataFrame) -> list[tuple[str, float, float]]:
    """Contiguous (label, t0_ms, t1_ms) spans covering the full series."""
    if data.empty:
        return []
    d = data.sort_values("millis")
    labels = d["label_name"].to_numpy()
    millis = d["millis"].to_numpy(dtype=float)
    spans: list[tuple[str, float, float]] = []
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[start]:
            spans.append((str(labels[start]), float(millis[start]), float(millis[i - 1])))
            start = i
    spans.append((str(labels[start]), float(millis[start]), float(millis[-1])))
    return spans


def break_on_gaps(
    t: np.ndarray, y: np.ndarray, gap_min: float
) -> tuple[np.ndarray, np.ndarray]:
    """Insert NaNs so matplotlib does not draw across long inactive gaps."""
    if len(t) < 2:
        return t, y
    dt = np.diff(t)
    cuts = np.where(dt > gap_min)[0]
    if len(cuts) == 0:
        return t, y
    t_out: list[float] = []
    y_out: list[float] = []
    prev = 0
    for cut in cuts:
        t_out.extend(t[prev : cut + 1].tolist())
        y_out.extend(y[prev : cut + 1].tolist())
        t_out.append(np.nan)
        y_out.append(np.nan)
        prev = cut + 1
    t_out.extend(t[prev:].tolist())
    y_out.extend(y[prev:].tolist())
    return np.asarray(t_out, dtype=float), np.asarray(y_out, dtype=float)


def shade_labels(ax, spans: list[tuple[str, float, float]], t0: float) -> None:
    for lab, a, b in spans:
        ax.axvspan(
            (a - t0) / 60_000.0,
            (b - t0) / 60_000.0,
            facecolor=LABEL_COLORS.get(lab, "#cccccc"),
            edgecolor="none",
            alpha=0.18,
            zorder=0,
        )


def _mode_label(s: pd.Series) -> str:
    m = s.mode()
    return str(m.iloc[0] if len(m) else s.iloc[0])


def prepare_series(
    data: pd.DataFrame,
    ref_sensor: int | None,
    bin_ms: float,
) -> tuple[pd.DataFrame, str]:
    """Return plot series and a short sensor description for the title."""
    if ref_sensor is not None:
        sub = data[data["logical_id"] == ref_sensor].sort_values("millis")
        if sub.empty:
            raise SystemExit(f"No rows for logical_id={ref_sensor}")
        return sub, f"logical_id={ref_sensor}"

    d = data.copy()
    d["t_bin"] = (d["millis"] // bin_ms) * bin_ms
    sub = (
        d.groupby(
            ["t_bin", "profile_id", "profile_name", "heater_step"],
            as_index=False,
        )
        .agg(
            millis=("millis", "mean"),
            log_r=("log_r", "mean"),
            humidity_pct=("humidity_pct", "mean"),
            label_name=("label_name", _mode_label),
            n_sensors=("logical_id", "nunique"),
        )
        .sort_values("millis")
    )
    n = int(data["logical_id"].nunique())
    return sub, f"mean of {n} sensors"


def humidity_series(
    data: pd.DataFrame,
    ref_sensor: int | None,
    bin_ms: float,
    smooth_sec: float,
) -> pd.DataFrame:
    """Mean humidity per time bin, then centered rolling smooth."""
    src = data if ref_sensor is None else data[data["logical_id"] == ref_sensor]
    src = src.dropna(subset=["humidity_pct"]).copy()
    src["t_bin"] = (src["millis"] // bin_ms) * bin_ms
    rh = (
        src.groupby("t_bin", as_index=False)
        .agg(millis=("millis", "mean"), humidity_pct=("humidity_pct", "mean"))
        .sort_values("millis")
    )
    if rh.empty or smooth_sec <= 0 or bin_ms <= 0:
        return rh
    win = max(1, int(round(smooth_sec * 1000.0 / bin_ms)))
    if win % 2 == 0:
        win += 1
    rh["humidity_pct"] = (
        rh["humidity_pct"].rolling(window=win, center=True, min_periods=1).mean()
    )
    return rh


def plot_raw(
    data: pd.DataFrame,
    out: Path,
    ref_sensor: int | None,
    gap_sec: float,
    bin_ms: float,
    rh_smooth_sec: float,
) -> Path:
    sub, sensor_desc = prepare_series(data, ref_sensor, bin_ms)
    if sub.empty:
        raise SystemExit("No rows to plot after sensor selection/averaging")

    profiles = (
        sub.groupby(["profile_id", "profile_name"], sort=True)
        .size()
        .reset_index(name="n")
        .sort_values("profile_id")
    )
    if profiles.empty:
        raise SystemExit("No heater profiles found in readings")

    t0 = float(data["millis"].min())
    spans = label_spans(data)
    cmap = plt.get_cmap("jet")
    norm = Normalize(vmin=0, vmax=9)

    fig, ax = plt.subplots(figsize=(14, 6.5))
    shade_labels(ax, spans, t0)

    for i, row in enumerate(profiles.itertuples(index=False)):
        pid, pname = int(row.profile_id), str(row.profile_name)
        sl = sub[sub["profile_id"] == pid].sort_values("millis")
        if sl.empty:
            continue
        t_min = (sl["millis"].to_numpy(dtype=float) - t0) / 60_000.0
        y = sl["log_r"].to_numpy(dtype=float)
        edge = PROFILE_LINE_COLORS[i % len(PROFILE_LINE_COLORS)]
        ax.scatter(
            t_min,
            y,
            c=sl["heater_step"].to_numpy(),
            cmap=cmap,
            norm=norm,
            s=28,
            marker="o",
            edgecolors=edge,
            linewidths=0.45,
            zorder=3,
        )

    ax_rh = ax.twinx()
    rh = humidity_series(data, ref_sensor, bin_ms, rh_smooth_sec)
    if not rh.empty:
        t_rh = (rh["millis"].to_numpy(dtype=float) - t0) / 60_000.0
        y_rh = rh["humidity_pct"].to_numpy(dtype=float)
        ax_rh.plot(
            t_rh,
            y_rh,
            color=HUMIDITY_COLOR,
            linewidth=1.8,
            alpha=0.95,
            zorder=1,
            label="humidity",
        )
    ax_rh.set_ylabel("humidity (%RH)", color=HUMIDITY_COLOR)
    ax_rh.tick_params(axis="y", colors=HUMIDITY_COLOR)
    ax_rh.spines["right"].set_color(HUMIDITY_COLOR)

    ax.set_xlabel("minutes from first sample")
    ax.set_ylabel("log10(gas resistance / ohm)")
    fig.suptitle(
        f"Raw heater-sweep time series  ·  {sensor_desc}",
        y=0.98,
        fontsize=12,
    )

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=[ax, ax_rh],
        pad=0.08,
        fraction=0.03,
    )
    cbar.set_label("heater_step")
    cbar.set_ticks(range(10))

    protocol_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#888888",
            markeredgecolor=PROFILE_LINE_COLORS[i % len(PROFILE_LINE_COLORS)],
            markeredgewidth=1.2,
            markersize=7,
            linestyle="none",
        )
        for i in range(len(profiles))
    ]
    protocol_labels = [str(r.profile_name) for r in profiles.itertuples(index=False)]
    protocol_handles.append(Line2D([0], [0], color=HUMIDITY_COLOR, lw=1.8))
    protocol_labels.append("humidity")
    label_handles = [
        Patch(
            facecolor=LABEL_COLORS[lab],
            edgecolor="none",
            alpha=0.35,
            label=LABEL_DISPLAY[lab],
        )
        for lab in LABELS
        if any(s[0] == lab for s in spans)
    ]

    # Single legend row above the axes (outside the plot box, inside the figure).
    all_handles = protocol_handles + label_handles
    all_labels = protocol_labels + [
        f"bg: {LABEL_DISPLAY[lab]}"
        for lab in LABELS
        if any(s[0] == lab for s in spans)
    ]
    ax.legend(
        all_handles,
        all_labels,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=min(8, len(all_handles)),
        frameon=False,
        fontsize=8,
        borderaxespad=0.0,
        handlelength=1.6,
        columnspacing=1.0,
    )

    ax.set_axisbelow(False)
    for spine in ax.spines.values():
        spine.set_zorder(4)
    fig.subplots_adjust(left=0.07, right=0.82, top=0.82, bottom=0.10)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "pretty_raw_timeseries.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    print(f"Loading {args.csv}")
    data = load_readings(args.csv)
    if data.empty:
        raise SystemExit("No reading rows after ignoring markers")

    sensors = sorted(data["logical_id"].unique().tolist())
    if args.ref_sensor is not None and args.ref_sensor not in sensors:
        raise SystemExit(
            f"logical_id={args.ref_sensor} not in data; present: {sensors}"
        )

    print(f"  reading rows : {len(data)}")
    print(f"  sensors      : {sensors}")
    if args.ref_sensor is None:
        print(f"  series       : mean across {len(sensors)} sensors (bin={args.bin_ms:g} ms)")
    else:
        print(f"  series       : logical_id={args.ref_sensor}")
    print(f"  rh smooth    : {args.rh_smooth_sec:g} s")
    print(f"  duration     : {(data['millis'].max() - data['millis'].min()) / 60_000.0:.1f} min")
    for lab, n in data["label_name"].value_counts().reindex(LABELS, fill_value=0).items():
        print(f"    {lab:<8} {int(n):>6} rows")

    path = plot_raw(
        data,
        args.out,
        args.ref_sensor,
        args.gap_sec,
        args.bin_ms,
        args.rh_smooth_sec,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
