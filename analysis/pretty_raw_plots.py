#!/usr/bin/env python3
"""Pretty raw time-series plots for BME688 cage-odor logs.

Keeps every reading row. Marker rows (label_change, profile_change, etc.)
are ignored — nothing is trimmed after a label swap.

One line per heater profile, with filled markers colored by heater_step
(parula). Label spans are drawn as background bands.

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
from matplotlib.colors import LinearSegmentedColormap, Normalize
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
# Distinct line colors so the four protocols stay readable under parula markers.
PROFILE_LINE_COLORS = (
    "#1b1b1b",
    "#2a6f97",
    "#6a4c93",
    "#bc4749",
)

# MATLAB-like parula stops (RGB in 0–1).
_PARULA_STOPS = [
    (0.2081, 0.1663, 0.5292),
    (0.2116, 0.1898, 0.5777),
    (0.2123, 0.2138, 0.6270),
    (0.2082, 0.2386, 0.6771),
    (0.1959, 0.2645, 0.7279),
    (0.1707, 0.2919, 0.7792),
    (0.1253, 0.3242, 0.8303),
    (0.0591, 0.3598, 0.8683),
    (0.0117, 0.3875, 0.8820),
    (0.0060, 0.4086, 0.8828),
    (0.0165, 0.4266, 0.8786),
    (0.0329, 0.4430, 0.8720),
    (0.0498, 0.4586, 0.8641),
    (0.0629, 0.4737, 0.8554),
    (0.0723, 0.4887, 0.8467),
    (0.0779, 0.5040, 0.8384),
    (0.0793, 0.5200, 0.8312),
    (0.0749, 0.5375, 0.8263),
    (0.0641, 0.5570, 0.8240),
    (0.0488, 0.5772, 0.8218),
    (0.0343, 0.5966, 0.8197),
    (0.0265, 0.6137, 0.8183),
    (0.0239, 0.6287, 0.8198),
    (0.0231, 0.6418, 0.8262),
    (0.0228, 0.6535, 0.8395),
    (0.0267, 0.6642, 0.8599),
    (0.0433, 0.6743, 0.8869),
    (0.0788, 0.6872, 0.9046),
    (0.1273, 0.7033, 0.9013),
    (0.1803, 0.7210, 0.8860),
    (0.2334, 0.7385, 0.8661),
    (0.2849, 0.7550, 0.8449),
    (0.3346, 0.7703, 0.8230),
    (0.3832, 0.7846, 0.8005),
    (0.4310, 0.7983, 0.7771),
    (0.4783, 0.8115, 0.7526),
    (0.5254, 0.8243, 0.7268),
    (0.5726, 0.8367, 0.6994),
    (0.6201, 0.8486, 0.6702),
    (0.6682, 0.8598, 0.6389),
    (0.7168, 0.8701, 0.6051),
    (0.7659, 0.8794, 0.5684),
    (0.8152, 0.8875, 0.5282),
    (0.8641, 0.8941, 0.4840),
    (0.9115, 0.8988, 0.4348),
    (0.9557, 0.9006, 0.3794),
    (0.9853, 0.9000, 0.3219),
    (0.9958, 0.9012, 0.2727),
    (0.9888, 0.9124, 0.2421),
    (0.9735, 0.9331, 0.2325),
    (0.9560, 0.9530, 0.2360),
    (0.9422, 0.9679, 0.2473),
    (0.9339, 0.9769, 0.2652),
    (0.9299, 0.9809, 0.2866),
    (0.9279, 0.9819, 0.3075),
    (0.9259, 0.9811, 0.3259),
    (0.9230, 0.9790, 0.3418),
    (0.9185, 0.9760, 0.3556),
    (0.9123, 0.9725, 0.3681),
    (0.9048, 0.9686, 0.3796),
    (0.8963, 0.9644, 0.3905),
    (0.8872, 0.9599, 0.4010),
    (0.8778, 0.9551, 0.4114),
    (0.8684, 0.9500, 0.4218),
]


def parula_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("parula", _PARULA_STOPS, N=256)


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
        help="logical_id to plot (default: lowest present)",
    )
    p.add_argument(
        "--gap-sec",
        type=float,
        default=60.0,
        help="Break a profile line when samples are farther apart than this (default: 60)",
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
        # Extend each band halfway to the next sample so swaps look contiguous.
        ax.axvspan(
            (a - t0) / 60_000.0,
            (b - t0) / 60_000.0,
            facecolor=LABEL_COLORS.get(lab, "#cccccc"),
            edgecolor="none",
            alpha=0.18,
            zorder=0,
        )


def plot_raw(
    data: pd.DataFrame,
    out: Path,
    ref_sensor: int,
    gap_sec: float,
) -> Path:
    sub = data[data["logical_id"] == ref_sensor].sort_values("millis")
    if sub.empty:
        raise SystemExit(f"No rows for logical_id={ref_sensor}")

    profiles = (
        sub.groupby(["profile_id", "profile_name"], sort=True)
        .size()
        .reset_index(name="n")
        .sort_values("profile_id")
    )
    if profiles.empty:
        raise SystemExit("No heater profiles found in readings")

    t0 = float(sub["millis"].min())
    spans = label_spans(sub)
    cmap = parula_cmap()
    norm = Normalize(vmin=0, vmax=9)
    gap_min = gap_sec / 60.0

    fig, ax = plt.subplots(figsize=(14, 5.5))
    shade_labels(ax, spans, t0)

    for i, row in enumerate(profiles.itertuples(index=False)):
        pid, pname = int(row.profile_id), str(row.profile_name)
        sl = sub[sub["profile_id"] == pid].sort_values("millis")
        if sl.empty:
            continue
        t_min = (sl["millis"].to_numpy(dtype=float) - t0) / 60_000.0
        y = sl["log_r"].to_numpy(dtype=float)
        t_line, y_line = break_on_gaps(t_min, y, gap_min)
        line_color = PROFILE_LINE_COLORS[i % len(PROFILE_LINE_COLORS)]
        ax.plot(
            t_line,
            y_line,
            color=line_color,
            linewidth=1.1,
            alpha=0.85,
            solid_capstyle="round",
            zorder=2,
            label=pname,
        )
        ax.scatter(
            t_min,
            y,
            c=sl["heater_step"].to_numpy(),
            cmap=cmap,
            norm=norm,
            s=28,
            marker="o",
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )

    ax.set_xlabel("minutes from first sample")
    ax.set_ylabel("log10(gas resistance / ohm)")
    ax.set_title(
        f"Raw heater-sweep time series  ·  logical_id={ref_sensor}  ·  "
        "lines = protocol, markers = heater_step (parula)"
    )

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        pad=0.015,
        fraction=0.03,
    )
    cbar.set_label("heater_step")
    cbar.set_ticks(range(10))

    protocol_handles = [
        Line2D([0], [0], color=PROFILE_LINE_COLORS[i % len(PROFILE_LINE_COLORS)], lw=1.5)
        for i in range(len(profiles))
    ]
    protocol_labels = [str(r.profile_name) for r in profiles.itertuples(index=False)]
    label_handles = [
        Patch(facecolor=LABEL_COLORS[lab], edgecolor="none", alpha=0.35, label=LABEL_DISPLAY[lab])
        for lab in LABELS
        if any(s[0] == lab for s in spans)
    ]
    leg1 = ax.legend(
        protocol_handles,
        protocol_labels,
        title="protocol",
        loc="upper left",
        framealpha=0.92,
    )
    ax.add_artist(leg1)
    if label_handles:
        ax.legend(
            handles=label_handles,
            title="label (background)",
            loc="upper right",
            framealpha=0.92,
        )

    ax.set_axisbelow(False)
    for spine in ax.spines.values():
        spine.set_zorder(4)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "pretty_raw_timeseries.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    print(f"Loading {args.csv}")
    data = load_readings(args.csv)
    if data.empty:
        raise SystemExit("No reading rows after ignoring markers")

    sensors = sorted(data["logical_id"].unique().tolist())
    ref = args.ref_sensor if args.ref_sensor is not None else sensors[0]
    if ref not in sensors:
        raise SystemExit(f"logical_id={ref} not in data; present: {sensors}")

    print(f"  reading rows : {len(data)}")
    print(f"  sensors      : {sensors}")
    print(f"  ref sensor   : {ref}")
    print(f"  duration     : {(data['millis'].max() - data['millis'].min()) / 60_000.0:.1f} min")
    for lab, n in data["label_name"].value_counts().reindex(LABELS, fill_value=0).items():
        print(f"    {lab:<8} {int(n):>6} rows")

    path = plot_raw(data, args.out, ref, args.gap_sec)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
