#!/usr/bin/env python3
"""Explore and classify BME688 cage-odor logs (UNKNOWN / CLEAN / SOILED).

UNKNOWN is treated as ambient air (out of the cage) and is kept for
exploration. The classifier is binary: CLEAN vs SOILED only.

Usage:
    python analysis/analyze_cage.py path/to/bme688_log_YYYYMMDD_HHMMSS.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LABELS = ("UNKNOWN", "CLEAN", "SOILED")
CAGE_LABELS = ("CLEAN", "SOILED")
PAIR_SPECS = (
    ("clean_soiled", "CLEAN", "SOILED"),
    ("unknown_clean", "UNKNOWN", "CLEAN"),
    ("unknown_soiled", "UNKNOWN", "SOILED"),
)
LABEL_COLORS = {
    "UNKNOWN": "#7f7f7f",
    "CLEAN": "#1f77b4",
    "SOILED": "#ff7f0e",
}
STEP_COLS = [f"r{i}" for i in range(10)]
AVG_BLOCK = 20  # ~30 s of 1.4 s cycles


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Explore BME688 cage logs and train a CLEAN vs SOILED classifier."
    )
    p.add_argument("csv", type=Path, help="Path to bme688_log_*.csv")
    p.add_argument(
        "--drop-after-label-min",
        type=float,
        default=10.0,
        help="Minutes of readings to drop after each label_change (default: 10)",
    )
    p.add_argument(
        "--test-frac",
        type=float,
        default=0.3,
        help="Last fraction of each CLEAN/SOILED block held out for test (default: 0.3)",
    )
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
        help="logical_id for time-series plots (default: lowest present)",
    )
    return p.parse_args()


def load_log(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path, comment="#")
    needed = {
        "millis",
        "label_name",
        "marker",
        "logical_id",
        "profile_id",
        "profile_name",
        "profile_cycle",
        "heater_step",
        "gas_resistance_ohm",
        "temperature_c",
        "humidity_pct",
        "gas_valid",
        "heat_stable",
    }
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"CSV is missing columns: {sorted(missing)}")

    marker = df["marker"]
    is_marker = marker.notna() & marker.astype(str).str.strip().ne("") & marker.astype(
        str
    ).str.lower().ne("nan")
    markers = df.loc[is_marker].copy()
    data = df.loc[~is_marker].copy()
    return data, markers


def _to_numeric(data: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = data.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def clean_readings(
    data: pd.DataFrame, markers: pd.DataFrame, drop_after_label_min: float
) -> tuple[pd.DataFrame, dict]:
    """Keep UNKNOWN/CLEAN/SOILED; drop invalid, first profile visit, swap window."""
    qa: dict = {}
    n0 = len(data)

    data = _to_numeric(
        data,
        [
            "millis",
            "logical_id",
            "profile_id",
            "profile_cycle",
            "heater_step",
            "heater_temp_c",
            "heater_dur_ms",
            "gas_resistance_ohm",
            "temperature_c",
            "humidity_pct",
            "gas_valid",
            "heat_stable",
        ],
    )
    data["label_name"] = data["label_name"].astype(str).str.strip().str.upper()
    data = data[data["label_name"].isin(LABELS)]
    qa["rows_after_label_filter"] = int(len(data))

    before_valid = len(data)
    data = data[
        (data["gas_valid"] == 1)
        & (data["heat_stable"] == 1)
        & data["gas_resistance_ohm"].gt(0)
        & data["heater_step"].between(0, 9)
    ]
    qa["rows_dropped_invalid"] = int(before_valid - len(data))

    before_cycle = len(data)
    min_cycle = data.groupby("profile_id")["profile_cycle"].transform("min")
    data = data[data["profile_cycle"] > min_cycle]
    qa["rows_dropped_first_profile_cycle"] = int(before_cycle - len(data))

    drop_ms = float(drop_after_label_min) * 60_000.0
    swap_windows: list[tuple[float, float]] = []
    if "millis" in markers.columns and "marker" in markers.columns:
        m = markers.copy()
        m["millis"] = pd.to_numeric(m["millis"], errors="coerce")
        m["marker"] = m["marker"].astype(str).str.strip()
        changes = m.loc[m["marker"] == "label_change", "millis"].dropna().to_numpy()
        for t0 in changes:
            swap_windows.append((float(t0), float(t0) + drop_ms))

    in_swap = np.zeros(len(data), dtype=bool)
    millis = data["millis"].to_numpy()
    for t0, t1 in swap_windows:
        in_swap |= (millis >= t0) & (millis < t1)
    qa["rows_dropped_swap_window"] = int(in_swap.sum())
    qa["swap_windows"] = swap_windows
    qa["rows_raw"] = n0
    data = data.loc[~in_swap].copy()
    for c in ("logical_id", "profile_id", "profile_cycle", "heater_step"):
        data[c] = data[c].astype(int)
    qa["rows_kept"] = int(len(data))
    return data, qa


def print_qa(data: pd.DataFrame, qa: dict) -> None:
    print("=== QA ===")
    print(f"  raw reading rows           : {qa['rows_raw']}")
    print(f"  after UNKNOWN/CLEAN/SOILED : {qa['rows_after_label_filter']}")
    print(f"  dropped invalid/unstable   : {qa['rows_dropped_invalid']}")
    print(f"  dropped first profile cycle: {qa['rows_dropped_first_profile_cycle']}")
    print(f"  dropped swap window        : {qa['rows_dropped_swap_window']}")
    print(f"  kept                       : {qa['rows_kept']}")
    sensors = sorted(data["logical_id"].dropna().unique().tolist()) if len(data) else []
    print(f"  sensors (logical_id)       : {sensors}")
    if data.empty:
        return
    d = data.sort_values("millis")
    changed = d["label_name"] != d["label_name"].shift()
    d = d.assign(_block=changed.cumsum())
    blocks = (
        d.groupby(["_block", "label_name"], as_index=False)
        .agg(t0=("millis", "min"), t1=("millis", "max"), n=("millis", "size"))
    )
    blocks["minutes"] = (blocks["t1"] - blocks["t0"]) / 60_000.0
    print("  label blocks (after filters):")
    for _, row in blocks.iterrows():
        print(
            f"    {row['label_name']:<8}  {row['minutes']:7.1f} min  "
            f"{int(row['n']):>8} rows"
        )
    totals = blocks.groupby("label_name")["minutes"].sum()
    print("  total minutes by label:")
    for lab in LABELS:
        if lab in totals.index:
            print(f"    {lab:<8}  {totals[lab]:7.1f} min")


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    if pooled == 0:
        return 0.0
    return float((b.mean() - a.mean()) / pooled)


def _auc_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Separability AUC treating b as the positive class.

    The score is flipped when class b has lower values, so 1.0 is perfect
    separation in either direction. Cohen's d keeps the sign.
    """
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    y = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
    s = np.concatenate([a, b])
    if np.unique(s).size < 2:
        return 0.5
    raw = float(roc_auc_score(y, s))
    return max(raw, 1.0 - raw)


def heater_step_scores(data: pd.DataFrame) -> pd.DataFrame:
    data = data.assign(log_r=np.log10(data["gas_resistance_ohm"]))
    rows = []
    sensors = sorted(data["logical_id"].unique())
    grouped = data.groupby(["profile_id", "profile_name", "heater_step"], sort=True)
    for (pid, pname, step), g in grouped:
        counts = {lab: int((g["label_name"] == lab).sum()) for lab in LABELS}
        by_lab = {
            lab: g.loc[g["label_name"] == lab, "log_r"].to_numpy() for lab in LABELS
        }
        rec = {
            "profile_id": int(pid),
            "profile_name": pname,
            "heater_step": int(step),
            "heater_temp_c": (
                float(g["heater_temp_c"].median())
                if "heater_temp_c" in g.columns
                else np.nan
            ),
            "n_unknown": counts["UNKNOWN"],
            "n_clean": counts["CLEAN"],
            "n_soiled": counts["SOILED"],
        }
        for key, lab_a, lab_b in PAIR_SPECS:
            rec[f"d_{key}"] = _cohens_d(by_lab[lab_a], by_lab[lab_b])
            rec[f"auc_{key}"] = _auc_1d(by_lab[lab_a], by_lab[lab_b])

        signs = []
        for sid in sensors:
            sg = g[g["logical_id"] == sid]
            d = _cohens_d(
                sg.loc[sg["label_name"] == "CLEAN", "log_r"].to_numpy(),
                sg.loc[sg["label_name"] == "SOILED", "log_r"].to_numpy(),
            )
            if np.isfinite(d) and d != 0:
                signs.append(np.sign(d))
        n_signed = len(signs)
        if n_signed:
            majority = np.sign(np.median(signs))
            n_agree = int(np.sum(np.array(signs) == majority))
        else:
            majority = np.nan
            n_agree = 0
        rec["n_sensors"] = int(len(sensors))
        rec["n_sensors_same_sign_clean_soiled"] = n_agree
        rec["frac_same_sign_clean_soiled"] = (
            n_agree / n_signed if n_signed else float("nan")
        )
        rec["majority_sign_clean_soiled"] = majority
        rows.append(rec)
    scores = pd.DataFrame(rows)
    if not scores.empty:
        scores = scores.sort_values(
            ["auc_clean_soiled", "d_clean_soiled"],
            ascending=False,
            key=lambda c: c.abs() if c.name == "d_clean_soiled" else c,
        )
    return scores


def _shade_swaps(ax, windows: list[tuple[float, float]], t_unit: float) -> None:
    labeled = False
    for t0, t1 in windows:
        ax.axvspan(
            t0 / t_unit,
            t1 / t_unit,
            color="k",
            alpha=0.08,
            label="swap window" if not labeled else None,
        )
        labeled = True


def _minutes(millis: np.ndarray, t0: float) -> np.ndarray:
    return (millis - t0) / 60_000.0


def plot_timeseries(
    data: pd.DataFrame,
    scores: pd.DataFrame,
    ref_sensor: int,
    swap_windows: list[tuple[float, float]],
    out: Path,
) -> None:
    sub = data[data["logical_id"] == ref_sensor].sort_values("millis")
    if sub.empty:
        print(f"  skip timeseries: no rows for logical_id={ref_sensor}")
        return
    if scores.empty or scores["auc_clean_soiled"].isna().all():
        step = int(sub["heater_step"].mode().iloc[0])
        pname = str(sub["profile_name"].mode().iloc[0])
    else:
        best = scores.sort_values("auc_clean_soiled", ascending=False).iloc[0]
        step = int(best["heater_step"])
        pname = best["profile_name"]
    hit = sub[(sub["profile_name"] == pname) & (sub["heater_step"] == step)]
    if hit.empty:
        hit = sub[sub["heater_step"] == step]
    if not hit.empty:
        sub = hit

    t0 = float(data["millis"].min())
    fig, ax = plt.subplots(figsize=(12, 4))
    for lab in LABELS:
        sl = sub[sub["label_name"] == lab]
        if sl.empty:
            continue
        ax.scatter(
            _minutes(sl["millis"].to_numpy(), t0),
            np.log10(sl["gas_resistance_ohm"]),
            s=4,
            alpha=0.45,
            c=LABEL_COLORS[lab],
            label=lab if lab != "UNKNOWN" else "UNKNOWN (air)",
            linewidths=0,
        )
    shifted = [(a - t0, b - t0) for a, b in swap_windows]
    _shade_swaps(ax, shifted, 60_000.0)
    ax.set_xlabel("minutes from first kept sample")
    ax.set_ylabel("log10(gas resistance / ohm)")
    ax.set_title(
        f"Ref sensor logical_id={ref_sensor}  "
        f"profile={pname}  heater_step={step}"
    )
    ax.legend(markerscale=4, loc="best")
    fig.tight_layout()
    fig.savefig(out / "timeseries_logR.png", dpi=140)
    plt.close(fig)


def plot_th(
    data: pd.DataFrame,
    ref_sensor: int,
    swap_windows: list[tuple[float, float]],
    out: Path,
) -> None:
    sub = data[data["logical_id"] == ref_sensor].sort_values("millis")
    if sub.empty:
        return
    # One point per ~cycle so T/RH is not 10x overplotted.
    sub = sub[sub["heater_step"] == sub["heater_step"].min()]
    t0 = float(data["millis"].min())
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, col, ylab in (
        (axes[0], "humidity_pct", "humidity (%RH)"),
        (axes[1], "temperature_c", "temperature (C)"),
    ):
        for lab in LABELS:
            sl = sub[sub["label_name"] == lab]
            if sl.empty:
                continue
            ax.plot(
                _minutes(sl["millis"].to_numpy(), t0),
                sl[col],
                ".",
                markersize=2,
                alpha=0.6,
                color=LABEL_COLORS[lab],
                label=lab if lab != "UNKNOWN" else "UNKNOWN (air)",
            )
        shifted = [(a - t0, b - t0) for a, b in swap_windows]
        _shade_swaps(ax, shifted, 60_000.0)
        ax.set_ylabel(ylab)
        ax.legend(markerscale=4, loc="best")
    axes[1].set_xlabel("minutes from first kept sample")
    axes[0].set_title(
        f"T / RH confound check  (logical_id={ref_sensor}; "
        "a class gap that tracks RH is not odor)"
    )
    fig.tight_layout()
    fig.savefig(out / "timeseries_TH.png", dpi=140)
    plt.close(fig)


def plot_boxplots(data: pd.DataFrame, out: Path) -> None:
    data = data.assign(log_r=np.log10(data["gas_resistance_ohm"]))
    profiles = list(data.groupby(["profile_id", "profile_name"]).size().index)
    n_prof = len(profiles)
    if n_prof == 0:
        return
    fig, axes = plt.subplots(
        n_prof, 10, figsize=(22, 3.2 * n_prof), sharex=True, squeeze=False
    )
    order = list(LABELS)
    for i, (pid, pname) in enumerate(profiles):
        for step in range(10):
            ax = axes[i][step]
            sl = data[(data["profile_id"] == pid) & (data["heater_step"] == step)]
            vals = []
            for lab in order:
                v = sl.loc[sl["label_name"] == lab, "log_r"].to_numpy()
                vals.append(v if len(v) else np.array([np.nan]))
            bp = ax.boxplot(
                vals,
                positions=[1, 2, 3],
                widths=0.6,
                patch_artist=True,
                showfliers=False,
            )
            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels(["air", "C", "S"])
            for patch, lab in zip(bp["boxes"], order):
                patch.set_facecolor(LABEL_COLORS[lab])
                patch.set_alpha(0.7)
            if i == 0:
                ax.set_title(f"step {step}")
            if step == 0:
                ax.set_ylabel(f"{pname}\nlog10(R)")
            ax.tick_params(labelsize=8)
    fig.suptitle("log10(R) by label  (UNKNOWN=air, C=CLEAN, S=SOILED)")
    fig.tight_layout()
    fig.savefig(out / "boxplots_logR.png", dpi=120)
    plt.close(fig)


def build_fingerprints(data: pd.DataFrame) -> pd.DataFrame:
    d = data.sort_values(["logical_id", "profile_id", "profile_cycle", "millis"]).copy()
    wrap = d.groupby(
        ["logical_id", "profile_id", "profile_cycle"], sort=False
    )["heater_step"].diff()
    d["fp_id"] = (
        wrap.lt(0).groupby(
            [d["logical_id"], d["profile_id"], d["profile_cycle"]]
        ).cumsum()
    )
    keys = ["logical_id", "profile_id", "profile_name", "profile_cycle", "fp_id"]
    r = d.pivot_table(
        index=keys,
        columns="heater_step",
        values="gas_resistance_ohm",
        aggfunc="first",
    )
    r.columns = [f"r{int(c)}" for c in r.columns]
    for c in STEP_COLS:
        if c not in r.columns:
            r[c] = np.nan
    r = r[STEP_COLS]
    complete = r.notna().all(axis=1)

    def _mode(s: pd.Series) -> str:
        m = s.mode()
        return str(m.iloc[0]) if len(m) else str(s.iloc[0])

    meta = d.groupby(keys).agg(
        label_name=("label_name", _mode),
        n_labels=("label_name", "nunique"),
        millis=("millis", "median"),
        temperature_c=("temperature_c", "mean"),
        humidity_pct=("humidity_pct", "mean"),
    )
    fp = meta.join(r, how="inner")
    fp = fp.loc[complete & (fp["n_labels"] == 1)].drop(columns="n_labels")
    fp = fp.reset_index()
    for c in STEP_COLS:
        fp[c] = np.log10(fp[c])
    return fp


def time_holdout(fp: pd.DataFrame, test_frac: float) -> tuple[pd.Index, pd.Index]:
    train_parts = []
    test_parts = []
    for lab in CAGE_LABELS:
        sub = fp[fp["label_name"] == lab].sort_values("millis")
        n = len(sub)
        if n < 4:
            raise SystemExit(
                f"Not enough {lab} fingerprints for a time holdout (n={n})."
            )
        n_test = max(1, int(round(n * test_frac)))
        n_train = n - n_test
        if n_train < 1:
            raise SystemExit(f"test-frac={test_frac} leaves no {lab} training rows.")
        train_parts.append(sub.index[:n_train])
        test_parts.append(sub.index[n_train:])
    train_idx = train_parts[0].append(train_parts[1])
    test_idx = test_parts[0].append(test_parts[1])
    return train_idx, test_idx


def feature_matrix(fp: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = fp[STEP_COLS + ["temperature_c", "humidity_pct"]].to_numpy(dtype=float)
    y = (fp["label_name"] == "SOILED").astype(int).to_numpy()
    return x, y


def fit_threshold(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (direction, threshold) on a 1-D score. Predict SOILED if direction*x > t."""
    pos, neg = x[y == 1], x[y == 0]
    direction = 1.0 if np.nanmean(pos) >= np.nanmean(neg) else -1.0
    score = direction * x
    fpr, tpr, thresh = roc_curve(y, score)
    j = tpr - fpr
    # sklearn puts inf as the first threshold; skip non-finite.
    finite = np.isfinite(thresh)
    if not finite.any():
        return direction, float(np.median(score))
    k = int(np.argmax(j[finite]))
    t = float(thresh[finite][k])
    return direction, t


def predict_threshold(
    x: np.ndarray, direction: float, t: float
) -> tuple[np.ndarray, np.ndarray]:
    score = direction * x
    pred = (score >= t).astype(int)
    # Map score to a 0-1 monotone for AUC (already oriented).
    return pred, score


def block_average(
    fp: pd.DataFrame, proba: np.ndarray, y: np.ndarray, block: int = AVG_BLOCK
) -> tuple[np.ndarray, np.ndarray]:
    tmp = fp.assign(_p=proba, _y=y)
    ys, ps = [], []
    for _, g in tmp.groupby("logical_id", sort=False):
        g = g.sort_values("millis")
        for start in range(0, len(g) - block + 1, block):
            chunk = g.iloc[start : start + block]
            if chunk["_y"].nunique() != 1:
                continue
            ys.append(int(chunk["_y"].iloc[0]))
            ps.append(float(chunk["_p"].mean()))
    if not ys:
        return np.array([]), np.array([])
    return np.asarray(ys), np.asarray(ps)


def _acc_auc(
    y: np.ndarray,
    score: np.ndarray,
    pred: np.ndarray | None = None,
    decision_t: float = 0.5,
) -> tuple[float, float]:
    if len(y) == 0 or np.unique(y).size < 2:
        return float("nan"), float("nan")
    if pred is None:
        pred = (score >= decision_t).astype(int)
    acc = float(accuracy_score(y, pred))
    auc = float(roc_auc_score(y, score))
    return acc, auc


def classify(
    fp: pd.DataFrame, scores: pd.DataFrame, test_frac: float
) -> pd.DataFrame:
    cage = fp[fp["label_name"].isin(CAGE_LABELS)].copy()
    if cage["label_name"].nunique() < 2:
        print("=== Classifier ===")
        print("  skip: need both CLEAN and SOILED fingerprints.")
        return pd.DataFrame()

    train_idx, test_idx = time_holdout(cage, test_frac)
    train, test = cage.loc[train_idx], cage.loc[test_idx]
    print("=== Classifier (CLEAN vs SOILED, time holdout) ===")
    print(
        f"  train: CLEAN={int((train.label_name=='CLEAN').sum())}  "
        f"SOILED={int((train.label_name=='SOILED').sum())}"
    )
    print(
        f"  test : CLEAN={int((test.label_name=='CLEAN').sum())}  "
        f"SOILED={int((test.label_name=='SOILED').sum())}"
    )

    # Baseline: best CLEAN-vs-SOILED heater step from exploration, threshold on train.
    if scores.empty or scores["auc_clean_soiled"].isna().all():
        best_step = 0
        best_profile = None
        print("  baseline: no exploration AUC; using heater_step 0")
    else:
        ranked = scores.dropna(subset=["auc_clean_soiled"]).sort_values(
            "auc_clean_soiled", ascending=False
        )
        best = ranked.iloc[0]
        best_step = int(best["heater_step"])
        best_profile = best["profile_name"]
        print(
            f"  baseline step: {best_profile} heater_step={best_step}  "
            f"explore AUC={best['auc_clean_soiled']:.3f}  "
            f"d={best['d_clean_soiled']:.3f}  "
            f"same-sign sensors={int(best['n_sensors_same_sign_clean_soiled'])}/"
            f"{int(best['n_sensors'])}"
        )

    col = f"r{best_step}"
    y_tr = (train["label_name"] == "SOILED").astype(int).to_numpy()
    y_test = (test["label_name"] == "SOILED").astype(int).to_numpy()
    direction, thresh = fit_threshold(train[col].to_numpy(), y_tr)
    base_pred, base_score = predict_threshold(
        test[col].to_numpy(), direction, thresh
    )
    base_acc, base_auc = _acc_auc(y_test, base_score, pred=base_pred)
    print(f"  baseline 1-cycle  acc={base_acc:.3f}  AUC={base_auc:.3f}")

    x_train, y_train = feature_matrix(train)
    x_test, _ = feature_matrix(test)
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "lr",
                LogisticRegression(max_iter=2000, solver="lbfgs"),
            ),
        ]
    )
    model.fit(x_train, y_train)
    model_proba = model.predict_proba(x_test)[:, 1]
    model_acc, model_auc = _acc_auc(y_test, model_proba)
    print(f"  10-D+T/RH 1-cycle acc={model_acc:.3f}  AUC={model_auc:.3f}")

    y_avg_b, s_avg_b = block_average(test, base_score, y_test)
    y_avg_m, p_avg_m = block_average(test, model_proba, y_test)
    base_pred_avg = (s_avg_b >= thresh).astype(int) if len(s_avg_b) else np.array([])
    base_acc_avg, base_auc_avg = _acc_auc(y_avg_b, s_avg_b, pred=base_pred_avg)
    model_acc_avg, model_auc_avg = _acc_auc(y_avg_m, p_avg_m)
    print(
        f"  baseline ~{AVG_BLOCK}-cycle acc={base_acc_avg:.3f}  AUC={base_auc_avg:.3f}"
        f"  (n_blocks={len(y_avg_b)})"
    )
    print(
        f"  10-D+T/RH ~{AVG_BLOCK}-cycle acc={model_acc_avg:.3f}  "
        f"AUC={model_auc_avg:.3f}  (n_blocks={len(y_avg_m)})"
    )

    print("  per-sensor test AUC (10-D+T/RH):")
    per_sensor = []
    for sid, g in test.groupby("logical_id"):
        yt = (g["label_name"] == "SOILED").astype(int).to_numpy()
        if np.unique(yt).size < 2:
            print(f"    logical_id={int(sid)}  skip (one class only)")
            continue
        pr = model.predict_proba(feature_matrix(g)[0])[:, 1]
        acc, auc = _acc_auc(yt, pr)
        print(f"    logical_id={int(sid)}  acc={acc:.3f}  AUC={auc:.3f}")
        per_sensor.append({"logical_id": int(sid), "acc": acc, "auc": auc})

    air = fp[fp["label_name"] == "UNKNOWN"]
    mean_p_air = float("nan")
    if len(air):
        p_air = model.predict_proba(feature_matrix(air)[0])[:, 1]
        mean_p_air = float(np.mean(p_air))
        print(
            f"  UNKNOWN (air) mean P(SOILED)={mean_p_air:.3f}  "
            f"(n={len(air)}; not used in accuracy)"
        )
    else:
        print("  UNKNOWN (air): no fingerprints to score")

    if np.isfinite(model_auc) and np.isfinite(base_auc) and model_auc <= base_auc + 0.01:
        print(
            "  The 10-D model is no better than the best single heater step. "
            "Freeze that profile/step on the firmware rather than deploying "
            "a larger model."
        )

    metrics = pd.DataFrame(
        [
            {
                "model": "baseline_best_step",
                "profile_name": best_profile,
                "heater_step": best_step,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "acc_1cycle": base_acc,
                "auc_1cycle": base_auc,
                "acc_avg20": base_acc_avg,
                "auc_avg20": base_auc_avg,
                "unknown_mean_p_soiled": np.nan,
            },
            {
                "model": "logistic_10d_TH",
                "profile_name": "all_in_features",
                "heater_step": -1,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "acc_1cycle": model_acc,
                "auc_1cycle": model_auc,
                "acc_avg20": model_acc_avg,
                "auc_avg20": model_auc_avg,
                "unknown_mean_p_soiled": mean_p_air,
            },
        ]
    )
    return metrics


def main() -> int:
    args = parse_args()
    if not args.csv.is_file():
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        return 1
    if not 0.0 < args.test_frac < 1.0:
        print("--test-frac must be between 0 and 1", file=sys.stderr)
        return 1

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.csv}")
    data, markers = load_log(args.csv)
    data, qa = clean_readings(data, markers, args.drop_after_label_min)
    print_qa(data, qa)
    if data.empty:
        print("No rows left after cleaning.", file=sys.stderr)
        return 1

    data = data.assign(log_r=np.log10(data["gas_resistance_ohm"]))
    ref = args.ref_sensor
    if ref is None:
        ref = int(data["logical_id"].min())
    elif ref not in set(data["logical_id"].unique()):
        print(
            f"--ref-sensor {ref} not in data; using {int(data['logical_id'].min())}"
        )
        ref = int(data["logical_id"].min())

    print("=== Exploration ===")
    scores = heater_step_scores(data)
    scores_path = out / "heater_step_scores.csv"
    scores.to_csv(scores_path, index=False)
    print(f"  wrote {scores_path}")
    if not scores.empty:
        show = scores.dropna(subset=["auc_clean_soiled"]).head(8)
        print("  top CLEAN vs SOILED heater conditions:")
        for _, r in show.iterrows():
            print(
                f"    {r['profile_name']:<18} step={int(r['heater_step'])}  "
                f"T={r['heater_temp_c']:.0f}C  "
                f"d={r['d_clean_soiled']:+.3f}  "
                f"AUC={r['auc_clean_soiled']:.3f}  "
                f"air-vs-CLEAN AUC={r['auc_unknown_clean']:.3f}  "
                f"air-vs-SOILED AUC={r['auc_unknown_soiled']:.3f}  "
                f"sign-agree={r['frac_same_sign_clean_soiled']:.2f}"
            )

    plot_timeseries(data, scores, ref, qa["swap_windows"], out)
    plot_th(data, ref, qa["swap_windows"], out)
    plot_boxplots(data, out)
    print(f"  wrote plots under {out}")

    print("=== Fingerprints ===")
    fp = build_fingerprints(data)
    print(f"  complete 10-step cycles: {len(fp)}")
    for lab in LABELS:
        n = int((fp["label_name"] == lab).sum())
        print(f"    {lab:<8} {n}")
    fp.to_csv(out / "fingerprints.csv", index=False)

    metrics = classify(fp, scores, args.test_frac)
    if not metrics.empty:
        metrics.to_csv(out / "metrics.csv", index=False)
        print(f"  wrote {out / 'metrics.csv'}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
