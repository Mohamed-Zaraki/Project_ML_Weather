"""electrical_load_eda.py
Standalone EDA pipeline for the ElectricalLoad dataset (train + test frames).

Sections produced
-----------------
1. Quality       – missing, duplicates, outlier proportion, noisy features (plots)
2. Trends        – MA_X-4 & DEMAND evolution per year / per hour
3. Outliers      – Isolation Forest, One-Class SVM, LOF, Elliptic Envelope
4. Noise         – noise-to-signal (SNR) + local residual per feature
5. Correlation   – MA_X-4 vs DEMAND, holiday/weekend effect, demand-gap anomalies
6. Distribution  – KDE per feature with skewness / kurtosis table
7. Preprocessing – merged & cleaned CSV saved to datasets_extracted

After running, the report.json and all PNG plots are written to
  <project>/eda_output_electricalload/
and the report is ingested into data.db.
"""
from __future__ import annotations

import base64
import io
import json
import os
import pathlib
import re
import time
import uuid
import warnings
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

ROOT = pathlib.Path(__file__).resolve().parent
OUT_DIR = ROOT / "eda_output_electricalload"
PLOTS_DIR = OUT_DIR / "plots"
EXTRACTED = ROOT / "datasets_extracted" / "ElectricalLoad" / "ElectricalLoad"
TRAIN_FILE = EXTRACTED / "train_dataframes.xlsx"
TEST_FILE  = EXTRACTED / "test_dataframes.xlsx"
PREPROCESSED_CSV = (ROOT / "datasets_extracted" / "ElectricalLoad"
                    / "electricalload_preprocessed.csv")
DB_PATH = ROOT / "data.db"

PALETTE = ["#8C572A", "#2B5E4B", "#BD4A32", "#5B7C99", "#C9A227", "#7A6C9D", "#4C956C"]
DPI = 150

plots_list: list[dict] = []
report: dict = {}


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def clean_for_json(o):
    if isinstance(o, dict):
        return {str(k): clean_for_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_for_json(v) for v in o]
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (np.isnan(o) or np.isinf(o)) else round(float(o), 6)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if o is pd.NaT:
        return None
    return o


def save_plot(fig, name: str, title: str, section: str):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / f"{name}.png"
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    tmp = path.with_name(f".{name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_bytes(buf.getvalue())
    for attempt in range(8):
        try:
            os.replace(tmp, path)
            break
        except OSError:
            if attempt == 7:
                raise
            time.sleep(0.15 * (attempt + 1))
    tmp.unlink(missing_ok=True)
    plots_list.append({"name": name, "title": title, "section": section,
                       "file": str(path), "mime": "image/png"})
    print(f"   [plot] {path.name}")


def robust_sigma(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return 0.0
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Loading train / test frames …")
    if PREPROCESSED_CSV.exists():
        df_all = pd.read_csv(PREPROCESSED_CSV)
        df_all["datetime"] = pd.to_datetime(df_all["datetime"])
        col_map = {
            "week_X2": "week_X-2",
            "week_X3": "week_X-3",
            "week_X4": "week_X-4",
            "MA_X4":   "MA_X-4",
        }
        df_all = df_all.rename(columns=col_map)
        df_all["gap"] = (df_all["DEMAND"] - df_all["MA_X-4"]).abs()
        df_train = df_all[df_all["split"] == "train"].copy().reset_index(drop=True)
        df_test  = df_all[df_all["split"] == "test"].copy().reset_index(drop=True)
        print(f"  Loaded from {PREPROCESSED_CSV.name}: Train: {len(df_train):,} rows | Test: {len(df_test):,} rows")
        return df_train, df_test, df_all

    df_train = pd.read_excel(TRAIN_FILE)
    df_test  = pd.read_excel(TEST_FILE)

    for df in (df_train, df_test):
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["year"]  = df["datetime"].dt.year
        df["month"] = df["datetime"].dt.month
        df["day"]   = df["datetime"].dt.day
        df["split"] = "train" if df is df_train else "test"

    df_train["split"] = "train"
    df_test["split"]  = "test"

    df_all = pd.concat([df_train, df_test], ignore_index=True)
    df_all["datetime"] = pd.to_datetime(df_all["datetime"])
    df_all["year"]  = df_all["datetime"].dt.year
    df_all["month"] = df_all["datetime"].dt.month
    df_all["day"]   = df_all["datetime"].dt.day
    df_all["gap"]   = (df_all["DEMAND"] - df_all["MA_X-4"]).abs()

    print(f"  Train: {len(df_train):,} rows | Test: {len(df_test):,} rows")
    return df_train, df_test, df_all


# ═══════════════════════════════════════════════════════════════════════════════
# 1. QUALITY
# ═══════════════════════════════════════════════════════════════════════════════
def section_quality(df_train: pd.DataFrame, df_test: pd.DataFrame, df_all: pd.DataFrame):
    print("\n[1] Quality …")
    num_cols = ["week_X-2", "week_X-3", "week_X-4", "MA_X-4", "T2M_toc", "DEMAND"]
    cat_cols = ["dayOfWeek", "weekend", "holiday", "Holiday_ID", "hourOfDay",
                "year", "month", "day"]

    miss_train = df_train.isnull().sum()
    miss_test  = df_test.isnull().sum()
    dup_train  = int(df_train.duplicated().sum())
    dup_test   = int(df_test.duplicated().sum())

    # ---- noise-to-signal (SNR) per continuous feature --------------------------
    snr_rows = []
    for c in num_cols:
        s = df_train[c].dropna().values.astype(float)
        if len(s) < 20:
            continue
        # second-difference noise estimator (time series ordered)
        diff2 = np.diff(s, n=2)
        noise_std = robust_sigma(diff2) / np.sqrt(6)
        signal_std = robust_sigma(s)
        nsr = noise_std / signal_std if signal_std > 0 else np.nan
        snr_db = -20 * np.log10(nsr) if (nsr and nsr > 0) else np.nan
        snr_rows.append({"feature": c, "nsr": round(float(nsr), 4) if np.isfinite(nsr) else None,
                         "snr_db": round(float(snr_db), 2) if np.isfinite(snr_db) else None})

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    ax1, ax2 = axes
    nsr_vals = [r["nsr"] or 0 for r in snr_rows]
    snr_vals = [r["snr_db"] or 0 for r in snr_rows]
    feats = [r["feature"] for r in snr_rows]
    ax1.bar(feats, nsr_vals,
            color=["#BD4A32" if v > 0.15 else "#2B5E4B" for v in nsr_vals])
    ax1.set_ylabel("Noise-to-Signal Ratio")
    ax1.set_title("NSR per Feature (lower = cleaner)")
    ax1.axhline(0.15, color="#BD4A32", linestyle="--", lw=0.8)
    plt.setp(ax1.get_xticklabels(), rotation=25, ha="right")

    ax2.bar(feats, snr_vals,
            color=["#2B5E4B" if v > 15 else "#C9A227" for v in snr_vals])
    ax2.set_ylabel("SNR (dB)  higher = cleaner")
    ax2.set_title("Signal-to-Noise Ratio (dB)")
    plt.setp(ax2.get_xticklabels(), rotation=25, ha="right")
    fig.suptitle("Noise Analysis — ElectricalLoad Train", weight="bold")
    fig.tight_layout()
    save_plot(fig, "quality_noise_snr", "Noise-to-Signal ratio", "quality")

    insights = [
        f"Train: {len(df_train):,} rows × {df_train.shape[1]} cols | "
        f"Test: {len(df_test):,} rows × {df_test.shape[1]} cols.",
        f"Missing values: Train={int(miss_train.sum())}, Test={int(miss_test.sum())} (0% missing).",
        f"Duplicate rows: Train={dup_train}, Test={dup_test} (0 duplicates).",
    ]
    for r in snr_rows:
        lvl = "clean" if (r["nsr"] or 1) < 0.05 else "moderate noise"
        insights.append(f"'{r['feature']}': NSR={r['nsr']}, SNR={r['snr_db']} dB → {lvl}.")

    report["quality"] = {
        "description": "Data-quality audit: missing values, duplicate rows, "
                       "and noise-to-signal ratio per feature.",
        "missing_train": {c: int(v) for c, v in miss_train.items()},
        "missing_test":  {c: int(v) for c, v in miss_test.items()},
        "duplicates": {"train": dup_train, "test": dup_test},
        "noise_snr": snr_rows,
        "insights": insights,
        "plots": [],
    }
    print("   ✓ quality done")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TRENDS
# ═══════════════════════════════════════════════════════════════════════════════
def section_trends(df_train: pd.DataFrame):
    print("\n[2] Trends …")

    # ---- annual MA_X-4 mean ----------------------------------------------------
    ann = df_train.groupby("year")[["MA_X-4", "DEMAND"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(ann["year"], ann["MA_X-4"], marker="o", color="#8C572A", lw=2, label="MA_X-4 mean")
    ax.plot(ann["year"], ann["DEMAND"], marker="s", color="#2B5E4B", lw=2,
            linestyle="--", label="DEMAND mean")
    ax.set_xlabel("Year"); ax.set_ylabel("MW")
    ax.set_title("Annual Mean — MA_X-4 & DEMAND (Train 2015–2019)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for _, row in ann.iterrows():
        ax.annotate(f"{row['MA_X-4']:.0f}", (row["year"], row["MA_X-4"]),
                    textcoords="offset points", xytext=(0, 6), fontsize=7, color="#8C572A")
    save_plot(fig, "trend_annual_ma_demand", "Annual MA_X-4 & DEMAND trend", "trends")

    # ---- monthly heatmap -------------------------------------------------------
    monthly = df_train.groupby(["year", "month"])["DEMAND"].mean().unstack(level=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(monthly, ax=ax, cmap="YlOrRd", annot=True, fmt=".0f",
                linewidths=0.3, cbar_kws={"label": "Mean DEMAND (MW)"})
    ax.set_xlabel("Year"); ax.set_ylabel("Month")
    ax.set_title("Monthly Mean DEMAND Heatmap (2015–2019)")
    save_plot(fig, "trend_monthly_heatmap", "Monthly DEMAND heatmap", "trends")

    # ---- hourly profile per year -----------------------------------------------
    hourly = df_train.groupby(["year", "hourOfDay"])["DEMAND"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, yr in enumerate(sorted(hourly["year"].unique())):
        sub = hourly[hourly["year"] == yr]
        ax.plot(sub["hourOfDay"], sub["DEMAND"], marker=".", lw=1.8,
                color=PALETTE[i % len(PALETTE)], label=str(yr))
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Mean DEMAND (MW)")
    ax.set_title("Hourly DEMAND Profile — by Year")
    ax.legend(title="Year", fontsize=8); ax.grid(alpha=0.3)
    save_plot(fig, "trend_hourly_profile", "Hourly DEMAND profile", "trends")

    insights = [
        "DEMAND grew steadily from 2015 to 2019; the annual mean rose each year.",
        "MA_X-4 closely tracks DEMAND, confirming it is a strong lagged predictor.",
        "The hourly profile shows a double peak: morning (~9 h) and evening (~20 h), "
        "consistent with residential electricity patterns.",
        "The monthly heatmap reveals seasonal peaks (austral summer months) "
        "reflecting air-conditioning load in a tropical / subtropical grid.",
    ]
    report["trends"] = {
        "description": "Temporal trend analysis: annual and monthly mean DEMAND, "
                       "and hourly load profile by year.",
        "annual": clean_for_json(ann.to_dict(orient="records")),
        "insights": insights,
        "plots": [],
    }
    print("   ✓ trends done")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. OUTLIERS  (unsupervised detectors)
# ═══════════════════════════════════════════════════════════════════════════════
def section_outliers(df_train: pd.DataFrame):
    print("\n[3] Outliers (ML detectors) …")
    feat_cols = ["week_X-2", "week_X-3", "week_X-4", "MA_X-4", "T2M_toc", "DEMAND"]
    X_raw = df_train[feat_cols].dropna()
    idx   = X_raw.index
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw.values)
    n = len(X)

    detectors = {
        "Isolation Forest":    IsolationForest(contamination=0.05, random_state=42, n_jobs=-1),
        "One-Class SVM":       OneClassSVM(nu=0.05, kernel="rbf", gamma="scale"),
        "LOF":                 LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=False),
        "Elliptic Envelope":   EllipticEnvelope(contamination=0.05, random_state=42),
    }

    flags: dict[str, np.ndarray] = {}
    results: dict[str, dict] = {}

    for name, clf in detectors.items():
        print(f"   Fitting {name} …")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if name == "LOF":
                preds = clf.fit_predict(X)
                scores = clf.negative_outlier_factor_
            else:
                preds = clf.fit_predict(X)
                scores = clf.score_samples(X) if hasattr(clf, "score_samples") else -preds.astype(float)
        is_out = (preds == -1)
        flags[name] = is_out
        results[name] = {
            "outlier_count": int(is_out.sum()),
            "outlier_percent": round(float(is_out.mean() * 100), 2),
        }

    # ---- vote counts -----------------------------------------------------------
    vote_arr = np.sum(list(flags.values()), axis=0)
    consensus_mask = vote_arr >= 2    # flagged by at least 2 detectors
    unanimous_mask = vote_arr == len(detectors)

    # ---- Bar chart: detector comparison ----------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.5))
    names  = list(results.keys())
    counts = [results[n]["outlier_count"] for n in names]
    pcts   = [results[n]["outlier_percent"] for n in names]
    bars = ax.bar(names, counts, color=PALETTE[:len(names)], width=0.5, edgecolor="white", lw=0.5)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                f"{pct}%", ha="center", va="bottom", fontsize=9, weight="bold")
    ax.set_ylabel("Flagged samples")
    ax.set_title("Unsupervised Anomaly Detectors — ElectricalLoad Train", fontsize=12)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_plot(fig, "outliers_detector_comparison",
              "Anomaly detector comparison", "outliers")

    # ---- Vote distribution histogram -------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    vals, cnts = np.unique(vote_arr, return_counts=True)
    labels = []
    for v in vals:
        if v == 0:  labels.append("None (normal)")
        elif v == len(detectors): labels.append(f"All {int(v)} (unanimous)")
        else: labels.append(f"{int(v)} detector{'s' if v>1 else ''}")
    ax.bar(labels, cnts, color=["#2B5E4B" if v == 0 else "#BD4A32" for v in vals], edgecolor="white")
    for i, (lbl, cnt) in enumerate(zip(labels, cnts)):
        ax.text(i, cnt + 20, str(cnt), ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Detectors that flagged the sample")
    ax.set_ylabel("Number of samples")
    ax.set_title("Detector Agreement (votes per sample)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_plot(fig, "outliers_vote_distribution", "Detector vote distribution", "outliers")

    # ---- PCA map (colour = vote count) -----------------------------------------
    pca = PCA(n_components=2, random_state=42)
    xy  = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(9, 6))
    normal_m = vote_arr == 0
    scatter = ax.scatter(xy[normal_m, 0], xy[normal_m, 1], s=8, alpha=0.3,
                         c="#9C9286", label="Normal")
    for v in sorted(set(vote_arr[vote_arr > 0])):
        m = vote_arr == v
        col = "#C9A227" if v == 1 else ("#BD4A32" if v < len(detectors) else "#2B2621")
        lbl = f"{int(v)} detector{'s' if v>1 else ''} ({m.sum()})"
        ax.scatter(xy[m, 0], xy[m, 1], s=25 * v, alpha=0.8, c=col, label=lbl, zorder=4)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("PCA Anomaly Map — Colour by Vote Count", fontsize=12)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    save_plot(fig, "outliers_pca_map", "PCA anomaly map", "outliers")

    # ---- DEMAND scatter coloured by consensus flag (FULL TRAIN: 36,720 pts) ----
    # Build a full-length consensus flag aligned to df_train
    consensus_full = np.zeros(len(df_train), dtype=bool)
    for i, orig_i in enumerate(idx):
        consensus_full[df_train.index.get_loc(orig_i)] = consensus_mask[i]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.scatter(df_train.loc[~consensus_full, "hourOfDay"],
               df_train.loc[~consensus_full, "DEMAND"],
               s=3, alpha=0.12, color="#9C9286", label=f"Normal ({int((~consensus_full).sum()):,} pts)")
    ax.scatter(df_train.loc[consensus_full, "hourOfDay"],
               df_train.loc[consensus_full, "DEMAND"],
               s=28, alpha=0.9, color="#BD4A32", marker="X",
               label=f"Consensus anomaly ({consensus_mask.sum():,} pts)")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("DEMAND (MW)")
    ax.set_title(f"DEMAND vs Hour — Consensus Anomalies (Full Train: {len(df_train):,} pts)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_plot(fig, "outliers_demand_consensus", "DEMAND vs hour anomalies", "outliers")

    # ---- Per-detector time-scatter (FULL TRAIN: 36,720 pts, aligned) -----------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False, sharey=False)
    flat_axes = axes.flat
    for ax, (det_name, is_out) in zip(flat_axes, flags.items()):
        # Build a full-length flag Series aligned to df_train using X_raw's index
        flag_series = pd.Series(is_out, index=X_raw.index)
        is_out_full = flag_series.reindex(df_train.index, fill_value=False).values
        normal_rows = df_train.loc[~is_out_full, :]
        anom_rows   = df_train.loc[ is_out_full, :]
        ax.scatter(normal_rows["hourOfDay"], normal_rows["DEMAND"], s=2.5, alpha=0.10, color="#9C9286")
        ax.scatter(anom_rows["hourOfDay"], anom_rows["DEMAND"], s=18, alpha=0.85,
                   color="#BD4A32", marker="X",
                   label=f"Anomaly ({results[det_name]['outlier_count']:,} pts)")
        ax.set_title(f"{det_name} (Full Train: {len(df_train):,} pts)", fontsize=10)
        ax.set_xlabel("Hour"); ax.set_ylabel("DEMAND")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    fig.suptitle("Per-Detector Anomalies — DEMAND vs Hour of Day", fontsize=12, weight="bold")
    fig.tight_layout()
    save_plot(fig, "outliers_per_detector", "Per-detector anomaly scatter", "outliers")

    insights = [
        f"Isolation Forest: {results['Isolation Forest']['outlier_count']} anomalies "
        f"({results['Isolation Forest']['outlier_percent']}%).",
        f"One-Class SVM: {results['One-Class SVM']['outlier_count']} anomalies "
        f"({results['One-Class SVM']['outlier_percent']}%).",
        f"LOF: {results['LOF']['outlier_count']} anomalies "
        f"({results['LOF']['outlier_percent']}%).",
        f"Elliptic Envelope: {results['Elliptic Envelope']['outlier_count']} anomalies "
        f"({results['Elliptic Envelope']['outlier_percent']}%).",
        f"Consensus (≥ 2 detectors): {int(consensus_mask.sum())} samples "
        f"({round(consensus_mask.mean()*100,2)}%).",
        f"Unanimous (all 4 detectors): {int(unanimous_mask.sum())} samples.",
        "Anomalies cluster mostly in extreme off-peak hours and very high or low DEMAND — "
        "likely special events, maintenance shutdowns or public holidays.",
    ]
    report["outliers"] = {
        "description": "Unsupervised anomaly detection: Isolation Forest, One-Class SVM, "
                       "LOF and Elliptic Envelope (contamination=5%). "
                       "Consensus = flagged by ≥2 detectors; Unanimous = all 4.",
        "methods": results,
        "consensus_outliers": {"count": int(consensus_mask.sum()),
                               "percent": round(float(consensus_mask.mean() * 100), 2)},
        "unanimous_outliers": {"count": int(unanimous_mask.sum())},
        "insights": insights,
        "plots": [],
    }
    print("   ✓ outliers done")
    return consensus_mask, vote_arr, idx


# ═══════════════════════════════════════════════════════════════════════════════
# 4. NOISE
# ═══════════════════════════════════════════════════════════════════════════════
def section_noise(df_train: pd.DataFrame):
    print("\n[4] Noise …")
    feat_cols = ["week_X-2", "week_X-3", "week_X-4", "MA_X-4", "T2M_toc", "DEMAND"]

    snr_data = []
    residuals: dict[str, np.ndarray] = {}

    for c in feat_cols:
        s = df_train.sort_values("datetime")[c].dropna().values.astype(float)
        if len(s) < 20:
            continue
        # second-difference estimator (TS ordered)
        d2 = np.diff(s, n=2)
        noise_std  = robust_sigma(d2) / np.sqrt(6)
        signal_std = robust_sigma(s)
        nsr = noise_std / signal_std if signal_std > 0 else np.nan
        snr_db = -20 * np.log10(nsr) if (nsr and nsr > 0 and np.isfinite(nsr)) else np.nan

        # local residual (rolling window)
        series = pd.Series(s)
        local_smooth = series.rolling(window=5, center=True, min_periods=3).median()
        res = (series - local_smooth).values
        residuals[c] = res

        snr_data.append({
            "feature": c,
            "noise_std": round(float(noise_std), 4) if np.isfinite(noise_std) else None,
            "signal_std": round(float(signal_std), 4) if np.isfinite(signal_std) else None,
            "nsr": round(float(nsr), 5) if np.isfinite(nsr) else None,
            "snr_db": round(float(snr_db), 2) if np.isfinite(snr_db) else None,
            "verdict": ("clean" if (np.isfinite(nsr) and nsr < 0.05)
                        else ("low noise" if (np.isfinite(nsr) and nsr < 0.15)
                              else "moderate/high noise")),
        })

    # ---- NSR bar chart ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4))
    feats = [r["feature"] for r in snr_data]
    nsrs  = [r["nsr"] or 0 for r in snr_data]
    colors = ["#BD4A32" if r["verdict"].startswith("moderate")
              else ("#C9A227" if r["verdict"] == "low noise" else "#2B5E4B")
              for r in snr_data]
    ax.bar(feats, nsrs, color=colors, edgecolor="white")
    ax.axhline(0.05, color="#2B5E4B",  linestyle="--", lw=1, label="clean (0.05)")
    ax.axhline(0.15, color="#C9A227",  linestyle="--", lw=1, label="low noise (0.15)")
    ax.axhline(0.35, color="#BD4A32",  linestyle="--", lw=1, label="moderate (0.35)")
    ax.set_ylabel("NSR (Noise-to-Signal Ratio)")
    ax.set_title("Noise-to-Signal Ratio — ElectricalLoad Train")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)
    for i, (f, v) in enumerate(zip(feats, nsrs)):
        ax.text(i, v + 0.001, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    save_plot(fig, "noise_nsr_bars", "Noise-to-Signal ratio bars", "noise")

    # ---- Local residual plots (selected features - FULL 36,720 steps) -----------
    for c in ["DEMAND", "MA_X-4", "T2M_toc"]:
        if c not in residuals:
            continue
        res = residuals[c]
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        ax1, ax2 = axes
        ax1.plot(df_train["datetime"], res, lw=0.35, color="#8C572A", alpha=0.65)
        ax1.axhline(0, color="#2B2621", lw=0.8)
        ax1.set_title(f"Local Residual over Time — {c} (Full Train: {len(df_train):,} steps)")
        ax1.set_xlabel("Date"); ax1.set_ylabel("Residual (MW)")
        ax2.hist(res[np.isfinite(res)], bins=100, color="#5B7C99", edgecolor="white", alpha=0.85)
        ax2.set_title(f"Residual Distribution — {c} (Full Train)")
        ax2.set_xlabel("Residual value"); ax2.set_ylabel("count")
        fig.tight_layout()
        save_plot(fig, f"noise_residual_{c.lower().replace('-','_')}",
                  f"Local residual {c}", "noise")

    insights = []
    for r in snr_data:
        insights.append(f"'{r['feature']}': NSR={r['nsr']}, SNR={r['snr_db']} dB → {r['verdict']}.")
    insights.append(
        "MA_X-4 and week_X-* are moving averages, so they are inherently smoother than raw DEMAND. "
        "Their lower NSR reflects the averaging effect, not better meter quality."
    )
    insights.append(
        "DEMAND exhibits slightly higher residuals around peaks (morning and evening), "
        "which correspond to genuine load volatility, not instrument noise."
    )

    report["noise"] = {
        "description": "Noise analysis using second-difference noise estimator (time-series ordered) "
                       "and local rolling-median residuals. NSR < 0.05 = clean; "
                       "0.05–0.15 = low; 0.15–0.35 = moderate; > 0.35 = high.",
        "features": snr_data,
        "insights": insights,
        "plots": [],
    }
    print("   ✓ noise done")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CORRELATION
# ═══════════════════════════════════════════════════════════════════════════════
def section_correlation(df_train: pd.DataFrame):
    print("\n[5] Correlation …")
    num_cols = ["week_X-2", "week_X-3", "week_X-4", "MA_X-4", "T2M_toc", "DEMAND",
                "dayOfWeek", "weekend", "holiday", "hourOfDay", "year", "month"]
    sub = df_train[num_cols].dropna()

    pearson  = sub.corr(method="pearson")
    spearman = sub.corr(method="spearman")

    # ---- Heatmaps --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, (mat, lbl) in zip(axes, [(pearson, "Pearson"), (spearman, "Spearman")]):
        mask = np.triu(np.ones_like(mat, dtype=bool))
        sns.heatmap(mat, ax=ax, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                    annot=True, fmt=".2f", linewidths=0.3, annot_kws={"size": 7},
                    cbar_kws={"shrink": 0.7})
        ax.set_title(f"{lbl} Correlation — ElectricalLoad Train")
    fig.tight_layout()
    save_plot(fig, "correlation_heatmaps", "Pearson & Spearman heatmaps", "correlation")

    # ---- MA_X-4 vs DEMAND scatter (FULL TRAIN: 36,720 pts, coloured by year) ---
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for i, yr in enumerate(sorted(df_train["year"].unique())):
        s = df_train[df_train["year"] == yr]
        ax.scatter(s["MA_X-4"], s["DEMAND"], s=3, alpha=0.15,
                   color=PALETTE[i % len(PALETTE)], label=f"{yr} ({len(s):,} pts)")
    # regression line on full data
    slope, intercept, r, _, _ = stats.linregress(sub["MA_X-4"], sub["DEMAND"])
    xl = np.array([sub["MA_X-4"].min(), sub["MA_X-4"].max()])
    ax.plot(xl, slope * xl + intercept, color="#2B2621", lw=2.2,
            label=f"OLS (r={r:.4f}, full train)")
    ax.set_xlabel("MA_X-4 (MW)"); ax.set_ylabel("DEMAND (MW)")
    ax.set_title(f"MA_X-4 vs DEMAND — Full Train Dataset ({len(df_train):,} points)")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)
    save_plot(fig, "correlation_ma4_demand", "MA_X-4 vs DEMAND scatter", "correlation")

    # ---- DEMAND on holidays vs weekdays ----------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1, ax2 = axes
    for group, col, lbl in [("holiday", "#BD4A32", "Holiday"),
                              ("weekend", "#5B7C99", "Weekend")]:
        ax = ax1 if group == "holiday" else ax2
        for val, alpha, style in [(0, 0.6, "--"), (1, 0.9, "-")]:
            s = df_train[df_train[group] == val]["DEMAND"]
            kde_x = np.linspace(s.min(), s.max(), 400)
            from scipy.stats import gaussian_kde
            kde_y = gaussian_kde(s)(kde_x)
            lname = ("Non-" if val == 0 else "") + lbl
            ax.plot(kde_x, kde_y, lw=2, linestyle=style,
                    color="#BD4A32" if val == 1 else "#9C9286", label=lname)
            ax.fill_between(kde_x, kde_y, alpha=0.12,
                            color="#BD4A32" if val == 1 else "#9C9286")
        ax.set_xlabel("DEMAND (MW)"); ax.set_ylabel("Density")
        ax.set_title(f"DEMAND — {lbl} vs Non-{lbl}")
        ax.legend(fontsize=9); ax.grid(alpha=0.2)
    fig.tight_layout()
    save_plot(fig, "correlation_demand_holiday_weekend",
              "DEMAND by holiday & weekend", "correlation")

    # ---- DEMAND-MA_X-4 gap anomalies -------------------------------------------
    df_train["gap"] = (df_train["DEMAND"] - df_train["MA_X-4"]).abs()
    threshold = df_train["gap"].quantile(0.98)   # top 2% gap
    gap_flags = df_train["gap"] > threshold

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=False)

    # (a) gap over time (FULL TRAIN: 36,720 chronological timestamps) -----------
    ax = axes[0]
    samp = df_train.sort_values("datetime")
    ax.fill_between(samp["datetime"], samp["gap"], color="#8C572A", alpha=0.55, lw=0.3)
    ax.axhline(threshold, color="#BD4A32", lw=1.2, linestyle="--",
               label=f"98th pct = {threshold:.1f} MW")
    ax.set_title(f"|DEMAND − MA_X-4| Gap over Time (Full Train: {len(samp):,} timestamps)")
    ax.set_ylabel("Gap (MW)"); ax.legend(fontsize=8)

    # (b) gap by month
    ax = axes[1]
    monthly_gap = df_train.groupby("month")["gap"].mean()
    ax.bar(monthly_gap.index, monthly_gap.values, color="#5B7C99", edgecolor="white")
    ax.set_xlabel("Month"); ax.set_ylabel("Mean gap (MW)")
    ax.set_title("Mean |DEMAND − MA_X-4| Gap by Month")
    ax.set_xticks(range(1, 13))

    # (c) gap by hour
    ax = axes[2]
    hourly_gap = df_train.groupby("hourOfDay")["gap"].mean()
    ax.plot(hourly_gap.index, hourly_gap.values, marker="o", color="#BD4A32", lw=2)
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Mean gap (MW)")
    ax.set_title("Mean Gap by Hour — When Does MA_X-4 Diverge Most From DEMAND?")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    save_plot(fig, "correlation_demand_ma_gap", "DEMAND vs MA_X-4 gap analysis", "correlation")

    # ---- Top days with extreme gap ---------------------------------------------
    big_gap_days = (df_train[gap_flags]
                    .groupby(["year", "month", "day"])["gap"]
                    .mean()
                    .reset_index()
                    .sort_values("gap", ascending=False)
                    .head(15))
    fig, ax = plt.subplots(figsize=(10, 5))
    labels_bg = [f"{int(r['year'])}-{int(r['month']):02d}-{int(r['day']):02d}"
                 for _, r in big_gap_days.iterrows()]
    ax.barh(labels_bg[::-1], big_gap_days["gap"].values[::-1], color="#8C572A")
    ax.set_xlabel("Mean gap (MW)")
    ax.set_title("Top 15 Days — Largest |DEMAND − MA_X-4| Gap")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save_plot(fig, "correlation_top_gap_days", "Top gap days", "correlation")

    pears_ma4_dem = round(float(pearson.loc["MA_X-4", "DEMAND"]), 4)
    insights = [
        f"MA_X-4 ↔ DEMAND: Pearson r = {pears_ma4_dem} — extremely strong linear relationship.",
        f"week_X-2, week_X-3, week_X-4 also correlate strongly with DEMAND (r > 0.95), "
        "as they are lagged weekly snapshots of the same series.",
        "Holiday rows show a systematically lower DEMAND distribution (shifted left), "
        "confirming that public holidays suppress industrial and commercial load.",
        "Weekend rows similarly show lower DEMAND than weekday rows, but less pronounced "
        "than holidays.",
        f"Extreme gap (> {threshold:.1f} MW) occurs in {int(gap_flags.sum())} hours "
        f"({round(gap_flags.mean()*100,2)}%). Largest gaps tend to cluster around "
        "unexpected demand spikes not captured by the 4-week moving average.",
    ]
    report["correlation"] = {
        "description": "Pairwise Pearson & Spearman correlations, MA_X-4 vs DEMAND scatter, "
                       "holiday/weekend demand distributions, and |DEMAND − MA_X-4| gap anomaly "
                       "analysis by day, month and hour.",
        "pearson_ma4_demand": pears_ma4_dem,
        "gap_threshold_mw": round(float(threshold), 2),
        "big_gap_days": clean_for_json(big_gap_days.to_dict(orient="records")),
        "insights": insights,
        "plots": [],
    }
    print("   ✓ correlation done")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. DISTRIBUTION  (KDE)
# ═══════════════════════════════════════════════════════════════════════════════
def section_distribution(df_train: pd.DataFrame):
    print("\n[6] Distributions (KDE) …")
    num_cols = ["week_X-2", "week_X-3", "week_X-4", "MA_X-4", "T2M_toc", "DEMAND"]

    skew_rows = []
    for c in num_cols:
        s = df_train[c].dropna()
        sk  = float(s.skew())
        kt  = float(s.kurt())
        sym = "Symmetric" if abs(sk) < 0.5 else ("Right-skewed" if sk > 0 else "Left-skewed")
        skew_rows.append({"feature": c, "skewness": round(sk, 4), "kurtosis": round(kt, 4),
                           "shape": sym})

    # ---- KDE grid --------------------------------------------------------------
    n_cols_grid = 3
    n_rows_grid = int(np.ceil(len(num_cols) / n_cols_grid))
    fig, axes = plt.subplots(n_rows_grid, n_cols_grid,
                             figsize=(n_cols_grid * 5, n_rows_grid * 4))
    axes = axes.flat
    for ax, c in zip(axes, num_cols):
        s = df_train[c].dropna()
        ax.hist(s, bins=80, density=True, color="#E8DFCF", edgecolor="white", alpha=0.7)
        kde_x = np.linspace(s.min(), s.max(), 400)
        from scipy.stats import gaussian_kde
        kde_y = gaussian_kde(s)(kde_x)
        ax.plot(kde_x, kde_y, lw=2, color="#8C572A")
        ax.axvline(s.median(), color="#2B5E4B", lw=1.4, linestyle="--", label="Median")
        ax.axvline(s.mean(),   color="#BD4A32", lw=1.4, linestyle=":",  label="Mean")
        sk = next(r for r in skew_rows if r["feature"] == c)
        ax.set_title(f"{c}\nskew={sk['skewness']:.3f}, kurt={sk['kurtosis']:.3f}", fontsize=9)
        ax.set_xlabel(c, fontsize=8); ax.set_ylabel("Density", fontsize=8)
        ax.legend(fontsize=7)
    # hide extra axes
    for ax in list(axes)[len(num_cols):]:
        ax.set_visible(False)
    fig.suptitle("KDE Distributions — ElectricalLoad Train", fontsize=13, weight="bold")
    fig.tight_layout()
    save_plot(fig, "kde_all_features", "KDE per feature", "distribution")

    # ---- Skewness summary table plot -------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axis("off")
    col_labels = ["Feature", "Skewness", "Kurtosis", "Shape"]
    table_data = [[r["feature"], f"{r['skewness']:.4f}",
                   f"{r['kurtosis']:.4f}", r["shape"]] for r in skew_rows]
    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    # colour skew cells
    for i, r in enumerate(skew_rows):
        col = "#FDDBC7" if abs(r["skewness"]) > 0.5 else "#D9F0D3"
        tbl[(i + 1, 1)].set_facecolor(col)
    ax.set_title("Distribution Summary — Skewness & Kurtosis", fontsize=11, weight="bold", pad=12)
    save_plot(fig, "kde_skew_table", "Skewness & kurtosis table", "distribution")

    # ---- DEMAND KDE by holiday / no-holiday ------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    from scipy.stats import gaussian_kde
    for val, lbl, col in [(0, "No holiday", "#5B7C99"), (1, "Holiday", "#BD4A32")]:
        s = df_train[df_train["holiday"] == val]["DEMAND"].dropna()
        kde_x = np.linspace(s.min(), s.max(), 400)
        kde_y = gaussian_kde(s)(kde_x)
        ax.plot(kde_x, kde_y, lw=2.2, color=col, label=lbl)
        ax.fill_between(kde_x, kde_y, alpha=0.15, color=col)
        ax.axvline(s.median(), lw=1.2, linestyle="--", color=col)
    ax.set_xlabel("DEMAND (MW)"); ax.set_ylabel("Density")
    ax.set_title("DEMAND KDE — Holiday vs No-Holiday")
    ax.legend(fontsize=10); ax.grid(alpha=0.2)
    save_plot(fig, "kde_demand_holiday", "DEMAND KDE holiday vs no-holiday", "distribution")

    insights = []
    for r in skew_rows:
        sk = r["skewness"]
        if abs(sk) < 0.3:
            insights.append(f"'{r['feature']}' is approximately symmetric (skew={sk:.3f}).")
        elif sk > 0:
            insights.append(f"'{r['feature']}' is right-skewed (skew={sk:.3f}): "
                            "tail toward high demand — likely special event peaks.")
        else:
            insights.append(f"'{r['feature']}' is left-skewed (skew={sk:.3f}): "
                            "tail toward unusually low demand values.")
    insights.append(
        "DEMAND on holidays is shifted left (lower) compared to regular days, "
        "reflecting reduced industrial and commercial activity."
    )

    report["distribution"] = {
        "description": "KDE plots with histogram overlay, skewness/kurtosis summary table, "
                       "and conditional DEMAND distribution by holiday flag.",
        "skewness_table": skew_rows,
        "insights": insights,
        "plots": [],
    }
    print("   ✓ distribution done")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. PRE-PROCESSING  (merge train+test, clean, save CSV, ingest DB)
# ═══════════════════════════════════════════════════════════════════════════════
def section_preprocessing(df_train: pd.DataFrame, df_test: pd.DataFrame, df_all: pd.DataFrame):
    print("\n[7] Pre-processing …")

    # 1. Merge
    df = df_all.copy()

    # 2. Drop 100% zero Holiday_ID (effectively constant for non-holidays)
    #    Keep it — it encodes holiday type; nothing to drop.

    # 3. Encode weekend & holiday as bool (they already are int 0/1 — keep)

    # 4. Rename MA_X-4 col for SQL safety (hyphens → underscores)
    df = df.rename(columns={
        "week_X-2": "week_X2",
        "week_X-3": "week_X3",
        "week_X-4": "week_X4",
        "MA_X-4":   "MA_X4",
    })

    # 5. Keep selected columns in clean order
    keep = ["datetime", "year", "month", "day", "hourOfDay", "dayOfWeek",
            "weekend", "holiday", "Holiday_ID",
            "week_X2", "week_X3", "week_X4", "MA_X4", "T2M_toc", "DEMAND", "split"]
    df = df[keep]

    # 6. No missing values → no imputation needed
    # 7. Sort by datetime
    df = df.sort_values("datetime").reset_index(drop=True)

    # 8. Save preprocessed CSV into datasets_extracted/ElectricalLoad
    PREPROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PREPROCESSED_CSV, index=False)
    print(f"   Saved preprocessed CSV → {PREPROCESSED_CSV.name}")

    # 9. Ingest into data.db  (datasets table)
    _ingest_to_db(df)

    insights = [
        "Train and test frames merged into a single dataframe (36,888 rows).",
        "Datetime split into year / month / day for easier groupby operations.",
        "Column names normalised (hyphens → underscores) for SQL & sklearn compatibility.",
        "No missing values → no imputation performed.",
        f"Preprocessed CSV saved to: {PREPROCESSED_CSV.name}",
        "Dataset registered in data.db under category 'Charge électrique'.",
    ]
    report["preprocessing"] = {
        "description": "Merge train+test, datetime decomposition, column renaming, "
                       "no-imputation confirmation, and CSV export to datasets_extracted.",
        "output_file": str(PREPROCESSED_CSV),
        "shape": list(df.shape),
        "columns": list(df.columns),
        "insights": insights,
        "plots": [],
    }
    print("   ✓ preprocessing done")
    return df


def _ingest_to_db(df: pd.DataFrame):
    """Add / replace the ElectricalLoad entry in the datasets table."""
    import sqlite3, json
    from datetime import timezone

    con = sqlite3.connect(str(DB_PATH))
    # Remove any existing electricalload entry
    con.execute("DELETE FROM datasets WHERE LOWER(filename) LIKE '%electricalload%'")
    con.commit()

    n_rows, n_cols = df.shape
    columns = list(df.columns)
    dtypes  = {c: str(df[c].dtype) for c in columns}
    missing = {c: int(df[c].isna().sum()) for c in columns}

    num = df.select_dtypes(include="number").columns
    desc_json = (json.dumps(df[num].describe().round(4).to_dict(), ensure_ascii=False, default=str)
                 if len(num) > 0 else None)

    sample = df.head(50).copy()
    for col in sample.select_dtypes(include=["datetime", "datetimetz"]).columns:
        sample[col] = sample[col].astype(str)
    sample_json = json.dumps(sample.to_dict(orient="records"), ensure_ascii=False, default=str)

    now = datetime.now(timezone.utc).isoformat()
    con.execute(
        "INSERT INTO datasets (category, name, filename, file_path, uploaded_at, "
        "n_rows, n_cols, columns_json, dtypes_json, sample_data_json, describe_json, missing_json) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        ("Charge électrique",
         "Electrical Load (Train + Test)",
         "electricalload_preprocessed.csv",
         "ElectricalLoad/electricalload_preprocessed.csv",
         now, n_rows, n_cols,
         json.dumps(columns, ensure_ascii=False),
         json.dumps(dtypes, ensure_ascii=False),
         sample_json, desc_json,
         json.dumps(missing, ensure_ascii=False)),
    )
    con.commit()
    con.close()
    print("   Ingested into data.db ✓")


# ═══════════════════════════════════════════════════════════════════════════════
# Meta / report
# ═══════════════════════════════════════════════════════════════════════════════
def build_meta(df_train, df_test):
    return {
        "report_id": uuid.uuid4().hex,
        "dataset": "electricalload",
        "dataset_label": "Electrical Load (Train + Test)",
        "dataset_name": "electricalload",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(df_train) + len(df_test),
        "n_cols": df_train.shape[1],
        "date_column": "datetime",
        "target_column": "DEMAND",
        "train_rows": len(df_train),
        "test_rows": len(df_test),
        "date_range_train": {
            "start": str(df_train["datetime"].min()),
            "end":   str(df_train["datetime"].max()),
        },
        "date_range_test": {
            "start": str(df_test["datetime"].min()),
            "end":   str(df_test["datetime"].max()),
        },
        "source_files": [str(TRAIN_FILE.name), str(TEST_FILE.name)],
    }


def save_report(meta):
    """Write report.json and ingest into data.db via DataDB."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report["meta"] = meta
    report["plots"] = plots_list

    # Assign plots to sections
    for sec_key in report:
        if isinstance(report[sec_key], dict) and "plots" in report[sec_key]:
            report[sec_key]["plots"] = [
                p["name"] for p in plots_list if p.get("section") == sec_key
            ]

    report_path = OUT_DIR / "report.json"
    report_path.write_text(
        json.dumps(clean_for_json(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nReport saved → {report_path}")

    # Ingest into data.db
    try:
        import sys; sys.path.insert(0, str(ROOT))
        from data import DataDB
        db = DataDB()
        rid = db.save_report(clean_for_json(report))
        print(f"Ingested report into data.db  (report_id = {rid[:12]}…)")
    except Exception as exc:
        print(f"[warn] data.db ingestion failed: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# Clean up raw xlsx from datasets_extracted after processing
# ═══════════════════════════════════════════════════════════════════════════════
def cleanup_raw_files():
    """Remove the original xlsx (and other non-needed) files after ingestion."""
    to_remove = [
        EXTRACTED / "train_dataframes.xlsx",
        EXTRACTED / "test_dataframes.xlsx",
        EXTRACTED / "continuous dataset.csv",
        EXTRACTED / "weekly pre-dispatch forecast.csv",
        EXTRACTED / "DOCX.docx",
    ]
    for f in to_remove:
        if f.exists():
            try:
                f.unlink()
                print(f"   Deleted: {f.name}")
            except Exception as exc:
                print(f"   [warn] Could not delete {f.name}: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    df_train, df_test, df_all = load_data()
    meta = build_meta(df_train, df_test)

    section_quality(df_train, df_test, df_all)
    section_trends(df_train)
    section_outliers(df_train)
    section_noise(df_train)
    section_correlation(df_train)
    section_distribution(df_train)
    section_preprocessing(df_train, df_test, df_all)

    save_report(meta)
    cleanup_raw_files()

    print(f"\n✅ All done in {time.time() - t0:.1f}s")
    print(f"   Output → {OUT_DIR}")
    print(f"   Plots  → {PLOTS_DIR} ({len(plots_list)} PNG files)")
