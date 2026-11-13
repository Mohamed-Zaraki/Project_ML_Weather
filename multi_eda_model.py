"""multi_eda_model.py - General Model layer (MVC) for the multi-dataset project.

The weather project stays in ``weather_model.py``.  This module applies the same
EDA / ML philosophy to four tabular datasets:

    * COVID-19 laboratory data
    * Parkinson's disease biomarkers
    * Alcohol detection (sensors + computer vision)
    * Heart-attack / heart-disease clinical data

Eight sections are produced (all JSON-friendly, no UI code):

  1. Description        : dtypes, head, describe, skewness, kurtosis, memory
  2. Contradictions     : impossible / contradictory records (bp = 0, chol = 0,
                          p-tau > t-tau, "no alcohol" class with alcohol > 0, ...)
  3. Missing & duplicates: explicit NaNs, hidden missing zeros, duplicate rows/cols
  4. Outliers           : classical (IQR, Z, modified Z, percentile) + a consensus
                          of six *unsupervised* detectors (Isolation Forest,
                          One-Class SVM, LOF, Elliptic Envelope, DBSCAN, Mahalanobis)
  5. Noise              : noise-to-signal statistic adapted to unordered tables
                          (local k-NN residual / robust signal spread), SNR(dB),
                          Hampel spikes, stuck readings, out-of-range values
  6. Correlation        : Pearson, Spearman, clustered heatmap, VIF, top pairs,
                          target correlation, class-conditional distributions
  7. KDE                : distribution shape / modality, optionally by target class
  8. Model phase        : target balance, sklearn Pipeline + ColumnTransformer,
                          stratified hold-out + cross-validation, confusion matrix,
                          impurity / permutation / mutual-information importances
"""
from __future__ import annotations

import base64
import io
import json
import re
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: the model never opens a GUI window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import DataDB
from quality_multi import (
    KNN_NEIGHBOURS, NSR_LEVELS, SPIKE_K, SPIKE_MIN_FRAC, SPIKE_PCT_FLAG,
    STUCK_RUN, _clean_numeric_matrix, compute_noise_metrics, detect_contradictions,
    detect_ml_outliers, get_physical_range, knn_neighbor_indices, mask_impossible,
    noise_levels, robust_sigma,
)

# Ravishing but sober palette shared by every plot
PALETTE = ["#8C572A", "#2B5E4B", "#BD4A32", "#5B7C99", "#C9A227", "#7A6C9D", "#4C956C"]


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def clean_for_json(o):
    """Recursively convert numpy / pandas objects into JSON-serialisable types."""
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


def records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", date_format="iso"))


def slug(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(text)).strip("_").lower()


def _numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


class BaseDatasetEDAModel:
    """Generic EDA + ML pipeline for one tabular dataset."""

    label = "Dataset"
    dataset_description = ""

    def __init__(self, csv_path: str | Path, dataset_key: str, target_col: str,
                 drop_cols: list[str] | None = None,
                 category_cols: list[str] | None = None,
                 out_dir: str | Path = "eda_output", dpi: int = 150,
                 embed_base64: bool = False):
        self.csv_path = Path(csv_path)
        self.dataset_key = dataset_key.lower()
        self.target_col = target_col
        self.drop_cols = drop_cols or []
        self.user_cat_cols = category_cols or []
        self.out_dir = Path(out_dir)
        self.plots_dir = self.out_dir / "plots"
        self.dpi = dpi
        self.embed_base64 = embed_base64
        self.plots: list[dict] = []
        self.report: dict = {}
        self.db_report_id: str | None = None

    # ---------------------------------------------------------- loading
    def load(self):
        self.raw = pd.read_csv(self.csv_path)
        df = self.raw.copy()

        valid_drops = [c for c in self.drop_cols if c in df.columns]
        if valid_drops:
            df = df.drop(columns=valid_drops)
        self.df = df

        target = self.target_col if self.target_col in df.columns else None
        self.target_present = target is not None

        all_cols = [c for c in df.columns if c != target]
        num_all = df[all_cols].select_dtypes(include="number").columns.tolist()
        self.constant_cols = [c for c in num_all if df[c].nunique(dropna=True) <= 1]
        self.num_cols = [c for c in num_all if c not in self.constant_cols]

        # Continuous vs low-cardinality numeric codes (sex, cp, angina, ...)
        self.continuous_cols = [c for c in self.num_cols if df[c].nunique(dropna=True) > 10]
        self.low_card_cols = [c for c in self.num_cols if c not in self.continuous_cols]

        # Non-numeric columns (+ declared categorical columns that are not numeric)
        non_numeric = [c for c in df.columns
                       if c != target and not pd.api.types.is_numeric_dtype(df[c])]
        self.cat_cols = list(dict.fromkeys(non_numeric))
        self.group_cols = [c for c in self.user_cat_cols if c in df.columns]

        # Physically impossible readings -> NaN
        self.df_valid, self.invalid_counts = mask_impossible(df, self.dataset_key)

        # Contradictory records are detected on the *raw* data on purpose
        self.contradiction_report = detect_contradictions(df, self.dataset_key)

        self._neighbours: np.ndarray | None = None

    # ---------------------------------------------------------- plotting
    def _save_plot(self, fig, name: str, title: str, section: str):
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        path = self.plots_dir / f"{name}.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        entry = {"name": name, "title": title, "section": section,
                 "file": str(path), "mime": "image/png"}
        if self.embed_base64:
            entry["base64"] = base64.b64encode(path.read_bytes()).decode()
        self.plots.append(entry)

    def _section(self, key: str, fn):
        before = len(self.plots)
        result = fn()
        result["plots"] = [p["name"] for p in self.plots[before:]]
        self.report[key] = result

    # ---------------------------------------------------------- 1. description
    def describe(self):
        buf = io.StringIO()
        self.raw.info(buf=buf)

        cols_for_stats = self.num_cols + self.constant_cols
        desc_num = self.df_valid[cols_for_stats].describe().T
        if len(desc_num):
            desc_num["skew_valid"] = self.df_valid[desc_num.index].skew()
            desc_num["kurtosis_valid"] = self.df_valid[desc_num.index].kurt()
            desc_num["n_unique"] = [int(self.df[c].nunique(dropna=True)) for c in desc_num.index]

        desc_cat = pd.DataFrame()
        if self.cat_cols:
            desc_cat = self.df[self.cat_cols].describe().T
            desc_cat["n_unique"] = [int(self.df[c].nunique(dropna=True)) for c in desc_cat.index]

        value_counts = {}
        for c in self.cat_cols + self.low_card_cols:
            vc = self.df[c].value_counts(dropna=False).head(12)
            value_counts[c] = {("Missing" if pd.isna(k) else str(k)): int(v) for k, v in vc.items()}

        insights = [
            f"{self.label}: '{self.csv_path.name}' contains {len(self.raw):,} rows and "
            f"{self.raw.shape[1]} columns.",
            f"Analysed features: {len(self.num_cols)} numeric "
            f"({len(self.continuous_cols)} continuous, {len(self.low_card_cols)} coded/low-cardinality), "
            f"{len(self.cat_cols)} categorical. Target: '{self.target_col}'.",
        ]
        if self.drop_cols:
            dropped = [c for c in self.drop_cols if c in self.raw.columns]
            if dropped:
                insights.append(f"Excluded identifier / non-predictive column(s): {dropped}.")
        if self.constant_cols:
            insights.append(f"Constant / zero-variance column(s) excluded: {self.constant_cols}.")

        skewed = [c for c in desc_num.index if abs(desc_num.loc[c, "skew_valid"]) > 1.5]
        if skewed:
            insights.append(f"Strongly skewed features (|skew| > 1.5): {skewed}.")

        if self.invalid_counts:
            total_bad = sum(self.invalid_counts.values())
            insights.append(
                f"{total_bad} physically impossible / corrupted reading(s) across "
                f"{len(self.invalid_counts)} feature(s) {list(self.invalid_counts.keys())} "
                "were masked as NaN before statistics, correlations and modelling."
            )

        cr = self.contradiction_report
        if cr["rows_flagged"]:
            insights.append(
                f"{cr['rows_flagged']:,} rows ({cr['rows_flagged_percent']}%) contain at least "
                f"one contradictory record across {cr['triggered_rules']} rule(s); see the "
                "'Contradictions' section."
            )
        else:
            insights.append("No contradictory record was detected by the domain rules.")

        return {
            "description": "Structure of the dataset: dtypes / non-null counts, first rows, summary "
                           "statistics, skewness, kurtosis, cardinality and categorical value counts.",
            "dataset_description": self.dataset_description,
            "info": {
                "text": buf.getvalue(),
                "memory_kb": round(float(self.raw.memory_usage(deep=True).sum()) / 1024, 2),
                "columns": [{"name": c, "dtype": str(self.raw[c].dtype),
                             "non_null": int(self.raw[c].notna().sum()),
                             "n_unique": int(self.raw[c].nunique()),
                             "missing": int(self.raw[c].isna().sum())} for c in self.raw.columns],
            },
            "head": records(self.raw.head(10)),
            "describe_numeric": records(desc_num.reset_index().rename(columns={"index": "feature"}))
            if len(desc_num) else [],
            "describe_categorical": records(desc_cat.reset_index().rename(columns={"index": "feature"}))
            if len(desc_cat) else [],
            "value_counts": value_counts,
            "constant_columns": self.constant_cols,
            "dropped_columns": self.drop_cols,
            "feature_roles": {
                "continuous_numeric": self.continuous_cols,
                "low_cardinality_numeric": self.low_card_cols,
                "categorical": self.cat_cols,
                "target": self.target_col,
            },
            "insights": insights,
        }

    # ---------------------------------------------------------- 2. contradictions
    def _plot_contradiction_rule(self, rule: dict, name: str):
        """Plot the two most relevant columns of a rule, flagging the bad rows."""
        cols = [c for c in rule["columns"] if c in self.df.columns]
        if not cols:
            return
        bad_idx = [i for i in rule["row_indices"] if i in self.df.index]
        if not bad_idx:
            return

        numeric = [c for c in cols if pd.api.types.is_numeric_dtype(self.df[c])]
        fig, ax = plt.subplots(figsize=(7.5, 4.8))

        if len(numeric) >= 2:
            a, b = numeric[0], numeric[1]
            sub = self.df[[a, b]].dropna()
            if len(sub) > 3000:
                sub = sub.sample(3000, random_state=42)
            ax.scatter(sub[a], sub[b], s=14, alpha=0.35, color="#9C9286", label="valid rows")
            bad = self.df.loc[self.df.index.isin(bad_idx), [a, b]].dropna()
            ax.scatter(bad[a], bad[b], s=38, alpha=0.9, color="#BD4A32", marker="X",
                       label=f"contradictory ({len(bad_idx)})")
            ax.set_xlabel(a)
            ax.set_ylabel(b)
            # If ranges differ wildly (e.g. 1.7e14), a log axis makes the plot readable
            if (sub[a].abs().max() > 1e6) or (sub[b].abs().max() > 1e6):
                ax.set_xscale("symlog")
        else:
            c = numeric[0] if numeric else cols[0]
            s = self.df[c].dropna()
            if len(s) > 3000:
                s = s.sample(3000, random_state=42)
            if pd.api.types.is_numeric_dtype(s):
                ax.hist(s, bins=50, color="#9C9286", alpha=0.8)
                bad = self.df.loc[self.df.index.isin(bad_idx), c].dropna()
                if len(bad):
                    ax.scatter(bad, np.full(len(bad), ax.get_ylim()[1] * 0.05),
                               color="#BD4A32", marker="X", s=40, label="contradictory")
            else:
                vc = self.df[c].astype(str).value_counts().head(10)
                ax.bar(vc.index, vc.values, color="#9C9286")
                ax.set_ylabel("count")
            ax.set_xlabel(c)
            ok = self.df.loc[~self.df.index.isin(bad_idx), c].dropna()
            if pd.api.types.is_numeric_dtype(ok) and len(ok) > 2 and ok.std() > 0:
                ax.set_title(f"{rule['label']}")
        ax.legend(fontsize=8)
        ax.set_title(f"{rule['label']} - {rule['count']} flagged row(s)")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        self._save_plot(fig, name, rule["label"], "contradictions")

    def contradictions(self):
        cr = self.contradiction_report
        triggered = [r for r in cr["rules"] if r["count"] > 0]

        insights = []
        if not triggered:
            insights.append("No contradictory record detected by the domain rules.")
        else:
            insights.append(
                f"{cr['rows_flagged']:,} of {len(self.df):,} rows ({cr['rows_flagged_percent']}%) "
                f"contain at least one contradictory value; {cr['total_flags']} contradictions "
                f"in total. The worst row contains {cr['max_contradictions_in_one_row']}."
            )
            for r in triggered[:8]:
                examples = r["examples"][0] if r["examples"] else {}
                detail = ", ".join(f"{k}={v}" for k, v in examples.items() if k != "row")
                sev = "CRITICAL" if r["severity"] == "critical" else "warning"
                insights.append(
                    f"[{sev}] {r['label']}: {r['count']} row(s) ({r['percent']}%)"
                    + (f" - example row {examples.get('row')}: {detail}." if detail else ".")
                )
            if cr["rows_flagged_percent"] > 10:
                insights.append(
                    "Because more than 10% of rows are affected, these values are treated as "
                    "missing for the statistical analyses (see 'impossible_values_masked')."
                )

        # Rule-count bar chart
        if triggered:
            top = sorted(triggered, key=lambda r: r["count"])[-15:]
            fig, ax = plt.subplots(figsize=(10, max(3.2, 0.42 * len(top) + 1.5)))
            colors = ["#BD4A32" if r["severity"] == "critical" else "#E0A458" for r in top]
            ax.barh([r["label"] for r in top], [r["count"] for r in top], color=colors)
            for i, r in enumerate(top):
                ax.text(r["count"], i, f"  {r['count']} ({r['percent']}%)",
                        va="center", fontsize=8)
            ax.set_xlabel("flagged rows")
            ax.set_title(f"Contradictory / impossible records - {self.label}")
            self._save_plot(fig, "contradictions_rules", "Contradiction rules", "contradictions")

            # Row-level contradiction count
            per_row = np.asarray(cr["per_row_counts"], dtype=int)
            fig, ax = plt.subplots(figsize=(7, 3.8))
            vals, counts = np.unique(per_row, return_counts=True)
            ax.bar(vals, counts, color="#8C572A")
            ax.set_yscale("log")
            ax.set_xlabel("number of contradictions in a row")
            ax.set_ylabel("rows (log scale)")
            ax.set_title("How many contradictions per row?")
            self._save_plot(fig, "contradictions_per_row", "Contradictions per row", "contradictions")

            # Scatter for the most frequent rule with >= 2 columns
            for r in sorted(triggered, key=lambda x: -x["count"]):
                if len([c for c in r["columns"] if c in self.df.columns]) >= 2:
                    self._plot_contradiction_rule(r, f"contradiction_{slug(r['id'])}")
                    break

        return {
            "description": "Domain rules that detect logically / biologically contradictory records "
                           "(e.g. blood pressure = 0, cholesterol = 0, p-tau > total tau, "
                           "class 'no alcohol' with a positive alcohol measurement). "
                           "Flagged values are reported raw and are excluded from later statistics.",
            "summary": {
                "total_rules": cr["total_rules"],
                "triggered_rules": cr["triggered_rules"],
                "total_flags": cr["total_flags"],
                "rows_flagged": cr["rows_flagged"],
                "rows_flagged_percent": cr["rows_flagged_percent"],
                "max_contradictions_in_one_row": cr["max_contradictions_in_one_row"],
            },
            "rules": cr["rules"],
            "insights": insights,
        }

    # ---------------------------------------------------------- 3. missing + duplicates
    def missing_duplicates(self):
        n = len(self.raw)
        miss = self.raw.isna().sum()
        miss_cols = [{"name": c, "count": int(miss[c]),
                      "percent": round(float(miss[c]) / n * 100, 3)} for c in self.raw.columns]

        dup = self.raw.duplicated(keep="first")
        dup_count = int(dup.sum())
        dup_examples = records(self.raw.loc[dup].head(5)) if dup_count else []

        # Exactly duplicated feature columns
        dup_cols = []
        seen = {}
        for c in self.raw.columns:
            key = tuple(pd.util.hash_pandas_object(self.raw[c].astype(str), index=False))
            if key in seen:
                dup_cols.append({"column": c, "duplicate_of": seen[key]})
            else:
                seen[key] = c

        # Hidden missing: zeros where the valid range starts above zero
        hidden_zeros = {}
        for c in self.num_cols:
            rng = get_physical_range(self.dataset_key, c)
            if not rng or rng[0] <= 0:
                continue
            cnt = int((self.df[c] == 0).sum())
            if cnt:
                hidden_zeros[c] = cnt

        insights = []
        bad = sorted([m for m in miss_cols if m["count"] > 0], key=lambda m: -m["percent"])
        if not bad:
            insights.append("No explicit missing (NaN) values in the raw file.")
        else:
            insights.append(f"{sum(m['count'] for m in bad)} missing cells across {len(bad)} column(s).")
            for m in bad[:6]:
                kind = "categorical" if m["name"] in self.cat_cols else "numeric"
                fix = "mode / 'Unknown' label" if kind == "categorical" else "median or model-based imputation"
                insights.append(f"'{m['name']}': {m['count']} missing ({m['percent']}%) - suggested fix: {fix}.")

        if dup_count:
            insights.append(
                f"{dup_count} exact duplicate rows ({dup_count / n * 100:.2f}%). "
                "They can inflate correlations and cross-validation scores; drop_duplicates() "
                "is recommended unless they are genuine repeated measurements."
            )
        else:
            insights.append("No exact duplicate rows.")

        if dup_cols:
            insights.append(f"Exactly duplicated feature column(s): {dup_cols}.")

        if hidden_zeros:
            for c, cnt in hidden_zeros.items():
                insights.append(
                    f"Hidden missing data: '{c}' has {cnt} zero(s) ({cnt / n * 100:.1f}%) while "
                    f"the valid range is {get_physical_range(self.dataset_key, c)}. "
                    "These zeros are placeholders, not measurements."
                )

        # Missing-value plot
        if bad:
            fig, ax = plt.subplots(figsize=(9, max(3.2, 0.32 * len(bad) + 1)))
            ax.barh([m["name"] for m in bad][::-1], [m["percent"] for m in bad][::-1], color="#BD4A32")
            ax.set_xlabel("% missing")
            ax.set_title(f"Missing values per column - {self.label}")
            self._save_plot(fig, "missing_values", "Missing values per column", "missing_duplicates")

        # Hidden-zero plot
        if hidden_zeros:
            fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(hidden_zeros) + 1)))
            cols = sorted(hidden_zeros, key=lambda c: -hidden_zeros[c])
            ax.barh(cols[::-1], [hidden_zeros[c] for c in cols][::-1], color="#E0A458")
            ax.set_xlabel("zeros where zero is impossible")
            ax.set_title("Hidden missing values coded as 0")
            self._save_plot(fig, "hidden_missing_zeros", "Hidden missing zeros", "missing_duplicates")

        return {
            "description": "Completeness check: explicit NaN cells, exact duplicate rows and columns, "
                           "and hidden missing values coded as zero in physiologically positive features.",
            "missing": {"total_cells": int(miss.sum()), "columns": miss_cols},
            "hidden_missing_zeros": hidden_zeros,
            "duplicates": {"count": dup_count,
                           "percent": round(dup_count / n * 100, 3),
                           "examples": dup_examples},
            "duplicate_columns": dup_cols,
            "insights": insights,
        }

    # ---------------------------------------------------------- 4. outliers
    def outliers(self):
        rows = []
        for c in self.continuous_cols:
            s = self.df_valid[c].dropna()
            if len(s) < 5:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            iqr_mask = (s < lo) | (s > hi)
            z = np.abs(stats.zscore(s)) if s.std() > 0 else np.zeros(len(s))
            med = s.median()
            mad = robust_sigma(s)
            mod_z = np.abs(0.6745 * (s - med) / mad) if mad > 0 else np.zeros(len(s))
            p_lo, p_hi = s.quantile(0.01), s.quantile(0.99)
            p_mask = (s < p_lo) | (s > p_hi)
            rows.append({
                "feature": c, "min": float(s.min()), "max": float(s.max()),
                "q1": float(q1), "q3": float(q3), "iqr": float(iqr),
                "lower_fence": float(lo), "upper_fence": float(hi),
                "iqr_outliers": int(iqr_mask.sum()), "iqr_percent": round(float(iqr_mask.mean() * 100), 2),
                "z_outliers": int((z > 3).sum()), "z_percent": round(float((z > 3).mean() * 100), 2),
                "modified_z_outliers": int((mod_z > 3.5).sum()),
                "modified_z_percent": round(float((mod_z > 3.5).mean() * 100), 2),
                "percentile_outliers": int(p_mask.sum()),
                "percentile_percent": round(float(p_mask.mean() * 100), 2),
            })

        ml = detect_ml_outliers(self.df_valid[self.continuous_cols], contamination=0.05)
        insights = []
        if rows:
            top = max(rows, key=lambda r: r["iqr_percent"])
            insights.append(
                f"Most outlier-prone feature (IQR): '{top['feature']}' "
                f"({top['iqr_outliers']} points, {top['iqr_percent']}%)."
            )
            diff = [r["feature"] for r in rows if r["iqr_outliers"] > 3 * max(r["z_outliers"], 1)]
            if diff:
                insights.append(
                    f"IQR flags far more points than the Z-score for {diff}: heavy-tailed or skewed "
                    "distribution, so those extremes may be genuine."
                )

        if "error" not in ml:
            for name in ml.get("methods_used", []):
                m = ml["methods"][name]
                insights.append(f"{name.replace('_', ' ').title()}: {m['outlier_count']} anomalies "
                                f"({m['outlier_percent']}%).")
            c = ml["consensus_outliers"]
            insights.append(
                f"Consensus anomalie(s): {c['count']} samples ({c['percent']}%) flagged by at least "
                f"half of the {len(ml['methods_used'])} unsupervised detectors."
            )
            if ml["unanimous_outliers"]["count"]:
                insights.append(f"{ml['unanimous_outliers']['count']} samples are flagged by ALL detectors.")
            if ml.get("agreement"):
                best = ml["agreement"][0]
                insights.append(f"Most similar detectors: {best['a']} / {best['b']} "
                                f"(Jaccard {best['jaccard']:.2f}).")

        # Boxplots
        if self.continuous_cols:
            z_df = self.df_valid[self.continuous_cols].apply(
                lambda s: (s - s.mean()) / s.std() if s.std() else s
            )
            fig, ax = plt.subplots(figsize=(max(8, len(self.continuous_cols) * 0.7), 5))
            sns.boxplot(data=z_df, ax=ax, color="#E8DFCF", fliersize=2)
            ax.set_title(f"Standardised boxplots (IQR outliers) - {self.label}")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            self._save_plot(fig, "outliers_boxplots", "Standardised boxplots", "outliers")

        # Method comparison + vote histogram + PCA map
        if "error" not in ml:
            methods = ml["methods"]
            names = list(methods.keys())
            counts = [methods[n]["outlier_count"] for n in names]
            fig, ax = plt.subplots(figsize=(9, 4.2))
            ax.bar([n.replace("_", " ").title() for n in names], counts,
                   color=[PALETTE[i % len(PALETTE)] for i in range(len(names))])
            for i, v in enumerate(counts):
                ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
            ax.set_ylabel("flagged samples")
            ax.set_title(f"Unsupervised outlier detectors - {self.label}")
            plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
            self._save_plot(fig, "outliers_methods", "Outlier detector comparison", "outliers")

            votes = np.asarray(ml["vote_counts"])
            fig, ax = plt.subplots(figsize=(7, 3.8))
            vals, cnt = np.unique(votes, return_counts=True)
            ax.bar(vals, cnt, color="#2B5E4B")
            ax.set_xlabel("number of detectors flagging the sample")
            ax.set_ylabel("samples")
            ax.set_title(f"Detector agreement - {self.label}")
            self._save_plot(fig, "outliers_votes", "Detector agreement", "outliers")

            X, cols = _clean_numeric_matrix(self.df_valid[self.continuous_cols])
            if X is not None and X.shape[1] >= 2:
                pca = PCA(n_components=2, random_state=42)
                xy = pca.fit_transform(X)
                consensus = np.zeros(len(X), dtype=bool)
                for i in ml["consensus_outliers"]["indices"]:
                    consensus[i] = True
                unanimous = np.zeros(len(X), dtype=bool)
                for i in ml["unanimous_outliers"]["indices"]:
                    unanimous[i] = True
                fig, ax = plt.subplots(figsize=(7.5, 5.5))
                ax.scatter(xy[~consensus, 0], xy[~consensus, 1], s=14, alpha=0.35,
                           color="#9C9286", label="normal")
                ax.scatter(xy[consensus, 0], xy[consensus, 1], s=32, alpha=0.85,
                           color="#BD4A32", label=f"consensus ({consensus.sum()})")
                ax.scatter(xy[unanimous, 0], xy[unanimous, 1], s=70, facecolors="none",
                           edgecolors="#2B2621", linewidths=1.2, label=f"unanimous ({unanimous.sum()})")
                ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
                ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
                ax.set_title(f"PCA anomaly map - {self.label}")
                ax.legend(fontsize=8)
                self._save_plot(fig, "outliers_pca", "PCA anomaly map", "outliers")

            # Feature-level Isolation Forest
            fl = ml.get("feature_level_isolation_forest", {})
            if fl:
                top = sorted(fl.items(), key=lambda kv: -kv[1]["outliers"])[:15]
                fig, ax = plt.subplots(figsize=(9, max(3.4, 0.35 * len(top) + 1)))
                ax.barh([k for k, _ in top][::-1], [v["outliers"] for _, v in top][::-1],
                        color="#8C572A")
                ax.set_xlabel("feature-level anomalies")
                ax.set_title("Which features drive the anomalies?")
                self._save_plot(fig, "outliers_feature_drivers", "Feature-level anomalies", "outliers")

        consensus = ml.get("consensus_outliers", {}) if "error" not in ml else {}
        unanimous = ml.get("unanimous_outliers", {}) if "error" not in ml else {}
        return {
            "description": "Classical univariate methods (IQR fences, Z-score, modified Z-score / MAD, "
                           "1-99% percentile) plus a consensus of six unsupervised multivariate "
                           "detectors (Isolation Forest, One-Class SVM, LOF, Elliptic Envelope, "
                           "DBSCAN, robust Mahalanobis).",
            "summary": {
                "features_checked": len(rows),
                "methods_used": len(ml.get("methods_used", [])),
                "consensus_count": consensus.get("count"),
                "consensus_percent": consensus.get("percent"),
                "unanimous_count": unanimous.get("count"),
                "mean_iqr_percent": round(float(np.mean([r["iqr_percent"] for r in rows])), 2)
                if rows else None,
                "max_iqr_percent": max((r["iqr_percent"] for r in rows), default=None),
            },
            "classical_features": rows,
            "machine_learning_outliers": ml,
            "insights": insights,
        }

    # ---------------------------------------------------------- 5. noise
    def _local_residual(self, col: str, nb: np.ndarray) -> np.ndarray:
        raw = self.df[col].to_numpy(dtype=float)
        rng = get_physical_range(self.dataset_key, col)
        if rng:
            raw = np.where((raw < rng[0]) | (raw > rng[1]), np.nan, raw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            expected = np.nanmedian(raw[nb], axis=1) if nb.size else np.full(len(raw), np.nan)
        return raw - expected

    def noise(self):
        rows = []
        residual_cache: dict[str, np.ndarray] = {}

        zero_inflated = [
            c for c in self.continuous_cols
            if self.df[c].notna().sum() >= 30 and self.df[c].nunique(dropna=True) > 10
            and float((self.df[c] == 0).mean()) >= 0.5
        ]
        cols = [c for c in self.continuous_cols
                if self.df[c].notna().sum() >= 30 and self.df[c].nunique(dropna=True) > 10
                and c not in zero_inflated]

        nb = np.zeros((0, 0), dtype=int)
        if cols:
            nb = knn_neighbor_indices(self.df_valid[cols], k=KNN_NEIGHBOURS)[0]
            if nb.size and nb.shape[0] != len(self.df):
                nb = np.zeros((0, 0), dtype=int)

        for c in cols:
            s = self.df[c]
            rng = get_physical_range(self.dataset_key, c)
            m = compute_noise_metrics(c, s, valid_range=rng,
                                      neighbour_idx=nb if nb.size else None)
            rows.append(m)
            if nb.size:
                residual_cache[c] = self._local_residual(c, nb)

        noisy = [r for r in rows if r["has_noise"]]
        errs = [r for r in rows if r["has_errors"]]

        insights = []
        insights.append(
            "Noise-to-signal is estimated from a local k-NN smoother in feature space "
            f"(k={KNN_NEIGHBOURS}), because these tables are not time series: row order is "
            "arbitrary, so the weather second-difference formula would measure row order."
        )
        if zero_inflated:
            insights.append(
                f"Skipped zero-inflated feature(s) {zero_inflated}: with >=50% zeros, "
                "k-NN residuals and Hampel spikes describe the zero/non-zero split rather than "
                "measurement noise."
            )
        if noisy:
            insights.append(f"Moderate/high random noise in {len(noisy)} feature(s): "
                            f"{[r['feature'] for r in noisy]}.")
        else:
            insights.append("No moderate/high random noise detected in the tested continuous features.")
        if errs:
            insights.append(f"Data-integrity problems in {len(errs)} feature(s): "
                            f"{[r['feature'] for r in errs]}.")
            for r in errs[:5]:
                insights.append(f"'{r['feature']}': " + "; ".join(r["error_reasons"]))
        else:
            insights.append("No out-of-range values or stuck readings among the tested features.")

        by_nsr = sorted(rows, key=lambda r: -r["noise_to_signal"])
        if by_nsr:
            insights.append(
                f"Noisiest: '{by_nsr[0]['feature']}' (NSR {by_nsr[0]['noise_to_signal']:.2f}, "
                f"SNR {by_nsr[0]['snr_db']} dB); cleanest: '{by_nsr[-1]['feature']}' "
                f"(NSR {by_nsr[-1]['noise_to_signal']:.2f})."
            )
        good = [r["feature"] for r in rows if not (r["has_noise"] or r["has_errors"])]
        if good:
            insights.append(f"Clean features: {good}.")

        # NSR bar chart
        if rows:
            fig, ax = plt.subplots(figsize=(9, max(3.2, 0.38 * len(rows) + 1)))
            colors = {"negligible": "#2B5E4B", "low": "#8DAA91",
                      "moderate": "#E0A458", "high": "#BD4A32"}
            srt = sorted(rows, key=lambda r: r["noise_to_signal"])
            ax.barh([r["feature"] for r in srt], [r["noise_to_signal"] for r in srt],
                    color=[colors.get(r["verdict"], "#8DAA91") for r in srt])
            for lv in NSR_LEVELS:
                ax.axvline(lv, color="#2B2621", ls=":", lw=1)
            ax.set_xlim(0, max(0.5, min(1.0, by_nsr[0]["noise_to_signal"] * 1.15)) if by_nsr else 0.5)
            ax.set_xlabel("noise-to-signal ratio (0 = smooth, 1 = pure noise)")
            ax.set_title(f"Noise level per feature - {self.label}")
            self._save_plot(fig, "noise_levels", "Noise level per feature", "noise")

        # Residual histograms for the worst features
        show: list[str] = []
        for r in by_nsr[:3]:
            if r["feature"] not in show:
                show.append(r["feature"])
        for r in errs:
            if r["feature"] not in show:
                show.append(r["feature"])
        for c in show[:4]:
            res = residual_cache.get(c)
            if res is None or np.isfinite(res).sum() < 10:
                continue
            sig = robust_sigma(res)
            fig, ax = plt.subplots(figsize=(6.5, 3.8))
            ax.hist(res[np.isfinite(res)], bins=60, color="#8C572A", alpha=0.85)
            if sig > 0:
                for k in (-SPIKE_K, SPIKE_K):
                    ax.axvline(k * sig, color="#BD4A32", ls="--", lw=1)
            ax.set_title(f"Local residual - {c} (dashed = spike limit)")
            ax.set_xlabel("reading - median of similar records")
            self._save_plot(fig, f"noise_residual_{slug(c)}", f"Noise residual - {c}", "noise")

        # NSR vs signal spread (why some features are naturally noisy)
        if len(rows) >= 3:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.scatter([r["signal_sigma"] for r in rows], [r["noise_to_signal"] for r in rows],
                       s=42, color="#5B7C99")
            for r in rows:
                ax.annotate(r["feature"], (r["signal_sigma"], r["noise_to_signal"]),
                            fontsize=7, xytext=(3, 3), textcoords="offset points")
            ax.set_xscale("symlog")
            ax.set_xlabel("signal spread (robust sigma)")
            ax.set_ylabel("noise-to-signal ratio")
            ax.set_title("Noise vs natural spread")
            self._save_plot(fig, "noise_vs_signal", "Noise vs signal spread", "noise")

        return {
            "description": "Noise-to-signal analysis for tabular data: local k-NN residual noise "
                           "divided by the robust signal spread, SNR(dB), Hampel spikes, repeated "
                           "readings and out-of-range values. Each feature receives a verdict.",
            "method": {
                "noise_to_signal": (
                    "Robust sigma of the residual between each reading and the median of its "
                    f"{KNN_NEIGHBOURS} nearest neighbours in standardised feature space, divided "
                    "by the robust sigma of the feature. 0 = perfectly predictable / smooth, "
                    "1 = as large as the signal itself. This is the tabular adaptation of the "
                    "weather second-difference NSR (row order is meaningless in these tables)."
                ),
                "snr_db": "10*log10(signal variance / noise variance); > 20 dB is very clean, < 5 dB is noisy.",
                "spikes": f"Residual larger than {SPIKE_K} robust sigmas and "
                          f"{int(SPIKE_MIN_FRAC * 100)}% of the signal spread.",
                "stuck_sensor": f">= {STUCK_RUN} identical consecutive readings on a continuous feature.",
                "out_of_range": "Values outside the biological / physical range; masked before stats.",
            },
            "thresholds": noise_levels(),
            "features": rows,
            "summary": {
                "features_checked": len(rows),
                "features_excluded": [c for c in self.continuous_cols if c not in cols],
                "zero_inflated_excluded": zero_inflated,
                "noisy_features": [r["feature"] for r in noisy],
                "error_features": [r["feature"] for r in errs],
                "noisy_feature_percent": round(len(noisy) / len(rows) * 100, 2) if rows else 0.0,
                "error_feature_percent": round(len(errs) / len(rows) * 100, 2) if rows else 0.0,
                "mean_noise_to_signal": round(float(np.mean([r["noise_to_signal"] for r in rows])), 4)
                if rows else None,
            },
            "insights": insights,
        }

    # ---------------------------------------------------------- 6. correlation
    def _vif(self) -> list[dict]:
        cols = [c for c in self.num_cols if self.df_valid[c].notna().sum() > 10]
        if len(cols) < 2:
            return []
        corr = self.df_valid[cols].corr().to_numpy()
        # Drop perfectly collinear columns for a stable pseudo-inverse
        try:
            inv = np.linalg.pinv(corr)
            vif = np.diag(inv)
        except Exception:
            return []
        out = [{"feature": c, "vif": round(float(v), 3)} for c, v in zip(cols, vif)]
        return sorted(out, key=lambda x: -x["vif"])

    def correlation(self):
        cols = [c for c in self.num_cols if self.df_valid[c].notna().sum() > 2]
        insights = []
        if len(cols) < 2:
            return {"description": "Correlation analysis", "insights": ["Not enough numeric columns."]}

        pear = self.df_valid[cols].corr()
        spear = self.df_valid[cols].corr(method="spearman")

        pairs = []
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                r, rs = pear.loc[a, b], spear.loc[a, b]
                if pd.isna(r):
                    continue
                strength = ("very strong" if abs(r) >= 0.8 else "strong" if abs(r) >= 0.6
                            else "moderate" if abs(r) >= 0.4 else "weak" if abs(r) >= 0.2 else "negligible")
                pairs.append({
                    "a": a, "b": b, "pearson": float(r), "spearman": float(rs),
                    "strength": strength, "direction": "positive" if r > 0 else "negative",
                    "abs_pearson": abs(float(r)),
                })
        pairs.sort(key=lambda p: -p["abs_pearson"])

        redundant = [p for p in pairs if p["abs_pearson"] >= 0.85]
        for p in redundant[:10]:
            insights.append(f"Redundancy / collinearity: '{p['a']}' vs '{p['b']}' (r = {p['pearson']:.2f}). "
                            "Keep only one of them in a linear model.")
        for p in pairs[:5]:
            insights.append(f"{p['strength'].capitalize()} {p['direction']} correlation: "
                            f"'{p['a']}' vs '{p['b']}' (r = {p['pearson']:.2f}, "
                            f"rho = {p['spearman']:.2f}).")

        nonlin = [p for p in pairs if abs(p["spearman"] - p["pearson"]) > 0.15]
        for p in nonlin[:4]:
            insights.append(f"'{p['a']}' vs '{p['b']}': Pearson {p['pearson']:.2f} but Spearman "
                            f"{p['spearman']:.2f} - monotonic but non-linear (or outlier-driven).")
        weak = [c for c in cols if pear[c].drop(c, errors="ignore").abs().max() < 0.15]
        if weak:
            insights.append(f"Almost independent of every other feature: {weak}.")

        vif = self._vif()
        high_vif = [v for v in vif if v["vif"] >= 10][:8]
        if high_vif:
            insights.append(f"High VIF (>= 10, multicollinearity): "
                            f"{[(v['feature'], v['vif']) for v in high_vif]}.")

        # Pearson heatmap
        annot = len(cols) <= 16
        fig, ax = plt.subplots(figsize=(max(7, len(cols) * 0.72), max(5, len(cols) * 0.62)))
        sns.heatmap(pear, annot=annot, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1,
                    linewidths=0.5, ax=ax, annot_kws={"size": 7})
        ax.set_title(f"Pearson correlation - {self.label}")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        self._save_plot(fig, "correlation_pearson", "Pearson correlation heatmap", "correlation")

        # Spearman heatmap
        fig, ax = plt.subplots(figsize=(max(7, len(cols) * 0.72), max(5, len(cols) * 0.62)))
        sns.heatmap(spear, annot=annot, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1,
                    linewidths=0.5, ax=ax, annot_kws={"size": 7})
        ax.set_title(f"Spearman rank correlation - {self.label}")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        self._save_plot(fig, "correlation_spearman", "Spearman correlation heatmap", "correlation")

        # Clustered heatmap: visually groups redundant blocks
        if len(cols) >= 3:
            try:
                cg = sns.clustermap(pear.fillna(0), cmap="coolwarm", center=0, vmin=-1, vmax=1,
                                    figsize=(max(8, len(cols) * 0.72), max(7, len(cols) * 0.62)),
                                    linewidths=0.4)
                cg.figure.suptitle(f"Clustered correlation - {self.label}", y=1.02)
                self._save_plot(cg.figure, "correlation_clustered", "Clustered correlation", "correlation")
            except Exception:
                plt.close("all")

        # Top pairs bar chart
        top = pairs[:12]
        if top:
            fig, ax = plt.subplots(figsize=(9, max(3.5, 0.4 * len(top) + 1)))
            labels = [f"{p['a']} / {p['b']}" for p in top][::-1]
            ax.barh(labels, [p["pearson"] for p in top][::-1],
                    color=["#2B5E4B" if p["pearson"] >= 0 else "#BD4A32" for p in top][::-1])
            ax.axvline(0, color="#2B2621", lw=1)
            ax.set_xlabel("Pearson r")
            ax.set_title("Strongest feature pairs")
            self._save_plot(fig, "correlation_top_pairs", "Strongest correlation pairs", "correlation")

        # Scatter of the 2 strongest pairs
        for p in pairs[:2]:
            sub = self.df_valid[[p["a"], p["b"]]].dropna()
            if len(sub) > 4000:
                sub = sub.sample(4000, random_state=42)
            fig, ax = plt.subplots(figsize=(6.5, 4.8))
            sns.regplot(data=sub, x=p["a"], y=p["b"], ax=ax,
                        scatter_kws={"s": 14, "alpha": 0.35},
                        line_kws={"color": "#BD4A32"})
            ax.set_title(f"{p['a']} vs {p['b']} (r = {p['pearson']:.2f})")
            self._save_plot(fig, f"correlation_scatter_{slug(p['a'])}__{slug(p['b'])}",
                            f"Scatter {p['a']} vs {p['b']}", "correlation")

        # Correlation with the target
        target_corr = []
        if self.target_present:
            y = _numeric(self.df[self.target_col])
            if y.nunique(dropna=True) > 1:
                for c in self.continuous_cols:
                    x = _numeric(self.df_valid[c])
                    pair = pd.concat([x, y], axis=1).dropna()
                    if len(pair) > 5 and pair.iloc[:, 0].std() > 0 and pair.iloc[:, 1].std() > 0:
                        r = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
                        if np.isfinite(r):
                            target_corr.append({"feature": c, "correlation": round(r, 4),
                                                "abs_correlation": abs(r)})
                target_corr.sort(key=lambda x: -x["abs_correlation"])
                if target_corr:
                    top_t = target_corr[:15]
                    fig, ax = plt.subplots(figsize=(9, max(3.4, 0.38 * len(top_t) + 1)))
                    ax.barh([r["feature"] for r in top_t][::-1],
                            [r["correlation"] for r in top_t][::-1],
                            color=["#2B5E4B" if r["correlation"] >= 0 else "#BD4A32"
                                   for r in top_t][::-1])
                    ax.axvline(0, color="#2B2621", lw=1)
                    ax.set_xlabel("Pearson correlation with target")
                    ax.set_title(f"Feature vs target '{self.target_col}'")
                    self._save_plot(fig, "correlation_with_target", "Correlation with target", "correlation")
                    insights.append(f"Strongest target correlates: "
                                    f"{[(r['feature'], r['correlation']) for r in target_corr[:5]]}.")

                    # Class-conditional distributions for the 3 strongest features
                    if y.nunique(dropna=True) <= 6:
                        for r in target_corr[:3]:
                            d = pd.concat([self.df_valid[[r["feature"]]], y.rename("target")], axis=1).dropna()
                            if d["target"].nunique() < 2:
                                continue
                            fig, ax = plt.subplots(figsize=(6.5, 4.2))
                            sns.boxplot(data=d, x="target", y=r["feature"], ax=ax, color="#E8DFCF")
                            sns.stripplot(data=d, x="target", y=r["feature"], ax=ax,
                                          color="#BD4A32", size=2, alpha=0.25)
                            ax.set_title(f"{r['feature']} by '{self.target_col}'")
                            self._save_plot(fig, f"correlation_target_{slug(r['feature'])}",
                                            f"{r['feature']} by target", "correlation")

        # Categorical group summary
        by_category = {}
        if self.group_cols:
            g = self.group_cols[0]
            if self.df[g].nunique() <= 12:
                grp = self.df_valid.groupby(self.df[g], dropna=False)[self.continuous_cols].mean(numeric_only=True)
                by_category = {"category": g, "means": records(grp.reset_index())}
                for c in self.continuous_cols[:3]:
                    d = self.df_valid[[g, c]].dropna()
                    if d[g].nunique() < 2:
                        continue
                    fig, ax = plt.subplots(figsize=(6.5, 4.2))
                    sns.boxplot(data=d, x=g, y=c, ax=ax, color="#E8DFCF")
                    ax.set_title(f"{c} by {g}")
                    self._save_plot(fig, f"correlation_group_{slug(c)}", f"{c} by {g}", "correlation")

        return {
            "description": "Pearson (linear) and Spearman (rank) correlation matrices, clustered "
                           "correlation blocks, top pairs, target correlation, VIF and "
                           "class-conditional distributions.",
            "pearson": pear.round(4).to_dict(),
            "spearman": spear.round(4).to_dict(),
            "top_pairs": pairs[:15],
            "redundant_pairs": redundant,
            "vif": vif,
            "target_correlation": target_corr,
            "by_category": by_category,
            "insights": insights,
        }

    # ---------------------------------------------------------- 7. KDE
    def kde(self):
        feats, skipped = [], []
        cols = self.continuous_cols or self.num_cols
        for c in cols:
            s = self.df_valid[c].dropna()
            if len(s) < 5 or s.nunique() < 3:
                skipped.append({"feature": c, "reason": "constant or too few distinct values"})
                continue

            fig, ax = plt.subplots(figsize=(6, 3.9))
            sns.kdeplot(s, fill=True, color="#8C572A", ax=ax, warn_singular=False)
            ax.axvline(s.mean(), color="#2B5E4B", ls="--", label=f"mean {s.mean():.2f}")
            ax.axvline(s.median(), color="#BD4A32", ls=":", label=f"median {s.median():.2f}")
            ax.set_title(f"KDE - {c}")
            ax.legend(fontsize=8)
            self._save_plot(fig, f"kde_{slug(c)}", f"KDE of {c}", "kde")

            sk = float(s.skew())
            shape = "right-skewed" if sk > 0.5 else "left-skewed" if sk < -0.5 else "roughly symmetric"
            try:
                dens = stats.gaussian_kde(s)(np.linspace(s.min(), s.max(), 200))
                modes = int(len(find_peaks(dens, prominence=0.10 * dens.max())[0]))
            except Exception:
                modes = 0

            feats.append({
                "feature": c, "mean": round(float(s.mean()), 3), "median": round(float(s.median()), 3),
                "std": round(float(s.std()), 3), "skew": round(sk, 3),
                "kurtosis": round(float(s.kurt()), 3), "n_peaks": modes, "shape": shape,
            })

        # Class-conditional KDE for the strongest target correlates
        if self.target_present and self.df[self.target_col].nunique(dropna=True) <= 6:
            tc = self.report.get("correlation", {}).get("target_correlation", [])[:3]
            for r in tc:
                c = r["feature"]
                d = pd.concat([self.df_valid[c], _numeric(self.df[self.target_col]).rename("target")],
                              axis=1).dropna()
                if d["target"].nunique() < 2 or len(d) < 20:
                    continue
                fig, ax = plt.subplots(figsize=(6.5, 4))
                for i, (cls, sub) in enumerate(d.groupby("target")):
                    if len(sub) > 3:
                        sns.kdeplot(sub[c], fill=True, alpha=0.35, ax=ax,
                                    color=PALETTE[i % len(PALETTE)], label=f"class {cls:g}")
                ax.set_title(f"KDE of {c} by '{self.target_col}'")
                ax.legend(fontsize=8)
                self._save_plot(fig, f"kde_by_target_{slug(c)}", f"KDE of {c} by target", "kde")

        insights = []
        for f in feats[:8]:
            extra = " (multimodal - possible subgroups)" if f["n_peaks"] > 1 else ""
            insights.append(f"'{f['feature']}' is {f['shape']} (skew {f['skew']:.2f}, "
                            f"{f['n_peaks']} peak(s)){extra}.")
        return {
            "description": "Kernel density estimates of continuous features with mean/median markers, "
                           "modality detection, skewness and (for the strongest target correlates) "
                           "class-conditional densities.",
            "features": feats,
            "skipped": skipped,
            "insights": insights,
        }

    # ---------------------------------------------------------- 8. model phase
    def model_phase(self):
        target = self.target_col
        desc = ("Model phase: target balance, sklearn Pipeline + ColumnTransformer, Random Forest "
                "on a stratified hold-out with 5-fold cross-validation, confusion matrix, and "
                "impurity / permutation / mutual-information feature importances.")
        if not self.target_present:
            return {"description": desc, "insights": [f"Target column '{target}' not found."]}

        insights = []
        s_target = self.df[target]
        total = len(s_target)
        counts_raw = s_target.value_counts(dropna=False)
        class_counts = {("Missing" if pd.isna(k) else str(k)): int(v) for k, v in counts_raw.items()}
        class_pcts = {k: round(v / total * 100, 2) for k, v in class_counts.items()}

        clean = self.df_valid[self.df_valid[target].notna()].copy()
        y = clean[target].astype(str)
        if y.nunique() < 2:
            return {"description": desc, "insights": [f"Target '{target}' has a single class."]}

        non_null = s_target.dropna().value_counts()
        imbalance_ratio = round(float(non_null.iloc[0]) / max(float(non_null.iloc[1]), 1), 2) \
            if len(non_null) >= 2 else None
        is_balanced = bool(imbalance_ratio is not None and imbalance_ratio <= 1.5)
        severity = ("balanced" if is_balanced else "moderate imbalance"
                    if imbalance_ratio and imbalance_ratio <= 3 else "severe imbalance")
        insights.append(f"Target '{target}': {class_counts} ({severity}, ratio {imbalance_ratio}:1).")

        # Target-balance plot
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        keys = list(class_counts.keys())
        vals = [class_counts[k] for k in keys]
        ax.bar(keys, vals, color=PALETTE[:len(keys)])
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v}\n({class_pcts[keys[i]]}%)", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"Target balance - {target}")
        ax.set_ylabel("samples")
        self._save_plot(fig, "model_target_balance", "Target class balance", "model_phase")

        # Preprocessing
        num_features = [c for c in self.num_cols if c in clean.columns]
        cat_features = [c for c in self.cat_cols if c in clean.columns]
        X = clean[num_features + cat_features]
        num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                             ("scaler", StandardScaler())])
        cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                             ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
        preprocessor = ColumnTransformer(
            transformers=[("num", num_pipe, num_features), ("cat", cat_pipe, cat_features)],
            remainder="drop",
        )

        # Stratified hold-out
        stratify = y if y.value_counts().min() >= 2 else None
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify)
        rf = RandomForestClassifier(n_estimators=200, max_depth=14, random_state=42,
                                    n_jobs=-1, class_weight="balanced")
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", rf)]).fit(Xtr, ytr)
        preds = pipe.predict(Xte)
        classes = sorted(y.unique())

        acc = float(accuracy_score(yte, preds))
        bal = float(balanced_accuracy_score(yte, preds))
        f1m = float(f1_score(yte, preds, average="macro", zero_division=0))
        recalls = recall_score(yte, preds, labels=classes, average=None, zero_division=0)
        baseline = float((yte == ytr.mode().iloc[0]).mean()) if len(ytr.mode()) else 0.0
        cm = confusion_matrix(yte, preds, labels=classes)

        roc_auc = None
        if len(classes) == 2:
            try:
                proba = pipe.predict_proba(Xte)[:, 1]
                roc_auc = float(roc_auc_score((yte == classes[1]).astype(int), proba))
            except Exception:
                roc_auc = None

        cv_mean = cv_std = None
        try:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(pipe, X, y, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
            cv_mean, cv_std = float(np.mean(scores)), float(np.std(scores))
        except Exception:
            pass

        insights.append(
            f"Hold-out: balanced accuracy {bal:.3f}, macro F1 {f1m:.3f}, accuracy {acc:.3f} "
            f"(majority baseline {baseline:.3f})"
            + (f"; ROC-AUC {roc_auc:.3f}" if roc_auc is not None else "")
            + (f"; 5-fold CV balanced accuracy {cv_mean:.3f} +/- {cv_std:.3f}." if cv_mean is not None else ".")
        )

        # Confusion matrix plot
        fig, ax = plt.subplots(figsize=(5.8, 4.8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.set_title(f"Confusion matrix - {target}")
        self._save_plot(fig, "model_confusion_matrix", "Confusion matrix", "model_phase")

        # Importances
        feat_names = list(pipe.named_steps["preprocessor"].get_feature_names_out())
        raw_imp = pipe.named_steps["classifier"].feature_importances_
        feats = num_features + cat_features
        orig = {f: 0.0 for f in feats}
        for nm, imp in zip(feat_names, raw_imp):
            for f in feats:
                if nm == f"num__{f}" or nm == f"cat__{f}" or nm.startswith(f"cat__{f}_"):
                    orig[f] += float(imp)
                    break
        impurity = [{"rank": i + 1, "feature": f, "importance": round(v, 5),
                     "percentage": round(v * 100, 2)}
                    for i, (f, v) in enumerate(sorted(orig.items(), key=lambda kv: -kv[1]))]

        pi = permutation_importance(pipe, Xte, yte, scoring="balanced_accuracy",
                                    n_repeats=5, random_state=42, n_jobs=-1)
        order = np.argsort(-pi.importances_mean)
        permutation = [{"rank": r + 1, "feature": Xte.columns[j],
                        "drop_points": round(float(pi.importances_mean[j]) * 100, 3),
                        "std_points": round(float(pi.importances_std[j]) * 100, 3)}
                       for r, j in enumerate(order)]

        # Mutual information on the transformed space
        mutual = []
        try:
            Xt = pipe.named_steps["preprocessor"].transform(X)
            mi = mutual_info_classif(Xt, y, discrete_features=False, random_state=42)
            agg: dict[str, float] = {}
            for nm, v in zip(feat_names, mi):
                base = re.sub(r"^(num__|cat__)", "", nm)
                # For one-hot columns, keep the full dummy name; numeric names stay clean
                agg[base] = agg.get(base, 0.0) + float(v)
            mutual = [{"rank": i + 1, "feature": f, "mutual_information": round(v, 5)}
                      for i, (f, v) in enumerate(sorted(agg.items(), key=lambda kv: -kv[1])[:20])]
        except Exception:
            mutual = []

        # Importance plots
        for data, key, title, color, fname in (
            (impurity, "percentage", "Impurity importance (Random Forest)", "#2B5E4B", "model_importance_impurity"),
            (permutation, "drop_points", "Permutation importance (test set)", "#BD4A32", "model_importance_permutation"),
            (mutual, "mutual_information", "Mutual information", "#8C572A", "model_importance_mutual_info"),
        ):
            top = data[:15]
            if not top:
                continue
            fig, ax = plt.subplots(figsize=(9, max(3.4, 0.38 * len(top) + 1)))
            ax.barh([d["feature"] for d in top][::-1], [d[key] for d in top][::-1], color=color)
            ax.set_xlabel(key.replace("_", " "))
            ax.set_title(title)
            self._save_plot(fig, fname, title, "model_phase")

        if impurity:
            insights.append(f"Top impurity predictors: "
                            f"{[(d['feature'], d['percentage']) for d in impurity[:5]]}.")
        if permutation:
            insights.append(f"Top permutation impact: '{permutation[0]['feature']}' "
                            f"({permutation[0]['drop_points']:.2f} points of balanced accuracy).")
        if mutual:
            insights.append(f"Top mutual-information features: "
                            f"{[(d['feature'], d['mutual_information']) for d in mutual[:5]]}.")

        # Leakage / spurious predictor checks
        if bal >= 0.99:
            insights.append("WARNING: near-perfect score (>= 99%). Check for target leakage or a "
                            "duplicated target-derived feature.")
        if impurity and impurity[0]["percentage"] > 60:
            insights.append(f"'{impurity[0]['feature']}' carries {impurity[0]['percentage']:.1f}% of "
                            "the impurity importance - verify it is a legitimate predictor.")
        if self.dataset_key == "parkinson" and "Subject_id" not in self.df.columns:
            pass  # identifier already dropped in the subclass

        return {
            "description": desc,
            "target_balance": {
                "target_feature": target, "is_balanced": is_balanced,
                "imbalance_ratio": imbalance_ratio, "severity": severity,
                "total_samples": total, "valid_samples": int(len(clean)),
                "class_counts": class_counts, "class_percentages": class_pcts,
            },
            "preprocessing": {
                "numerical_features": num_features,
                "categorical_features": cat_features,
                "pipeline": "Median imputation + StandardScaler (numeric); most-frequent "
                            "imputation + OneHotEncoder (categorical).",
                "train_test_split": "stratified 75% / 25%",
            },
            "evaluation": {
                "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
                "accuracy": round(acc, 4), "balanced_accuracy": round(bal, 4),
                "f1_macro": round(f1m, 4), "baseline_accuracy": round(baseline, 4),
                "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
                "cv_balanced_accuracy_mean": round(cv_mean, 4) if cv_mean is not None else None,
                "cv_balanced_accuracy_std": round(cv_std, 4) if cv_std is not None else None,
                "per_class_recall": {str(c): round(float(r), 4) for c, r in zip(classes, recalls)},
                "confusion_matrix": cm.tolist(), "classes": classes,
            },
            "feature_importance": {
                "model": "RandomForestClassifier(n_estimators=200, class_weight='balanced')",
                "impurity_ranking": impurity,
                "permutation_ranking": permutation,
                "mutual_information_ranking": mutual,
            },
            "insights": insights,
        }

    # ---------------------------------------------------------- 9. data quality
    def data_quality(self):
        """Consolidated Phase-I indicators, each expressed as a percentage."""
        meta = self.report["meta"]
        n_rows, n_cols = meta["n_rows"], meta["n_cols"]
        total_cells = max(n_rows * n_cols, 1)

        md = self.report["missing_duplicates"]
        explicit_missing = int(md["missing"]["total_cells"])
        hidden_zeros = int(sum(md.get("hidden_missing_zeros", {}).values()))
        missing_cells = explicit_missing + hidden_zeros
        missing_pct = 100 * missing_cells / total_cells

        dup_count = int(md["duplicates"]["count"])
        dup_pct = float(md["duplicates"]["percent"])

        cr = self.report["contradictions"]["summary"]
        contradiction_pct = float(cr["rows_flagged_percent"])

        invalid_cells = int(sum(meta.get("impossible_values_masked", {}).values()))
        invalid_pct = 100 * invalid_cells / total_cells

        noise = self.report["noise"]["summary"]
        tested = int(noise.get("features_checked", 0))
        noisy_n = len(noise.get("noisy_features", []))
        error_n = len(noise.get("error_features", []))
        noise_pct = 100 * noisy_n / tested if tested else 0.0

        out = self.report["outliers"].get("summary", {})
        outlier_count = int(out.get("consensus_count") or 0)
        outlier_pct = float(out.get("consensus_percent") or 0.0)

        score = 100 - (
            0.30 * missing_pct + 0.10 * dup_pct + 0.20 * contradiction_pct
            + 0.15 * invalid_pct + 0.10 * noise_pct + 0.15 * outlier_pct
        )
        score = round(float(np.clip(score, 0, 100)), 1)

        def status(pct: float, good: float = 1.0, warn: float = 5.0) -> str:
            return "good" if pct <= good else "warning" if pct <= warn else "critical"

        indicators = [
            {"indicator": "Completeness", "value": round(100 - missing_pct, 2), "unit": "%",
             "issue_percent": round(missing_pct, 2), "status": status(missing_pct),
             "detail": f"{missing_cells} effective missing cells "
                       f"({explicit_missing} NaN + {hidden_zeros} hidden zeros)"},
            {"indicator": "Duplicate rows", "value": round(dup_pct, 2), "unit": "%",
             "issue_percent": round(dup_pct, 2), "status": status(dup_pct),
             "detail": f"{dup_count} exact duplicate row(s)"},
            {"indicator": "Contradictory rows", "value": round(contradiction_pct, 2), "unit": "%",
             "issue_percent": round(contradiction_pct, 2), "status": status(contradiction_pct),
             "detail": f"{cr['rows_flagged']} rows / {cr['total_flags']} contradictions"},
            {"indicator": "Out-of-range cells", "value": round(invalid_pct, 2), "unit": "%",
             "issue_percent": round(invalid_pct, 2), "status": status(invalid_pct),
             "detail": f"{invalid_cells} values outside the physical range (masked)"},
            {"indicator": "Noisy features", "value": round(noise_pct, 2), "unit": "%",
             "issue_percent": round(noise_pct, 2), "status": status(noise_pct, 20, 50),
             "detail": f"{noisy_n}/{tested} tested continuous features flagged"},
            {"indicator": "Outlier samples", "value": round(outlier_pct, 2), "unit": "%",
             "issue_percent": round(outlier_pct, 2), "status": status(outlier_pct, 5, 10),
             "detail": f"{outlier_count} samples flagged by the unsupervised consensus"},
        ]

        quality_label = "good" if score >= 90 else "moderate" if score >= 75 else "poor"
        weakest = max(indicators, key=lambda i: i["issue_percent"])
        insights = [
            f"Overall data quality score: {score}/100 ({quality_label}).",
            f"Completeness {100 - missing_pct:.2f}% | duplicates {dup_pct:.2f}% | "
            f"contradictions {contradiction_pct:.2f}% | out-of-range {invalid_pct:.2f}%.",
            f"Noise affects {noise_pct:.1f}% of tested features ({noisy_n}/{tested}); "
            f"{outlier_pct:.2f}% of samples are consensus outliers.",
            f"Weakest indicator: {weakest['indicator']} ({weakest['issue_percent']}%).",
        ]

        # Indicator bar chart (issue percentages; lower is better)
        fig, ax = plt.subplots(figsize=(9, 4.0))
        colors = {"good": "#2B5E4B", "warning": "#E0A458", "critical": "#BD4A32"}
        order = [i for i in indicators if i["indicator"] != "Overall quality score"]
        ax.barh([i["indicator"] for i in order][::-1],
                [i["issue_percent"] for i in order][::-1],
                color=[colors.get(i["status"], "#8DAA91") for i in order][::-1])
        for idx, i in enumerate(order[::-1]):
            ax.text(i["issue_percent"], idx, f"  {i['issue_percent']}%", va="center", fontsize=8)
        ax.set_xlabel("affected share (%) - lower is better")
        ax.set_title(f"Data quality indicators - {self.label}")
        self._save_plot(fig, "data_quality_indicators", "Data quality indicators", "data_quality")

        return {
            "description": "Consolidated Phase-I quality indicators: completeness, duplicates, "
                           "contradictions, out-of-range values, noisy features and outlier samples, "
                           "all expressed as percentages and combined into one quality score.",
            "quality_score": score,
            "quality_label": quality_label,
            "indicators": indicators,
            "dashboard": {
                "dataset": self.label,
                "rows": n_rows,
                "columns": n_cols,
                "quality_score": score,
                "quality_label": quality_label,
                "contradictions_percent": contradiction_pct,
                "noise_percent": round(noise_pct, 2),
                "outliers_percent": round(outlier_pct, 2),
            },
            "insights": insights,
        }

    # ---------------------------------------------------- 10. interpretation
    def interpretation(self):
        """Turn the EDA findings into an interpretation and an action plan."""
        dq = self.report["data_quality"]
        md = self.report["missing_duplicates"]
        cr = self.report["contradictions"]["summary"]
        noise = self.report["noise"]["summary"]
        out = self.report["outliers"].get("summary", {})
        corr = self.report["correlation"]
        model = self.report["model_phase"]
        ev = model.get("evaluation", {})
        fi = model.get("feature_importance", {})
        imp = (fi.get("impurity_ranking") or [None])[0]
        perm = (fi.get("permutation_ranking") or [None])[0]
        top_pairs = corr.get("top_pairs", [])
        top_pair = top_pairs[0] if top_pairs else None
        target = model.get("target_balance", {})
        hidden = md.get("hidden_missing_zeros", {})
        invalid_cells = int(sum(self.report["meta"].get("impossible_values_masked", {}).values()))

        findings = [f"Quality score: {dq['quality_score']}/100 ({dq['quality_label']})."]
        if cr["rows_flagged"]:
            findings.append(
                f"{cr['rows_flagged']} contradictory rows ({cr['rows_flagged_percent']}%) "
                f"across {cr['triggered_rules']} triggered rule(s).")
        if hidden:
            worst = max(hidden, key=hidden.get)
            findings.append(
                f"Hidden missing values: '{worst}' has {hidden[worst]} zero placeholder(s).")
        if noise.get("noisy_features"):
            findings.append(
                f"{len(noise['noisy_features'])} feature(s) show noise "
                f"({noise.get('noisy_feature_percent', 0)}% of those tested); "
                f"mean noise-to-signal {noise.get('mean_noise_to_signal')}.")
        if out.get("consensus_count"):
            findings.append(
                f"{out['consensus_count']} samples ({out['consensus_percent']}%) are consensus "
                f"outliers across {out.get('methods_used')} unsupervised detectors.")
        if top_pair:
            findings.append(
                f"Strongest association: '{top_pair['a']}' / '{top_pair['b']}' "
                f"(Pearson r={top_pair['pearson']:.2f}, Spearman {top_pair['spearman']:.2f}).")
        if imp:
            findings.append(
                f"Top model driver (impurity): '{imp['feature']}' "
                f"({imp['percentage']}% of importance).")
        if ev.get("balanced_accuracy") is not None:
            findings.append(
                f"Predictive check: balanced accuracy {ev['balanced_accuracy']:.3f}, "
                f"macro-F1 {ev.get('f1_macro')}, CV {ev.get('cv_balanced_accuracy_mean')}.")

        recommendations: list[dict] = []

        def add(priority: str, area: str, finding: str, action: str) -> None:
            recommendations.append({"priority": priority, "area": area,
                                    "finding": finding, "action": action})

        if cr["rows_flagged"]:
            add("high", "Contradictions",
                f"{cr['rows_flagged']} row(s), {cr['rows_flagged_percent']}% of the file.",
                "Correct the source values where possible; otherwise keep them masked as missing "
                "and document the domain rule that was violated.")
        if hidden:
            worst = max(hidden, key=hidden.get)
            add("high", "Hidden missing values",
                f"'{worst}' contains {hidden[worst]} zeros where zero is not a valid measurement.",
                "Re-code these zeros as NaN before imputation; never let a placeholder enter "
                "scaling, correlation or model training.")
        if md["missing"]["total_cells"]:
            add("high", "Explicit missing values",
                f"{md['missing']['total_cells']} NaN cell(s) across the file.",
                "Impute numeric features with the median and categorical features with the mode "
                "(already done inside the sklearn Pipeline); drop columns above 50% missing.")
        if md["duplicates"]["count"]:
            dup_priority = "high" if md["duplicates"]["percent"] >= 10 else "medium"
            add(dup_priority, "Duplicate rows",
                f"{md['duplicates']['count']} exact duplicate row(s) "
                f"({md['duplicates']['percent']}%).",
                "Drop exact duplicates before cross-validation unless they are genuine repeated "
                "measurements; otherwise scores become optimistic.")
        if invalid_cells:
            add("medium", "Out-of-range values",
                f"{invalid_cells} value(s) outside the physical range were masked.",
                "Add the same range checks to the ingestion step and reject/log invalid readings "
                "instead of silently accepting them.")
        if noise.get("noisy_features"):
            add("medium", "Noise",
                f"{noise.get('noisy_feature_percent')}% of tested features are flagged as noisy.",
                "Review the measurement protocol for the highest-NSR features; aggregate repeated "
                "readings or apply smoothing, and prefer robust scaling/tree models.")
        if out.get("consensus_count"):
            add("medium", "Outliers",
                f"{out.get('consensus_percent')}% of samples are consensus outliers.",
                "Inspect them with the domain expert; use robust models or winsorisation only "
                "when extremes are confirmed to be measurement errors.")
        strong = [p for p in top_pairs if abs(p["pearson"]) >= 0.8]
        if strong:
            p = strong[0]
            add("medium", "Redundant features",
                f"'{p['a']}' and '{p['b']}' are strongly correlated (r={p['pearson']:.2f}).",
                "Keep one of the pair, combine them, or use PCA / regularisation to avoid "
                "unstable importance and multicollinearity.")
        if target.get("severity") not in (None, "balanced"):
            add("medium", "Class imbalance",
                f"Target imbalance ratio {target.get('imbalance_ratio')} "
                f"({target.get('severity')}).",
                "Use class weights and stratified CV (already configured) and report balanced "
                "accuracy / macro-F1 rather than raw accuracy.")
        if imp:
            add("low", "Feature importance",
                f"'{imp['feature']}' is the strongest impurity driver; "
                f"permutation top is '{perm['feature'] if perm else 'n/a'}'.",
                "Retain the top drivers but confirm they are available at prediction time; "
                "drop features whose permutation importance is near zero.")
        if ev.get("balanced_accuracy") is not None and ev["balanced_accuracy"] < 0.80:
            add("high", "Model performance",
                f"Balanced accuracy is only {ev['balanced_accuracy']:.3f}.",
                "Collect more data, simplify the feature set, or revisit the label definition "
                "before deploying the model.")

        priority_rank = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda r: (priority_rank.get(r["priority"], 3), r["area"]))

        applied = [
            {"step": "Physical range validation", "status": "applied",
             "reason": f"{invalid_cells} impossible reading(s) masked before statistics"},
            {"step": "Contradiction masking", "status": "applied",
             "reason": f"{cr['rows_flagged']} contradictory row(s) excluded from later statistics"},
            {"step": "Median / mode imputation", "status": "applied",
             "reason": "Numeric median and categorical most-frequent imputation in the Pipeline"},
            {"step": "StandardScaler + OneHotEncoder", "status": "applied",
             "reason": "Preprocessing inside the sklearn ColumnTransformer"},
            {"step": "Stratified hold-out + cross-validation", "status": "applied",
             "reason": "75/25 stratified split and 5-fold CV"},
        ]
        pending = [
            {"step": "Drop exact duplicates", "status": "recommended",
             "reason": f"{md['duplicates']['count']} duplicate row(s) detected"},
            {"step": "Re-code hidden zeros as missing", "status": "recommended",
             "reason": f"{sum(hidden.values())} placeholder zero(s) detected"},
            {"step": "Remove redundant features", "status": "recommended",
             "reason": f"{len(strong)} pair(s) with |r| >= 0.8"},
            {"step": "Outlier review / robust modelling", "status": "recommended",
             "reason": f"{out.get('consensus_count', 0)} consensus outlier(s)"},
        ]

        summary = " ".join(findings[:4])
        if recommendations:
            top = recommendations[0]
            next_action = f"First action ({top['priority']}): {top['action']}"
        else:
            next_action = "No blocking data-quality action detected."
        insights = findings + [next_action]

        # Recommendation priority chart
        if recommendations:
            counts = {p: sum(1 for r in recommendations if r["priority"] == p)
                      for p in ("high", "medium", "low")}
            fig, ax = plt.subplots(figsize=(6.5, 3.6))
            ax.bar(list(counts), list(counts.values()),
                   color=["#BD4A32", "#E0A458", "#2B5E4B"])
            for idx, v in enumerate(counts.values()):
                ax.text(idx, v, str(v), ha="center", va="bottom", fontsize=9)
            ax.set_ylabel("recommendations")
            ax.set_title(f"Action plan by priority - {self.label}")
            self._save_plot(fig, "interpretation_priorities", "Recommendations by priority",
                            "interpretation")

        return {
            "description": "Interpretation of the EDA results and the concrete actions to apply "
                           "from the data description: what to fix, what to mask, what to keep for "
                           "modelling, and which checks are already active in the Pipeline.",
            "summary": summary,
            "findings": findings,
            "recommendations": recommendations,
            "preprocessing_plan": applied + pending,
            "insights": insights,
        }

    # ---------------------------------------------------------- run / save
    def run(self):
        self.load()
        stamp = datetime.now()
        cr = self.contradiction_report
        self.report = {
            "meta": {
                "report_id": f"{slug(self.dataset_key)}_{stamp:%Y%m%d_%H%M%S}",
                "dataset_name": self.dataset_key,
                "dataset_label": self.label,
                "dataset_description": self.dataset_description,
                "csv_path": str(self.csv_path),
                "generated_at": stamp.isoformat(timespec="seconds"),
                "n_rows": len(self.raw), "n_cols": self.raw.shape[1],
                "numeric_columns": self.num_cols,
                "continuous_columns": self.continuous_cols,
                "categorical_columns": self.cat_cols,
                "target_column": self.target_col,
                "impossible_values_masked": self.invalid_counts,
                "contradictory_rows": cr["rows_flagged"],
                "contradictory_rows_percent": cr["rows_flagged_percent"],
            }
        }
        self._section("description", self.describe)
        self._section("contradictions", self.contradictions)
        self._section("missing_duplicates", self.missing_duplicates)
        self._section("outliers", self.outliers)
        self._section("noise", self.noise)
        self._section("correlation", self.correlation)
        self._section("kde", self.kde)
        self._section("model_phase", self.model_phase)
        # These two aggregate the sections above; they are re-ordered below so a
        # Phase-I reader sees the teacher's flow: quality -> noise -> outliers ->
        # interpretation.
        self._section("data_quality", self.data_quality)
        self._section("interpretation", self.interpretation)
        self.report["plots"] = self.plots
        order = [
            "meta", "description", "data_quality", "contradictions", "missing_duplicates",
            "noise", "outliers", "correlation", "kde", "model_phase", "interpretation",
            "plots",
        ]
        self.report = {k: self.report[k] for k in order if k in self.report}
        return clean_for_json(self.report)

    def save(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        rep = clean_for_json(self.report)

        # Remove PNGs left over from an older run so the folder always matches
        # the plots listed in this report (renamed/deleted plots do not linger).
        plots_dir = self.out_dir / "plots"
        if plots_dir.exists():
            keep = {f"{p['name']}.png" for p in rep["plots"]}
            for old in plots_dir.glob("*.png"):
                if old.name not in keep:
                    old.unlink()

        out_file = self.out_dir / "report.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)

        rid = rep["meta"]["report_id"]
        core = {k: v for k, v in rep.items() if k != "plots"}
        core["_id"] = rid
        db_docs = {
            "reports": [core],
            "plots": [{"_id": f"{rid}__{p['name']}", "report_id": rid, **p} for p in rep["plots"]],
        }
        with open(self.out_dir / "db_documents.json", "w", encoding="utf-8") as f:
            json.dump(db_docs, f, ensure_ascii=False)

        # Also store the report (sections JSON + PNG blobs) in data.db.
        try:
            self.db_report_id = DataDB().save_report(rep, keep_last=1)
        except Exception as exc:  # a DB problem must not lose the JSON output
            self.db_report_id = None
            print(f"[warn] {rid}: could not store report in data.db: {exc}")
        return str(out_file)


# --------------------------------------------------------------------------- #
# Dataset-specific subclasses
# --------------------------------------------------------------------------- #
class CovidEDAModel(BaseDatasetEDAModel):
    label = "COVID-19 Laboratory"
    dataset_description = (
        "Laboratory measurements from 1,736 patients: blood counts, liver/kidney markers, "
        "electrolytes and inflammatory markers. Target SARSCov = 0 (negative) / 1 (positive). "
        "Several columns contain missing codes (blank CK, zero ages) and one impossible calcium value."
    )

    def __init__(self, csv_path: str = "datasets_extracted/Covid19/dataset2-covid19.csv",
                 out_dir: str = "eda_output_covid19"):
        super().__init__(csv_path=csv_path, dataset_key="covid", target_col="SARSCov",
                         category_cols=["Sex"], out_dir=out_dir)


class HeartEDAModel(BaseDatasetEDAModel):
    """Comprehensive heart-attack / heart-disease cohort (1190 patients)."""
    label = "Heart Attack (comprehensive)"
    dataset_description = (
        "Combined Cleveland / Hungary / Switzerland heart-disease cohort (1,190 patients). "
        "Clinical predictors: age, sex, chest-pain type, resting blood pressure, cholesterol, "
        "fasting blood sugar, resting ECG, maximum heart rate, exercise angina, oldpeak and "
        "ST slope. Target 1 = heart disease present. Cholesterol = 0 and blood pressure = 0 "
        "are placeholder codes, not measurements."
    )

    def __init__(self,
                 csv_path: str = "datasets_extracted/heartdiseasedatasetcomprehensive/heart_statlog_cleveland_hungary_final.csv",
                 out_dir: str = "eda_output_heart"):
        super().__init__(
            csv_path=csv_path, dataset_key="heart", target_col="target",
            category_cols=["sex", "chest pain type", "fasting blood sugar",
                           "resting ecg", "exercise angina", "ST slope"],
            out_dir=out_dir,
        )


class HeartAttackEDAModel(BaseDatasetEDAModel):
    """Classic UCI heart-attack file (303 patients, 14 columns)."""
    label = "Heart Attack (UCI)"

    def __init__(self, csv_path: str = "datasets_extracted/Heart Attack Analysis/heart.csv",
                 out_dir: str = "eda_output_heartattack"):
        super().__init__(
            csv_path=csv_path, dataset_key="heartattack", target_col="output",
            category_cols=["sex", "cp", "fbs", "restecg", "exng", "slp", "caa", "thall"],
            out_dir=out_dir,
        )


class ParkinsonEDAModel(BaseDatasetEDAModel):
    label = "Parkinson's Disease"

    def __init__(self, csv_path: str = "datasets_extracted/parkinson/parkinson/data1.csv",
                 out_dir: str = "eda_output_parkinson"):
        super().__init__(
            csv_path=csv_path, dataset_key="parkinson",
            target_col="Group (Normal/PD/SWEDD - 1/2/3)",
            drop_cols=["Subject_id"], out_dir=out_dir,
        )


class AlcoholEDAModel(BaseDatasetEDAModel):
    label = "Alcohol Detection"

    def __init__(self, csv_path: str = "datasets_extracted/alcohol/alcohol/alcohol.csv",
                 out_dir: str = "eda_output_alcohol"):
        super().__init__(csv_path=csv_path, dataset_key="alcohol", target_col="CLASS",
                         out_dir=out_dir)
