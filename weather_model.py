"""
weather_model.py  -  MODEL layer (MVC) for the Weather EDA project.

Responsibility : load the weather CSV, run the analysis, produce plots and a
                 JSON-serialisable report. NO UI code here (no plt.show(), no Flask).
Controller use : model = WeatherEDAModel("weather.csv"); report = model.run(); model.save()
View use       : read eda_output/report.json  (or the db_documents.json collections)

Pipeline (same order as the assignment)
  1. Description   : df.info(), df.head(), df.describe()
  2. Missing values + duplicates
  3. Outliers      : IQR fences + Z-score
  4. Noise         : robust noise-to-signal ratio, SNR (dB), von Neumann smoothness, spike (Hampel)
                     detection, stuck sensors, impossible values -> verdict per feature
  4b. Monthly trends: your resample("MS") mean/q10/q90/std bands + month-by-month lines
  5. Correlation   : Pearson + Spearman, heatmap, top pairs, scatter of strongest
                     pairs, histograms per weather 'Summary' (FacetGrid)
  6. KDE           : one density plot per numeric feature
"""

import io
import re
import json
import base64
import warnings
import calendar
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # headless: Model never opens windows
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score

from quality import (SPIKE_K, SPIKE_PCT_FLAG, SPIKE_MIN_FRAC, STUCK_RUN, NSR_LEVELS, CIRCULAR_KEYS,
                     robust_sigma, physical_range, mask_impossible)

MONTH_STYLE = {                             # from your plot_month / label_color code
    1: ("January", "blue"),    2: ("February", "green"),   3: ("March", "orange"),
    4: ("April", "gold"),      5: ("May", "red"),          6: ("June", "violet"),
    7: ("July", "purple"),     8: ("August", "black"),     9: ("September", "brown"),
    10: ("October", "darkblue"), 11: ("November", "grey"), 12: ("December", "pink"),
}


# ------------------------------------------------------------------ utils
def clean(o):
    """Recursively turn numpy / pandas objects into plain JSON types (NaN -> None)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (np.isnan(o) or np.isinf(o)) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if o is pd.NaT:
        return None
    return o


def records(df):
    """DataFrame -> list of dicts (NaN -> None, timestamps -> ISO strings)."""
    return json.loads(df.to_json(orient="records", date_format="iso"))


def slug(text):
    return re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()


# ------------------------------------------------------------------ model
class WeatherEDAModel:
    def __init__(self, csv_path, out_dir="eda_output", category_col="Summary", target_col="Precip Type",
                 top_categories=8, dpi=150, embed_base64=True):
        self.csv_path = Path(csv_path)
        self.out_dir = Path(out_dir)
        self.plots_dir = self.out_dir / "plots"
        self.category_col = category_col
        self.target_col = target_col
        self.top_categories = top_categories
        self.dpi = dpi
        self.embed_base64 = embed_base64
        self.plots = []
        self.report = {}

    # ---------------------------------------------------------- loading
    def load(self):
        self.raw = pd.read_csv(self.csv_path)
        df = self.raw.copy()
        self.date_col = None
        for c in df.columns:
            if any(k in c.lower() for k in ("date", "time")):
                parsed = pd.to_datetime(df[c], utc=True, errors="coerce")
                if parsed.notna().mean() > 0.8:
                    self.date_col = c
                    df[c] = parsed.dt.tz_convert(None)
                    break
        if self.date_col:
            df = df.dropna(subset=[self.date_col]).sort_values(self.date_col)
            df = df.set_index(self.date_col)
        self.df = df
        all_num = df.select_dtypes(include="number").columns.tolist()
        self.constant_cols = [c for c in all_num if df[c].nunique(dropna=True) <= 1]
        self.num_cols = [c for c in all_num if c not in self.constant_cols]
        self.cat_cols = df.select_dtypes(exclude="number").columns.tolist()
        # df_valid = same data with physically impossible readings set to NaN (e.g. pressure = 0).
        # Outliers, correlation, KDE and monthly analysis use it so bad readings cannot distort them.
        self.df_valid, self.invalid_counts = mask_impossible(df)
        # key features used by the assignment snippets (fallback: first numeric cols)
        pref = [c for c in ("Temperature (C)", "Humidity") if c in self.num_cols]
        self.key_feats = pref + [c for c in self.num_cols if c not in pref][: 2 - len(pref)]

    # ---------------------------------------------------------- plotting
    def _save_plot(self, fig, name, title, section):
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        path = self.plots_dir / f"{name}.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        entry = {"name": name, "title": title, "section": section,
                 "file": str(path), "mime": "image/png"}
        if self.embed_base64:
            entry["base64"] = base64.b64encode(path.read_bytes()).decode()
        self.plots.append(entry)

    def _section(self, key, fn):
        """Run one analysis step and attach the names of the plots it produced."""
        before = len(self.plots)
        result = fn()
        result["plots"] = [p["name"] for p in self.plots[before:]]
        self.report[key] = result

    # ---------------------------------------------------------- 1. description
    def describe(self):
        buf = io.StringIO()
        self.raw.info(buf=buf)
        desc_num = self.df_valid[self.num_cols + self.constant_cols].describe().T   # valid readings only (pressure = 0 would give std 117)
        desc_num["skew_valid"] = self.df_valid[desc_num.index].skew()
        desc_num["kurtosis_valid"] = self.df_valid[desc_num.index].kurt()
        desc_cat = self.df[self.cat_cols].describe().T if self.cat_cols else pd.DataFrame()
        insights = [
            f"The dataset has {len(self.raw):,} rows and {self.raw.shape[1]} columns "
            f"({len(self.num_cols) + len(self.constant_cols)} numeric, {len(self.cat_cols)} categorical).",
        ]
        if self.date_col:
            idx = self.df.index
            insights.append(f"Time range: {idx.min():%Y-%m-%d} to {idx.max():%Y-%m-%d} "
                            f"(column '{self.date_col}', used as the time index).")
        if self.constant_cols:
            insights.append(f"Constant column(s) with no information: {self.constant_cols} "
                            "- excluded from correlation, KDE and noise analysis.")
        skewed = [c for c in desc_num.index if abs(desc_num.loc[c, "skew_valid"]) > 1 and c not in self.constant_cols]
        if skewed:
            insights.append(f"Strongly skewed features (|skew| > 1, impossible values excluded): {skewed}.")
        for c, k in self.invalid_counts.items():
            insights.append(f"'{c}' contains {k} physically impossible readings - they are excluded from "
                            "outlier, correlation, KDE and monthly analysis (see the Noise section).")
        return {
            "description": "Structure of the dataset: dtypes and non-null counts (info), "
                           "first rows (head) and summary statistics (describe).",
            "info": {
                "text": buf.getvalue(),
                "memory_kb": round(self.raw.memory_usage(deep=True).sum() / 1024, 2),
                "columns": [{"name": c, "dtype": str(self.raw[c].dtype),
                             "non_null": int(self.raw[c].notna().sum()),
                             "n_unique": int(self.raw[c].nunique())} for c in self.raw.columns],
            },
            "head": records(self.raw.head(10)),
            "describe_numeric": records(desc_num.reset_index().rename(columns={"index": "feature"})),
            "describe_categorical": records(desc_cat.reset_index().rename(columns={"index": "feature"}))
            if len(desc_cat) else [],
            "constant_columns": self.constant_cols,
            "insights": insights,
        }

    # ---------------------------------------------------------- 2. missing + duplicates
    def missing_duplicates(self):
        n = len(self.raw)
        miss = self.raw.isna().sum()
        miss_cols = [{"name": c, "count": int(miss[c]), "percent": round(miss[c] / n * 100, 3)}
                     for c in self.raw.columns]
        dup = self.raw.duplicated(keep="first")
        insights = []
        bad = sorted([m for m in miss_cols if m["count"] > 0], key=lambda m: -m["percent"])
        if not bad:
            insights.append("No missing values.")
        for m in bad:
            kind = "categorical" if m["name"] in self.cat_cols else "numeric"
            fix = "fill with the mode or an 'Unknown' label" if kind == "categorical" \
                else "interpolate over time or fill with the median"
            insights.append(f"'{m['name']}' has {m['count']} missing values ({m['percent']}%) - "
                            f"{kind}; suggested fix: {fix}.")
        insights.append(f"{int(dup.sum())} fully duplicated rows ({dup.mean() * 100:.3f}%)"
                        + (" - remove with drop_duplicates()." if dup.any() else "."))
        if bad:
            fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(bad))))
            ax.barh([m["name"] for m in bad][::-1], [m["percent"] for m in bad][::-1], color="#BD4A32")
            ax.set_xlabel("% missing")
            ax.set_title("Missing values per column")
            self._save_plot(fig, "missing_values", "Missing values per column", "missing_duplicates")
        return {
            "description": "Completeness check: missing cells per column and exact duplicate rows.",
            "missing": {"total_cells": int(miss.sum()), "columns": miss_cols},
            "duplicates": {"count": int(dup.sum()), "percent": round(dup.mean() * 100, 3),
                           "row_indexes": self.raw.index[dup].tolist()[:500]},
            "insights": insights,
        }

    # ---------------------------------------------------------- 3. outliers
    def outliers(self):
        rows = []
        for c in self.num_cols:
            s = self.df_valid[c].dropna()
            q1, q3 = s.quantile(.25), s.quantile(.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            iqr_mask = (s < lo) | (s > hi)
            z = np.abs(stats.zscore(s))
            rows.append({"feature": c, "min": s.min(), "max": s.max(), "q1": q1, "q3": q3, "iqr": iqr,
                         "lower_fence": lo, "upper_fence": hi,
                         "iqr_outliers": int(iqr_mask.sum()), "iqr_percent": round(iqr_mask.mean() * 100, 3),
                         "z_outliers": int((z > 3).sum()), "z_percent": round((z > 3).mean() * 100, 3)})
        insights = []
        flagged = sorted([r for r in rows if r["iqr_percent"] > 0], key=lambda r: -r["iqr_percent"])
        if flagged:
            top = flagged[0]
            insights.append(f"Most outlier-prone feature: '{top['feature']}' "
                            f"({top['iqr_outliers']} IQR outliers, {top['iqr_percent']}%).")
        clean_f = [r["feature"] for r in rows if r["iqr_outliers"] == 0]
        if clean_f:
            insights.append(f"No IQR outliers in: {clean_f}.")
        diff = [r["feature"] for r in rows if r["iqr_outliers"] > 5 * max(r["z_outliers"], 1)]
        if diff:
            insights.append(f"IQR flags far more points than Z-score for {diff}: skewed or heavy-tailed "
                            "distributions - check them before deleting anything (weather extremes are real).")
        if self.num_cols:
            z_df = (self.df_valid[self.num_cols] - self.df_valid[self.num_cols].mean()) / self.df_valid[self.num_cols].std()
            fig, ax = plt.subplots(figsize=(max(8, len(self.num_cols) * 1.0), 5))
            sns.boxplot(data=z_df, ax=ax, color="#E8DFCF", fliersize=2)
            ax.set_title("Box plots (standardised) - points beyond whiskers are IQR outliers")
            plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
            self._save_plot(fig, "boxplots", "Box plots of all numeric features", "outliers")
        return {
            "description": "Extreme values flagged with two methods: IQR fences (Q1-1.5*IQR, Q3+1.5*IQR) "
                           "and Z-score (|z| > 3).",
            "features": rows, "insights": insights,
        }

    # ---------------------------------------------------------- 4. noise
    def _noise_metrics(self, col, s):
        """All noise tests for one feature.
        Returns (metrics, expected, spike_mask, residual, resid_sigma). Impossible readings are
        excluded from the noise tests and reported separately as data errors."""
        raw = s.to_numpy(dtype=float)
        n = len(raw)
        rng = physical_range(col, s)
        bad = ((raw < rng[0]) | (raw > rng[1])) if rng else np.zeros(n, dtype=bool)
        v = np.where(bad, np.nan, raw)                       # valid readings only
        ok = ~np.isnan(v)

        # (a) NOISE LEVEL - window-free estimator. For 'smooth signal + random noise' the 2nd difference
        #     x[t+1] - 2x[t] + x[t-1] cancels the signal and leaves sqrt(6) * noise_sigma.
        #     MAD makes it robust, so spikes / bad readings do not inflate the estimate.
        scale = robust_sigma(v[ok])                          # robust spread of the signal itself
        if scale == 0:
            scale = float(np.nanstd(v, ddof=1))
        d2 = v[2:] - 2 * v[1:-1] + v[:-2]
        sigma = robust_sigma(d2[~np.isnan(d2)]) / np.sqrt(6)
        nsr = float(min(sigma / scale, 1.0)) if scale > 0 else 0.0
        var_sig = scale ** 2 - sigma ** 2
        snr_db = float(10 * np.log10(var_sig / sigma ** 2)) if sigma > 0 and var_sig > 0 else None

        # (b) SMOOTHNESS - von Neumann ratio (~0 smooth, ~2 pure white noise) and lag-1 autocorrelation
        var = float(np.nanvar(v))
        vn = float(np.nanmean(np.diff(v) ** 2) / var) if var > 0 else 0.0
        pair = ok[:-1] & ok[1:]
        lag1 = float(np.corrcoef(v[:-1][pair], v[1:][pair])[0, 1])

        # (c) SPIKES - a reading that (1) is far from the median of its 4 neighbours (itself excluded),
        #     (2) is large compared with the feature's normal spread, and (3) is ISOLATED: the neighbours
        #     before and after agree with each other, so it is a one-point blip and not a real step change
        #     (weather front, wrap-around, level shift).
        nb = np.full((n, 4), np.nan)
        for j, k in enumerate((-2, -1, 1, 2)):
            if k < 0:
                nb[-k:, j] = v[:k]
            else:
                nb[:-k, j] = v[k:]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            expected = np.nanmedian(nb, axis=1)
            left, right = np.nanmean(nb[:, :2], axis=1), np.nanmean(nb[:, 2:], axis=1)
        r = v - expected
        rs = robust_sigma(r[~np.isnan(r)])
        a = np.abs(np.nan_to_num(r))
        mask = (a > max(SPIKE_K * rs, SPIKE_MIN_FRAC * scale)) & (np.abs(right - left) < 0.5 * a)
        a = a * mask   # keep only the strongest point of each cluster (non-maximum suppression, +/-2 points)
        mask &= a >= pd.Series(a).rolling(5, center=True, min_periods=1).max().to_numpy()
        pos = [p for p in np.argsort(-a)[:5] if mask[p]]
        examples = [{"at": s.index[p], "value": raw[p], "expected": expected[p]} for p in pos]

        # (d) STUCK SENSOR - longest run of identical consecutive readings (continuous features only)
        run_id = np.cumsum(np.r_[True, raw[1:] != raw[:-1]])
        counts = np.bincount(run_id)
        longest = int(counts.max())
        stuck_val = float(raw[np.where(run_id == counts.argmax())[0][0]])
        stuck = bool(longest >= STUCK_RUN and s.nunique() > 50)

        # (e) IMPOSSIBLE VALUES - outside the physical range of the measured quantity
        n_range = int(bad.sum())

        verdict = ("negligible" if nsr < NSR_LEVELS[0] else "low" if nsr < NSR_LEVELS[1]
                   else "moderate" if nsr < NSR_LEVELS[2] else "high")
        spike_pct = round(mask.mean() * 100, 3)
        noise_reasons, error_reasons = [], []
        if nsr >= NSR_LEVELS[1]:
            noise_reasons.append(f"{verdict} random fluctuation (noise is {nsr * 100:.0f}% of the signal's spread)")
        if spike_pct > SPIKE_PCT_FLAG:
            noise_reasons.append(f"{int(mask.sum())} isolated spikes ({spike_pct}%)")
        if n_range:
            error_reasons.append(f"{n_range} physically impossible values (valid range {rng[0]} to {rng[1]})")
        if stuck:
            error_reasons.append(f"stuck at {stuck_val:.4g} for {longest} consecutive readings "
                                 "(probably a sensor outage)")
        metrics = {
            "feature": col, "n": n, "noise_sigma": sigma, "noise_to_signal": nsr, "snr_db": snr_db,
            "verdict": verdict, "von_neumann": vn, "lag1_autocorr": lag1,
            "spikes": int(mask.sum()), "spike_percent": spike_pct, "spike_examples": examples,
            "longest_flat_run": longest, "stuck_sensor": stuck, "stuck_value": stuck_val,
            "valid_range": list(rng) if rng else None, "out_of_range": n_range,
            "has_noise": bool(noise_reasons), "has_errors": bool(error_reasons),
            "noise_reasons": noise_reasons, "error_reasons": error_reasons,
        }
        idx = s.index
        return (metrics, pd.Series(expected, index=idx), pd.Series(mask, index=idx),
                pd.Series(r, index=idx), rs)

    def _plot_noise_view(self, col, s, smooth, mask, resid, rs, verdict):
        start = len(s) // 2
        seg = slice(start, start + 336)                      # ~14 days of hourly data
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
        x, y, sm, mk = s.index[seg], s.values[seg], smooth.values[seg], mask.values[seg]
        a1.plot(x, y, color="#9C9286", lw=1, label="raw")
        a1.plot(x, sm, color="#8C572A", lw=1.8, label="expected (median of 4 neighbours)")
        a1.scatter(x[mk], y[mk], color="#BD4A32", zorder=3, s=25, label="spike")
        a1.set_title(f"{col} - zoom on a 2-week window")
        a1.legend(loc="lower right", fontsize=8)
        plt.setp(a1.get_xticklabels(), rotation=25, ha="right")
        res = resid.dropna()
        if rs > 0:
            res = res.clip(-8 * rs, 8 * rs)
        a2.hist(res, bins=60, color="#8C572A", alpha=.8)
        if rs > 0:
            for k in (-SPIKE_K, SPIKE_K):
                a2.axvline(k * rs, color="#BD4A32", ls="--")
        a2.set_title("Residual (raw - expected); dashed = spike limit")
        fig.suptitle(f"Noise check: {col}  [{verdict}]", y=1.03, fontweight="bold")
        self._save_plot(fig, f"noise_view_{slug(col)}", f"Noise check - {col}", "noise")

    def noise(self):
        rows, cache, skipped = [], {}, []
        for c in self.num_cols:
            if any(k in c.lower() for k in CIRCULAR_KEYS):
                skipped.append({"feature": c, "reason": "circular variable (0 and 360 degrees are the same "
                                                        "direction), so linear noise tests are not meaningful"})
                continue
            s = self.df[c].dropna()
            if len(s) < 30:
                continue
            m, smooth, mask, resid, rs = self._noise_metrics(c, s)
            rows.append(m)
            cache[c] = (s, smooth, mask, resid, rs)
        out = {
            "description": "Is there noise in the data, and how much? Each numeric feature is tested for random "
                           "fluctuation and isolated spikes (= NOISE) and for stuck sensors and physically "
                           "impossible readings (= DATA ERRORS), then given a verdict.",
            "method": {
                "noise_to_signal": "Robust noise sigma (MAD of the 2nd difference / sqrt(6)) divided by the "
                                   "feature's robust spread (MAD-based, immune to bad readings). 0 = perfectly "
                                   "smooth, 1 = pure random noise. Window-free, so the daily and seasonal cycles "
                                   "are NOT counted as noise.",
                "snr_db": "10*log10(signal variance / noise variance). Above 20 dB is very clean; below 5 dB is noisy.",
                "von_neumann": "Mean squared step-to-step change / variance. Near 0 = smooth curve, near 2 = random.",
                "spikes": f"A reading more than {SPIKE_K} robust sigmas AND more than {int(SPIKE_MIN_FRAC * 100)}% "
                          "of the feature's spread away from the median of its 4 neighbours, while the neighbours "
                          "agree with each other (one-point blip, not a real step change).",
                "stuck_sensor": f"At least {STUCK_RUN} identical consecutive readings on a continuous feature.",
                "out_of_range": "Values outside the physically possible range of the quantity. They are excluded "
                                "from the noise tests (and from outliers, correlation, KDE and monthly analysis) "
                                "and reported as data errors.",
            },
            "thresholds": {"negligible": f"< {NSR_LEVELS[0]}", "low": f"< {NSR_LEVELS[1]}",
                           "moderate": f"< {NSR_LEVELS[2]}", "high": f">= {NSR_LEVELS[2]}",
                           "noise_flag_from": f">= {NSR_LEVELS[1]} (moderate) or > {SPIKE_PCT_FLAG}% spikes"},
            "features": rows, "skipped": skipped,
        }
        noisy = [r for r in rows if r["has_noise"]]
        errs = [r for r in rows if r["has_errors"]]
        out["dataset_summary"] = {
            "features_checked": len(rows),
            "features_with_noise": len(noisy), "noisy_features": [r["feature"] for r in noisy],
            "has_noise": bool(noisy),
            "features_with_errors": len(errs), "error_features": [r["feature"] for r in errs],
            "has_errors": bool(errs),
            "mean_noise_to_signal": float(np.mean([r["noise_to_signal"] for r in rows])) if rows else None,
        }
        if not rows:
            out["insights"] = ["Not enough data to test for noise."]
            return out
        ins = []
        if noisy:
            ins.append(f"Random noise: YES in {len(noisy)} of {len(rows)} features: {[r['feature'] for r in noisy]}.")
        else:
            ins.append(f"Random noise: NO significant noise - all {len(rows)} tested features are smooth.")
        if errs:
            ins.append(f"Data errors: YES in {len(errs)} feature(s): {[r['feature'] for r in errs]}.")
        else:
            ins.append("Data errors: none (no impossible or stuck readings).")
        by_nsr = sorted(rows, key=lambda r: -r["noise_to_signal"])
        ins.append(f"Noisiest: '{by_nsr[0]['feature']}' (noise/signal {by_nsr[0]['noise_to_signal']:.2f}); "
                   f"cleanest: '{by_nsr[-1]['feature']}' ({by_nsr[-1]['noise_to_signal']:.2f}).")
        for r in rows:
            if r["has_noise"] or r["has_errors"]:
                ins.append(f"'{r['feature']}': " + "; ".join(r["noise_reasons"] + r["error_reasons"]) + ".")
        good = [r["feature"] for r in rows if not (r["has_noise"] or r["has_errors"])]
        if good:
            ins.append(f"Clean features (no action needed): {good}.")
        for sk in skipped:
            ins.append(f"'{sk['feature']}' not tested: {sk['reason']}.")
        if errs:
            ins.append("Recommended cleaning: set impossible/stuck readings to NaN and interpolate over time. "
                       "The outlier, correlation, KDE and monthly sections already ignore impossible values.")
        if any(r["feature"].lower().startswith("wind speed") and r["has_noise"] for r in rows):
            ins.append("Note: wind speed fluctuates naturally (gusts), so its noise flag is expected - "
                       "smooth it (rolling median / daily mean) rather than deleting points.")

        # --- plots
        fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(rows))))
        colors = {"negligible": "#2B5E4B", "low": "#8DAA91", "moderate": "#E0A458", "high": "#BD4A32"}
        srt = sorted(rows, key=lambda r: r["noise_to_signal"])
        ax.barh([r["feature"] for r in srt], [r["noise_to_signal"] for r in srt],
                color=[colors[r["verdict"]] for r in srt])
        for lv in NSR_LEVELS:
            ax.axvline(lv, color="#2B2621", ls=":", lw=1)
        ax.set_xlim(0, max(0.5, min(1, by_nsr[0]["noise_to_signal"] * 1.15)))
        ax.set_xlabel("noise-to-signal ratio (0 = smooth, 1 = pure noise)")
        ax.set_title("Noise level per feature  (dotted lines = verdict thresholds)")
        self._save_plot(fig, "noise_levels", "Noise level per feature", "noise")

        show = list(dict.fromkeys(self.key_feats + [by_nsr[0]["feature"]] + [r["feature"] for r in errs]))
        for c in show:
            if c in cache:
                s, smooth, mask, resid, rs = cache[c]
                verdict = next(r["verdict"] for r in rows if r["feature"] == c)
                self._plot_noise_view(c, s, smooth, mask, resid, rs, verdict)
        out["insights"] = ins
        return out

    # ---------------------------------------------------------- 4b. monthly trends (your resample code)
    def monthly_trends(self):
        out = {"description": "Monthly aggregation of every feature (mean, 10th/90th percentile, std), "
                              "seasonal averages and month-by-month trend lines across the years.",
               "insights": []}
        if not self.date_col:
            out["insights"].append("No date column - monthly analysis skipped.")
            return out
        df, ins = self.df_valid, out["insights"]
        monthly = {}
        for c in self.num_cols:
            m = df[c].resample("MS").agg(
                mean="mean",
                low=lambda v: v.quantile(0.10),
                high=lambda v: v.quantile(0.90),
                std="std",
            )
            m.index.name = "month"
            monthly[c] = records(m.reset_index())
        out["monthly_stats"] = monthly

        season = {}
        for c in self.key_feats:
            by_m = df[c].groupby(df.index.month).mean()
            season[c] = {calendar.month_name[int(k)]: float(v) for k, v in by_m.items()}
            ins.append(f"'{c}' averages highest in {calendar.month_name[int(by_m.idxmax())]} "
                       f"({by_m.max():.2f}) and lowest in {calendar.month_name[int(by_m.idxmin())]} "
                       f"({by_m.min():.2f}).")
        out["seasonality"] = season

        df_monthly_mean = df[self.num_cols].resample("MS").mean()
        for c in self.key_feats:
            fig, ax = plt.subplots(figsize=(14, 8))
            for mth in range(1, 13):
                label, color = MONTH_STYLE[mth]
                md = df_monthly_mean[c][df_monthly_mean.index.month == mth].dropna()
                if len(md):
                    sns.lineplot(x=md.index.year, y=md.values, label=label, color=color, marker="o", ax=ax)
            ax.set_title(f"{c} - each month across the years")
            ax.set_xlabel("YEAR")
            ax.set_ylabel(c)
            self._save_plot(fig, f"month_lines_{slug(c)}", f"Month-by-month trend - {c}", "monthly_trends")
            ms = monthly[c]
            worst = max((r for r in ms if r["std"] is not None), key=lambda r: r["std"])
            ins.append(f"Most variable month for '{c}': {worst['month'][:7]} (std {worst['std']:.2f}).")
        return out

    # ---------------------------------------------------------- 5. correlation
    def correlation(self):
        cols = self.num_cols
        out = {"description": "Linear (Pearson) and rank (Spearman) relationships between numeric features, "
                              "plus how the main variables behave for each weather 'Summary'.",
               "insights": []}
        if len(cols) < 2:
            out["insights"].append("Need at least two non-constant numeric columns.")
            return out
        pear = self.df_valid[cols].corr()
        spear = self.df_valid[cols].corr(method="spearman")
        pairs = []
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                r, rs = pear.loc[a, b], spear.loc[a, b]
                pairs.append({"a": a, "b": b, "pearson": r, "spearman": rs,
                              "strength": ("very strong" if abs(r) >= .8 else "strong" if abs(r) >= .6 else
                                           "moderate" if abs(r) >= .4 else "weak" if abs(r) >= .2 else "negligible"),
                              "direction": "positive" if r > 0 else "negative"})
        pairs.sort(key=lambda p: -abs(p["pearson"]))
        out["pearson"] = pear.round(4).to_dict()
        out["spearman"] = spear.round(4).to_dict()
        out["pairs"] = pairs

        ins = out["insights"]
        for p in [p for p in pairs if abs(p["pearson"]) >= .6][:6]:
            ins.append(f"{p['strength'].capitalize()} {p['direction']} link between '{p['a']}' and '{p['b']}' "
                       f"(r = {p['pearson']:.2f}).")
        redundant = [p for p in pairs if abs(p["pearson"]) >= .9]
        for p in redundant:
            ins.append(f"'{p['a']}' and '{p['b']}' are nearly redundant (|r| >= 0.9) - keep one to avoid "
                       "multicollinearity in a model.")
        nonlin = [p for p in pairs if abs(p["spearman"] - p["pearson"]) > .15]
        for p in nonlin[:3]:
            ins.append(f"'{p['a']}' vs '{p['b']}': Pearson {p['pearson']:.2f} but Spearman {p['spearman']:.2f} "
                       "- the relationship is monotonic but not linear (or distorted by outliers/zeros).")
        weak = [c for c in cols if pear[c].drop(c).abs().max() < .2]
        if weak:
            ins.append(f"Almost independent of every other feature: {weak}.")

        fig, ax = plt.subplots(figsize=(max(7, len(cols) * 1.0), max(5, len(cols) * 0.85)))
        sns.heatmap(pear, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1,
                    linewidths=.5, ax=ax)
        ax.set_title("Pearson correlation matrix")
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
        self._save_plot(fig, "correlation_heatmap", "Pearson correlation heatmap", "correlation")

        # scatter of the 2 strongest pairs
        for p in pairs[:2]:
            d = self.df_valid[[p["a"], p["b"]]].dropna()
            d = d.sample(min(len(d), 5000), random_state=0)
            fig, ax = plt.subplots(figsize=(6.5, 5))
            sns.regplot(data=d, x=p["a"], y=p["b"], ax=ax, scatter_kws={"s": 5, "alpha": .25},
                        line_kws={"color": "#BD4A32"})
            ax.set_title(f"{p['a']} vs {p['b']}  (r = {p['pearson']:.2f})")
            self._save_plot(fig, f"scatter_{slug(p['a'])}__{slug(p['b'])}",
                            f"Scatter {p['a']} vs {p['b']}", "correlation")

        # --- weather Summary analysis (your FacetGrid code, for Apparent Temp + Humidity)
        cat = self.category_col
        if cat in self.df.columns:
            facet_feats = [c for c in ("Apparent Temperature (C)", "Humidity") if c in cols] or self.key_feats
            d_all = self.df_valid[[cat] + facet_feats].dropna()
            top = d_all[cat].value_counts().head(self.top_categories).index
            d_all = d_all.assign(**{cat: d_all[cat].where(d_all[cat].isin(top), "Other")})
            for f in facet_feats:
                with sns.axes_style("darkgrid"):
                    g = sns.FacetGrid(d_all, hue=cat, height=7, aspect=1.5)
                    g.map(sns.histplot, f).add_legend()
                    g.figure.suptitle(f"Analysis of Weather Conditions with {f}", y=1.02)
                self._save_plot(g.figure, f"facet_{slug(f)}_by_{slug(cat)}",
                                f"{f} distribution by {cat}", "correlation")
            grp = d_all.groupby(cat)[facet_feats].mean().round(3)
            out["by_category"] = {"category": cat,
                                  "note": f"top {self.top_categories} categories kept, the rest grouped as 'Other'",
                                  "means": records(grp.reset_index())}
            for f in facet_feats:
                hi, lo = grp[f].idxmax(), grp[f].idxmin()
                ins.append(f"Average '{f}' is highest for '{hi}' ({grp[f].max():.2f}) and lowest for "
                           f"'{lo}' ({grp[f].min():.2f}).")
        return out

    # ---------------------------------------------------------- 6. KDE
    def kde(self):
        feats, skipped = [], []
        for c in self.num_cols + self.constant_cols:
            s = self.df_valid[c].dropna()
            if len(s) < 2 or s.nunique() < 2:
                skipped.append({"feature": c, "reason": "constant or too few values"})
                continue
            fig, ax = plt.subplots(figsize=(6, 3.8))
            sns.kdeplot(s, fill=True, color="#8C572A", ax=ax)
            ax.axvline(s.mean(), color="#2B5E4B", ls="--", label=f"mean = {s.mean():.2f}")
            ax.axvline(s.median(), color="#BD4A32", ls=":", label=f"median = {s.median():.2f}")
            ax.set_title(f"KDE - {c}")
            ax.legend()
            self._save_plot(fig, f"kde_{slug(c)}", f"KDE of {c}", "kde")
            sk = float(s.skew())
            shape = "right-skewed" if sk > .5 else "left-skewed" if sk < -.5 else "roughly symmetric"
            dens = stats.gaussian_kde(s)(np.linspace(s.min(), s.max(), 200))
            modes = len(find_peaks(dens, prominence=0.10 * dens.max())[0])   # ignore tiny bumps
            feats.append({"feature": c, "mean": s.mean(), "median": s.median(), "std": s.std(),
                          "skew": sk, "kurtosis": float(s.kurt()), "n_peaks": modes, "shape": shape})
        insights = [f"'{f['feature']}' is {f['shape']} with {f['n_peaks']} peak(s)"
                    + (" - multimodal, probably mixed regimes (day/night or seasons)." if f["n_peaks"] > 1 else ".")
                    for f in feats]
        return {"description": "Kernel density estimate of every non-constant numeric feature, with mean and "
                               "median markers, to show distribution shape.",
                "features": feats, "skipped": skipped, "insights": insights}

    # ---------------------------------------------------------- 7. model phase (target balance, preprocessing, RF + evaluation)
    def model_phase(self):
        """Class balance of the target, sklearn preprocessing (Pipeline + ColumnTransformer) and a
        RandomForestClassifier evaluated on a TIME-BASED hold-out (last 25% of the series), with both
        impurity and permutation importances. Works on the cleaned data (impossible readings -> NaN)."""
        target_col = self.target_col
        desc = ("Model phase: target class balance, preprocessing with sklearn Pipeline + ColumnTransformer, "
                "Random Forest evaluated on a time-based hold-out, and feature importance (impurity and permutation).")
        if target_col not in self.df.columns:
            return {"description": desc, "insights": [f"Target column '{target_col}' not found - model phase skipped."]}
        insights = []

        # (1) Class balance (all rows, including missing)
        s_target = self.df[target_col]
        total_samples = len(s_target)
        nan_count = int(s_target.isna().sum())
        class_counts, class_pcts = {}, {}
        for k, v in s_target.value_counts(dropna=False).items():
            name = "Missing / Not recorded" if pd.isna(k) else str(k)
            class_counts[name] = int(v)
            class_pcts[name] = round(float(v) / total_samples * 100, 2)
        non_null_counts = s_target.dropna().value_counts()
        imbalance_ratio, is_balanced = None, False
        if len(non_null_counts) >= 2:
            imbalance_ratio = round(float(non_null_counts.iloc[0]) / max(float(non_null_counts.iloc[1]), 1), 2)
            is_balanced = bool(imbalance_ratio <= 1.5)
        severity = ("not measurable" if imbalance_ratio is None else "severely imbalanced" if imbalance_ratio >= 5
                    else "imbalanced" if imbalance_ratio > 1.5 else "balanced")
        listing = ", ".join(f"'{k}': {v:,} ({class_pcts[k]:.1f}%)" for k, v in class_counts.items())
        insights.append(f"Class balance on '{target_col}': {severity}. {listing}.")
        if imbalance_ratio and not is_balanced:
            insights.append(f"Imbalance ratio {imbalance_ratio}:1 - without class weighting a model would be biased "
                            "towards the majority class, so class_weight='balanced' is used and balanced accuracy / F1 "
                            "are reported instead of accuracy alone.")

        # (2) Preprocessing, on CLEANED data (physically impossible readings are NaN -> median-imputed)
        d = self.df_valid[self.df_valid[target_col].notna()]
        num_features = list(self.num_cols)
        cat_features = [self.category_col] if self.category_col in d.columns else []
        X, y = d[num_features + cat_features], d[target_col].astype(str)
        num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        cat_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
        preprocessor = ColumnTransformer([("num", num_pipeline, num_features), ("cat", cat_pipeline, cat_features)],
                                         remainder="drop")
        insights.append(
            f"Preprocessing: {len(num_features)} numerical features {num_features} and {len(cat_features)} categorical "
            f"feature(s) {cat_features}; ColumnTransformer = numerical Pipeline (median imputation + StandardScaler) "
            "+ categorical Pipeline (most-frequent imputation + OneHotEncoder). Impossible readings "
            f"({sum(self.invalid_counts.values())} cells) are treated as missing, not as real values.")

        # (3) Time-based split: train on the past, test on the most recent 25% (hourly data is autocorrelated,
        #     a random split would leak neighbouring hours between train and test)
        n_test = max(1, int(len(d) * 0.25))
        n_train = len(d) - n_test
        Xtr, Xte, ytr, yte = X.iloc[:n_train], X.iloc[n_train:], y.iloc[:n_train], y.iloc[n_train:]

        def span(idx):
            return f"{idx.min():%Y-%m-%d} to {idx.max():%Y-%m-%d}" if self.date_col else f"rows {idx.min()}-{idx.max()}"

        rf = RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1,
                                    class_weight="balanced")
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", rf)]).fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        classes = sorted(y.unique())
        acc = float(accuracy_score(yte, pred))
        bal = float(balanced_accuracy_score(yte, pred))
        f1 = float(f1_score(yte, pred, average="macro", zero_division=0))
        baseline = float((yte == ytr.mode().iloc[0]).mean())
        recalls = recall_score(yte, pred, labels=classes, average=None, zero_division=0)
        evaluation = {
            "split": "time-based hold-out (train = first 75%, test = last 25%)",
            "n_train": int(n_train), "n_test": int(n_test),
            "train_period": span(Xtr.index), "test_period": span(Xte.index),
            "accuracy": acc, "balanced_accuracy": bal, "f1_macro": f1, "baseline_accuracy": baseline,
            "per_class_recall": {c: float(r) for c, r in zip(classes, recalls)},
        }
        insights.append(f"Hold-out evaluation ({evaluation['test_period']}): accuracy {acc:.3f} (always predicting the "
                        f"majority class gives {baseline:.3f}), balanced accuracy {bal:.3f}, macro F1 {f1:.3f}.")
        if bal >= 0.995:
            insights.append("Near-perfect score: check that no feature gives away the target (data leakage).")

        # (4) Importances: impurity (train fit) and permutation (on the unseen test period)
        names = list(pipe.named_steps["preprocessor"].get_feature_names_out())
        raw_imp = pipe.named_steps["classifier"].feature_importances_
        feats = num_features + cat_features
        orig = {f: 0.0 for f in feats}
        for nm, imp in zip(names, raw_imp):
            for f in feats:
                if nm.startswith(f"num__{f}") or nm.startswith(f"cat__{f}"):
                    orig[f] += float(imp)
                    break
        impurity = [{"rank": i + 1, "feature": f, "importance": round(v, 4), "percentage": round(v * 100, 2)}
                    for i, (f, v) in enumerate(sorted(orig.items(), key=lambda kv: -kv[1]))]
        transformed = [{"rank": i + 1, "feature": re.sub(r"^(num__|cat__)", "", nm), "importance": round(float(v), 4),
                        "percentage": round(float(v) * 100, 2)}
                       for i, (nm, v) in enumerate(sorted(zip(names, raw_imp), key=lambda x: -x[1]))]
        pi = permutation_importance(pipe, Xte, yte, scoring="balanced_accuracy", n_repeats=3,
                                    random_state=42, n_jobs=-1)
        order = np.argsort(-pi.importances_mean)
        permutation = [{"rank": r + 1, "feature": Xte.columns[j], "drop_points": round(float(pi.importances_mean[j]) * 100, 2),
                        "std_points": round(float(pi.importances_std[j]) * 100, 2)} for r, j in enumerate(order)]
        insights.append(f"Permutation importance (drop in balanced accuracy when a feature is shuffled on the test period): "
                        f"'{permutation[0]['feature']}' {permutation[0]['drop_points']:.1f} pts"
                        + (f", then '{permutation[1]['feature']}' {permutation[1]['drop_points']:.1f} pts." if len(permutation) > 1 else "."))
        top = impurity[:2]
        insights.append(f"Impurity importance (Random Forest splits): '{top[0]['feature']}' {top[0]['percentage']:.1f}%"
                        + (f", '{top[1]['feature']}' {top[1]['percentage']:.1f}%." if len(top) > 1 else "."))
        if len(top) > 1 and all(t["feature"] in self.num_cols for t in top):
            r = float(self.df_valid[top[0]["feature"]].corr(self.df_valid[top[1]["feature"]]))
            if abs(r) >= 0.9:
                insights.append(f"'{top[0]['feature']}' and '{top[1]['feature']}' are almost the same signal (r = {r:.2f}): "
                                "impurity importance splits the credit between them, so read them together.")

        # leakage hint: category values that name the target (e.g. 'Rain', 'Snow') and are almost pure
        for cf in cat_features:
            hit = d[cf].astype(str).str.contains(r"rain|snow|drizzle|sleet|flurr", case=False, regex=True)
            if hit.any():
                purity = d[hit].groupby(cf)[target_col].agg(lambda s: s.value_counts(normalize=True).iloc[0])
                share = float(d[cf].isin(purity[purity >= 0.95].index).mean() * 100)
                if share >= 1:
                    insights.append(f"Possible leakage: {share:.1f}% of rows have a '{cf}' value that names precipitation "
                                    f"and almost determines '{target_col}' (>= 95% one class). Compare with a model without '{cf}'.")

        return {
            "description": desc,
            "target_balance": {
                "target_feature": target_col, "is_balanced": is_balanced, "imbalance_ratio": imbalance_ratio,
                "severity": severity, "total_observations": total_samples, "valid_observations": int(len(d)),
                "missing_count": nan_count, "class_counts": class_counts, "class_percentages": class_pcts,
            },
            "preprocessing": {
                "numerical_features": num_features, "categorical_features": cat_features, "target_feature": target_col,
                "numerical_pipeline": ["SimpleImputer(strategy='median')", "StandardScaler()"],
                "categorical_pipeline": ["SimpleImputer(strategy='most_frequent')", "OneHotEncoder(handle_unknown='ignore')"],
                "transformer_method": "ColumnTransformer",
                "total_input_features": len(feats), "total_transformed_features": len(names),
                "samples_trained": int(n_train),
            },
            "evaluation": evaluation,
            "feature_importance": {
                "model": "RandomForestClassifier", "n_estimators": 100, "max_depth": 16,
                "class_weight": "balanced", "random_state": 42,
                "ranked_features": impurity, "permutation_ranking": permutation,
                "detailed_transformed_ranking": transformed[:15],
            },
            "insights": insights,
        }

    # ---------------------------------------------------------- run / save
    def run(self):
        self.load()
        stamp = datetime.now()
        self.report = {"meta": {
            "report_id": f"{slug(self.csv_path.stem)}_{stamp:%Y%m%d_%H%M%S}",
            "dataset": self.csv_path.name, "generated_at": stamp.isoformat(timespec="seconds"),
            "n_rows": len(self.raw), "n_cols": self.raw.shape[1], "date_column": self.date_col,
            "numeric_columns": self.num_cols, "constant_columns": self.constant_cols,
            "categorical_columns": self.cat_cols,
            "impossible_values_excluded": self.invalid_counts}}
        self._section("description", self.describe)
        self._section("missing_duplicates", self.missing_duplicates)
        self._section("outliers", self.outliers)
        self._section("noise", self.noise)
        self._section("monthly_trends", self.monthly_trends)
        self._section("correlation", self.correlation)
        self._section("kde", self.kde)
        self._section("model_phase", self.model_phase)
        self.report["plots"] = self.plots
        return clean(self.report)

    def to_db_documents(self):
        """Two collections/tables: 'reports' (light) and 'plots' (one doc per image).
        Splitting avoids MongoDB's 16 MB document limit when images are embedded."""
        rep = clean(self.report)
        rid = rep["meta"]["report_id"]
        core = {k: v for k, v in rep.items() if k != "plots"}
        core["_id"] = rid
        return {"reports": [core],
                "plots": [{"_id": f"{rid}__{p['name']}", "report_id": rid, **p} for p in rep["plots"]]}

    def save(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.out_dir / "report.json", "w", encoding="utf-8") as f:
            json.dump(clean(self.report), f, ensure_ascii=False, indent=2)
        with open(self.out_dir / "db_documents.json", "w", encoding="utf-8") as f:
            json.dump(self.to_db_documents(), f, ensure_ascii=False)
        return str(self.out_dir / "report.json")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        sys.exit("Usage: python weather_model.py weather.csv [output_dir]")
    m = WeatherEDAModel(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "eda_output")
    m.run()
    print("Saved:", m.save(), "|", len(m.plots), "plots")