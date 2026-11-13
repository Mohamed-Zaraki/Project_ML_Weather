"""quality_multi.py - Shared data-quality logic for the multi-dataset EDA project.

This module is the *logic* layer for COVID-19, Parkinson, Alcohol and Heart-Attack
datasets.  It contains no UI / Streamlit code.

Main building blocks
--------------------
1. ``DATASET_RANGES``      physical / biological valid ranges per dataset.
2. ``mask_impossible``     impossible readings -> NaN so they cannot distort stats.
3. ``detect_contradictions`` explicit *contradictory* records (e.g. blood pressure
   = 0, cholesterol = 0, class says "no alcohol" while alcohol > 0, p-tau > t-tau).
4. ``detect_ml_outliers``  a *consensus of six unsupervised detectors*:
   Isolation Forest, One-Class SVM, Local Outlier Factor, Elliptic Envelope,
   DBSCAN and robust Mahalanobis distance.
5. ``compute_noise_metrics`` noise-to-signal statistic adapted to *tabular*
   (unordered) data: local k-NN residual noise / robust signal spread, plus
   SNR(dB), Hampel spikes, stuck-reading runs and out-of-range counts.

Why the noise method differs from the weather project
------------------------------------------------------
Weather rows are an ordered time series, so the weather model estimates noise with
the second difference ``x[t+1] - 2x[t] + x[t-1]`` (which cancels the smooth signal).
Medical / sensor tables here are *cross-sectional*: row order is arbitrary, so a
second difference would measure row order, not noise.  The same noise-to-signal
idea is therefore estimated from a local smoother in feature space: each reading is
compared with the median of its k nearest neighbours, and the robust spread of
those residuals is divided by the robust spread of the signal.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope, MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# --------------------------------------------------------------------------- #
# Thresholds
# --------------------------------------------------------------------------- #
SPIKE_K = 3.5                      # Hampel spike threshold (in robust sigmas)
SPIKE_PCT_FLAG = 1.0               # feature "noisy" if > 1% points are spikes
SPIKE_MIN_FRAC = 0.4               # spike deviation relative to feature spread
STUCK_RUN = 10                     # >= 10 identical consecutive readings
NSR_LEVELS = (0.05, 0.15, 0.35)    # negligible / low / moderate / high NSR
KNN_NEIGHBOURS = 8                 # neighbours used by the tabular noise estimator
KNN_MAX_ROWS = 6000                # cap for computing the neighbour graph

# --------------------------------------------------------------------------- #
# Biological / physical valid ranges (min, max)
# --------------------------------------------------------------------------- #
DATASET_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "alcohol": {
        "alcohol": (0.0, 10.0),            # alcohol concentration (ml/L)
        "temp_amb": (10.0, 45.0),          # ambient temperature (deg C)
        "temp_fac_max": (15.0, 42.0),      # maximum facial temperature
        "temp_fac_min": (5.0, 35.0),       # minimum facial temperature
        "eyes": (1.0, 30.0),               # pupil aspect ratio
    },
    "heart": {
        "age": (18.0, 100.0),
        "sex": (0.0, 1.0),
        "chest pain type": (1.0, 4.0),
        "resting bp s": (70.0, 250.0),     # 0 is physiologically impossible
        "cholesterol": (50.0, 650.0),      # 0 is missing data, not a real value
        "fasting blood sugar": (0.0, 1.0),
        "resting ecg": (0.0, 2.0),
        "max heart rate": (60.0, 220.0),
        "exercise angina": (0.0, 1.0),
        "oldpeak": (-0.01, 7.0),           # ST depression cannot be negative
        "st slope": (1.0, 3.0),            # 0 is an invalid code
        "target": (0.0, 1.0),
    },
    "heartattack": {
        "age": (18.0, 100.0),
        "sex": (0.0, 1.0),
        "cp": (0.0, 3.0),
        "trtbps": (70.0, 250.0),           # 0 impossible
        "chol": (50.0, 650.0),             # 0 impossible / missing
        "fbs": (0.0, 1.0),
        "restecg": (0.0, 2.0),
        "thalachh": (60.0, 220.0),         # 0 impossible
        "exng": (0.0, 1.0),
        "oldpeak": (0.0, 7.0),
        "slp": (0.0, 2.0),
        "caa": (0.0, 4.0),
        "thall": (1.0, 3.0),               # 0 = missing code
        "output": (0.0, 1.0),
    },
    "parkinson": {
        "a-beta": (50.0, 1500.0),          # CSF A-beta42 pg/mL
        "alpha-syn": (100.0, 10000.0),     # CSF alpha-synuclein pg/mL
        "p-tau": (1.0, 200.0),             # CSF p-tau pg/mL
        "t-tau": (10.0, 500.0),            # CSF total tau pg/mL
        "p-tau/a-beta": (0.0001, 5.0),
        "t-tau/a-beta": (0.0001, 5.0),
        "p-tau/t-tau": (0.0001, 5.0),
        "upsit": (0.0, 40.0),              # smell identification score
        "rembdsq": (0.0, 13.0),            # REM sleep behaviour score
        "caudate_r": (0.1, 6.0),           # DaTscan binding ratio
        "caudate_l": (0.1, 6.0),
        "putamen_r": (0.1, 6.0),
        "putamen_l": (0.1, 6.0),
        "group (normal/pd/swedd - 1/2/3)": (0.0, 2.0),
    },
    "covid": {
        "sex": (0.0, 1.0),
        "age": (1.0, 115.0),               # 0 is usually unrecorded
        "ca": (1.5, 3.5),                  # serum calcium mmol/L
        "ck": (10.0, 8000.0),              # creatine kinase U/L
        "crea": (0.2, 16.0),               # creatinine mg/dL
        "alp": (10.0, 1000.0),             # alkaline phosphatase U/L
        "ggt": (5.0, 3500.0),              # gamma-GT U/L
        "glu": (20.0, 900.0),              # glucose mg/dL
        "ast": (5.0, 1500.0),
        "alt": (5.0, 1000.0),
        "ldh": (50.0, 6000.0),
        "pcr": (0.1, 500.0),               # C-reactive protein mg/L
        "kal": (1.5, 8.0),                 # potassium mmol/L
        "nat": (110.0, 175.0),             # sodium mmol/L
        "urea": (5.0, 500.0),
        "wbc": (0.5, 80.0),
        "rbc": (1.0, 8.0),
        "hgb": (4.0, 22.0),
        "hct": (12.0, 65.0),
        "mcv": (50.0, 140.0),
        "mch": (15.0, 50.0),
        "mchc": (25.0, 45.0),
        "plt1": (10.0, 900.0),
        "ne": (0.0, 100.0),
        "ly": (0.0, 100.0),
        "mo": (0.0, 100.0),
        "eo": (0.0, 100.0),
        "ba": (0.0, 100.0),
        "net": (0.0, 100.0),
        "lyt": (0.0, 100.0),
        "mot": (0.0, 100.0),
        "eot": (0.0, 100.0),
        "bat": (0.0, 100.0),
        "sarscov": (0.0, 1.0),
    },
}

# Labels used in reports / plots
DATASET_LABELS = {
    "alcohol": "Alcohol Detection",
    "heart": "Heart Attack (comprehensive)",
    "heartattack": "Heart Attack (UCI)",
    "parkinson": "Parkinson's Disease",
    "covid": "COVID-19 Laboratory",
}


# --------------------------------------------------------------------------- #
# Basic helpers
# --------------------------------------------------------------------------- #
def robust_sigma(x) -> float:
    """MAD-based standard deviation: insensitive to outliers / corrupt values."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def get_physical_range(dataset_key: str, col_name: str) -> tuple[float, float] | None:
    """Look up biological / physical bounds for a dataset column."""
    name = str(col_name).strip().lower()
    ranges = DATASET_RANGES.get(str(dataset_key).lower(), {})
    if name in ranges:
        return ranges[name]
    return None


def mask_impossible(df: pd.DataFrame, dataset_key: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return a copy where physically impossible readings are replaced by NaN.

    The original frame is never modified.  Returns ``(clean_df, {col: count})``.
    """
    out = df.copy()
    counts: dict[str, int] = {}
    for c in out.select_dtypes("number").columns:
        s = out[c].dropna()
        if s.empty or s.nunique() <= 1:
            continue
        rng = get_physical_range(dataset_key, c)
        if not rng:
            continue
        bad = (out[c] < rng[0]) | (out[c] > rng[1])
        n_bad = int(bad.sum())
        if n_bad:
            out.loc[bad, c] = np.nan
            counts[c] = n_bad
    return out, counts


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Column as float Series, or an all-NaN Series when the column is absent."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _outside(s: pd.Series, lo: float, hi: float) -> pd.Series:
    return s.notna() & ((s < lo) | (s > hi))


def _not_in(s: pd.Series, allowed) -> pd.Series:
    allowed = list(allowed)
    return s.notna() & ~s.round(6).isin(allowed)


def _rows(df: pd.DataFrame, mask: pd.Series, cols: list[str], limit: int = 5) -> list[dict]:
    """A few example rows (raw values) for a contradiction rule."""
    cols = [c for c in dict.fromkeys(cols) if c in df.columns][:8]
    if not cols:
        return []
    sub = df.loc[mask.reindex(df.index, fill_value=False), cols].head(limit)
    if sub.empty:
        return []
    out = []
    for idx, row in sub.iterrows():
        rec = {"row": int(idx) if isinstance(idx, (int, np.integer)) else str(idx)}
        for c in cols:
            v = row[c]
            if pd.isna(v):
                rec[c] = None
            elif isinstance(v, (np.integer, int)):
                rec[c] = int(v)
            elif isinstance(v, (np.floating, float)):
                rec[c] = round(float(v), 4)
            else:
                rec[c] = str(v)
        out.append(rec)
    return out


# --------------------------------------------------------------------------- #
# Contradiction detection (the "wind = 0 while pressure = 0" section)
# --------------------------------------------------------------------------- #
def _rules_for(df: pd.DataFrame, dataset_key: str) -> list[dict]:
    """Build the list of contradiction rules for one dataset.

    A rule is a dict with: id, label, description, columns, severity and a
    boolean mask aligned with ``df.index``.
    """
    key = str(dataset_key).lower()
    idx = df.index
    none = pd.Series(False, index=idx)
    rules: list[dict] = []

    def add(rule_id, label, description, cols, mask, severity="critical"):
        rules.append({
            "id": rule_id, "label": label, "description": description,
            "columns": list(dict.fromkeys(cols)), "severity": severity,
            "mask": pd.Series(mask, index=idx).fillna(False).astype(bool),
        })

    if key == "alcohol":
        alcohol, cls = _col(df, "ALCOHOL"), _col(df, "CLASS")
        t_amb, t_max, t_min = _col(df, "TEMP_AMB"), _col(df, "TEMP_FAC_MAX"), _col(df, "TEMP_FAC_MIN")
        eyes = _col(df, "EYES")

        add("alcohol_but_no_class",
            "Alcohol detected while class says 'no alcohol'",
            "The class label says no alcohol, yet a non-trivial alcohol concentration "
            "(>= 0.5 ml/L) was measured. These rows contradict the label.",
            ["ALCOHOL", "CLASS"], (cls == 0) & (alcohol >= 0.5))
        add("temp_min_gt_max",
            "Face temperature minimum > maximum",
            "The minimum facial temperature is larger than the maximum. This is "
            "logically impossible and usually a corrupted sensor value (e.g. 1.7e14).",
            ["TEMP_FAC_MIN", "TEMP_FAC_MAX"], t_min > t_max)
        add("face_min_above_ambient",
            "Face minimum temperature far above ambient",
            "The minimum face temperature is more than 5 degC above ambient; the "
            "minimum should be below or close to ambient.",
            ["TEMP_FAC_MIN", "TEMP_AMB"], t_min > (t_amb + 5))
        add("eyes_impossible",
            "Impossible pupil ratio",
            "Pupil aspect ratio outside 1-30. Values of 1e13-1e14 are corrupt "
            "sensor readings, not measurements.",
            ["EYES"], _outside(eyes, 1.0, 30.0))
        add("face_temp_impossible",
            "Impossible face temperature",
            "Face temperature outside the plausible 5-42 degC range.",
            ["TEMP_FAC_MIN", "TEMP_FAC_MAX"],
            _outside(t_min, 5.0, 35.0) | _outside(t_max, 15.0, 42.0))
        add("alcohol_out_of_range",
            "Alcohol concentration out of range",
            "Alcohol concentration outside 0-10 ml/L.",
            ["ALCOHOL"], _outside(alcohol, 0.0, 10.0))
        add("class_invalid",
            "Invalid class code",
            "The class must be 0 (no alcohol) or 1 (alcohol present).",
            ["CLASS"], _not_in(cls, [0, 1]))
        return rules

    if key in ("heart", "heartattack"):
        if key == "heart":
            age, sex = _col(df, "age"), _col(df, "sex")
            bp, chol = _col(df, "resting bp s"), _col(df, "cholesterol")
            hr, oldpeak = _col(df, "max heart rate"), _col(df, "oldpeak")
            cp, ecg = _col(df, "chest pain type"), _col(df, "resting ecg")
            fbs, angina = _col(df, "fasting blood sugar"), _col(df, "exercise angina")
            slope, target = _col(df, "ST slope"), _col(df, "target")
            add("bp_zero", "Blood pressure = 0",
                "A resting blood pressure of 0 mmHg is incompatible with life: the "
                "value is a missing/placeholder code, not a measurement.",
                ["resting bp s"], bp == 0)
            add("chol_zero", "Cholesterol = 0",
                "Total cholesterol of 0 mg/dL is physiologically impossible. In this "
                "dataset 0 is a placeholder for a missing lipid panel.",
                ["cholesterol"], chol == 0)
            add("oldpeak_negative", "Negative ST depression",
                "oldpeak (ST depression induced by exercise) cannot be negative.",
                ["oldpeak"], oldpeak < 0)
            add("st_slope_invalid", "Invalid ST slope code",
                "ST slope must be 1 (up), 2 (flat) or 3 (down); 0 is an invalid code.",
                ["ST slope"], _not_in(slope, [1, 2, 3]))
            add("heart_rate_impossible", "Impossible maximum heart rate",
                "Maximum heart rate outside 60-220 bpm (0 is a missing code).",
                ["max heart rate"], _outside(hr, 60, 220))
            add("blood_pressure_impossible", "Impossible resting blood pressure",
                "Resting blood pressure outside 70-250 mmHg (0 is reported separately).",
                ["resting bp s"], _outside(bp, 70, 250) & (bp != 0))
            add("cholesterol_impossible", "Impossible cholesterol",
                "Cholesterol outside 100-650 mg/dL (0 is reported separately).",
                ["cholesterol"], _outside(chol, 100, 650) & (chol != 0))
            add("age_impossible", "Impossible age",
                "Age outside 18-100 years.", ["age"], _outside(age, 18, 100))
            add("invalid_codes", "Invalid categorical codes",
                "Coded clinical variables outside their documented code lists "
                "(sex 0/1, chest pain 1-4, fasting sugar 0/1, resting ECG 0-2, "
                "exercise angina 0/1, target 0/1).",
                ["sex", "chest pain type", "fasting blood sugar", "resting ecg",
                 "exercise angina", "target"],
                _not_in(sex, [0, 1]) | _not_in(cp, [1, 2, 3, 4]) | _not_in(fbs, [0, 1])
                | _not_in(ecg, [0, 1, 2]) | _not_in(angina, [0, 1]) | _not_in(target, [0, 1]))
        else:
            age, sex = _col(df, "age"), _col(df, "sex")
            bp, chol = _col(df, "trtbps"), _col(df, "chol")
            hr, oldpeak = _col(df, "thalachh"), _col(df, "oldpeak")
            cp, ecg = _col(df, "cp"), _col(df, "restecg")
            slope, caa = _col(df, "slp"), _col(df, "caa")
            thall, target = _col(df, "thall"), _col(df, "output")
            add("bp_zero", "Blood pressure = 0",
                "A resting blood pressure of 0 mmHg is incompatible with life.",
                ["trtbps"], bp == 0)
            add("chol_zero", "Cholesterol = 0",
                "Total cholesterol of 0 mg/dL is physiologically impossible.",
                ["chol"], chol == 0)
            add("hr_zero", "Maximum heart rate = 0",
                "A maximum heart rate of 0 bpm is impossible (missing code).",
                ["thalachh"], hr == 0)
            add("oldpeak_negative", "Negative ST depression",
                "oldpeak cannot be negative.", ["oldpeak"], oldpeak < 0)
            add("thall_missing_code", "Invalid thallium code",
                "thall must be 1, 2 or 3; 0 is the missing-value code.",
                ["thall"], _not_in(thall, [1, 2, 3]))
            add("caa_impossible", "Impossible number of major vessels",
                "caa (coloured major vessels) must be between 0 and 4.",
                ["caa"], _outside(caa, 0, 4))
            add("invalid_codes", "Invalid categorical codes",
                "Coded variables outside their documented ranges (cp 0-3, restecg 0-2, "
                "slp 0-2, sex 0/1, output 0/1).",
                ["sex", "cp", "restecg", "slp", "output"],
                _not_in(sex, [0, 1]) | _not_in(cp, [0, 1, 2, 3]) | _not_in(ecg, [0, 1, 2])
                | _not_in(slope, [0, 1, 2]) | _not_in(target, [0, 1]))
            add("age_impossible", "Impossible age",
                "Age outside 18-100 years.", ["age"], _outside(age, 18, 100))
        return rules

    if key == "parkinson":
        ab, asyn = _col(df, "A-beta"), _col(df, "alpha-syn")
        ptau, ttau = _col(df, "p-tau"), _col(df, "T-tau")
        r1, r2, r3 = _col(df, "P-tau/A-beta"), _col(df, "T-tau/A-beta"), _col(df, "P-tau/t-tau")
        upsit, rem = _col(df, "UPSIT"), _col(df, "REMBDSQ")
        dat = [_col(df, c) for c in ("CAUDATE_R", "CAUDATE_L", "PUTAMEN_R", "PUTAMEN_L")]
        group = _col(df, "Group (Normal/PD/SWEDD - 1/2/3)")

        add("biomarker_nonpositive", "Non-positive CSF biomarker",
            "A-beta, alpha-synuclein, p-tau and t-tau are concentrations and must be "
            "strictly positive.", ["A-beta", "alpha-syn", "p-tau", "T-tau"],
            (ab <= 0) | (asyn <= 0) | (ptau <= 0) | (ttau <= 0))
        add("datscan_nonpositive", "Non-positive DaTscan binding",
            "Caudate / putamen binding ratios must be positive.",
            ["CAUDATE_R", "CAUDATE_L", "PUTAMEN_R", "PUTAMEN_L"],
            (dat[0] <= 0) | (dat[1] <= 0) | (dat[2] <= 0) | (dat[3] <= 0))
        add("ptau_gt_ttau", "p-tau greater than total tau",
            "Phosphorylated tau is a fraction of total tau, so p-tau > t-tau is "
            "biologically contradictory.",
            ["p-tau", "T-tau"], ptau > ttau)
        add("ratio_gt_one", "p-tau / t-tau ratio greater than 1",
            "The p-tau / t-tau ratio cannot exceed 1 because p-tau is part of t-tau.",
            ["P-tau/t-tau", "p-tau", "T-tau"], r3 > 1)
        ratio_err = (
            ((r1 - ptau / ab).abs() / r1.abs().clip(lower=1e-9) > 0.01)
            | ((r2 - ttau / ab).abs() / r2.abs().clip(lower=1e-9) > 0.01)
            | ((r3 - ptau / ttau).abs() / r3.abs().clip(lower=1e-9) > 0.01)
        )
        add("ratio_mismatch", "Ratio does not match its components",
            "The published ratio columns do not equal the ratio of their own "
            "components (tolerance 1%), which indicates a transcription error.",
            ["P-tau/A-beta", "T-tau/A-beta", "P-tau/t-tau"], ratio_err, "warning")
        add("upsit_impossible", "UPSIT outside 0-40",
            "The smell identification test is scored 0-40.",
            ["UPSIT"], _outside(upsit, 0, 40))
        add("rembdsq_impossible", "REMBDSQ outside 0-13",
            "The REM sleep behaviour questionnaire is scored 0-13.",
            ["REMBDSQ"], _outside(rem, 0, 13))
        add("group_invalid", "Invalid diagnostic group",
            "Group codes are 0 (normal), 1 (PD) and 2 (SWEDD) in the documentation, "
            "so any other code is invalid.",
            ["Group (Normal/PD/SWEDD - 1/2/3)"], _not_in(group, [0, 1, 2]))
        if "Subject_id" in df.columns:
            add("duplicate_subject", "Duplicate subject identifier",
                "The same Subject_id appears more than once, so rows are not "
                "independent patients.", ["Subject_id"],
                df["Subject_id"].duplicated(keep=False), "warning")
        return rules

    if key == "covid":
        num = df.select_dtypes("number")
        negative = pd.Series(False, index=idx)
        for c in num.columns:
            if c == "SARSCov":
                continue
            negative |= (num[c] < 0)
        add("negative_measurement", "Negative laboratory value",
            "Concentration / count measurements cannot be negative.",
            list(num.columns)[:8], negative)

        age, ca = _col(df, "Age"), _col(df, "CA")
        add("age_zero", "Age = 0",
            "Age 0 is not a plausible adult COVID-19 record; it is an unrecorded age.",
            ["Age"], age == 0)
        add("calcium_impossible", "Impossible serum calcium",
            "Serum calcium outside 1.5-3.5 mmol/L. Values such as 20.25 are lab errors.",
            ["CA"], _outside(ca, 1.5, 3.5))
        zero_labs = ["CK", "CREA", "ALP", "GGT", "GLU", "AST", "ALT", "LDH", "PCR",
                     "KAL", "NAT", "UREA", "WBC", "RBC", "HGB", "HCT", "MCV", "MCH",
                     "MCHC", "PLT1"]
        z = pd.Series(False, index=idx)
        for c in zero_labs:
            z |= (_col(df, c) == 0)
        add("zero_lab", "Laboratory value = 0",
            "A zero in a concentration or blood count is used here as a missing code, "
            "not as a real measurement.", zero_labs, z)
        diff = df[["NE", "LY", "MO", "EO", "BA"]].sum(axis=1, min_count=5)
        add("differential_sum", "Differential white-cell percentages do not sum to 100",
            "NE + LY + MO + EO + BA are percentages of white cells and must sum to "
            "about 100% (tolerance 2 points).",
            ["NE", "LY", "MO", "EO", "BA"], (diff - 100).abs() > 2)
        mchc_calc = _col(df, "HGB") / _col(df, "HCT") * 100
        add("mchc_mismatch", "MCHC inconsistent with haemoglobin / haematocrit",
            "MCHC is derived as HGB / HCT * 100. A mismatch > 15% indicates a "
            "transcription or unit error.",
            ["MCHC", "HGB", "HCT"],
            (_col(df, "MCHC") - mchc_calc).abs() / _col(df, "MCHC").abs().clip(lower=1e-9) > 0.15,
            "warning")
        mcv_calc = _col(df, "HCT") / _col(df, "RBC") * 10
        add("mcv_mismatch", "MCV inconsistent with haematocrit / RBC",
            "MCV is derived as HCT / RBC * 10 (mismatch > 15%).",
            ["MCV", "HCT", "RBC"],
            (_col(df, "MCV") - mcv_calc).abs() / _col(df, "MCV").abs().clip(lower=1e-9) > 0.15,
            "warning")
        mch_calc = _col(df, "HGB") / _col(df, "RBC") * 10
        add("mch_mismatch", "MCH inconsistent with haemoglobin / RBC",
            "MCH is derived as HGB / RBC * 10 (mismatch > 15%).",
            ["MCH", "HGB", "RBC"],
            (_col(df, "MCH") - mch_calc).abs() / _col(df, "MCH").abs().clip(lower=1e-9) > 0.15,
            "warning")
        add("invalid_codes", "Invalid coded value",
            "Sex and SARSCov must be 0 or 1.",
            ["Sex", "SARSCov"],
            _not_in(_col(df, "Sex"), [0, 1]) | _not_in(_col(df, "SARSCov"), [0, 1]))
        return rules

    return rules


def detect_contradictions(df: pd.DataFrame, dataset_key: str) -> dict:
    """Detect logically / biologically contradictory records.

    Returns a JSON-friendly dict with every rule, the number of flagged rows,
    a few raw examples, and the overall row-level contradiction count.
    """
    rules = _rules_for(df, dataset_key)
    any_mask = pd.Series(False, index=df.index)
    out_rules = []
    n = max(len(df), 1)

    for r in rules:
        mask = r["mask"]
        cnt = int(mask.sum())
        any_mask |= mask
        out_rules.append({
            "id": r["id"], "label": r["label"], "description": r["description"],
            "columns": r["columns"], "severity": r["severity"],
            "count": cnt, "percent": round(cnt / n * 100, 3),
            "row_indices": [int(i) for i in df.index[mask][:100]],
            "examples": _rows(df, mask, r["columns"]),
        })

    out_rules.sort(key=lambda x: (-x["count"], x["id"]))
    per_row = pd.Series(0, index=df.index)
    for r in rules:
        per_row = per_row.add(r["mask"].astype(int), fill_value=0)
    flagged_rows = int((per_row > 0).sum())

    return {
        "total_rules": len(out_rules),
        "triggered_rules": int(sum(1 for r in out_rules if r["count"] > 0)),
        "total_flags": int(sum(r["count"] for r in out_rules)),
        "rows_flagged": flagged_rows,
        "rows_flagged_percent": round(flagged_rows / n * 100, 3),
        "max_contradictions_in_one_row": int(per_row.max()) if len(per_row) else 0,
        "per_row_counts": [int(v) for v in per_row.to_numpy()],
        "rules": out_rules,
    }


# --------------------------------------------------------------------------- #
# Unsupervised outlier ensemble
# --------------------------------------------------------------------------- #
def _clean_numeric_matrix(df_numeric: pd.DataFrame):
    """Median-impute + standardise a numeric frame; drop unusable columns."""
    if df_numeric.empty:
        return None, []
    sub = df_numeric.copy()
    sub = sub.loc[:, sub.notna().any(axis=0)]
    sub = sub.loc[:, sub.nunique(dropna=True) > 1]
    if sub.shape[1] == 0 or len(sub) < 10:
        return None, []
    sub = sub.fillna(sub.median(numeric_only=True))
    sub = sub.fillna(0.0)
    X = StandardScaler().fit_transform(sub.to_numpy(dtype=float))
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, list(sub.columns)


def detect_ml_outliers(df_numeric: pd.DataFrame, contamination: float = 0.05) -> dict:
    """Consensus of six unsupervised anomaly detectors.

    Methods: Isolation Forest, One-Class SVM, Local Outlier Factor,
    Elliptic Envelope, DBSCAN (density) and robust Mahalanobis distance.
    A sample is a "consensus outlier" when at least half of the available
    detectors flag it; "unanimous" means every detector agrees.
    """
    X, cols = _clean_numeric_matrix(df_numeric)
    if X is None:
        return {"error": "Not enough usable numeric data for ML outlier detection"}

    n, dim = X.shape
    if n < 30:
        return {"error": "At least 30 samples are required for ML outlier detection"}

    methods: dict[str, dict] = {}

    def record(name: str, mask: np.ndarray, note: str = ""):
        mask = np.asarray(mask, dtype=bool)
        methods[name] = {
            "outlier_count": int(mask.sum()),
            "outlier_percent": round(float(mask.mean() * 100), 2),
            "indices": np.flatnonzero(mask)[:100].tolist(),
            "mask": mask,
            "note": note,
        }

    # 1. Isolation Forest -----------------------------------------------------
    try:
        iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        record("isolation_forest", iso.fit_predict(X) == -1)
    except Exception as exc:  # pragma: no cover - defensive
        record("isolation_forest", np.zeros(n, dtype=bool), f"failed: {exc}")

    # 2. One-Class SVM --------------------------------------------------------
    try:
        ocsvm = OneClassSVM(nu=contamination, kernel="rbf", gamma="scale")
        record("one_class_svm", ocsvm.fit_predict(X) == -1)
    except Exception as exc:  # pragma: no cover
        record("one_class_svm", np.zeros(n, dtype=bool), f"failed: {exc}")

    # 3. Local Outlier Factor -------------------------------------------------
    try:
        k = min(20, max(5, n // 10), n - 1)
        lof = LocalOutlierFactor(n_neighbors=k, contamination=contamination)
        record("local_outlier_factor", lof.fit_predict(X) == -1,
               f"k = {k} nearest neighbours")
    except Exception as exc:  # pragma: no cover
        record("local_outlier_factor", np.zeros(n, dtype=bool), f"failed: {exc}")

    # 4. Elliptic Envelope (robust Gaussian) ---------------------------------
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            env = EllipticEnvelope(contamination=contamination, random_state=42,
                                   support_fraction=None)
            env_pred = env.fit_predict(X)
        record("elliptic_envelope", env_pred == -1)
    except Exception as exc:  # pragma: no cover
        record("elliptic_envelope", np.zeros(n, dtype=bool), f"failed: {exc}")

    # 5. DBSCAN (density) -----------------------------------------------------
    try:
        nn = NearestNeighbors(n_neighbors=min(6, n)).fit(X)
        dist, _ = nn.kneighbors(X)
        eps = float(np.quantile(dist[:, -1], 0.90)) * 1.25
        eps = max(eps, 1e-3)
        min_samples = max(5, min(2 * dim, 20))
        db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(X)
        record("dbscan", db.labels_ == -1,
               f"eps = {eps:.3f}, min_samples = {min_samples}")
    except Exception as exc:  # pragma: no cover
        record("dbscan", np.zeros(n, dtype=bool), f"failed: {exc}")

    # 6. Robust Mahalanobis distance -----------------------------------------
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mcd = MinCovDet(random_state=42).fit(X)
            dist = mcd.mahalanobis(X)
        threshold = float(np.quantile(dist, 1.0 - contamination))
        record("mahalanobis", dist > threshold, "robust minimum-covariance estimate")
    except Exception as exc:  # pragma: no cover
        record("mahalanobis", np.zeros(n, dtype=bool), f"failed: {exc}")

    usable = [name for name, m in methods.items() if not m.get("note", "").startswith("failed")]
    if not usable:
        return {"error": "All unsupervised detectors failed"}

    vote_matrix = np.vstack([methods[name]["mask"] for name in usable])
    votes = vote_matrix.sum(axis=0)
    half = max(2, int(np.ceil(len(usable) / 2)))
    consensus = votes >= half
    unanimous = votes == len(usable)

    # Pairwise agreement (Jaccard) between detectors
    agreement = []
    for i, a in enumerate(usable):
        for b in usable[i + 1:]:
            ma, mb = methods[a]["mask"], methods[b]["mask"]
            union = int((ma | mb).sum())
            inter = int((ma & mb).sum())
            agreement.append({"a": a, "b": b,
                              "jaccard": round(inter / union, 4) if union else 1.0})

    # Feature-level Isolation Forest: which features drive the anomalies
    feature_level = {}
    for col in cols:
        s = df_numeric[col].astype(float)
        if s.dropna().nunique() <= 1:
            continue
        fill = s.fillna(s.median())
        z = ((fill - fill.median()) / max(robust_sigma(fill), 1e-9)).to_numpy().reshape(-1, 1)
        try:
            p = IsolationForest(contamination=contamination, random_state=42).fit_predict(z)
            feature_level[col] = {
                "outliers": int((p == -1).sum()),
                "percent": round(float((p == -1).mean() * 100), 2),
            }
        except Exception:  # pragma: no cover
            continue

    # Strip internal masks before returning
    public_methods = {}
    for name, m in methods.items():
        public_methods[name] = {k: v for k, v in m.items() if k != "mask"}

    return {
        "total_samples": n,
        "n_features_used": len(cols),
        "contamination_rate": contamination,
        "methods_used": usable,
        "methods": public_methods,
        "consensus_rule": f"flagged by at least {half} of {len(usable)} detectors",
        "consensus_outliers": {
            "count": int(consensus.sum()),
            "percent": round(float(consensus.mean() * 100), 2),
            "indices": np.flatnonzero(consensus)[:100].tolist(),
        },
        "unanimous_outliers": {
            "count": int(unanimous.sum()),
            "percent": round(float(unanimous.mean() * 100), 2),
            "indices": np.flatnonzero(unanimous)[:100].tolist(),
        },
        "vote_counts": [int(v) for v in votes],
        "vote_histogram": {str(k): int((votes == k).sum()) for k in range(len(usable) + 1)},
        "agreement": sorted(agreement, key=lambda x: -x["jaccard"]),
        "feature_level_isolation_forest": feature_level,
    }


# --------------------------------------------------------------------------- #
# Noise-to-signal for tabular data
# --------------------------------------------------------------------------- #
def knn_neighbor_indices(df_numeric: pd.DataFrame, k: int = KNN_NEIGHBOURS) -> tuple[np.ndarray, np.ndarray]:
    """Return (neighbour_indices, distances) for every row.

    Neighbours are searched in the standardised multivariate feature space so a
    reading is compared with *similar records*, not with the next arbitrary CSV
    row.  The point itself is excluded from its own neighbour list.
    """
    X, _ = _clean_numeric_matrix(df_numeric)
    if X is None:
        return np.zeros((0, 0), dtype=int), np.zeros((0, 0))
    n = len(X)
    k = min(k, max(1, n - 1))
    nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(X)
    dist, idx = nn.kneighbors(X)
    return idx[:, 1:], dist[:, 1:]


def compute_noise_metrics(col: str, s: pd.Series,
                          valid_range: tuple[float, float] | None = None,
                          neighbour_idx: np.ndarray | None = None) -> dict:
    """Noise-to-signal statistics for one feature.

    ``neighbour_idx`` must contain row positions into ``s`` (see
    :func:`knn_neighbor_indices`).  When absent, the classical second-difference
    estimator is used (only meaningful for ordered series).
    """
    raw = s.to_numpy(dtype=float)
    n = len(raw)
    bad = np.zeros(n, dtype=bool)
    if valid_range:
        bad = (raw < valid_range[0]) | (raw > valid_range[1])
    v = np.where(bad, np.nan, raw)
    ok = np.isfinite(v)

    scale = robust_sigma(v[ok])
    if scale == 0:
        scale = float(np.nanstd(v[ok], ddof=1)) if ok.sum() > 1 else 0.0

    # (a) NOISE LEVEL --------------------------------------------------------
    if neighbour_idx is not None and len(neighbour_idx) == n and neighbour_idx.size:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            expected = np.nanmedian(v[neighbour_idx], axis=1)
        resid = v - expected
        resid_ok = ok & np.isfinite(resid)
        sigma = robust_sigma(resid[resid_ok])
        nsr_method = f"k-NN local residual (k={neighbour_idx.shape[1]})"
    else:
        # Weather-style second difference: valid only for ordered series
        if n > 2:
            d2 = v[2:] - 2 * v[1:-1] + v[:-2]
            sigma = robust_sigma(d2[np.isfinite(d2)]) / np.sqrt(6)
        else:
            sigma = 0.0
        expected = np.full(n, np.nan)
        resid = v - np.nanmean(v[ok]) if ok.any() else np.zeros(n)
        resid_ok = ok
        nsr_method = "second difference (ordered series)"

    nsr = float(min(sigma / scale, 1.0)) if scale > 0 else 0.0

    # (b) SNR in dB ----------------------------------------------------------
    var_sig = scale ** 2 - sigma ** 2
    snr_db = float(10 * np.log10(var_sig / sigma ** 2)) if (sigma > 0 and var_sig > 0) else None

    # (c) Spikes (Hampel on the local residual) ------------------------------
    r = resid[resid_ok]
    resid_sigma = robust_sigma(r)
    thresh = max(SPIKE_K * resid_sigma, SPIKE_MIN_FRAC * scale)
    spike_mask = np.zeros(n, dtype=bool)
    if thresh > 0:
        spike_mask = ok & np.isfinite(resid) & (np.abs(resid) > thresh)
    spike_cnt = int(spike_mask.sum())
    spike_pct = round(float(spike_cnt / max(n, 1) * 100), 2)
    spike_values = raw[spike_mask][:5].tolist()

    # (d) Repeated / stuck readings ------------------------------------------
    run_id = np.cumsum(np.r_[True, raw[1:] != raw[:-1]]) if n else np.array([], dtype=int)
    counts = np.bincount(run_id) if len(run_id) else np.array([0])
    longest_run = int(counts.max()) if len(counts) else 0
    stuck_val = float(raw[np.where(run_id == counts.argmax())[0][0]]) if len(counts) else None
    zero_frac = float(np.mean(raw == 0)) if n else 0.0
    # A long run of zeros in a zero-inflated feature is not a stuck sensor; a
    # run outside the valid range is an out-of-range error, not a stuck value.
    stuck_flag = bool(
        longest_run >= STUCK_RUN
        and s.nunique() > 20
        and not (stuck_val is not None and valid_range
                 and not (valid_range[0] <= stuck_val <= valid_range[1]))
        and not (stuck_val == 0 and zero_frac > 0.20)
    )

    # (e) Out-of-range values ------------------------------------------------
    n_range = int(bad.sum())

    verdict = ("negligible" if nsr < NSR_LEVELS[0] else "low" if nsr < NSR_LEVELS[1]
               else "moderate" if nsr < NSR_LEVELS[2] else "high")

    # A feature is called *noisy* only when high local variability is backed by
    # spike evidence; an independent but accurate feature can have a high local
    # NSR (it is simply not predictable from its neighbours), so NSR alone would
    # over-flag it.
    noisy = ((nsr >= 0.25 and spike_pct > SPIKE_PCT_FLAG)
             or spike_pct > 2 * SPIKE_PCT_FLAG)

    noise_reasons, error_reasons = [], []
    if noisy and nsr >= NSR_LEVELS[1]:
        noise_reasons.append(f"{verdict} local random fluctuation (NSR {nsr * 100:.1f}% of signal spread)")
    elif noisy:
        noise_reasons.append(f"many isolated spikes ({spike_pct}%) despite a low overall NSR")
    if spike_pct > SPIKE_PCT_FLAG and nsr >= 0.25:
        noise_reasons.append(f"{spike_cnt} isolated spikes ({spike_pct}%)")
    if n_range:
        error_reasons.append(f"{n_range} values outside the physical range {valid_range}")
    if stuck_flag:
        error_reasons.append(f"repeated value {stuck_val:.4g} for {longest_run} consecutive rows")

    return {
        "feature": col,
        "n_samples": n,
        "signal_sigma": round(scale, 4),
        "noise_sigma": round(sigma, 4),
        "noise_to_signal": round(nsr, 4),
        "snr_db": round(snr_db, 2) if snr_db is not None else None,
        "verdict": verdict,
        "nsr_method": nsr_method,
        "spikes_count": spike_cnt,
        "spike_percent": spike_pct,
        "spike_examples": [round(float(x), 4) for x in spike_values],
        "longest_flat_run": longest_run,
        "zero_fraction": round(zero_frac, 4),
        "stuck_sensor": stuck_flag,
        "stuck_value": stuck_val,
        "out_of_range_count": n_range,
        "valid_range": list(valid_range) if valid_range else None,
        "has_noise": bool(noise_reasons),
        "has_errors": bool(error_reasons),
        "noise_reasons": noise_reasons,
        "error_reasons": error_reasons,
    }


def noise_levels() -> dict:
    """Threshold labels for report display."""
    return {
        "negligible": f"< {NSR_LEVELS[0]}",
        "low": f"< {NSR_LEVELS[1]}",
        "moderate": f"< {NSR_LEVELS[2]}",
        "high": f">= {NSR_LEVELS[2]}",
    }
