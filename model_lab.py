"""
model_lab.py - Phase II modelling & evaluation lab.

Replicates the offline study done in the sibling project
``Projet2_Pre_Learning_ML`` but for the four medical datasets of this project:

  * three algorithms compared side by side: KNN, Random Forest (Ensemble), SVM
  * manual hyper-parameter loops (k, n_estimators, kernel/C) recorded as
    *sensitivity curves* (no GridSearchCV: every candidate is a visible row)
  * automatic scaler benchmark (StandardScaler / MinMaxScaler / RobustScaler)
  * Phase-II robustness study evaluated on the *same clean test set*:
        baseline / IQR-cleaned / feature noise / noise -> gaussian smoothed /
        injected extreme outliers
  * hold-out metrics: accuracy, balanced accuracy, macro precision/recall/F1,
    ROC-AUC (binary or macro one-vs-rest), train-vs-test gap diagnosis
  * split + fitted pipelines serialised to ``storage/model_lab/*.pkl``

All returned objects are plain Python structures so the caller can dump them
into ``report.json`` / SQLite without further conversion.
"""
from __future__ import annotations

import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------- #
# Model catalogue (same trio as the whiteboard / Phase II requirement)
# --------------------------------------------------------------------------- #
MODELS = {
    "KNN": ("K-Nearest Neighbors (KNN)", KNeighborsClassifier),
    "Ensemble": ("Random Forest (Ensemble)", RandomForestClassifier),
    "SVM": ("Support Vector Machine (SVM)", SVC),
}

# Manual candidate grids - each row is a CV point of the sensitivity curve.
CANDIDATES = {
    "KNN": [
        {"n_neighbors": k, "metric": m, "weights": "distance"}
        for k in (1, 3, 5, 7, 11, 15)
        for m in ("euclidean", "manhattan")
    ],
    "Ensemble": [
        {"n_estimators": n, "max_depth": 10, "random_state": 42,
         "n_jobs": -1, "class_weight": "balanced"}
        for n in (25, 50, 100, 200)
    ],
    "SVM": [
        {"kernel": "rbf", "C": c, "gamma": "scale", "random_state": 42,
         "max_iter": 3000}
        for c in (0.5, 1.0, 5.0)
    ] + [
        {"kernel": k, "C": 1.0, "random_state": 42, "max_iter": 3000}
        for k in ("linear", "poly")
    ],
}

SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}

# Parameters that are plumbing, not model choices -> hidden from the report.
NOISE_PARAMS = {"random_state", "probability", "max_iter", "n_jobs"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _preprocessor(num_features, cat_features, scaler_name, imputer="median"):
    transformers = [(
        "num",
        Pipeline([("imputer", SimpleImputer(strategy=imputer)),
                  ("scaler", SCALERS[scaler_name]())]),
        num_features,
    )]
    if cat_features:
        transformers.append((
            "cat",
            Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("encoder", OneHotEncoder(handle_unknown="ignore",
                                                sparse_output=False))]),
            cat_features,
        ))
    return ColumnTransformer(transformers, remainder="drop")


def _pipeline(num_features, cat_features, scaler_name, estimator, imputer="median"):
    return Pipeline([("preprocessor", _preprocessor(num_features, cat_features,
                                                    scaler_name, imputer)),
                     ("classifier", estimator)])


def overfitting_label(train_acc, test_acc, cv_mean=None):
    """Same thresholds as Projet2: gap between train and test accuracy."""
    gap = train_acc - test_acc
    if gap > 0.15:
        return ("High Overfitting Risk", "danger",
                f"Train {train_acc*100:.1f}% >> Test {test_acc*100:.1f}% "
                f"(gap {gap*100:.1f} points).")
    if gap > 0.07:
        return ("Moderate Overfitting", "warning",
                f"Train {train_acc*100:.1f}% > Test {test_acc*100:.1f}% "
                f"(gap {gap*100:.1f} points).")
    if gap < -0.05:
        return ("Underfitting", "info",
                f"Test {test_acc*100:.1f}% >= Train {train_acc*100:.1f}%: "
                "the model is not exploiting the training signal.")
    cv_txt = f" ~ CV {cv_mean*100:.1f}%" if cv_mean is not None else ""
    return ("Well-Balanced", "success",
            f"Train {train_acc*100:.1f}%{cv_txt} ~ Test {test_acc*100:.1f}%.")


def _safe_cv(y, folds=5):
    """Shrink the fold count so every class is represented in every fold."""
    counts = pd.Series(y).value_counts()
    k = int(min(folds, counts.min()))
    if k < 2:
        return None
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=42)


def _clean_params(params):
    return {k: (v if not isinstance(v, np.integer) else int(v))
            for k, v in params.items() if k not in NOISE_PARAMS}


# --------------------------------------------------------------------------- #
# 1. Scaler benchmark
# --------------------------------------------------------------------------- #
def _best_scaler(X_train, y_train, num_features, cat_features, cv, imputer="median"):
    """Pick the scaler with the best baseline CV balanced accuracy (KNN k=5)."""
    best_name, best_score, table = "standard", -1.0, []
    for name in SCALERS:
        pipe = _pipeline(num_features, cat_features, name,
                         KNeighborsClassifier(n_neighbors=5), imputer)
        try:
            score = float(cross_val_score(pipe, X_train, y_train, cv=cv,
                                          scoring="balanced_accuracy", n_jobs=1).mean())
        except Exception:
            score = -1.0
        table.append({"scaler": name, "cv_balanced_accuracy": round(score, 4)})
        if score > best_score:
            best_name, best_score = name, score
    return best_name, table


# --------------------------------------------------------------------------- #
# 2. Manual tuning loop (sensitivity curve)
# --------------------------------------------------------------------------- #
def _tune(model_key, X_train, y_train, num_features, cat_features, scaler, cv, imputer="median"):
    est_cls = MODELS[model_key][1]
    curve, best_params, best_score = [], None, -np.inf

    for params in CANDIDATES[model_key]:
        pipe = _pipeline(num_features, cat_features, scaler, est_cls(**params), imputer)
        try:
            scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                     scoring="balanced_accuracy", n_jobs=1)
            mean_s, std_s = float(scores.mean()), float(scores.std())
        except Exception:
            mean_s, std_s = float("nan"), float("nan")
        curve.append({"params": _clean_params(params),
                      "cv_balanced_accuracy": None if np.isnan(mean_s) else round(mean_s, 4),
                      "cv_std": None if np.isnan(std_s) else round(std_s, 4)})
        if not np.isnan(mean_s) and mean_s > best_score:
            best_score, best_params = mean_s, params

    if best_params is None:                      # every candidate failed
        best_params = CANDIDATES[model_key][0]

    # Refit the winner; SVM needs probability=True for ROC curves.
    final = dict(best_params)
    if model_key == "SVM":
        final["probability"] = True
    fitted = _pipeline(num_features, cat_features, scaler, est_cls(**final), imputer)
    fitted.fit(X_train, y_train)
    return fitted, best_params, curve


# --------------------------------------------------------------------------- #
# 3. Robustness scenarios (Phase II: noise & outliers)
# --------------------------------------------------------------------------- #
def _iqr_filter(X, y, factor=1.5):
    """Drop rows where >=15% of continuous features fall outside 1.5*IQR."""
    numeric = X.select_dtypes(include=[np.number])
    numeric = numeric.loc[:, numeric.nunique() > 2]
    if numeric.shape[1] == 0:
        return X, y, 0
    q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
    iqr = (q3 - q1).replace(0, np.nan)
    outside = ((numeric < q1 - factor * iqr) | (numeric > q3 + factor * iqr))
    mask = outside.sum(axis=1) >= max(1, int(numeric.shape[1] * 0.15))
    if (~mask).sum() < 0.5 * len(X):             # too aggressive -> stricter rule
        mask = outside.sum(axis=1) >= max(2, int(numeric.shape[1] * 0.25))
    return X[~mask].reset_index(drop=True), y[~mask].reset_index(drop=True), int(mask.sum())


def _add_noise(X, level=0.18, seed=42):
    rng = np.random.RandomState(seed)
    out = X.copy()
    cols = out.select_dtypes(include=[np.number]).columns
    std = out[cols].std().replace(0, np.nan).fillna(1.0)
    out[cols] = out[cols] + rng.normal(0, level * std.values, size=(len(out), len(cols)))
    return out


def _smooth(X, sigma=1.2):
    out = X.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        out[col] = gaussian_filter1d(out[col].values.astype(float), sigma=sigma)
    return out


def _inject_outliers(X, contamination=0.08, factor=3.5, seed=42):
    rng = np.random.RandomState(seed)
    out = X.copy().reset_index(drop=True)
    cols = list(out.select_dtypes(include=[np.number]).columns)
    if not cols:
        return out, 0
    idx = rng.choice(len(out), size=max(2, int(len(out) * contamination)), replace=False)
    std = out[cols].std().replace(0, np.nan).fillna(1.0)
    for i in idx:
        chosen = rng.choice(len(cols), size=min(3, len(cols)), replace=False)
        for j in chosen:
            out.iloc[i, out.columns.get_loc(cols[j])] += rng.choice([-1, 1]) * factor * std.iloc[j]
    return out, len(idx)


def _robustness(models, X_train, y_train, X_test, y_test,
                num_features, cat_features, scaler, imputer):
    X_clean, y_clean, removed = _iqr_filter(X_train, y_train)
    X_noisy = _add_noise(X_train)
    X_smoothed = _smooth(X_noisy)
    X_out, injected = _inject_outliers(X_train)

    scenarios = [
        ("baseline", "Baseline (standard)", X_train, y_train,
         "Standard deduplicated training set."),
        ("without_outliers", "Without outliers (IQR)", X_clean, y_clean,
         f"{removed} row(s) removed with the IQR rule."),
        ("with_noise", "With feature noise", X_noisy, y_train,
         "18% Gaussian perturbation on every continuous feature."),
        ("noise_smoothed", "Noise -> Gaussian smoothed", X_smoothed, y_train,
         "Noise injected then suppressed with gaussian_filter1d (sigma=1.2)."),
        ("with_outliers", "With injected outliers", X_out, y_train,
         f"{injected} row(s) spiked at 3.5 sd on up to 3 features."),
    ]

    scores = {}
    for key, label, X_tr, y_tr, desc in scenarios:
        scores[key] = {"label": label, "description": desc, "scores": {}}
        for mkey, fitted in models.items():
            est_cls = MODELS[mkey][1]
            params = {k: v for k, v in fitted.named_steps["classifier"].get_params().items()
                      if not k.startswith("_") and k != "probability"}
            pipe = _pipeline(num_features, cat_features, scaler, est_cls(**params), imputer)
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_test)
            scores[key]["scores"][mkey] = {
                "accuracy": round(float(accuracy_score(y_test, pred)), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(y_test, pred)), 4),
                "f1_macro": round(float(f1_score(y_test, pred, average="macro",
                                                 zero_division=0)), 4),
            }

    comparison = []
    for mkey in models:
        base = scores["baseline"]["scores"][mkey]
        rows = {k: scores[k]["scores"][mkey] for k in scores}
        comparison.append({
            "model_key": mkey,
            "model_name": MODELS[mkey][0],
            "baseline_acc": rows["baseline"]["accuracy"],
            "without_outliers_acc": rows["without_outliers"]["accuracy"],
            "noisy_acc": rows["with_noise"]["accuracy"],
            "smoothed_acc": rows["noise_smoothed"]["accuracy"],
            "outliers_acc": rows["with_outliers"]["accuracy"],
            "noise_delta": round(rows["with_noise"]["accuracy"] - base["accuracy"], 4),
            "smooth_recovery": round(rows["noise_smoothed"]["accuracy"]
                                     - rows["with_noise"]["accuracy"], 4),
            "outlier_delta": round(rows["with_outliers"]["accuracy"] - base["accuracy"], 4),
            "iqr_cleanup_delta": round(rows["without_outliers"]["accuracy"]
                                       - base["accuracy"], 4),
        })
    return {"scenarios": scores, "comparison": comparison,
            "stats": {"outliers_removed": removed, "outliers_injected": injected,
                      "rows_used": int(len(X_train))}}


# --------------------------------------------------------------------------- #
# 4. Metric block for one fitted pipeline
# --------------------------------------------------------------------------- #
def _evaluate(pipe, X_train, y_train, X_test, y_test, cv):
    pred = pipe.predict(X_test)
    pred_train = pipe.predict(X_train)
    classes = list(pipe.named_steps["classifier"].classes_)

    test_acc = float(accuracy_score(y_test, pred))
    train_acc = float(accuracy_score(y_train, pred_train))
    bal = float(balanced_accuracy_score(y_test, pred))

    cv_mean = cv_std = None
    if cv is not None:
        try:
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                        scoring="balanced_accuracy", n_jobs=1)
            cv_mean, cv_std = float(cv_scores.mean()), float(cv_scores.std())
        except Exception:
            pass

    # ROC: binary -> curve of the positive class, multiclass -> micro-average.
    auc_val, fpr, tpr, roc_mode = None, [0.0, 1.0], [0.0, 1.0], "binary"
    try:
        proba = pipe.predict_proba(X_test)
        if len(classes) == 2:
            fpr, tpr, _ = roc_curve((y_test == classes[1]).astype(int), proba[:, 1])
            auc_val = float(roc_auc_score((y_test == classes[1]).astype(int), proba[:, 1]))
        else:
            roc_mode = "micro"
            # One-vs-rest matrix (same class order as predict_proba), flattened
            # so the curve is a micro-average across all classes.
            y_bin = np.stack([(y_test.to_numpy() == c).astype(int)
                              for c in classes], axis=1).ravel()
            fpr, tpr, _ = roc_curve(y_bin, proba.ravel())
            auc_val = float(roc_auc_score(y_test, proba, multi_class="ovr",
                                          average="macro", labels=classes))
    except Exception:
        pass

    cm = confusion_matrix(y_test, pred, labels=classes)
    status, badge, desc = overfitting_label(train_acc, test_acc, cv_mean)

    return {
        "classes": [str(c) for c in classes],
        "test_accuracy": round(test_acc, 4),
        "train_accuracy": round(train_acc, 4),
        "balanced_accuracy": round(bal, 4),
        "cv_balanced_accuracy_mean": None if cv_mean is None else round(cv_mean, 4),
        "cv_balanced_accuracy_std": None if cv_std is None else round(cv_std, 4),
        "precision_macro": round(float(precision_score(y_test, pred, average="macro",
                                                       zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_test, pred, average="macro",
                                                 zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_test, pred, average="macro", zero_division=0)), 4),
        "auc": None if auc_val is None else round(auc_val, 4),
        "roc": {"mode": roc_mode,
                "fpr": [round(float(x), 4) for x in fpr],
                "tpr": [round(float(x), 4) for x in tpr]},
        "confusion_matrix": cm.astype(int).tolist(),
        "overfitting_status": status,
        "overfitting_badge": badge,
        "overfitting_desc": desc,
    }


# --------------------------------------------------------------------------- #
# 5. Pickle persistence
# --------------------------------------------------------------------------- #
def _save_pickles(dataset_key, stamp, X_train, X_test, y_train, y_test,
                  feature_names, models, scaler, storage_dir):
    root = Path(storage_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {}

    split_path = root / f"{dataset_key}_{stamp}_split.pkl"
    with open(split_path, "wb") as fh:
        pickle.dump({
            "dataset_id": dataset_key, "timestamp": datetime.now().isoformat(timespec="seconds"),
            "feature_names": feature_names,
            "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        }, fh)
    paths["split"] = str(split_path)

    for key, pipe in models.items():
        p = root / f"{dataset_key}_{key}_{stamp}.pkl"
        with open(p, "wb") as fh:
            pickle.dump(pipe, fh)
        paths[key] = str(p)
    paths["scaler"] = scaler
    return paths


# --------------------------------------------------------------------------- #
# 6. Public entry point
# --------------------------------------------------------------------------- #
def run_lab(df, target, num_features, cat_features, dataset_key="dataset",
            storage_dir="storage/model_lab", test_size=0.25, cv_folds=5,
            random_state=42, persist=True):
    """Run the full Phase-II study on one cleaned dataframe.

    Parameters
    ----------
    df            : cleaned rows (impossible values already masked)
    target        : target column name
    num_features  : numeric predictors
    cat_features  : categorical predictors (disjoint from num_features)
    persist       : write split/model pickles to ``storage_dir``

    Returns
    -------
    dict with keys: scaler, scaler_table, cv_folds, split, models, best_model,
    sensitivity, noise_outlier, pickles
    """
    cat_features = [c for c in cat_features if c not in num_features]
    X = df[num_features + cat_features].copy()
    y = df[target].astype(str).str.strip()

    cv = _safe_cv(y, cv_folds)
    stratify = y if (cv is not None) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify)

    scaler, scaler_table = _best_scaler(X_train, y_train, num_features, cat_features,
                                        cv or 3)

    models, sensitivity, metrics = {}, {}, {}
    for key in MODELS:
        pipe, best_params, curve = _tune(key, X_train, y_train, num_features,
                                         cat_features, scaler, cv or 3)
        models[key] = pipe
        sensitivity[key] = curve
        block = _evaluate(pipe, X_train, y_train, X_test, y_test, cv)
        block.update({
            "model_key": key,
            "model_name": MODELS[key][0],
            "best_params": _clean_params(best_params),
            "cv_points": len(curve),
        })
        metrics[key] = block

    best_key = max(metrics, key=lambda k: (metrics[k]["f1_macro"],
                                           metrics[k]["test_accuracy"]))
    noise_outlier = _robustness(models, X_train, y_train, X_test, y_test,
                                num_features, cat_features, scaler, "median")

    pickles = {}
    if persist:
        try:
            pickles = _save_pickles(dataset_key, datetime.now().strftime("%Y%m%d_%H%M%S"),
                                    X_train, X_test, y_train, y_test,
                                    list(X.columns), models, scaler, storage_dir)
            meta_path = Path(storage_dir) / f"{dataset_key}_{datetime.now():%Y%m%d_%H%M%S}_lab.json"
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump({"dataset": dataset_key, "target": target, "scaler": scaler,
                           "best_model": best_key,
                           "models": {k: v["best_params"] for k, v in metrics.items()},
                           "pickles": pickles}, fh, indent=2)
            pickles["metadata"] = str(meta_path)
        except Exception as exc:                 # persistence must never break the report
            pickles = {"error": str(exc)}

    return {
        "scaler": scaler,
        "scaler_table": scaler_table,
        "cv_folds": (cv.get_n_splits() if cv is not None else None),
        "split": {"test_size": test_size, "n_train": int(len(X_train)),
                  "n_test": int(len(X_test)), "n_features": int(X.shape[1])},
        "models": [metrics[k] for k in MODELS],
        "best_model": best_key,
        "sensitivity": sensitivity,
        "noise_outlier": noise_outlier,
        "pickles": pickles,
    }
