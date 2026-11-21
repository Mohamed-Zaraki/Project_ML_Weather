"""ensemble_all_datasets.py
Ensemble Learning Benchmark across 5 Datasets:
  - Weather
  - Heart Disease
  - Alcohol Detection
  - Parkinson's Disease
  - COVID-19

Algorithms compared:
  1. AdaBoost
  2. XGBoost
  3. LightGBM
  4. CatBoost
  5. Gradient Boosting

Evaluation Metrics:
  - Accuracy, Balanced Accuracy, F1-Macro, Precision, Recall, ROC-AUC
  - 5-Fold Stratified Cross-Validation
"""
from __future__ import annotations

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import pathlib
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent
OUT  = ROOT / "evaluation_ensemble"
OUT.mkdir(exist_ok=True)

# Elegant professional palette
PALETTE = ["#2b5c8f", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
DPI = 160

# ── Dataset definitions ───────────────────────────────────────────────────────
DATASETS = {
    "Weather": {
        "path": ROOT / "datasets_extracted" / "weather-dataset" / "weatherHistory.csv",
        "target": "Precip Type",
        "drop": ["Formatted Date", "Daily Summary", "Loud Cover"],
        "sample": 6000,  # Stratified sample for fast & stable cross-validation
    },
    "Heart Disease": {
        "path": ROOT / "datasets_extracted" / "Heart Attack Analysis" / "heart.csv",
        "target": "output",
        "drop": [],
        "sample": None,
    },
    "Alcohol": {
        "path": ROOT / "datasets_extracted" / "alcohol" / "alcohol" / "alcohol.csv",
        "target": "CLASS",
        "drop": [],
        "sample": None,
    },
    "Parkinson": {
        "path": ROOT / "datasets_extracted" / "parkinson" / "parkinson" / "data1.csv",
        "target": "Group (Normal/PD/SWEDD - 1/2/3)",
        "drop": ["Subject_id"],
        "sample": None,
    },
    "COVID-19": {
        "path": ROOT / "datasets_extracted" / "Covid19" / "dataset2-covid19.csv",
        "target": "SARSCov",
        "drop": [],
        "sample": None,
    },
}


def load_dataset(cfg):
    df = pd.read_csv(cfg["path"])
    target = cfg["target"]
    df = df.dropna(subset=[target])

    drop_cols = [c for c in cfg["drop"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Encode categorical predictor columns
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if col == target:
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Encode target to 0..K-1
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(df[target])
    df[target] = y_encoded

    # Stratified subsample if requested
    if cfg["sample"] is not None and len(df) > cfg["sample"]:
        df, _ = train_test_split(df, train_size=cfg["sample"],
                                 stratify=df[target], random_state=42)

    X = df.drop(columns=[target])
    y = df[target].values

    # Impute numeric NaNs
    imp = SimpleImputer(strategy="median")
    X_imputed = imp.fit_transform(X)

    return X_imputed, y, le_target.classes_


def get_models():
    """Return dictionary of the 5 ensemble algorithms."""
    return {
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100, learning_rate=0.8, random_state=42
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=42, eval_metric="logloss", verbosity=0
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=42, verbose=-1
        ),
        "CatBoost": cb.CatBoostClassifier(
            iterations=100, depth=4, learning_rate=0.1,
            random_state=42, verbose=0
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        ),
    }


# ── Run Benchmark ─────────────────────────────────────────────────────────────
all_results = {}
all_predictions = {}

print("=" * 80)
print("ENSEMBLE LEARNING BENCHMARK — 5 ALGORITHMS ACROSS 5 DATASETS")
print("=" * 80)

for ds_name, cfg in DATASETS.items():
    print(f"\n📂 Processing Dataset: [{ds_name}] ...")
    X, y, class_names = load_dataset(cfg)
    n_classes = len(np.unique(y))
    print(f"   Samples: {X.shape[0]} | Features: {X.shape[1]} | Target: {cfg['target']} ({n_classes} classes)")

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models_dict = get_models()
    min_count = pd.Series(y_train).value_counts().min()
    cv_folds = min(5, min_count)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    ds_res = {}
    ds_pred = {}

    for mname, model in models_dict.items():
        t0 = time.time()
        model.fit(X_train_sc, y_train)
        fit_time = time.time() - t0

        y_pred = model.predict(X_test_sc)
        if hasattr(y_pred, "ravel"):
            y_pred = y_pred.ravel().astype(int)

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train_sc, y_train, cv=cv, scoring="accuracy", n_jobs=1)
        cv_mean = cv_scores.mean()

        # ROC-AUC & Probabilities
        auc_score = None
        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test_sc)
                if n_classes == 2:
                    auc_score = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    auc_score = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
            except Exception:
                pass

        ds_res[mname] = {
            "Accuracy": acc,
            "Balanced Acc": bal_acc,
            "F1-Macro": f1,
            "Precision": prec,
            "Recall": rec,
            "CV Accuracy": cv_mean,
            "AUC": auc_score,
            "Fit Time (s)": fit_time,
        }
        ds_pred[mname] = {
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "n_classes": n_classes,
        }
        print(f"   ✓ {mname:<18} -> Acc: {acc:.4f} | F1: {f1:.4f} | CV: {cv_mean:.4f} ({fit_time:.2f}s)")

    all_results[ds_name] = ds_res
    all_predictions[ds_name] = ds_pred

# ── Save CSV Metrics ──────────────────────────────────────────────────────────
metrics_records = []
for ds_name, ds_res in all_results.items():
    for mname, m in ds_res.items():
        metrics_records.append({
            "Dataset": ds_name,
            "Algorithm": mname,
            "Accuracy": m["Accuracy"],
            "Balanced Acc": m["Balanced Acc"],
            "F1-Macro": m["F1-Macro"],
            "Precision": m["Precision"],
            "Recall": m["Recall"],
            "CV Accuracy": m["CV Accuracy"],
            "AUC": m["AUC"],
        })
metrics_df = pd.DataFrame(metrics_records)
metrics_df.to_csv(OUT / "ensemble_metrics_summary.csv", index=False)

# ── INDIVIDUAL DATASET PLOTS ──────────────────────────────────────────────────
algos = list(get_models().keys())
safe_names = {
    "Weather": "weather",
    "Heart Disease": "heart_disease",
    "Alcohol": "alcohol",
    "Parkinson": "parkinson",
    "COVID-19": "covid19",
}

for ds_name, ds_res in all_results.items():
    sname = safe_names[ds_name]
    preds = all_predictions[ds_name]

    # ━━━ Plot 1: Accuracy & CV Comparison Bar ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(algos))
    w = 0.35

    test_accs = [ds_res[m]["Accuracy"] for m in algos]
    cv_accs   = [ds_res[m]["CV Accuracy"] for m in algos]

    b1 = ax.bar(x - w/2, test_accs, w, label="Test Accuracy", color="#2b5c8f", edgecolor="white")
    b2 = ax.bar(x + w/2, cv_accs,   w, label="5-Fold CV Accuracy", color="#ff7f0e", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"Ensemble Models Comparison — {ds_name} Dataset", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)

    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.015, f"{h:.3f}", ha="center", fontsize=9, fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.015, f"{h:.3f}", ha="center", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT / f"{sname}_01_accuracy_comparison.png", dpi=DPI)
    plt.close(fig)

    # ━━━ Plot 2: Confusion Matrices Grid ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i, m in enumerate(algos):
        ax = axes[i]
        cm = confusion_matrix(preds[m]["y_test"], preds[m]["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    annot_kws={"size": 11, "weight": "bold"})
        ax.set_title(f"{m}\nAcc: {ds_res[m]['Accuracy']:.3f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle(f"Confusion Matrices for Ensemble Models — {ds_name}", fontsize=13, fontweight="bold", y=1.05)
    fig.tight_layout()
    fig.savefig(OUT / f"{sname}_02_confusion_matrices.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # ━━━ Plot 3: ROC Curves (if binary) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if preds[algos[0]]["n_classes"] == 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, m in enumerate(algos):
            y_proba = preds[m]["y_proba"]
            if y_proba is not None:
                fpr, tpr, _ = roc_curve(preds[m]["y_test"], y_proba[:, 1])
                auc_val = ds_res[m]["AUC"]
                ax.plot(fpr, tpr, lw=2, color=PALETTE[i],
                        label=f"{m} (AUC = {auc_val:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Random Guess")
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title(f"ROC Curves — {ds_name} Dataset", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, loc="lower right")
        fig.tight_layout()
        fig.savefig(OUT / f"{sname}_03_roc_curves.png", dpi=DPI)
        plt.close(fig)

print("  ✓ Individual dataset plots generated successfully.")
print(f"\n🎉 All ensemble evaluation plots successfully generated in: {OUT}\n")

