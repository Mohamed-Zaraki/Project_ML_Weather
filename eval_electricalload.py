"""eval_electricalload.py
Evaluation Phase — ElectricalLoad Dataset
Predicting Electrical Demand (DEMAND) using:
  1. Linear Regression
  2. Polynomial Regression (degree 2)
  3. Decision Tree Regressor

Evaluation metrics (ONLY Test and CV):
  - R² (Test, CV)
  - RMSE (Test, CV)
"""
from __future__ import annotations

import pathlib
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "datasets_extracted" / "ElectricalLoad" / "electricalload_preprocessed.csv"
OUT  = ROOT / "evaluation_electricalload"
OUT.mkdir(exist_ok=True)

# Elegant color palette
PALETTE = ["#2b5c8f", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
DPI = 160

# ── 1. Load Data ──────────────────────────────────────────────────────────────
print("Loading ElectricalLoad dataset...")
df = pd.read_csv(DATA)

feature_cols = [c for c in df.columns if c not in ("datetime", "DEMAND", "split")]
target_col   = "DEMAND"

if "split" in df.columns:
    train_df = df[df["split"] == "train"].copy()
    test_df  = df[df["split"] == "test"].copy()
else:
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

X_train = train_df[feature_cols].values
y_train = train_df[target_col].values
X_test  = test_df[feature_cols].values
y_test  = test_df[target_col].values

print(f"Data ready: Train={X_train.shape[0]} samples, Test={X_test.shape[0]} samples, Features={X_train.shape[1]}")

# ── 2. Define Algorithms ──────────────────────────────────────────────────────
models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LinearRegression()),
    ]),
    "Polynomial Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("poly",   PolynomialFeatures(degree=2, include_bias=False)),
        ("model",  LinearRegression()),
    ]),
    "Decision Tree": Pipeline([
        ("model", DecisionTreeRegressor(max_depth=12, min_samples_leaf=10, random_state=42)),
    ]),
}

# ── 3. Train & Evaluate (ONLY Test and CV) ─────────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = {}
predictions = {}

for name, pipe in models.items():
    print(f"  Training {name}...")
    pipe.fit(X_train, y_train)

    y_pred_test = pipe.predict(X_test)

    r2_test   = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Cross-validation scores (CV R2 and CV RMSE)
    cv_r2   = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="r2", n_jobs=1).mean()
    cv_rmse = (-cross_val_score(pipe, X_train, y_train, cv=kf, scoring="neg_root_mean_squared_error", n_jobs=1)).mean()

    results[name] = {
        "R² Test":   r2_test,
        "CV R²":     cv_r2,
        "RMSE Test": rmse_test,
        "CV RMSE":   cv_rmse,
    }
    predictions[name] = {"test": y_pred_test}

# Print clean summary table with ONLY Test and CV
print("\n" + "=" * 66)
print(f"{'Algorithm':<25} {'R² Test':>10} {'CV R²':>10} {'RMSE Test':>10} {'CV RMSE':>10}")
print("-" * 66)
for name, m in results.items():
    print(f"{name:<25} {m['R² Test']:>10.4f} {m['CV R²']:>10.4f} {m['RMSE Test']:>10.2f} {m['CV RMSE']:>10.2f}")
print("=" * 66 + "\n")

# Save table to CSV
summary_df = pd.DataFrame(results).T
summary_df.to_csv(OUT / "metrics_summary.csv", index_label="Algorithm")

# ── 4. Generate Clean Plots (ONLY Test and CV) ─────────────────────────────────
algo_names = list(results.keys())

# ━━━ Plot 1: R² Comparison (Test vs CV) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, ax = plt.subplots(figsize=(9, 5.5))
x = np.arange(len(algo_names))
w = 0.32

r1 = ax.bar(x - w/2, [results[n]["R² Test"] for n in algo_names], w, label="R² Test", color="#2ca02c", edgecolor="white")
r2 = ax.bar(x + w/2, [results[n]["CV R²"]   for n in algo_names], w, label="CV R²",   color="#ff7f0e", edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels(algo_names, fontsize=11, fontweight="bold")
ax.set_ylabel("R² Score (Higher is Better)", fontsize=12)
ax.set_title("R² Score Comparison (Test vs CV) — ElectricalLoad", fontsize=13, fontweight="bold", pad=12)
ax.set_ylim(0, 1.1)
ax.legend(frameon=True, fontsize=11)

for rects in [r1, r2]:
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 0.015, f"{h:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

fig.tight_layout()
fig.savefig(OUT / "01_r2_comparison.png", dpi=DPI)
plt.close(fig)
print("  ✓ Saved 01_r2_comparison.png")

# ━━━ Plot 2: RMSE Comparison (Test vs CV) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, ax = plt.subplots(figsize=(9, 5.5))
r1 = ax.bar(x - w/2, [results[n]["RMSE Test"] for n in algo_names], w, label="RMSE Test", color="#d62728", edgecolor="white")
r2 = ax.bar(x + w/2, [results[n]["CV RMSE"]   for n in algo_names], w, label="CV RMSE",   color="#9467bd", edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels(algo_names, fontsize=11, fontweight="bold")
ax.set_ylabel("RMSE in MW (Lower is Better)", fontsize=12)
ax.set_title("RMSE Comparison (Test vs CV) — ElectricalLoad", fontsize=13, fontweight="bold", pad=12)
ax.legend(frameon=True, fontsize=11)

for rects in [r1, r2]:
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 1.2, f"{h:.1f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

fig.tight_layout()
fig.savefig(OUT / "02_rmse_comparison.png", dpi=DPI)
plt.close(fig)
print("  ✓ Saved 02_rmse_comparison.png")

# ━━━ Plot 3: Combined Direct Comparison (R² & RMSE) ━━━━━━━━━━━━━━━━━━━━━━━━
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Subplot 1: R² Test & CV
ax1.bar(x - w/2, [results[n]["R² Test"] for n in algo_names], w, label="Test R²", color="#2ca02c")
ax1.bar(x + w/2, [results[n]["CV R²"]   for n in algo_names], w, label="CV R²",   color="#ff7f0e")
ax1.set_xticks(x)
ax1.set_xticklabels(algo_names, fontsize=10, fontweight="bold")
ax1.set_ylabel("R² Score", fontsize=11)
ax1.set_title("R² Score (Test vs CV)", fontsize=12, fontweight="bold")
ax1.set_ylim(0, 1.1)
ax1.legend()
for i, n in enumerate(algo_names):
    ax1.text(i - w/2, results[n]["R² Test"] + 0.02, f"{results[n]['R² Test']:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax1.text(i + w/2, results[n]["CV R²"] + 0.02,   f"{results[n]['CV R²']:.3f}", ha="center", fontsize=9, fontweight="bold")

# Subplot 2: RMSE Test & CV
ax2.bar(x - w/2, [results[n]["RMSE Test"] for n in algo_names], w, label="Test RMSE", color="#d62728")
ax2.bar(x + w/2, [results[n]["CV RMSE"]   for n in algo_names], w, label="CV RMSE",   color="#9467bd")
ax2.set_xticks(x)
ax2.set_xticklabels(algo_names, fontsize=10, fontweight="bold")
ax2.set_ylabel("RMSE (MW)", fontsize=11)
ax2.set_title("RMSE Error (Test vs CV)", fontsize=12, fontweight="bold")
ax2.legend()
for i, n in enumerate(algo_names):
    ax2.text(i - w/2, results[n]["RMSE Test"] + 2, f"{results[n]['RMSE Test']:.1f}", ha="center", fontsize=9, fontweight="bold")
    ax2.text(i + w/2, results[n]["CV RMSE"] + 2,   f"{results[n]['CV RMSE']:.1f}", ha="center", fontsize=9, fontweight="bold")

fig.suptitle("Performance Overview: R² and RMSE (Test & CV Only)", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "03_r2_and_rmse_overview.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("  ✓ Saved 03_r2_and_rmse_overview.png")

# ━━━ Plot 4: Actual vs Predicted Demand (Test Set) ━━━━━━━━━━━━━━━━━━━━━━━━━
fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

for i, (name, col) in enumerate(zip(algo_names, colors)):
    ax = axes[i]
    y_pred = predictions[name]["test"]
    ax.scatter(y_test, y_pred, alpha=0.6, s=25, color=col, edgecolor="none")
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "k--", lw=1.8, label="Ideal (y = x)")
    ax.set_title(f"{name}\nR² = {results[name]['R² Test']:.3f} | RMSE = {results[name]['RMSE Test']:.1f}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Actual Demand (MW)", fontsize=11)
    ax.set_ylabel("Predicted Demand (MW)", fontsize=11)
    ax.legend(loc="upper left", fontsize=10)

fig.suptitle("Actual vs. Predicted Electrical Load Demand (Test Set)", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "04_actual_vs_predicted.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("  ✓ Saved 04_actual_vs_predicted.png")

# ━━━ Plot 5: Residuals Distribution (Test Set) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
for i, (name, col) in enumerate(zip(algo_names, colors)):
    ax = axes[i]
    res = y_test - predictions[name]["test"]
    ax.hist(res, bins=25, color=col, alpha=0.75, edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", lw=1.5)
    ax.set_title(f"{name} Residuals\nMean: {res.mean():.1f} | Std: {res.std():.1f}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Residual (Actual − Predicted)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)

fig.suptitle("Error (Residual) Distribution on Test Set", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "05_residuals_distribution.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("  ✓ Saved 05_residuals_distribution.png")

# ━━━ Plot 6: Metric Scoreboard Heatmap (ONLY Test & CV) ━━━━━━━━━━━━━━━━━━━━
fig, ax = plt.subplots(figsize=(8, 3.5))
table_display = summary_df[["R² Test", "CV R²", "RMSE Test", "CV RMSE"]]
sns.heatmap(table_display, annot=True, fmt=".3f", cmap="Blues", cbar=False,
            linewidths=2, linecolor="white", annot_kws={"size": 13, "weight": "bold"}, ax=ax)
ax.set_title("Scoreboard: R² and RMSE (Test vs CV Only)", fontsize=13, fontweight="bold", pad=12)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11, fontweight="bold")
ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "06_scoreboard_heatmap.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("  ✓ Saved 06_scoreboard_heatmap.png")

print(f"\nAll plots and metrics ready in {OUT}!")
