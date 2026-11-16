"""weather_regression.py : MODEL de l'évaluation météo (régressions), même protocole que regression_weather.py.

Pour chaque cible (Apparent Temperature, Humidity, Wind Speed, Wind Bearing, Visibility) prédite par UNE seule variable
(Température, puis Pression) : Linéaire, Polynomiale (degré 3) et Random Forest, comparés par R2 et RMSE sur un test de 20 %.
Dataset inchangé (aucune ligne supprimée), comme dans le script d'origine.

Usage :  python weather_regression.py weatherHistory.csv latest_weather_report.json
         -> ajoute la section « regression_lab » au rapport JSON (aucun affichage ici, la vue lit le JSON).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

PREDICTORS = ["Temperature (C)", "Pressure (millibars)"]
TARGETS = ["Apparent Temperature (C)", "Humidity", "Wind Speed (km/h)", "Wind Bearing (degrees)", "Visibility (km)"]
MODELS = ["Linéaire", "Polynomiale", "Random Forest"]
SCATTER_POINTS = 800          # échantillon de points gardé dans le JSON pour le nuage
GRID_POINTS = 120             # points des courbes de chaque modèle


def _new_models() -> dict:
    return {
        "Linéaire": LinearRegression(),
        "Polynomiale": make_pipeline(PolynomialFeatures(3), LinearRegression()),
        "Random Forest": RandomForestRegressor(n_estimators=50, min_samples_leaf=50, n_jobs=-1, random_state=0),
    }


def run(csv_path: str | Path) -> dict:
    df = pd.read_csv(csv_path)
    rows, curves, scatter = [], {}, {}
    for pred in PREDICTORS:
        for target in TARGETS:
            X, y = df[[pred]], df[target]
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
            grid = pd.DataFrame({pred: np.linspace(X[pred].min(), X[pred].max(), GRID_POINTS)})
            key = f"{pred}|{target}"
            curves[key] = {"x": [round(float(v), 4) for v in grid[pred]]}
            for name, model in _new_models().items():
                model.fit(X_tr, y_tr)
                p = model.predict(X_te)
                rows.append({"predictor": pred, "target": target, "model": name,
                             "r2": round(float(r2_score(y_te, p)), 4),
                             "rmse": round(float(np.sqrt(mean_squared_error(y_te, p))), 4)})
                curves[key][name] = [round(float(v), 4) for v in model.predict(grid)]
            samp = df[[pred, target]].sample(SCATTER_POINTS, random_state=0)
            scatter[key] = {"x": [round(float(v), 4) for v in samp[pred]], "y": [round(float(v), 4) for v in samp[target]]}
    n_zero_pressure = int((df["Pressure (millibars)"] == 0).sum())
    return {
        "description": "Régression simple : chaque variable est prédite par la Température seule, puis par la Pression seule. "
                       "Trois modèles comparés (Linéaire, Polynomiale degré 3, Random Forest) sur un test de 20 %.",
        "protocol": {"test_size": 0.2, "random_state": 42, "n_rows": int(len(df)), "poly_degree": 3,
                     "rf": "RandomForestRegressor(n_estimators=50, min_samples_leaf=50)",
                     "rows_removed": 0, "zero_pressure_rows": n_zero_pressure},
        "predictors": PREDICTORS, "targets": TARGETS, "models": MODELS,
        "results": rows, "curves": curves, "scatter": scatter,
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("Usage: python weather_regression.py weatherHistory.csv latest_weather_report.json")
    out = run(sys.argv[1])
    report_path = Path(sys.argv[2])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["regression_lab"] = out
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    best = pd.DataFrame(out["results"]).sort_values("r2", ascending=False).iloc[0]
    print(f"OK : {len(out['results'])} résultats -> {report_path} | meilleur : {best['model']} "
          f"({best['predictor']} -> {best['target']}, R2={best['r2']})")
