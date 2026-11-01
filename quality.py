"""quality.py : règles de qualité des données PARTAGÉES par le Model (weather_model.py).

Une seule définition des plages physiques, des seuils de bruit et du nettoyage, réutilisable par toute
autre vue ou script. Ne dépend que de numpy et pandas (aucun Streamlit).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---- seuils de détection du bruit
SPIKE_K = 4                      # pic = plus de 4 sigmas robustes de la médiane des 4 voisins
SPIKE_PCT_FLAG = 1.0             # une variable est « bruitée » si plus de 1 % de ses points sont des pics
SPIKE_MIN_FRAC = 0.5             # ... et si l'écart dépasse 50 % de l'étalement normal de la variable
STUCK_RUN = 24                   # >= 24 valeurs identiques consécutives = capteur figé
NSR_LEVELS = (0.05, 0.15, 0.35)  # négligeable / faible / modéré / élevé
CIRCULAR_KEYS = ("bearing",)     # variables circulaires : les tests linéaires n'ont pas de sens

# (min, max) physiquement possibles, repérés par le nom de colonne
PHYSICAL_RANGES = {
    "temperature": (-90, 60), "humidity": (0, 1), "pressure": (850, 1090),
    "wind speed": (0, 150), "wind bearing": (0, 360), "visibility": (0, 100),
}


def robust_sigma(x) -> float:
    """Écart-type estimé par la MAD : insensible aux valeurs extrêmes."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else 0.0


def physical_range(col: str, s: pd.Series):
    name = str(col).lower()
    for key, rng in PHYSICAL_RANGES.items():
        if key in name:
            if key == "humidity" and s.quantile(.99) > 1.5:      # humidité stockée en %
                return (0, 100)
            return rng
    return None


def is_circular(col: str) -> bool:
    return any(k in str(col).lower() for k in CIRCULAR_KEYS)


def mask_impossible(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Copie de df où les mesures physiquement impossibles (ex. pression = 0) deviennent NaN.
    Retourne (df nettoyé, {colonne: nombre de valeurs masquées}). Le df d'origine n'est pas modifié."""
    out = df.copy()
    counts: dict[str, int] = {}
    for c in out.select_dtypes("number").columns:
        s = out[c].dropna()
        if s.empty or s.nunique() <= 1:
            continue
        rng = physical_range(c, s)
        if rng:
            bad = (out[c] < rng[0]) | (out[c] > rng[1])
            if bad.any():
                out.loc[bad, c] = np.nan
                counts[c] = int(bad.sum())
    return out, counts
