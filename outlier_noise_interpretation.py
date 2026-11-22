"""outlier_noise_interpretation.py : guide de lecture et synthèse des outliers / du bruit.

Partagé par la vue générique (`outlier_view.py`) et la vue ElectricalLoad
(`electrical_load_view.py`) : il transforme les chiffres du rapport en phrases
compréhensibles et en décisions concrètes, sans recalculer les métriques.
"""
from __future__ import annotations

import streamlit as st

import dashboard_data as dd
from theme import section_title


def _stretch(fn, *args, **kwargs):
    try:
        return fn(*args, width="stretch", **kwargs)
    except TypeError:
        return fn(*args, use_container_width=True, **kwargs)


GUIDE = [
    ("Outlier", "Une observation rare, éloignée du reste des données. Ce n'est **pas** automatiquement une erreur."),
    ("Barrières IQR", "Hors de `[Q1 − 1,5 × IQR ; Q3 + 1,5 × IQR]`. Sensible aux distributions à queue lourde."),
    ("Z-score / Z modifié", "Distance à la moyenne en écarts-types. Le Z modifié (MAD) est plus robuste aux extrêmes."),
    ("Consensus", "Ligne signalée par au moins 2 détecteurs. Plus le nombre d'accords est élevé, plus l'anomalie est fiable."),
    ("Bruit (NSR)", "Ratio bruit/signal `σ_bruit / σ_signal`. Plus il est bas, plus la série est propre."),
    ("SNR (dB)", "Rapport signal/bruit en décibels. Plus il est haut, plus le signal domine le bruit."),
    ("Seuils NSR", "< 0,05 : propre · 0,05–0,15 : faible · 0,15–0,35 : modéré · > 0,35 : élevé."),
]


def render_outlier_noise_interpretation(rep: dict, key: str) -> None:
    """Synthèse + guide de lecture + interprétation par variable + décisions."""
    info = dd.outlier_noise_interpretation(rep)

    section_title("Interprétation des outliers et du bruit", "synthèse calculée à partir du rapport")
    st.info(info["headline"])
    for fact in info["facts"]:
        st.markdown(f"- {fact}")

    with st.expander("📖 Comment lire ces indicateurs ?"):
        for term, definition in GUIDE:
            st.markdown(f"**{term}** — {definition}")
        if dd.is_electrical(rep):
            st.caption(
                "Pour ElectricalLoad, le bruit est estimé par les secondes différences de la série horaire. "
                "Les anomalies sont détectées dans un espace multivarié : il n'y a donc pas de taux d'outliers "
                "par colonne, mais un consensus global sur les lignes."
            )

    section_title("Interprétation par variable", "ce que les chiffres disent, et quoi faire")
    if dd.is_electrical(rep):
        table = dd.electrical_noise_interpretation_df(rep)
    else:
        table = dd.interpret_variables(rep)
    if table.empty:
        st.info("Aucune variable à interpréter dans ce rapport.")
    else:
        _stretch(st.dataframe, table, hide_index=True)

    section_title("Décision recommandée", "actions déduites des indicateurs")
    for decision in info["decisions"]:
        st.markdown(f"- ✅ {decision}")
