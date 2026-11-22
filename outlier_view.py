"""outlier_view.py : vue « Outliers & bruit » au thème du dashboard (remplace les images matplotlib des boxplots).

Boxplots interactifs (Altair) reconstruits depuis les statistiques du rapport, outliers, bruit et interprétation.
Aucune logique de calcul ici : tout vient de dashboard_data.py ; les graphiques viennent de dashboard_charts.py.
"""
from __future__ import annotations

import streamlit as st

import dashboard_charts as ch
import dashboard_data as dd
from outlier_noise_interpretation import render_outlier_noise_interpretation
from theme import kpi_row, section_title


def render_boxplot(rep: dict, key: str, max_features: int = 15) -> bool:
    """Boxplot au thème du dashboard. Renvoie False s'il n'y a rien à tracer."""
    box, ext = dd.boxplot_frames(rep, normalized=True)
    if box.empty:
        return False
    n = len(box)
    order = st.radio("Trier par", ["% d'outliers", "ordre du fichier"], horizontal=True, key=f"bx_sort_{key}")
    if order == "% d'outliers":
        box = box.sort_values("pct_out", ascending=False)
    if n > max_features:
        top = st.slider("Variables affichées", 5, n, min(max_features, n), key=f"bx_n_{key}")
        box = box.head(top)
    ext = ext[ext["feature"].isin(box["feature"])] if not ext.empty else ext
    st.altair_chart(ch.boxplot(box, ext, True))
    st.caption("Boîte = Q1→Q3, trait blanc = médiane, moustaches = barrières IQR (1,5 × IQR), points rouges = valeurs extrêmes. "
               "Axe normalisé : écart à la médiane en nombre d'IQR, pour comparer des variables d'échelles différentes.")
    return True


def render_outliers_noise(rep: dict, key: str) -> None:
    """Page complète : KPI, boxplots, outliers, bruit, interprétation."""
    feats = dd.outlier_features(rep)
    nz = dd.noise_df(rep)
    if not feats and nz.empty:
        st.info("Ce rapport ne contient ni section outliers ni section bruit.")
        return

    s = (rep.get("outliers", {}) or {}).get("summary") or {}
    mean_iqr = s.get("mean_iqr_percent")
    if mean_iqr is None and feats:
        mean_iqr = sum(f.get("iqr_percent") or 0 for f in feats) / len(feats)
    noisy = nz[nz["verdict"].isin(["moderate", "high"])] if not nz.empty else nz
    kpi_row([
        ("Variables analysées", len(feats), "outliers univariés"),
        ("Outliers IQR moyen", "n/a" if mean_iqr is None else f"{mean_iqr:.2f} %", "part des lignes hors barrières",
         "warn" if (mean_iqr or 0) >= 1 else "good"),
        ("Outliers consensus", "n/a" if s.get("consensus_percent") is None else f"{s['consensus_percent']:.2f} %",
         "6 détecteurs multivariés" if s else "non calculé pour ce rapport", "info"),
        ("Variables bruitées", f"{len(noisy)} / {len(nz)}" if not nz.empty else "n/a", "bruit modéré ou élevé",
         "bad" if (not nz.empty and len(noisy) / len(nz) >= .5) else "warn" if len(noisy) else "good"),
    ])

    render_outlier_noise_interpretation(rep, key)

    section_title("Boxplots", "valeurs extrêmes par variable")
    if not render_boxplot(rep, key):
        st.info("Pas de statistiques de boîte dans ce rapport.")

    c1, c2 = st.columns(2)
    with c1:
        section_title("Outliers selon la méthode", "% de lignes hors limites")
        long = dd.outlier_long(rep)
        if long.empty:
            st.info("Pas de détail par méthode.")
        else:
            st.altair_chart(ch.classical_outliers(long))
    with c2:
        section_title("Bruit par variable", "bruit/signal, plus bas = plus propre")
        if nz.dropna(subset=["noise_to_signal"]).empty:
            st.info("Pas de données de bruit.")
        else:
            st.altair_chart(ch.noise_ratio(nz))

    notes = [t for sec in ("outliers", "noise") for t in ((rep.get(sec, {}) or {}).get("insights") or [])]
    if notes:
        with st.expander("Constats détaillés du rapport (anglais)"):
            st.markdown("\n".join(f"- {t.replace('$', chr(92) + '$')}" for t in notes))
