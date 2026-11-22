"""electrical_load_view.py : vues spécifiques au jeu ElectricalLoad dans le tableau de bord.

Le rapport `eda_output_electricalload/report.json` a un schéma différent des rapports
Phase I (sections quality / trends / outliers / noise / correlation / distribution /
preprocessing au lieu de data_quality + model_phase). Ce module affiche :
  * le détail outliers & bruit (4 détecteurs, NSR/SNR) ;
  * la corrélation MA_X-4 ↔ DEMAND et les jours à fort écart ;
  * l'évaluation des régresseurs lue depuis evaluation_electricalload/metrics_summary.csv.

Aucun calcul de modèle ici : tout provient des fichiers générés par electrical_load_eda.py
et eval_electricalload.py.
"""
from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import dashboard_charts as ch
import dashboard_data as dd
from outlier_noise_interpretation import render_outlier_noise_interpretation
from theme import attention_list, kpi_row, section_title

ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "evaluation_electricalload"
EVAL_IMAGES = [
    ("01_r2_comparison.png", "R² : test vs validation croisée"),
    ("02_rmse_comparison.png", "RMSE : test vs validation croisée"),
    ("03_r2_and_rmse_overview.png", "Vue d'ensemble R² / RMSE"),
    ("04_actual_vs_predicted.png", "Valeurs réelles vs prédites"),
    ("05_residuals_distribution.png", "Distribution des résidus"),
    ("06_scoreboard_heatmap.png", "Classement des algorithmes"),
]
COLORS = {"R² Test": "#4F46E5", "R² CV": "#06B6D4", "RMSE Test": "#F59E0B", "RMSE CV": "#EF4444"}


def _stretch(fn, *args, **kwargs):
    try:
        return fn(*args, width="stretch", **kwargs)
    except TypeError:
        return fn(*args, use_container_width=True, **kwargs)


def _metrics_df() -> pd.DataFrame:
    """Tableau de bord de eval_electricalload.py (colonnes simplifiées pour Altair)."""
    path = EVAL_DIR / "metrics_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.rename(columns={"Algorithm": "Algorithme", "R² Test": "R2 Test", "CV R²": "R2 CV",
                            "RMSE Test": "RMSE Test", "CV RMSE": "RMSE CV"})
    for col in ("R2 Test", "R2 CV", "RMSE Test", "RMSE CV"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=[c for c in ("R2 Test", "R2 CV") if c in df])


def render_electrical_evaluation(rep: dict, key: str) -> None:
    """KPI, comparaison R²/RMSE, tableau et graphiques d'évaluation de la charge électrique."""
    metrics = _metrics_df()
    if metrics.empty:
        st.info("Évaluation introuvable. Relancez `py -3 eval_electricalload.py` pour générer "
                "`evaluation_electricalload/metrics_summary.csv` et ses graphiques.")
    else:
        best_r2 = metrics.loc[metrics["R2 Test"].idxmax()]
        best_rmse = metrics.loc[metrics["RMSE Test"].idxmin()]
        meta = rep.get("meta", {}) or {}
        kpi_row([
            ("Meilleur R² test", f"{best_r2['R2 Test']:.3f}", str(best_r2["Algorithme"]), "good"),
            ("Meilleur RMSE test", f"{best_rmse['RMSE Test']:.1f} MW", str(best_rmse["Algorithme"]), "good"),
            ("R² validation croisée", f"{best_r2['R2 CV']:.3f}", "meilleur modèle en test", "info"),
            ("Jeu de test", str(meta.get("test_rows", "—")), f"{meta.get('n_rows', '—')} relevés au total"),
        ])

        gap = best_r2["R2 Test"] - best_r2["R2 CV"]
        pts = [("good", f"{best_r2['Algorithme']} domine en test (R² = {best_r2['R2 Test']:.3f})",
                f"RMSE de {best_r2['RMSE Test']:.1f} MW sur le jeu de test, contre "
                f"{metrics['RMSE Test'].max():.1f} MW pour le moins bon modèle.")]
        if abs(gap) >= 0.05:
            pts.append(("warn" if gap < 0 else "info",
                        f"Écart test / validation croisée de {abs(gap):.3f} pour le meilleur modèle",
                        "Un R² test nettement inférieur au R² CV peut indiquer une série de test courte "
                        f"({meta.get('test_rows', '?')} heures) ou une dérive saisonnière."))
        attention_list(pts)

        section_title("Comparaison des régresseurs", "R² : plus haut = meilleur · RMSE en MW : plus bas = meilleur")
        for value_vars, title in (([c for c in ("R2 Test", "R2 CV") if c in metrics], "R²"),
                                  ([c for c in ("RMSE Test", "RMSE CV") if c in metrics], "RMSE (MW)")):
            if not value_vars:
                continue
            long = metrics.melt(id_vars="Algorithme", value_vars=value_vars,
                                var_name="Mesure", value_name="Valeur")
            long["Mesure"] = long["Mesure"].replace({"R2 Test": "R² Test", "R2 CV": "R² CV"})
            bars = alt.Chart(long).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("Algorithme:N", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Valeur:Q", title=title),
                xOffset="Mesure:N",
                color=alt.Color("Mesure:N", scale=alt.Scale(
                    domain=list(COLORS), range=list(COLORS.values())),
                    legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["Algorithme", "Mesure", alt.Tooltip("Valeur:Q", format=".4f")])
            st.altair_chart(bars.properties(width="container", height=280))

        section_title("Tableau des métriques", "Test = jeu de 168 heures · CV = validation croisée 5 folds")
        show = metrics.rename(columns={"R2 Test": "R² test", "R2 CV": "R² CV",
                                       "RMSE Test": "RMSE test", "RMSE CV": "RMSE CV"})
        _stretch(st.dataframe, show.round(4), hide_index=True)

    section_title("Graphiques générés par eval_electricalload.py", "comparaisons, courbe réelle/prédite et résidus")
    cols = st.columns(2)
    shown = 0
    for i, (name, caption) in enumerate(EVAL_IMAGES):
        path = EVAL_DIR / name
        if not path.exists():
            continue
        with cols[i % 2]:
            _stretch(st.image, str(path), caption=caption)
        shown += 1
    if not shown:
        st.info("Aucun graphique d'évaluation présent dans `evaluation_electricalload/`.")


def render_electrical_outliers_noise(rep: dict, key: str) -> None:
    """Vue outliers & bruit adaptée aux 4 détecteurs et à l'estimateur NSR du rapport ElectricalLoad."""
    o = rep.get("outliers", {}) or {}
    det = dd.outlier_detectors_df(rep)
    cons = o.get("consensus_outliers") or {}
    unanimous = (o.get("unanimous_outliers") or {}).get("count", 0)
    nz = dd.noise_df(rep)
    noisy = nz[nz["verdict"].isin(["moderate", "high"])] if not nz.empty else nz

    kpi_row([
        ("Détecteurs", len(det), "contamination 5 %", "info"),
        ("Outliers consensus", "n/a" if cons.get("percent") is None else f"{cons['percent']:.2f} %",
         f"{cons.get('count', 0)} lignes signalées par ≥ 2 détecteurs",
         "warn" if (cons.get("percent") or 0) >= 5 else "good"),
        ("Accord unanime", unanimous, "lignes signalées par les 4 détecteurs"),
        ("Variables bruitées", f"{len(noisy)} / {len(nz)}" if not nz.empty else "n/a",
         "bruit modéré ou élevé (NSR)", "good" if noisy.empty else "warn"),
    ])

    render_outlier_noise_interpretation(rep, key)

    c1, c2 = st.columns(2)
    with c1:
        section_title("Détecteurs multivariés", "% d'échantillons signalés")
        if det.empty:
            st.info("Aucun détail de détecteur dans ce rapport.")
        else:
            st.altair_chart(ch.detectors(det))
    with c2:
        section_title("Bruit par variable", "NSR (bruit/signal), plus bas = plus propre")
        if nz.dropna(subset=["noise_to_signal"]).empty:
            st.info("Pas de données de bruit.")
        else:
            st.altair_chart(ch.noise_ratio(nz))

    notes = [t for sec in ("outliers", "noise") for t in ((rep.get(sec) or {}).get("insights") or [])]
    if notes:
        with st.expander("Constats détaillés du rapport (anglais)"):
            st.markdown("\n".join(f"- {t}" for t in notes))


def render_electrical_correlation(rep: dict, key: str) -> None:
    """Corrélation du prédicteur décalé MA_X-4 avec DEMAND et jours à fort écart."""
    corr = rep.get("correlation", {}) or {}
    meta = rep.get("meta", {}) or {}
    gaps = dd.electrical_gap_days_df(rep)
    r = corr.get("pearson_ma4_demand")
    threshold = corr.get("gap_threshold_mw")

    kpi_row([
        ("MA_X-4 ↔ DEMAND", "n/a" if r is None else f"{r:.4f}", "corrélation de Pearson", "good"),
        ("Seuil d'écart", "n/a" if threshold is None else f"{threshold:.1f} MW", "|DEMAND − MA_X-4|", "info"),
        ("Jours atypiques", len(gaps), "plus grands écarts du rapport"),
        ("Cible", str(meta.get("target_column", "DEMAND")), str(meta.get("n_rows", "—")) + " relevés"),
    ])

    pts = []
    if r is not None:
        pts.append(("good", f"Lien quasi linéaire entre MA_X-4 et DEMAND (r = {r:.4f})",
                    "La moyenne mobile décalée suit étroitement la charge : elle reste le prédicteur principal, "
                    "à condition de préserver l'ordre chronologique train → test."))
    if len(gaps):
        threshold_txt = "n/a" if threshold is None else f"{threshold:.1f} MW"
        pts.append(("warn", f"{len(gaps)} jours présentent un écart supérieur à {threshold_txt}",
                    f"Le pic atteint {gaps.iloc[0]['gap']:.0f} MW le {gaps.iloc[0]['date']} : "
                    "événements spéciaux, jours fériés ou arrêts de production à vérifier."))
    if pts:
        attention_list(pts)

    if not gaps.empty:
        section_title("Plus grands écarts |DEMAND − MA_X-4|", "par jour, en MW")
        bars = alt.Chart(gaps).mark_bar(cornerRadiusEnd=4, color="#4F46E5").encode(
            x=alt.X("gap:Q", title="écart (MW)"),
            y=alt.Y("date:N", sort="-x", title=None),
            tooltip=[alt.Tooltip("date:N", title="jour"), alt.Tooltip("gap:Q", format=".1f", title="écart (MW)"),
                     alt.Tooltip("seuil:Q", format=".1f", title="seuil")])
        st.altair_chart(bars.properties(width="container", height=max(220, 28 * len(gaps) + 40)))

    notes = corr.get("insights") or []
    if notes:
        with st.expander("Constats détaillés du rapport (anglais)"):
            st.markdown("\n".join(f"- {t}" for t in notes))
