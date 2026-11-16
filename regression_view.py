"""regression_view.py : onglet « Éval. Modèles » du dataset météo (régressions simples), au thème du dashboard.

Lit la section « regression_lab » du rapport (calculée par weather_regression.py). Aucun calcul ici.
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from theme import attention_list, kpi_row, section_title

INFO, GOOD, WARN, BAD, MUTED = "#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#64748B"
COLORS = {"Linéaire": INFO, "Polynomiale": GOOD, "Random Forest": WARN}


def _fit(c, h):
    return c.properties(width="container", height=h)


def _short(s: str) -> str:
    return s.split(" (")[0]


def render_regression_lab(rep: dict, key: str) -> None:
    lab = rep.get("regression_lab")
    if not lab:
        st.info("Ce rapport ne contient pas d'évaluation par régression. "
                "Lancez `python weather_regression.py weatherHistory.csv latest_weather_report.json`.")
        return
    res = pd.DataFrame(lab["results"])
    models, preds, targets = lab["models"], lab["predictors"], lab["targets"]
    proto = lab["protocol"]

    # ── KPI + lecture rapide ──
    mean_r2 = res.groupby("model")["r2"].mean().sort_values(ascending=False)
    best = res.sort_values("r2", ascending=False).iloc[0]
    winners = res.loc[res.groupby(["predictor", "target"])["r2"].idxmax()]["model"].value_counts()
    kpi_row([
        ("Régressions", f"{len(res) // len(models)}", f"{len(preds)} prédicteurs × {len(targets)} cibles", "info"),
        ("Meilleur modèle (R² moyen)", mean_r2.index[0], f"R² moyen {mean_r2.iloc[0]:.3f}", "good"),
        ("Meilleur R²", f"{best['r2']:.3f}", f"{_short(best['predictor'])} → {_short(best['target'])}", "good"),
        ("Jeu de test", f"{int(proto['test_size'] * 100)} %", f"{proto['n_rows']:,} lignes, aucune retirée".replace(",", " ")),
    ])
    pts = [("good", f"{mean_r2.index[0]} gagne le plus souvent ({int(winners.get(mean_r2.index[0], 0))} régressions sur "
                    f"{len(res) // len(models)})", " · ".join(f"{m} : R² moyen {v:.3f}" for m, v in mean_r2.items()))]
    weak = res.groupby(["predictor", "target"])["r2"].max().sort_values().head(1)
    (wp, wt), wv = weak.index[0], float(weak.iloc[0])
    pts.append(("warn", f"Cas le plus faible : {_short(wp)} → {_short(wt)}",
                f"Même le meilleur modèle n'atteint que R² = {wv:.3f} : la variable explicative ne suffit pas seule."))
    lin = res[res.model == "Linéaire"].set_index(["predictor", "target"])["r2"]
    rf = res[res.model == "Random Forest"].set_index(["predictor", "target"])["r2"]
    gain = (rf - lin).sort_values(ascending=False)
    if gain.iloc[0] > 0.05:
        (gp, gt) = gain.index[0]
        pts.append(("info", f"Non-linéarité : {_short(gp)} → {_short(gt)}",
                    f"La Random Forest gagne {gain.iloc[0]:.2f} point de R² sur la droite (linéaire {lin[gain.index[0]]:.2f} "
                    f"→ forêt {rf[gain.index[0]]:.2f})."))
    if proto.get("zero_pressure_rows"):
        pts.append(("warn", f"{proto['zero_pressure_rows']} pressions égales à 0 conservées",
                    "Mesures impossibles (capteur bloqué) laissées dans les données, comme dans le script d'origine : "
                    "elles pèsent sur les régressions avec la Pression."))
    attention_list(pts)

    # ── Une régression : nuage + courbes, barres de R² ──
    section_title("Une régression en détail", "choisissez le prédicteur et la variable à prédire")
    c1, c2 = st.columns([1, 1])
    with c1:
        pred = st.radio("Prédicteur (utilisé seul)", preds, horizontal=True, key=f"rg_pred_{key}")
    with c2:
        target = st.selectbox("Variable prédite", targets, key=f"rg_tgt_{key}")
    k = f"{pred}|{target}"
    sc, cv = lab["scatter"][k], lab["curves"][k]
    sub = res[(res.predictor == pred) & (res.target == target)].copy()
    best_m = sub.sort_values("r2", ascending=False).iloc[0]["model"]

    left, right = st.columns([3, 2], gap="medium")
    with left:
        pts_df = pd.DataFrame({"x": sc["x"], "y": sc["y"]})
        cur = pd.DataFrame([{"x": x, "y": y, "modèle": m} for m in models for x, y in zip(cv["x"], cv[m])])
        dots = alt.Chart(pts_df).mark_circle(size=14, opacity=0.25, color=MUTED).encode(
            x=alt.X("x:Q", title=pred, scale=alt.Scale(zero=False)), y=alt.Y("y:Q", title=target, scale=alt.Scale(zero=False)))
        lines = alt.Chart(cur).mark_line(strokeWidth=2.5).encode(
            x="x:Q", y="y:Q", color=alt.Color("modèle:N", scale=alt.Scale(domain=models, range=[COLORS[m] for m in models]),
                                               legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["modèle:N", alt.Tooltip("x:Q", format=".2f"), alt.Tooltip("y:Q", format=".2f")])
        st.altair_chart(_fit(dots + lines, 340))
        st.caption(f"Nuage : {len(pts_df)} points tirés au hasard parmi {proto['n_rows']:,}; ".replace(",", " ")
                   + "les courbes montrent ce que chaque modèle prédit sur toute la plage du prédicteur.")
    with right:
        bars = alt.Chart(sub).mark_bar(cornerRadiusEnd=5, size=34).encode(
            x=alt.X("model:N", sort=models, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("r2:Q", title="R² (1 = parfait)", scale=alt.Scale(domain=[min(0, float(sub["r2"].min())), 1.05])),
            color=alt.Color("model:N", scale=alt.Scale(domain=models, range=[COLORS[m] for m in models]), legend=None),
            tooltip=["model:N", alt.Tooltip("r2:Q", format=".4f", title="R²"), alt.Tooltip("rmse:Q", format=".3f", title="RMSE")])
        txt = bars.mark_text(dy=-8, fontSize=12, fontWeight="bold").encode(text=alt.Text("r2:Q", format=".3f"))
        st.altair_chart(_fit(bars + txt, 340))
        st.markdown(f"**Meilleur : {best_m}** · RMSE "
                    + " · ".join(f"{r['model']} {r['rmse']:.3g}" for _, r in sub.iterrows()))

    # ── Comparaison globale ──
    section_title("Comparaison globale des modèles", "R² de chaque régression, un panneau par prédicteur")
    g = st.columns(len(preds), gap="medium")
    for col, p in zip(g, preds):
        with col:
            d = res[res.predictor == p].assign(cible=lambda x: x["target"].map(_short))
            ch = alt.Chart(d).mark_bar(cornerRadiusEnd=3).encode(
                x=alt.X("cible:N", sort=[_short(t) for t in targets], title=None, axis=alt.Axis(labelAngle=-40, labelOverlap=False, labelLimit=140)),
                y=alt.Y("r2:Q", title="R²", scale=alt.Scale(domain=[min(0, float(d["r2"].min())), 1.05])),
                xOffset=alt.XOffset("model:N", sort=models),
                color=alt.Color("model:N", scale=alt.Scale(domain=models, range=[COLORS[m] for m in models]),
                                legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["cible:N", "model:N", alt.Tooltip("r2:Q", format=".4f"), alt.Tooltip("rmse:Q", format=".3f")])
            st.markdown(f"**Prédire avec {_short(p)} seule**")
            st.altair_chart(_fit(ch, 340))

    section_title("Tous les résultats")
    table = res.pivot_table(index=["predictor", "target"], columns="model", values=["r2", "rmse"]).round(4)
    table.columns = [f"{'R²' if a == 'r2' else 'RMSE'} · {b}" for a, b in table.columns]
    table = table.reset_index().rename(columns={"predictor": "Prédicteur", "target": "Cible"})
    try:
        st.dataframe(table, hide_index=True, width="stretch")
    except TypeError:
        st.dataframe(table, hide_index=True, use_container_width=True)
    with st.expander("Protocole"):
        st.markdown(f"- Séparation train/test : {int(proto['test_size'] * 100)} % de test, `random_state={proto['random_state']}`\n"
                    f"- Polynomiale : degré {proto['poly_degree']} · Forêt : `{proto['rf']}`\n"
                    f"- Lignes supprimées : {proto['rows_removed']} (dataset inchangé)")
