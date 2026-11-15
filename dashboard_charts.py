"""dashboard_charts.py : graphiques Altair au thème du dashboard (teal, mêmes couleurs que theme.py).

Chaque fonction reçoit un DataFrame produit par dashboard_data.py et renvoie un graphique ;
l'affichage (st.altair_chart) se fait dans project_dashboard.py.
"""
from __future__ import annotations

import altair as alt
import pandas as pd

TEAL, TEAL_2 = "#4F46E5", "#06B6D4"
GOOD, WARN, BAD, INFO, MUTED = "#10B981", "#F59E0B", "#EF4444", "#3B82F6", "#64748B"
INK = "#0F172A"
LAYER_COLORS = [TEAL, TEAL_2, INFO, WARN, MUTED]


def fit(c, height: int = 280):
    """Largeur adaptative + hauteur fixée."""
    return c.properties(width="container", height=height)


def _h(n: int, per: int = 28, lo: int = 160) -> int:
    return max(lo, per * n + 40)


# ───────────────────────── valeurs extrêmes ─────────────────────────
def boxplot(box: pd.DataFrame, extremes: pd.DataFrame, normalized: bool = True):
    """Boîtes à moustaches construites depuis les statistiques du rapport (Q1, Q3, barrières IQR, médiane)."""
    order = list(box["feature"])
    y = alt.Y("feature:N", sort=order, title=None)
    tip = [alt.Tooltip("feature:N", title="variable"), alt.Tooltip("q1:Q", format=".3f"),
           alt.Tooltip("median:Q", format=".3f", title="médiane"), alt.Tooltip("q3:Q", format=".3f"),
           alt.Tooltip("vmin:Q", format=".3f", title="min"), alt.Tooltip("vmax:Q", format=".3f", title="max"),
           alt.Tooltip("n_out:Q", title="outliers IQR"), alt.Tooltip("pct_out:Q", format=".2f", title="% outliers")]
    base = alt.Chart(box).encode(y=y)
    xtitle = "écart à la médiane (en nombre d'IQR)" if normalized else "valeur"
    # l'axe suit les boîtes (pas les extrêmes) ; les extrêmes hors cadre sont collés au bord (triangle) avec leur vraie valeur en infobulle
    lo = min(-3.0, float(box["x_wlo"].min()) - 0.5)
    hi = max(3.0, float(box["x_whi"].max()) + 0.5)
    if not extremes.empty:
        extremes = extremes.copy()
        extremes["hors_cadre"] = (extremes["x"] > hi) | (extremes["x"] < lo)
        extremes["x_aff"] = extremes["x"].clip(lo, hi)
    dom = alt.Scale(domain=[lo - 0.4, hi + 0.8])
    whisker = base.mark_rule(color=MUTED, strokeWidth=2).encode(x=alt.X("x_wlo:Q", title=xtitle, scale=dom), x2=alt.X2("x_whi"))
    body = base.mark_bar(opacity=0.9, size=16, cornerRadius=3).encode(
        x=alt.X("x_q1:Q", scale=dom), x2=alt.X2("x_q3"), tooltip=tip,
        color=alt.Color("niveau:N", scale=alt.Scale(domain=["faible", "modéré", "élevé"], range=[TEAL_2, WARN, BAD]),
                        legend=alt.Legend(orient="bottom", title="outliers IQR")) if "niveau" in box else alt.value(TEAL_2))
    med = base.mark_tick(color="white", thickness=3, size=16).encode(x=alt.X("x_med:Q", scale=dom))
    layers = whisker + body + med
    if not extremes.empty:
        pts = alt.Chart(extremes).mark_point(color=BAD, filled=True, size=70, opacity=0.9).encode(
            x=alt.X("x_aff:Q", scale=dom), y=y,
            shape=alt.Shape("hors_cadre:N", scale=alt.Scale(domain=[False, True], range=["circle", "triangle-right"]), legend=None),
            tooltip=[alt.Tooltip("feature:N", title="variable"), alt.Tooltip("valeur:Q", format=".3f", title="valeur extrême (réelle)"),
                     alt.Tooltip("n_out:Q", title="outliers IQR")])
        layers = layers + pts
    return fit(layers, _h(len(box), 30))


def classical_outliers(long: pd.DataFrame):
    order = list(dict.fromkeys(long["feature"]))
    return fit(alt.Chart(long).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("feature:N", sort=order, title=None), x=alt.X("pourcent:Q", title="% de lignes hors limites"),
        yOffset="methode:N",
        color=alt.Color("methode:N", scale=alt.Scale(range=LAYER_COLORS), legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["feature", "methode", alt.Tooltip("pourcent:Q", format=".2f")]), _h(len(order), 44, 200))


def detectors(df: pd.DataFrame):
    return fit(alt.Chart(df).mark_bar(cornerRadiusEnd=5, color=TEAL).encode(
        x=alt.X("pourcent:Q", title="% d'échantillons signalés"), y=alt.Y("detecteur:N", sort="-x", title=None),
        tooltip=["detecteur", "n", alt.Tooltip("pourcent:Q", format=".2f")]), _h(len(df), 34, 200))


def votes(df: pd.DataFrame):
    return fit(alt.Chart(df).mark_bar(cornerRadiusEnd=5, color=TEAL_2).encode(
        x=alt.X("votes:O", title="nombre de détecteurs qui signalent la ligne", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("echantillons:Q", title="échantillons", scale=alt.Scale(type="sqrt")),
        tooltip=["votes", "echantillons"]), 240)


def agreement(df: pd.DataFrame):
    order = list(dict.fromkeys(df["a"]))
    base = alt.Chart(df).encode(x=alt.X("a:N", sort=order, title=None, axis=alt.Axis(labelAngle=-35)),
                                y=alt.Y("b:N", sort=order, title=None))
    heat = base.mark_rect(cornerRadius=4).encode(
        color=alt.Color("jaccard:Q", scale=alt.Scale(domain=[0, 1], range=["#eaf4f4", TEAL]), legend=None),
        tooltip=["a", "b", alt.Tooltip("jaccard:Q", format=".2f")])
    txt = base.mark_text(fontSize=11).encode(
        text=alt.Text("jaccard:Q", format=".2f"),
        color=alt.condition("datum.jaccard > 0.5", alt.value("white"), alt.value(INK)))
    return fit(heat + txt, 300)


def feature_outliers(df: pd.DataFrame):
    return fit(alt.Chart(df).mark_bar(cornerRadiusEnd=4, color=INFO).encode(
        x=alt.X("pourcent:Q", title="% signalé par Isolation Forest"), y=alt.Y("feature:N", sort="-x", title=None),
        tooltip=["feature", "n", alt.Tooltip("pourcent:Q", format=".2f")]), _h(len(df)))


# ───────────────────────── bruit ─────────────────────────
VERDICT_SCALE = alt.Scale(domain=["negligible", "low", "moderate", "high"], range=[GOOD, TEAL_2, WARN, BAD])


def noise_ratio(nz: pd.DataFrame):
    nz = nz.dropna(subset=["noise_to_signal"]).sort_values("noise_to_signal", ascending=False).head(20)
    return fit(alt.Chart(nz).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("noise_to_signal:Q", title="bruit / signal"), y=alt.Y("feature:N", sort="-x", title=None),
        color=alt.Color("verdict:N", scale=VERDICT_SCALE, legend=alt.Legend(orient="bottom", title="niveau")),
        tooltip=["feature", alt.Tooltip("noise_to_signal:Q", format=".3f"), alt.Tooltip("snr_db:Q", format=".1f"), "verdict"]),
        _h(len(nz)))


def noise_snr(nz: pd.DataFrame):
    nz = nz.dropna(subset=["snr_db"]).sort_values("snr_db")
    return fit(alt.Chart(nz).mark_bar(cornerRadiusEnd=4, color=TEAL).encode(
        x=alt.X("snr_db:Q", title="rapport signal/bruit (dB), plus haut = plus propre"),
        y=alt.Y("feature:N", sort="x", title=None),
        tooltip=["feature", alt.Tooltip("snr_db:Q", format=".1f")]), _h(len(nz)))


def noise_scatter(nz: pd.DataFrame):
    d = nz.dropna(subset=["noise_sigma", "signal_sigma"])
    d = d[(d["noise_sigma"] > 0) & (d["signal_sigma"] > 0)]
    pts = alt.Chart(d).mark_circle(size=110, opacity=0.85).encode(
        x=alt.X("signal_sigma:Q", scale=alt.Scale(type="log"), title="écart-type du signal (échelle log)"),
        y=alt.Y("noise_sigma:Q", scale=alt.Scale(type="log"), title="écart-type du bruit (échelle log)"),
        color=alt.Color("verdict:N", scale=VERDICT_SCALE, legend=alt.Legend(orient="bottom", title="niveau")),
        tooltip=["feature", alt.Tooltip("signal_sigma:Q", format=".3g"), alt.Tooltip("noise_sigma:Q", format=".3g"), "verdict"])
    return fit(pts, 320)


def spikes(nz: pd.DataFrame):
    d = nz.dropna(subset=["spike_pct"]).sort_values("spike_pct", ascending=False).head(20)
    return fit(alt.Chart(d).mark_bar(cornerRadiusEnd=4, color=WARN).encode(
        x=alt.X("spike_pct:Q", title="% de pics isolés"), y=alt.Y("feature:N", sort="-x", title=None),
        tooltip=["feature", alt.Tooltip("spike_pct:Q", format=".2f"), "spikes"]), _h(len(d)))


# ───────────────────────── corrélations ─────────────────────────
def corr_heatmap(long: pd.DataFrame):
    order = list(dict.fromkeys(long["a"]))
    n = len(order)
    base = alt.Chart(long).encode(x=alt.X("a:N", sort=order, title=None, axis=alt.Axis(labelAngle=-45)),
                                  y=alt.Y("b:N", sort=order, title=None))
    heat = base.mark_rect().encode(
        color=alt.Color("value:Q", scale=alt.Scale(domain=[-1, 0, 1], range=[INFO, "#f2f5f6", BAD]),
                        legend=alt.Legend(title="r", orient="right")),
        tooltip=["a", "b", alt.Tooltip("value:Q", format=".3f")])
    layers = heat
    if n <= 14:
        layers = heat + base.mark_text(fontSize=10).encode(
            text=alt.Text("value:Q", format=".2f"),
            color=alt.condition("abs(datum.value) > 0.5", alt.value("white"), alt.value(INK)))
    return fit(layers, max(260, min(1100, 34 * n + 60)))


def top_pairs(cp: pd.DataFrame):
    return fit(alt.Chart(cp).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("pearson:Q", scale=alt.Scale(domain=[-1, 1]), title="corrélation de Pearson"),
        y=alt.Y("paire:N", sort=alt.EncodingSortField("pearson", order="descending"), title=None),
        color=alt.Color("pearson:Q", scale=alt.Scale(domain=[-1, 0, 1], range=[INFO, "#d9dedf", BAD]), legend=None),
        tooltip=["paire", alt.Tooltip("pearson:Q", format=".3f"), alt.Tooltip("spearman:Q", format=".3f"), "strength"]),
        _h(len(cp), 30))


def target_corr(tc: pd.DataFrame):
    d = tc.sort_values("abs_correlation", ascending=False).head(15)
    order = list(d["feature"])
    return fit(alt.Chart(d).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("correlation:Q", scale=alt.Scale(domain=[-1, 1]), title="corrélation avec la cible"),
        y=alt.Y("feature:N", sort=order, title=None),
        color=alt.Color("correlation:Q", scale=alt.Scale(domain=[-1, 0, 1], range=[INFO, "#d9dedf", BAD]), legend=None),
        tooltip=["feature", alt.Tooltip("correlation:Q", format=".3f")]), _h(len(d)))


# ───────────────────────── qualité & interprétation ─────────────────────────
def indicators(ind: pd.DataFrame):
    return fit(alt.Chart(ind).mark_bar(cornerRadiusEnd=6).encode(
        x=alt.X("issue_percent:Q", title="% du dataset concerné"), y=alt.Y("indicator:N", sort="-x", title=None),
        color=alt.Color("status:N", scale=alt.Scale(domain=["good", "warning", "critical"], range=[GOOD, WARN, BAD]),
                        legend=alt.Legend(orient="bottom", title="statut")),
        tooltip=["indicator", "issue_percent", "status", "detail"]), _h(len(ind), 36))


def contradictions(df: pd.DataFrame):
    return fit(alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("pourcent:Q", title="% des lignes"), y=alt.Y("regle:N", sort="-x", title=None),
        color=alt.Color("severite:N", scale=alt.Scale(domain=["critical", "warning", "info"], range=[BAD, WARN, INFO]),
                        legend=alt.Legend(orient="bottom", title="sévérité")),
        tooltip=["regle", "n", alt.Tooltip("pourcent:Q", format=".2f"), "severite"]), _h(len(df)))


def missing(df: pd.DataFrame):
    d = df.assign(total=df["manquants"] + df["zeros_caches"]).sort_values("total", ascending=False).head(15)
    long = d.melt(id_vars="feature", value_vars=["manquants", "zeros_caches"], var_name="type", value_name="n")
    return fit(alt.Chart(long).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("n:Q", title="cellules"), y=alt.Y("feature:N", sort=list(d["feature"]), title=None),
        color=alt.Color("type:N", scale=alt.Scale(domain=["manquants", "zeros_caches"], range=[WARN, BAD]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["feature", "type", "n"]), _h(len(d)))


def priorities(rec: pd.DataFrame):
    d = rec.groupby("priority", as_index=False).size().rename(columns={"size": "n"})
    return fit(alt.Chart(d).mark_arc(innerRadius=55).encode(
        theta="n:Q",
        color=alt.Color("priority:N", scale=alt.Scale(domain=["high", "medium", "low"], range=[BAD, WARN, INFO]),
                        legend=alt.Legend(orient="bottom", title="priorité")),
        tooltip=["priority", "n"]), 240)


def plan_status(plan: pd.DataFrame):
    d = plan.groupby("status", as_index=False).size().rename(columns={"size": "n"})
    return fit(alt.Chart(d).mark_arc(innerRadius=55).encode(
        theta="n:Q",
        color=alt.Color("status:N", scale=alt.Scale(domain=["applied", "recommended"], range=[GOOD, WARN]),
                        legend=alt.Legend(orient="bottom", title="statut")),
        tooltip=["status", "n"]), 240)


# ───────────────────────── modèle ─────────────────────────
def importance(imp: pd.DataFrame):
    return fit(alt.Chart(imp).mark_bar(cornerRadiusEnd=4, color=TEAL).encode(
        x=alt.X("value:Q", title=None), y=alt.Y("feature:N", sort="-x", title=None),
        tooltip=["feature", alt.Tooltip("value:Q", format=".2f")]), _h(len(imp)))


def class_balance(bal: pd.DataFrame):
    return fit(alt.Chart(bal).mark_bar(cornerRadiusEnd=4, color=TEAL_2).encode(
        x=alt.X("classe:N", title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("effectif:Q"),
        tooltip=["classe", "effectif", alt.Tooltip("pourcentage:Q", format=".2f")]), 240)


def confusion(cm: pd.DataFrame):
    mid = float(cm["n"].max()) / 2
    base = alt.Chart(cm).encode(x=alt.X("prédit:N", title="prédit"), y=alt.Y("réel:N", title="réel"))
    heat = base.mark_rect(cornerRadius=6).encode(color=alt.Color("n:Q", scale=alt.Scale(range=["#eaf4f4", TEAL]), legend=None))
    txt = base.mark_text(fontSize=16, fontWeight="bold").encode(
        text="n:Q", color=alt.condition(f"datum.n > {mid}", alt.value("white"), alt.value(INK)))
    return fit(heat + txt, 260)


def model_compare(mt: pd.DataFrame):
    long = mt.melt(id_vars="Dataset", value_vars=["Balanced accuracy", "F1 macro", "CV balanced acc."],
                   var_name="Métrique", value_name="Score")
    return fit(alt.Chart(long).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("Dataset:N", title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1])),
        xOffset="Métrique:N",
        color=alt.Color("Métrique:N", scale=alt.Scale(range=[TEAL, TEAL_2, "#9ad4d6"]), legend=alt.Legend(orient="bottom")),
        tooltip=["Dataset", "Métrique", alt.Tooltip("Score:Q", format=".3f")]), 300)


# ───────────────────────── synthèse du projet ─────────────────────────
def quality_scores(sc: pd.DataFrame):
    return fit(alt.Chart(sc).mark_bar(cornerRadiusEnd=6).encode(
        x=alt.X("Score qualité:Q", scale=alt.Scale(domain=[0, 100])), y=alt.Y("Dataset:N", sort="-x", title=None),
        color=alt.Color("Verdict:N", scale=alt.Scale(domain=["good", "moderate", "poor"], range=[GOOD, WARN, BAD]),
                        legend=alt.Legend(orient="bottom")),
        tooltip=["Dataset", "Score qualité", "Verdict"]), 240)


def issues_grouped(issues: pd.DataFrame):
    return fit(alt.Chart(issues).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("Dataset:N", title=None), x=alt.X("Valeur:Q", title="%"), yOffset="Indicateur:N",
        color=alt.Color("Indicateur:N", scale=alt.Scale(range=[BAD, WARN, INFO, TEAL_2]), legend=alt.Legend(orient="bottom")),
        tooltip=["Dataset", "Indicateur", alt.Tooltip("Valeur:Q", format=".2f")]), 300)


def loc_bars(code: pd.DataFrame):
    return fit(alt.Chart(code).mark_bar(cornerRadiusEnd=4).encode(
        x="Lignes:Q", y=alt.Y("Fichier:N", sort="-x", title=None),
        color=alt.Color("Couche:N", scale=alt.Scale(range=LAYER_COLORS), legend=alt.Legend(orient="bottom")),
        tooltip=["Fichier", "Couche", "Lignes", "Rôle"]), _h(len(code), 26, 260))


def layer_donut(by_layer: pd.DataFrame):
    return fit(alt.Chart(by_layer).mark_arc(innerRadius=60).encode(
        theta="Lignes:Q", color=alt.Color("Couche:N", scale=alt.Scale(range=LAYER_COLORS), legend=alt.Legend(orient="bottom")),
        tooltip=["Couche", "Lignes"]), 280)


def category_bars(cats: pd.DataFrame):
    return fit(alt.Chart(cats).mark_bar(cornerRadiusEnd=4, color=TEAL).encode(
        x="Tables:Q", y=alt.Y("Catégorie:N", sort="-x", title=None), tooltip=["Catégorie", "Tables", "Lignes"]),
        _h(len(cats), 28, 220))
