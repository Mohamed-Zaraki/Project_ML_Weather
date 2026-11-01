"""weather_view.py : vue Streamlit en lecture seule du rapport météo stocké dans weather.db.

Ne dépend que de weather_db.WeatherDB (aucun calcul statistique ici).
Utilisation autonome :  streamlit run weather_view.py
Utilisation dans app.py :  from weather_view import render_weather ; render_weather()
"""
from __future__ import annotations

import calendar
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from theme import ACCENT_2, MUTED, apply_theme, attention_list, hero, kpi_row, section_title
from weather_db import WeatherDB

DEFAULT_DB = Path(os.environ.get("WEATHER_DB", Path(__file__).parent / "weather.db"))
ACCENT = "#0E6B6F"

VERDICT = {
    "negligible": "🟢 négligeable",
    "low": "🟡 faible",
    "moderate": "🟠 modéré",
    "high": "🔴 élevé",
}
SECTION_TITLES = {
    "description": "Description",
    "missing_duplicates": "Manquants et doublons",
    "outliers": "Valeurs extrêmes",
    "noise": "Bruit",
    "monthly_trends": "Tendances mensuelles",
    "correlation": "Corrélations",
    "kde": "Distributions",
    "model_phase": "Préparation et modèle",
}


# ───────────────────────────── Accès aux données ─────────────────────────────

@st.cache_resource(show_spinner=False)
def _db(path: str) -> WeatherDB:
    return WeatherDB(path, read_only=True)


@st.cache_data(show_spinner=False, ttl=30)
def _reports(path: str) -> list[dict]:
    return _db(path).list_reports()


@st.cache_data(show_spinner=False)
def _report(path: str, rid: str) -> dict:
    return _db(path).get_report(rid, include_images=False)


@st.cache_data(show_spinner=False)
def _png(path: str, rid: str, name: str) -> bytes:
    return _db(path).get_plot_png(rid, name)


# ───────────────────────────── Helpers d'affichage ─────────────────────────────

def _stretch(fn, *args, **kwargs):
    """Largeur pleine, compatible avec les anciennes et nouvelles versions de Streamlit."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


def _table(rows, **kwargs) -> None:
    if rows is None or len(rows) == 0:
        st.caption("Aucune donnée.")
        return
    _stretch(st.dataframe, pd.DataFrame(rows), hide_index=True, **kwargs)


def _insights(sec: dict) -> None:
    items = sec.get("insights") or []
    if items:
        st.markdown("\n".join(f"- {t.replace('$', chr(92) + '$')}" for t in items))


def _plots_grid(rep: dict, section: str, path: str) -> None:
    """Graphiques de la section, affichés EN HAUT de l'onglet, sur 2 colonnes (visibles sans défiler)."""
    plots = [p for p in rep.get("plots", []) if p.get("section") == section]
    if not plots:
        return
    rid = rep["meta"]["report_id"]
    key = f"w_plot_{rid}_{section}"
    n = len(plots)
    idx = min(st.session_state.get(key, 0), n - 1)

    def go(delta: int) -> None:
        st.session_state[key] = (st.session_state.get(key, 0) + delta) % n

    if n > 1:
        c1, c2, c3 = st.columns([1, 3, 1])
        with c1:
            _stretch(st.button, "◀ Précédent", key=f"{key}_prev", on_click=go, args=(-1,))
        with c2:
            st.markdown(f"<div style='text-align:center;padding-top:.4rem'><b>Graphique {idx + 1} / {n}</b></div>",
                        unsafe_allow_html=True)
        with c3:
            _stretch(st.button, "Suivant ▶", key=f"{key}_next", on_click=go, args=(1,))
    p = plots[idx]
    try:
        st.image(_png(path, rid, p["name"]), caption=p.get("title") or p["name"])
    except KeyError:
        st.caption(f"Image manquante : {p['name']}")


def _head(sec: dict) -> None:
    if sec.get("description"):
        st.caption(sec["description"])
    _insights(sec)


# ───────────────────────────── Sections ─────────────────────────────

def _sec_description(rep, path):
    sec = rep["description"]
    _head(sec)
    m = rep["meta"]
    c = st.columns(4)
    c[0].metric("Lignes", f"{m['n_rows']:,}".replace(",", " "))
    c[1].metric("Colonnes", m["n_cols"])
    c[2].metric("Colonne de date", m.get("date_column") or "—")
    c[3].metric("Colonnes constantes", len(sec.get("constant_columns", [])))
    st.subheader("Colonnes")
    _table(sec["info"]["columns"])
    st.subheader("Statistiques numériques")
    _table(sec["describe_numeric"])
    st.subheader("Statistiques catégorielles")
    _table(sec["describe_categorical"])
    st.subheader("Premières lignes")
    _table(sec["head"])


def _sec_missing(rep, path):
    sec = rep["missing_duplicates"]
    _head(sec)
    dup = sec["duplicates"]
    c = st.columns(3)
    c[0].metric("Cellules manquantes", sec["missing"]["total_cells"])
    c[1].metric("Lignes en double", dup["count"])
    c[2].metric("Part de doublons (%)", dup["percent"])
    only_missing = st.checkbox("Colonnes avec valeurs manquantes seulement", True, key="w_missing_only")
    cols = sec["missing"]["columns"]
    _table([c for c in cols if c["count"] > 0] if only_missing else cols)


def _sec_outliers(rep, path):
    sec = rep["outliers"]
    _head(sec)
    st.caption("Les valeurs signalées sont à examiner, pas à supprimer automatiquement.")
    df = pd.DataFrame(sec["features"]).sort_values("iqr_percent", ascending=False)
    _stretch(st.dataframe, df, hide_index=True)


def _sec_noise(rep, path):
    sec = rep["noise"]
    _head(sec)
    s = sec["dataset_summary"]
    c = st.columns(4)
    c[0].metric("Variables analysées", s["features_checked"])
    c[1].metric("Avec bruit", s["features_with_noise"])
    c[2].metric("Avec erreurs", s["features_with_errors"])
    c[3].metric("Bruit/signal moyen", f"{s['mean_noise_to_signal']:.3f}".replace(".", ","))
    df = pd.DataFrame(sec["features"])
    df["verdict"] = df["verdict"].map(lambda v: VERDICT.get(v, v))
    cols = ["feature", "verdict", "noise_to_signal", "snr_db", "spikes", "spike_percent",
            "out_of_range", "has_noise", "has_errors"]
    _stretch(st.dataframe, df[[c for c in cols if c in df.columns]].sort_values("noise_to_signal", ascending=False),
             hide_index=True)
    problems = [f for f in sec["features"] if f.get("noise_reasons") or f.get("error_reasons")]
    for f in problems:
        with st.expander(f"{f['feature']} · {VERDICT.get(f['verdict'], f['verdict'])}"):
            for r in f.get("noise_reasons", []):
                st.markdown(f"- Bruit : {r}")
            for r in f.get("error_reasons", []):
                st.markdown(f"- Erreur : {r}")
    for sk in sec.get("skipped", []):
        st.caption(f"Ignorée : {sk['feature']} ({sk['reason']}).")
    with st.expander("Méthode et seuils"):
        for k, v in sec["method"].items():
            st.markdown(f"**{k}** : {v}")
        st.json(sec["thresholds"])


def _sec_monthly(rep, path):
    sec = rep["monthly_trends"]
    _head(sec)
    feats = list(sec["monthly_stats"])
    feat = st.selectbox("Variable", feats, key="w_month_feat")
    df = pd.DataFrame(sec["monthly_stats"][feat])
    df["month"] = pd.to_datetime(df["month"])
    band = alt.Chart(df).mark_area(opacity=0.2, color=ACCENT).encode(x="month:T", y="low:Q", y2="high:Q")
    line = alt.Chart(df).mark_line(color=ACCENT).encode(
        x=alt.X("month:T", title=None), y=alt.Y("mean:Q", title=feat, scale=alt.Scale(zero=False)),
        tooltip=["month:T", "mean:Q", "low:Q", "high:Q", "std:Q"])
    st.altair_chart((band + line).properties(width="container", height=280))
    st.subheader("Moyenne par mois de l'année")
    # Le Model ne calcule la saisonnalité que pour les variables clés : pour les autres, on la déduit
    # des moyennes mensuelles (moyenne des moyennes de chaque mois civil, toutes années confondues).
    seas_data = (sec.get("seasonality") or {}).get(feat)
    derived = not seas_data
    if derived:
        by_month = df.groupby(df["month"].dt.month)["mean"].mean()
        seas_data = {calendar.month_name[int(m)]: float(v) for m, v in by_month.items()}
    seas = pd.DataFrame(list(seas_data.items()), columns=["mois", "moyenne"])
    bars = alt.Chart(seas).mark_bar(color=ACCENT).encode(
        x=alt.X("mois:N", sort=list(seas["mois"]), title=None), y=alt.Y("moyenne:Q", title=feat),
        tooltip=["mois:N", alt.Tooltip("moyenne:Q", format=".2f")])
    st.altair_chart(bars.properties(width="container", height=240))
    if derived:
        st.caption("Calculé à partir des moyennes mensuelles (le rapport ne contient pas de saisonnalité pour cette variable).")


def _sec_correlation(rep, path):
    sec = rep["correlation"]
    _head(sec)
    kind = st.radio("Coefficient", ["pearson", "spearman"], horizontal=True, key="w_corr_kind")
    mat = pd.DataFrame(sec[kind])
    long = mat.reset_index(names="a").melt("a", var_name="b", value_name="r")
    heat = alt.Chart(long).mark_rect().encode(
        x=alt.X("b:N", title=None, sort=list(mat.columns)), y=alt.Y("a:N", title=None, sort=list(mat.index)),
        color=alt.Color("r:Q", scale=alt.Scale(domain=[-1, 0, 1], range=["#B04A08", "#F4F4F4", ACCENT]), title="r"),
        tooltip=["a:N", "b:N", alt.Tooltip("r:Q", format=".3f")])
    st.altair_chart(heat.properties(width="container", height=340))
    st.subheader("Paires de variables")
    pairs = pd.DataFrame(sec["pairs"])
    _stretch(st.dataframe, pairs.reindex(pairs["pearson"].abs().sort_values(ascending=False).index), hide_index=True)
    bc = sec.get("by_category")
    if bc:
        st.subheader(f"Moyennes par « {bc['category']} »")
        if bc.get("note"):
            st.caption(bc["note"])
        _table(bc["means"])


def _sec_kde(rep, path):
    sec = rep["kde"]
    _head(sec)
    _table(sec["features"])
    for sk in sec.get("skipped", []):
        st.caption(f"Ignorée : {sk['feature']} ({sk['reason']}).")


def _fr(n, nd: int = 0) -> str:
    """Nombre au format français : 85 224 ou 7,96."""
    return f"{n:,.{nd}f}".replace(",", " ").replace(".", ",")


def _importance_chart(df: pd.DataFrame, x: str, x_title: str, tip: list) -> None:
    chart = (
        alt.Chart(df).mark_bar(cornerRadiusEnd=5, size=16, color=ACCENT_2).encode(
            y=alt.Y("feature:N", sort=list(df["feature"]), title=None,
                    axis=alt.Axis(domain=False, ticks=False, labelLimit=240)),
            x=alt.X(f"{x}:Q", title=x_title, axis=alt.Axis(domain=False)),
            tooltip=tip)
        .properties(width="container", height=max(120, 30 * len(df))))
    st.altair_chart(chart)


def _sec_model_phase(rep, path):
    sec = rep["model_phase"]
    _head(sec)
    if "target_balance" not in sec:          # étape ignorée par le Model (cible absente)
        return
    tb, prep, fi = sec["target_balance"], sec["preprocessing"], sec["feature_importance"]
    ev = sec.get("evaluation")               # absent des anciens rapports
    ranked = fi["ranked_features"]
    perm = fi.get("permutation_ranking") or []
    counts, pcts = tb["class_counts"], tb["class_percentages"]
    ratio = tb.get("imbalance_ratio")
    top2 = sum(r["percentage"] for r in ranked[:2])

    cards = [
        ("Variable cible", tb["target_feature"], f"{_fr(tb['valid_observations'])} observations valides"),
        ("Déséquilibre", f"{_fr(ratio, 2)} : 1" if ratio else "—", "classe majoritaire / minoritaire",
         "good" if tb["is_balanced"] else "bad"),
        ("Variables", f"{prep['total_input_features']} → {prep['total_transformed_features']}",
         "avant → après One-Hot", "info"),
    ]
    if ev:
        gain = ev["accuracy"] - ev["baseline_accuracy"]
        cards += [
            ("Exactitude équilibrée", f"{_fr(ev['balanced_accuracy'] * 100, 1)} %", f"test : {ev['test_period']}",
             "good" if ev["balanced_accuracy"] >= 0.7 else "warn"),
            ("F1 macro", _fr(ev["f1_macro"], 3), f"exactitude {_fr(ev['accuracy'] * 100, 1)} % "
                                                  f"(référence {_fr(ev['baseline_accuracy'] * 100, 1)} %)",
             "good" if gain > 0.02 else "warn"),
        ]
    kpi_row(cards)

    points = []
    if not tb["is_balanced"] and ratio:
        points.append(("bad", "Cible très déséquilibrée",
                       f"{_fr(ratio, 2)} contre 1 : le modèle utilise class_weight={fi['class_weight']}, "
                       "lisez l'exactitude équilibrée et le F1, pas l'exactitude seule."))
    if tb.get("missing_count"):
        points.append(("warn", "Cible manquante",
                       f"{_fr(tb['missing_count'])} lignes sans {tb['target_feature']}, exclues de l'entraînement."))
    if ev and ev["accuracy"] - ev["baseline_accuracy"] <= 0.02:
        points.append(("warn", "Pas mieux que la classe majoritaire",
                       "L'exactitude ne dépasse pas celle d'un modèle qui prédit toujours la classe la plus fréquente."))
    if top2 >= 80:
        points.append(("warn", "Importance concentrée",
                       f"Deux variables pèsent {_fr(top2, 1)} % de l'importance d'impureté : lisez-les ensemble "
                       "(voir les observations ci-dessus)."))
    if points:
        attention_list(points)

    section_title("Équilibre des classes", tb["target_feature"])
    df = pd.DataFrame({"classe": list(counts), "effectif": list(counts.values()), "part": [pcts.get(k) for k in counts]})
    c1, c2 = st.columns([3, 2], gap="medium")
    with c1:
        chart = (
            alt.Chart(df).mark_bar(cornerRadiusEnd=5, size=22).encode(
                y=alt.Y("classe:N", sort="-x", title=None, axis=alt.Axis(domain=False, ticks=False, labelLimit=220)),
                x=alt.X("effectif:Q", title=None, axis=alt.Axis(domain=False)),
                color=alt.condition(alt.datum.classe == "Missing / Not recorded",
                                    alt.value(MUTED), alt.value(ACCENT)),
                tooltip=["classe:N", alt.Tooltip("effectif:Q", format=","), alt.Tooltip("part:Q", title="part (%)")])
            .properties(width="container", height=max(90, 44 * len(df))))
        st.altair_chart(chart)
    with c2:
        _table([{"classe": k, "effectif": v, "part (%)": pcts.get(k)} for k, v in counts.items()])

    section_title("Prétraitement", f"{prep['transformer_method']} · {_fr(prep['samples_trained'])} échantillons d'entraînement")
    col_num, col_cat = st.columns(2, gap="medium")
    with col_num:
        with st.container(border=True):
            st.markdown("**Variables numériques**")
            st.markdown(" → ".join(f"`{x}`" for x in prep["numerical_pipeline"]))
            st.caption(" · ".join(prep["numerical_features"]))
    with col_cat:
        with st.container(border=True):
            st.markdown("**Variables catégorielles**")
            st.markdown(" → ".join(f"`{x}`" for x in prep["categorical_pipeline"]))
            st.caption(" · ".join(prep["categorical_features"]) or "—")

    if ev:
        section_title("Évaluation", ev["split"])
        st.caption(f"Entraînement : {ev['train_period']} ({_fr(ev['n_train'])} lignes) · "
                   f"test : {ev['test_period']} ({_fr(ev['n_test'])} lignes).")
        _table([{"classe": k, "rappel (%)": round(v * 100, 1)} for k, v in ev["per_class_recall"].items()])

    section_title("Importance des variables",
                  f"{fi['model']} · {fi['n_estimators']} arbres · class_weight={fi['class_weight']}")
    detailed = fi.get("detailed_transformed_ranking") or []
    levels = (["Permutation (test)"] if perm else []) + ["Impureté (RF)"] + (["Après One-Hot"] if detailed else [])
    view = st.radio("Niveau", levels, horizontal=True, key="w_fi_view", label_visibility="collapsed") \
        if len(levels) > 1 else levels[0]
    if view == "Permutation (test)":
        dfi = pd.DataFrame(perm)
        _importance_chart(dfi, "drop_points", "Baisse d'exactitude équilibrée (points)",
                          ["rank:Q", "feature:N", alt.Tooltip("drop_points:Q", title="baisse (pts)", format=".2f"),
                           alt.Tooltip("std_points:Q", title="écart-type", format=".2f")])
        st.caption("Mesurée sur la période de test : on mélange une variable et on regarde de combien le modèle se dégrade. "
                   "Plus fiable que l'impureté quand deux variables sont presque identiques.")
    else:
        rows = detailed[:15] if view == "Après One-Hot" else ranked
        dfi = pd.DataFrame(rows)
        _importance_chart(dfi, "percentage", "Importance (%)",
                          ["rank:Q", "feature:N", alt.Tooltip("percentage:Q", title="%", format=".2f")])
        if view == "Après One-Hot":
            st.caption(f"15 premières variables sur {prep['total_transformed_features']}.")
    with st.expander("Tableaux complets"):
        _stretch(st.dataframe, pd.DataFrame(perm or detailed or ranked), hide_index=True)
        if perm:
            _stretch(st.dataframe, pd.DataFrame(ranked), hide_index=True)


RENDERERS = {
    "description": _sec_description,
    "missing_duplicates": _sec_missing,
    "outliers": _sec_outliers,
    "noise": _sec_noise,
    "monthly_trends": _sec_monthly,
    "correlation": _sec_correlation,
    "kde": _sec_kde,
    "model_phase": _sec_model_phase,
}


# ───────────────────────────── Point d'entrée ─────────────────────────────

def render_weather(db_path: str | Path = DEFAULT_DB, show_header: bool = True) -> dict | None:
    """Affiche le rapport et le renvoie (sans images), ou None s'il n'y a rien à afficher.
    show_header=False : sans bandeau (quand la page hôte a déjà son propre titre)."""
    path = str(db_path)
    if not Path(path).exists():
        st.error(f"Base introuvable : {path}")
        st.caption("Placez weather.db à côté de l'application ou définissez WEATHER_DB=/chemin/weather.db.")
        return None
    reports = _reports(path)
    if not reports:
        st.info("weather.db ne contient aucun rapport.")
        return None

    # Liste de tous les rapports sauvegardés dans weather.db (du plus récent au plus ancien)
    ids = [r["report_id"] for r in reports]
    labels = {r["report_id"]: f"{r['generated_at'][:16].replace('T', ' ')} · {r['dataset']} · {r['n_rows']:,} lignes".replace(",", " ")
              for r in reports}
    st.sidebar.markdown(f"**Rapports enregistrés** ({len(ids)})")
    rid = st.sidebar.radio("Rapports enregistrés", ids, format_func=labels.get, key="w_report",
                           label_visibility="collapsed")
    if st.sidebar.button("Actualiser la liste", key="w_refresh"):
        st.cache_data.clear()
        st.rerun()
    st.sidebar.caption(f"Base : {Path(path).name}")
    rep = _report(path, rid)
    m = rep["meta"]

    when = m["generated_at"][:16].replace("T", " ")
    rows = f"{m['n_rows']:,}".replace(",", " ")
    if show_header:
        hero("Rapport de dataset", m["dataset"],
             [("Généré le", when), ("Lignes", rows), ("Colonnes", m["n_cols"])])

    names = [n for n in rep if n not in ("meta", "plots")]
    tabs = st.tabs([SECTION_TITLES.get(n, n) for n in names])
    for tab, name in zip(tabs, names):
        with tab:
            _plots_grid(rep, name, path)         # graphiques en haut, avant les tableaux
            fn = RENDERERS.get(name)
            if fn is None:                       # section inconnue : affichage générique
                _head(rep[name])
                st.json(rep[name], expanded=1)
            else:
                fn(rep, path)
    return rep


if __name__ == "__main__":
    st.set_page_config(page_title="Rapport météo", layout="wide")
    apply_theme()
    render_weather()
