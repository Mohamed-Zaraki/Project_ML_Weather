"""project_dashboard.py : tableau de bord de synthèse du projet ML Weather.

Montre ce qui a été réalisé : catalogue de données, EDA, qualité Phase I, modèles,
prétraitement, architecture du code. Tout est lu depuis les fichiers du projet.

Lancer (depuis la racine du projet) :
    streamlit run project_dashboard.py
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Tableau de bord · ML Weather", page_icon="📊", layout="wide")

import dashboard_data as dd  # noqa: E402
from dataset_cards import render_dataset_cards  # noqa: E402
from dataset_overview import render_dataset_detail  # noqa: E402
from outlier_view import render_boxplot, render_outliers_noise  # noqa: E402
from theme import apply_theme, attention_list, brand, hero, kpi_row, section_title  # noqa: E402

apply_theme()
st.markdown(
    """<style>
.tl { position: relative; margin: .4rem 0 1.2rem 0; padding-left: 1.6rem; border-left: 2px solid var(--line); }
.tl-i { position: relative; margin-bottom: 1rem; }
.tl-i::before { content: ""; position: absolute; left: -2.02rem; top: .35rem; width: .9rem; height: .9rem;
  border-radius: 50%; background: var(--accent-2); border: 3px solid var(--soft-2); }
.tl-d { font-family: 'Geist Mono', ui-monospace, monospace; font-size: .7rem; letter-spacing: .08em; opacity: .6; }
.tl-t { font-weight: 600; }
.tl-x { font-size: .85rem; opacity: .75; }
</style>""",
    unsafe_allow_html=True,
)

TEAL, TEAL_2 = "#0E6B6F", "#14A3A8"
GOOD, WARN, BAD, INFO, MUTED = "#2B7A4B", "#B04A08", "#C0392B", "#3B6FB6", "#7A8A90"


# ───────────────────────── utilitaires ─────────────────────────
def stretch(fn, *args, **kwargs):
    """Compatibilité entre versions de Streamlit (width='stretch' vs use_container_width)."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except TypeError:
        return fn(*args, use_container_width=True, **kwargs)


def chart(c: alt.Chart, height: int = 280):
    st.altair_chart(c.properties(width="container", height=height))


@st.cache_data(show_spinner="Lecture des rapports…")
def get_reports():
    return dd.load_reports()


@st.cache_data(show_spinner=False)
def get_db():
    return dd.db_stats()



SEC_FR = {"missing_duplicates": "Manquants et doublons", "contradictions": "Contradictions", "data_quality": "Indicateurs de qualité",
          "outliers": "Outliers", "noise": "Bruit", "correlation": "Corrélations", "kde": "Distributions",
          "monthly_trends": "Tendances mensuelles", "model_phase": "Modèle", "interpretation": "Interprétation"}


def plots_carousel(sections: list[str], title: str | None = None) -> None:
    """Graphiques PNG des sections données, dans l'onglet qui les concerne : un à la fois, boutons ◀ / ▶."""
    for sec in sections:
        items = [p for p in dd.plots_for(rep) if p["section"] == sec and "boxplot" not in p["path"].stem.lower()]
        if not items:
            continue
        n = len(items)
        k = f"gal_{sel}_{sec}"
        idx = min(st.session_state.get(k, 0), n - 1)

        def go(delta: int, k=k, n=n) -> None:
            st.session_state[k] = (st.session_state.get(k, 0) + delta) % n

        section_title(title or f"Graphiques · {SEC_FR.get(sec, sec)}", f"{n} graphique(s)")
        if n > 1:
            c1, c2, c3 = st.columns([1, 3, 1])
            with c1:
                stretch(st.button, "◀ Précédent", key=f"{k}_prev", on_click=go, args=(-1,))
            with c2:
                st.markdown(f"<div style='text-align:center;padding-top:.4rem'><b>{idx + 1} / {n}</b> · {items[idx]['title']}</div>",
                            unsafe_allow_html=True)
            with c3:
                stretch(st.button, "Suivant ▶", key=f"{k}_next", on_click=go, args=(1,))
        st.image(str(items[idx]["path"]), caption=items[idx]["title"])


reports = get_reports()
if not reports:
    st.error("Aucun rapport trouvé. Placez ce fichier à la racine du projet (à côté de data.db et des dossiers eda_output_*).")
    st.stop()
db = get_db()
ov = dd.overview_table(reports)
mt = dd.model_table(reports)

# ───────────────────────── sélection (accueil ↔ page dataset) ─────────────────────────
brand("Tableau de bord du projet")
sel = st.session_state.get("home_dataset")
if sel not in reports:
    sel = None
if sel is not None:
    if st.sidebar.button("← Retour aux datasets", use_container_width=True, key="back_home"):
        st.session_state["home_dataset"] = None
        st.rerun()
    st.sidebar.markdown("**Changer de dataset**")
    for k in reports:
        if k != sel and st.sidebar.button(dd.label(k), key=f"side_{k}", use_container_width=True):
            st.session_state["home_dataset"] = k
            st.rerun()
st.sidebar.divider()
st.sidebar.caption("Sources lues : rapports JSON, data.db et scripts .py du dossier du projet.")
rep = reports[sel] if sel else None

# ───────────────────────── bandeau ─────────────────────────
total_rows = int(ov["Lignes"].sum())
if sel is None:
    hero(
        "Projet ML · Analyse météo et données de santé",
        "Ce qui a été réalisé",
        [
            ("Datasets analysés", str(len(reports))),
            ("Lignes", f"{total_rows:,}".replace(",", " ")),
            ("Graphiques", str(int(ov["Graphiques"].sum()))),
            ("Insights", str(int(ov["Insights"].sum()))),
        ],
    )
    section_title("Choisir un dataset", "cliquez sur « Ouvrir » pour détailler : données, qualité, outliers & bruit, modèle")
    cards = []
    for _, r in ov.iterrows():
        q = r["Score qualité"]
        cards.append({"key": r["key"], "name": dd.label(r["key"]),
                      "subtitle": f"{int(r['Lignes']):,} lignes · {int(r['Colonnes'])} colonnes".replace(",", " ")
                                  + ("" if pd.isna(q) else f" · qualité {q:.0f}/100")})
    render_dataset_cards(cards)
    tabs = dict(zip([0, 2, 4], st.tabs(["Vue d'ensemble", "Comparatif des modèles", "Code & architecture"])))
else:
    _r = ov[ov["key"] == sel].iloc[0]
    hero("Dataset", dd.label(sel),
         [("Lignes", f"{int(_r['Lignes']):,}".replace(",", " ")), ("Colonnes", str(int(_r["Colonnes"]))),
          ("Cible", str(_r["Cible"])), ("Graphiques", str(int(_r["Graphiques"])))])
    tabs = dict(zip([7, 1, 8, 9, 10, 6, 3],
                    st.tabs(["Données", "Qualité", "Outliers & bruit", "Corrélations", "Distributions", "Modèle", "Prétraitement"])))

# ═════════════════════════ 1. VUE D'ENSEMBLE ═════════════════════════
if 0 in tabs:
    with tabs[0]:
        scored = ov.dropna(subset=["Score qualité"])
        kpis = [
            ("Datasets avec rapport", len(reports), "météo + 4 jeux de santé", "info"),
            ("Qualité moyenne", f"{scored['Score qualité'].mean():.1f}/100", "4 datasets scorés (Phase I)", "good"),
        ]
        if not mt.empty:
            kpis.append(("Balanced accuracy moyenne", f"{mt['Balanced accuracy'].mean():.3f}",
                         f"{len(mt)} modèles évalués", "good"))
        if db:
            kpis.append(("Tables en base", db["counts"]["datasets"], f"{db['n_categories']} catégories · data.db {db['size_mb']} Mo"))
        kpi_row(kpis)

        section_title("Points d'attention", "calculés à partir des rapports")
        attention_list(dd.attention_points(reports))

        section_title("Parcours du projet")
        html = "".join(
            f'<div class="tl-i"><div class="tl-d">{s["date"] or "—"}</div><div class="tl-t">{s["title"]}</div>'
            f'<div class="tl-x">{s["detail"]}</div></div>'
            for s in dd.timeline(reports, db)
        )
        st.markdown(f'<div class="tl">{html}</div>', unsafe_allow_html=True)

        section_title("Comparatif des datasets")
        show = ov.drop(columns=["key"]).copy()
        stretch(st.dataframe, show, hide_index=True,
                column_config={
                    "Score qualité": st.column_config.ProgressColumn("Score qualité", min_value=0, max_value=100, format="%.1f"),
                    "Lignes": st.column_config.NumberColumn("Lignes", format="%d"),
                })

        c1, c2 = st.columns(2)
        with c1:
            section_title("Score de qualité Phase I", "plus haut = meilleur")
            sc = scored[["Dataset", "Score qualité", "Verdict"]]
            chart(alt.Chart(sc).mark_bar(cornerRadiusEnd=6).encode(
                x=alt.X("Score qualité:Q", scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("Dataset:N", sort="-x", title=None),
                color=alt.Color("Verdict:N", scale=alt.Scale(domain=["good", "moderate", "poor"], range=[GOOD, WARN, BAD]),
                                legend=alt.Legend(orient="bottom")),
                tooltip=["Dataset", "Score qualité", "Verdict"]), 240)
        with c2:
            section_title("Problèmes détectés (% du dataset)", "doublons, contradictions, bruit, outliers")
            issues = ov.melt(id_vars="Dataset", value_vars=["Doublons %", "Contradictions %", "Bruit %", "Outliers %"],
                             var_name="Indicateur", value_name="Valeur").dropna()
            chart(alt.Chart(issues).mark_bar(cornerRadiusEnd=4).encode(
                y=alt.Y("Dataset:N", title=None),
                x=alt.X("Valeur:Q", title="%"),
                yOffset="Indicateur:N",
                color=alt.Color("Indicateur:N", scale=alt.Scale(range=[BAD, WARN, INFO, TEAL_2]),
                                legend=alt.Legend(orient="bottom")),
                tooltip=["Dataset", "Indicateur", alt.Tooltip("Valeur:Q", format=".2f")]), 300)
        st.caption("La météo n'a pas de section « data_quality » : bruit = part des variables testées jugées bruitées, "
                   "doublons issus de la section manquants/doublons ; contradictions et outliers consensuels non calculés.")

# ═════════════════════════ 2. QUALITÉ ═════════════════════════
if 1 in tabs:
    with tabs[1]:
        st.subheader(dd.label(sel))
        row = ov[ov["key"] == sel].iloc[0]
        q = row["Score qualité"]
        kpi_row([
            ("Lignes × colonnes", f"{int(row['Lignes']):,}".replace(",", " ") + f" × {int(row['Colonnes'])}", f"cible : {row['Cible']}"),
            ("Score qualité", "n/a" if pd.isna(q) else f"{q:.1f}/100", row["Verdict"] if isinstance(row["Verdict"], str) else "section absente",
             {"good": "good", "moderate": "warn", "poor": "bad"}.get(row["Verdict"], "neutral")),
            ("Doublons", "n/a" if pd.isna(row["Doublons %"]) else f"{row['Doublons %']:.2f} %", None,
             "bad" if (pd.notna(row["Doublons %"]) and row["Doublons %"] >= 5) else "good"),
            ("Bruit", "n/a" if pd.isna(row["Bruit %"]) else f"{row['Bruit %']:.0f} %", "variables bruitées", "warn"),
        ])

        ind = dd.indicators_df(rep)
        if not ind.empty:
            section_title("Indicateurs de qualité", "part du dataset concernée par chaque problème")
            chart(alt.Chart(ind).mark_bar(cornerRadiusEnd=6).encode(
                x=alt.X("issue_percent:Q", title="% concerné"),
                y=alt.Y("indicator:N", sort="-x", title=None),
                color=alt.Color("status:N", scale=alt.Scale(domain=["good", "warning", "critical"], range=[GOOD, WARN, BAD]),
                                legend=alt.Legend(orient="bottom", title="statut")),
                tooltip=["indicator", "issue_percent", "status", "detail"]), max(180, 36 * len(ind)))
            stretch(st.dataframe, ind[["indicator", "issue_percent", "status", "detail"]], hide_index=True)

        nz = dd.noise_df(rep)
        c1, c2 = st.columns(2)
        with c1:
            section_title("Bruit par variable", "rapport bruit/signal (plus bas = plus propre)")
            if nz.empty:
                st.info("Pas de données de bruit.")
            else:
                nz = nz.sort_values("noise_to_signal", ascending=False).head(15)
                chart(alt.Chart(nz).mark_bar(cornerRadiusEnd=4).encode(
                    x=alt.X("noise_to_signal:Q", title="bruit / signal"),
                    y=alt.Y("feature:N", sort="-x", title=None),
                    color=alt.Color("verdict:N", scale=alt.Scale(domain=["negligible", "low", "moderate", "high"],
                                                                  range=[GOOD, TEAL_2, WARN, BAD]),
                                    legend=alt.Legend(orient="bottom")),
                    tooltip=["feature", alt.Tooltip("noise_to_signal:Q", format=".3f"), "snr_db", "verdict"]),
                    max(200, 26 * len(nz)))
        with c2:
            section_title("Corrélations les plus fortes", "Pearson")
            cp = dd.corr_pairs_df(rep)
            if cp.empty:
                st.info("Pas de corrélations.")
            else:
                chart(alt.Chart(cp).mark_bar(cornerRadiusEnd=4).encode(
                    x=alt.X("pearson:Q", scale=alt.Scale(domain=[-1, 1])),
                    y=alt.Y("paire:N", sort=alt.EncodingSortField("pearson", order="descending"), title=None),
                    color=alt.Color("pearson:Q", scale=alt.Scale(domain=[-1, 0, 1], range=[INFO, "#d9dedf", BAD]), legend=None),
                    tooltip=["paire", alt.Tooltip("pearson:Q", format=".3f"), alt.Tooltip("spearman:Q", format=".3f"), "strength"]),
                    max(200, 30 * len(cp)))

        plots_carousel(["data_quality", "missing_duplicates", "contradictions"])

# ═════════════════════════ 3. MODÈLES ═════════════════════════
if 2 in tabs:
    with tabs[2]:
        section_title("Comparaison des modèles", "Random Forest pondéré · évaluation sur jeu de test + validation croisée")
        if mt.empty:
            st.info("Aucune évaluation de modèle dans les rapports.")
        else:
            long = mt.melt(id_vars="Dataset", value_vars=["Balanced accuracy", "F1 macro", "CV balanced acc."],
                           var_name="Métrique", value_name="Score")
            chart(alt.Chart(long).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("Dataset:N", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1])),
                xOffset="Métrique:N",
                color=alt.Color("Métrique:N", scale=alt.Scale(range=[TEAL, TEAL_2, "#9ad4d6"]), legend=alt.Legend(orient="bottom")),
                tooltip=["Dataset", "Métrique", alt.Tooltip("Score:Q", format=".3f")]), 300)
            stretch(st.dataframe, mt.drop(columns=["key"]), hide_index=True)


# ═════════════════════════ 3b. MODÈLE (détail du dataset) ═════════════════════════
if 6 in tabs:
    with tabs[6]:
        st.subheader(f"Détail · {dd.label(sel)}")
        ev = rep.get("model_phase", {}).get("evaluation")
        if ev:
            lift = ev["balanced_accuracy"] - ev["baseline_accuracy"]
            kpi_row([
                ("Balanced accuracy", f"{ev['balanced_accuracy']:.3f}", f"accuracy de base {ev['baseline_accuracy']:.3f}", "good"),
                ("F1 macro", f"{ev['f1_macro']:.3f}", None),
                ("ROC AUC", f"{ev['roc_auc']:.4f}" if ev.get("roc_auc") is not None else "n/a", None),
                ("Validation croisée", f"{ev['cv_balanced_accuracy_mean']:.3f}", f"± {ev['cv_balanced_accuracy_std']:.3f}"),
                ("Gain vs base", f"{lift:+.3f}", "balanced acc. − accuracy de base", "info"),
            ])
        else:
            st.info("Ce rapport ne contient pas d'évaluation du modèle (cas du rapport météo enregistré le 25 oct.).")

        c1, c2 = st.columns(2)
        with c1:
            imp = dd.importance_df(rep)
            if not imp.empty:
                section_title("Importance des variables", imp["method"].iloc[0])
                chart(alt.Chart(imp).mark_bar(cornerRadiusEnd=4, color=TEAL).encode(
                    x=alt.X("value:Q", title=None),
                    y=alt.Y("feature:N", sort="-x", title=None),
                    tooltip=["feature", alt.Tooltip("value:Q", format=".2f")]), max(200, 28 * len(imp)))
        with c2:
            bal = dd.class_balance_df(rep)
            if not bal.empty:
                section_title("Équilibre des classes", "variable cible")
                chart(alt.Chart(bal).mark_bar(cornerRadiusEnd=4, color=TEAL_2).encode(
                    x=alt.X("classe:N", title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("effectif:Q"),
                    tooltip=["classe", "effectif", alt.Tooltip("pourcentage:Q", format=".2f")]), 240)

        cm = dd.confusion_df(rep)
        if not cm.empty:
            section_title("Matrice de confusion", "jeu de test")
            base = alt.Chart(cm).encode(x=alt.X("prédit:N", title="prédit"), y=alt.Y("réel:N", title="réel"))
            heat = base.mark_rect(cornerRadius=6).encode(color=alt.Color("n:Q", scale=alt.Scale(range=["#eaf4f4", TEAL]), legend=None))
            txt = base.mark_text(fontSize=16, fontWeight="bold").encode(
                text="n:Q", color=alt.condition(alt.datum.n > float(cm["n"].max()) / 2, alt.value("white"), alt.value("#10222A")))
            chart(heat + txt, 260)

        plots_carousel(["model_phase"])

# ═════════════════════════ 4. PRÉTRAITEMENT ═════════════════════════
if 3 in tabs:
    with tabs[3]:
        st.subheader(dd.label(sel))
        plan = dd.plan_df(rep)
        if plan.empty:
            st.info("Ce rapport n'a pas de section « interpretation » (cas du rapport météo).")
            pre = rep.get("model_phase", {}).get("preprocessing")
            if pre:
                section_title("Pipeline de prétraitement du modèle météo")
                st.write("**Numérique** : " + " → ".join(pre.get("numerical_pipeline", [])))
                st.write("**Catégoriel** : " + " → ".join(pre.get("categorical_pipeline", [])))
                st.caption(f"{len(pre.get('numerical_features', []))} variables numériques, "
                           f"{len(pre.get('categorical_features', []))} catégorielle(s), cible {pre.get('target_feature')}.")
        else:
            n_app = int((plan["status"] == "applied").sum())
            n_rec = int((plan["status"] == "recommended").sum())
            kpi_row([("Étapes appliquées", n_app, "déjà dans le pipeline", "good"),
                     ("Étapes recommandées", n_rec, "restent à faire", "warn")])
            c1, c2 = st.columns(2)
            with c1:
                section_title("Plan de prétraitement")
                stretch(st.dataframe, plan.rename(columns={"step": "Étape", "status": "Statut", "reason": "Constat"}),
                        hide_index=True)
            with c2:
                section_title("Recommandations priorisées")
                rec = dd.recommendations_df(rep)
                stretch(st.dataframe, rec.rename(columns={"priority": "Priorité", "area": "Zone", "finding": "Constat",
                                                          "action": "Action"}), hide_index=True)

        plots_carousel(["interpretation"])

# ═════════════════════════ 5. CODE & ARCHITECTURE ═════════════════════════
if 4 in tabs:
    with tabs[4]:
        code = dd.code_inventory()
        kpi_row([
            ("Scripts Python", len(code), "hors dashboard"),
            ("Lignes de code", f"{int(code['Lignes'].sum()):,}".replace(",", " "), None),
            ("Plus gros module", code.iloc[0]["Fichier"], f"{int(code.iloc[0]['Lignes'])} lignes"),
        ])
        c1, c2 = st.columns([3, 2])
        with c1:
            section_title("Lignes de code par fichier", "couleur = couche de l'architecture")
            chart(alt.Chart(code).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("Lignes:Q"), y=alt.Y("Fichier:N", sort="-x", title=None),
                color=alt.Color("Couche:N", scale=alt.Scale(range=[TEAL, TEAL_2, INFO, WARN, MUTED]),
                                legend=alt.Legend(orient="bottom")),
                tooltip=["Fichier", "Couche", "Lignes", "Rôle"]), max(260, 26 * len(code)))
        with c2:
            section_title("Lignes par couche")
            by_layer = code.groupby("Couche", as_index=False)["Lignes"].sum()
            chart(alt.Chart(by_layer).mark_arc(innerRadius=60).encode(
                theta="Lignes:Q",
                color=alt.Color("Couche:N", scale=alt.Scale(range=[TEAL, TEAL_2, INFO, WARN, MUTED]),
                                legend=alt.Legend(orient="bottom")),
                tooltip=["Couche", "Lignes"]), 280)
        stretch(st.dataframe, code, hide_index=True)

        if db:
            section_title("Base SQLite data.db", f"{db['size_mb']} Mo")
            c = db["counts"]
            kpi_row([("Rapports", c["reports"]), ("Sections", c["sections"]), ("Insights", c["insights"]),
                     ("Graphiques (BLOB)", c["plots"]), ("Tables importées", c["datasets"])])
            cats = db["categories"]
            chart(alt.Chart(cats).mark_bar(cornerRadiusEnd=4, color=TEAL).encode(
                x=alt.X("Tables:Q"), y=alt.Y("Catégorie:N", sort="-x", title=None),
                tooltip=["Catégorie", "Tables", "Lignes"]), max(220, 28 * len(cats)))

# ═════════════════════════ 0b. DONNÉES (comme la vue d'ensemble de app.py) ═════════════════════════
if 7 in tabs:
    with tabs[7]:
        render_dataset_detail(dd.ROOT / "data.db", dd.DB_DATASET_IDS[sel], key_prefix=f"ds_{sel}")

# ═════════════════════════ 2b. OUTLIERS & BRUIT ═════════════════════════
if 8 in tabs:
    with tabs[8]:
        render_outliers_noise(rep, sel)
        plots_carousel(["outliers", "noise"])

# ═════════════════════════ CORRÉLATIONS & DISTRIBUTIONS ═════════════════════════
if 9 in tabs:
    with tabs[9]:
        plots_carousel(["correlation"], "Graphiques · Corrélations")
if 10 in tabs:
    with tabs[10]:
        plots_carousel(["kde"], "Graphiques · Distributions")
        plots_carousel(["monthly_trends"], "Graphiques · Tendances mensuelles")
