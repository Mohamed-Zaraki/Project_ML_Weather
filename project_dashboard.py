"""project_dashboard.py : tableau de bord de synthèse du projet ML Weather & Santé.

Montre ce qui a été réalisé : catalogue de données, EDA, qualité Phase I, modèles,
prétraitement, architecture du code et explications visuelles des problématiques métier.

Lancer (depuis la racine du projet) :
    streamlit run project_dashboard.py
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Data Description & Quality · ML Dashboard", page_icon="📊", layout="wide")

import dashboard_data as dd  # noqa: E402
import dashboard_charts as ch  # noqa: E402
from dataset_cards import render_dataset_cards  # noqa: E402
from dataset_overview import render_dataset_detail  # noqa: E402
from outlier_view import render_boxplot, render_outliers_noise  # noqa: E402
from theme import apply_theme, attention_list, brand, hero, kpi_row, section_title  # noqa: E402

apply_theme()
st.markdown(
    """<style>
.tl { position: relative; margin: .5rem 0 1.3rem 0; padding-left: 1.8rem; border-left: 2px solid var(--line); }
.tl-i { position: relative; margin-bottom: 1.1rem; }
.tl-i::before { content: ""; position: absolute; left: -2.25rem; top: .35rem; width: 1rem; height: 1rem;
  border-radius: 50%; background: linear-gradient(135deg, var(--accent), var(--accent-2));
  border: 3px solid var(--soft-2); box-shadow: 0 0 10px rgba(99, 102, 241, 0.4); }
.tl-d { font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: .72rem; letter-spacing: .08em; opacity: .7; font-weight: 600; color: var(--accent-2); }
.tl-t { font-weight: 700; font-size: .95rem; margin-top: .1rem; }
.tl-x { font-size: .84rem; opacity: .8; line-height: 1.35; }

.problem-card {
  border: 1px solid rgba(226, 232, 240, 0.95); border-radius: 18px; padding: 1.2rem 1.3rem;
  background: #FFFFFF; height: 100%;
  box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.04);
  transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease;
  position: relative; overflow: hidden;
}
.problem-card:hover {
  transform: translateY(-4px); border-color: var(--accent-2);
  box-shadow: 0 14px 30px -8px rgba(79, 70, 229, 0.14);
}
.problem-card-h { display: flex; align-items: center; justify-content: space-between; margin-bottom: .6rem; }
.problem-card-t { font-weight: 800; font-size: 1.05rem; color: #0F172A; }
.problem-card-b { font-size: .84rem; color: #334155; line-height: 1.45; }
</style>""",
    unsafe_allow_html=True,
)

# ───────────────────────── Couleurs vibrantes modernes ─────────────────────────
TEAL, TEAL_2 = "#4F46E5", "#06B6D4"
GOOD, WARN, BAD, INFO, MUTED = "#10B981", "#F59E0B", "#EF4444", "#3B82F6", "#64748B"


# ───────────────────────── Dictionnaire des problématiques métier ─────────────────────────
PROBLEMS_DATA = {
    "weather": {
        "title": "Prévision Météorologique & Pannes de Capteurs Barométriques",
        "tag": "Météorologie · 96 453 lignes",
        "icon": "🌤️",
        "accent": "#0284C7",
        "problem": (
            "Prédire les conditions météorologiques à partir de 96 453 relevés horaires sur 10 ans. "
            "Le défi critique est la défaillance matérielle des baromètres : 1 288 enregistrements présentent une "
            "pression atmosphérique aberrante de 0 millibar (capteur bloqué), en plus de 3 192 doublons stricts."
        ),
        "impact": "Un modèle entraîné sur ces zéros apprendrait de fausses lois physiques et fausserait toutes les prévisions de tempêtes.",
        "solution": "Exclusion systématique des pressions nulles, imputation robuste, analyse des tendances mensuelles et Random Forest pondéré.",
    },
    "covid": {
        "title": "Diagnostic Rapide Covid-19 en Laboratoire d'Urgence",
        "tag": "Covid-19 · 1 736 patients",
        "icon": "🧪",
        "accent": "#0891B2",
        "problem": (
            "Prédire l'infection Covid-19 dès l'admission à partir de 34 biomarqueurs sanguins et biochimiques. "
            "La difficulté majeure réside dans un taux massif de valeurs manquantes (les panels complets ne sont pas prescrits à tous les patients) "
            "et des corrélations non-linéaires complexes entre marqueurs immunitaires."
        ),
        "impact": "Un faux négatif peut provoquer un foyer épidémique intrahospitalier ; l'accuracy classique est trompeuse à cause du déséquilibre.",
        "solution": "Pipeline avec SimpleImputer médian/mode, réduction de dimension, validation croisée stratifiée et optimisation du F1 Macro.",
    },
    "parkinson": {
        "title": "Dépistage Précoce de la Maladie de Parkinson",
        "tag": "Neurologie · 584 patients",
        "icon": "🧠",
        "accent": "#7C3AED",
        "problem": (
            "Détecter la maladie neurodégénérative à partir de 15 biomarqueurs moteurs (tremblements, démarche) et cognitifs. "
            "Le piège central est le déséquilibre marqué des classes (70% malades / 30% sains) et le bruit cinématique (tremblement physiologique vs pathologique)."
        ),
        "impact": "Un classifieur naïf prédisant toujours la classe majoritaire obtiendrait 70% d'accuracy tout en ratant 100% des cas sains.",
        "solution": "Consensus de 6 détecteurs d'anomalies multivariés, calcul du SNR, et évaluation impérative par Balanced Accuracy.",
    },
    "alcohol": {
        "title": "Détection d'Alcoolémie sans Contact par Thermographie & Vision",
        "tag": "Biométrie · 390 observations",
        "icon": "🍺",
        "accent": "#D97706",
        "problem": (
            "Détecter l'état d'ébriété sans éthylotest invasif via la thermographie faciale infrarouge (températures faciales min/max) et l'oculométrie. "
            "Le problème identifié est la présence de contradictions directes d'étiquetage (sujets classés sains avec taux d'alcoolémie mesuré > 0) "
            "et la sensibilité thermique à l'environnement ambiant."
        ),
        "impact": "Les contradictions logiques perturbent les frontières de décision du modèle et génèrent des faux positifs inacceptables.",
        "solution": "Audit automatisé des contradictions, analyse des paires fortement corrélées (températures faciale/ambiante) et Random Forest.",
    },
    "heart": {
        "title": "Évaluation du Risque de Crise Cardiaque (Infarctus)",
        "tag": "Cardiologie · 1 190 dossiers cliniques",
        "icon": "❤️",
        "accent": "#E11D48",
        "problem": (
            "Pronostiquer le risque vital d'accident cardiovasculaire à partir de paramètres cliniques (angine, ECG, fréquence cardiaque, cholestérol). "
            "Le piège de ce dataset est la présence de fausses valeurs nulles masquées : des patients avec cholestérol = 0 ou pression artérielle = 0, "
            "ce qui est physiologiquement impossible chez un individu vivant."
        ),
        "impact": "Ces zéros cachés faussent les calculs statistiques et dégradent la calibration des probabilités de risque vital.",
        "solution": "Détection automatique des zéros aberrants masqués comme valeurs manquantes, imputation ciblée et évaluation ROC-AUC.",
    },
}


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


SEC_FR = {
    "missing_duplicates": "Manquants et doublons",
    "contradictions": "Contradictions",
    "data_quality": "Indicateurs de qualité",
    "outliers": "Outliers",
    "noise": "Bruit",
    "correlation": "Corrélations",
    "kde": "Distributions",
    "monthly_trends": "Tendances mensuelles",
    "model_phase": "Modèle",
    "interpretation": "Interprétation",
}


def plots_carousel(sections: list[str], title: str | None = None,
                   skip: tuple[str, ...] = ()) -> None:
    """Graphiques PNG des sections données, dans l'onglet qui les concerne : un à la fois, boutons ◀ / ▶."""
    for sec in sections:
        items = [p for p in dd.plots_for(rep) if p["section"] == sec
                 and "boxplot" not in p["path"].stem.lower()
                 and p["path"].stem not in skip]
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
                st.markdown(
                    f"<div style='text-align:center;padding-top:.4rem'><b>{idx + 1} / {n}</b> · {items[idx]['title']}</div>",
                    unsafe_allow_html=True,
                )
            with c3:
                stretch(st.button, "Suivant ▶", key=f"{k}_next", on_click=go, args=(1,))
        st.image(str(items[idx]["path"]), caption=items[idx]["title"])


def correlation_panel(rep: dict, key: str) -> None:
    """Matrice de corrélation interactive + paires + VIF (même logique que la vue météo)."""
    kind_label = st.radio(
        "Coefficient",
        ["Pearson (linéaire)", "Spearman (rangs)"],
        horizontal=True,
        key=f"corr_kind_{key}",
    )
    kind = "pearson" if kind_label.startswith("Pearson") else "spearman"
    long = dd.corr_matrix_df(rep, kind)
    if long.empty:
        st.info("Ce rapport ne contient pas de matrice de corrélation exploitable.")
        return

    n = long["a"].nunique()
    section_title(
        f"Matrice de corrélation · {kind_label}",
        f"{n} variables · survolez une cellule pour la valeur exacte",
    )
    st.altair_chart(ch.corr_heatmap(long))

    c1, c2 = st.columns([3, 2])
    with c1:
        section_title("Paires les plus corrélées", "triées par |r| décroissant")
        pairs = dd.corr_pairs_df(rep, top=30)
        if pairs.empty:
            st.info("Pas de paires disponibles.")
        else:
            stretch(
                st.dataframe,
                pairs.rename(columns={"paire": "Paire", "pearson": "Pearson",
                                      "spearman": "Spearman", "strength": "Force"}),
                hide_index=True,
            )
    with c2:
        section_title("Redondance (VIF)", "VIF ≥ 10 = colinéarité problématique")
        vif = dd.vif_df(rep, top=10)
        if vif.empty:
            st.info("VIF non calculé pour ce rapport.")
        else:
            vmax = max(20.0, float(vif["vif"].max()))
            stretch(
                st.dataframe,
                vif.rename(columns={"feature": "Variable", "vif": "VIF"}),
                hide_index=True,
                column_config={"VIF": st.column_config.ProgressColumn(
                    "VIF", min_value=0, max_value=vmax, format="%.1f")},
            )


def model_lab_panel(rep: dict, key: str) -> None:
    """Phase II modelling lab: KPIs, benchmark table, robustness deltas. Charts come from the database."""
    lab = rep.get("model_lab") or {}
    models = dd.lab_models_df(rep)
    if models.empty:
        st.info("Ce rapport ne contient pas l'étude de modélisation Phase II. "
                "Relancez `py -3 run_all_models.py` pour la générer.")
        return

    split = lab.get("split") or {}
    best_key = lab.get("best_model")
    best = models[models["key"] == best_key]
    best = best.iloc[0] if not best.empty else models.iloc[0]
    scaler = str(lab.get("scaler") or "—").capitalize()
    kpi_row([
        ("Meilleur modèle", str(best["model"]).split(" (")[0],
         f"macro F1 {best['f1']:.3f}" if pd.notna(best["f1"]) else "",
         "good"),
        ("Test accuracy", f"{best['test_acc'] * 100:.1f} %",
         f"train {best['train_acc'] * 100:.1f} %" if pd.notna(best["train_acc"]) else ""),
        ("Validation croisée", f"{best['cv'] * 100:.1f} %" if pd.notna(best["cv"]) else "n/a",
         f"écart-type {best['cv_std'] * 100:.1f} pts" if pd.notna(best["cv_std"]) else ""),
        ("ROC AUC", f"{best['auc']:.3f}" if pd.notna(best["auc"]) else "n/a",
         "séparabilité probabiliste"),
        ("Scaler retenu", f"{scaler}Scaler", f"{int(lab.get('cv_folds') or 0)} folds"),
        ("Split", f"{split.get('n_train', '?')} / {split.get('n_test', '?')}",
         f"{split.get('n_features', '?')} variables"),
    ])

    section_title("Benchmark des trois algorithmes",
                  "Tuning manuel par validation croisée · évaluation sur jeu de test indépendant")
    show = models.rename(columns={
        "model": "Modèle", "params": "Paramètres retenus", "test_acc": "Test",
        "train_acc": "Train", "cv": "CV", "bal_acc": "Balanced acc.", "precision": "Précision",
        "recall": "Rappel", "f1": "F1 macro", "auc": "AUC", "status": "Diagnostic",
    }).drop(columns=["key", "cv_std", "badge", "desc"])
    for c in ("Test", "Train", "CV", "Balanced acc.", "Précision", "Rappel", "F1 macro"):
        show[c] = (show[c] * 100).round(2)
    stretch(st.dataframe, show, hide_index=True)

    # Noise & outlier bar chart lives in the database (lab_noise_outlier) and
    # is shown by the model_lab carousel below. Keep only the delta table.
    stats = (lab.get("noise_outliers") or {}).get("stats") or {}
    deltas = dd.lab_deltas_df(rep)
    section_title("Écarts bruit et outliers vs baseline",
                  f"IQR : {stats.get('outliers_removed', 0)} ligne(s) retirée(s) · "
                  f"{stats.get('outliers_injected', 0)} outlier(s) injecté(s)")
    if deltas.empty:
        st.info("Aucune donnée de robustesse pour ce rapport.")
    else:
        stretch(st.dataframe, deltas.style.format("{:+.2f} pts", subset=deltas.columns[1:]),
                hide_index=True)
        st.caption("Lecture : points d'accuracy gagnés (+) ou perdus (−) par rapport à la baseline. "
                   "« Récupération lissage » mesure ce que le filtre gaussien rattrape après ajout de bruit. "
                   "Les graphiques (benchmark, sensibilité, bruit) sont dans les graphiques enregistrés ci-dessous.")

    insights = lab.get("insights") or []
    if insights:
        section_title("Synthèse de l'évaluation", "Constats calculés à partir des métriques")
        st.markdown("".join(
            f"<div class='tl-i'><div class='tl-t'>{i + 1}. </div><div class='tl-x'>{t}</div></div>"
            for i, t in enumerate(insights)), unsafe_allow_html=True)

    if lab.get("pickles"):
        pk = lab["pickles"]
        section_title("Artefacts sauvegardés (pickle)", "Réutilisables sans ré-entraînement")
        st.code("\n".join(f"{k}: {v}" for k, v in pk.items() if isinstance(v, str)) or "—",
                language="text")


reports = get_reports()

if not reports:
    st.error("Aucun rapport trouvé. Placez ce fichier à la racine du projet (à côté de data.db et des dossiers eda_output_*).")
    st.stop()
db = get_db()
ov = dd.overview_table(reports)
mt = dd.model_table(reports)

# ───────────────────────── sélection (accueil ↔ page dataset) ─────────────────────────
brand("Audit & Qualité des Données", title="Data Description", logo="DD")
sel = st.session_state.get("home_dataset")
if sel not in reports:
    sel = None
if sel is not None:
    st.sidebar.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
    if st.sidebar.button("← Retour aux datasets", use_container_width=True, key="back_home"):
        st.session_state["home_dataset"] = None
        st.rerun()
    st.sidebar.markdown(f"""
    <div style="background: rgba(79, 70, 229, 0.05); border: 1px solid rgba(79, 70, 229, 0.22); border-radius: 16px; padding: 1rem 1.1rem; margin-top: 1.2rem;">
      <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; font-weight: 700; color: #4F46E5; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.35rem;">Jeu de Données Actif</div>
      <div style="font-weight: 800; font-size: 1.05rem; color: #0F172A; margin-bottom: 0.45rem;">{dd.label(sel)}</div>
      <div style="font-size: 0.8rem; color: #64748B; line-height: 1.4;">Pour explorer un autre dataset, cliquez sur le bouton ci-dessus pour revenir à la sélection d'accueil.</div>
    </div>
    """, unsafe_allow_html=True)
st.sidebar.divider()
st.sidebar.caption("Sources lues : rapports JSON, data.db et scripts .py du projet.")
rep = reports[sel] if sel else None

# ───────────────────────── bandeau principal ─────────────────────────
total_rows = int(ov["Lignes"].sum())
if sel is None:
    hero(
        "Projet Machine Learning & Qualité des Données",
        "Analyse Météo & Datasets Médicaux Multi-Domaines",
        [
            ("Datasets audités", str(len(reports))),
            ("Observations totales", f"{total_rows:,}".replace(",", " ")),
            ("Visualisations générées", str(int(ov["Graphiques"].sum()))),
            ("Insights analytiques", str(int(ov["Insights"].sum()))),
        ],
    )

    # ── Section Problématique : Vue synthétique de l'architecture du projet ──
    st.markdown(
        """
        <div class="problem-box">
          <div class="problem-title">
            <span>🎯</span> Problématique Centrale du Projet : De la Donnée Bruitée aux Prédictions Fiables
          </div>
          <div class="problem-desc">
            Dans le monde réel, les données météorologiques et les dossiers médicaux ne sont jamais propres : capteurs barométriques
            bloqués à 0, contradictions d'étiquetage, valeurs biologiques impossibles (cholestérol = 0), et fort déséquilibre de classes.
            Ce projet implémente un <b>pipeline complet en 4 étapes</b> pour auditer, nettoyer et modéliser ces données de façon robuste.
          </div>
          <div class="pipeline-grid">
            <div class="pipeline-step">
              <span class="step-badge">1</span>
              <div class="step-t">⚠️ Ingestion & Audit Qualité</div>
              <div class="step-d">Détection des anomalies matérielles (1 288 pressions nulles), des contradictions logiques et des doublons stricts.</div>
            </div>
            <div class="pipeline-step">
              <span class="step-badge">2</span>
              <div class="step-t">🔍 Consensus 6 Détecteurs</div>
              <div class="step-d">Isolation Forest, LOF, One-Class SVM, Elliptic Env., DBSCAN & Mahalanobis pour isoler les vrais outliers multivariés.</div>
            </div>
            <div class="pipeline-step">
              <span class="step-badge">3</span>
              <div class="step-t">⚙️ Prétraitement Robuste</div>
              <div class="step-d">Pipeline Scikit-Learn (imputation médiane, One-Hot Encoding, scaling robuste et filtrage du bruit capteur SNR).</div>
            </div>
            <div class="pipeline-step">
              <span class="step-badge">4</span>
              <div class="step-t">🏆 Modèles ML Équilibrés</div>
              <div class="step-d">Random Forest avec pondération de classes, évalué par <b>Balanced Accuracy</b> et <b>F1 Macro</b> pour neutraliser les biais.</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_title("Choisir un jeu de données", "Cliquez sur « Explorer ce dataset » pour analyser les données, la qualité, les outliers et les modèles")
    cards = []
    for _, r in ov.iterrows():
        q = r["Score qualité"]
        cards.append({
            "key": r["key"],
            "name": dd.label(r["key"]),
            "subtitle": f"{int(r['Lignes']):,} lignes · {int(r['Colonnes'])} colonnes"
                        + ("" if pd.isna(q) else f" · score qualité {q:.0f}/100"),
        })
    render_dataset_cards(cards)

    tabs = dict(zip([0, 5, 2, 4], st.tabs([
        "📊 Vue d'ensemble",
        "🎯 Problématique & Défis",
        "🤖 Comparatif des modèles",
        "💻 Code & Architecture",
    ])))
else:
    _r = ov[ov["key"] == sel].iloc[0]
    hero("Jeu de Données Sélectionné", dd.label(sel),
         [("Observations", f"{int(_r['Lignes']):,}".replace(",", " ")),
          ("Variables", str(int(_r["Colonnes"]))),
          ("Variable Cible", str(_r["Cible"])),
          ("Graphiques EDA", str(int(_r["Graphiques"])))])

    # ── Spotlight Problématique du dataset ouvert ──
    if sel in PROBLEMS_DATA:
        pinfo = PROBLEMS_DATA[sel]
        st.markdown(
            f"""
            <div class="problem-box" style="border-left: 5px solid {pinfo['accent']};">
              <div class="problem-title">
                <span>{pinfo['icon']}</span> {pinfo['title']}
                <span class="chip" style="margin-left: auto; font-size: .75rem;"><b>{pinfo['tag']}</b></span>
              </div>
              <div class="problem-desc">
                <p><b>Le Défi Concret :</b> {pinfo['problem']}</p>
                <p><span style="color: var(--bad); font-weight: 700;">⚠️ Risque & Piège :</span> {pinfo['impact']}</p>
                <p><span style="color: var(--good); font-weight: 700;">✅ Approche Appliquée :</span> {pinfo['solution']}</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    tabs = dict(zip([7, 1, 8, 9, 10, 11, 3],
                    st.tabs(["📋 Données", "💎 Qualité", "🔬 Outliers & Bruit", "🔗 Corrélations",
                             "📈 Distributions", "🧪 Éval. Modèles", "⚙️ Prétraitement"])))

# ═════════════════════════ 1. VUE D'ENSEMBLE ═════════════════════════
if 0 in tabs:
    with tabs[0]:
        scored = ov.dropna(subset=["Score qualité"])
        kpis = [
            ("Jeux avec rapport", len(reports), "Météo + 4 datasets de santé", "info"),
            ("Qualité moyenne", f"{scored['Score qualité'].mean():.1f}/100", "4 datasets scorés (Phase I)", "good"),
        ]
        if not mt.empty:
            kpis.append(("Balanced accuracy moy.", f"{mt['Balanced accuracy'].mean():.3f}",
                         f"{len(mt)} modèles Random Forest évalués", "good"))
        if db:
            kpis.append(("Tables en base", db["counts"]["datasets"], f"{db['n_categories']} catégories · data.db {db['size_mb']} Mo"))
        kpi_row(kpis)

        section_title("Points d'attention prioritaires", "Anomalies et constats majeurs calculés à partir des rapports")
        attention_list(dd.attention_points(reports))

        section_title("Chronologie du projet & Jalons")
        html = "".join(
            f'<div class="tl-i"><div class="tl-d">{s["date"] or "—"}</div><div class="tl-t">{s["title"]}</div>'
            f'<div class="tl-x">{s["detail"]}</div></div>'
            for s in dd.timeline(reports, db)
        )
        st.markdown(f'<div class="tl">{html}</div>', unsafe_allow_html=True)

        section_title("Tableau comparatif des jeux de données")
        show = ov.drop(columns=["key"]).copy()
        stretch(st.dataframe, show, hide_index=True,
                column_config={
                    "Score qualité": st.column_config.ProgressColumn("Score qualité", min_value=0, max_value=100, format="%.1f"),
                    "Lignes": st.column_config.NumberColumn("Lignes", format="%d"),
                })

        c1, c2 = st.columns(2)
        with c1:
            section_title("Scores de qualité Phase I", "Score synthétique /100")
            sc = scored[["Dataset", "Score qualité", "Verdict"]]
            chart(alt.Chart(sc).mark_bar(cornerRadiusEnd=6).encode(
                x=alt.X("Score qualité:Q", scale=alt.Scale(domain=[0, 100]), title="Score de qualité"),
                y=alt.Y("Dataset:N", sort="-x", title=None),
                color=alt.Color("Verdict:N", scale=alt.Scale(domain=["good", "moderate", "poor"], range=[GOOD, WARN, BAD]),
                                legend=alt.Legend(orient="bottom", title="Verdict")),
                tooltip=["Dataset", "Score qualité", "Verdict"]), 240)
        with c2:
            section_title("Répartition des anomalies détectées (% du jeu)", "Doublons, contradictions, bruit, outliers")
            issues = ov.melt(id_vars="Dataset", value_vars=["Doublons %", "Contradictions %", "Bruit %", "Outliers %"],
                             var_name="Indicateur", value_name="Valeur").dropna()
            chart(alt.Chart(issues).mark_bar(cornerRadiusEnd=4).encode(
                y=alt.Y("Dataset:N", title=None),
                x=alt.X("Valeur:Q", title="Pourcentage concerné (%)"),
                yOffset="Indicateur:N",
                color=alt.Color("Indicateur:N", scale=alt.Scale(range=[BAD, WARN, INFO, TEAL_2]),
                                legend=alt.Legend(orient="bottom")),
                tooltip=["Dataset", "Indicateur", alt.Tooltip("Valeur:Q", format=".2f")]), 300)
        st.caption("Note : la météo n'a pas de section « data_quality » Phase I standardisée : bruit = part des variables testées jugées bruitées, "
                   "doublons issus de la section manquants/doublons.")

# ═════════════════════════ NOUVEAU ONGLET : PROBLÉMATIQUE & DÉFIS ═════════════════════════
if 5 in tabs:
    with tabs[5]:
        section_title("Les 5 Problématiques Métier & Défis Techniques", "Comprendre les pièges réels de chaque domaine")
        st.markdown(
            """
            Chaque dataset représente un cas d'étude concret où un modèle de Machine Learning naïf
            échouerait s'il était appliqué directement sans audit préalable de qualité.
            """
        )

        for p_key, p in PROBLEMS_DATA.items():
            st.markdown(
                f"""
                <div class="problem-card" style="margin-bottom: 1rem; border-left: 6px solid {p['accent']};">
                  <div class="problem-card-h">
                    <div class="problem-card-t">{p['icon']} {p['title']}</div>
                    <span class="chip"><b>{p['tag']}</b></span>
                  </div>
                  <div class="problem-card-b">
                    <p style="margin: .3rem 0;"><b>Le Problème Métier :</b> {p['problem']}</p>
                    <p style="margin: .3rem 0;"><span style="color: var(--bad); font-weight: 700;">⚠️ Risque & Piège Algorithmique :</span> {p['impact']}</p>
                    <p style="margin: .3rem 0;"><span style="color: var(--good); font-weight: 700;">✅ Solution Technique :</span> {p['solution']}</p>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        section_title("Tableau Synthétique des Défis de Qualité")
        summary_df = pd.DataFrame([
            {"Domaine": "🌤️ Météo (weatherHistory)", "Taille": "96 453 lignes", "Anomalie Critique": "1 288 pressions nulles (0 mbar)", "Risque": "Fausse relation physique", "Traitement": "Exclusion des zéros & Random Forest"},
            {"Domaine": "🧪 COVID-19 (Laboratoire)", "Taille": "1 736 patients", "Anomalie Critique": "Données manquantes massives", "Risque": "Biais d'échantillonnage", "Traitement": "Imputation médiane & F1 Macro"},
            {"Domaine": "🧠 Maladie de Parkinson", "Taille": "584 patients", "Anomalie Critique": "Déséquilibre de classe fort", "Risque": "Accuracy trompeuse", "Traitement": "Balanced Accuracy & K-Fold stratifié"},
            {"Domaine": "🍺 Détection d'Alcoolémie", "Taille": "390 observations", "Anomalie Critique": "Contradictions logiques (classe 0, alcool > 0)", "Risque": "Frontière de décision corrompue", "Traitement": "Audit des contradictions & filtre bruit"},
            {"Domaine": "❤️ Crise Cardiaque", "Taille": "1 190 patients", "Anomalie Critique": "Cholestérol = 0 (zéros masqués)", "Risque": "Valeur biologique absurde", "Traitement": "Imputation ciblée & ROC-AUC"},
        ])
        stretch(st.dataframe, summary_df, hide_index=True)

# ═════════════════════════ 2. QUALITÉ ═════════════════════════
if 1 in tabs:
    with tabs[1]:
        st.subheader(dd.label(sel))
        row = ov[ov["key"] == sel].iloc[0]
        q = row["Score qualité"]
        kpi_row([
            ("Dimensions", f"{int(row['Lignes']):,}".replace(",", " ") + f" × {int(row['Colonnes'])}", f"Cible : {row['Cible']}"),
            ("Score de qualité", "n/a" if pd.isna(q) else f"{q:.1f}/100", row["Verdict"] if isinstance(row["Verdict"], str) else "section absente",
             {"good": "good", "moderate": "warn", "poor": "bad"}.get(row["Verdict"], "neutral")),
            ("Taux de doublons", "n/a" if pd.isna(row["Doublons %"]) else f"{row['Doublons %']:.2f} %", None,
             "bad" if (pd.notna(row["Doublons %"]) and row["Doublons %"] >= 5) else "good"),
            ("Variables bruitées", "n/a" if pd.isna(row["Bruit %"]) else f"{row['Bruit %']:.0f} %", "rapport signal/bruit dégradé", "warn"),
        ])

        ind = dd.indicators_df(rep)
        if not ind.empty:
            section_title("Indicateurs de qualité détaillés", "Part du dataset impactée par chaque type de défaut")
            chart(alt.Chart(ind).mark_bar(cornerRadiusEnd=6).encode(
                x=alt.X("issue_percent:Q", title="% concerné"),
                y=alt.Y("indicator:N", sort="-x", title=None),
                color=alt.Color("status:N", scale=alt.Scale(domain=["good", "warning", "critical"], range=[GOOD, WARN, BAD]),
                                legend=alt.Legend(orient="bottom", title="Statut")),
                tooltip=["indicator", "issue_percent", "status", "detail"]), max(180, 36 * len(ind)))
            stretch(st.dataframe, ind[["indicator", "issue_percent", "status", "detail"]], hide_index=True)

        nz = dd.noise_df(rep)
        c1, c2 = st.columns(2)
        with c1:
            section_title("Bruit par variable (Noise-to-Signal)", "Plus bas = signal plus propre et stable")
            if nz.empty:
                st.info("Pas de données de bruit disponibles.")
            else:
                nz = nz.sort_values("noise_to_signal", ascending=False).head(15)
                chart(alt.Chart(nz).mark_bar(cornerRadiusEnd=4).encode(
                    x=alt.X("noise_to_signal:Q", title="Ratio Bruit / Signal"),
                    y=alt.Y("feature:N", sort="-x", title=None),
                    color=alt.Color("verdict:N", scale=alt.Scale(domain=["negligible", "low", "moderate", "high"],
                                                                  range=[GOOD, TEAL_2, WARN, BAD]),
                                    legend=alt.Legend(orient="bottom", title="Verdict")),
                    tooltip=["feature", alt.Tooltip("noise_to_signal:Q", format=".3f"), "snr_db", "verdict"]),
                    max(200, 26 * len(nz)))
        with c2:
            section_title("Corrélations linéaires les plus fortes", "Coefficients de Pearson")
            cp = dd.corr_pairs_df(rep)
            if cp.empty:
                st.info("Pas de corrélations disponibles.")
            else:
                chart(alt.Chart(cp).mark_bar(cornerRadiusEnd=4).encode(
                    x=alt.X("pearson:Q", scale=alt.Scale(domain=[-1, 1]), title="Pearson"),
                    y=alt.Y("paire:N", sort=alt.EncodingSortField("pearson", order="descending"), title=None),
                    color=alt.Color("pearson:Q", scale=alt.Scale(domain=[-1, 0, 1], range=[INFO, "#CBD5E1", BAD]), legend=None),
                    tooltip=["paire", alt.Tooltip("pearson:Q", format=".3f"), alt.Tooltip("spearman:Q", format=".3f"), "strength"]),
                    max(200, 30 * len(cp)))

        plots_carousel(["data_quality", "missing_duplicates", "contradictions"])

# ═════════════════════════ 3. MODÈLES ═════════════════════════
if 2 in tabs:
    with tabs[2]:
        section_title("Comparatif des performances Machine Learning", "Random Forest pondéré · Évaluation test hold-out + validation croisée")
        if mt.empty:
            st.info("Aucune évaluation de modèle trouvée dans les rapports.")
        else:
            long = mt.melt(id_vars="Dataset", value_vars=["Balanced accuracy", "F1 macro", "CV balanced acc."],
                           var_name="Métrique", value_name="Score")
            chart(alt.Chart(long).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("Dataset:N", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1]), title="Score de performance"),
                xOffset="Métrique:N",
                color=alt.Color("Métrique:N", scale=alt.Scale(range=[TEAL, TEAL_2, "#818CF8"]), legend=alt.Legend(orient="bottom")),
                tooltip=["Dataset", "Métrique", alt.Tooltip("Score:Q", format=".3f")]), 300)
            stretch(st.dataframe, mt.drop(columns=["key"]), hide_index=True)

# ═════════════════════════ 3c. ÉVALUATION DES MODÈLES (Phase II) ═════════════════════════
if 11 in tabs:
    with tabs[11]:
        model_lab_panel(rep, sel)
        plots_carousel(["model_lab"], "Graphiques · Évaluation des modèles")
        plots_carousel(["model_phase"], "Graphiques · Random Forest (importance des variables)")

# ═════════════════════════ 4. PRÉTRAITEMENT ═════════════════════════
if 3 in tabs:
    with tabs[3]:
        st.subheader(dd.label(sel))
        plan = dd.plan_df(rep)
        if plan.empty:
            st.info("Ce rapport n'a pas de section d'interprétation explicite.")
            pre = rep.get("model_phase", {}).get("preprocessing")
            if pre:
                section_title("Pipeline de prétraitement du modèle météo")
                st.write("**Pipeline numérique** : " + " → ".join(pre.get("numerical_pipeline", [])))
                st.write("**Pipeline catégoriel** : " + " → ".join(pre.get("categorical_pipeline", [])))
                st.caption(f"{len(pre.get('numerical_features', []))} variables numériques, "
                           f"{len(pre.get('categorical_features', []))} catégorielle(s), cible : {pre.get('target_feature')}.")
        else:
            n_app = int((plan["status"] == "applied").sum())
            n_rec = int((plan["status"] == "recommended").sum())
            kpi_row([("Étapes appliquées", n_app, "Déjà actives dans le pipeline", "good"),
                     ("Étapes recommandées", n_rec, "Recommandations futures", "warn")])
            c1, c2 = st.columns(2)
            with c1:
                section_title("Plan de prétraitement appliqué")
                stretch(st.dataframe, plan.rename(columns={"step": "Étape", "status": "Statut", "reason": "Constat"}),
                        hide_index=True)
            with c2:
                section_title("Recommandations d'amélioration")
                rec = dd.recommendations_df(rep)
                stretch(st.dataframe, rec.rename(columns={"priority": "Priorité", "area": "Zone", "finding": "Constat",
                                                          "action": "Action"}), hide_index=True)

        plots_carousel(["interpretation"])

# ═════════════════════════ 5. CODE & ARCHITECTURE ═════════════════════════
if 4 in tabs:
    with tabs[4]:
        code = dd.code_inventory()
        kpi_row([
            ("Scripts Python", len(code), "Composants du projet"),
            ("Lignes de code totales", f"{int(code['Lignes'].sum()):,}".replace(",", " "), "Architecture MVC"),
            ("Module le plus dense", code.iloc[0]["Fichier"], f"{int(code.iloc[0]['Lignes'])} lignes"),
        ])
        c1, c2 = st.columns([3, 2])
        with c1:
            section_title("Volume de code par module", "Couleur = Couche logicielle (Model, View, Stockage)")
            chart(alt.Chart(code).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("Lignes:Q", title="Nombre de lignes"), y=alt.Y("Fichier:N", sort="-x", title=None),
                color=alt.Color("Couche:N", scale=alt.Scale(range=[TEAL, TEAL_2, INFO, WARN, MUTED]),
                                legend=alt.Legend(orient="bottom", title="Couche")),
                tooltip=["Fichier", "Couche", "Lignes", "Rôle"]), max(260, 26 * len(code)))
        with c2:
            section_title("Répartition par couche")
            by_layer = code.groupby("Couche", as_index=False)["Lignes"].sum()
            chart(alt.Chart(by_layer).mark_arc(innerRadius=60).encode(
                theta="Lignes:Q",
                color=alt.Color("Couche:N", scale=alt.Scale(range=[TEAL, TEAL_2, INFO, WARN, MUTED]),
                                legend=alt.Legend(orient="bottom")),
                tooltip=["Couche", "Lignes"]), 280)
        stretch(st.dataframe, code, hide_index=True)

        if db:
            section_title("Base relationnelle SQLite data.db", f"Taille : {db['size_mb']} Mo")
            c = db["counts"]
            kpi_row([("Rapports stockés", c["reports"]), ("Sections", c["sections"]), ("Insights", c["insights"]),
                     ("Graphiques BLOB", c["plots"]), ("Catalogues de données", c["datasets"])])
            cats = db["categories"]
            chart(alt.Chart(cats).mark_bar(cornerRadiusEnd=4, color=TEAL).encode(
                x=alt.X("Tables:Q", title="Nombre de tables"), y=alt.Y("Catégorie:N", sort="-x", title=None),
                tooltip=["Catégorie", "Tables", "Lignes"]), max(220, 28 * len(cats)))

# ═════════════════════════ 0b. DONNÉES ═════════════════════════
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
        correlation_panel(rep, sel)
        plots_carousel(["correlation"], "Graphiques · Corrélations",
                       skip=("correlation_pearson", "correlation_spearman"))
if 10 in tabs:
    with tabs[10]:
        plots_carousel(["kde"], "Graphiques · Distributions")
        plots_carousel(["monthly_trends"], "Graphiques · Tendances mensuelles")
