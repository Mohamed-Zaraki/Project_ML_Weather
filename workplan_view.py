"""workplan_view.py : checklist interactive du plan de travail sur la page d'accueil.

Les cases sont cochées au départ d'après les artefacts réellement présents (rapports JSON,
data.db, modèles) ; l'utilisateur peut les ajuster pendant la session. Aucune donnée n'est
écrite : l'état vit dans st.session_state et peut être réinitialisé à tout moment.
"""
from __future__ import annotations

import streamlit as st

from theme import esc, kpi_row

# (id, libellé, détail, phase) — l'ordre des phases est celui du livrable.
PHASES = [
    {"name": "Phase 0 · Cadrage & ingestion", "icon": "📥", "color": "#06B6D4"},
    {"name": "Phase I · Audit qualité & EDA", "icon": "🔍", "color": "#4F46E5"},
    {"name": "Phase II · Modélisation", "icon": "🤖", "color": "#7C3AED"},
    {"name": "Phase III · Restitution", "icon": "📊", "color": "#10B981"},
]

TASKS = [
    {"id": "catalogue", "phase": 0, "label": "Importer les datasets dans data.db",
     "detail": "Catalogue SQLite, aperçus et schémas (populate_datasets.py, dataset_overview.py).",
     "default": lambda reports, db: db is not None},
    {"id": "sources", "phase": 0, "label": "Documenter les sources et les schémas",
     "detail": "Fiches dataset : colonnes, types, catégories et volume de chaque jeu.",
     "default": lambda reports, db: len(reports) >= 5},
    {"id": "quality", "phase": 1, "label": "Calculer les indicateurs de qualité /100",
     "detail": "Complétude, doublons, contradictions, bruit et outliers par consensus.",
     "default": lambda reports, db: sum(bool(r.get("data_quality") or r.get("quality")) for r in reports.values()) >= 5},
    {"id": "eda", "phase": 1, "label": "Réaliser l'EDA des 6 domaines",
     "detail": "Météo, COVID-19, Parkinson, alcool, crise cardiaque et charge électrique.",
     "default": lambda reports, db: "electricalload" in reports},
    {"id": "outliers", "phase": 1, "label": "Analyser les outliers et le bruit",
     "detail": "Détecteurs multivariés, NSR/SNR, graphiques et interprétations enregistrés.",
     "default": lambda reports, db: any(r.get("outliers") or r.get("noise") for r in reports.values())},
    {"id": "classification", "phase": 2, "label": "Modèles de classification (RF / KNN / SVM)",
     "detail": "Tuning par validation croisée, F1 macro, balanced accuracy et tests de robustesse.",
     "default": lambda reports, db: any(r.get("model_lab") or (r.get("model_phase") or {}).get("evaluation")
                                       for r in reports.values())},
    {"id": "regression", "phase": 2, "label": "Modèles de régression (météo + DEMAND)",
     "detail": "R²/RMSE, courbes réelles vs prédites et distribution des résidus.",
     "default": lambda reports, db: bool((reports.get("weather") or {}).get("regression_lab"))
                                    and "electricalload" in reports},
    {"id": "dashboard", "phase": 3, "label": "Construire le dashboard Streamlit",
     "detail": "Page d'accueil, fiches dataset et onglets qualité / outliers / modèles.",
     "default": lambda reports, db: True},
    {"id": "validation", "phase": 3, "label": "Synthèse finale et validation",
     "detail": "Relecture des métriques, rapport de synthèse et préparation de la présentation.",
     "default": lambda reports, db: False},
]

CSS = """<style>
.wp-phase { display:flex; align-items:center; gap:.55rem; margin:.9rem 0 .35rem; font-size:.92rem; }
.wp-badge { display:inline-flex; align-items:center; justify-content:center; width:1.6rem; height:1.6rem;
  border-radius:9px; background:rgba(79,70,229,.08); font-size:.95rem; }
.wp-count { margin-left:auto; font-family:'JetBrains Mono', monospace; font-size:.7rem; font-weight:700;
  color:#64748B; background:#F1F5F9; padding:.15rem .45rem; border-radius:20px; }
.wp-detail { font-size:.76rem; color:#64748B; line-height:1.3; margin:-.45rem 0 .55rem 1.85rem; }
.wp-done { text-decoration:line-through; opacity:.65; }
</style>"""


def _stretch(fn, *args, **kwargs):
    try:
        return fn(*args, width="stretch", **kwargs)
    except TypeError:
        return fn(*args, use_container_width=True, **kwargs)


def render_workplan(reports: dict, db: dict | None) -> None:
    """Checklist interactive : KPI d'avancement, barre de progression, tâches par phase et reset."""
    defaults = {t["id"]: bool(t["default"](reports, db)) for t in TASKS}
    for task_id, value in defaults.items():
        st.session_state.setdefault(f"wp_{task_id}", value)

    done = [t for t in TASKS if st.session_state[f"wp_{t['id']}"]]
    pct = round(100 * len(done) / len(TASKS))
    next_task = next((t for t in TASKS if not st.session_state[f"wp_{t['id']}"]), None)

    st.markdown(CSS, unsafe_allow_html=True)
    kpi_row([
        ("Avancement du plan", f"{pct} %", f"{len(done)} / {len(TASKS)} étapes cochées",
         "good" if pct == 100 else "info"),
        ("Prochaine étape", next_task["label"] if next_task else "Plan terminé 🎉",
         next_task["detail"] if next_task else "Toutes les étapes sont validées",
         "warn" if next_task else "good"),
    ])
    try:
        st.progress(pct / 100, text=f"{pct} % du plan de travail réalisé")
    except TypeError:
        st.progress(pct / 100)

    cols = st.columns(2, gap="medium")
    for idx, phase in enumerate(PHASES):
        tasks = [t for t in TASKS if t["phase"] == idx]
        n_done = sum(st.session_state[f"wp_{t['id']}"] for t in tasks)
        with cols[idx % 2]:
            st.markdown(
                f'<div class="wp-phase"><span class="wp-badge">{phase["icon"]}</span>'
                f'<b>{esc(phase["name"])}</b><span class="wp-count">{n_done}/{len(tasks)}</span></div>',
                unsafe_allow_html=True,
            )
            for task in tasks:
                st.checkbox(task["label"], key=f"wp_{task['id']}", help=task["detail"])
                cls = "wp-detail wp-done" if st.session_state[f"wp_{task['id']}"] else "wp-detail"
                st.markdown(f'<div class="{cls}">{esc(task["detail"])}</div>', unsafe_allow_html=True)

    def reset() -> None:
        # Exécuté avant l'instantiation des checkboxes : Streamlit interdit de modifier
        # une clé de widget après sa création dans le même run.
        for task_id, value in defaults.items():
            st.session_state[f"wp_{task_id}"] = value

    _stretch(st.button, "↺ Réinitialiser le plan", key="wp_reset", on_click=reset)
