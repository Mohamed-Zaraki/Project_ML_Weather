"""dataset_workplan_view.py : plan de travail propre à chaque dataset du dashboard.

Contrairement au plan global de la page d'accueil, cette checklist est calculée
à partir des artefacts réellement présents pour LE dataset ouvert : catalogue SQLite,
rapport EDA, score qualité, outliers/bruit, corrélations, graphiques, modèle et
dictionnaire des colonnes. L'état est sessionnel (aucune écriture sur disque).
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

import dashboard_data as dd
from theme import esc, kpi_row, section_title
from workplan_view import CSS as WORKPLAN_CSS, PHASES

ROOT = Path(__file__).resolve().parent
METRICS_ELEC = ROOT / "evaluation_electricalload" / "metrics_summary.csv"


def _stretch(fn, *args, **kwargs):
    try:
        return fn(*args, width="stretch", **kwargs)
    except TypeError:
        return fn(*args, use_container_width=True, **kwargs)


def _has_evaluation(key: str, rep: dict) -> bool:
    if key == "electricalload":
        return METRICS_ELEC.exists()
    if rep.get("model_lab") or (rep.get("model_phase") or {}).get("evaluation"):
        return True
    return bool(rep.get("regression_lab"))


def _quality_text(rep: dict) -> str:
    if dd.is_electrical(rep):
        q = dd.electrical_quality(rep)
        return f"score {q['score']}/100 ({q['verdict']}), dérivé du rapport EDA"
    dq = rep.get("data_quality") or {}
    if dq.get("quality_score") is not None:
        return f"score {dq['quality_score']}/100 ({dq.get('quality_label', 'n/a')})"
    if rep.get("missing_duplicates") or rep.get("noise") or rep.get("outliers"):
        return "complétude, doublons et bruit audités (rapport sans score /100 standardisé)"
    return "score indisponible dans ce rapport"


def _build_tasks(key: str, rep: dict) -> list[dict]:
    meta = rep.get("meta") or {}
    n_rows = meta.get("n_rows", "?")
    n_cols = meta.get("n_cols", "?")
    n_plots = len(rep.get("plots") or [])
    o = rep.get("outliers") or {}
    n_det = len(o.get("methods") or o.get("machine_learning_outliers", {}).get("methods_used") or [])
    if dd.is_electrical(rep):
        n_det = len(o.get("methods") or {})
    db_id = dd.resolve_dataset_id(dd.ROOT / "data.db", key, rep)
    n_features = len((rep.get("noise") or {}).get("features") or [])
    corr = rep.get("correlation") or {}
    n_corr = len(corr.get("top_pairs") or corr.get("pairs") or [])
    if dd.is_electrical(rep) and corr.get("pearson_ma4_demand") is not None:
        n_corr = 1
    robustness = bool((rep.get("model_lab") or {}).get("noise_outliers"))
    if dd.is_electrical(rep):
        robustness = n_det >= 4 and n_features > 0
    if key == "weather":
        robustness = bool(rep.get("noise") and rep.get("outliers"))

    specific = {
        "weather": "Série horaire : split temporel 75 % passé / 25 % récent, target `Precip Type` très déséquilibrée.",
        "covid": "Biologie clinique : 54 % de manquants sur CK et valeurs « Missing » cachées dans EOT/BAT.",
        "parkinson": "Marqueurs LCR + IRM : exclure `Subject_id` et splitter par sujet pour éviter les fuites.",
        "alcohol": "Capteurs température/yeux : distribution de `ALCOHOL` très asymétrique (transformation log).",
        "heart": "Données cliniques : les `cholesterol = 0` sont des valeurs manquantes masquées à imputer.",
        "electricalload": "Série de 36 888 relevés horaires : split chronologique et pics de demande à préserver.",
    }.get(key, "")

    return [
        {"id": "catalogue", "phase": 0, "label": f"Importer {key} dans data.db",
         "detail": f"Catalogue SQLite et aperçu de 50 lignes (id {db_id or 'introuvable'}). "
                   f"Volume catalogué : {n_rows} lignes × {n_cols} colonnes.",
         "default": db_id is not None},
        {"id": "schema", "phase": 0, "label": "Documenter le schéma et les sources",
         "detail": f"Colonnes, types et catégorie enregistrés ; fichier source : "
                   f"{str(meta.get('csv_path') or meta.get('source_files') or 'datasets_extracted/').split('\\')[-1]}.",
         "default": bool(n_cols)},
        {"id": "quality", "phase": 1, "label": "Audit qualité et score /100",
         "detail": _quality_text(rep) + ". Complétude, doublons, contradictions, bruit et outliers.",
         "default": bool((rep.get("data_quality") or {}).get("quality_score")
                         or dd.is_electrical(rep)
                         or (rep.get("missing_duplicates") and (rep.get("noise") or rep.get("outliers"))))},
        {"id": "outliers", "phase": 1, "label": "Analyser les outliers et le bruit",
         "detail": (f"{n_det} détecteurs, {n_features} variables testées pour le bruit. "
                    + (specific if dd.is_electrical(rep) else "")),
         "default": bool(rep.get("outliers") or rep.get("noise"))},
        {"id": "corr", "phase": 1, "label": "Étudier les corrélations et redondances",
         "detail": f"{n_corr} paire(s) / analyse VIF "
                   + ("; lien MA_X-4 ↔ DEMAND et jours à fort écart."
                      if dd.is_electrical(rep) else "conservées dans le rapport EDA."),
         "default": bool(corr)},
        {"id": "eda", "phase": 1, "label": "Réaliser l'EDA et enregistrer les graphiques",
         "detail": f"{n_plots} graphiques PNG présents dans le rapport {key}.",
         "default": n_plots >= 5},
        {"id": "model", "phase": 2, "label": "Entraîner et évaluer le modèle",
         "detail": ("R²/RMSE des 3 régresseurs dans evaluation_electricalload/metrics_summary.csv."
                    if key == "electricalload" else
                    "Validation croisée, métriques test et courbes/ matrices sauvegardées dans le rapport."),
         "default": _has_evaluation(key, rep)},
        {"id": "robust", "phase": 2, "label": "Tester la robustesse bruit / outliers",
         "detail": ("Stress tests baseline / IQR / bruit injecté / outliers injectés."
                    if rep.get("model_lab") else
                    "Analyse du bruit NSR et du consensus de détecteurs comme diagnostic de robustesse."),
         "default": robustness},
        {"id": "dictionary", "phase": 3, "label": "Rédiger l'interprétation de chaque colonne",
         "detail": f"Dictionnaire métier des {n_cols} variables : rôle, signification, statistiques et recommandation.",
         "default": True},
        {"id": "dashboard", "phase": 3, "label": "Intégrer le dataset au dashboard",
         "detail": "Carte d'accueil, onglets qualité / outliers / corrélations / modèles et suivi du plan.",
         "default": True},
    ]


def render_dataset_workplan(key: str, rep: dict) -> None:
    """Checklist interactive limitée au dataset affiché."""
    tasks = _build_tasks(key, rep)
    defaults = {t["id"]: bool(t["default"]) for t in tasks}
    for task_id, value in defaults.items():
        st.session_state.setdefault(f"dwp_{key}_{task_id}", value)

    done = [t for t in tasks if st.session_state[f"dwp_{key}_{t['id']}"]]
    pct = round(100 * len(done) / len(tasks))
    nxt = next((t for t in tasks if not st.session_state[f"dwp_{key}_{t['id']}"]), None)

    st.markdown(WORKPLAN_CSS, unsafe_allow_html=True)
    section_title("Plan de travail du dataset",
                  f"{dd.label(key)} · {len(done)} / {len(tasks)} étapes validées")
    kpi_row([
        ("Plan de travail", f"{pct} %", f"{dd.label(key)}", "good" if pct == 100 else "info"),
        ("Prochaine étape", nxt["label"] if nxt else "Plan terminé 🎉",
         nxt["detail"] if nxt else "Toutes les étapes sont validées", "warn" if nxt else "good"),
    ])
    try:
        st.progress(pct / 100, text=f"{pct} % du plan « {key} » réalisé")
    except TypeError:
        st.progress(pct / 100)

    cols = st.columns(2, gap="medium")
    for idx, phase in enumerate(PHASES):
        phase_tasks = [t for t in tasks if t["phase"] == idx]
        n_done = sum(st.session_state[f"dwp_{key}_{t['id']}"] for t in phase_tasks)
        with cols[idx % 2]:
            st.markdown(
                f'<div class="wp-phase"><span class="wp-badge">{phase["icon"]}</span>'
                f'<b>{esc(phase["name"])}</b><span class="wp-count">{n_done}/{len(phase_tasks)}</span></div>',
                unsafe_allow_html=True,
            )
            for task in phase_tasks:
                st.checkbox(task["label"], key=f"dwp_{key}_{task['id']}", help=task["detail"])
                cls = "wp-detail wp-done" if st.session_state[f"dwp_{key}_{task['id']}"] else "wp-detail"
                st.markdown(f'<div class="{cls}">{esc(task["detail"])}</div>', unsafe_allow_html=True)

    def reset() -> None:
        for task_id, value in defaults.items():
            st.session_state[f"dwp_{key}_{task_id}"] = value

    _stretch(st.button, "↺ Réinitialiser le plan du dataset", key=f"dwp_reset_{key}", on_click=reset)
