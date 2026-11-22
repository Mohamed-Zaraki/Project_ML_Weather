"""dashboard_data.py : couche de données du tableau de bord du projet.

Aucune dépendance à Streamlit : tout est calculé à partir des fichiers du projet
(rapports JSON, data.db, scripts .py), donc le tableau de bord reflète toujours
l'état réel du dossier. Utilisé par project_dashboard.py.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

# clé -> (libellé affiché, fichier rapport)
REPORT_FILES: dict[str, tuple[str, Path]] = {
    "weather": ("Météo (weatherHistory)", ROOT / "latest_weather_report.json"),
    "covid": ("COVID-19 · Laboratoire", ROOT / "eda_output_covid19" / "report.json"),
    "parkinson": ("Maladie de Parkinson", ROOT / "eda_output_parkinson" / "report.json"),
    "alcohol": ("Détection d'alcool", ROOT / "eda_output_alcohol" / "report.json"),
    "heart": ("Crise cardiaque", ROOT / "eda_output_heart" / "report.json"),
    "electricalload": ("Charge électrique", ROOT / "eda_output_electricalload" / "report.json"),
}

# Rôle de chaque script (d'après README.md et les docstrings du projet)
FILE_ROLES: dict[str, tuple[str, str]] = {
    "weather_model.py": ("Model", "Analyse complète de weatherHistory.csv + Random Forest"),
    "multi_eda_model.py": ("Model", "Pipeline EDA + ML des datasets COVID, Parkinson, alcool et cœur"),
    "electrical_load_eda.py": ("Model", "EDA 7 sections de la charge électrique (qualité, tendances, outliers, bruit, corrélation, distributions, prétraitement)"),
    "eval_electricalload.py": ("Model", "Évaluation R²/RMSE de 3 régresseurs sur DEMAND + 6 graphiques"),
    "electrical_load_pipeline.py": ("Pipeline", "Intègre ElectricalLoad dans run_all_models.py (garde-fou aperçus 50 lignes)"),
    "quality.py": ("Model", "Règles de qualité partagées (plages, seuils de bruit, nettoyage)"),
    "quality_multi.py": ("Model", "Indicateurs de qualité Phase I pour les datasets multiples"),
    "data.py": ("Stockage", "Couche SQLite : datasets, rapports, insights, graphiques (BLOB)"),
    "weather_db.py": ("Stockage", "Alias de compatibilité de DataDB (rapports météo)"),
    "populate_datasets.py": ("Stockage", "Import des fichiers de données dans data.db"),
    "export_dataset_previews.py": ("Stockage", "Recrée les chemins datasets_extracted/ depuis les aperçus 50 lignes de data.db"),
    "export_weather_json.py": ("Stockage", "Export du dernier rapport météo en JSON"),
    "run_all_models.py": ("Pipeline", "Exécute le pipeline EDA + ML sur les 5 datasets du projet"),
    "weather_view.py": ("View", "Vue en lecture seule d'un rapport (un onglet par section)"),
    "dataset_overview.py": ("View", "Vue d'ensemble : import et exploration des datasets"),
    "column_dictionary.py": ("View", "Dictionnaire métier et interprétation de chaque colonne, par dataset"),
    "dataset_workplan_view.py": ("View", "Plan de travail interactif propre à chaque dataset"),
    "outlier_noise_interpretation.py": ("View", "Guide de lecture et synthèse interprétative des outliers et du bruit"),
    "outlier_view.py": ("View", "Onglet outliers/bruit : KPI, boxplots, détecteurs et bruit par variable"),
    "electrical_load_view.py": ("View", "Vues charge électrique : outliers/bruit, corrélation et évaluation R²/RMSE"),
    "workplan_view.py": ("View", "Checklist interactive du plan de travail sur la page d'accueil"),
    "app.py": ("View", "Application Streamlit : navigation et routage"),
    "theme.py": ("View", "Habillage visuel partagé (CSS, cartes KPI)"),
}


# ───────────────────────── chargement ─────────────────────────
def load_reports(root: Path = ROOT) -> dict[str, dict]:
    """Charge les rapports JSON présents ; ignore silencieusement ceux qui manquent."""
    out: dict[str, dict] = {}
    for key, (_, path) in REPORT_FILES.items():
        p = root / path.relative_to(ROOT)
        if p.exists():
            out[key] = json.loads(p.read_text(encoding="utf-8"))
    return out


def label(key: str) -> str:
    return REPORT_FILES[key][0]


def _sections(rep: dict):
    for name, sec in rep.items():
        if isinstance(sec, dict):
            yield name, sec


def n_insights(rep: dict) -> int:
    return sum(len(sec.get("insights", []) or []) for _, sec in _sections(rep))


def _indicator(rep: dict, name: str):
    for ind in rep.get("data_quality", {}).get("indicators", []) or []:
        if ind.get("indicator") == name:
            return ind.get("value")
    return None


def target_name(rep: dict) -> str | None:
    meta = rep.get("meta", {})
    return (meta.get("target_column")
            or rep.get("model_phase", {}).get("target_balance", {}).get("target_feature"))


# ───────────────────── charge électrique (schéma EDA spécifique) ─────────────────────
# Le rapport eda_output_electricalload/report.json n'utilise pas la section
# « data_quality » standard des 4 datasets Phase I. On dérive ici les mêmes
# indicateurs avec la pondération de multi_eda_model.py.
ELECTRICAL_KEY = "electricalload"
_VERDICT_ALIASES = {
    "clean": "negligible", "negligible": "negligible", "négligeable": "negligible",
    "low": "low", "low noise": "low", "faible": "low",
    "moderate": "moderate", "moderate noise": "moderate", "modéré": "moderate",
    "high": "high", "high noise": "high", "élevé": "high",
}


def is_electrical(rep_or_key: dict | str) -> bool:
    if isinstance(rep_or_key, str):
        return rep_or_key == ELECTRICAL_KEY
    meta = rep_or_key.get("meta", {}) or {}
    return (meta.get("dataset") == ELECTRICAL_KEY
            or "quality" in rep_or_key and "data_quality" not in rep_or_key
            and "pearson_ma4_demand" in (rep_or_key.get("correlation") or {}))


def _electrical_verdict(raw: str | None) -> str:
    return _VERDICT_ALIASES.get(str(raw or "").strip().lower(), "negligible")


def electrical_quality(rep: dict) -> dict:
    """Score /100 et indicateurs de la charge électrique, dans le format du dashboard."""
    q = rep.get("quality", {}) or {}
    meta = rep.get("meta", {}) or {}
    n_rows = int(meta.get("n_rows") or 0)
    n_cols = int(meta.get("n_cols") or 0)

    missing_cells = sum((q.get("missing_train") or {}).values()) + sum((q.get("missing_test") or {}).values())
    missing_pct = 100 * missing_cells / max(1, n_rows * n_cols)
    dup_cells = sum((q.get("duplicates") or {}).values())
    dup_pct = 100 * dup_cells / max(1, n_rows)

    features = (rep.get("noise") or {}).get("features") or []
    noisy_n = sum(1 for f in features if _electrical_verdict(f.get("verdict")) in ("moderate", "high"))
    noise_pct = 100 * noisy_n / len(features) if features else 0.0

    o = rep.get("outliers") or {}
    cons = o.get("consensus_outliers") or {}
    out_pct = float(cons.get("percent") or 0.0)
    out_n = int(cons.get("count") or 0)

    # Même pondération que multi_eda_model.BaseDatasetEDAModel.quality()
    # (contradictions et valeurs hors-plage = 0 pour cette série temporelle).
    score = round(max(0.0, min(100.0, 100 - (
        0.30 * missing_pct + 0.10 * dup_pct + 0.15 * noise_pct + 0.15 * out_pct
    ))), 1)
    verdict = "good" if score >= 90 else "moderate" if score >= 75 else "poor"

    def status(pct: float, good: float = 1.0, warn: float = 5.0) -> str:
        return "good" if pct <= good else "warning" if pct <= warn else "critical"

    indicators = [
        {"indicator": "Completeness", "value": round(100 - missing_pct, 2), "unit": "%",
         "issue_percent": round(missing_pct, 2), "status": status(missing_pct),
         "detail": f"{missing_cells} cellules manquantes sur {n_rows * n_cols:,}".replace(",", " ")},
        {"indicator": "Duplicate rows", "value": round(dup_pct, 2), "unit": "%",
         "issue_percent": round(dup_pct, 2), "status": status(dup_pct),
         "detail": f"{dup_cells} lignes dupliquées (train + test)"},
        {"indicator": "Noisy features", "value": round(noise_pct, 2), "unit": "%",
         "issue_percent": round(noise_pct, 2), "status": status(noise_pct, 20, 50),
         "detail": f"{noisy_n}/{len(features)} variables à bruit modéré ou élevé (NSR)"},
        {"indicator": "Outlier samples", "value": round(out_pct, 2), "unit": "%",
         "issue_percent": round(out_pct, 2), "status": status(out_pct, 5, 10),
         "detail": f"{out_n} échantillons signalés par ≥ 2 détecteurs sur 4"},
    ]
    return {"score": score, "verdict": verdict, "missing_pct": missing_pct,
            "dup_pct": dup_pct, "noise_pct": noise_pct, "outlier_pct": out_pct,
            "indicators": indicators}


def outlier_detectors_df(rep: dict) -> pd.DataFrame:
    """Détecteurs du rapport ElectricalLoad (4 méthodes, contamination 5 %)."""
    methods = (rep.get("outliers") or {}).get("methods") or {}
    rows = [{"detecteur": name,
             "n": int(v.get("outlier_count") or 0),
             "pourcent": float(v.get("outlier_percent") or 0.0)}
            for name, v in methods.items()]
    return pd.DataFrame(rows).sort_values("pourcent", ascending=False).reset_index(drop=True)


def electrical_gap_days_df(rep: dict, top: int = 15) -> pd.DataFrame:
    """Jours où |DEMAND − MA_X-4| dépasse le seuil, du plus grand écart au plus petit."""
    corr = rep.get("correlation") or {}
    threshold = float(corr.get("gap_threshold_mw") or 0.0)
    rows = []
    for g in corr.get("big_gap_days") or []:
        try:
            date = f"{int(g['year']):04d}-{int(g['month']):02d}-{int(g['day']):02d}"
        except (KeyError, TypeError, ValueError):
            date = str(g)
        rows.append({"date": date, "gap": float(g.get("gap") or 0.0), "seuil": threshold})
    return pd.DataFrame(rows).sort_values("gap", ascending=False).head(top).reset_index(drop=True)


def electrical_trends_df(rep: dict) -> pd.DataFrame:
    """Moyennes annuelles MA_X-4 / DEMAND du rapport ElectricalLoad."""
    rows = (rep.get("trends") or {}).get("annual") or []
    return pd.DataFrame(rows).rename(columns={"year": "annee"})


# ───────────────────────── tableaux de synthèse ─────────────────────────
def overview_table(reports: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for key, rep in reports.items():
        meta = rep.get("meta", {})
        dq = rep.get("data_quality", {})
        dash = dq.get("dashboard", {})
        noise = rep.get("noise", {})
        if is_electrical(rep):  # rapport EDA spécifique (pas de section data_quality)
            eq = electrical_quality(rep)
            rows.append({
                "key": key,
                "Dataset": label(key),
                "Lignes": meta.get("n_rows"),
                "Colonnes": meta.get("n_cols"),
                "Cible": target_name(rep),
                "Score qualité": eq["score"],
                "Verdict": eq["verdict"],
                "Doublons %": round(eq["dup_pct"], 2),
                "Contradictions %": 0.0,
                "Bruit %": round(eq["noise_pct"], 2),
                "Outliers %": round(eq["outlier_pct"], 2),
                "Graphiques": len(rep.get("plots", []) or []),
                "Insights": n_insights(rep),
                "Généré le": (meta.get("generated_at") or "")[:16].replace("T", " "),
            })
            continue
        if dash:  # datasets multiples
            noise_pct = dash.get("noise_percent")
            outl_pct = dash.get("outliers_percent")
            contra_pct = dash.get("contradictions_percent")
            dup_pct = _indicator(rep, "Duplicate rows")
        else:  # rapport météo : pas de section data_quality
            ds = noise.get("dataset_summary", {})
            checked = ds.get("features_checked") or 0
            noise_pct = round(100 * ds.get("features_with_noise", 0) / checked, 1) if checked else None
            outl_pct = None
            contra_pct = None
            dup_pct = rep.get("missing_duplicates", {}).get("duplicates", {}).get("percent")
        rows.append({
            "key": key,
            "Dataset": label(key),
            "Lignes": meta.get("n_rows"),
            "Colonnes": meta.get("n_cols"),
            "Cible": target_name(rep),
            "Score qualité": dq.get("quality_score"),
            "Verdict": dq.get("quality_label"),
            "Doublons %": dup_pct,
            "Contradictions %": contra_pct,
            "Bruit %": noise_pct,
            "Outliers %": outl_pct,
            "Graphiques": len(rep.get("plots", []) or []),
            "Insights": n_insights(rep),
            "Généré le": (meta.get("generated_at") or "")[:16].replace("T", " "),
        })
    return pd.DataFrame(rows)


def model_table(reports: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for key, rep in reports.items():
        ev = rep.get("model_phase", {}).get("evaluation")
        if not ev:
            continue
        rows.append({
            "key": key,
            "Dataset": label(key),
            "Balanced accuracy": ev.get("balanced_accuracy"),
            "F1 macro": ev.get("f1_macro"),
            "ROC AUC": ev.get("roc_auc"),
            "CV balanced acc.": ev.get("cv_balanced_accuracy_mean"),
            "CV écart-type": ev.get("cv_balanced_accuracy_std"),
            "Accuracy de base": ev.get("baseline_accuracy"),
            "Train": ev.get("n_train"),
            "Test": ev.get("n_test"),
        })
    return pd.DataFrame(rows)


# ───────────────────────── détail d'un dataset ─────────────────────────
def indicators_df(rep: dict) -> pd.DataFrame:
    if is_electrical(rep):
        return pd.DataFrame(electrical_quality(rep)["indicators"])
    return pd.DataFrame(rep.get("data_quality", {}).get("indicators", []) or [])


def noise_df(rep: dict) -> pd.DataFrame:
    rows = []
    for f in rep.get("noise", {}).get("features", []) or []:
        rows.append({
            "feature": f.get("feature"),
            "noise_to_signal": f.get("noise_to_signal", f.get("nsr")),
            "snr_db": f.get("snr_db"),
            # Les verdicts ElectricalLoad ("clean", "low noise") sont alignés
            # sur le domaine des couleurs Altair (negligible/low/moderate/high).
            "verdict": _electrical_verdict(f.get("verdict")),
            "noise_sigma": f.get("noise_sigma", f.get("noise_std")),
            "signal_sigma": f.get("signal_sigma", f.get("signal_std")),
            "spikes %": f.get("spike_percent", f.get("spikes_percent")),
        })
    return pd.DataFrame(rows)


def importance_df(rep: dict, top: int = 10) -> pd.DataFrame:
    fi = rep.get("model_phase", {}).get("feature_importance", {}) or {}
    perm = fi.get("permutation_ranking")
    if perm:
        df = pd.DataFrame(perm)[["feature", "drop_points"]].rename(columns={"drop_points": "value"})
        df["method"] = "Permutation (points de balanced accuracy perdus)"
    elif fi.get("ranked_features"):
        df = pd.DataFrame(fi["ranked_features"])[["feature", "percentage"]].rename(columns={"percentage": "value"})
        df["method"] = "Impureté Random Forest (%)"
    else:
        return pd.DataFrame(columns=["feature", "value", "method"])
    return df.sort_values("value", ascending=False).head(top).reset_index(drop=True)


def class_balance_df(rep: dict) -> pd.DataFrame:
    tb = rep.get("model_phase", {}).get("target_balance", {}) or {}
    counts = tb.get("class_counts") or {}
    pcts = tb.get("class_percentages") or {}
    return pd.DataFrame([{"classe": str(k), "effectif": v, "pourcentage": pcts.get(k)} for k, v in counts.items()])


def confusion_df(rep: dict) -> pd.DataFrame:
    ev = rep.get("model_phase", {}).get("evaluation") or {}
    cm = ev.get("confusion_matrix")
    if not cm:
        return pd.DataFrame(columns=["réel", "prédit", "n"])
    classes = [str(c) for c in (ev.get("classes") or range(len(cm)))]
    rows = [{"réel": classes[i], "prédit": classes[j], "n": v}
            for i, r in enumerate(cm) for j, v in enumerate(r)]
    return pd.DataFrame(rows)


def corr_pairs_df(rep: dict, top: int = 8) -> pd.DataFrame:
    corr = rep.get("correlation", {})
    if is_electrical(rep) and corr.get("pearson_ma4_demand") is not None:
        r = float(corr["pearson_ma4_demand"])
        return pd.DataFrame([{
            "a": "MA_X-4", "b": "DEMAND", "paire": "MA_X-4  ×  DEMAND",
            "pearson": r, "spearman": r,
            "strength": "très forte" if abs(r) >= 0.8 else "forte" if abs(r) >= 0.6 else "modérée",
        }])
    pairs = corr.get("top_pairs") or corr.get("pairs") or []
    df = pd.DataFrame(pairs)
    if df.empty:
        return df
    df["paire"] = df["a"] + "  ×  " + df["b"]
    df["abs"] = df["pearson"].abs()
    return df.sort_values("abs", ascending=False).head(top)[["paire", "pearson", "spearman", "strength"]]


def corr_matrix_df(rep: dict, kind: str = "pearson") -> pd.DataFrame:
    """Long-format correlation matrix (a, b, value) for an Altair heatmap."""
    corr = rep.get("correlation", {}) or {}
    mat = corr.get(kind) or {}
    df = pd.DataFrame(mat)
    if df.empty:
        return pd.DataFrame(columns=["a", "b", "value"])
    long = df.reset_index(names="a").melt("a", var_name="b", value_name="value")
    return long.dropna(subset=["value"])


def vif_df(rep: dict, top: int = 10) -> pd.DataFrame:
    """Most collinear features (VIF >= 10 are the actionable ones)."""
    vif = (rep.get("correlation", {}) or {}).get("vif") or []
    df = pd.DataFrame(vif)
    return df.head(top) if not df.empty else df


def plan_df(rep: dict) -> pd.DataFrame:
    return pd.DataFrame(rep.get("interpretation", {}).get("preprocessing_plan", []) or [])


# ───────────────────────── modelling lab (Phase II) ─────────────────────────
def lab_models_df(rep: dict) -> pd.DataFrame:
    """Benchmark table: one row per algorithm (KNN / Ensemble / SVM)."""
    lab = rep.get("model_lab") or {}
    rows = []
    for m in lab.get("models", []) or []:
        params = ", ".join(f"{k}={v}" for k, v in (m.get("best_params") or {}).items())
        rows.append({
            "key": m.get("model_key"),
            "model": m.get("model_name"),
            "params": params,
            "test_acc": m.get("test_accuracy"),
            "train_acc": m.get("train_accuracy"),
            "cv": m.get("cv_balanced_accuracy_mean"),
            "cv_std": m.get("cv_balanced_accuracy_std"),
            "bal_acc": m.get("balanced_accuracy"),
            "precision": m.get("precision_macro"),
            "recall": m.get("recall_macro"),
            "f1": m.get("f1_macro"),
            "auc": m.get("auc"),
            "status": m.get("overfitting_status"),
            "badge": m.get("overfitting_badge"),
            "desc": m.get("overfitting_desc"),
        })
    return pd.DataFrame(rows)


def lab_deltas_df(rep: dict) -> pd.DataFrame:
    """Impact of each perturbation vs baseline, in accuracy points."""
    comp = ((rep.get("model_lab") or {}).get("noise_outliers") or {}).get("comparison") or []
    rows = []
    for c in comp:
        rows.append({
            "Modèle": c.get("model_name", "").split(" (")[0],
            "Sans outliers": (c.get("iqr_cleanup_delta") or 0) * 100,
            "Bruit": (c.get("noise_delta") or 0) * 100,
            "Récupération lissage": (c.get("smooth_recovery") or 0) * 100,
            "Outliers injectés": (c.get("outlier_delta") or 0) * 100,
        })
    return pd.DataFrame(rows)


def recommendations_df(rep: dict) -> pd.DataFrame:
    df = pd.DataFrame(rep.get("interpretation", {}).get("recommendations", []) or [])
    if df.empty:
        return df
    order = {"high": 0, "medium": 1, "low": 2}
    return df.sort_values("priority", key=lambda s: s.map(order).fillna(9)).reset_index(drop=True)


def plots_for(rep: dict, root: Path = ROOT) -> list[dict]:
    """Graphiques du rapport avec chemin résolu (les JSON contiennent des chemins Windows)."""
    out = []
    for p in rep.get("plots", []) or []:
        if isinstance(p, str):
            name = p
            title = name
            section = "distribution"
            raw = ""
        else:
            name = p.get("name")
            title = p.get("title") or name
            section = p.get("section") or "distribution"
            raw = p.get("file") or ""
        # Les rapports ElectricalLoad stockent des chemins absolus d'une autre machine.
        candidates = []
        if raw:
            r = raw.replace("\\", "/")
            candidates.append(Path(r) if Path(r).is_absolute() else root / r)
        if is_electrical(rep) and name:
            candidates.append(root / "eda_output_electricalload" / "plots" / f"{name}.png")
        path = next((c for c in candidates if c.exists()), None)
        if path is None:
            continue
        out.append({"title": title, "section": section, "path": path})
    return out


# ───────────────────────── outliers, bruit, interprétation ─────────────────────────
# clé du rapport -> id historique de la table principale dans data.db
# Les lignes peuvent changer après une ré-ingestion: utiliser resolve_dataset_id()
# pour retrouver le fichier correspondant dynamiquement.
DB_DATASET_IDS = {"weather": 134, "covid": 72, "parkinson": 109, "alcohol": 70, "heart": 96, "electricalload": 135}


def resolve_dataset_id(db_path: Path, key: str, rep: dict) -> int | None:
    """Resolve the catalogue row for a report, regardless of SQLite row renumbering."""
    csv_path = str((rep.get("meta") or {}).get("csv_path") or "")
    basename = Path(csv_path.replace("\\", "/")).name.lower()
    normalized = csv_path.replace("\\", "/").strip("/").lower()

    try:
        with sqlite3.connect(db_path) as con:
            rows = con.execute(
                "SELECT id, filename, file_path FROM datasets"
            ).fetchall()
    except sqlite3.Error:
        return DB_DATASET_IDS.get(key)

    if basename:
        exact_path = [
            r[0] for r in rows
            if str(r[2]).replace("\\", "/").strip("/").lower() == normalized
        ]
        if exact_path:
            return exact_path[0]

        exact_name = [r[0] for r in rows if str(r[1]).lower() == basename]
        if len(exact_name) == 1:
            return exact_name[0]

    fallback = DB_DATASET_IDS.get(key)
    if fallback is not None:
        with sqlite3.connect(db_path) as con:
            if con.execute(
                "SELECT 1 FROM datasets WHERE id = ?", (fallback,)
            ).fetchone():
                return fallback
    return None


def outlier_features(rep: dict) -> list[dict]:
    o = rep.get("outliers", {}) or {}
    return o.get("classical_features") or o.get("features") or []


def boxplot_frames(rep: dict, normalized: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Boîtes à moustaches reconstruites depuis les statistiques du rapport (Q1, médiane, Q3, barrières IQR, min, max).
    normalized : écart à la médiane en nombre d'IQR, pour comparer des variables d'échelles différentes."""
    med = {r["feature"]: r.get("50%") for r in rep.get("description", {}).get("describe_numeric", []) or []}
    rows, ext = [], []
    for f in outlier_features(rep):
        q1, q3, iqr = f.get("q1"), f.get("q3"), f.get("iqr")
        if q1 is None or q3 is None:
            continue
        m = med.get(f["feature"])
        m = (q1 + q3) / 2 if m is None else m
        lo = max(f["min"], f["lower_fence"]) if f.get("lower_fence") is not None else f["min"]
        hi = min(f["max"], f["upper_fence"]) if f.get("upper_fence") is not None else f["max"]
        sc = (lambda v: (v - m) / iqr) if (normalized and iqr) else (lambda v: v)
        pct = f.get("iqr_percent") or 0.0
        rows.append({"feature": f["feature"], "q1": q1, "median": m, "q3": q3, "vmin": f["min"], "vmax": f["max"],
                     "x_q1": sc(q1), "x_med": sc(m), "x_q3": sc(q3), "x_wlo": sc(lo), "x_whi": sc(hi),
                     "n_out": f.get("iqr_outliers") or 0, "pct_out": pct,
                     "niveau": "élevé" if pct >= 5 else "modéré" if pct >= 1 else "faible"})
        if f["min"] < lo:
            ext.append({"feature": f["feature"], "x": sc(f["min"]), "valeur": f["min"], "n_out": f.get("iqr_outliers") or 0})
        if f["max"] > hi:
            ext.append({"feature": f["feature"], "x": sc(f["max"]), "valeur": f["max"], "n_out": f.get("iqr_outliers") or 0})
    return pd.DataFrame(rows), pd.DataFrame(ext)


def outlier_long(rep: dict, top: int = 12) -> pd.DataFrame:
    rows = []
    for f in sorted(outlier_features(rep), key=lambda f: -(f.get("iqr_percent") or 0))[:top]:
        for m, k in (("IQR", "iqr_percent"), ("Z-score", "z_percent"), ("Z modifié (MAD)", "modified_z_percent"),
                     ("Percentiles 1-99", "percentile_percent")):
            if f.get(k) is not None:
                rows.append({"feature": f["feature"], "methode": m, "pourcent": f[k]})
    return pd.DataFrame(rows)


def interpret_variables(rep: dict) -> pd.DataFrame:
    """Une ligne de lecture par variable : outliers (IQR vs Z) + bruit -> constat et action."""
    nz = {r["feature"]: r for r in noise_df(rep).to_dict("records")}
    rows = []
    for f in outlier_features(rep):
        iqr_p, z_p = f.get("iqr_percent") or 0.0, f.get("z_percent") or 0.0
        n = nz.get(f["feature"], {})
        verdict = n.get("verdict")
        if iqr_p >= 5 and z_p < iqr_p / 2:
            out_txt = f"{iqr_p:.1f} % hors barrières IQR, {z_p:.1f} % en Z-score : queue lourde, extrêmes probablement réels."
            action = "Conserver ; échelle robuste (RobustScaler) ou transformation log."
        elif iqr_p >= 5:
            out_txt = f"{iqr_p:.1f} % d'outliers, confirmés par le Z-score ({z_p:.1f} %)."
            action = "Vérifier la source des mesures extrêmes avant de les garder."
        elif iqr_p >= 1:
            out_txt = f"{iqr_p:.1f} % d'outliers : présence modérée."
            action = "Surveiller ; pas de suppression automatique."
        else:
            out_txt = f"{iqr_p:.2f} % d'outliers : distribution propre."
            action = "Aucune action sur les extrêmes."
        if verdict in ("moderate", "high"):
            nz_txt = f"bruit {'élevé' if verdict == 'high' else 'modéré'} (bruit/signal {n.get('noise_to_signal', float('nan')):.2f})"
            action += " Lisser ou agréger les mesures répétées."
        elif verdict in ("negligible", "low"):
            nz_txt = f"bruit {'négligeable' if verdict == 'negligible' else 'faible'}"
        else:
            nz_txt = "bruit non évalué"
        rows.append({"Variable": f["feature"], "Outliers IQR %": iqr_p, "Bruit/signal": n.get("noise_to_signal"),
                     "Bruit": verdict or "—", "Interprétation": f"{out_txt} Et {nz_txt}.", "Action": action})
    return pd.DataFrame(rows).sort_values("Outliers IQR %", ascending=False).reset_index(drop=True)


def electrical_noise_interpretation_df(rep: dict) -> pd.DataFrame:
    """Table de lecture du bruit ElectricalLoad (NSR/SNR) quand l'outlier est multivarié."""
    rows = []
    for r in noise_df(rep).to_dict("records"):
        nsr = r.get("noise_to_signal")
        snr = r.get("snr_db")
        verdict = str(r.get("verdict") or "—").lower()
        if verdict == "negligible":
            interp = "Signal très propre : les écarts locaux sont négligeables."
            action = "Conserver sans lissage."
        elif verdict == "low":
            interp = "Bruit faible : la variation observée est surtout du signal utile."
            action = "Conserver ; le lissage n'apporterait rien."
        elif verdict == "moderate":
            interp = "Bruit modéré : une part notable des micro-variations est aléatoire."
            action = "Tester une agrégation horaire ou un lissage léger."
        elif verdict == "high":
            interp = "Bruit élevé : les micro-variations dominent le signal."
            action = "Lisser / agréger avant modélisation et vérifier le capteur."
        else:
            interp = "Verdict de bruit indisponible."
            action = "Vérifier le rapport EDA."
        if pd.notna(nsr) and pd.notna(snr):
            interp += f" NSR {nsr:.3f} · SNR {snr:.1f} dB."
        elif pd.notna(nsr):
            interp += f" NSR {nsr:.3f}."
        rows.append({
            "Variable": r.get("feature"),
            "NSR": nsr,
            "SNR (dB)": snr,
            "Verdict": r.get("verdict") or "—",
            "Interprétation": interp,
            "Action": action,
        })
    return pd.DataFrame(rows)


def outlier_noise_interpretation(rep: dict) -> dict:
    """Synthèse prête à afficher : chiffres clés, faits et décisions pour outliers + bruit."""
    nz = noise_df(rep)
    bad = nz[nz["verdict"].isin(["moderate", "high"])] if not nz.empty else nz
    worst = nz.sort_values("noise_to_signal", ascending=False).iloc[0] if (
        not nz.empty and nz["noise_to_signal"].notna().any()) else None

    if is_electrical(rep):
        o = rep.get("outliers") or {}
        cons = o.get("consensus_outliers") or {}
        unanimous = (o.get("unanimous_outliers") or {}).get("count", 0)
        det = outlier_detectors_df(rep)
        cons_pct = cons.get("percent")
        headline = (
            f"{len(det)} détecteurs multivariés · consensus "
            f"{'n/a' if cons_pct is None else f'{cons_pct:.2f} %'} "
            f"· {len(bad)} variable(s) sur {len(nz)} en bruit modéré/élevé"
        )
        facts = []
        if cons_pct is not None:
            facts.append(
                f"Le consensus (≥ 2 détecteurs) retient {cons.get('count', 0)} lignes, soit {cons_pct:.2f} %. "
                f"Ce niveau correspond à la contamination de 5 % paramétrée dans les détecteurs : "
                f"il s'agit de comportements suspects, pas nécessairement d'erreurs de mesure."
            )
        if unanimous:
            facts.append(
                f"{unanimous} lignes sont signalées par les 4 détecteurs : ce sont les anomalies les plus sûres "
                "(pointes de charge, jours fériés, arrêts ou événements spéciaux)."
            )
        if worst is not None and pd.notna(worst.get("noise_to_signal")):
            snr_txt = (f" et SNR {worst['snr_db']:.1f} dB" if pd.notna(worst.get("snr_db")) else "")
            facts.append(
                f"Le bruit reste globalement faible : NSR maximal {worst['noise_to_signal']:.3f} "
                f"pour {worst['feature']}{snr_txt}. Les variables lissées (week_X-*, MA_X-4) ont un NSR "
                "plus bas par construction, pas une meilleure qualité de capteur."
            )
        decisions = [
            "Ne pas supprimer aveuglément les lignes de consensus : les conserver comme indicateur d'événement "
            "ou les expliquer via `holiday`, `weekend` et `Holiday_ID`.",
            "Garder les retards `week_X-*` et `MA_X-4`, mais vérifier qu'ils sont calculés uniquement sur le passé "
            "avant entraînement (risque de fuite temporelle).",
            "Aucun lissage nécessaire sur `DEMAND` et `T2M_toc` : le bruit estimé est faible et les résidus aux "
            "pics reflètent la volatilité réelle du réseau.",
        ]
        return {"headline": headline, "facts": facts, "decisions": decisions}

    feats = outlier_features(rep)
    s = (rep.get("outliers") or {}).get("summary") or {}
    mean_iqr = s.get("mean_iqr_percent")
    if mean_iqr is None and feats:
        mean_iqr = sum(f.get("iqr_percent") or 0 for f in feats) / len(feats)
    top = max(feats, key=lambda f: f.get("iqr_percent") or 0) if feats else None
    cons_pct = s.get("consensus_percent")
    headline = (
        f"{len(feats)} variable(s) analysée(s) · IQR moyen "
        f"{'n/a' if mean_iqr is None else f'{mean_iqr:.2f} %'} · "
        f"consensus {'n/a' if cons_pct is None else f'{cons_pct:.2f} %'} · "
        f"{len(bad)} variable(s) sur {len(nz)} en bruit modéré/élevé"
    )
    facts = []
    if top is not None:
        p = top.get("iqr_percent") or 0
        z = top.get("z_percent") or 0
        facts.append(
            f"{top['feature']} est la variable la plus touchée : {p:.2f} % hors barrières IQR "
            f"({top.get('iqr_outliers', 0)} valeurs) contre {z:.2f} % en Z-score. "
            + ("L'écart IQR ≫ Z indique une queue lourde : ces extrêmes sont souvent réels."
               if p >= 2 * max(z, 0.01) and p >= 1 else
               "Les deux méthodes convergent : les extrêmes sont confirmés.")
            + (" Attention : dans ce jeu, `Pressure = 0` code un capteur non renseigné, pas une vraie mesure."
               if "pressure" in str(top.get("feature", "")).lower() else "")
        )
    if cons_pct is not None:
        facts.append(
            f"La détection multivariée signale {s.get('consensus_count', '?')} lignes ({cons_pct:.2f} %) "
            f"par au moins 2 des 6 détecteurs, dont {s.get('unanimous_count', 0)} à l'unanimité."
        )
    if worst is not None and pd.notna(worst.get("noise_to_signal")):
        snr_txt = (f" et SNR {worst['snr_db']:.1f} dB" if pd.notna(worst.get("snr_db")) else "")
        facts.append(
            f"La variable la plus bruitée est {worst['feature']} "
            f"(bruit/signal {worst['noise_to_signal']:.2f}{snr_txt})."
        )
    decisions = []
    if top is not None and (top.get("iqr_percent") or 0) >= 5 and (top.get("z_percent") or 0) < (top.get("iqr_percent") or 0) / 2:
        decisions.append("Conserver les extrêmes de « queue lourde » et utiliser une échelle robuste ou une transformation log.")
    else:
        decisions.append("Aucune suppression automatique : vérifier la source des valeurs extrêmes avant décision.")
    if bad.empty:
        decisions.append("Le bruit est négligeable ou faible sur toutes les variables : pas de lissage nécessaire.")
    else:
        decisions.append("Pour les variables en bruit modéré/élevé, tester l'agrégation ou le lissage, puis comparer les métriques du modèle.")
    if cons_pct is not None and cons_pct >= 5:
        decisions.append("Analyser les lignes de consensus avant nettoyage : elles peuvent être des cas rares mais légitimes.")
    return {"headline": headline, "facts": facts, "decisions": decisions}


def outlier_noise_points(rep: dict) -> list[tuple[str, str, str]]:
    """Points d'interprétation globaux (ton, titre, détail) pour la page Outliers & bruit."""
    if is_electrical(rep):
        o = rep.get("outliers") or {}
        cons = o.get("consensus_outliers") or {}
        unanimous = (o.get("unanimous_outliers") or {}).get("count", 0)
        det = outlier_detectors_df(rep)
        pts = []
        if cons.get("percent") is not None:
            pts.append(("warn" if cons["percent"] >= 5 else "info",
                        f"{cons['percent']:.2f} % d'outliers par consensus ({cons.get('count', 0)} lignes)",
                        f"Signalées par au moins 2 des {len(det)} détecteurs ; {unanimous} lignes sont "
                        "signalées par les 4. Ce sont les pointes de charge du réseau : ne pas les supprimer aveuglément."))
        nz = noise_df(rep).dropna(subset=["noise_to_signal"])
        if not nz.empty:
            bad = nz[nz["verdict"].isin(["moderate", "high"])]
            worst = nz.sort_values("noise_to_signal", ascending=False).iloc[0]
            pts.append(("good" if bad.empty else "warn",
                        f"Bruit : {len(bad)} variable(s) sur {len(nz)} en bruit modéré ou élevé",
                        f"NSR maximal {worst['noise_to_signal']:.3f} pour {worst['feature']} "
                        f"(SNR {worst['snr_db']:.1f} dB) : la série horaire est globalement propre."))
        return pts

    pts = []
    feats = outlier_features(rep)
    if feats:
        top = max(feats, key=lambda f: f.get("iqr_percent") or 0)
        p = top.get("iqr_percent") or 0
        pts.append(("bad" if p >= 5 else "warn" if p >= 1 else "good",
                    f"Variable la plus touchée : {top['feature']} ({p:.2f} % d'outliers IQR)",
                    f"{top.get('iqr_outliers', 0)} valeurs hors de [{top['lower_fence']:.3g} ; {top['upper_fence']:.3g}]."))
        heavy = [f["feature"] for f in feats if (f.get("iqr_percent") or 0) >= 2 * max(f.get("z_percent") or 0, 0.01)
                 and (f.get("iqr_percent") or 0) >= 1]
        if heavy:
            pts.append(("info", f"{len(heavy)} variable(s) à queue lourde",
                        "L'IQR signale bien plus de points que le Z-score (" + ", ".join(heavy[:6])
                        + ("…" if len(heavy) > 6 else "") + ") : ces extrêmes sont souvent réels, à ne pas supprimer."))
    s = (rep.get("outliers", {}) or {}).get("summary") or {}
    if s.get("consensus_percent") is not None:
        pts.append(("warn" if s["consensus_percent"] >= 5 else "info",
                    f"{s['consensus_percent']:.2f} % d'échantillons outliers par consensus",
                    f"{s['consensus_count']} lignes signalées par plusieurs des 6 détecteurs multivariés "
                    f"({s.get('unanimous_count', 0)} à l'unanimité)."))
    nz = noise_df(rep).dropna(subset=["noise_to_signal"])
    if not nz.empty:
        bad = nz[nz["verdict"].isin(["moderate", "high"])]
        worst = nz.sort_values("noise_to_signal", ascending=False).iloc[0]
        pts.append(("bad" if len(bad) / len(nz) >= .5 else "warn" if len(bad) else "good",
                    f"Bruit : {len(bad)} variable(s) sur {len(nz)} modérées ou élevées",
                    f"Plus bruitée : {worst['feature']} (bruit/signal {worst['noise_to_signal']:.2f}, "
                    f"SNR {worst['snr_db']:.1f} dB)." if pd.notna(worst["snr_db"]) else
                    f"Plus bruitée : {worst['feature']} (bruit/signal {worst['noise_to_signal']:.2f})."))
    return pts


# ───────────────────────── base SQLite & code ─────────────────────────
def db_stats(db_path: Path | None = None) -> dict | None:
    db_path = db_path or ROOT / "data.db"
    if not db_path.exists():
        return None
    con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        counts = {t: con.execute(f'select count(*) from "{t}"').fetchone()[0]
                  for t in ("reports", "sections", "insights", "plots", "datasets")}
        cats = pd.DataFrame(
            con.execute("select category as Catégorie, count(*) as Tables, sum(n_rows) as Lignes "
                        "from datasets group by category order by 2 desc").fetchall(),
            columns=["Catégorie", "Tables", "Lignes"])
        n_cat = con.execute("select count(distinct category) from datasets").fetchone()[0]
    finally:
        con.close()
    return {"counts": counts, "categories": cats, "n_categories": n_cat,
            "size_mb": round(db_path.stat().st_size / 1e6, 1)}


def code_inventory(root: Path = ROOT) -> pd.DataFrame:
    rows = []
    for f in sorted(root.glob("*.py")):
        if f.name in ("dashboard_data.py", "project_dashboard.py"):
            continue
        n = sum(1 for _ in f.open(encoding="utf-8", errors="ignore"))
        layer, role = FILE_ROLES.get(f.name, ("Autre", ""))
        rows.append({"Fichier": f.name, "Couche": layer, "Lignes": n, "Rôle": role})
    return pd.DataFrame(rows).sort_values("Lignes", ascending=False).reset_index(drop=True)


# ───────────────────────── points d'attention & parcours ─────────────────────────
def attention_points(reports: dict[str, dict]) -> list[tuple[str, str, str]]:
    """Constats calculés à partir des rapports (ton, titre, détail)."""
    pts: list[tuple[str, str, str]] = []
    ov = overview_table(reports)
    for _, r in ov.iterrows():
        name = r["Dataset"]
        if pd.notna(r["Doublons %"]) and r["Doublons %"] >= 5:
            pts.append(("bad", f"{name} : {r['Doublons %']:.1f} % de doublons exacts",
                        "Le rapport recommande de les retirer avant la validation croisée : "
                        "sinon les scores du modèle sont optimistes."))
        if pd.notna(r["Contradictions %"]) and r["Contradictions %"] >= 10:
            pts.append(("warn", f"{name} : {r['Contradictions %']:.1f} % de lignes contradictoires",
                        "Lignes masquées dans les statistiques ; à corriger à la source si possible."))
        if pd.notna(r["Bruit %"]) and r["Bruit %"] >= 80:
            pts.append(("warn", f"{name} : {r['Bruit %']:.0f} % des variables bruitées",
                        "Rapport bruit/signal élevé sur la majorité des variables testées."))
    for key, rep in reports.items():
        tb = rep.get("model_phase", {}).get("target_balance", {}) or {}
        ratio = tb.get("imbalance_ratio")
        if ratio and ratio >= 2:
            pts.append(("info", f"{label(key)} : classes déséquilibrées ({ratio:.2f} pour 1)",
                        "Pondération des classes utilisée ; lire la balanced accuracy et le F1, pas l'accuracy seule."))
        if key == "weather" and not rep.get("model_phase", {}).get("evaluation"):
            pts.append(("info", "Météo : évaluation du modèle absente du rapport enregistré",
                        "weather_model.py sait la calculer ; relancer python weather_db.py weatherHistory.csv data.db."))
        if is_electrical(rep):
            corr = rep.get("correlation") or {}
            gaps = electrical_gap_days_df(rep, 1)
            if corr.get("pearson_ma4_demand") is not None:
                pts.append(("info",
                            f"Charge électrique : MA_X-4 corrélée à DEMAND (r = {corr['pearson_ma4_demand']:.4f})",
                            "Prédicteur puissant mais sensible aux fuites temporelles ; le split train/test "
                            "doit rester chronologique (2015-2019 → avril 2019)."))
            if not gaps.empty:
                pts.append(("warn",
                            f"Charge électrique : écart de {gaps.iloc[0]['gap']:.0f} MW le {gaps.iloc[0]['date']}",
                            f"Au-dessus du seuil de {gaps.iloc[0]['seuil']:.0f} MW : pic de demande réel "
                            "(jour férié, météo ou arrêt de production) à conserver comme signal."))
    for key, rep in reports.items():
        imp = importance_df(rep, top=2)
        if len(imp) == 2 and imp.loc[1, "value"] > 0 and imp.loc[0, "value"] / imp.loc[1, "value"] >= 5 \
                and "Permutation" in imp.loc[0, "method"]:
            pts.append(("warn", f"{label(key)} : une seule variable porte le modèle",
                        f"{imp.loc[0, 'feature']} fait perdre {imp.loc[0, 'value']:.1f} points de balanced accuracy "
                        f"une fois permutée, contre {imp.loc[1, 'value']:.1f} pour la suivante."))
    mt = model_table(reports)
    if not mt.empty:
        best = mt.sort_values("Balanced accuracy", ascending=False).iloc[0]
        pts.append(("good", f"Score le plus élevé : {best['Dataset']}",
                    f"Balanced accuracy {best['Balanced accuracy']:.3f}, ROC AUC {best['ROC AUC']:.4f} "
                    f"(validation croisée {best['CV balanced acc.']:.3f})."))
    return pts


def timeline(reports: dict[str, dict], db: dict | None) -> list[dict]:
    """Étapes du projet, chacune adossée à un fichier ou un chiffre vérifiable."""
    steps = []
    if db:
        c = db["counts"]
        steps.append({"date": "", "title": "Catalogue de données (Phase 0)",
                      "detail": f"{c['datasets']} tables importées dans data.db, réparties en {db['n_categories']} catégories "
                                "(populate_datasets.py, dataset_overview.py)."})
    w = reports.get("weather")
    if w:
        tb = w.get("model_phase", {}).get("target_balance", {})
        steps.append({"date": (w["meta"].get("generated_at") or "")[:10],
                      "title": "EDA météo + phase modèle",
                      "detail": f"weatherHistory.csv : {w['meta']['n_rows']:,} lignes".replace(",", " ")
                                + f", {len(w.get('plots', []))} graphiques, "
                                  f"cible Precip Type déséquilibrée ({tb.get('imbalance_ratio')} pour 1), Random Forest pondéré."})
    multi = [k for k in reports if k != "weather"]
    if multi:
        dates = sorted({(reports[k]["meta"].get("generated_at") or "")[:10] for k in multi})
        steps.append({"date": dates[-1],
                      "title": f"Pipeline EDA + ML sur {len(multi)} datasets",
                      "detail": ", ".join(label(k) for k in multi)
                                + " : qualité Phase I, bruit, outliers (6 détecteurs), contradictions, corrélations, modèle "
                                  "(run_all_models.py)."})
    e = reports.get(ELECTRICAL_KEY)
    if e:
        eq = electrical_quality(e)
        r_ma = (e.get("correlation") or {}).get("pearson_ma4_demand", "n/a")
        steps.append({"date": (e["meta"].get("generated_at") or "")[:10],
                      "title": "Charge électrique : EDA temporelle + régression",
                      "detail": f"{e['meta'].get('n_rows'):,} relevés horaires".replace(",", " ")
                                + f", score qualité {eq['score']}/100, 4 détecteurs d'anomalies, "
                                  f"r MA_X-4/DEMAND = {r_ma} "
                                  "(electrical_load_eda.py + eval_electricalload.py)."})
    steps.append({"date": "", "title": "Application Streamlit en architecture MVC",
                  "detail": "Model (weather_model.py, multi_eda_model.py, electrical_load_eda.py) · "
                            "stockage (data.py, export_dataset_previews.py) · "
                            "vues (app.py, project_dashboard.py, electrical_load_view.py, workplan_view.py, "
                             "dataset_workplan_view.py, column_dictionary.py, outlier_noise_interpretation.py)."})
    return steps
