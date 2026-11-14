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
}

# Rôle de chaque script (d'après README.md et les docstrings du projet)
FILE_ROLES: dict[str, tuple[str, str]] = {
    "weather_model.py": ("Model", "Analyse complète de weatherHistory.csv + Random Forest"),
    "multi_eda_model.py": ("Model", "Pipeline EDA + ML des 4 datasets (COVID, Parkinson, alcool, cœur)"),
    "quality.py": ("Model", "Règles de qualité partagées (plages, seuils de bruit, nettoyage)"),
    "quality_multi.py": ("Model", "Indicateurs de qualité Phase I pour les datasets multiples"),
    "data.py": ("Stockage", "Couche SQLite : datasets, rapports, insights, graphiques (BLOB)"),
    "weather_db.py": ("Stockage", "Alias de compatibilité de DataDB (rapports météo)"),
    "populate_datasets.py": ("Stockage", "Import des fichiers de données dans data.db"),
    "export_weather_json.py": ("Stockage", "Export du dernier rapport météo en JSON"),
    "run_all_models.py": ("Pipeline", "Exécute le pipeline sur les 4 datasets demandés"),
    "weather_view.py": ("View", "Vue en lecture seule d'un rapport (un onglet par section)"),
    "dataset_overview.py": ("View", "Vue d'ensemble : import et exploration des datasets"),
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


# ───────────────────────── tableaux de synthèse ─────────────────────────
def overview_table(reports: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for key, rep in reports.items():
        meta = rep.get("meta", {})
        dq = rep.get("data_quality", {})
        dash = dq.get("dashboard", {})
        noise = rep.get("noise", {})
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
    return pd.DataFrame(rep.get("data_quality", {}).get("indicators", []) or [])


def noise_df(rep: dict) -> pd.DataFrame:
    rows = [{
        "feature": f.get("feature"),
        "noise_to_signal": f.get("noise_to_signal"),
        "snr_db": f.get("snr_db"),
        "verdict": f.get("verdict"),
        "spikes %": f.get("spike_percent", f.get("spikes_percent")),
    } for f in rep.get("noise", {}).get("features", []) or []]
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
    pairs = corr.get("top_pairs") or corr.get("pairs") or []
    df = pd.DataFrame(pairs)
    if df.empty:
        return df
    df["paire"] = df["a"] + "  ×  " + df["b"]
    df["abs"] = df["pearson"].abs()
    return df.sort_values("abs", ascending=False).head(top)[["paire", "pearson", "spearman", "strength"]]


def plan_df(rep: dict) -> pd.DataFrame:
    return pd.DataFrame(rep.get("interpretation", {}).get("preprocessing_plan", []) or [])


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
        raw = (p.get("file") or "").replace("\\", "/")
        path = root / raw if raw else None
        if path is not None and not path.exists():
            path = None
        out.append({"title": p.get("title") or p.get("name"), "section": p.get("section") or "autre", "path": path})
    return [p for p in out if p["path"] is not None]


# ───────────────────────── outliers, bruit, interprétation ─────────────────────────
# clé du rapport -> id de la table principale dans data.db (table `datasets`)
DB_DATASET_IDS = {"weather": 68, "covid": 3, "parkinson": 43, "alcohol": 1, "heart": 32}


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


def outlier_noise_points(rep: dict) -> list[tuple[str, str, str]]:
    """Points d'interprétation globaux (ton, titre, détail) pour la page Outliers & bruit."""
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
    steps.append({"date": "", "title": "Application Streamlit en architecture MVC",
                  "detail": "Model (weather_model.py, multi_eda_model.py) · stockage (data.py) · "
                            "vues (app.py, weather_view.py, dataset_overview.py, theme.py)."})
    return steps
