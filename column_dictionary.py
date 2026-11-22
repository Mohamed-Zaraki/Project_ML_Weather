"""column_dictionary.py : dictionnaire et interprétation métier des colonnes, par dataset.

Complète la fiche « 📋 Données » du dashboard : pour chaque variable, on affiche son rôle
(cible, identifiant, horodatage, prédicteur...), sa signification métier en français, ses
statistiques observées, son taux de valeurs manquantes, son signal bruit/outliers et la
recommandation de prétraitement. Les définitions sont curées pour les 6 datasets du projet ;
un mécanisme heuristique couvre les colonnes inconnues.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

import dashboard_data as dd
from theme import kpi_row, section_title

# (rôle, signification, recommandation)
GLOSSARY: dict[str, dict[str, tuple[str, str, str]]] = {
    "weather": {
        "Formatted Date": ("Horodatage", "Date et heure de la mesure, avec décalage UTC (+0200).",
                           "Convertir en datetime, trier chronologiquement puis dériver mois/heure."),
        "Summary": ("Prédicteur catégoriel", "Description textuelle des conditions météo (27 modalités, ex. « Partly Cloudy »).",
                    "Encoder en one-hot ou regrouper les modalités rares."),
        "Precip Type": ("Cible", "Type de précipitation observé : rain ou snow. Variable à prédire.",
                        "Ne jamais utiliser comme prédicteur ; déséquilibre à gérer (class_weight='balanced')."),
        "Temperature (C)": ("Prédicteur numérique", "Température de l'air en degrés Celsius.",
                            "Conserver ; relation directe avec la cible (neige si T ≤ 0 °C)."),
        "Apparent Temperature (C)": ("Prédicteur numérique", "Température ressentie (wind chill / indice de chaleur).",
                                     "Très corrélée à Temperature : tester l'une ou l'autre pour éviter la redondance."),
        "Humidity": ("Prédicteur numérique", "Humidité relative, entre 0 et 1.",
                     "Conserver telle quelle (déjà normalisée)."),
        "Wind Speed (km/h)": ("Prédicteur numérique", "Vitesse du vent en km/h.",
                              "Conserver ; distribution à queue lourde, envisager une transformation log."),
        "Wind Bearing (degrees)": ("Prédicteur circulaire", "Direction d'origine du vent en degrés (0-360°).",
                                   "Variable circulaire : préférer sin/cos plutôt que la valeur brute."),
        "Visibility (km)": ("Prédicteur numérique", "Visibilité horizontale en kilomètres.",
                            "Conserver après vérification des valeurs extrêmes."),
        "Loud Cover": ("Constant", "Couverture nuageuse : toujours 0 dans ce jeu de données.",
                       "Supprimer : aucune variance, aucun apport au modèle."),
        "Pressure (millibars)": ("Prédicteur numérique", "Pression atmosphérique en millibars ; 0 = capteur non renseigné.",
                                 "Remplacer les 0 par NaN puis imputer ou exclure la variable."),
        "Daily Summary": ("Prédicteur catégoriel", "Résumé quotidien du temps, redondant avec Summary.",
                          "Étudier la redondance avec Summary avant de conserver."),
    },
    "covid": {
        "Sex": ("Prédicteur catégoriel", "Sexe du patient (0 / 1).", "Encoder en binaire ; faible cardinalité."),
        "Age": ("Prédicteur numérique", "Âge du patient en années.", "Conserver ; mise à l'échelle recommandée."),
        "CA": ("Prédicteur numérique", "Calcium sérique (CA).", "Imputer les 4 % de manquants puis standardiser."),
        "CK": ("Prédicteur numérique", "Créatine kinase, marqueur de lésion musculaire.",
               "54 % de valeurs manquantes : imputation médiane ou retrait à évaluer."),
        "CREA": ("Prédicteur numérique", "Créatinine, marqueur de la fonction rénale.", "Imputer les 4 % de manquants."),
        "ALP": ("Prédicteur numérique", "Phosphatase alcaline, marqueur hépatique / osseux.", "Imputer les 24 % de manquants."),
        "GGT": ("Prédicteur numérique", "Gamma-glutamyl transférase, enzyme hépatique.", "Imputer les 26 % de manquants."),
        "GLU": ("Prédicteur numérique", "Glycémie (glucose sanguin).", "Imputer et vérifier les valeurs physiologiques extremes."),
        "AST": ("Prédicteur numérique", "Aspartate aminotransférase, enzyme hépatique.", "Imputer les 4 % de manquants."),
        "ALT": ("Prédicteur numérique", "Alanine aminotransférase, enzyme hépatique.", "Imputer les 4 % de manquants."),
        "LDH": ("Prédicteur numérique", "Lactate déshydrogénase, marqueur d'inflammation / lyse cellulaire.",
                "Imputer les 10 % de manquants ; distribution à queue lourde."),
        "PCR": ("Prédicteur numérique", "Protéine C-réactive (CRP), marqueur d'inflammation.",
                "Imputer les 6 % de manquants ; forte asymétrie à traiter."),
        "KAL": ("Prédicteur numérique", "Potassium sérique (Kalium).", "Imputer les 4 % de manquants."),
        "NAT": ("Prédicteur numérique", "Sodium sérique (Natrium).", "Imputer les 4 % de manquants."),
        "UREA": ("Prédicteur numérique", "Urée sanguine, marqueur de la fonction rénale.", "44 % de manquants : imputation prudente."),
        "WBC": ("Prédicteur numérique", "Leucocytes (globules blancs), 10⁹/L.", "Conserver ; standardiser."),
        "RBC": ("Prédicteur numérique", "Érythrocytes (globules rouges), 10¹²/L.", "Conserver ; standardiser."),
        "HGB": ("Prédicteur numérique", "Hémoglobine, g/dL.", "Conserver ; vérifier les valeurs extrêmes."),
        "HCT": ("Prédicteur numérique", "Hématocrite, proportion de globules rouges.", "Conserver ; corrélé à HGB/HCT."),
        "MCV": ("Prédicteur numérique", "Volume globulaire moyen, fL.", "Conserver ; une mise à l'échelle reste recommandée."),
        "MCH": ("Prédicteur numérique", "Teneur corpusculaire moyenne en hémoglobine, pg.", "Conserver."),
        "MCHC": ("Prédicteur numérique", "Concentration corpusculaire moyenne en hémoglobine, g/dL.", "Conserver."),
        "PLT1": ("Prédicteur numérique", "Plaquettes (thrombocytes), 10⁹/L.", "Conserver ; vérifier les queues de distribution."),
        "NE": ("Prédicteur numérique", "Neutrophiles, pourcentage de la formule leucocytaire.", "Conserver ; corrélé à NET."),
        "LY": ("Prédicteur numérique", "Lymphocytes, pourcentage de la formule leucocytaire.", "Conserver ; corrélé à LYT."),
        "MO": ("Prédicteur numérique", "Monocytes, pourcentage de la formule leucocytaire.", "Conserver ; corrélé à MOT."),
        "EO": ("Prédicteur numérique", "Éosinophiles, pourcentage de la formule leucocytaire.", "Conserver ; corrélé à EOT."),
        "BA": ("Prédicteur numérique", "Basophiles, pourcentage de la formule leucocytaire.", "Conserver ; corrélé à BAT."),
        "NET": ("Prédicteur numérique", "Neutrophiles absolus (count), dérivé de NE et WBC.", "Redondant avec NE : tester sans lui."),
        "LYT": ("Prédicteur numérique", "Lymphocytes absolus (count), dérivé de LY et WBC.", "Redondant avec LY : tester sans lui."),
        "MOT": ("Prédicteur numérique", "Monocytes absolus (count), dérivé de MO et WBC.", "Redondant avec MO : tester sans lui."),
        "EOT": ("Prédicteur numérique", "Éosinophiles absolus (count) ; la chaîne « Missing » cache des valeurs manquantes.",
                "Convertir « Missing » en NaN avant imputation."),
        "BAT": ("Prédicteur numérique", "Basophiles absolus (count) ; la chaîne « Missing » cache des valeurs manquantes.",
                "Convertir « Missing » en NaN avant imputation."),
        "SARSCov": ("Cible", "Résultat SARS-CoV-2 (0 / 1) à prédire.", "Ne jamais utiliser comme prédicteur ; stratifier les splits."),
    },
    "parkinson": {
        "Subject_id": ("Identifiant", "Identifiant du sujet ; plusieurs lignes peuvent venir du même patient.",
                       "Exclure du modèle et utiliser un split par sujet pour éviter les fuites."),
        "A-beta": ("Prédicteur numérique", "Peptide amyloïde β dans le liquide céphalo-rachidien.",
                   "Marqueur biologique direct ; standardiser."),
        "alpha-syn": ("Prédicteur numérique", "Alpha-synucléine, protéine impliquée dans la maladie de Parkinson.",
                      "Marqueur clé ; standardiser."),
        "p-tau": ("Prédicteur numérique", "Protéine tau phosphorylée (p-tau).", "Conserver ; lien avec la neurodégénérescence."),
        "T-tau": ("Prédicteur numérique", "Protéine tau totale.", "Conserver ; corrélée à p-tau."),
        "P-tau/A-beta": ("Prédicteur dérivé", "Ratio p-tau / A-beta, marqueur combiné de pathologie amyloïde.",
                         "Déjà normalisé ; vérifier la redondance avec ses composantes."),
        "T-tau/A-beta": ("Prédicteur dérivé", "Ratio T-tau / A-beta.", "Conserver ou tester sans les variables sources."),
        "P-tau/t-tau": ("Prédicteur dérivé", "Ratio p-tau / T-tau, indicateur de phosphorylation relative.",
                        "Conserver ; échelle très différente des autres colonnes."),
        "UPSIT": ("Prédicteur numérique", "Score olfactif UPSIT (University of Pennsylvania Smell Identification Test).",
                  "Le trouble olfactif est un signe précoce : prédicteur très informatif."),
        "REMBDSQ": ("Prédicteur numérique", "Score du questionnaire sur le trouble du comportement en sommeil paradoxal (RBD).",
                    "Signe prodromal connu ; conserver."),
        "CAUDATE_R": ("Prédicteur numérique", "Volume IRM du noyau caudé droit.", "Conserver ; mise à l'échelle recommandée."),
        "CAUDATE_L": ("Prédicteur numérique", "Volume IRM du noyau caudé gauche.", "Conserver ; symétrie avec CAUDATE_R."),
        "PUTAMEN_R": ("Prédicteur numérique", "Volume IRM du putamen droit.", "Conserver ; zone touchée dans Parkinson."),
        "PUTAMEN_L": ("Prédicteur numérique", "Volume IRM du putamen gauche.", "Conserver ; zone touchée dans Parkinson."),
        "Group (Normal/PD/SWEDD - 1/2/3)": ("Cible", "Diagnostic : 1 = Normal, 2 = Parkinson (PD), 3 = SWEDD.",
                                             "Ne jamais utiliser comme prédicteur ; encoder les 3 classes."),
    },
    "alcohol": {
        "ALCOHOL": ("Prédicteur numérique", "Concentration d'alcool mesurée par le capteur (g/L ou % selon l'appareil).",
                    "Distribution très asymétrique : transformation log ou robust scaling."),
        "TEMP_AMB": ("Prédicteur numérique", "Température ambiante en °C au moment de la mesure.",
                     "Conserver ; faible variance dans l'échantillon."),
        "TEMP_FAC_MAX": ("Prédicteur numérique", "Température faciale maximale relevée par capteur infrarouge.",
                         "Conserver ; peut capter l'effet vasodilatateur de l'alcool."),
        "TEMP_FAC_MIN": ("Prédicteur numérique", "Température faciale minimale relevée par capteur infrarouge.",
                         "Conserver ; former un écart avec TEMP_FAC_MAX si utile."),
        "EYES": ("Prédicteur numérique", "Indicateur de l'état / mouvement des yeux (somnolence, mesure oculaire).",
                 "Conserver ; complète les signes physiques."),
        "CLASS": ("Cible", "Classe à prédire : 0 = sobre, 1 = sous l'effet de l'alcool.",
                  "Ne jamais utiliser comme prédicteur ; équilibrer les classes si déséquilibrées."),
    },
    "heart": {
        "age": ("Prédicteur numérique", "Âge du patient en années.", "Conserver ; standardiser."),
        "sex": ("Prédicteur catégoriel", "Sexe du patient (1 = homme, 0 = femme selon le codage UCI).", "Encoder en binaire."),
        "chest pain type": ("Prédicteur catégoriel", "Type de douleur thoracique (1 = angine typique, 4 = asymptomatique, codage UCI).",
                            "Encoder en one-hot ; 4 = asymptomatique est le profil le plus à risque."),
        "resting bp s": ("Prédicteur numérique", "Pression artérielle au repos, en mm Hg.",
                         "Conserver ; vérifier les valeurs physiologiques."),
        "cholesterol": ("Prédicteur numérique", "Cholestérol sérique en mg/dL ; 0 = valeur manquante masquée.",
                        "Remplacer les 0 par NaN puis imputer (piège majeur de ce jeu)."),
        "fasting blood sugar": ("Prédicteur binaire", "Glycémie à jeun > 120 mg/dL (0 / 1).", "Conserver tel quel."),
        "resting ecg": ("Prédicteur catégoriel", "Résultat de l'ECG au repos (0 = normal, 1 = ST-T anormal, 2 = hypertrophie).",
                        "Encoder en one-hot."),
        "max heart rate": ("Prédicteur numérique", "Fréquence cardiaque maximale atteinte au test d'effort.",
                           "Conserver ; valeur basse = facteur de risque."),
        "exercise angina": ("Prédicteur binaire", "Angine provoquée par l'effort (0 / 1).", "Conserver tel quel."),
        "oldpeak": ("Prédicteur numérique", "Dépression du segment ST à l'effort, en mm (indicateur d'ischémie).",
                    "Conserver ; peut être négatif, ne pas supprimer ces valeurs."),
        "ST slope": ("Prédicteur catégoriel", "Pente du segment ST au pic d'effort (1 = ascendante, 2 = plate, 3 = descendante).",
                     "Encoder en one-hot ; 3 = descendante est le signe ischémique le plus fort."),
        "target": ("Cible", "Présence de maladie coronarienne (0 = absent, 1 = présent).",
                   "Ne jamais utiliser comme prédicteur ; lire la balanced accuracy."),
    },
    "electricalload": {
        "datetime": ("Horodatage", "Horodatage horaire de la mesure de charge (2015-01 → 2019-04).",
                     "Trier chronologiquement ; ne pas faire de K-fold aléatoire."),
        "year": ("Composante temporelle", "Année de la mesure.", "Conserver pour capter la tendance longue."),
        "month": ("Composante temporelle", "Mois de l'année (1-12).", "Encoder en cyclique sin/cos ou one-hot."),
        "day": ("Composante temporelle", "Jour du mois (1-31).", "Conserver avec prudence (peu de signal isolé)."),
        "hourOfDay": ("Composante temporelle", "Heure de la journée (0-23), pilote du profil de charge.",
                      "Encoder en cyclique sin/cos pour préserver la continuité 23h → 0h."),
        "dayOfWeek": ("Composante temporelle", "Jour de la semaine (1 = lundi … 7 = dimanche).",
                      "Encoder en cyclique ou one-hot ; distinguer week-end."),
        "weekend": ("Indicateur binaire", "1 si la mesure correspond à un week-end, 0 sinon.",
                    "Conserver ; profil de consommation différent."),
        "holiday": ("Indicateur binaire", "1 si le jour est férié, 0 sinon.",
                    "Conserver ; les jours fériés expliquent certains pics / creux."),
        "Holiday_ID": ("Identifiant catégoriel", "Identifiant du jour férié (ou 0 / vide hors férié).",
                       "Encoder en one-hot ou réduire à holiday pour éviter la fragmentation."),
        "week_X-2": ("Prédicteur décalé", "Charge du même jour, deux semaines auparavant (retard hebdomadaire).",
                     "Très corrélé à la cible : vérifier qu'il est calculé sur le passé uniquement."),
        "week_X-3": ("Prédicteur décalé", "Charge du même jour, trois semaines auparavant.",
                     "Conserver avec les autres retards ; surveiller la redondance."),
        "week_X-4": ("Prédicteur décalé", "Charge du même jour, quatre semaines auparavant.",
                     "Conserver ; complète MA_X-4."),
        "MA_X-4": ("Prédicteur décalé", "Moyenne mobile de la charge sur les 4 dernières semaines.",
                   "Prédicteur principal (r = 0,9205 avec DEMAND) mais sensible aux fuites temporelles."),
        "T2M_toc": ("Prédicteur météo", "Température de l'air à 2 mètres, en °C (source météo).",
                    "Conserver ; explique les pics de chauffage / climatisation."),
        "DEMAND": ("Cible", "Demande électrique horaire en MW, à prédire.",
                   "Ne jamais utiliser comme prédicteur ; split train/test par date."),
        "split": ("Partition", "Indicateur train / test défini chronologiquement (2015-2019 → avril 2019).",
                  "Utiliser uniquement pour séparer les jeux ; exclure des features."),
    },
}

# Noms alternatifs présents dans la table SQLite avant normalisation des noms.
_ALIASES = {
    "week_X2": "week_X-2", "week_X3": "week_X-3", "week_X4": "week_X-4",
    "MA_X4": "MA_X-4",
}


def _stretch(fn, *args, **kwargs):
    try:
        return fn(*args, width="stretch", **kwargs)
    except TypeError:
        return fn(*args, use_container_width=True, **kwargs)


def _meta_from_db(key: str, rep: dict) -> dict:
    try:
        dataset_id = dd.resolve_dataset_id(dd.ROOT / "data.db", key, rep)
        if not dataset_id:
            return {}
        con = sqlite3.connect(f"file:{(dd.ROOT / 'data.db').as_posix()}?mode=ro", uri=True)
        row = con.execute(
            "select columns_json, dtypes_json, missing_json, describe_json from datasets where id = ?",
            (dataset_id,)).fetchone()
        con.close()
        if not row:
            return {}
        return {
            "id": dataset_id,
            "columns": json.loads(row[0] or "[]"),
            "dtypes": json.loads(row[1] or "{}"),
            "missing": json.loads(row[2] or "{}"),
            "describe": json.loads(row[3] or "{}"),
        }
    except (sqlite3.Error, json.JSONDecodeError, TypeError):
        return {}


def _fmt_num(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{f:.4g}"


def _stats_text(col: str, rep: dict, db_meta: dict, num_stats: dict, cat_stats: dict, vc: dict) -> str:
    row = num_stats.get(col)
    if not row:
        row = db_meta.get("describe", {}).get(col)
    if row:
        try:
            txt = (f"min {_fmt_num(row.get('min'))} · méd {_fmt_num(row.get('50%'))} "
                   f"· max {_fmt_num(row.get('max'))}")
        except Exception:
            txt = "—"
        n_unique = row.get("n_unique")
        if n_unique is not None:
            txt += f" · {int(n_unique)} valeurs distinctes"
        return txt
    cat = cat_stats.get(col)
    if cat:
        return f"{cat.get('unique')} modalités · top « {cat.get('top')} » ({cat.get('freq')})"
    if col in vc:
        top = sorted(vc[col].items(), key=lambda kv: -kv[1])[:3]
        return "modalités : " + ", ".join(f"{k} ({v})" for k, v in top)
    return "—"


def _signal_text(col: str, rep: dict) -> str:
    key = _ALIASES.get(col, col)
    if dd.is_electrical(rep):
        for f in (rep.get("noise") or {}).get("features") or []:
            if f.get("feature") == key:
                return (f"NSR {f.get('nsr', 0):.3f} · SNR {f.get('snr_db', 0):.1f} dB "
                        f"· {f.get('verdict', '—')}")
        return "—"
    for f in dd.outlier_features(rep):
        if f.get("feature") == key and f.get("iqr_percent") is not None:
            return f"IQR {f['iqr_percent']:.2f} % d'outliers"
    for f in dd.noise_df(rep).to_dict("records"):
        if f.get("feature") == key:
            return f"bruit {f.get('verdict', '—')} (NSR {f.get('noise_to_signal', 0):.3f})"
    return "—"


def _infer(col: str, rep: dict, dtype: str, target: str | None) -> tuple[str, str, str]:
    low = col.lower()
    if col == target:
        return "Cible", "Variable cible du modèle.", "Ne jamais utiliser comme prédicteur."
    if "id" in low and ("subject" in low or low.endswith("_id")):
        return "Identifiant", "Identifiant d'enregistrement ou de sujet.", "Exclure du modèle (risque de fuite)."
    if any(t in low for t in ("date", "datetime", "year", "month", "day", "hour")):
        return "Composante temporelle", "Composante de date / heure de la mesure.", "Trier chronologiquement et encoder en cyclique."
    if dd.is_electrical(rep) and col == "split":
        return "Partition", "Indicateur train / test.", "Utiliser uniquement pour séparer les jeux."
    if "/" in col or col.startswith(("MA_", "week_X")):
        return "Prédicteur dérivé", "Variable calculée à partir d'autres mesures.", "Vérifier l'absence de fuite avant entraînement."
    if "str" in str(dtype).lower() or "object" in str(dtype).lower():
        return "Prédicteur catégoriel", "Variable qualitative.", "Encoder en one-hot ou cibler les modalités rares."
    return "Prédicteur numérique", "Variable numérique explicative.", "Vérifier l'échelle, les extrêmes et les valeurs manquantes."


def build_column_df(key: str, rep: dict) -> pd.DataFrame:
    """Tableau d'interprétation de toutes les colonnes d'un dataset."""
    desc = rep.get("description") or {}
    db_meta = _meta_from_db(key, rep)

    columns = [c["name"] for c in (desc.get("info", {}) or {}).get("columns", []) if c.get("name")]
    if not columns:
        columns = list(db_meta.get("columns") or [])
    if not columns and dd.is_electrical(rep):
        columns = list((rep.get("preprocessing") or {}).get("columns") or [])

    info_cols = {c["name"]: c for c in (desc.get("info", {}) or {}).get("columns", []) if c.get("name")}
    num_stats = {r["feature"]: r for r in desc.get("describe_numeric", []) or []}
    cat_stats = {r["feature"]: r for r in desc.get("describe_categorical", []) or []}
    vc = desc.get("value_counts") or {}
    roles = desc.get("feature_roles") or {}
    target = dd.target_name(rep) or roles.get("target")
    constant = set(desc.get("constant_columns") or [])

    # Taux de manquants : rapport complet si disponible, sinon aperçu SQLite.
    missing_map: dict[str, float] = {}
    md = rep.get("missing_duplicates") or {}
    for c in md.get("missing", {}).get("columns", []) or []:
        missing_map[c.get("name")] = float(c.get("percent") or 0.0)
    if not missing_map and dd.is_electrical(rep):
        q = rep.get("quality") or {}
        n = int((rep.get("meta") or {}).get("n_rows") or 1)
        all_missing = {**q.get("missing_train", {}), **q.get("missing_test", {})}
        missing_map = {c: 100 * float(v or 0) / n for c, v in all_missing.items()}
    if not missing_map:
        preview_n = max(1, len(desc.get("head") or []) or 50)
        missing_map = {c: 100 * float(db_meta.get("missing", {}).get(c, 0)) / preview_n for c in columns}

    glossary = GLOSSARY.get(key, {})
    rows = []
    for col in columns:
        normalized = _ALIASES.get(col, col)
        role, meaning, advice = glossary.get(normalized, _infer(col, rep,
                                                                info_cols.get(col, {}).get("dtype")
                                                                or db_meta.get("dtypes", {}).get(col, ""), target))
        if col in constant and role != "Cible":
            role, advice = "Constant", "Supprimer : aucune variance."
        if col == target:
            role, meaning, advice = ("Cible", meaning, advice)
        dtype = info_cols.get(col, {}).get("dtype") or db_meta.get("dtypes", {}).get(col, "—")
        rows.append({
            "Colonne": col,
            "Rôle": role,
            "Type": str(dtype),
            "Signification": meaning,
            "Statistiques observées": _stats_text(col, rep, db_meta, num_stats, cat_stats, vc),
            "Manquants %": round(missing_map.get(col, 0.0), 2),
            "Signal (outlier / bruit)": _signal_text(col, rep),
            "Recommandation": advice,
        })
    return pd.DataFrame(rows)


def render_column_dictionary(key: str, rep: dict) -> None:
    """Vue Streamlit : KPI, filtres et tableau interprétatif de chaque colonne."""
    df = build_column_df(key, rep)
    if df.empty:
        st.info("Aucune colonne n'a pu être lue pour ce dataset.")
        return

    n_target = int((df["Rôle"] == "Cible").sum())
    n_missing = int((df["Manquants %"] > 0).sum())
    n_id = int(df["Rôle"].isin(["Identifiant", "Constant", "Partition"]).sum())
    kpi_row([
        ("Colonnes documentées", len(df), "dictionnaire métier par variable", "good"),
        ("Variables cibles", n_target, dd.label(key), "info"),
        ("Colonnes avec manquants", n_missing, f"sur {len(df)} colonnes", "warn" if n_missing else "good"),
        ("À exclure / constantes", n_id, "identifiants, partitions, constantes", "info"),
    ])

    section_title("Dictionnaire des colonnes",
                  "rôle métier, statistiques observées et recommandation de prétraitement")
    c1, c2 = st.columns([2, 1])
    with c1:
        search = st.text_input("Rechercher une colonne ou un mot-clé", "", key=f"coldict_search_{key}",
                               placeholder="Ex. température, age, DEMAND…")
    with c2:
        role_filter = st.selectbox("Filtrer par rôle", ["Tous"] + sorted(df["Rôle"].unique().tolist()),
                                   key=f"coldict_role_{key}")

    shown = df
    if search.strip():
        q = search.strip().lower()
        shown = shown[shown.apply(lambda r: q in " ".join(str(v).lower() for v in r), axis=1)]
    if role_filter != "Tous":
        shown = shown[shown["Rôle"] == role_filter]

    st.caption(f"{len(shown)} colonne(s) affichée(s) sur {len(df)}. "
               "Les statistiques proviennent du rapport EDA complet lorsqu'il est disponible, "
               "sinon de l'aperçu stocké dans data.db.")
    _stretch(st.dataframe, shown.reset_index(drop=True), hide_index=True)

    with st.expander("📖 Voir la définition détaillée d'une colonne"):
        selected = st.selectbox("Colonne", df["Colonne"].tolist(), key=f"coldict_detail_{key}")
        row = df[df["Colonne"] == selected].iloc[0]
        st.markdown(f"**{row['Colonne']}** · *{row['Rôle']}* · `{row['Type']}`")
        st.write(row["Signification"])
        st.caption(f"Statistiques : {row['Statistiques observées']} · "
                   f"Manquants : {row['Manquants %']} % · Signal : {row['Signal (outlier / bruit)']}")
        st.info(row["Recommandation"])

    csv = df.to_csv(index=False).encode("utf-8-sig")
    _stretch(st.download_button, f"⬇️ Télécharger le dictionnaire « {key} » (CSV)",
             csv, file_name=f"dictionnaire_colonnes_{key}.csv", mime="text/csv",
             key=f"coldict_dl_{key}")
