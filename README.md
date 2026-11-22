# Analyse météo : modèle, base SQLite et application Streamlit

Architecture MVC : le **Model** calcule et stocke, la **vue** affiche.

| Fichier | Rôle |
|---|---|
| `weather_model.py` | Model : analyse complète de `weatherHistory.csv` (description, manquants, extrêmes, bruit, tendances, corrélations, distributions, modèle Random Forest). |
| `quality.py` | Règles de qualité partagées (plages physiques, seuils de bruit, nettoyage). |
| `data.py` | Couche de stockage SQLite (`data.db`) : datasets, rapports JSON par section, insights et graphiques PNG (BLOB). Chaque exécution **ajoute** un rapport. |
| `weather_db.py` | Alias de compatibilité de `data.DataDB` (ne liste que les rapports météo). |
| `weather_view.py` | Vue en lecture seule d'un rapport (un onglet par section). |
| `dataset_overview.py` | Fiche d'un dataset : aperçu, types, statistiques et export CSV. |
| `column_dictionary.py` | Dictionnaire métier : rôle, signification et recommandation pour **chaque colonne** des 6 datasets. |
| `workplan_view.py` | Plan de travail global de la page d'accueil (checklist interactive, état sessionnel). |
| `dataset_workplan_view.py` | Plan de travail **propre au dataset ouvert**, calculé à partir des artefacts présents. |
| `outlier_view.py` | Onglet outliers/bruit des datasets génériques : KPI, boxplots, détecteurs et bruit par variable. |
| `outlier_noise_interpretation.py` | Guide de lecture et synthèse interprétative partagée des outliers et du bruit. |
| `electrical_load_view.py` | Vues spécifiques ElectricalLoad : bruit/outliers, corrélations, évaluation R²/RMSE. |
| `project_dashboard.py` | Dashboard principal : accueil, fiches dataset, qualité, outliers, corrélations, modèles et prétraitement. |
| `app.py` | Application : la barre latérale liste tous les rapports enregistrés, le rapport choisi s'affiche. |
| `theme.py` | Habillage visuel. |

## Structure Phase I (recommandations du cours)

| Point du cours | Où le trouver |
|---|---|
| Home Page | `data_quality.dashboard` (KPI prêts pour une page d'accueil) + `data.py stats` |
| Data Quality Indicators | section `data_quality` : score /100 + complétude, doublons, contradictions, hors-plages, bruit, outliers en % |
| Noise (%) | section `noise` : `summary.noisy_feature_percent`, NSR, SNR(dB), spikes |
| Outliers (%) | section `outliers` : `summary.consensus_percent` + 6 détecteurs non supervisés (OCSVM, Isolation Forest, LOF, Elliptic Envelope, DBSCAN, Mahalanobis) |
| Interpretation | section `interpretation` : constats + recommandations priorisées |
| Apply Data Description Recommendations | `interpretation.preprocessing_plan` : actions déjà appliquées vs actions restantes |

## Lancer l'application

```bash
pip install -r requirements.txt
streamlit run project_dashboard.py
# ou l'application historique, centrée sur les rapports :
streamlit run app.py
```

Autre base : `DATA_DB=/chemin/data.db streamlit run app.py`.

### Plans de travail et dictionnaire des colonnes

Dans `project_dashboard.py`, ouvrir un dataset puis l'onglet **📋 Données** affiche, pour ce
dataset uniquement :

1. **Plan de travail du dataset** — 10 étapes réparties sur 4 phases (ingestion, audit/EDA,
   modélisation, restitution). Les cases pré-cochées reflètent les artefacts réellement présents
   (catalogue SQLite, rapport EDA, graphiques, métriques, dictionnaire). L'état est sessionnel :
   « Réinitialiser le plan du dataset » restaure les coches calculées.
2. **Dictionnaire des colonnes** — une ligne par variable avec son rôle (cible, identifiant,
   horodatage, partition, prédicteur…), sa signification métier en français, son type, ses
   statistiques observées, son taux de manquants, son signal bruit/outliers et sa recommandation
   de prétraitement. La recherche, le filtre par rôle, la fiche détaillée et l'export CSV sont
   disponibles. Les définitions sont curées pour les 6 datasets et complétées par des heuristiques
   pour toute colonne inconnue.

### Interprétation des outliers et du bruit

L'onglet **🔬 Outliers & Bruit** affiche désormais, pour chaque dataset, une interprétation partagée :

- une synthèse chiffrée (IQR moyen, consensus multivarié, variables bruitées) ;
- un guide de lecture des notions : outlier, barrières IQR, Z-score, consensus, NSR, SNR et seuils ;
- un tableau « Interprétation par variable » avec constat et action recommandée ;
- une liste de décisions concrètes (supprimer ou conserver, lisser, vérifier une fuite temporelle…).

Pour ElectricalLoad, l'anomalie étant détectée globalement par 4 détecteurs sans taux par colonne,
le tableau présente le bruit NSR/SNR de chaque variable et rappelle que les pointes de charge
sont des événements métier à ne pas effacer aveuglément.

Le plan global de la page d'accueil reste accessible séparément dans **📊 Vue d'ensemble**.

## Créer un nouveau rapport

```bash
pip install -r requirements-model.txt
python weather_db.py weatherHistory.csv data.db
# ou, pour les cinq datasets du projet (4 classifications + 1 régression) :
python run_all_models.py
# charge électrique seule :
python run_all_models.py electricalload
```

`run_all_models.py` traite COVID-19, Parkinson, alcool, cœur et ElectricalLoad. Pour ce dernier,
`electrical_load_pipeline.py` lit `eda_output_electricalload/report.json` et les métriques
`evaluation_electricalload/metrics_summary.csv` (R²/RMSE). Tant que `datasets_extracted/_APERCU_ONLY.txt`
est présent, il **ne régénère pas** le rapport complet à partir des aperçus de 50 lignes.

Le rapport est ajouté à `data.db` ; dans l'application, « Actualiser la liste » le fait apparaître.
Pour réimporter les `report.json` déjà générés : `python data.py ingest`.
Pour ne garder que les N derniers rapports d'un dataset : `db.save_report(report, keep_last=N)`.

## Aperçus des datasets (dossiers de données brutes absents)

`datasets_extracted/` est dans `.gitignore` : les CSV/XLSX d'origine ne sont donc pas versionnés.
Les datasets existent néanmoins dans `data.db` (table `datasets` : 65 jeux, 358 889 lignes cataloguées,
aperçus de 50 lignes et statistiques). Pour les rendre consultables comme fichiers :

```bash
python export_dataset_previews.py
```

Cela crée :
- `data/previews/<id>__<fichier>.csv` pour chaque dataset et un `manifest.csv` de correspondance
  (catégorie, nom de fichier d'origine, nombre de lignes réelles, nombre de lignes exportées) ;
- les chemins attendus dans `datasets_extracted/` (ex. `Covid19/dataset2-covid19.csv`,
  `parkinson/parkinson/data1.csv`) avec ces mêmes aperçus, **sans écraser un vrai fichier existant** ;
- un marqueur `datasets_extracted/_APERCU_ONLY.txt` rappelant que ce sont des aperçus de 50 lignes.

⚠️ Ces fichiers ne sont pas les jeux complets. Remplacez-les par les vrais fichiers bruts puis supprimez
`_APERCU_ONLY.txt` avant de relancer `python populate_datasets.py` (qui remplace le catalogue).
Tant que le marqueur est présent — ou si aucun fichier source n'est trouvé — `populate_datasets.py`
refuse de vider `data.db`.

## Notes sur la phase modèle

- Cible `Precip Type`, très déséquilibrée : `class_weight="balanced"`, on lit l'exactitude **équilibrée** et le F1.
- Évaluation sur une coupure **temporelle** (75 % passé / 25 % récent).
- Importance par **permutation** sur la période de test (plus fiable que l'impureté quand `Temperature` et `Apparent Temperature` sont presque identiques).
- Sur ce jeu, `Precip Type` suit une règle de température (neige si T ≤ 0 °C, pluie sinon) : le score est parfait et la prédiction triviale.

## Phase II : évaluation des modèles (`model_lab.py`)

Le tableau de bord expose un onglet **🧪 Éval. Modèles** qui reprend le protocole du projet
`Projet2_Pre_Learning_ML` pour les quatre datasets médicaux. La charge électrique, problème de
régression, a son propre onglet avec R²/RMSE (`eval_electricalload.py`) :

| Élément | Détail |
|---|---|
| Algorithmes | KNN, Random Forest (Ensemble), SVM |
| Réglage | boucles de validation croisée explicites (k = 1…15, n_estimators = 25…200, noyaux rbf/linear/poly et C) — aucune GridSearch cachée |
| Prétraitement | imputation médiane + OneHotEncoder + scaler choisi automatiquement (Standard / MinMax / Robust) |
| Métriques | accuracy train/test, balanced accuracy, précision/rappel/F1 macro, ROC-AUC (binaire ou micro-moyenne), matrice de confusion |
| Diagnostic | écart train/test : Well-Balanced / Moderate overfitting / High overfitting / Underfitting |
| Robustesse | baseline, sans outliers (IQR), bruit gaussien 18 %, bruit → lissage gaussien, outliers injectés |
| Artefacts | `storage/model_lab/*.pkl` (split + pipelines ajustés) et un `*_lab.json` de métadonnées |

Le calcul se fait hors ligne dans `run_all_models.py` (section `model_lab` du rapport) ;
le dashboard reste en lecture seule. Pour régénérer un dataset :

```bash
python run_all_models.py alcohol
```

