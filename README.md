# Analyse météo : modèle, base SQLite et application Streamlit

Architecture MVC : le **Model** calcule et stocke, la **vue** affiche.

| Fichier | Rôle |
|---|---|
| `weather_model.py` | Model : analyse complète de `weatherHistory.csv` (description, manquants, extrêmes, bruit, tendances, corrélations, distributions, modèle Random Forest). |
| `quality.py` | Règles de qualité partagées (plages physiques, seuils de bruit, nettoyage). |
| `data.py` | Couche de stockage SQLite (`data.db`) : datasets, rapports JSON par section, insights et graphiques PNG (BLOB). Chaque exécution **ajoute** un rapport. |
| `weather_db.py` | Alias de compatibilité de `data.DataDB` (ne liste que les rapports météo). |
| `weather_view.py` | Vue en lecture seule d'un rapport (un onglet par section). |
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
streamlit run app.py
```

Autre base : `DATA_DB=/chemin/data.db streamlit run app.py`.

## Créer un nouveau rapport

```bash
pip install -r requirements-model.txt
python weather_db.py weatherHistory.csv data.db
# ou, pour les quatre datasets du projet :
python run_all_models.py
```

Le rapport est ajouté à `data.db` ; dans l'application, « Actualiser la liste » le fait apparaître.
Pour réimporter les `report.json` déjà générés : `python data.py ingest`.
Pour ne garder que les N derniers rapports d'un dataset : `db.save_report(report, keep_last=N)`.

## Notes sur la phase modèle

- Cible `Precip Type`, très déséquilibrée : `class_weight="balanced"`, on lit l'exactitude **équilibrée** et le F1.
- Évaluation sur une coupure **temporelle** (75 % passé / 25 % récent).
- Importance par **permutation** sur la période de test (plus fiable que l'impureté quand `Temperature` et `Apparent Temperature` sont presque identiques).
- Sur ce jeu, `Precip Type` suit une règle de température (neige si T ≤ 0 °C, pluie sinon) : le score est parfait et la prédiction triviale.
