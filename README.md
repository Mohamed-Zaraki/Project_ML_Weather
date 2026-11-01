# Analyse météo : modèle, base SQLite et application Streamlit

Architecture MVC : le **Model** calcule et stocke, la **vue** affiche.

| Fichier | Rôle |
|---|---|
| `weather_model.py` | Model : analyse complète de `weatherHistory.csv` (description, manquants, extrêmes, bruit, tendances, corrélations, distributions, modèle Random Forest). |
| `quality.py` | Règles de qualité partagées (plages physiques, seuils de bruit, nettoyage). |
| `weather_db.py` | Stockage SQLite (`weather.db`) : un rapport = JSON par section + graphiques PNG. Chaque exécution **ajoute** un rapport. |
| `weather_view.py` | Vue en lecture seule d'un rapport (un onglet par section). |
| `app.py` | Application : la barre latérale liste tous les rapports enregistrés, le rapport choisi s'affiche. |
| `theme.py` | Habillage visuel. |

## Lancer l'application

```bash
pip install -r requirements.txt
streamlit run app.py
```

Autre base : `WEATHER_DB=/chemin/weather.db streamlit run app.py`.

## Créer un nouveau rapport

```bash
pip install -r requirements-model.txt
python weather_db.py weatherHistory.csv weather.db
```

Le rapport est ajouté à `weather.db` ; dans l'application, « Actualiser la liste » le fait apparaître.
Pour ne garder que les N derniers rapports d'un dataset : `db.save_report(report, keep_last=N)`.

## Notes sur la phase modèle

- Cible `Precip Type`, très déséquilibrée : `class_weight="balanced"`, on lit l'exactitude **équilibrée** et le F1.
- Évaluation sur une coupure **temporelle** (75 % passé / 25 % récent).
- Importance par **permutation** sur la période de test (plus fiable que l'impureté quand `Temperature` et `Apparent Temperature` sont presque identiques).
- Sur ce jeu, `Precip Type` suit une règle de température (neige si T ≤ 0 °C, pluie sinon) : le score est parfait et la prédiction triviale.
