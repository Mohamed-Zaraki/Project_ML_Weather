"""app.py : application Streamlit d'analyse météo.

Affiche, en lecture seule, les rapports enregistrés dans weather.db (produits par weather_model.py).
La barre latérale liste TOUS les rapports sauvegardés ; le rapport choisi s'affiche en onglets.

Lancer :  streamlit run app.py
Autre base : WEATHER_DB=/chemin/weather.db streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from theme import apply_theme, brand
from weather_view import DEFAULT_DB, render_weather

st.set_page_config(page_title="Analyse météo", layout="wide")
apply_theme()
brand("rapports d'analyse météo")
render_weather(DEFAULT_DB)
