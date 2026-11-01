"""app.py : application Streamlit d'analyse météo.

Phase 0 - Vue d'ensemble   : importer, sauvegarder et explorer les datasets.
Phase 1+ - Rapports EDA    : afficher les rapports enregistres dans weather.db.

La barre laterale permet de naviguer entre les deux phases.

Lancer :  streamlit run app.py
Autre base : WEATHER_DB=/chemin/weather.db streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from theme import apply_theme, brand
from weather_view import DEFAULT_DB, render_weather
from dataset_overview import render_dataset_overview

st.set_page_config(page_title="Analyse météo", layout="wide")
apply_theme()
brand("Analyse météo · ML")

# ── Navigation laterale ───────────────────────────────────────────────────────
PAGES = {
    "📊 Vue d'ensemble des datasets": "overview",
    "🔬 Rapports d'analyse EDA":       "reports",
}

page_label = st.sidebar.radio(
    "Navigation",
    list(PAGES.keys()),
    key="app_page",
    label_visibility="collapsed",
)
page = PAGES[page_label]
st.sidebar.divider()

# ── Routing ───────────────────────────────────────────────────────────────────
if page == "overview":
    render_dataset_overview(DEFAULT_DB)
else:
    render_weather(DEFAULT_DB, show_header=True)
