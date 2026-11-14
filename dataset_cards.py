"""dataset_cards.py : page d'accueil, grille de cartes de datasets (image + nom).

Chaque carte montre l'image du dataset ; le bouton « Ouvrir » sélectionne le dataset.
Pour utiliser vos propres images : déposez assets/datasets/<clé>.png|jpg|webp|svg
(clés : weather, covid, parkinson, alcohol, heart) ; elles remplacent les illustrations par défaut.
"""
from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

from theme import esc

ASSETS = Path(__file__).resolve().parent / "assets" / "datasets"
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".svg": "image/svg+xml"}

_BG = ('<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{a}"/>'
       '<stop offset="1" stop-color="{b}"/></linearGradient></defs><rect width="400" height="220" fill="url(#g)"/>'
       '<circle cx="360" cy="30" r="90" fill="#fff" opacity=".07"/><circle cx="30" cy="210" r="70" fill="#fff" opacity=".06"/>')

_ART = {
    # soleil + nuage + gouttes
    "weather": ("#0E6B6F", "#14A3A8",
                '<circle cx="165" cy="95" r="34" fill="#FFD166"/>'
                + "".join(f'<line x1="165" y1="95" x2="{165 + 52 * c:.0f}" y2="{95 + 52 * s:.0f}" stroke="#FFD166" stroke-width="5" stroke-linecap="round"/>'
                          for c, s in [(1, 0), (.7, .7), (0, 1), (-.7, .7), (-1, 0), (-.7, -.7), (0, -1), (.7, -.7)])
                + '<circle cx="165" cy="95" r="34" fill="#FFD166"/>'
                '<path d="M170 150a28 28 0 0 1 8-54 38 38 0 0 1 72 10 26 26 0 0 1 4 44z" fill="#fff" opacity=".95"/>'
                '<g stroke="#fff" stroke-width="5" stroke-linecap="round"><line x1="195" y1="168" x2="189" y2="182"/>'
                '<line x1="220" y1="168" x2="214" y2="182"/><line x1="245" y1="168" x2="239" y2="182"/></g>'),
    # particule virale
    "covid": ("#2B5F8A", "#3B8FB6",
              '<g transform="translate(200 110)">'
              + "".join(f'<g transform="rotate({a})"><line x1="0" y1="-38" x2="0" y2="-62" stroke="#fff" stroke-width="5"/>'
                        f'<circle cx="0" cy="-66" r="8" fill="#fff"/></g>' for a in range(0, 360, 30))
              + '<circle r="40" fill="#fff"/><circle cx="-12" cy="-8" r="7" fill="#3B8FB6"/>'
                '<circle cx="14" cy="10" r="9" fill="#3B8FB6"/><circle cx="-6" cy="16" r="5" fill="#3B8FB6"/></g>'),
    # réseau de neurones
    "parkinson": ("#5B4B8A", "#8B6FC0",
                  '<g stroke="#fff" stroke-width="3" opacity=".85"><line x1="95" y1="80" x2="185" y2="55"/>'
                  '<line x1="95" y1="80" x2="170" y2="130"/><line x1="185" y1="55" x2="270" y2="90"/>'
                  '<line x1="170" y1="130" x2="270" y2="90"/><line x1="170" y1="130" x2="230" y2="175"/>'
                  '<line x1="270" y1="90" x2="320" y2="140"/><line x1="230" y1="175" x2="320" y2="140"/>'
                  '<line x1="95" y1="80" x2="110" y2="165"/><line x1="110" y1="165" x2="230" y2="175"/></g>'
                  '<g fill="#fff"><circle cx="95" cy="80" r="14"/><circle cx="185" cy="55" r="11"/>'
                  '<circle cx="170" cy="130" r="17"/><circle cx="270" cy="90" r="13"/>'
                  '<circle cx="230" cy="175" r="12"/><circle cx="320" cy="140" r="10"/><circle cx="110" cy="165" r="9"/></g>'),
    # verre + bulles
    "alcohol": ("#B06A1F", "#E0A040",
                '<path d="M140 50h120l-18 100q-3 14-17 14h-50q-14 0-17-14z" fill="#fff" opacity=".95"/>'
                '<path d="M149 95h102l-9 55q-3 14-17 14h-50q-14 0-17-14z" fill="#F4B942"/>'
                '<rect x="193" y="164" width="14" height="30" fill="#fff"/><rect x="165" y="192" width="70" height="8" rx="4" fill="#fff"/>'
                '<g fill="#fff" opacity=".9"><circle cx="180" cy="125" r="5"/><circle cx="210" cy="140" r="4"/>'
                '<circle cx="222" cy="112" r="5"/><circle cx="195" cy="105" r="3"/></g>'),
    # coeur + électrocardiogramme
    "heart": ("#A8322B", "#D9594C",
              '<path d="M200 170C120 120 110 70 150 55c25-9 42 5 50 22 8-17 25-31 50-22 40 15 30 65-50 115z" fill="#fff" opacity=".95"/>'
              '<polyline points="60,125 140,125 160,95 185,155 210,70 235,125 340,125" fill="none" stroke="#A8322B" '
              'stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>'),
}
_DEFAULT = ("#0E6B6F", "#14A3A8", '<rect x="140" y="60" width="30" height="100" rx="6" fill="#fff"/>'
            '<rect x="185" y="90" width="30" height="70" rx="6" fill="#fff"/><rect x="230" y="40" width="30" height="120" rx="6" fill="#fff"/>')


def _data_uri(key: str) -> str:
    for ext, mime in MIME.items():
        f = ASSETS / f"{key}{ext}"
        if f.exists():
            return f"data:{mime};base64,{base64.b64encode(f.read_bytes()).decode()}"
    a, b, art = _ART.get(key, _DEFAULT)
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 220">{_BG.format(a=a, b=b)}{art}</svg>'
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


CARD_CSS = """<style>
.dc { border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; background: var(--soft);
  transition: transform .15s ease, border-color .15s ease; }
.dc:hover { transform: translateY(-3px); border-color: var(--accent-2); }
.dc img { width: 100%; height: 150px; object-fit: cover; display: block; border-radius: 0 !important; }
.dc-b { padding: .75rem .95rem .85rem; }
.dc-t { font-family: 'Instrument Serif', Georgia, serif; font-size: 1.45rem; line-height: 1.1; }
.dc-m { font-size: .76rem; opacity: .65; margin-top: .25rem; }
</style>"""


def render_dataset_cards(items: list[dict], state_key: str = "home_dataset", cols: int = 3) -> None:
    """items : [{key, name, rows, cols, subtitle}]. Un clic sur « Ouvrir » écrit la clé dans st.session_state[state_key]."""
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    for start in range(0, len(items), cols):
        row = st.columns(cols, gap="medium")
        for col, it in zip(row, items[start:start + cols]):
            with col:
                st.markdown(
                    f'<div class="dc"><img src="{_data_uri(it["key"])}" alt="{esc(it["name"])}"/>'
                    f'<div class="dc-b"><div class="dc-t">{esc(it["name"])}</div>'
                    f'<div class="dc-m">{esc(it.get("subtitle", ""))}</div></div></div>',
                    unsafe_allow_html=True)
                if st.button("Ouvrir ›", key=f"open_{it['key']}", use_container_width=True):
                    st.session_state[state_key] = it["key"]
                    st.rerun()
