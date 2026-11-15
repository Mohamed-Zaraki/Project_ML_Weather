"""dataset_cards.py : page d'accueil, grille de cartes de datasets (illustrations vectorielles + nom).

Chaque carte montre l'illustration représentative du problème ; le bouton « Ouvrir » sélectionne le dataset.
Pour utiliser vos propres images : déposez assets/datasets/<clé>.png|jpg|webp|svg
(clés : weather, covid, parkinson, alcohol, heart) ; elles remplaceront automatiquement ces illustrations.
"""
from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

from theme import esc

ASSETS = Path(__file__).resolve().parent / "assets" / "datasets"
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".svg": "image/svg+xml"}

_BG = (
    '<defs>'
    '<linearGradient id="bg_{key}" x1="0%" y1="0%" x2="100%" y2="100%">'
    '<stop offset="0%" stop-color="{a}"/>'
    '<stop offset="100%" stop-color="{b}"/>'
    '</linearGradient>'
    '<filter id="glow_{key}" x="-20%" y="-20%" width="140%" height="140%">'
    '<feGaussianBlur stdDeviation="6" result="blur"/>'
    '<feComposite in="SourceGraphic" in2="blur" operator="over"/>'
    '</filter>'
    '</defs>'
    '<rect width="400" height="220" fill="url(#bg_{key})"/>'
    '<circle cx="360" cy="20" r="110" fill="#ffffff" opacity=".05"/>'
    '<circle cx="40" cy="200" r="90" fill="#ffffff" opacity=".04"/>'
    '<path d="M0 170 Q100 140 200 170 T400 170 L400 220 L0 220 Z" fill="#ffffff" opacity=".03"/>'
)

_ART = {
    # ── METEO : Soleil, nuage pluvieux, éclairs, thermomètre & isobares ──
    "weather": (
        "#0F172A", "#0369A1",
        # Isobares atmosphériques
        '<path d="M-20 70 Q120 40 240 70 T420 50" fill="none" stroke="#38BDF8" stroke-width="1.5" stroke-dasharray="4 4" opacity=".35"/>'
        '<path d="M-20 120 Q100 90 260 120 T420 100" fill="none" stroke="#38BDF8" stroke-width="1.5" stroke-dasharray="4 4" opacity=".25"/>'
        # Soleil radiant doré
        '<g transform="translate(145, 80)">'
        '<circle cx="0" cy="0" r="32" fill="#FBBF24" opacity=".9" filter="url(#glow_weather)"/>'
        + "".join(
            f'<line x1="{40*c:.1f}" y1="{40*s:.1f}" x2="{50*c:.1f}" y2="{50*s:.1f}" stroke="#FDE047" stroke-width="3.5" stroke-linecap="round"/>'
            for c, s in [(1, 0), (.7, .7), (0, 1), (-.7, .7), (-1, 0), (-.7, -.7), (0, -1), (.7, -.7)]
        ) +
        '</g>'
        # Nuage orageux volumétrique
        '<path d="M165 135 a26 26 0 0 1 8 -50 a38 38 0 0 1 70 8 a30 30 0 0 1 20 42 z" fill="#F8FAFC" opacity=".95"/>'
        '<path d="M185 140 a20 20 0 0 1 6 -38 a28 28 0 0 1 54 6 a24 24 0 0 1 14 32 z" fill="#E2E8F0" opacity=".7"/>'
        # Pluie & éclairs
        '<polygon points="205,142 195,165 208,165 198,188 222,158 210,158" fill="#FACC15"/>'
        '<g stroke="#38BDF8" stroke-width="3" stroke-linecap="round">'
        '<line x1="228" y1="148" x2="222" y2="168"/>'
        '<line x1="245" y1="148" x2="239" y2="168"/>'
        '<line x1="262" y1="148" x2="256" y2="168"/>'
        '</g>'
        # Thermomètre digital de précision
        '<g transform="translate(315, 55)">'
        '<rect x="0" y="0" width="16" height="85" rx="8" fill="#1E293B" stroke="#475569" stroke-width="2"/>'
        '<rect x="4" y="25" width="8" height="60" rx="4" fill="#EF4444"/>'
        '<circle cx="8" cy="85" r="14" fill="#EF4444"/>'
        '<circle cx="8" cy="85" r="7" fill="#FCA5A5"/>'
        '<line x1="20" y1="20" x2="26" y2="20" stroke="#94A3B8" stroke-width="2"/>'
        '<line x1="20" y1="40" x2="28" y2="40" stroke="#94A3B8" stroke-width="2"/>'
        '<line x1="20" y1="60" x2="26" y2="60" stroke="#94A3B8" stroke-width="2"/>'
        '</g>'
    ),

    # ── COVID-19 : Particule virale 3D, éprouvette de laboratoire & brin ADN ──
    "covid": (
        "#0F172A", "#0E7490",
        # Brin d'ADN en arrière-plan
        '<path d="M60 40 Q80 80 60 120 T60 200" fill="none" stroke="#06B6D4" stroke-width="3" opacity=".4"/>'
        '<path d="M90 40 Q70 80 90 120 T90 200" fill="none" stroke="#22D3EE" stroke-width="3" opacity=".4"/>'
        + "".join(f'<line x1="62" y1="{y}" x2="88" y2="{y+8}" stroke="#67E8F9" stroke-width="2" opacity=".45"/>' for y in range(50, 190, 22)) +
        # Particule virale sphérique avec spicules
        '<g transform="translate(225, 105)">'
        + "".join(
            f'<g transform="rotate({a})">'
            f'<line x1="0" y1="-38" x2="0" y2="-62" stroke="#22D3EE" stroke-width="4.5" stroke-linecap="round"/>'
            f'<circle cx="0" cy="-66" r="7" fill="#F43F5E"/>'
            f'</g>'
            for a in range(0, 360, 30)
        ) +
        '<circle r="42" fill="#0891B2" opacity=".95"/>'
        '<circle r="36" fill="#06B6D4" opacity=".85"/>'
        '<circle cx="-14" cy="-10" r="8" fill="#F43F5E" opacity=".8"/>'
        '<circle cx="16" cy="12" r="10" fill="#F43F5E" opacity=".8"/>'
        '<circle cx="-8" cy="18" r="6" fill="#F43F5E" opacity=".8"/>'
        '<circle cx="14" cy="-14" r="7" fill="#F43F5E" opacity=".8"/>'
        '</g>'
        # Tube à essai de laboratoire
        '<g transform="translate(330, 45) rotate(20)">'
        '<rect x="0" y="0" width="22" height="110" rx="11" fill="none" stroke="#E2E8F0" stroke-width="3"/>'
        '<rect x="3" y="45" width="16" height="60" rx="8" fill="#06B6D4" opacity=".85"/>'
        '<circle cx="11" cy="70" r="3" fill="#FFFFFF" opacity=".7"/>'
        '<circle cx="8" cy="85" r="2.5" fill="#FFFFFF" opacity=".7"/>'
        '<circle cx="14" cy="95" r="2" fill="#FFFFFF" opacity=".7"/>'
        '</g>'
    ),

    # ── PARKINSON : Réseau neuronal, synapses & ondes de tremblements moteurs ──
    "parkinson": (
        "#1E1B4B", "#6D28D9",
        # Silhouette cérébrale stylisée
        '<path d="M120 150 C90 150 70 120 70 95 C70 60 100 45 135 45 C150 45 165 52 175 62 C190 48 215 45 235 55 C260 65 270 95 265 125 C260 150 235 170 200 170 Z"'
        ' fill="#4C1D95" opacity=".5" stroke="#A78BFA" stroke-width="2"/>'
        # Onde oscillatoire de tremblement (tremor wave ~5Hz)
        '<path d="M30 185 Q50 170 70 185 T110 185 T150 185 Q165 155 180 205 T210 185 Q225 145 240 215 T270 185 T310 185 T370 185"'
        ' fill="none" stroke="#F43F5E" stroke-width="3.5" stroke-linecap="round"/>'
        # Réseau de synapses et neurones interconnectés
        '<g stroke="#C4B5FD" stroke-width="2.5" opacity=".85">'
        '<line x1="110" y1="80" x2="155" y2="65"/><line x1="110" y1="80" x2="140" y2="120"/>'
        '<line x1="155" y1="65" x2="205" y2="80"/><line x1="140" y1="120" x2="205" y2="80"/>'
        '<line x1="140" y1="120" x2="180" y2="145"/><line x1="205" y1="80" x2="245" y2="110"/>'
        '<line x1="180" y1="145" x2="245" y2="110"/><line x1="110" y1="80" x2="100" y2="135"/>'
        '</g>'
        # Nœuds synaptiques luminescents
        '<g fill="#F5F3FF">'
        '<circle cx="110" cy="80" r="8" fill="#DDD6FE"/><circle cx="110" cy="80" r="4" fill="#7C3AED"/>'
        '<circle cx="155" cy="65" r="7" fill="#DDD6FE"/><circle cx="155" cy="65" r="3.5" fill="#7C3AED"/>'
        '<circle cx="140" cy="120" r="9" fill="#DDD6FE"/><circle cx="140" cy="120" r="4.5" fill="#7C3AED"/>'
        '<circle cx="205" cy="80" r="8" fill="#DDD6FE"/><circle cx="205" cy="80" r="4" fill="#7C3AED"/>'
        '<circle cx="180" cy="145" r="7" fill="#DDD6FE"/><circle cx="180" cy="145" r="3.5" fill="#7C3AED"/>'
        '<circle cx="245" cy="110" r="9" fill="#DDD6FE"/><circle cx="245" cy="110" r="4.5" fill="#7C3AED"/>'
        '<circle cx="100" cy="135" r="6" fill="#DDD6FE"/>'
        '</g>'
    ),

    # ── ALCOOL : Thermographie faciale infrarouge & oculométrie optique ──
    "alcohol": (
        "#1C1917", "#B45309",
        # Carte thermique faciale (zones frontales et nasales)
        '<g transform="translate(140, 45)">'
        '<ellipse cx="50" cy="65" rx="42" ry="55" fill="#3B82F6" opacity=".5"/>'
        # Front chaud
        '<ellipse cx="50" cy="35" rx="30" ry="16" fill="#EF4444" opacity=".85" filter="url(#glow_alcohol)"/>'
        '<ellipse cx="50" cy="35" rx="20" ry="10" fill="#FBBF24" opacity=".95"/>'
        # Yeux & joues
        '<circle cx="34" cy="55" r="9" fill="#F59E0B" opacity=".8"/>'
        '<circle cx="66" cy="55" r="9" fill="#F59E0B" opacity=".8"/>'
        '<circle cx="34" cy="55" r="4" fill="#EF4444"/>'
        '<circle cx="66" cy="55" r="4" fill="#EF4444"/>'
        # Zone nasale et buccale
        '<ellipse cx="50" cy="75" rx="10" ry="12" fill="#10B981" opacity=".8"/>'
        '<ellipse cx="50" cy="95" rx="16" ry="8" fill="#3B82F6" opacity=".7"/>'
        '</g>'
        # Réticule de ciblage et capteurs oculaires
        '<g stroke="#F59E0B" stroke-width="2" fill="none">'
        '<circle cx="174" cy="100" r="16" stroke-dasharray="3 3"/>'
        '<line x1="174" y1="80" x2="174" y2="120"/>'
        '<line x1="154" y1="100" x2="194" y2="100"/>'
        '<circle cx="206" cy="100" r="16" stroke-dasharray="3 3"/>'
        '<line x1="206" y1="80" x2="206" y2="120"/>'
        '<line x1="186" y1="100" x2="226" y2="100"/>'
        '</g>'
        # Jauge de taux d'alcool (BAC)
        '<g transform="translate(290, 60)">'
        '<path d="M0 70 A45 45 0 0 1 70 70" fill="none" stroke="#334155" stroke-width="10" stroke-linecap="round"/>'
        '<path d="M0 70 A45 45 0 0 1 45 32" fill="none" stroke="#F59E0B" stroke-width="10" stroke-linecap="round"/>'
        '<line x1="35" y1="70" x2="48" y2="40" stroke="#EF4444" stroke-width="3.5" stroke-linecap="round"/>'
        '<circle cx="35" cy="70" r="6" fill="#E2E8F0"/>'
        '<text x="20" y="92" fill="#FBBF24" font-family="sans-serif" font-size="11" font-weight="bold">BAC %</text>'
        '</g>'
    ),

    # ── CARDIOLOGIE : Cœur humain, tracé ECG dynamique & monitoring ──
    "heart": (
        "#450A0A", "#BE123C",
        # Silhouette cardiaque anatomique stylisée
        '<path d="M200 175 C120 120 100 65 145 48 C175 36 195 55 200 70 C205 55 225 36 255 48 C300 65 280 120 200 175 Z"'
        ' fill="#E11D48" opacity=".95" filter="url(#glow_heart)"/>'
        # Artères coronaires lumineuses
        '<path d="M195 72 Q180 105 165 125 M180 105 Q195 118 190 145" fill="none" stroke="#FECDD3" stroke-width="2.5" opacity=".7"/>'
        # Tracé d'électrocardiogramme (ECG Complexe QRS)
        '<path d="M40 115 L120 115 L135 115 L145 85 L158 155 L175 45 L190 135 L205 115 L280 115 L292 98 L305 125 L320 115 L360 115"'
        ' fill="none" stroke="#10B981" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round"/>'
        # Moniteur cardiaque / Télémétrie
        '<g transform="translate(295, 35)">'
        '<rect x="0" y="0" width="75" height="42" rx="8" fill="#1E293B" stroke="#475569" stroke-width="1.5"/>'
        '<text x="10" y="24" fill="#10B981" font-family="monospace" font-size="18" font-weight="bold">82</text>'
        '<text x="44" y="22" fill="#94A3B8" font-family="sans-serif" font-size="9">BPM</text>'
        '<circle cx="62" cy="12" r="3" fill="#EF4444"/>'
        '</g>'
    ),
}

_DEFAULT = (
    "#0F172A", "#334155",
    '<rect x="140" y="60" width="30" height="100" rx="6" fill="#38BDF8"/>'
    '<rect x="185" y="90" width="30" height="70" rx="6" fill="#818CF8"/>'
    '<rect x="230" y="40" width="30" height="120" rx="6" fill="#34D399"/>'
)


def _data_uri(key: str) -> str:
    for ext, mime in MIME.items():
        f = ASSETS / f"{key}{ext}"
        if f.exists():
            return f"data:{mime};base64,{base64.b64encode(f.read_bytes()).decode()}"
    a, b, art = _ART.get(key, _DEFAULT)
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 220">{_BG.format(key=key, a=a, b=b)}{art}</svg>'
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode()


CARD_CSS = """<style>
.dc {
  border: 1px solid rgba(226, 232, 240, 0.95); border-radius: 20px; overflow: hidden; background: #FFFFFF;
  box-shadow: 0 4px 18px -2px rgba(15, 23, 42, 0.04);
  transition: transform .22s cubic-bezier(0.16, 1, 0.3, 1), border-color .22s ease, box-shadow .22s ease;
  position: relative;
}
.dc:hover {
  transform: translateY(-5px); border-color: var(--accent-2);
  box-shadow: 0 16px 36px -10px rgba(79, 70, 229, 0.18);
}
.dc img { width: 100%; height: 160px; object-fit: cover; display: block; border-radius: 0 !important; }
.dc-b { padding: .95rem 1.15rem 1.05rem; }
.dc-t { font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.25rem; font-weight: 800; line-height: 1.2; letter-spacing: -.02em; color: #0F172A; }
.dc-m { font-size: .8rem; color: #64748B; margin-top: .35rem; line-height: 1.35; }
</style>"""


def render_dataset_cards(items: list[dict], state_key: str = "home_dataset", cols: int = 3) -> None:
    """items : [{key, name, rows, cols, subtitle}]. Un clic sur « Ouvrir » sélectionne le dataset."""
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    for start in range(0, len(items), cols):
        row = st.columns(cols, gap="medium")
        for col, it in zip(row, items[start:start + cols]):
            with col:
                st.markdown(
                    f'<div class="dc"><img src="{_data_uri(it["key"])}" alt="{esc(it["name"])}"/>'
                    f'<div class="dc-b"><div class="dc-t">{esc(it["name"])}</div>'
                    f'<div class="dc-m">{esc(it.get("subtitle", ""))}</div></div></div>',
                    unsafe_allow_html=True,
                )
                if st.button("Explorer ce dataset ›", key=f"open_{it['key']}", use_container_width=True):
                    st.session_state[state_key] = it["key"]
                    st.rerun()
