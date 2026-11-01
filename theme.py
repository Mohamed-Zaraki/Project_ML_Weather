"""theme.py : habillage visuel partagé (CSS, bandeau, cartes KPI, points d'attention).

Aucun calcul ici : uniquement de la présentation. Utilisé par app.py et weather_view.py.
Les couleurs de surface sont des transparences (rgba) : le rendu suit le thème clair ou sombre de Streamlit.
"""
from __future__ import annotations

import html

import streamlit as st

ACCENT = "#0E6B6F"
ACCENT_2 = "#14A3A8"
WARN = "#B04A08"
BAD = "#C0392B"
GOOD = "#2B7A4B"
MUTED = "#7A8A90"

TONES = {"neutral", "good", "warn", "bad", "info"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500&display=swap');

:root {
  --accent: #0E6B6F; --accent-2: #14A3A8;
  --warn: #B04A08; --bad: #C0392B; --good: #2B7A4B; --info: #3B6FB6;
  --line: rgba(128,128,128,.24); --soft: rgba(128,128,128,.07); --soft-2: rgba(128,128,128,.12);
  --radius: 18px;
}
html, body, .stApp, [data-testid="stSidebar"], button, input, textarea, select {
  font-family: 'Geist', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.block-container { padding-top: 1.8rem; max-width: 1280px; }
header[data-testid="stHeader"] { background: transparent; }

/* ---------- Bandeau ---------- */
.hero {
  position: relative; overflow: hidden; border: 1px solid var(--line); border-radius: 26px;
  padding: 1.6rem 1.8rem 1.4rem; margin: 0 0 1.3rem;
  background:
    radial-gradient(900px 260px at 0% -20%, rgba(20,163,168,.30), transparent 65%),
    radial-gradient(600px 220px at 100% 120%, rgba(14,107,111,.18), transparent 60%),
    var(--soft);
}
.hero .eyebrow { font-family: 'Geist Mono', ui-monospace, Menlo, Consolas, monospace; font-size: .7rem;
  letter-spacing: .18em; text-transform: uppercase; font-weight: 500; color: var(--accent-2); }
.hero .title, .title { font-family: 'Instrument Serif', Georgia, serif; font-size: 3.1rem; line-height: 1.04;
  letter-spacing: -.015em; margin: .35rem 0 .8rem; padding: 0; overflow-wrap: anywhere; font-weight: 400; }
.chips { display: flex; flex-wrap: wrap; gap: .45rem; }
.chip { display: inline-flex; align-items: center; gap: .4rem; font-size: .78rem; padding: .28rem .7rem;
  border-radius: 999px; background: var(--soft-2); border: 1px solid var(--line); }
.chip b { font-weight: 600; }
.eyebrow { font-family: 'Geist Mono', ui-monospace, monospace; font-size: .7rem; letter-spacing: .16em;
  text-transform: uppercase; font-weight: 500; color: var(--accent-2); }
.meta { font-size: .85rem; opacity: .7; margin-bottom: 1.2rem; }

/* ---------- Cartes KPI ---------- */
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: .8rem; margin: .2rem 0 1.2rem; }
.kpi { position: relative; border: 1px solid var(--line); border-radius: var(--radius); padding: .95rem 1.05rem .9rem;
  background: var(--soft); overflow: hidden; }
.kpi::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--accent-2); opacity: .85; }
.kpi.tone-good::before { background: var(--good); }
.kpi.tone-warn::before { background: var(--warn); }
.kpi.tone-bad::before { background: var(--bad); }
.kpi.tone-info::before { background: var(--info); }
.kpi-l { font-family: 'Geist Mono', ui-monospace, monospace; font-size: .68rem; letter-spacing: .08em;
  text-transform: uppercase; opacity: .65; }
.kpi-v { font-family: 'Instrument Serif', Georgia, serif; font-size: 2.35rem; line-height: 1.1; margin: .15rem 0 .1rem;
  letter-spacing: -.01em; overflow-wrap: anywhere; }
.kpi-h { font-size: .78rem; opacity: .65; }

/* ---------- Points d'attention ---------- */
.attn-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: .7rem; margin-bottom: 1.1rem; }
.attn { display: flex; gap: .75rem; align-items: flex-start; border: 1px solid var(--line); border-radius: 14px;
  padding: .75rem .9rem; background: var(--soft); }
.attn-dot { flex: 0 0 auto; width: .6rem; height: .6rem; border-radius: 50%; margin-top: .42rem; background: var(--info); }
.attn.tone-bad .attn-dot { background: var(--bad); box-shadow: 0 0 0 4px rgba(192,57,43,.18); }
.attn.tone-warn .attn-dot { background: var(--warn); box-shadow: 0 0 0 4px rgba(176,74,8,.16); }
.attn.tone-good .attn-dot { background: var(--good); box-shadow: 0 0 0 4px rgba(43,122,75,.16); }
.attn-t { font-weight: 600; font-size: .9rem; }
.attn-d { font-size: .8rem; opacity: .75; margin-top: .1rem; overflow-wrap: anywhere; }

/* ---------- Titres de bloc ---------- */
.sec-h { display: flex; align-items: baseline; gap: .6rem; margin: 1.3rem 0 .5rem; }
.sec-h .t { font-weight: 600; font-size: 1.02rem; letter-spacing: -.005em; }
.sec-h .s { font-size: .78rem; opacity: .6; }

/* ---------- Onglets en pastilles ---------- */
[data-baseweb="tab-list"] { gap: .4rem; border-bottom: none !important; flex-wrap: wrap; padding-bottom: .5rem; }
button[data-baseweb="tab"] { height: auto; padding: .45rem 1rem; border-radius: 999px; background: var(--soft);
  border: 1px solid transparent; transition: background .15s ease, transform .15s ease; }
button[data-baseweb="tab"]:hover { background: var(--soft-2); }
button[data-baseweb="tab"] p { margin: 0; font-size: .88rem; font-weight: 500; }
button[data-baseweb="tab"][aria-selected="true"] { background: var(--accent); }
button[data-baseweb="tab"][aria-selected="true"] p { color: #fff; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none !important; }

/* ---------- Métriques, conteneurs, boutons ---------- */
[data-testid="stMetric"] { background: var(--soft); border: 1px solid var(--line); border-radius: 16px; padding: .85rem 1rem; }
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMetric"] { background: transparent; border: none; padding: 0; }
[data-testid="stMetricValue"] { font-family: 'Instrument Serif', Georgia, serif; font-size: 2.3rem; font-weight: 400; letter-spacing: -.01em; }
[data-testid="stMetricLabel"] p { font-family: 'Geist Mono', ui-monospace, monospace; font-size: .72rem; opacity: .7; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: var(--radius); }
.stButton > button, .stDownloadButton > button { border-radius: 12px; font-weight: 500; border: 1px solid var(--line); }
.stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--accent-2); color: var(--accent-2); }
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }
[data-testid="stImage"] img { border-radius: 14px; }
[data-testid="stExpander"] { border-radius: 14px; }

/* ---------- Barre latérale ---------- */
[data-testid="stSidebar"] { border-right: 1px solid var(--line); }
.brand { display: flex; align-items: center; gap: .6rem; font-family: 'Instrument Serif', Georgia, serif;
  font-size: 1.9rem; line-height: 1; letter-spacing: -.01em; margin: 0 0 .25rem; }
.brand .logo { display: inline-grid; place-items: center; width: 2rem; height: 2rem; border-radius: 10px; color: #fff;
  font-family: 'Geist', sans-serif; font-size: .95rem; font-weight: 700;
  background: linear-gradient(135deg, var(--accent-2), var(--accent)); }
.brand-sub { font-size: .8rem; opacity: .65; margin-bottom: .9rem; }
.ver { display: inline-block; font-family: 'Geist Mono', monospace; font-size: .62rem; letter-spacing: .08em;
  padding: .1rem .45rem; border-radius: 999px; border: 1px solid var(--line); opacity: .75; vertical-align: middle; }

@keyframes rise { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
.hero, .kpi, .attn { animation: rise .45s ease both; }
@media (prefers-reduced-motion: reduce) { .hero, .kpi, .attn { animation: none; } }
@media (max-width: 640px) { .hero .title, .title { font-size: 2.2rem; } .hero { padding: 1.2rem; } }
</style>
"""


def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(s) -> str:
    return html.escape(str(s))


def _tone(t: str) -> str:
    return t if t in TONES else "neutral"


def hero(eyebrow: str, title: str, chips: list[tuple[str, str]] | None = None) -> None:
    """Bandeau d'en-tête. chips : liste de (libellé, valeur)."""
    chips_html = "".join(f'<span class="chip">{esc(k)} <b>{esc(v)}</b></span>' for k, v in (chips or []))
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{esc(eyebrow)}</div>'
        f'<h1 class="title">{esc(title)}</h1><div class="chips">{chips_html}</div></div>',
        unsafe_allow_html=True,
    )


def kpi_row(items: list[tuple]) -> None:
    """Rangée de cartes. items : (libellé, valeur, indication=None, ton='neutral')."""
    cards = []
    for it in items:
        label, value = it[0], it[1]
        hint = it[2] if len(it) > 2 else None
        tone = _tone(it[3]) if len(it) > 3 else "neutral"
        h = f'<div class="kpi-h">{esc(hint)}</div>' if hint else ""
        cards.append(f'<div class="kpi tone-{tone}"><div class="kpi-l">{esc(label)}</div>'
                     f'<div class="kpi-v">{esc(value)}</div>{h}</div>')
    st.markdown(f'<div class="kpis">{"".join(cards)}</div>', unsafe_allow_html=True)


def attention_list(items: list[tuple[str, str, str]]) -> None:
    """Points d'attention. items : (ton, titre, détail)."""
    cards = [f'<div class="attn tone-{_tone(t)}"><span class="attn-dot"></span><div>'
             f'<div class="attn-t">{esc(title)}</div><div class="attn-d">{esc(detail)}</div></div></div>'
             for t, title, detail in items]
    st.markdown(f'<div class="attn-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def section_title(title: str, sub: str | None = None) -> None:
    s = f'<span class="s">{esc(sub)}</span>' if sub else ""
    st.markdown(f'<div class="sec-h"><span class="t">{esc(title)}</span>{s}</div>', unsafe_allow_html=True)


def brand(subtitle: str, sidebar=None) -> None:
    target = sidebar or st.sidebar
    target.markdown(
        f'<div class="brand"><span class="logo">R</span>Rapports <span class="ver">v2</span></div>'
        f'<div class="brand-sub">{esc(subtitle)}</div>',
        unsafe_allow_html=True,
    )
