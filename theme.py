"""theme.py : habillage visuel partagé (CSS, bandeau, cartes KPI, points d'attention).

Aucun calcul ici : uniquement de la présentation. Utilisé par project_dashboard.py, app.py et weather_view.py.
"""
from __future__ import annotations

import html
import streamlit as st

ACCENT = "#4F46E5"     # Indigo vibrant
ACCENT_2 = "#06B6D4"   # Cyan électrique
WARN = "#F59E0B"       # Ambre
BAD = "#EF4444"        # Rouge corail
GOOD = "#10B981"       # Émeraude
INFO = "#3B82F6"       # Bleu électrique
MUTED = "#64748B"      # Ardoise moderne

TONES = {"neutral", "good", "warn", "bad", "info"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
  --accent: #4F46E5; --accent-2: #06B6D4;
  --warn: #F59E0B; --bad: #EF4444; --good: #10B981; --info: #3B82F6;
  --bg-page: #F8FAFC;
  --bg-card: #FFFFFF;
  --line: rgba(226, 232, 240, 0.85);
  --soft: #FFFFFF;
  --soft-2: rgba(241, 245, 249, 0.85);
  --radius: 18px;
}

html, body, .stApp {
  font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
  background: linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%) !important;
  color: #0F172A !important;
}

.block-container { padding-top: 1.8rem; max-width: 1320px; }
header[data-testid="stHeader"] { background: transparent; }

/* ---------- Barre latérale professionnelle ---------- */
[data-testid="stSidebar"] {
  background: #FFFFFF !important;
  border-right: 1px solid rgba(226, 232, 240, 0.85) !important;
  box-shadow: 2px 0 16px rgba(15, 23, 42, 0.02) !important;
}
[data-testid="stSidebar"] * {
  color: #1E293B;
}

.brand { display: flex; align-items: center; gap: .7rem; margin: 0 0 .35rem; }
.brand .logo {
  display: inline-grid; place-items: center; width: 2.3rem; height: 2.3rem; border-radius: 12px; color: #fff !important;
  font-family: 'Plus Jakarta Sans', sans-serif; font-size: .95rem; font-weight: 800; flex-shrink: 0;
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35);
}
.brand-text {
  display: flex; align-items: center; gap: .45rem; font-family: 'Plus Jakarta Sans', sans-serif;
  font-size: 1.22rem; font-weight: 800; line-height: 1.15; letter-spacing: -.02em; color: #0F172A !important;
}
.brand-sub { font-size: .8rem; color: #64748B !important; margin-bottom: 1.1rem; }
.ver {
  display: inline-block; font-family: 'JetBrains Mono', monospace; font-size: .62rem; letter-spacing: .06em;
  padding: .12rem .48rem; border-radius: 999px; background: rgba(79, 70, 229, 0.08);
  border: 1px solid rgba(79, 70, 229, 0.22); color: var(--accent) !important; font-weight: 700; vertical-align: middle;
}

/* ---------- BOUTON RETOUR AUX DATASETS (DESIGN HEROIQUE) ---------- */
div[data-testid="stSidebar"] div.stButton > button {
  background: linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%) !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 16px !important;
  font-weight: 700 !important;
  font-size: 0.92rem !important;
  padding: 0.75rem 1.2rem !important;
  box-shadow: 0 6px 20px -2px rgba(79, 70, 229, 0.4) !important;
  transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1) !important;
  letter-spacing: -0.01em !important;
}
div[data-testid="stSidebar"] div.stButton > button p {
  color: #ffffff !important;
  font-weight: 700 !important;
}
div[data-testid="stSidebar"] div.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 10px 25px -2px rgba(79, 70, 229, 0.55) !important;
  filter: brightness(1.08) !important;
  color: #ffffff !important;
}
div[data-testid="stSidebar"] div.stButton > button:active {
  transform: translateY(0px) !important;
}

/* ---------- Bandeau Hero Moderne ---------- */
.hero {
  position: relative; overflow: hidden; border: 1px solid rgba(99, 102, 241, 0.22); border-radius: 24px;
  padding: 1.8rem 2.2rem 1.6rem; margin: 0 0 1.4rem;
  background:
    radial-gradient(1000px 300px at 0% -20%, rgba(79, 70, 229, 0.14), transparent 70%),
    radial-gradient(800px 260px at 100% 120%, rgba(6, 182, 212, 0.12), transparent 65%),
    #FFFFFF;
  box-shadow: 0 10px 30px -8px rgba(79, 70, 229, 0.08), 0 2px 6px rgba(0,0,0,0.02);
}
.hero .eyebrow { font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: .74rem;
  letter-spacing: .16em; text-transform: uppercase; font-weight: 700; color: var(--accent); }
.hero .title, .title { font-family: 'Plus Jakarta Sans', sans-serif; font-size: 2.6rem; line-height: 1.12;
  letter-spacing: -.03em; margin: .4rem 0 .9rem; padding: 0; overflow-wrap: anywhere; font-weight: 800;
  background: linear-gradient(135deg, #0F172A 50%, #4F46E5);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.chips { display: flex; flex-wrap: wrap; gap: .5rem; }
.chip { display: inline-flex; align-items: center; gap: .45rem; font-size: .82rem; padding: .35rem .9rem;
  border-radius: 999px; background: rgba(79, 70, 229, 0.06); border: 1px solid rgba(79, 70, 229, 0.2);
  transition: transform .15s ease, border-color .15s ease; color: #1E293B; }
.chip:hover { transform: translateY(-1px); border-color: var(--accent-2); }
.chip b { font-weight: 800; color: var(--accent); }
.eyebrow { font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: .72rem; letter-spacing: .16em;
  text-transform: uppercase; font-weight: 700; color: var(--accent); }
.meta { font-size: .88rem; opacity: .75; margin-bottom: 1.2rem; }

/* ---------- Cartes KPI ---------- */
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(195px, 1fr)); gap: .9rem; margin: .3rem 0 1.3rem; }
.kpi {
  position: relative; border: 1px solid rgba(226, 232, 240, 0.9); border-radius: var(--radius); padding: 1.1rem 1.25rem 1rem;
  background: #FFFFFF; overflow: hidden;
  box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.04);
  transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
}
.kpi:hover {
  transform: translateY(-3px); box-shadow: 0 12px 28px -6px rgba(79, 70, 229, 0.12);
  border-color: rgba(99, 102, 241, 0.35);
}
.kpi::before { content: ""; position: absolute; left: 0; top: 0; right: 0; height: 3.5px;
  background: linear-gradient(90deg, var(--accent), var(--accent-2)); }
.kpi.tone-good::before { background: linear-gradient(90deg, #10B981, #34D399); }
.kpi.tone-warn::before { background: linear-gradient(90deg, #F59E0B, #FBBF24); }
.kpi.tone-bad::before { background: linear-gradient(90deg, #EF4444, #F87171); }
.kpi.tone-info::before { background: linear-gradient(90deg, #3B82F6, #60A5FA); }
.kpi-l { font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: .7rem; letter-spacing: .08em;
  text-transform: uppercase; color: #64748B; font-weight: 700; }
.kpi-v { font-family: 'Plus Jakarta Sans', sans-serif; font-size: 2.2rem; line-height: 1.15; margin: .2rem 0 .15rem;
  letter-spacing: -.02em; font-weight: 800; overflow-wrap: anywhere; color: #0F172A; }
.kpi-h { font-size: .8rem; color: #64748B; }

/* ---------- Points d'attention ---------- */
.attn-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: .8rem; margin-bottom: 1.2rem; }
.attn {
  display: flex; gap: .85rem; align-items: flex-start; border: 1px solid rgba(226, 232, 240, 0.9); border-radius: 16px;
  padding: .95rem 1.1rem; background: #FFFFFF;
  box-shadow: 0 2px 10px rgba(15, 23, 42, 0.02);
  transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
}
.attn:hover { transform: translateY(-2px); border-color: rgba(99, 102, 241, 0.35); box-shadow: 0 8px 20px -6px rgba(0,0,0,0.06); }
.attn-dot { flex: 0 0 auto; width: .65rem; height: .65rem; border-radius: 50%; margin-top: .42rem; background: var(--info); }
.attn.tone-bad .attn-dot { background: var(--bad); box-shadow: 0 0 0 4px rgba(239,68,68,.25); }
.attn.tone-warn .attn-dot { background: var(--warn); box-shadow: 0 0 0 4px rgba(245,158,11,.22); }
.attn.tone-good .attn-dot { background: var(--good); box-shadow: 0 0 0 4px rgba(16,185,129,.22); }
.attn-t { font-weight: 700; font-size: .92rem; color: #0F172A; }
.attn-d { font-size: .82rem; color: #475569; margin-top: .15rem; overflow-wrap: anywhere; line-height: 1.4; }

/* ---------- Titres de bloc ---------- */
.sec-h { display: flex; align-items: baseline; gap: .7rem; margin: 1.5rem 0 .6rem; }
.sec-h .t { font-weight: 800; font-size: 1.15rem; letter-spacing: -.02em; color: #0F172A; }
.sec-h .s { font-size: .82rem; color: #64748B; font-weight: 500; }

/* ---------- Onglets Modernes ---------- */
[data-baseweb="tab-list"] { gap: .5rem; border-bottom: none !important; flex-wrap: wrap; padding-bottom: .6rem; }
button[data-baseweb="tab"] {
  height: auto; padding: .52rem 1.2rem; border-radius: 999px; background: #FFFFFF;
  border: 1px solid rgba(226, 232, 240, 0.95); transition: all .2s ease;
  box-shadow: 0 2px 6px rgba(0,0,0,0.02);
}
button[data-baseweb="tab"]:hover { background: #F8FAFC; border-color: rgba(99, 102, 241, 0.35); }
button[data-baseweb="tab"] p { margin: 0; font-size: .9rem; font-weight: 600; color: #475569; }
button[data-baseweb="tab"][aria-selected="true"] {
  background: linear-gradient(135deg, #4F46E5, #06B6D4) !important;
  border-color: transparent !important;
  box-shadow: 0 4px 16px rgba(79, 70, 229, 0.35);
}
button[data-baseweb="tab"][aria-selected="true"] p { color: #fff !important; font-weight: 700; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none !important; }

/* ---------- Métriques, conteneurs, boutons généraux ---------- */
[data-testid="stMetric"] {
  background: #FFFFFF; border: 1px solid rgba(226, 232, 240, 0.95); border-radius: 16px; padding: .95rem 1.1rem;
  box-shadow: 0 2px 8px rgba(0,0,0,0.02);
}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMetric"] { background: transparent; border: none; padding: 0; }
[data-testid="stMetricValue"] { font-family: 'Plus Jakarta Sans', sans-serif; font-size: 2.2rem; font-weight: 800; letter-spacing: -.02em; color: #0F172A; }
[data-testid="stMetricLabel"] p { font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: .74rem; color: #64748B; font-weight: 700; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: var(--radius); }

.stButton > button, .stDownloadButton > button {
  border-radius: 12px; font-weight: 600; border: 1px solid rgba(226, 232, 240, 0.95);
  background: #FFFFFF; color: #0F172A;
  box-shadow: 0 2px 6px rgba(0,0,0,0.02);
  transition: all .2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: var(--accent); color: var(--accent);
  box-shadow: 0 4px 14px rgba(79, 70, 229, 0.18);
  transform: translateY(-1px);
}
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid rgba(226, 232, 240, 0.95); background: #FFFFFF; }
[data-testid="stImage"] img { border-radius: 14px; box-shadow: 0 8px 24px -6px rgba(0,0,0,0.08); }
[data-testid="stExpander"] { border-radius: 14px; border: 1px solid rgba(226, 232, 240, 0.95); background: #FFFFFF; }

/* ---------- Showcase Problématique & Pipeline ---------- */
.problem-box {
  border: 1px solid rgba(99, 102, 241, 0.22); border-radius: 20px;
  padding: 1.4rem 1.6rem; margin: 0 0 1.5rem;
  background: linear-gradient(135deg, rgba(79, 70, 229, 0.05) 0%, rgba(6, 182, 212, 0.03) 100%), #FFFFFF;
  box-shadow: 0 6px 20px -4px rgba(0,0,0,0.03);
}
.problem-title {
  font-size: 1.35rem; font-weight: 800; letter-spacing: -.02em; margin-bottom: .4rem;
  display: flex; align-items: center; gap: .6rem; color: #0F172A;
}
.problem-desc {
  font-size: .88rem; color: #334155; line-height: 1.5; margin-bottom: 1rem; max-width: 950px;
}
.pipeline-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: .9rem; margin: 1rem 0 .5rem;
}
.pipeline-step {
  border: 1px solid rgba(226, 232, 240, 0.95); border-radius: 16px; padding: 1.05rem 1.15rem;
  background: #FFFFFF; position: relative;
  box-shadow: 0 2px 8px rgba(0,0,0,0.02);
  transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease;
}
.pipeline-step:hover {
  transform: translateY(-3px); border-color: var(--accent-2);
  box-shadow: 0 10px 24px -6px rgba(79, 70, 229, 0.12);
}
.step-badge {
  display: inline-flex; align-items: center; justify-content: center;
  width: 28px; height: 28px; border-radius: 8px; font-weight: 800; font-size: .78rem;
  background: linear-gradient(135deg, var(--accent), var(--accent-2)); color: #fff; margin-bottom: .6rem;
}
.step-t { font-weight: 700; font-size: .95rem; margin-bottom: .25rem; color: #0F172A; }
.step-d { font-size: .8rem; color: #64748B; line-height: 1.4; }

@keyframes rise { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
.hero, .kpi, .attn, .problem-box { animation: rise .45s ease both; }
@media (prefers-reduced-motion: reduce) { .hero, .kpi, .attn, .problem-box { animation: none; } }
@media (max-width: 640px) { .hero .title, .title { font-size: 2.1rem; } .hero { padding: 1.2rem; } }
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


def brand(subtitle: str, title: str = "Data Description", logo: str = "DD", sidebar=None) -> None:
    target = sidebar or st.sidebar
    target.markdown(
        f'<div class="brand"><span class="logo">{esc(logo)}</span>'
        f'<div class="brand-text"><span>{esc(title)}</span> <span class="ver">v2.0</span></div></div>'
        f'<div class="brand-sub">{esc(subtitle)}</div>',
        unsafe_allow_html=True,
    )
