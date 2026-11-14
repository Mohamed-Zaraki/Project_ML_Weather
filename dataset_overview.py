"""dataset_overview.py  –  Overview phase (Phase 0) for the Weather ML project.

Responsibilities
----------------
* Store and read datasets from the SQLite database (data.db via data.py).
* Allows browsing all decompressed & stored datasets (Covid19, Parkinson, Heart Disease,
  Weather, Solar, Electrical Load, etc.).
* Category filter + search filter.
* Shows an interactive overview table of 50 lines (or fewer lines chosen by user).
* Shows column metadata, dtypes, missing values, and descriptive statistics.
* Lets user upload any new CSV / Excel dataset to be saved directly into the database.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from data import DataDB
from theme import ACCENT, hero, kpi_row, section_title

PREVIEW_MAX_ROWS = 50


# ── Database helpers (all SQL lives in data.py) ───────────────────────────────

def _ensure_schema(db_path: str) -> None:
    DataDB(db_path)


def _list_categories(db_path: str) -> list[str]:
    try:
        return DataDB(db_path, read_only=True).list_categories()
    except Exception:
        return []


def _list_datasets(db_path: str, category: str | None = None, search: str | None = None) -> list[dict]:
    try:
        return DataDB(db_path, read_only=True).list_datasets(category=category, search=search)
    except Exception:
        return []


def _get_dataset(db_path: str, row_id: int) -> tuple[pd.DataFrame, dict]:
    return DataDB(db_path, read_only=True).get_dataset(row_id)


def _save_uploaded_dataset(db_path: str, name: str, df: pd.DataFrame, category: str = "Import utilisateur") -> int:
    return DataDB(db_path).save_dataset(name, df, category=category)


def _delete_dataset(db_path: str, row_id: int) -> None:
    DataDB(db_path).delete_dataset(row_id)


# ── UI Helpers ────────────────────────────────────────────────────────────────

def _stretch(fn, *args, **kwargs):
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


def _read_upload_file(file) -> pd.DataFrame | None:
    name = file.name.lower()
    try:
        if name.endswith(".csv"):
            raw = file.read()
            for sep in (",", ";", "\t", "|"):
                try:
                    df = pd.read_csv(io.BytesIO(raw), sep=sep, encoding="utf-8-sig")
                    if len(df.columns) > 1:
                        return df
                except Exception:
                    continue
            return pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
        elif name.endswith((".xls", ".xlsx", ".xlsm")):
            return pd.read_excel(file)
        else:
            st.warning(f"Format non supporté pour « {file.name} » (formats supportés : CSV, Excel).")
            return None
    except Exception as exc:
        st.error(f"Erreur lors de la lecture de « {file.name} » : {exc}")
        return None


def render_dataset_detail(db_path: str | Path, selected_id: int, key_prefix: str = "ov") -> None:
    """Détail d'un dataset (KPI, aperçu, colonnes, statistiques, export) : partagé par la vue d'ensemble et l'accueil."""
    db_path = str(db_path)
    # ── Affichage des métadonnées du dataset choisi ──────────────────────
    try:
        df_preview, meta = _get_dataset(db_path, selected_id)
    except KeyError:
        st.error("Dataset introuvable.")
        return

    total_missing = sum(meta["missing"].values())
    cols_with_missing = sum(1 for v in meta["missing"].values() if v > 0)
    preview_count = min(PREVIEW_MAX_ROWS, len(df_preview))

    # KPI row
    kpi_row([
        ("Catégorie", meta["category"], meta["filename"]),
        ("Lignes totales", f"{meta['n_rows']:,}".replace(",", " "), f"Enregistré dans la base"),
        ("Colonnes", str(meta["n_cols"]), f"{len(meta['columns'])} variables"),
        ("Aperçu en base", f"{preview_count} lignes", f"≤ 50 instances enregistrées", "info"),
        ("Valeurs manquantes", f"{total_missing:,}".replace(",", " "),
         f"{cols_with_missing} col. concernée(s)" if total_missing > 0 else "Aucune valeur nulle",
         "warn" if total_missing > 0 else "good"),
    ])

    # ── Tableau d'aperçu (50 lignes ou moins) ────────────────────────────
    section_title("Tableau d'aperçu", f"{preview_count} lignes d'instances chargées depuis la base")

    col_ctrl1, col_ctrl2 = st.columns([2, 3])
    with col_ctrl1:
        n_rows_slider = st.slider(
            "Nombre de lignes d'instances à afficher",
            min_value=min(5, preview_count),
            max_value=max(5, preview_count),
            value=min(preview_count, 50),
            step=5,
            key=f"{key_prefix}_rows_slider",
        )
    with col_ctrl2:
        st.caption(
            f"💡 Ce tableau affiche jusqu'à **{n_rows_slider} lignes** de données réelles stockées dans SQLite. "
            f"Taille totale du dataset : **{meta['n_rows']:,} lignes**.".replace(",", " ")
        )

    # Affichage du tableau de données (50 lignes ou moins)
    df_display = df_preview.head(n_rows_slider)
    _stretch(st.dataframe, df_display, hide_index=False)

    # ── Détail des colonnes & types ──────────────────────────────────────
    with st.expander("📋 Informations sur les colonnes et types", expanded=False):
        col_info = pd.DataFrame([
            {
                "Colonne": c,
                "Type": meta["dtypes"].get(c, "—"),
                "Valeurs manquantes (échantillon)": meta["missing"].get(c, 0),
                "% manquant": round(100 * meta["missing"].get(c, 0) / max(preview_count, 1), 2),
            }
            for c in meta["columns"]
        ])
        _stretch(st.dataframe, col_info, hide_index=True)

    # ── Statistiques descriptives ─────────────────────────────────────────
    if meta.get("describe"):
        with st.expander("📊 Statistiques numériques (describe)", expanded=False):
            df_desc = pd.DataFrame(meta["describe"]).T.reset_index().rename(columns={"index": "Colonne"})
            _stretch(st.dataframe, df_desc, hide_index=True)

    # ── Bouton de téléchargement ──────────────────────────────────────────
    csv_bytes = df_preview.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label=f"⬇️ Télécharger l'extrait ({len(df_preview)} lignes en CSV)",
        data=csv_bytes,
        file_name=f"{meta['filename']}_preview.csv",
        mime="text/csv",
        key=f"{key_prefix}_dl_btn",
    )


# ── Main Render Function ──────────────────────────────────────────────────────

def render_dataset_overview(db_path: str | Path) -> None:
    """Renders the complete Overview phase."""
    db_path = str(db_path)
    _ensure_schema(db_path)

    all_cats = _list_categories(db_path)
    all_datasets_count = len(_list_datasets(db_path))

    hero(
        "Phase 0 · Vue d'ensemble",
        "Exploration & Stockage des Datasets",
        [
            ("Base de données", Path(db_path).name),
            ("Datasets stockés", f"{all_datasets_count}"),
            ("Catégories", f"{len(all_cats)}"),
        ],
    )

    # ── Section 1 : Sélection & Filtrage des Datasets ─────────────────────────
    st.subheader("🔍 Sélectionner un dataset")

    col_filter1, col_filter2 = st.columns([1, 2])
    with col_filter1:
        cat_options = ["Toutes les catégories"] + all_cats
        selected_cat = st.selectbox("Catégorie / Domaine", cat_options, key="ov_cat_filter")
    with col_filter2:
        search_kw = st.text_input("Rechercher par nom de dataset ou fichier", "", key="ov_search_kw", placeholder="Ex: covid, heart, weather, parkinson...")

    datasets = _list_datasets(db_path, category=selected_cat, search=search_kw)

    if not datasets:
        st.warning("Aucun dataset ne correspond aux filtres sélectionnés.")
    else:
        # Build selection dict
        labels = {
            d["id"]: f"[{d['category']}]  {d['name']} ({d['filename']})  —  {d['n_rows']:,} lignes × {d['n_cols']} colonnes".replace(",", " ")
            for d in datasets
        }
        ids = [d["id"] for d in datasets]

        col_sel, col_del = st.columns([5, 1])
        with col_sel:
            selected_id = st.selectbox(
                f"Dataset ({len(datasets)} disponible{'s' if len(datasets) > 1 else ''})",
                ids,
                format_func=labels.get,
                key="ov_selected_id",
            )
        with col_del:
            st.write("")
            st.write("")
            if st.button("🗑️ Supprimer", key="ov_del_btn", help="Supprimer ce dataset de la base"):
                _delete_dataset(db_path, selected_id)
                st.cache_data.clear()
                st.rerun()

        render_dataset_detail(db_path, selected_id)

    # ── Section 2 : Importation de nouveaux datasets ──────────────────────────
    st.divider()
    st.subheader("➕ Ajouter un nouveau dataset à la base")
    st.caption("Vous pouvez importer d'autres fichiers CSV ou Excel. Ils seront enregistrés immédiatement dans la base SQLite.")

    with st.expander("Importer un fichier CSV ou Excel", expanded=False):
        up_cat = st.selectbox(
            "Catégorie pour le nouveau dataset",
            all_cats + ["Autre"],
            key="ov_new_cat",
        )
        uploaded_files = st.file_uploader(
            "Sélectionner des fichiers",
            type=["csv", "xls", "xlsx", "xlsm"],
            accept_multiple_files=True,
            key="ov_file_uploader",
        )
        if uploaded_files:
            for uf in uploaded_files:
                df_up = _read_upload_file(uf)
                if df_up is not None:
                    new_id = _save_uploaded_dataset(db_path, uf.name, df_up, category=up_cat)
                    st.success(f"✅ « {uf.name} » ajouté avec succès à la base (ID : {new_id}, {len(df_up):,} lignes).".replace(",", " "))
            st.cache_data.clear()
            st.rerun()
