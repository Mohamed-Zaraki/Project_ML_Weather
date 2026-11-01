"""populate_datasets.py - Ingests all extracted datasets into weather.db.
"""
from __future__ import annotations

import json
import os
import pathlib
import sqlite3
import time
from datetime import datetime, timezone

import pandas as pd

DB_PATH = pathlib.Path(r"c:\Users\Moham\Documents\GitHub\Sumestre_7_TP\ML_SUM_7\Project_ML_Weather\weather.db")
EXTRACTED_ROOT = pathlib.Path(r"c:\Users\Moham\Documents\GitHub\Sumestre_7_TP\ML_SUM_7\Project_ML_Weather\datasets_extracted")

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    category         TEXT NOT NULL,
    name             TEXT NOT NULL,
    filename         TEXT NOT NULL,
    file_path        TEXT NOT NULL,
    uploaded_at      TEXT NOT NULL,
    n_rows           INTEGER NOT NULL,
    n_cols           INTEGER NOT NULL,
    columns_json     TEXT NOT NULL,
    dtypes_json      TEXT NOT NULL,
    sample_data_json TEXT NOT NULL,
    describe_json    TEXT,
    missing_json     TEXT
);
CREATE INDEX IF NOT EXISTS idx_datasets_cat ON datasets(category, name);
"""

CATEGORY_MAP = [
    ("Covid19", ["covid"]),
    ("Alcoolémie / Santé", ["alcohol"]),
    ("Charge électrique", ["electricalload"]),
    ("Ferme solaire allemande", ["germansolarfarm", "pv_"]),
    ("Cardiologie & Crise cardiaque", ["heart"]),
    ("Maladie de Parkinson", ["parkinson"]),
    ("Production photovoltaïque (PV)", ["pvdataset"]),
    ("Irradiance solaire", ["solar irradiance", "irradiance"]),
    ("Trafic routier", ["traffic"]),
    ("Météorologie", ["weather"]),
]


def determine_category(path: pathlib.Path) -> str:
    path_str = str(path).lower()
    for cat_name, keywords in CATEGORY_MAP:
        if any(kw in path_str for kw in keywords):
            return cat_name
    return "Autres datasets"


def count_csv_rows(fpath: pathlib.Path) -> int:
    """Fast line counter for CSV files."""
    try:
        count = 0
        with open(fpath, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                count += chunk.count(b"\n")
        return max(0, count - 1)  # subtract header
    except Exception:
        return 0


def populate():
    t_start = time.time()
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)

    # Clear existing datasets to avoid duplicates on re-run
    con.execute("DELETE FROM datasets")
    con.commit()

    all_files = sorted(
        f for f in EXTRACTED_ROOT.rglob("*")
        if f.is_file() and f.suffix.lower() in [".csv", ".xlsx", ".xls"]
    )
    print(f"Found {len(all_files)} datasets to process.")

    inserted = 0
    now = datetime.now(timezone.utc).isoformat()

    for idx, fpath in enumerate(all_files, 1):
        cat = determine_category(fpath)
        name = fpath.stem.replace("_", " ").title()
        filename = fpath.name
        rel_path = str(fpath.relative_to(EXTRACTED_ROOT))

        print(f"[{idx}/{len(all_files)}] Processing ({cat}) {filename}...", end=" ", flush=True)

        try:
            if fpath.suffix.lower() == ".csv":
                total_rows = count_csv_rows(fpath)
                df_preview = pd.read_csv(fpath, nrows=50, encoding_errors="replace")
                if total_rows == 0:
                    total_rows = len(df_preview)
            else:
                # Excel file
                df_preview = pd.read_excel(fpath, nrows=50)
                total_rows = len(df_preview)

            n_cols = len(df_preview.columns)
            columns = [str(c) for c in df_preview.columns]
            dtypes = {str(c): str(df_preview[c].dtype) for c in df_preview.columns}

            # Missing values in preview
            missing = {str(c): int(df_preview[c].isna().sum()) for c in df_preview.columns}

            # Numeric describe on preview
            num_cols = df_preview.select_dtypes(include="number").columns
            if len(num_cols) > 0:
                desc = df_preview[num_cols].describe().round(4).to_dict()
                desc_json = json.dumps(desc, ensure_ascii=False)
            else:
                desc_json = None

            # Sample 50 rows records
            # Handle timestamps and NaNs safely for json serialization
            df_clean = df_preview.copy()
            for col in df_clean.select_dtypes(include=["datetime", "datetimetz"]).columns:
                df_clean[col] = df_clean[col].astype(str)

            records = df_clean.to_dict(orient="records")
            sample_json = json.dumps(records, ensure_ascii=False, default=str)

            con.execute(
                """
                INSERT INTO datasets (
                    category, name, filename, file_path, uploaded_at,
                    n_rows, n_cols, columns_json, dtypes_json,
                    sample_data_json, describe_json, missing_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cat,
                    name,
                    filename,
                    rel_path,
                    now,
                    total_rows,
                    n_cols,
                    json.dumps(columns, ensure_ascii=False),
                    json.dumps(dtypes, ensure_ascii=False),
                    sample_json,
                    desc_json,
                    json.dumps(missing, ensure_ascii=False),
                ),
            )
            inserted += 1
            print(f"OK ({total_rows:,} rows, {n_cols} cols)")
        except Exception as e:
            print(f"FAILED: {e}")

    con.commit()
    con.close()
    dt = time.time() - t_start
    print(f"\nSuccessfully stored {inserted} datasets in {DB_PATH.name} in {dt:.1f}s.")


if __name__ == "__main__":
    populate()
