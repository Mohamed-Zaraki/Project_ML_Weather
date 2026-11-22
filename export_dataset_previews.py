"""export_dataset_previews.py : rend les datasets consultables même sans les fichiers bruts.

Le dossier `datasets_extracted/` (fichiers d'origine) est ignoré par Git et absent ici.
Cette commande utilise les aperçus de 50 lignes déjà stockés dans la table `datasets` de
`data.db` pour :

1. écrire `data/previews/<id>__<fichier>.csv` + `manifest.csv` (preuve de catalogage) ;
2. recréer, **uniquement s'ils manquent**, les chemins attendus dans `datasets_extracted/`
   afin que les scripts (populate_datasets.py, EDA, eval) trouvent leurs fichiers.

Les fichiers recréés sont des APERÇUS (50 lignes), pas les jeux complets : un marqueur
`datasets_extracted/_APERCU_ONLY.txt` le rappelle. Remplacez-les par les vrais fichiers
quand vous les récupérez, puis supprimez le marqueur.

Lancer :  python export_dataset_previews.py
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data.db"
OUT_DIR = ROOT / "data" / "previews"
SOURCE_ROOT = ROOT / "datasets_extracted"
MARKER = SOURCE_ROOT / "_APERCU_ONLY.txt"
INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_filename(name: str) -> str:
    return INVALID.sub("_", name).strip().rstrip(".") or "dataset"


def main() -> None:
    if not DB_PATH.exists():
        raise SystemExit(f"Base introuvable : {DB_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "select id, category, name, filename, file_path, n_rows, n_cols, sample_data_json, uploaded_at "
        "from datasets order by id"
    ).fetchall()
    con.close()

    manifest = []
    source_written = 0
    for row in rows:
        try:
            records = json.loads(row["sample_data_json"] or "[]")
        except (TypeError, json.JSONDecodeError):
            records = []
        df = pd.DataFrame(records)

        filename = safe_filename(row["filename"] or row["name"] or "dataset")
        if not filename.lower().endswith(".csv"):
            filename += ".csv"
        out_name = f"{row['id']:03d}__{filename}"
        path = OUT_DIR / out_name
        df.to_csv(path, index=False, encoding="utf-8-sig")

        # Recrée le chemin attendu par les scripts, sans écraser un vrai fichier existant.
        rel = str(row["file_path"] or "").replace("\\", "/")
        if rel:
            target = SOURCE_ROOT / rel
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.suffix.lower() in (".xlsx", ".xls"):
                    try:
                        df.to_excel(target, index=False)
                    except Exception:
                        target = target.with_suffix(".csv")
                        df.to_csv(target, index=False, encoding="utf-8-sig")
                else:
                    df.to_csv(target, index=False, encoding="utf-8-sig")
                source_written += 1

        manifest.append({
            "id": row["id"],
            "categorie": row["category"],
            "nom": row["name"],
            "fichier_source": row["filename"],
            "lignes_reelles": row["n_rows"],
            "colonnes": row["n_cols"],
            "lignes_exportees": len(df),
            "fichier_apercu": out_name,
            "importe_le": row["uploaded_at"],
        })

    pd.DataFrame(manifest).to_csv(OUT_DIR / "manifest.csv", index=False, encoding="utf-8-sig")
    MARKER.write_text(
        "APERÇUS UNIQUEMENT — 50 lignes par fichier, reconstruits depuis data.db.\n"
        "Ce ne sont PAS les datasets complets. Remplacez ces fichiers par les vrais fichiers "
        "bruts (CSV/XLSX d'origine), puis supprimez ce marqueur avant de relancer populate_datasets.py.\n",
        encoding="utf-8",
    )
    total = sum(m["lignes_reelles"] for m in manifest)
    print(f"{len(manifest)} aperçus exportés dans {OUT_DIR.relative_to(ROOT)} "
          f"({total:,} lignes réelles cataloguées, 50 lignes par aperçu)".replace(",", " "))
    print(f"{source_written} chemins source recréés dans {SOURCE_ROOT.relative_to(ROOT)} "
          "(fichiers existants laissés intacts).")


if __name__ == "__main__":
    main()
