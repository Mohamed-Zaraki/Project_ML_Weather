"""data.py - SQLite storage layer for the whole project.

The database is ``data.db`` (formerly ``weather.db``) and holds two families of
objects:

* ``datasets``  - the catalogue used by the Phase 0 overview (metadata + preview)
* EDA reports   - one row per analysis run, with every JSON section, every
  insight sentence and every PNG plot (BLOB), plus small relational copies of
  the missing values, outliers, noise metrics and correlations.

Only the Python standard library is used (``sqlite3``), so the module can be
imported by the model, the CLI and the Streamlit views alike.

Typical use
-----------
    from data import DataDB

    db = DataDB()                       # data.db next to this file
    rid = db.save_report(model.run())   # store a whole report
    db.list_reports()                   # newest first
    db.get_report(rid)                  # same dict as report.json (+ images)
    db.get_plot_png(rid, "outliers_methods")
    db.query("SELECT dataset, COUNT(*) FROM reports GROUP BY dataset")

CLI
---
    py -3 data.py init
    py -3 data.py ingest                         # all eda_output_*/report.json
    py -3 data.py ingest eda_output_heart
    py -3 data.py list
    py -3 data.py stats
    py -3 data.py export <report_id> report.json [--images]
    py -3 data.py sql "SELECT dataset, COUNT(*) FROM reports GROUP BY dataset"
"""
from __future__ import annotations

import argparse
import base64
import json
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DB = PROJECT_ROOT / "data.db"

SCHEMA = """
PRAGMA foreign_keys = ON;

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

CREATE TABLE IF NOT EXISTS reports (
    report_id    TEXT PRIMARY KEY,
    dataset      TEXT NOT NULL,
    label        TEXT,
    generated_at TEXT NOT NULL,
    n_rows       INTEGER,
    n_cols       INTEGER,
    date_column  TEXT,
    target_column TEXT,
    meta_json    TEXT NOT NULL,
    report_json  TEXT
);

CREATE TABLE IF NOT EXISTS sections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id   TEXT NOT NULL REFERENCES reports(report_id) ON DELETE CASCADE,
    position    INTEGER NOT NULL,
    name        TEXT NOT NULL,
    description TEXT,
    data_json   TEXT NOT NULL,
    UNIQUE (report_id, name)
);

CREATE TABLE IF NOT EXISTS insights (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id TEXT NOT NULL REFERENCES reports(report_id) ON DELETE CASCADE,
    section   TEXT NOT NULL,
    position  INTEGER NOT NULL,
    text      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS plots (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id TEXT NOT NULL REFERENCES reports(report_id) ON DELETE CASCADE,
    position  INTEGER NOT NULL,
    name      TEXT NOT NULL,
    title     TEXT,
    section   TEXT,
    mime      TEXT DEFAULT 'image/png',
    file_path TEXT,
    image     BLOB,
    UNIQUE (report_id, name)
);

CREATE TABLE IF NOT EXISTS missing_columns (
    report_id TEXT NOT NULL REFERENCES reports(report_id) ON DELETE CASCADE,
    feature   TEXT NOT NULL,
    count     INTEGER,
    percent   REAL,
    PRIMARY KEY (report_id, feature)
);

CREATE TABLE IF NOT EXISTS outlier_stats (
    report_id     TEXT NOT NULL REFERENCES reports(report_id) ON DELETE CASCADE,
    feature       TEXT NOT NULL,
    min_value     REAL, max_value REAL, q1 REAL, q3 REAL, iqr REAL,
    lower_fence   REAL, upper_fence REAL,
    iqr_outliers  INTEGER, iqr_percent REAL,
    z_outliers    INTEGER, z_percent REAL,
    modified_z_outliers INTEGER, modified_z_percent REAL,
    percentile_outliers INTEGER, percentile_percent REAL,
    PRIMARY KEY (report_id, feature)
);

CREATE TABLE IF NOT EXISTS noise_features (
    report_id       TEXT NOT NULL REFERENCES reports(report_id) ON DELETE CASCADE,
    feature         TEXT NOT NULL,
    noise_to_signal REAL, snr_db REAL, verdict TEXT,
    von_neumann     REAL, lag1_autocorr REAL,
    spikes          INTEGER, spike_percent REAL,
    out_of_range    INTEGER, longest_flat_run INTEGER, stuck_sensor INTEGER,
    has_noise       INTEGER, has_errors INTEGER,
    zero_fraction   REAL,
    PRIMARY KEY (report_id, feature)
);

CREATE TABLE IF NOT EXISTS correlation_pairs (
    report_id TEXT NOT NULL REFERENCES reports(report_id) ON DELETE CASCADE,
    feature_a TEXT NOT NULL,
    feature_b TEXT NOT NULL,
    pearson   REAL, spearman REAL, strength TEXT, direction TEXT,
    PRIMARY KEY (report_id, feature_a, feature_b)
);

CREATE INDEX IF NOT EXISTS idx_insights_report ON insights(report_id, section, position);
CREATE INDEX IF NOT EXISTS idx_plots_report    ON plots(report_id, section);
CREATE INDEX IF NOT EXISTS idx_reports_dataset ON reports(dataset, generated_at);
"""

# Columns added after the first schema version. ``CREATE TABLE IF NOT EXISTS``
# does not alter an existing database, so they are applied explicitly.
MIGRATIONS: dict[str, dict[str, str]] = {
    "reports": {
        "label": "TEXT",
        "target_column": "TEXT",
        "report_json": "TEXT",
    },
    "plots": {
        "file_path": "TEXT",
    },
    "noise_features": {
        "zero_fraction": "REAL",
    },
    "outlier_stats": {
        "modified_z_outliers": "INTEGER",
        "modified_z_percent": "REAL",
        "percentile_outliers": "INTEGER",
        "percentile_percent": "REAL",
    },
}

SECTION_KEYS_NOT_IN_JSON = {"description", "insights"}


class DataDB:
    """Thin, explicit SQLite gateway used by every part of the project."""

    def __init__(self, db_path: str | Path = DEFAULT_DB, read_only: bool = False):
        self.db_path = str(Path(db_path))
        self.read_only = read_only
        if read_only:
            if not Path(self.db_path).exists():
                raise FileNotFoundError(f"Database not found: {self.db_path}")
        else:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as con:
                con.executescript(SCHEMA)
                self._migrate(con)

    # ------------------------------------------------------------------ core
    @contextmanager
    def _connect(self):
        """Yield a connection, commit/rollback and always close it."""
        if self.read_only:
            con = sqlite3.connect(Path(self.db_path).resolve().as_uri() + "?mode=ro", uri=True)
        else:
            con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        try:
            with con:
                yield con
        finally:
            con.close()

    @staticmethod
    def _migrate(con: sqlite3.Connection) -> None:
        for table, columns in MIGRATIONS.items():
            existing = {r[1] for r in con.execute(f'PRAGMA table_info("{table}")')}
            for name, decl in columns.items():
                if name not in existing:
                    con.execute(f'ALTER TABLE "{table}" ADD COLUMN "{name}" {decl}')
        # Older weather rows predate the ``label`` column.
        con.execute("UPDATE reports SET label = dataset WHERE label IS NULL")

    # --------------------------------------------------------------- datasets
    def list_categories(self) -> list[str]:
        with self._connect() as con:
            rows = con.execute("SELECT DISTINCT category FROM datasets "
                               "ORDER BY category").fetchall()
        return [r["category"] for r in rows if r["category"]]

    def list_datasets(self, category: str | None = None,
                      search: str | None = None) -> list[dict]:
        sql = ("SELECT id, category, name, filename, uploaded_at, n_rows, n_cols "
               "FROM datasets WHERE 1=1")
        params: list[str] = []
        if category and category != "Toutes les catégories":
            sql += " AND category = ?"
            params.append(category)
        if search and search.strip():
            sql += " AND (name LIKE ? OR filename LIKE ?)"
            pattern = f"%{search.strip()}%"
            params.extend([pattern, pattern])
        sql += " ORDER BY category, name"
        with self._connect() as con:
            return [dict(r) for r in con.execute(sql, params)]

    def get_dataset(self, row_id: int) -> tuple:
        """Return ``(preview_dataframe, metadata)`` for one stored dataset."""
        import pandas as pd

        with self._connect() as con:
            r = con.execute(
                "SELECT id, category, name, filename, file_path, uploaded_at, n_rows, n_cols, "
                "columns_json, dtypes_json, sample_data_json, describe_json, missing_json "
                "FROM datasets WHERE id = ?", (row_id,)).fetchone()
        if r is None:
            raise KeyError(f"Dataset {row_id} not found")
        df = pd.DataFrame(json.loads(r["sample_data_json"]))
        meta = {
            "id": r["id"], "category": r["category"], "name": r["name"],
            "filename": r["filename"], "file_path": r["file_path"],
            "uploaded_at": r["uploaded_at"], "n_rows": r["n_rows"], "n_cols": r["n_cols"],
            "columns": json.loads(r["columns_json"]),
            "dtypes": json.loads(r["dtypes_json"]),
            "describe": json.loads(r["describe_json"]) if r["describe_json"] else None,
            "missing": json.loads(r["missing_json"]) if r["missing_json"] else {},
        }
        return df, meta

    def save_dataset(self, name: str, df, category: str = "Import utilisateur") -> int:
        """Store an uploaded/previewed dataset; returns its row id."""
        now = datetime.now(timezone.utc).isoformat()
        columns = [str(c) for c in df.columns]
        dtypes = {str(c): str(df[c].dtype) for c in df.columns}
        missing = {str(c): int(df[c].isna().sum()) for c in df.columns}
        num_cols = df.select_dtypes(include="number").columns
        describe = (json.dumps(df[num_cols].describe().round(4).to_dict(), ensure_ascii=False)
                    if len(num_cols) > 0 else None)
        sample = df.head(50).copy()
        for col in sample.select_dtypes(include=["datetime", "datetimetz"]).columns:
            sample[col] = sample[col].astype(str)
        with self._connect() as con:
            cur = con.execute(
                "INSERT INTO datasets (category, name, filename, file_path, uploaded_at, "
                "n_rows, n_cols, columns_json, dtypes_json, sample_data_json, describe_json, "
                "missing_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (category, Path(name).stem.replace("_", " ").title(), name, f"uploaded/{name}",
                 now, len(df), len(df.columns), json.dumps(columns, ensure_ascii=False),
                 json.dumps(dtypes, ensure_ascii=False),
                 json.dumps(sample.to_dict(orient="records"), ensure_ascii=False, default=str),
                 describe, json.dumps(missing, ensure_ascii=False)))
            return cur.lastrowid

    def delete_dataset(self, row_id: int) -> None:
        with self._connect() as con:
            con.execute("DELETE FROM datasets WHERE id = ?", (row_id,))

    # ---------------------------------------------------------------- reports
    def save_report(self, report: dict, replace: bool = True,
                    keep_last: int | None = None) -> str:
        """Store one report atomically and return its ``report_id``.

        ``keep_last`` keeps only that many reports per dataset (older runs are
        deleted); ``None`` keeps every run.
        """
        meta = report["meta"]
        rid = meta["report_id"]
        dataset = str(meta.get("dataset") or meta.get("dataset_name") or "unknown")
        label = meta.get("dataset_label") or dataset
        generated_at = meta.get("generated_at") or datetime.now(timezone.utc).isoformat()

        with self._connect() as con:
            if replace:
                con.execute("DELETE FROM reports WHERE report_id = ?", (rid,))
            con.execute(
                "INSERT INTO reports (report_id, dataset, label, generated_at, n_rows, n_cols, "
                "date_column, target_column, meta_json, report_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (rid, dataset, label, generated_at, meta.get("n_rows"), meta.get("n_cols"),
                 meta.get("date_column"), meta.get("target_column"),
                 json.dumps(meta, ensure_ascii=False), self._report_json(report)))

            sections = [k for k, v in report.items()
                        if k not in ("meta", "plots") and isinstance(v, dict)]
            for pos, name in enumerate(sections):
                sec = report[name]
                data = {k: v for k, v in sec.items() if k not in SECTION_KEYS_NOT_IN_JSON}
                con.execute(
                    "INSERT INTO sections (report_id, position, name, description, data_json) "
                    "VALUES (?,?,?,?,?)",
                    (rid, pos, name, sec.get("description"),
                     json.dumps(data, ensure_ascii=False)))
                con.executemany(
                    "INSERT INTO insights (report_id, section, position, text) VALUES (?,?,?,?)",
                    [(rid, name, i, t) for i, t in enumerate(sec.get("insights", []))])

            for pos, p in enumerate(report.get("plots", [])):
                con.execute(
                    "INSERT INTO plots (report_id, position, name, title, section, mime, "
                    "file_path, image) VALUES (?,?,?,?,?,?,?,?)",
                    (rid, pos, p["name"], p.get("title"), p.get("section"),
                     p.get("mime", "image/png"), p.get("file"), self._image_bytes(p)))

            self._save_relational(con, rid, report)
            if keep_last:
                con.execute(
                    "DELETE FROM reports WHERE dataset = ? AND report_id NOT IN "
                    "(SELECT report_id FROM reports WHERE dataset = ? "
                    " ORDER BY generated_at DESC LIMIT ?)",
                    (dataset, dataset, keep_last))
        return rid

    @staticmethod
    def _report_json(report: dict) -> str:
        """Full report JSON without embedded image bytes (paths are kept)."""
        clone = dict(report)
        clone["plots"] = [{k: v for k, v in p.items() if k != "base64"}
                          for p in report.get("plots", [])]
        return json.dumps(clone, ensure_ascii=False)

    @staticmethod
    def _resolve_path(raw: str | None) -> Path | None:
        if not raw:
            return None
        path = Path(raw)
        if path.is_absolute():
            return path
        candidate = PROJECT_ROOT / path
        return candidate if candidate.exists() else Path.cwd() / path

    def _image_bytes(self, p: dict) -> bytes | None:
        if p.get("base64"):
            return base64.b64decode(p["base64"])
        path = self._resolve_path(p.get("file"))
        return path.read_bytes() if path and path.exists() else None

    @staticmethod
    def _save_relational(con: sqlite3.Connection, rid: str, report: dict) -> None:
        missing = (report.get("missing_duplicates", {})
                         .get("missing", {})
                         .get("columns", []))
        con.executemany(
            "INSERT INTO missing_columns (report_id, feature, count, percent) VALUES (?,?,?,?)",
            [(rid, c["name"], c.get("count"), c.get("percent")) for c in missing])

        rows = (report.get("outliers", {}).get("classical_features")
                or report.get("outliers", {}).get("features", []))
        con.executemany(
            "INSERT INTO outlier_stats (report_id, feature, min_value, max_value, q1, q3, iqr, "
            "lower_fence, upper_fence, iqr_outliers, iqr_percent, z_outliers, z_percent, "
            "modified_z_outliers, modified_z_percent, percentile_outliers, percentile_percent) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(rid, f["feature"], f.get("min", f.get("min_value")), f.get("max", f.get("max_value")),
              f.get("q1"), f.get("q3"), f.get("iqr"), f.get("lower_fence"), f.get("upper_fence"),
              f.get("iqr_outliers"), f.get("iqr_percent"), f.get("z_outliers"), f.get("z_percent"),
              f.get("modified_z_outliers"), f.get("modified_z_percent"),
              f.get("percentile_outliers"), f.get("percentile_percent")) for f in rows])

        noise = report.get("noise", {}).get("features", [])
        con.executemany(
            "INSERT INTO noise_features (report_id, feature, noise_to_signal, snr_db, verdict, "
            "von_neumann, lag1_autocorr, spikes, spike_percent, out_of_range, longest_flat_run, "
            "stuck_sensor, has_noise, has_errors, zero_fraction) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(rid, f["feature"], f.get("noise_to_signal"), f.get("snr_db"), f.get("verdict"),
              f.get("von_neumann"), f.get("lag1_autocorr"),
              f.get("spikes", f.get("spikes_count")), f.get("spike_percent"),
              f.get("out_of_range", f.get("out_of_range_count")),
              f.get("longest_flat_run"), int(bool(f.get("stuck_sensor"))),
              int(bool(f.get("has_noise"))), int(bool(f.get("has_errors"))),
              f.get("zero_fraction")) for f in noise])

        pairs = report.get("correlation", {}).get("pairs", [])
        con.executemany(
            "INSERT INTO correlation_pairs (report_id, feature_a, feature_b, pearson, spearman, "
            "strength, direction) VALUES (?,?,?,?,?,?,?)",
            [(rid, p["a"], p["b"], p.get("pearson"), p.get("spearman"),
              p.get("strength"), p.get("direction")) for p in pairs])

    # ------------------------------------------------------------------- read
    def list_reports(self, dataset: str | None = None,
                     dataset_like: str | None = None) -> list[dict]:
        sql = ("SELECT report_id, dataset, label, generated_at, n_rows, n_cols, "
               "target_column FROM reports WHERE 1=1")
        params: list[str] = []
        if dataset is not None:
            sql += " AND dataset = ?"
            params.append(dataset)
        if dataset_like is not None:
            sql += " AND dataset LIKE ?"
            params.append(dataset_like)
        sql += " ORDER BY generated_at DESC"
        with self._connect() as con:
            return [dict(r) for r in con.execute(sql, params)]

    def latest_report_id(self, dataset: str | None = None) -> str | None:
        rows = self.list_reports(dataset=dataset)
        return rows[0]["report_id"] if rows else None

    def get_report(self, report_id: str, include_images: bool = True) -> dict:
        """Rebuild the full report dict (same content as report.json)."""
        with self._connect() as con:
            head = con.execute("SELECT meta_json FROM reports WHERE report_id = ?",
                               (report_id,)).fetchone()
            if head is None:
                raise KeyError(f"unknown report_id: {report_id}")
            report: dict = {"meta": json.loads(head["meta_json"])}
            ins: dict[str, list[str]] = {}
            for r in con.execute("SELECT section, text FROM insights WHERE report_id = ? "
                                 "ORDER BY section, position", (report_id,)):
                ins.setdefault(r["section"], []).append(r["text"])
            for r in con.execute("SELECT name, description, data_json FROM sections "
                                 "WHERE report_id = ? ORDER BY position", (report_id,)):
                sec = {"description": r["description"], **json.loads(r["data_json"])}
                sec["insights"] = ins.get(r["name"], [])
                report[r["name"]] = sec
            plots = []
            for r in con.execute("SELECT name, title, section, mime, file_path, image "
                                 "FROM plots WHERE report_id = ? ORDER BY position", (report_id,)):
                p = {"name": r["name"], "title": r["title"], "section": r["section"],
                     "mime": r["mime"]}
                if r["file_path"]:
                    p["file"] = r["file_path"]
                if include_images and r["image"] is not None:
                    p["base64"] = base64.b64encode(r["image"]).decode()
                plots.append(p)
            report["plots"] = plots
        return report

    def get_report_json(self, report_id: str) -> dict:
        """Return the stored full JSON document (no image bytes)."""
        with self._connect() as con:
            row = con.execute("SELECT report_json FROM reports WHERE report_id = ?",
                              (report_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown report_id: {report_id}")
        if row["report_json"]:
            return json.loads(row["report_json"])
        return self.get_report(report_id, include_images=False)

    def get_section(self, report_id: str, name: str) -> dict:
        with self._connect() as con:
            r = con.execute("SELECT description, data_json FROM sections "
                            "WHERE report_id = ? AND name = ?", (report_id, name)).fetchone()
            if r is None:
                raise KeyError(f"unknown section: {name}")
            sec = {"description": r["description"], **json.loads(r["data_json"])}
            sec["insights"] = [x["text"] for x in con.execute(
                "SELECT text FROM insights WHERE report_id = ? AND section = ? "
                "ORDER BY position", (report_id, name))]
        return sec

    def get_plot_png(self, report_id: str, name: str) -> bytes:
        with self._connect() as con:
            r = con.execute("SELECT image FROM plots WHERE report_id = ? AND name = ?",
                            (report_id, name)).fetchone()
        if r is None or r["image"] is None:
            raise KeyError(f"unknown plot: {name}")
        return r["image"]

    def get_plot_base64(self, report_id: str, name: str) -> str:
        return base64.b64encode(self.get_plot_png(report_id, name)).decode()

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        if not sql.lstrip().lower().startswith(("select", "with")):
            raise ValueError("only SELECT queries are allowed here")
        with self._connect() as con:
            con.execute("PRAGMA query_only = ON")
            return [dict(r) for r in con.execute(sql, params)]

    def stats(self) -> dict:
        with self._connect() as con:
            counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                      for t in ("datasets", "reports", "sections", "insights", "plots")}
            counts["by_dataset"] = [dict(r) for r in con.execute(
                "SELECT dataset, label, COUNT(*) AS reports FROM reports "
                "GROUP BY dataset, label ORDER BY dataset")]
        return counts

    def delete_report(self, report_id: str) -> None:
        with self._connect() as con:
            con.execute("DELETE FROM reports WHERE report_id = ?", (report_id,))


# --------------------------------------------------------------------------- #
# CLI helpers
# --------------------------------------------------------------------------- #
def _report_files(paths: list[str]) -> list[Path]:
    if not paths:
        return sorted(PROJECT_ROOT.glob("eda_output_*/report.json"))
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            p = p / "report.json"
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.exists():
            files.append(p)
        else:
            print(f"[skip] {p} not found", file=sys.stderr)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage data.db (datasets + EDA reports).")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="database path")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="create / migrate the database")
    p_ingest = sub.add_parser("ingest", help="store report.json files")
    p_ingest.add_argument("paths", nargs="*", help="report.json or eda_output_* folders")
    sub.add_parser("list", help="list stored reports")
    sub.add_parser("stats", help="show table counts")
    p_export = sub.add_parser("export", help="write one report as JSON")
    p_export.add_argument("report_id")
    p_export.add_argument("output")
    p_export.add_argument("--images", action="store_true", help="include base64 images")
    p_sql = sub.add_parser("sql", help="run a read-only SQL query")
    p_sql.add_argument("query")

    args = parser.parse_args(argv)
    db = DataDB(args.db)

    if args.cmd == "init":
        print(f"Database ready: {db.db_path}")

    elif args.cmd == "ingest":
        files = _report_files(args.paths)
        if not files:
            print("No report.json found.", file=sys.stderr)
            return 1
        for path in files:
            report = json.loads(path.read_text(encoding="utf-8"))
            rid = db.save_report(report)
            print(f"[ok] {rid:42s} -> {path}")

    elif args.cmd == "list":
        for r in db.list_reports():
            print(f"{r['generated_at'][:19]}  {r['dataset']:12s}  "
                  f"{r['n_rows']!s:>6s} x {r['n_cols']!s:<3s}  {r['report_id']}")

    elif args.cmd == "stats":
        print(json.dumps(db.stats(), indent=2, ensure_ascii=False))

    elif args.cmd == "export":
        report = db.get_report(args.report_id, include_images=args.images)
        out = Path(args.output)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Exported {args.report_id} -> {out}")

    elif args.cmd == "sql":
        print(json.dumps(db.query(args.query), indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
