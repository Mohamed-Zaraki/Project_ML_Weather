"""electrical_load_pipeline.py : intégration d'ElectricalLoad dans run_all_models.py.

Les 4 autres datasets utilisent ``multi_eda_model.BaseDatasetEDAModel`` et produisent
une section ``model_phase`` standard. La charge électrique a son propre rapport
(``eda_output_electricalload/report.json`` : quality / trends / outliers / noise /
correlation / distribution / preprocessing) et sa propre évaluation
(``eval_electricalload.py`` : R² et RMSE sur 3 régresseurs).

Cette classe fournit la même interface que les modèles de ``run_all_models.py``
(``run()``, ``save()``, ``plots``, ``report``, ``db_report_id``) sans réécrire le
rapport complet quand seuls les aperçus de 50 lignes sont présents.

Usage : ``python run_all_models.py electricalload``
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "eda_output_electricalload" / "report.json"
METRICS = ROOT / "evaluation_electricalload" / "metrics_summary.csv"
MARKER = ROOT / "datasets_extracted" / "_APERCU_ONLY.txt"


class ElectricalLoadEDAModel:
    """Adaptateur lecture seule / régénération sûre pour le rapport ElectricalLoad."""

    is_regression = True

    def __init__(self):
        self.report: dict = {}
        self.plots: list[dict] = []
        self.metrics: pd.DataFrame = pd.DataFrame()
        self.db_report_id: str | None = None
        self.skipped_reason: str | None = None

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def read_metrics() -> pd.DataFrame:
        if not METRICS.exists():
            return pd.DataFrame()
        df = pd.read_csv(METRICS)
        df = df.rename(columns={"Algorithm": "Algorithme", "R² Test": "R2 Test",
                                "CV R²": "R2 CV", "RMSE Test": "RMSE Test",
                                "CV RMSE": "RMSE CV"})
        for col in ("R2 Test", "R2 CV", "RMSE Test", "RMSE CV"):
            if col in df:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _augment(self) -> None:
        """Ajoute les clés attendues par les lecteurs génériques (sans toucher au JSON)."""
        rep = self.report
        eq = self._quality_summary()
        rep.setdefault("data_quality", {
            "quality_score": eq["score"], "quality_label": eq["verdict"],
            "indicators": eq["indicators"],
            "dashboard": {"noise_percent": eq["noise_pct"], "outliers_percent": eq["outlier_pct"],
                          "contradictions_percent": 0.0},
        })
        rep.setdefault("contradictions", {"summary": {
            "rows_flagged": 0, "rows_flagged_percent": 0.0, "triggered_rules": 0, "total_flags": 0}})
        noise = rep.setdefault("noise", {})
        features = noise.get("features") or []
        noise.setdefault("summary", {
            "features_checked": len(features), "noisy_features": [],
            "error_features": [], "noisy_feature_percent": eq["noise_pct"]})
        out = rep.setdefault("outliers", {})
        out.setdefault("summary", {
            "consensus_count": (out.get("consensus_outliers") or {}).get("count", 0),
            "consensus_percent": (out.get("consensus_outliers") or {}).get("percent", 0.0),
            "unanimous_count": (out.get("unanimous_outliers") or {}).get("count", 0)})
        out.setdefault("machine_learning_outliers", {
            "methods_used": list((out.get("methods") or {}).keys()),
            "consensus_outliers": out.get("consensus_outliers") or {}})
        corr = rep.setdefault("correlation", {})
        corr.setdefault("top_pairs", [{
            "a": "MA_X-4", "b": "DEMAND", "pearson": corr.get("pearson_ma4_demand"),
            "spearman": corr.get("pearson_ma4_demand"), "strength": "très forte"}])
        if not self.metrics.empty:
            best = self.metrics.loc[self.metrics["R2 Test"].idxmax()]
            rep.setdefault("model_phase", {"evaluation": {
                "r2_test": float(best["R2 Test"]), "rmse_test": float(best["RMSE Test"]),
                "r2_cv": float(best["R2 CV"]), "rmse_cv": float(best["RMSE CV"]),
                "best_model": str(best["Algorithme"]),
            }})

    def _quality_summary(self) -> dict:
        try:
            import dashboard_data as dd
            return dd.electrical_quality(self.report)
        except Exception:
            return {"score": None, "verdict": "n/a", "noise_pct": 0.0, "outlier_pct": 0.0, "indicators": []}

    # ------------------------------------------------------------------ pipeline
    def run(self) -> dict:
        if MARKER.exists():
            self.skipped_reason = "aperçus 50 lignes (_APERCU_ONLY.txt) : rapport complet conservé"
        elif not REPORT.exists():
            self.skipped_reason = "rapport EDA introuvable : génération nécessaire"

        if self.skipped_reason is None:
            for script in ("electrical_load_eda.py", "eval_electricalload.py"):
                print(f"     -> python {script}")
                subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)

        if not REPORT.exists():
            raise FileNotFoundError(f"Rapport ElectricalLoad introuvable : {REPORT}")

        self.report = json.loads(REPORT.read_text(encoding="utf-8"))
        self.metrics = self.read_metrics()
        self.plots = list(self.report.get("plots") or [])
        self.db_report_id = (self.report.get("meta") or {}).get("report_id")
        self._augment()
        return self.report

    def save(self) -> str:
        """Aucune écriture quand on relit le rapport : on renvoie seulement son chemin."""
        return str(REPORT.relative_to(ROOT))

    # ------------------------------------------------------------------ reporting
    def summary_lines(self) -> list[str]:
        """Lignes affichées par run_all_models.py pour ce dataset de régression."""
        meta = self.report.get("meta") or {}
        eq = self._quality_summary()
        o = self.report.get("outliers") or {}
        cons = o.get("consensus_outliers") or {}
        corr = self.report.get("correlation") or {}
        lines = [
            f"     Description : {meta.get('n_rows', 'n/a')} rows x {meta.get('n_cols', 'n/a')} cols "
            f"(train {meta.get('train_rows', '?')} / test {meta.get('test_rows', '?')})",
            f"     Quality    : {eq['score']}/100 ({eq['verdict']})",
            f"     Outliers   : {len(o.get('methods') or {})} detectors, "
            f"consensus={cons.get('count', 'n/a')} ({cons.get('percent', 'n/a')}%)",
            f"     Noise      : {eq['noise_pct']:.0f}% of tested features (moderate/high NSR)",
            f"     Correlation: MA_X-4 <-> DEMAND r={corr.get('pearson_ma4_demand', 'n/a')}",
        ]
        if self.metrics.empty:
            lines.append("     Model      : métriques R²/RMSE absentes (relancer eval_electricalload.py)")
        else:
            best = self.metrics.loc[self.metrics["R2 Test"].idxmax()]
            lines.append(
                f"     Model      : {best['Algorithme']} | R2 test={best['R2 Test']:.4f} | "
                f"RMSE test={best['RMSE Test']:.1f} MW | R2 CV={best['R2 CV']:.4f}")
        if self.skipped_reason:
            lines.append(f"     Read-only  : {self.skipped_reason}")
        return lines
