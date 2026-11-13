"""run_all_models.py - Execute the 8-section EDA + ML pipeline on the four
requested datasets: COVID-19, Parkinson, Alcohol and Heart Attack.

Each dataset produces ``eda_output_<name>/report.json`` plus one PNG per plot.
The front-end is not touched: this script only builds the logic/report artefacts.

Usage
-----
    py -3 run_all_models.py
    py -3 run_all_models.py covid parkinson      # subset
"""
from __future__ import annotations

import sys
import time
import traceback

from multi_eda_model import (
    AlcoholEDAModel, CovidEDAModel, HeartAttackEDAModel, HeartEDAModel,
    ParkinsonEDAModel,
)

MODELS = {
    "alcohol": ("Alcohol Detection", AlcoholEDAModel),
    "heart": ("Heart Attack (comprehensive)", HeartEDAModel),
    "heartattack": ("Heart Attack (UCI)", HeartAttackEDAModel),
    "parkinson": ("Parkinson's Disease", ParkinsonEDAModel),
    "covid": ("COVID-19 Laboratory", CovidEDAModel),
}

# Default run: the four datasets requested by the assignment.
DEFAULT = ["covid", "parkinson", "alcohol", "heart"]


def _fmt(value):
    return "n/a" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)


def main(argv: list[str]):
    keys = argv or DEFAULT
    unknown = [k for k in keys if k not in MODELS]
    if unknown:
        raise SystemExit(f"Unknown dataset key(s): {unknown}. Choose from {list(MODELS)}")

    print("=" * 78)
    print("MULTI-DATASET EDA & ML PIPELINE")
    print("=" * 78)

    results = []
    for key in keys:
        label, cls = MODELS[key]
        t0 = time.time()
        print(f"\n---> {label} ...")
        try:
            model = cls()
            report = model.run()
            saved = model.save()
            duration = round(time.time() - t0, 2)

            ev = report["model_phase"].get("evaluation", {})
            cr = report["contradictions"]["summary"]
            noise = report["noise"]["summary"]
            ml = report["outliers"]["machine_learning_outliers"]
            consensus = ml.get("consensus_outliers", {})

            db_id = getattr(model, "db_report_id", None)
            print(f"     [OK] {duration}s | {len(model.plots)} plots | report: {saved}")
            print(f"     Database   : data.db | report_id={db_id or 'not stored'}")
            print(f"     Description : {report['meta']['n_rows']} rows x {report['meta']['n_cols']} cols")
            dq = report.get("data_quality", {})
            print(f"     Quality    : {dq.get('quality_score')}/100 ({dq.get('quality_label')})")
            print(f"     Contradictions: {cr['rows_flagged']} rows ({cr['rows_flagged_percent']}%), "
                  f"{cr['triggered_rules']} rules triggered")
            print(f"     Outliers   : {len(ml.get('methods_used', []))} unsupervised detectors, "
                  f"consensus={consensus.get('count', 'n/a')} ({consensus.get('percent', 'n/a')}%)")
            print(f"     Noise      : noisy={noise['noisy_features']} | errors={noise['error_features']}")
            print(f"     Correlation: top={[(p['a'], p['b']) for p in report['correlation']['top_pairs'][:3]]}")
            print(f"     Model      : balanced_acc={_fmt(ev.get('balanced_accuracy'))}, "
                  f"macro_F1={_fmt(ev.get('f1_macro'))}, "
                  f"CV={_fmt(ev.get('cv_balanced_accuracy_mean'))}")

            results.append({
                "dataset": label, "status": "SUCCESS", "duration": duration,
                "plots": len(model.plots), "report_file": saved,
                "rows": report["meta"]["n_rows"],
                "contradictory_rows": cr["rows_flagged"],
                "balanced_accuracy": ev.get("balanced_accuracy"),
                "f1_macro": ev.get("f1_macro"),
            })
        except Exception as exc:  # keep going with the other datasets
            print(f"     [FAILED] {exc}")
            traceback.print_exc()
            results.append({"dataset": label, "status": "FAILED", "error": str(exc)})

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for r in results:
        if r["status"] == "SUCCESS":
            print(f" - {r['dataset']}: OK | {r['plots']} plots | {r['rows']} rows | "
                  f"contradictions={r['contradictory_rows']} | "
                  f"bal_acc={_fmt(r['balanced_accuracy'])} | macro_F1={_fmt(r['f1_macro'])}")
        else:
            print(f" - {r['dataset']}: FAILED | {r['error']}")
    print("=" * 78)
    return results


if __name__ == "__main__":
    main(sys.argv[1:])
