"""Analyze matched comparator fold ablations without refitting the models."""

import argparse
import json
from pathlib import Path

import pandas as pd


def compare(run_dir: Path) -> pd.DataFrame:
    """Pair training-fold losses where the tested factor is the only change."""
    scores = pd.read_csv(run_dir / "search" / "comparators_folds.csv")
    contrasts = [
        ("MLP RPM", "mlp_vibration", "mlp_rpm", "rpm_degree", 0, 1),
        ("TCN RPM", "tcn_vibration", "tcn_rpm", "rpm_degree", 0, 1),
        ("Physics FIR order bank", "physics_linear", "physics_linear", "orders", False, True),
        ("NG-RC cubic library", "ngrc", "ngrc", "polynomial_degree", 2, 3),
        ("NG-RC angle and order features", "ngrc", "ngrc", ("angular", "orders"),
         (False, False), (True, True)),
    ]
    rows = []
    for label, reference, treatment, factor, off, on in contrasts:
        fields = (factor,) if isinstance(factor, str) else factor
        def side(model, level):
            return {
                (rec.fold, json.dumps({key: val for key, val in config.items()
                                       if key not in fields}, sort_keys=True)):
                (rec.trial, rec.mse)
                for rec in scores[scores.model == model].itertuples()
                for config in [json.loads(rec.config)]
                if tuple(config[key] for key in fields) == (
                    (level,) if isinstance(factor, str) else level
                )
            }
        a, b = side(reference, off), side(treatment, on)
        for fold, config in sorted(a.keys() & b.keys()):
            a_trial, a_mse = a[fold, config]
            b_trial, b_mse = b[fold, config]
            rows.append(dict(factor=label, fold=fold, reference=reference, treatment=treatment,
                             reference_trial=a_trial, treatment_trial=b_trial,
                             held_constant=config, reference_mse=a_mse, treatment_mse=b_mse,
                             delta_mse=b_mse-a_mse))
    table = pd.DataFrame(rows)
    table.to_csv(run_dir / "tables" / "comparators_matched_ablation.csv", index=False)
    summary = table.groupby("factor").delta_mse.agg(["mean", "median", "min", "max", "count"])
    summary.to_csv(run_dir / "tables" / "comparators_matched_ablation_summary.csv")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    print(compare(parser.parse_args().run_dir).to_string())
