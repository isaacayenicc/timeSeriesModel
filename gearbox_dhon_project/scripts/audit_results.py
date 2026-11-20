"""Recompute headline scores and primary PMI groups from saved numerical evidence."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gearbox_dhon.results import START, read_predictions, scores


def verify(run):
    source = np.load(run / "predictions/development_targets.npz", allow_pickle=False)
    y = source["target"]
    ix = source["zero_based_indices"]
    np.testing.assert_array_equal(ix, np.arange(START, 24_000))
    predictions = read_predictions(run, len(y) + START)
    table = pd.read_csv(run / "metrics/summary.csv").set_index("model")
    if set(predictions) != set(table.index):
        raise ValueError("Prediction/summary model set mismatch")
    diff = {}
    for key, values in predictions.items():
        recalculated = scores(y, values[START:])
        diff[key] = abs(recalculated["mse"] - table.loc[key, "mse"])
        if diff[key] > 1e-12 or table.loc[key, "count"] != len(ix):
            raise ValueError(f"Metric mismatch: {key} {diff[key]}")
    matched = table.loc["primary_dhon", "mse"]
    comparison = {ref: 100*(table.loc[ref, "mse"]-matched)/table.loc[ref, "mse"]
                  for ref in ("spa_lpv_arma", "lpv_arma_dense", "archive_dhon_optimized_corrected",
                              "archive_dhon_tau_context_extended_gated_corrected") if ref in table.index}
    pmi = {}
    path = run / "metrics/fault_per_record.csv"
    if path.exists():
        rows = pd.read_csv(path)
        grouped = pd.read_csv(run / "metrics/fault_group_summary.csv")
        for _, row in grouped.iterrows():
            subset = rows[(rows.model == row.model) & rows.eligible]
            a = subset.loc[subset.group == "A", row.metric].dropna().to_numpy()
            b = subset.loc[subset.group == "B", row.metric].dropna().to_numpy()
            auc = np.mean((b[:, None] > a[None, :]) + .5*(b[:, None] == a[None, :]))
            np.testing.assert_allclose([np.median(a), np.median(b), auc],
                                       [row.median_A, row.median_B, row.conditional_auc_B_high],
                                       atol=1e-12, rtol=0)
            np.testing.assert_equal([len(a), len(b)], [row.n_A, row.n_B])
            pmi[f"{row.model}/{row.metric}"] = {"n_a":len(a), "n_b":len(b), "auc":auc}
    result = {"status": "PASS", "scored_targets": len(y), "mse_abs_discrepancy": diff,
              "dhon_improvement_pct": comparison, "pmi_recomputed": pmi}
    print(json.dumps(result, indent=2))
    (run / "results_verification.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    verify(parser.parse_args().run_dir)
