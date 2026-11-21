"""Audit the expanded evidence without refitting or requiring raw acquisitions."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gearbox_expanded.protocol import FAMILIES, START, mse, write_json
from gearbox_expanded.study import ANCHORS, ANCESTOR


def verify(out, prior=None):
    summary = pd.read_csv(out/"comparison_metrics.csv").set_index("model")
    wanted = {*sum(FAMILIES.values(),()), *ANCHORS, ANCESTOR}
    assert wanted == set(summary.index)
    prior = prior or Path(__file__).resolve().parents[1]/"prior/gearbox_dhon_project"
    past = prior/"outputs"/"runs"/"development"/"predictions"
    anchors = {}
    with np.load(out/"matched_validation.npz",allow_pickle=False) as z:
        np.testing.assert_array_equal(z["target_indices"],np.arange(START,24_000))
        discrepancies = {slug:abs(mse(z["actual"],z[slug])-summary.loc[slug,"mse"]) for slug in wanted}
        for slug, file in [("primary_dhon","dhon.npz"),
                           ("archive_dhon_optimized_corrected","baseline.npz"),
                           (ANCESTOR,"baseline.npz")]:
            with np.load(past/file, allow_pickle=False) as saved:
                anchors[slug] = float(np.max(np.abs(z[slug]-saved[slug][START:])))
    if max(anchors.values()) > 1e-5:
        raise ValueError("Prior DHON archive replay mismatch")
    if max(discrepancies.values()) > 1e-12 or not (summary["count"]==24_000-START).all():
        raise ValueError("Metric or score-mask discrepancy")
    result = dict(status="PASS", families=len(sum(FAMILIES.values(),())), all_rows=len(summary),
                  targets=24_000-START, mse_max_discrepancy=max(discrepancies.values()),
                  prior_forecast_replay_max_abs=anchors)
    write_json(out/"numeric_audit.json",result)
    print(result)
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("outputs/expanded"))
    verify(parser.parse_args().out)
