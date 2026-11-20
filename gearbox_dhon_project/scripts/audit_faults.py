"""Recompute diagnostic summary invariants from saved per-record evidence."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from gearbox_dhon.faults import diagnostics


def audit(run: Path) -> dict:
    rows = pd.read_csv(run / "metrics" / "fault_per_record.csv")
    summary = pd.read_csv(run / "metrics" / "fault_group_summary.csv")
    status = json.loads((run / "metrics" / "fault_status.json").read_text())
    eligible = rows[rows.eligible]
    assert len(rows) == status["records"] * len(status["models"])
    assert not rows.duplicated(["record_id", "model"]).any()
    for _, group in rows.groupby("record_id"):
        assert group.first_sample.nunique() == group.target_count.nunique() == 1
        assert group.first_sample.iloc[0] == status["warmup_samples"]
        assert group.eligible.nunique() == 1
    assert rows.loc[rows.pmi_status == "ok", "pmi_course"].notna().all()
    courses = rows.dropna(subset=["pmi_course"])
    np.testing.assert_allclose(
        courses.pmi_course, courses.pmi_course_rho / (1 - courses.pmi_course_rho), atol=1e-13
    )
    for row in summary.itertuples():
        subset = eligible[eligible.model == row.model]
        a = subset[subset.group == "A"][row.metric].dropna().to_numpy()
        b = subset[subset.group == "B"][row.metric].dropna().to_numpy()
        assert (a.size, b.size) == (row.n_A, row.n_B)
        # Calculate AUC from ranks instead of the main routine's pairwise matrix.
        ranks = rankdata(np.r_[a, b])[a.size:]
        auc = (ranks.sum() - b.size * (b.size + 1) / 2) / (a.size * b.size)
        np.testing.assert_allclose(
            [np.median(a), np.median(b), np.median(b) - np.median(a), b.min() - a.max(), auc],
            [row.median_A, row.median_B, row.median_gap_B_minus_A,
             row.margin_min_B_minus_max_A, row.conditional_auc_B_high],
            atol=1e-13,
        )
        assert 0 <= row.conditional_auc_ci_low <= row.conditional_auc_ci_high <= 1
    rep_errors = {}
    with np.load(run / "predictions" / "fault_representative_traces.npz", allow_pickle=False) as traces:
        for row in rows.itertuples():
            key = f"{row.record_id}:{row.model}:residual"
            if key in traces:
                diagnostic, _ = diagnostics(
                    traces[key], traces[f"{row.record_id}:rpm"], traces[f"{row.record_id}:t"],
                    row.samples_per_rev,
                )
                residual_error = abs(diagnostic["pmi_course"] - row.pmi_course)
                rep_errors[key] = residual_error
                assert residual_error <= 2e-8
    stress = pd.read_csv(run / "metrics" / "rpm_stress.csv")
    pulses = pd.read_csv(run / "metrics" / "pulse_preservation.csv")
    robust = pd.read_csv(run / "metrics" / "robustness_summary.csv")
    for row in robust.itertuples():
        matching = stress[(stress.model == row.model) & (stress.scope == "predictor_only")
                          & (stress.condition != "input_noise_sd_0.05")]
        case = pulses[(pulses.model == row.model) & (pulses.pulse_kind == "periodic")]
        assert len(matching) == row.rpm_conditions
        assert len(case) == row.periodic_pmi_cases
        assert int((case.delta_pmi > 0).sum()) == row.periodic_pmi_positive
        assert row.mse_worst_condition == matching.loc[matching["mse"].idxmax(), "condition"]
        np.testing.assert_allclose(
            [matching["mse"].mean(), matching["mse"].max(), case.pulse_corr.median()],
            [row.rpm_mse_mean, row.rpm_mse_worst, row.median_periodic_pulse_corr],
            atol=1e-13,
        )
    pmi = eligible.pivot(index=["group", "record_id"], columns="model", values="pmi_course")
    comparisons = {}
    for reference in ("spa_lpv_arma", "archive_dhon_optimized_corrected"):
        if reference not in pmi or "primary_dhon" not in pmi:
            continue
        pair = pmi[[reference, "primary_dhon"]].dropna()
        a = pair.loc["A"].to_numpy()
        b = pair.loc["B"].to_numpy()
        rng = np.random.default_rng(7780)
        a_draw = rng.integers(a.shape[0], size=(1000, a.shape[0]))
        b_draw = rng.integers(b.shape[0], size=(1000, b.shape[0]))
        bootstrap_gaps = np.median(b[b_draw], axis=1) - np.median(a[a_draw], axis=1)
        gap = np.median(b, axis=0) - np.median(a, axis=0)
        bounds = np.quantile(bootstrap_gaps[:, 1] - bootstrap_gaps[:, 0], [.025, .975])
        comparisons[reference] = {
            "B_minus_A_median_gap_primary_minus_reference": gap[1] - gap[0],
            "CI_95_record_bootstrap": bounds.tolist(),
        }
    return {
        "status": "PASS",
        "record_model_rows": len(rows),
        "group_summary_rows_verified": len(summary),
        "representative_traces_verified": len(rep_errors),
        "max_absolute_representative_PMI_rounding_error": max(rep_errors.values()),
        "robustness_summary_rows_verified": len(robust),
        "paired_provisional_index_group_gap_differences": comparisons,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/runs/development"))
    args = parser.parse_args()
    result = audit(args.run_dir)
    payload = json.dumps(result, indent=2) + "\n"
    (args.run_dir / "metrics" / "fault_verification.json").write_text(payload, encoding="utf-8")
    print(payload, end="")
