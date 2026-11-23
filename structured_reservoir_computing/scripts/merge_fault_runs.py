"""Join independent live fault lanes after verifying raw data and mask identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from src_gearbox.diagnostics.fault_metrics import summarize
from src_gearbox.diagnostics.pmi import lecturer_score
from src_gearbox.diagnostics.visualize import plot


def _verify_numeric_outputs(out: Path, frame: pd.DataFrame) -> dict:
    paired = frame.groupby("record_id")[["first_target","target_count","eligible"]].nunique()
    assert paired.eq(1).all().all()
    assert frame.first_target.eq(1792).all()
    trace_count, maximum_error = 0, 0.
    trace_file = out / "fault_traces.npz"
    if trace_file.is_file():
        with np.load(trace_file, allow_pickle=False) as traces:
            for row in frame.itertuples():
                key = f"{row.record_id}:{row.model}:residual"
                if key not in traces:
                    continue
                prediction = lecturer_score(traces[key], traces[f"{row.record_id}:rpm"],
                                           traces[f"{row.record_id}:time"]).value
                np.testing.assert_allclose(prediction, row.lecturer_compatible, atol=2e-8, rtol=0)
                trace_count += 1
                maximum_error = max(maximum_error, abs(prediction-row.lecturer_compatible))
    rpm_count = 0
    rpm_file = out / "rpm_stress.csv"
    if rpm_file.is_file():
        rpm = pd.read_csv(rpm_file)
        succeeded = rpm.loc[rpm.status.eq("ok")]
        angle = succeeded.loc[succeeded.scope.eq("diagnostics")]
        np.testing.assert_allclose(angle.mse, angle.clean_mse, atol=1e-12, rtol=0)
        np.testing.assert_allclose(succeeded.delta_pmi, succeeded.lecturer_compatible-succeeded.clean_pmi,
                                   atol=1e-12, rtol=0)
        rpm_count = len(succeeded)
    pulse_count = 0
    pulse_file = out / "pulse_preservation.csv"
    if pulse_file.is_file():
        pulse = pd.read_csv(pulse_file).query("status == 'ok'")
        ratio = pulse.fault_distortion + 2*pulse.residual_pulse_projection - 1
        np.testing.assert_allclose(pulse.residual_energy_retention, ratio, atol=1e-12, rtol=0)
        assert pulse.residual_pulse_correlation.abs().le(1+1e-12).all()
        pulse_count = len(pulse)
    return {"representative_pmi_rows_verified": trace_count, "max_saved_trace_pmi_error": maximum_error,
            "successful_rpm_stress_rows_verified": rpm_count, "successful_pulse_rows_verified": pulse_count}


def merge(lanes: list[Path], out: Path, figures: Path, primary: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    manifests = [json.loads((lane / "fault_run_manifest.json").read_text()) for lane in lanes]
    assert all(m["target_mask"] == "1792:N" for m in manifests)
    inputs = manifests[0]["input_sha256"]
    assert all(m["input_sha256"] == inputs for m in manifests)
    counts = {}
    for table in ("fault_per_record", "fault_eligibility", "rpm_stress", "rpm_stress_regimes",
                  "rpm_stress_summary", "pulse_preservation", "pulse_summary"):
        seen = [pd.read_csv(p / f"{table}.csv") for p in lanes if (p / f"{table}.csv").is_file()]
        if not seen:
            continue
        if table == "fault_per_record":
            # Every lane computes the same raw diagnostic independently.
            profile = seen[0][seen[0].model.eq("raw")].sort_values("record_id")
            values = profile.drop(columns=["prediction_seconds","diagnostic_seconds"], errors="ignore")
            for other in seen[1:]:
                raw = other[other.model.eq("raw")].sort_values("record_id")
                pd.testing.assert_frame_equal(values.reset_index(drop=True),
                    raw[values.columns].reset_index(drop=True), atol=1e-12, rtol=0)
            frame = pd.concat([seen[0], *[other.loc[other.model.ne("raw")] for other in seen[1:]]],
                              ignore_index=True)
            assert not frame.duplicated(["record_id", "model"]).any()
        elif table == "fault_eligibility":
            frame = seen[0]
            for other in seen[1:]:
                pd.testing.assert_frame_equal(frame,other)
        else:
            frame = pd.concat(seen, ignore_index=True)
        frame.to_csv(out / f"{table}.csv", index=False)
        counts[table] = len(frame)

    for name in ("fault_traces", "pulse_example"):
        arrays = {}
        for lane in lanes:
            source = lane / f"{name}.npz"
            if source.is_file():
                with np.load(source, allow_pickle=False) as loaded:
                    for key in loaded.files:
                        if key in arrays:
                            np.testing.assert_array_equal(arrays[key], loaded[key])
                        else:
                            arrays[key] = loaded[key]
        if arrays:
            np.savez_compressed(out / f"{name}.npz", **arrays)
    summary = summarize(pd.read_csv(out / "fault_per_record.csv"), out, primary=primary)
    statuses = [json.loads((lane / "fault_status.json").read_text()) for lane in lanes
                if (lane / "fault_status.json").is_file()]
    common_status = statuses[0]
    frame = pd.read_csv(out / "fault_per_record.csv")
    common_status.update(models=list(frame.model.drop_duplicates()), model_record_rows=len(frame),
                         invalid_rows=int(frame.status.ne("ok").sum()),
                         elapsed_seconds=sum(d["elapsed_seconds"] for d in statuses), **summary)
    (out / "fault_status.json").write_text(json.dumps(common_status, indent=2) + "\n", encoding="utf-8")
    for name in ("matlab_conformance", "availability"):
        documents = [(lane / f"{name}.json").read_text(encoding="utf-8") for lane in lanes
                     if (lane / f"{name}.json").is_file()]
        if documents:
            assert all(json.loads(d) == json.loads(documents[0]) for d in documents)
            (out / f"{name}.json").write_text(documents[0], encoding="utf-8")
    audit = {
        "status": "PASS", "checks": ["exact raw file hashes match", "raw per-record diagnostics match",
            "record IDs and overlap policies match", "representative raw traces match",
            "conditional AUC rederived by ranks and pairwise indicator"], "counts": counts,
        "input_sha256": inputs, "checkpoint_lanes": {path.name: m for path, m in zip(lanes, manifests)},
        **summary, **_verify_numeric_outputs(out, frame),
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    plot(out, figures, primary=primary)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lanes", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "results/fault")
    parser.add_argument("--figures", type=Path, default=ROOT / "figures/fault")
    parser.add_argument("--primary", default="primary_pi_src")
    args = parser.parse_args()
    with threadpool_limits(1):
        print(json.dumps(merge(args.lanes, args.out, args.figures, args.primary), indent=2))
