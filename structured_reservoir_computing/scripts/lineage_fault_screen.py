"""Reconcile the frozen 21-model PMI screen against current inputs and checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from src_gearbox.data import sha256
from src_gearbox.diagnostics.fault_metrics import summarize

MODES = {
    "pmi_course": "lecturer_compatible", "pmi_lag_corrected": "lag_corrected",
    "pmi_math": "mathematically_specified", "pmi_unrelated_0p63_rev": "pmi_unrelated",
}


def screen(earlier: Path, live: Path, out: Path) -> dict:
    previous = earlier / "outputs" / "expanded"
    protected = previous / "diagnostics"
    prior_freeze_file = earlier / "prior/gearbox_dhon_project/outputs/runs/development/frozen_manifest.json"
    frozen = json.loads(prior_freeze_file.read_text(encoding="utf-8"))
    scope = json.loads((protected / "diagnostic_status.json").read_text(encoding="utf-8"))
    if scope["prior_freeze_sha256"] != sha256(prior_freeze_file):
        raise ValueError("Prior diagnostic freeze identity mismatch")
    lane = live / "comparator_live" if (live / "comparator_live").is_dir() else live
    actual = json.loads((lane / "fault_run_manifest.json").read_text(encoding="utf-8"))
    required = {k:v for k,v in actual["input_sha256"].items() if k.endswith(".mat")}
    if actual["target_mask"] != "1792:N" or required != frozen["development_inputs"]:
        raise ValueError("Actual raw MAT file hashes or score window differ from the earlier screen")
    if actual["input_sha256"]["fdt_featcal.m"] != scope["instructor_script_sha256"]:
        raise ValueError("Instructor MATLAB source hash changed")
    registry = json.loads((ROOT / "configs/baselines/selection.json").read_text(encoding="utf-8"))
    checkpoints = {}
    for name, model in registry["models"].items():
        local = ROOT / "checkpoints" / "baselines" / f"{name}.npz"
        old = previous / "models" / f"{name}.npz"
        signature = model["prior_checkpoint_sha256"]
        if sha256(local) != signature or sha256(old) != signature:
            raise ValueError(f"Checkpoint lineage digest failed for {name}")
        checkpoints[name] = signature

    archived = pd.read_csv(protected / "fault_screen_per_record.csv")
    fresh = pd.read_csv(live / "fault_per_record.csv")
    archived = archived[archived.model.isin(["raw", *checkpoints])].rename(columns=MODES)
    fields = ["model", "record_id", "group", "eligible", *MODES.values()]
    raw = archived.loc[archived.model.eq("raw"),fields].sort_values("record_id")
    matched = fresh.loc[fresh.model.eq("raw") & fresh.record_id.isin(raw.record_id), fields].sort_values("record_id")
    pd.testing.assert_frame_equal(raw.reset_index(drop=True), matched.reset_index(drop=True),
                                  atol=2e-12, rtol=0)
    if not archived.first_sample.eq(1792).all() or not archived.target_count.eq(30208).all():
        raise ValueError("Inherited predictions do not support the current course window")
    if (archived.groupby("model").record_id.nunique() != raw.record_id.nunique()).any():
        raise ValueError("Inherited prediction scopes differ")

    # Use live values for forecast roles already replayed on the full corpus.
    direct = fresh.loc[fresh.record_id.isin(raw.record_id), fields].copy()
    inherited = archived.loc[~archived.model.isin(direct.model), fields].copy()
    direct["scope"] = "live_new_project"
    inherited["scope"] = "protected_prior_same_checkpoints_and_raw_hashes"
    join = pd.concat([direct, inherited], ignore_index=True)
    if join.duplicated(["model", "record_id"]).any():
        raise ValueError("Duplicate diagnostic record model")
    out.mkdir(parents=True, exist_ok=True)
    join.to_csv(out / "fault_per_record.csv", index=False)
    summary = summarize(join, out, primary="primary_pi_src")
    proof = {
        "status": "PASS", "role": "supplementary 41-record screen; not the full diagnostic corpus",
        "record_ids": raw.record_id.tolist(), "label_basis": "provisional index only",
        "eligible_a": int(raw[raw.eligible & raw.group.eq("A")].record_id.nunique()),
        "eligible_b": int(raw[raw.eligible & raw.group.eq("B")].record_id.nunique()),
        "direct_models": sorted(direct.model.unique()), "inherited_models": sorted(inherited.model.unique()),
        "same_checkpoint_sha256": checkpoints,
        "input_data_sha256": required,
        "previous_screen_sha256": sha256(protected / "fault_screen_per_record.csv"),
        "previous_freeze_sha256": scope["prior_freeze_sha256"],
        "validation": "all checkpoint SHA-256 and raw MAT file hashes match; same mask and recomputed raw PMI to 2e-12",
        **summary,
    }
    (out / "screen_lineage.json").write_text(json.dumps(proof, indent=2)+"\n", encoding="utf-8")
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("earlier_expanded_project", type=Path)
    parser.add_argument("--live", type=Path, default=ROOT / "results/fault")
    parser.add_argument("--out", type=Path, default=ROOT / "results/fault/lineage_screen")
    args = parser.parse_args()
    with threadpool_limits(1):
        print(json.dumps(screen(args.earlier_expanded_project, args.live, args.out), indent=2))
