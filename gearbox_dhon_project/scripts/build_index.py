"""Write course traceability, asset index and package checksums."""
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = "outputs/runs/development"
FINAL = ROOT / "outputs/final"
ROWS = [
    ("QA01","Course/paper/thesis and archive source audit","","docs/prior_work_audit.md","PASS","Sources read, inputs hashed and archive methods inspected."),
    ("QA02","Data availability, units, duplicate checks and official split","","tables/record_inventory.csv","PARTIAL","Exact joint overlap excludes 61 and 63. Condition and original acquisition IDs are unavailable."),
    ("QA03","Standalone Python, pinned dependencies and local smoke","","README.md","PASS","Linux clean-extraction replay and smoke checked. PowerShell instructions supplied."),
    ("QA04","Executed physics EDA and feature evidence","","notebooks/01_Physics_Informed_EDA.ipynb","PASS","Training speed, harmonic tracks and guarded feature screen."),
    ("QA05","Conventional fit/replay and matched errors","","metrics/conventional_metrics.csv","PASS","Strict refits and archive replay on same targets. Paper source segment mismatch documented."),
    ("QA06","Archived DHON and NG-RC replay with correction controls","","metrics/archive_replay.csv","PARTIAL","Available checkpoints replay. Strongest ancestor supplied as forecast vectors only."),
    ("QA07","Learnable oscillator, constraints and numerical verification","","docs/mathematical_formulation.md","PASS","Gradient, recurrence, reset, prefix and causality tests plus matched fixed state."),
    ("QA08","Executed finite joint optimization","","search/dhon_trials.csv","PASS","21 candidates and 63 guarded folds with per-fold predictions and logs."),
    ("QA09","Selected primary and alternate freeze and replay","","frozen_manifest.json","PASS","Training-fold decision and exact checkpoint reload recorded."),
    ("QA10","Matched learned and conventional comparisons","","notebooks/05_Comparative_Model_Performance.ipynb","PASS","Nine trained/naive comparator families plus LPV and archive models."),
    ("QA11","Angular PMI and instructor diagnostic indexing","","docs/fault_diagnostics.md","PARTIAL","Both lags implemented and verified analytically. MATLAB helper and golden parity unavailable."),
    ("QA12","Per-record fault diagnostics with eligibility flags","","metrics/fault_per_record.csv","PASS","200 actual candidate records, overlap exclusion and provisional group-only interpretation."),
    ("QA13","Stress, pulse, oscillator, adapter and sensitivity checks","","metrics/robustness_summary.csv","PARTIAL","Controlled RPM stress, pulse injection, matched DHON ablations and seeds completed. A fully crossed ablation grid and production-streaming stress remain open."),
    ("QA14","Measured efficiency including CPU/GPU","","metrics/timings.csv","PARTIAL","Common CPU batch times and isolated component tables. GPU and streaming deadlines unmeasured."),
    ("QA15","Frozen independent assessment","","outputs/runs/blind_missing/data_availability.json","BLOCKED","Both held-out datasets absent. Inference-only frozen preflight executed."),
    ("QA16","Nine instructional notebooks executed","","notebook_execution.csv","PASS","Nine clean-kernel replay notebooks generated, with recorded times."),
    ("QA17","Numerical and pipeline QA","","pytest.log","PARTIAL","Unit suite, actual checkpoint replay and smoke completed. No GPU/MATLAB parity."),
    ("QA18","Plots, metrics, predictions, search and checkpoint provenance","","outputs/final/artifact_index.md","PASS","Inputs and code/checkpoint hashes, actual arrays, PNGs, PDFs and metrics indexed."),
    ("QA19","Companion, methods, reproducibility and report PDF/source","","docs/companion_guide.md","PARTIAL","All 18 study topics covered, with separate methods and notebook material. Guide length below aspirational 15,000–25,000 words."),
    ("QA20","Editable talks and speaker notes","","presentations/README.md","PASS","8 and 12 minutes, editable slides and render verification."),
    ("QA21","Course marking traceability","","docs/requirements_traceability.md","PASS","Requirements, weights, evidence and statuses tracked."),
    ("QA22","Source package, manifest and fresh replay","","CHECKSUMS.sha256","PASS","ZIP integrity verification and fresh-extraction checkpoint replay. Raw MAT files external."),
    ("P1","Progress organization and presentation",4,"presentations/progress_presentation.pptx","PASS","Eight timed slides plus backup."),
    ("P2","Progress presentation clarity",4,"presentations/progress_speaker_notes.md","PASS","Timed and structured notes, equations and physics figures."),
    ("P3","Progress algorithm understanding",8,"docs/mathematical_formulation.md","PASS","Causal AR/LPV, recurrence, gradient and PMI derivations."),
    ("P4","Progress algorithm novelty",4,"docs/dhon_results_interpretation.md","PASS","Oscillator dynamics are learned with matched fixed-state control."),
    ("F1","Final report organization and presentation",16,"docs/final_course_report.md","PASS","Report source and rendered PDF with measured results and context."),
    ("F2","Final presentation clarity",16,"presentations/final_presentation.pptx","PASS","Twelve-minute deck and speaker notes."),
    ("F3","Final algorithm performance",32,"metrics/summary.csv","PARTIAL","Development MSE, CPU and conditional PMI measured. Blind datasets, labels and GPU missing."),
    ("F4","Final algorithm novelty",16,"search/primary_dhon_learning.json","PASS","Constrained state parameters trained, inspected and ablated."),
]


def relative(evidence):
    return evidence if evidence.startswith(("docs/", "notebooks/", "outputs/", "presentations/")) or evidence in {
        "README.md", "CHECKSUMS.sha256"} else f"{RUN}/{evidence}"


def main():
    fields = ("requirement_id", "requirement", "weight_pct", "evidence", "status", "reason")
    filled = [(*row[:3], relative(row[3]), *row[4:]) for row in ROWS]
    FINAL.mkdir(parents=True, exist_ok=True)
    with (ROOT / "outputs/requirements_traceability.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(filled)
    md = ["# Course and master-prompt traceability", "",
          "Statuses refer to delivered evidence. A PASS is a verification label and does not predict a course grade.", "",
          "| ID | Requirement | Weight | Evidence | Status | Scope |", "| --- | --- | ---: | --- | --- | --- |"]
    md += [f"| {i} | {r} | {w} | [{Path(p).name}](../{p}) | {s} | {d} |" for i,r,w,p,s,d in filled]
    (ROOT / "docs/requirements_traceability.md").write_text("\n".join(md)+"\n", encoding="utf-8")
    checklist = ["# Completion checklist", "",
                 "Run `development`. These are verification statuses for the supplied brief and master prompt.", ""]
    checklist += [f"- **{s}** · {r}. {d} [Evidence](../../{p})" for i,r,w,p,s,d in filled if i.startswith("QA")]
    (FINAL / "completion_checklist.md").write_text("\n".join(checklist)+"\n", encoding="utf-8")
    index = ["# Artifact index", "",
             "Development run throughout, except the separate smoke and absent-input blind preflight.", "",
             "| Artifact | Evidence |", "| --- | --- |"]
    for p in ("README.md", "docs/reproduction_summary.md", "docs/final_course_report.md",
              "outputs/final/final_course_report.pdf", "docs/companion_guide.md",
              "docs/requirements_traceability.md", "outputs/final/completion_checklist.md",
              "outputs/final/RESULTS_SUMMARY.json", "notebooks/", "presentations/",
              f"{RUN}/metrics/summary.csv", f"{RUN}/metrics/fault_group_summary.csv",
              f"{RUN}/search/", f"{RUN}/checkpoints/", f"{RUN}/predictions/",
              f"{RUN}/results_verification.json", f"{RUN}/frozen_manifest.json"):
        index.append(f"| {Path(p).name} | [Open](../../{p}) |")
    index += ["", "## Figure index", "",
              "Developed from the run's actual EDA, fit, search and predictions except the illustrative architecture and oscillator fixtures. Vector PDF siblings are supplied where generated.", ""]
    for fig in sorted((ROOT / RUN / "figures").glob("*.png")):
        p = fig.relative_to(ROOT).as_posix()
        index.append(f"- {fig.stem.replace('_', ' ').capitalize()}. [PNG](../../{p})")
    (FINAL / "artifact_index.md").write_text("\n".join(index)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
