"""Assemble the SRC study companion and verify its evidence links."""

import csv
import json
import math
from pathlib import Path
import re

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT/"docs/companion_source"
TARGET = PROJECT/"docs/companion_guide.md"

NOTEBOOK_CHAPTERS = (
    ("00", "1, 6, 13, 22"),
    ("01", "2, 3, 9–11, 14"),
    ("02", "4, 5, 17, 21"),
    ("03", "7–12, 15"),
    ("04", "6, 15, 16"),
    ("05", "8, 11, 12, 15, 22"),
    ("06", "17, 18, 21, 23"),
    ("07", "10, 13, 19, 20"),
    ("08", "12, 18–20, 23"),
    ("09", "6, 18, 22, 23"),
    ("10", "6, 13, 20, 22–24"),
)


def csv_rows(relative):
    with (PROJECT/"results"/relative).open(encoding="utf-8",newline="") as file:
        return list(csv.DictReader(file))


def json_record(relative):
    return json.loads((PROJECT/"results"/relative).read_text(encoding="utf-8"))


def manuscript(name):
    return (SOURCE/f"{name}.md").read_text(encoding="utf-8").strip()


def diagnostics_chapter():
    text = manuscript("diagnostics")
    text = re.sub(r"(?m)^## ", "### ", text)
    text = re.sub(r"(?m)^### (Exercises|Full diagnostic replay used for release|Measured intervention results)$",
                  r"#### \1", text)
    text = re.sub(r"(?m)^# .+", "## 19. Fault detection from causal residuals", text, count=1)
    index = iter(range(16,19))
    text = re.sub(r"(?ms)^\$\$\n(.+?)\n\$\$$",
                  lambda match:"$$\n"+match[1]+rf" \tag{{{next(index)}}}"+"\n$$",text)
    return text


def table(headers, values):
    def row(columns):
        return "| "+" | ".join(str(c).replace("|",r"\|").replace("\n"," ") for c in columns)+" |"
    return "\n".join([row(headers),row(["---"]*len(headers)),*(row(columns) for columns in values)])


def fmt(value, places=6):
    if value == "" or value is None:
        return "—"
    number = float(value)
    return format(number,f".{places}f") if math.isfinite(number) else "—"


def evidence_appendix():
    primary = json_record("metrics/src_summary.json")
    comparison = json_record("metrics/comparison_summary.json")
    freeze = json_record("frozen/availability.json")
    fault = json_record("fault/fault_status.json")
    contrasts = csv_rows("ablations/identical_adapter_controls.csv")
    ranking = csv_rows("metrics/all_model_comparison.csv")
    local = csv_rows("metrics/local_statistical_reference.csv")
    external = csv_rows("metrics/published_paper_reference.csv")
    group = csv_rows("fault/fault_group_summary.csv")
    pulse = csv_rows("fault/pulse_summary.csv")
    stresses = csv_rows("fault/rpm_stress_summary.csv")
    seed = csv_rows("metrics/baseline_seed_sensitivity_summary.csv")
    timing = csv_rows("metrics/vibration_rpm_parity.csv")
    ablations = csv_rows("ablations/src_ablations.csv")
    paired = csv_rows("statistical_tests/paired_moving_block.csv")
    evidence = ["""## Generated evidence appendix

Tables in this appendix load the release ledgers. Rerun the builder after changing a result scope. A reported MSE is always in the project’s squared standardized vibration units unless an external source uses its own scale. Timing describes the measured CPU stages. A dash means a field lacks a measured value in the source row."""]
    outcomes = [
        ("Training-only selection", primary["trial_id"]),
        ("Guarded mean MSE", fmt(primary["fold_mse"])),
        ("Guarded stress MSE", fmt(primary["fold_stress_mse"])),
        ("Guarded utility", fmt(primary["utility"])),
        ("Training MSE on common mask", fmt(primary["train_mse"])),
        ("Development MSE", fmt(primary["validation_mse"])),
        ("Development RMSE", fmt(primary["validation_rmse"])),
        ("Development MAE", fmt(primary["validation_mae"])),
        ("Warm CPU median, milliseconds", fmt(primary["warm_median_ms"],2)),
        ("Selected retained feature columns", primary["feature_columns"]),
        ("Stored coefficients including proxy", primary["parameters"]),
        ("Selected history support, samples", primary["history"]),
        ("Target count", primary["targets"]),
        ("Frozen independent file", freeze["status"]),
        ("Checkpoint SHA-256", f"`{primary['checkpoint_sha256']}`"),
        ("Source commit", f"`{primary['source_commit']}`"),
        ("Target index SHA-256", f"`{primary['mask_sha256']}`"),
    ]
    evidence.extend(["### A.1. Primary measurement and aligned comparison",
                     table(("Primary record","Measured value"),outcomes),
                     f"The selected MSE exceeds the archived corrected SRC by {fmt(comparison['primary_minus_anchor'])} "
                     f"({abs(comparison['relative_improvement_over_archive_pct']):.2f}% above that archived value). "
                     f"It is {comparison['relative_improvement_over_required_baseline_pct']:.2f}% below the strongest required "
                     f"comparator. All {comparison['required_baselines_checked']} required families and five archival rows share the exact target mask. "
                     "The required families use prior tuned frozen fits replayed in the new project. Matched covariate and seed checks were newly fitted where identified.",
                     table(("Model","Scope","MSE","RMSE","MAE","CPU median, ms","Parameters"),
                           [(r["display"],r["scope"].replace("_"," "),fmt(r["mse"]),fmt(r["rmse"]),fmt(r["mae"]),
                             fmt(r["warm_median_ms"],2),fmt(r["parameters"],0)) for r in ranking]),
                     "The ranking bars show this table's same-mask errors. Timing and forensic paper rows are interpreted separately.",
                     "![Held-out development ranking with the primary, archive, and 21 required families](../figures/comparison/mse_ranking.png)"])
    evidence.extend(["### A.2. Paired uncertainty and capacity controls",
                     "The paired error direction below is comparator minus primary PI-SRC. A negative value favors the comparator on this record. "
                     "Bands are percentile intervals from contiguous within-record blocks. Their interpretation is conditional on this acquisition.",
                     table(("Comparator","Block length, samples","MSE difference","2.5th percentile","97.5th percentile"),
                           [(r["model"],r["block_samples"],fmt(r["delta_mse"]),fmt(r["interval_low"]),fmt(r["interval_high"]))
                            for r in paired if r["model"] in {"archived_src_corrected","linear","sparse_lpv_arma","cnn"}]),
                     "Each named ablation uses its own stored configuration and identical score window. These values do not change the selected model.",
                     table(("Control","Development MSE","Training MSE","Feature columns","Parameters"),
                           [(r["display"],fmt(r["mse"]),fmt(r["train_mse"]),r["feature_columns"],r["parameters"]) for r in ablations]),
                     "The same adapter settings also act on matched conventional and polynomial forecasts.",
                     table(("Controlled base","Uncorrected MSE","Matched correction MSE","Reduction"),
                           [(r["model"],fmt(r["uncorrected_mse"]),fmt(r["matched_correction_mse"]),fmt(r["gain_mse"]))
                            for r in contrasts])])
    evidence.extend(["### A.3. Input parity and seed sensitivity",
                     "Negative MSE differences indicate an advantage to measured speed compared with the fixed nominal-speed fit.",
                     table(("Model","Measured speed","Nominal speed","Measured minus nominal"),
                           [(r["model"],fmt(r["measured_mse"]),fmt(r["vibration_only_mse"]),
                             fmt(r["delta_measured_minus_vibration_only"])) for r in timing]),
                     "Seed dispersion uses three fits of the selected configuration on the same input split. "
                     "It measures optimization variability under the tested initializations.",
                     table(("Model","Seeds","Mean MSE","Sample standard deviation"),
                           [(r["model"],r["seeds"],fmt(r["mean_mse"]),fmt(r["sd_mse"])) for r in seed])])
    evidence.extend(["### A.4. External and local reference conventions",
                     "Published results below use the authors' reported analysis segment. The development table beneath it shows local "
                     "coefficients, timing scope and replay conventions. A literature MSE never enters the matched primary ranking.",
                     table(("Published model","Authors’ training MSE","Authors’ test MSE"),[
                         (r["model"],fmt(r["train_mse"],4),fmt(r["authors_test_segment_mse"],4)) for r in external]),
                     table(("Local model","Convention","Schedule","Start","Reported-window MSE","Common-mask MSE"),
                           [(r["model"],r["convention"].replace("_"," "),r["scheduling"],r["start"],fmt(r["development_mse"]),
                             fmt(r["common_mask_mse"])) for r in local])])
    evidence.extend(["### A.5. Measured diagnostic and sensor checks",
                     f"The full live cohort gives {fault['model_record_rows']} model-record results and "
                     f"{fault['eligible_candidates']} eligible records after removing the two exact overlaps. "
                     "The table uses the instructor-compatible definition with a one-sample indexing offset relative to exact shaft period. "
                     "Every group contrast is conditional on the file-index group. No classification threshold was tested.",
                     table(("Model","n A","n B","Median A","Median B","Median gap","Extreme margin","Conditional AUC"),
                           [(r["model"],r["n_A"],r["n_B"],fmt(r["median_A"]),fmt(r["median_B"]),fmt(r["median_gap_B_minus_A"]),
                             fmt(r["margin_min_B_minus_max_A"]),fmt(r["conditional_auc_B_high"],4))
                            for r in group if r["metric"]=="lecturer_compatible"]),
                     "![Conditional PMI distributions after excluding exact overlaps](../figures/fault/fault_group_distribution.png)",
                     "The stress conditions assess known interventions on the available development acquisition.",
                     table(("Model","Cases","Successful","Clean MSE","Mean stress MSE","Worst stress MSE","Worst condition"),
                           [(r["model"],r["scenarios"],r["successful"],fmt(r["clean_mse"]),fmt(r["mean_stress_mse"]),
                             fmt(r["worst_stress_mse"]),r["worst_condition"]) for r in stresses]),
                     "Synthetic pulse outcomes measure an intervention response. Their case counts and positive response counts are descriptive.",
                     table(("Model","Injection type","Cases","Positive PMI shifts","Median distortion","Median waveform correlation"),
                           [(r["model"],r["kind"],r["cases"],r["positive_delta_count"],fmt(r["median_distortion"],4),
                             fmt(r["median_pulse_correlation"],4)) for r in pulse])])
    return "\n\n".join(evidence)


def crosswalk():
    books = sorted((PROJECT/"notebooks").glob("*.ipynb"))
    notebook_rows = []
    for prefix, chapters in NOTEBOOK_CHAPTERS:
        book = next(path for path in books if path.name.startswith(prefix))
        relative = book.relative_to(PROJECT).as_posix()
        notebook_rows.append((f"[{book.stem.replace('_',' ')}](../{relative})",chapters))
    return "\n\n".join(["## Notebook and artifact crosswalk",
                         "Start with the relevant companion chapter, open its executed notebook, and locate the result ledger before running a new fit.",
                         table(("Executed notebook","Companion chapters"),notebook_rows),
                         table(("Evidence or deliverable","Companion chapters"),[
                             ("[Input manifest](../inputs/input_manifest.csv)","2, 3, 13"),
                             ("[Data preparation and EDA](eda_findings.md)","9–14"),
                             ("[Primary source model](../src/src_gearbox/src_model.py)","7–12, 15"),
                             ("[Checkpoint registry](../checkpoints/src_registry.json)","15, 22"),
                             ("[Trial ledger](../results/optimization/trials.csv)","16, 23"),
                             ("[Matched comparison](../results/metrics/all_model_comparison.csv)","17, 18, 21, 23"),
                             ("[Fault evidence](../results/fault/fault_group_summary.csv)","19, 20"),
                             ("[Pipeline](../scripts/run_pipeline.py)","22"),
                             ("[Office report and figures](../report/SRC_Final_Course_Report.pdf)","24"),
                         ])])


def anchor(text):
    return re.sub(r"[^\w -]","",text.lower()).replace(" ","-")


def check_links(text):
    fence = False
    missing = []
    target_anchors = {anchor(line) for line in re.findall(r"(?m)^#{1,6} (.+)$",text)}
    for line in text.splitlines():
        if line.startswith("```"):
            fence = not fence
        if not fence:
            for link in re.findall(r"\]\(([^)]*)\)",line):
                if not link or "://" in link:
                    continue
                if link.startswith("#"):
                    if link[1:] not in target_anchors:
                        missing.append(link)
                elif not (TARGET.parent/link.split("#")[0]).is_file():
                    missing.append(link)
    if missing:
        raise FileNotFoundError(f"Companion evidence links: {sorted(set(missing))}")


def main():
    sections = [manuscript("foundations"),manuscript("architecture"),manuscript("comparison"),
                diagnostics_chapter(),manuscript("synthesis")]
    if (SOURCE/"laboratory.md").is_file():
        sections.append(manuscript("laboratory"))
    sections += [evidence_appendix(),crosswalk(),manuscript("references")]
    text = "\n\n".join(sections)+"\n"
    headings = re.findall(r"(?m)^## (.+)$",text)
    toc = "\n".join(f"- [{title}](#{anchor(title)})" for title in headings if title!="Reading map")
    text = text.replace("## Reading map","## Contents\n\n"+toc+"\n\n## Reading map",1)
    control = set(c for c in text if ord(c) < 32 and c not in "\t\n")
    tags = [int(n) for n in re.findall(r"\\tag\{(\d+)\}",text)]
    chapters = [int(n) for n in re.findall(r"(?m)^## (\d+)\. ",text)]
    assert not control and tags[:22]==list(range(1,23)) and chapters[:25]==list(range(1,26))
    check_links(text)
    TARGET.write_text(text,encoding="utf-8",newline="\n")
    print(f"{len(text.split()):,} whitespace-delimited words, 25 numbered chapters, {len(tags)} numbered equations")
    print(f"{text.count('~~~mermaid')+text.count(chr(96)*3+'mermaid')} Mermaid figures, 11 notebook cross-references")
    print(f"Wrote {TARGET.relative_to(PROJECT)}. All local links verified.")


if __name__=="__main__":
    main()
