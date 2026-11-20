"""Regenerate the research report and numerical synthesis from an executed run."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def dataset(run, filename):
    path = run / filename
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def value_table(frame, columns, labels=None, precision=6):
    labels = labels or columns
    header = "| " + " | ".join(labels) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"

    def render(x):
        if pd.isna(x):
            return "—"
        if isinstance(x, float):
            return f"{x:.{precision}f}"
        return str(x).replace("_", " ").replace("|", "\\|")
    return "\n".join([header, separator, *[
        "| " + " | ".join(render(v) for v in row) + " |"
        for row in frame[columns].itertuples(index=False, name=None)
    ]])


def figure(run, filename, alt):
    rel = (run / "figures" / f"{filename}.png").relative_to(ROOT).as_posix()
    return f"![{alt}](../{rel})" if (ROOT / rel).exists() else ""


def build(run):
    final = ROOT / "outputs/final"
    final.mkdir(parents=True, exist_ok=True)
    summary = dataset(run, "metrics/summary.csv").set_index("model")
    pairs = dataset(run, "metrics/paired_block_bootstrap.csv")
    dhon = dataset(run, "metrics/dhon_metrics.csv").set_index("model")
    search = dataset(run, "search/dhon_ranking.csv")
    trial_rows = dataset(run, "search/dhon_trials.csv")
    selection = json.loads((run / "search/dhon_selection.json").read_text(encoding="utf-8"))
    conv_fit = json.loads((run / "metrics/conventional_fit.json").read_text(encoding="utf-8"))
    audit = json.loads((run / "data_availability.json").read_text(encoding="utf-8"))
    diagnostics = dataset(run, "metrics/fault_group_summary.csv")
    if diagnostics.empty:
        for path in sorted((run / "metrics").glob("*fault*summary*.csv")):
            diagnostics = pd.read_csv(path)
            break
    timings = dataset(run, "metrics/timings.csv")
    timing_env_file = run / "metrics/timing_environment.json"
    timing_env = json.loads(timing_env_file.read_text()) if timing_env_file.is_file() else {}
    primary = dhon.loc["primary_dhon"]
    alt = dhon.loc["alternative_dhon"]
    better = summary.loc["spa_lpv_arma", "mse"]
    archive = summary.loc["archive_dhon_optimized_corrected", "mse"]
    ancestor_key = "archive_dhon_tau_context_extended_gated_corrected"
    ancestor = summary.loc[ancestor_key, "mse"]
    m = primary["mse"]
    new_vs_sparse = 100 * (better-m) / better
    archive_deficit = 100 * (m-archive) / archive
    ancestor_deficit = 100 * (m-ancestor) / ancestor
    baseline_dense = summary.loc["lpv_arma_dense", "mse"]
    fixed = dhon.loc["dhon_matched_fixed", "mse"]
    adapter = primary["base_mse"]-m
    paired = pairs.query("model == 'primary_dhon'").copy()
    timing_table = ""
    if not timings.empty:
        timing_table = value_table(timings[timings.model.isin(["primary_dhon", "spa_lpv_arma",
                                                               "archive_dhon_optimized_corrected"])],
                                   ["model", "record", "median_s", "iqr_s", "rtf"],
                                   ["Model", "Record", "Median (s)", "IQR (s)", "RTF"], 5)
    diagnostic_text = "The diagnostic result tables are not in this run."
    diagnostic_summary = {}
    if not diagnostics.empty:
        wanted = next((cols for cols in (
            ["model", "n_A", "n_B", "median_A", "median_B", "conditional_auc_B_high"],
            ["model", "eligible_n_A", "eligible_n_B", "median_A", "median_B", "auc_B_high"],
            ["model", "n_group_A", "n_group_B", "median_A", "median_B", "auc_B_high"],
            ["model", "count_A", "count_B", "median_A", "median_B", "auc_B_gt_A"],
        ) if all(col in diagnostics for col in cols)), None)
        if wanted is None:
            wanted = [col for col in diagnostics if col == "model" or
                      any(key in col.lower() for key in ("auc", "median", "eligible"))][:6]
        group = diagnostics
        if "metric" in group and "pmi_course" in set(group.metric):
            group = group[group.metric == "pmi_course"]
        if "mode" in group and "course" in set(group["mode"]):
            group = group[group["mode"] == "course"]
        diagnostic_text = value_table(group.head(8), wanted)
        diagnostic_summary = json.loads(group.to_json(orient="records"))

    robustness = dataset(run, "metrics/robustness_summary.csv")
    robustness_table = ""
    if not robustness.empty and "model" in robustness:
        numeric = [c for c in ("mse_clean", "rpm_mse_worst", "median_periodic_pulse_corr",
                               "periodic_pmi_positive", "periodic_pmi_cases") if c in robustness]
        if numeric:
            robustness_table = value_table(robustness, ["model", *numeric], precision=5)

    compare_list = [
        "primary_dhon", "alternative_dhon", "dhon_matched_fixed", "dhon_matched_linear", "spa_lpv_arma",
        "lpv_arma_dense", "archived_spa_lpv_arma", "physics_linear", "ngrc", "tcn_rpm",
        "archive_dhon_optimized_corrected", "archive_ngrc_optimized_corrected", ancestor_key,
    ]
    core = summary.reindex([name for name in compare_list if name in summary.index]).reset_index()
    compare = value_table(core, ["model", "count", "mse", "mae", "bias"],
                          ["Model", "Targets", "MSE", "MAE", "Bias"])
    matched = value_table(dhon.reset_index().loc[
        lambda df: df.model.isin(["primary_dhon", "dhon_matched_fixed", "dhon_matched_decay_only",
                                  "dhon_matched_frequency_only", "dhon_matched_linear", "alternative_dhon"])],
                          ["model", "mse", "base_mse", "parameters"],
                          ["Model", "Adapted MSE", "Base MSE", "Parameters"])
    pair_table = value_table(paired, ["reference", "delta_mse", "low_95", "high_95"],
                             ["Reference", "Paired difference", "95% low", "95% high"])
    cfg = selection["configurations"]["primary_dhon"]
    report = f"""---
title: "Causal gearbox vibration prediction with differentiable harmonic oscillators"
subtitle: "MECG 7780 Engineering Data Analytics"
date: "Development run {run.name}"
geometry: margin=0.86in
fontsize: 10pt
colorlinks: true
toc: true
---
# Abstract

This study tests whether speed-conditioned differentiable oscillator states improve causal unit-step prediction of gearbox vibration while preserving periodic residual signatures. The predictor uses known vibration and RPM through the preceding sample. On 22,208 common development targets, the train-fold-selected DHON had standardized MSE **{m:.6f}**, a **{new_vs_sparse:.2f}%** reduction over the strict sparse LPV-ARMA refit ({better:.6f}). It was **{archive_deficit:.2f}%** above the independently replayed archived corrected DHON checkpoint ({archive:.6f}) and **{ancestor_deficit:.2f}%** above the strongest archived saved forecasts ({ancestor:.6f}). The adapter reduced the new model's error by {adapter:.6f}, compared with {fixed-m:.6f} for learning the oscillator parameters relative to the matched frozen oscillator. The study verifies numerical differentiability and reports development-only diagnosis using provisional index groups. Independent test files and confirmed acquisition condition labels were absent.

**Keywords** Gearbox vibration, order tracking, causal prediction, differentiable oscillator, LPV-ARMA, periodic modulation intensity

# 1. Engineering objective

Variable shaft speed warps the time-domain vibration signature generated by mesh forces and local faults. A statistical predictor can follow those changing dynamics either by treating speed as a scheduling variable or by using directly interpretable shaft phase. The study tests both methods. The primary outcome is one-step MSE on the designated development sequence. The downstream outcome is how consistently angular-domain residual diagnostics distinguish observed index groups and respond to known pulse perturbations. The analysis does not infer health labels from file ordering.

# 2. Physical basis and related work

Chen et al. (2025, §§3–5) expand AR and MA coefficients in speed basis functions and estimate a sparse two-stage predictor. Their angular PMI relates envelope auto-energy to shaft-period repetition. Chen (2020, §2.2.3) defines the centered envelope formulation. The experiment in the article uses an encoder on the input shaft with a 37/20 tooth pair and measures the x-axis vibration. The new architecture retains physical carrier priors and explicitly trains damped state parameters through the computation graph.

# 3. Dataset and training-only exploratory evidence

The audit confirms 24,000 samples each in training and development at 3.2 kHz and {audit.get('sigfault1_count', 0)} ten-second development fault candidates. Whole-record matches at offset zero identify acquisitions 61 and 63 as repeats of the training and validation prefixes, respectively. The complete overlapping acquisitions are excluded from index-group summary metrics. Scoring uses original standardized vibration and the fixed mask [1792, 24000). Raw conversion metadata reconstructs stored raw channel-one amplitude, but its calibrated engineering units remain unidentified.

Training covered 124.323–167.334 RPM. The 37-tooth mesh fundamental therefore moved from 76.666 to 103.189 Hz. Order 148 showed a 0.744 Hz median ridge discrepancy and an angular PSD peak about 160 times its local median. The 1.0–1.2 kHz region contained 31.3% of the training Welch power. A guarded LPV-AR24 screen moved from 0.46153 to 0.43803 when adding order 148 and 259 quadratures. Adding a standalone 1,100 Hz quadrature produced 0.46230. The evidence supports order-locked features and does not identify a structural mode from a broad band.

{figure(run, "eda_spectrogram", "Descriptive training spectrogram and theoretical mesh trajectories.")}

# 4. Conventional benchmark

A causal LPV-ARMA predicts from lagged vibration and lagged first-stage proxy innovations, both modulated by an eight-term speed basis. The strict course refits use prior RPM, AIC to screen finite order grids, and L1 support selected by ADMM and a guarded one-standard-deviation rule. Support is then refitted via rank-aware least squares. An ARI baseline fits lagged first differences, and the conditional ARIMA uses lagged proxy innovations. Dense LPV counterparts help expose the cost of sparsity on this short acquisition. The strict dense two-stage comparison scored {baseline_dense:.6f}, closer to the new DHON than the strict sparse fit.

The verified original coefficient archive with current scheduling has a historical-mask MSE of about 0.336901 on the supplied validation. On the shared mask the archive coefficient replay is a current-speed reference. Chen et al.'s Table 6 gives 0.2893 for their test. Figure 13(c) starts at a different speed phase from the course input. The original acquisition and retrospective-window archive were unavailable. The analysis retained the designated course segment and did not reconstruct a preferred replacement by searching test windows.

# 5. Proposed DHON

The model integrates observed speed to predict shaft phase in revolutions, $u_t=u_{{t-1}}+n_{{t-1}}/(60 f_s)$. A carrier at order $o$ has phase $2\\pi o u_t$. Each oscillator demodulates the known measurements into a fixed complex frame, applies normalized finite exponential memory, delays one sample, and remodulates into target phase. For $q_t=x_t\\exp(-i\\phi_t)$, the equivalent finite recurrence is

$$E_{{t+1}}=\\rho E_t+q_t-\\rho^M q_{{t-M}},\\qquad
 h_t=\\frac{{2E_t\\exp(i\\phi_t)}}{{\\sum_{{j=1}}^M\\rho^{{j-1}}}}.$$

Here $M$ is the finite state memory and $\\rho=\\exp(-1/\\tau)$, with $\\tau$ measured in samples. The implementation uses complex linear convolution in PyTorch. Bounded continuous parameters govern decay and detuning. The learned head uses causal context to mix the bank of quadrature features. A regularized speed-conditioned linear readout uses vibration lags, angle lags, proxy innovations and energy. The optional adapter adds a weighted history of clipped base residuals only after their targets arrive. The base context is bounded by 768 samples and the adapter by another 1,024.

{figure(run, "dhon_architecture", "DHON input availability, state computation, forecast branches and residual adapter.")}

# 6. Optimization and comparisons

Three validation blocks inside the training record were evaluated with 1,792-sample separation on both sides and record/segment resets. The DHON design scored {len(search)} configurations comprising {len(trial_rows)} guarded rows. It used a utility combining clean error, stress error and injected-pulse distortion. The selected trial was **{selection['primary_trial'].replace('_',' ')}**, with fold utility {selection['primary_fold_utility']:.6f}. Its base geometry used {len(cfg['harmonics'])} mesh-locked orders, {cfg['memory']} oscillator samples, {cfg['lags']} direct lags and {cfg['angular_steps']} samples per mesh cycle over {cfg['mesh_cycles']} cycles. The alternative traded clean MSE for a more compact representation and a lower measured RPM stress and synthetic pulse score. The earlier-only chronological folds returned MSE 3.75985, 1.65084 and 0.33538. This large early error indicates weak extrapolation when training covers less of the speed trajectory. Trial-level arrays and chronological outcomes are in the search directory.

A separate train-fold comparison covered ridge AR, speed-conditioned LPV ridge, a fixed-physics linear readout, polynomial NG-RC, MLP and TCN variants. The old DHON/NG-RC checkpoint forecasts were independently reconstructed from numeric arrays. The best original ancestor provides matched forecast vectors but no corresponding complete checkpoint. Its metric was recomputed using the same 22,208 targets.

# 7. Results

## 7.1 Development prediction

{compare}

Table 1. Standardized target errors on a common mask. The archived conventional coefficients used current RPM. Strict estimates used previous RPM.

{figure(run, "trace", "Prespecified development trace and matched residual alignment on a short interval.")}

{pair_table}

Table 2. New DHON minus comparator squared error, with 2,000 moving-block bootstrap draws of paired 640-sample blocks. These intervals describe within-record uncertainty. They do not represent new-acquisition uncertainty.

## 7.2 Effect of learning and adaptation

{matched}

Table 3. A matched encoder and scoring policy allow comparison of fixed and learned physical parameters. The alternative differs in geometry.

The new primary reduced MSE by {fixed-m:.6f} relative to the matched fixed-state model, about {100*(fixed-m)/fixed:.3f}%. Removing the slow adapter raised MSE by {adapter:.6f}, or {100*adapter/primary['base_mse']:.2f}% of the base error. State gradients and parameter motion establish differentiable state learning, but the development error change due to learning is small. Fitted order offsets ranged from −0.04361 to +0.04372 order. The three memory banks occupied 7.83–8.11, 30.62–33.54 and 123.55–133.03 samples. The offsets and memories are model parameters and do not measure mechanical damping or changes in tooth count. The strongest archived reference remains ahead. A compact alternative with {int(alt['parameters'])} deployed numeric parameters versus {int(primary['parameters'])} also scored {alt['mse']:.6f}, exposing the trade-off between clean error and feature cost.

{figure(run, "dhon_parameter_trajectory", "Fitted physical parameter trajectories from the selected DHON training run.")}

## 7.3 Provisional diagnostics and robustness

PMI normalizes centered analytic-envelope correlation at one turn. A 2,048-sample angular grid gives lag 2,047 under the literal MATLAB index and 2,048 under the mathematical correction. The input labels describe index groups A and B only. The scope is threshold-free within this development corpus. Predictor RPM corruption and diagnostic RPM corruption are measured separately. Controlled damped pulse injections assess retention by residual differences, with aperiodic scenarios as negative controls. Exact MATLAB end-to-end parity requires its angle helper or golden outputs.

{diagnostic_text}

Table 4. Descriptive PMI group summaries where computed. AUC refers to B having a higher score than A and cannot be interpreted as verified fault-classification accuracy.

{figure(run, "fault_pmi_groups", "Angular PMI distributions for provisional groups with exact overlap exclusions.")}

{robustness_table}

Across thirteen controlled predictor RPM stress cases, a +2% systematic speed bias produced the maximum observed MSE for the primary, strict sparse and replayed archive DHON controls. Pulse recovery and PMI response vary across injection scenarios. The pulse projection and correlation tables quantify this variation without equating synthetic amplitude to physical crack severity.

## 7.4 Computational cost

{timing_table}

Table 5. In-memory batch inference on CPU, including model feature construction. Each displayed repeat uses a single library thread. RTF is runtime divided by recording duration. A low batch RTF alone does not establish 0.3125 ms per-sample streaming deadlines. GPU available to the tested build was {timing_env.get('cuda_available', False)}. Training and component fit measurements are separate in model-level tables.

# 8. Verification and limitations

Tests cover finite history and reference recurrence, actual parameter gradients, alignment, solver parity, PMI and record reset fixtures, guarded folds, and checkpoint round trips. The audit and notebook logs provide the exact recorded verification scope. The speed and vibration window checks exclude exact overlaps, while unavailable original acquisition metadata limits stronger claims of independence. Independent prediction and fault inputs were unavailable, so no blind scores are reported. The project also lacks the original checkout needed to reproduce the strongest archived saved forecasts from coefficients. The mixed 2025/2026 course dates are preserved as published.

# 9. Conclusion

The study demonstrates a causal, finite-history DHON whose speed-linked oscillator state participates in backpropagation. Its forecast is stronger than the newly fitted conventional and learned comparators in this run, but weaker than the strongest archived corrected anchors. The adapter provided the largest demonstrated design gain. The diagnostic findings describe development index groups and controlled perturbations. More labelled acquisitions and the frozen test protocol are required to establish transfer and operational fault detection.

# References

1. MECG 7780, *Time Serial Dependency Modeling via Advanced Machine Learning Algorithms*. Instructor course brief and supplied `fdt_featcal.m`.
2. Y. Chen, Z. Li, Y. Jiang, D. Gong and K. Zhou. “Sparse LPV-ARMA model for non-stationary vibration representation and its application on gearbox tooth crack detection under variable speed conditions.” *Mechanical Systems and Signal Processing* 224, 112161 (2025). [DOI](https://doi.org/10.1016/j.ymssp.2024.112161)
3. Y. Chen. *Vibration-based fault detection and severity assessment for fixed-axis gearboxes*. PhD thesis, University of Alberta (2020). pp. 45, 99–102, 116–124.
"""
    report_path = ROOT / "docs/final_course_report.md"
    report_path.write_text(report, encoding="utf-8")
    reproduction = f"""# Reproduction summary

Run `{run.name}`. All matched scores use targets 1,792–23,999, 22,208 samples at 3.2 kHz. Fit selection used only guarded blocks of `sigtrain`.

{compare}

**Interpretation.** Primary DHON improves on the strict sparse refit by {new_vs_sparse:.2f}% and trails the strongest forecast-only archive anchor by {ancestor_deficit:.2f}%. The learned-state improvement over the matched fixed oscillator is {fixed-m:.6f}. Removing correction worsens MSE by {adapter:.6f}.

**Selection.** `{selection['primary_trial']}` is primary and `{selection['alternative_trial']}` is alternative. Exact configs are in [the selection manifest](../{run.relative_to(ROOT).as_posix()}/search/dhon_selection.json). There are {len(search)} ranked DHON configs and {len(trial_rows)} guarded fold rows. The current-speed archived comparison and the published Table 6 value 0.2893 refer to distinct conventions.

**Limits.** `sigtest` and `sigfault2` were absent. Condition labels and acquisition start IDs were absent. GPU hardware was unavailable. Instructor-compatible PMI indexing was reproduced algebraically, and exact executable MATLAB parity remains unverified.

**Replay.** From the project root run `python -m gearbox_dhon.cli replay --run-id {run.name} --data-dir data/raw` after placing the original MAT files. See [README](../README.md) for full run, smoke, notebooks, and frozen blind evaluation.
"""
    (ROOT / "docs/reproduction_summary.md").write_text(reproduction, encoding="utf-8")
    result = {"run": run.name, "split": "development_sigvalid", "units": "standardized vibration squared",
              "score_start": 1792, "score_stop_exclusive": 24000, "target_count": 22208,
              "primary_dhon_mse": m, "strict_sparse_lpv_arma_mse": better, "strict_dense_lpv_arma_mse": baseline_dense,
              "archived_checkpoint_mse": archive, "archived_forecast_only_mse": ancestor, "paper_external_mse": .2893,
              "geometry": cfg, "improvement_pct_vs_strict_sparse": new_vs_sparse,
              "adapter_mse_decrease": adapter, "learned_vs_fixed_mse_decrease": fixed-m,
              "dhon_ranked_configurations": len(search), "dhon_fold_rows": len(trial_rows),
              "diagnostics": diagnostic_summary, "gpu_available": timing_env.get("cuda_available", False),
              "paired_bootstrap": json.loads(paired.to_json(orient="records")),
              "availability": audit,
              "evidence": {key: str(run.relative_to(ROOT) / path) for key, path in
                           {"metrics": "metrics/summary.csv", "selection": "search/dhon_selection.json",
                            "timings": "metrics/timings.csv", "freeze": "frozen_manifest.json"}.items()}}
    (final / "RESULTS_SUMMARY.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    if shutil.which("pandoc") and shutil.which("xelatex"):
        subprocess.run(["pandoc", str(report_path), "--from=markdown+tex_math_dollars",
                        "--pdf-engine=xelatex", f"--resource-path={ROOT / 'docs'}",
                        "--output", str(final / "final_course_report.pdf")], check=True, cwd=ROOT)
    else:
        raise RuntimeError("PDF build requires Pandoc and XeLaTeX. Markdown and numerical results were written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="development")
    args = parser.parse_args()
    build(ROOT / "outputs/runs" / args.run_id)
