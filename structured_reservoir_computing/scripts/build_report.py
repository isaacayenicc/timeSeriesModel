"""Compose the SRC report and produce its editable DOCX and PDF review copy."""
from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile
from xml.etree import ElementTree as ET

from report_evidence import Evidence, ROOT, fmt, model_key, name, number

TITLE = "Physics-Informed Structured Reservoir Computing for Nonstationary Gearbox Vibration Modelling and Tooth Crack Detection"
REPORT = ROOT / "report"


def cell(value):
    return " ".join(str(value).split()).replace("|", "\\|")


def table(headers, records):
    items = [headers, ["---"] * len(headers), *records]
    return "\n".join("| " + " | ".join(cell(value) for value in row) + " |" for row in items)


def figure(e, label, description, *names):
    path = e.plot(*names)
    if path is None:
        return f"{label} evidence is unavailable in this release."
    return f"![{label}. {description}](../{path.relative_to(e.root).as_posix()})"


def diagram(output: Path, e):
    """Draw original scientific block figures for the report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    output.mkdir(parents=True, exist_ok=True)
    items = [
        ("workflow", [
            ("Course records", "Train · development\nCandidate acquisitions"),
            ("Guarded fitting", "Fit transforms and readouts\nSelect on training folds"),
            ("Matched forecast", "Frozen one-step inference\nCommon validation targets"),
            ("Assessment", "Residual PMI, stress\nand CPU cost"),
        ]),
        ("architecture", [
            ("Observed past", "Vibration and RPM\nthrough sample t−1"),
            ("Structured state", "Time and angular lags\nHarmonic and error states"),
            ("Learned readout", "Selected products\nGroup-scaled coefficient fit"),
            ("Causal output", "Base prediction +\noptional historical correction"),
        ]),
    ]
    for stem, blocks in items:
        fig, ax = plt.subplots(figsize=(10.2, 2.55), layout="constrained")
        ax.set(xlim=(0, 10.3), ylim=(0, 2.42))
        ax.axis("off")
        for i, (heading, body) in enumerate(blocks):
            x = 0.08 + 2.6 * i
            ax.add_patch(FancyBboxPatch((x, .50), 2.30, 1.45, boxstyle="round,pad=0.055,rounding_size=.045",
                                        linewidth=.8, facecolor="#e7f2f3" if i == 1 else "#f8fafb",
                                        edgecolor="#87a4b0"))
            ax.text(x + .12, 1.70, heading, va="top", fontsize=12, fontweight="bold", color="#0a5d5b")
            ax.text(x + .12, 1.26, body, va="top", fontsize=10.3, color="#21313e", linespacing=1.42)
            if i < 3:
                ax.add_patch(FancyArrowPatch((x + 2.36, 1.24), (x + 2.54, 1.24), arrowstyle="->",
                                             linewidth=1.25, mutation_scale=11, color="#086660"))
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{stem}.{extension}", dpi=350)
        plt.close(fig)
    trials = [row for row in e.csv("results/optimization/trials.csv") if row.get("status") == "DONE"]
    if trials:
        fig, ax = plt.subplots(figsize=(7.6, 4.15), layout="constrained")
        stages = sorted({row["stage"] for row in trials})
        for stage in stages:
            valid = [row for row in trials if row["stage"] == stage and number(row, "fold_mse") is not None]
            ax.scatter([number(row, "fold_stress_mse") for row in valid],
                       [number(row, "fold_mse") for row in valid], s=36, label=f"Stage {stage}", alpha=.85)
        chosen = e.selection.get("trial_id")
        selected = next((row for row in trials if row["trial_id"] == chosen), {})
        if selected:
            ax.scatter(number(selected, "fold_stress_mse"), number(selected, "fold_mse"),
                       s=138, facecolors="none", edgecolors="#153654", linewidth=1.6, label="Selected PI-SRC")
        ax.set(xlabel="Mean stressed training-fold MSE", ylabel="Mean clean training-fold MSE")
        ax.grid(alpha=.17)
        ax.legend(ncol=2, frameon=False, fontsize=9)
        for extension in ("png", "pdf"):
            fig.savefig(output / f"search_tradeoff.{extension}", dpi=350)
        plt.close(fig)


def model_table(e):
    rows = e.comparison
    return table(["Predictor", "MSE", "RMSE", "MAE", "Bias", "Targets"],
                 [(name(row), fmt(row, "mse"), fmt(row, "rmse"), fmt(row, "mae"),
                   fmt(row, "bias"), fmt(row, "n", digits=0)) for row in rows]) if rows else (
                 "The matched model table is unavailable at the time of this build.")


def timing_table(e):
    keys = {"primary-pi-src", "archived-src-corrected", "sparse-lpv-arma", "linear",
            "ridge", "ar", "cnn", "transformer", "tft", "svr"}
    return table(["Model", "Fit (s)", "CPU median (ms)", "IQR (ms)", "Parameters", "Bytes"],
                 [(name(r), fmt(r, "fit_s", digits=2), fmt(r, "warm_median_ms", digits=3),
                   fmt(r, "warm_iqr_ms", digits=3), fmt(r, "parameters", digits=0),
                   fmt(r, "checkpoint_bytes", digits=0)) for r in e.comparison if model_key(r) in keys]) if e.comparison else (
                 "A common timing table is unavailable at the time of this build.")


def split_table(e):
    rows = e.csv("results/record_overlap_audit.csv")
    status = e.json("results/data_inventory.json")
    evidence = "Read the per-record and split ledgers for exact identities and policy decisions."
    if rows and status:
        evidence = "The supplied source inventory and overlap audit retain both complete and excluded record identities."
    return table(("Input", "Role", "Status"), [
        ("sigtrain", "Fit, training-only exploration and guarded selection", "24,000 samples at 3,200 Hz"),
        ("sigvalid", "Matched development scoring after selection", "22,208 scored targets"),
        ("sigfault1", "Candidate diagnostics and integrity audit", evidence),
        ("sigtest, sigfault2", "Blind study following freeze", "Unavailable in the supplied release"),
    ])


def selected_table(e):
    selected = e.selection
    config = selected.get("config", {})
    if not isinstance(config, dict):
        config = {}
    def show(*keys):
        return " / ".join(str(config.get(k, "—")) for k in keys)
    return table(["Selected component", "Recorded value"],
                 [("Training trial", selected.get("trial_id", "Unavailable")),
                  ("Mean guarded MSE: clean, stressed", f"{fmt(selected, 'fold_mse')} / {fmt(selected, 'fold_stress_mse')}"),
                  ("Guarded pulse distortion", fmt(selected, "fold_pulse_distortion")),
                  ("Selection objective", fmt(selected, "utility")),
                  ("Effective raw support (samples)", fmt(selected, "history", digits=0)),
                  ("Fitted model coefficients / readout features", f"{fmt(selected, 'parameters', digits=0)} / {fmt(selected, 'feature_columns', digits=0)}"),
                  ("Training MSE on common mask", fmt(selected, "train_mse")),
                  ("Time lags / angular steps / mesh cycles", show("lags", "angular_steps", "mesh_cycles")),
                  ("Shaft orders / oscillator decay (samples)", show("harmonics", "taus")),
                  ("Oscillator memory / speed polynomial degree", show("memory", "speed_degree")),
                  ("Innovation order / lag count / clip", show("innovation_order", "innovation_lags", "innovation_clip")),
                  ("RMS and acceleration windows (samples)", show("energy_window", "acceleration_window")),
                  ("Polynomial degree / nonlinear feature cap", show("polynomial_degree", "nonlinear_cap")),
                  ("Solver / ridge weight", show("solver", "ridge")),
                  ("Adapter gain / decay / memory / clip", show("adapter_gain", "adapter_tau", "adapter_memory", "adapter_clip")),
                  ("Pulse preservation / augmentation weights", show("pulse_preservation", "augmentation"))])


def compact_fault_table(e):
    rows = e.fault
    if not rows:
        return "The candidate-record group table is unavailable."
    selected = [r for r in rows if not r.get("metric") or r.get("metric") in {"lecturer_compatible", "pmi_course", "pmi_literal", "PMI"}]
    rows = selected or rows
    return table(["Signal", "Median A", "Median B", "Gap B−A", "95% gap interval", "Conditional AUC"], [
        (name(r), fmt(r, "median_A", "group_a_median", digits=5),
         fmt(r, "median_B", "group_b_median", digits=5),
         fmt(r, "median_gap_B_minus_A", "median_gap", "pmi_median_gap", digits=5),
         f"{fmt(r, 'median_gap_ci_low', digits=4)} to {fmt(r, 'median_gap_ci_high', digits=4)}",
         fmt(r, "conditional_auc_B_high", "conditional_auc", "provisional_auc", "auc", digits=3))
        for r in rows[:14]])


def compact_stress_table(e):
    rows = e.stress
    if not rows:
        return "The stress summary table is unavailable."
    return table(["Model", "Clean MSE", "Worst tested MSE", "Worst tested RPM condition"], [
        (name(r), fmt(r, "clean_mse", "mse_clean"),
         fmt(r, "worst_stress_mse", "worst_mse", "rpm_mse_worst", "mse_worst"),
         r.get("worst_condition", "—").replace("bias_plus_2pct", "+2% bias").replace("bias_minus_2pct", "−2% bias").replace("_", " "))
        for r in rows[:14]])


def compact_ablation(e):
    rows = e.ablations
    if not rows:
        return "The matched ablation table is unavailable."
    keys = {
        "primary-pi-src", "pi-src-pure-mse", "pi-src-compact", "src-linear", "src-polynomial",
        "src-speed", "src-physics-linear", "pi-src-fixed", "src-generic", "pi-src-adapter-off",
        "pi-src-no-angle", "pi-src-no-carrier", "pi-src-no-innovation", "pi-src-no-augmentation",
        "pi-src-degree2", "pi-src-order12", "pi-src-fixed-hz-only",
    }
    return table(["Control", "Development MSE", "Training MSE", "Coefficients"], [
        (name(r), fmt(r, "mse", "validation_mse", "clean_mse"),
         fmt(r, "train_mse"), fmt(r, "parameters", digits=0))
        for r in rows if model_key(r) in keys])


def matched_adapter_table(e):
    rows = e.csv("results/ablations/identical_adapter_controls.csv")
    return table(["Predictor", "Base MSE", "Corrected MSE", "MSE decrease"], [
        (name(r), fmt(r, "uncorrected_mse"), fmt(r, "matched_correction_mse"), fmt(r, "gain_mse"))
        for r in rows])


def external_table(e):
    paper = e.csv("results/metrics/published_paper_reference.csv")
    local = e.csv("results/metrics/local_statistical_reference.csv")
    labels = {"ari": "ARI", "arima": "ARIMA", "spa_lpv_ar": "Sparse LPV–AR", "spa_lpv_arma": "Sparse LPV–ARMA"}
    rows = []
    for r in paper:
        key = r.get("model")
        strict = next((q for q in local if q.get("model") == key and q.get("convention") == "strict_development"), {})
        archived = next((q for q in local if q.get("model") == key
                         and q.get("convention") == "preserved_archive_current_rpm"), {})
        rows.append((labels.get(key, key), fmt(r, "authors_test_segment_mse", digits=4),
                     fmt(strict, "development_mse"), fmt(archived, "development_mse")))
    return table(["Model", "Paper test segment", "Strict development", "Archive reproduction"], rows)


def best_result(e):
    rows = sorted(e.measured, key=lambda row: number(row, "mse"))
    if not rows:
        return "Primary and comparator measurements remain unavailable in the saved result table."
    primary = number(e.primary, "mse")
    text = e.comparison_text()
    archive = number(e.archive_reference, "mse")
    if primary is not None and archive is not None:
        change = 100 * (primary - archive) / archive
        direction = "higher" if change >= 0 else "lower"
        text += f" The selected model scored {abs(change):.2f}% {direction} than the strongest replayed SRC anchor ({archive:.6f})."
    best = next((row for row in rows if row.get("scope") == "replayed comparator"), {})
    if best:
        text += f" The lowest requested comparator scored {fmt(best, 'mse')} ({name(best)})."
    return text


def ablation_interpretation(e):
    primary = number(e.primary, "mse")
    no_adapter = next((r for r in e.ablations if model_key(r) == "pi-src-adapter-off"), {})
    if not no_adapter:
        no_adapter = next((r for r in e.ablations if "primary-base" in model_key(r)
                           or "without-adapter" in model_key(r)), {})
    no_score = number(no_adapter, "mse")
    if primary is None or no_score is None:
        return "Use the reported same-readout control to assess the incremental correction effect."
    difference = no_score - primary
    return (f"Using the same fitted weights, the no-adapter MSE was {no_score:.6f}. "
            f"Applying the selected correction changed MSE by {-difference:+.6f}.")


def baselines_evidence(e):
    controls = e.csv("results/metrics/vibration_rpm_parity.csv")
    repeats = e.csv("results/metrics/baseline_seed_sensitivity_summary.csv")
    if not controls or not repeats:
        return ""
    changes = [(r, number(r, "delta_measured_minus_vibration_only")) for r in controls]
    better = sum(x is not None and x < 0 for _, x in changes)
    largest = min((row for row, delta in changes if delta is not None),
                  key=lambda r: number(r, "delta_measured_minus_vibration_only"))
    best = min(repeats, key=lambda r: number(r, "mean_mse") or float("inf"))
    return (f"A fixed-architecture refit with speed held at 145 RPM was compared across {len(controls)} comparator "
            f"families. Measured RPM improved development MSE in {better} instances. The strongest improvement "
            f"was for {name(largest).upper()} ({fmt(largest, 'delta_measured_minus_vibration_only', digits=5)} MSE). "
            f"Across the three-fit seed sensitivity checks, {name(best).upper()} had the lowest mean MSE "
            f"({fmt(best, 'mean_mse', digits=5)}, sample SD {fmt(best, 'sd_mse', digits=5)}). These limited controls "
            "describe specific replayed architectures and one development acquisition.")


def regime_evidence(e):
    strata = [r for r in e.csv("results/metrics/speed_and_time_strata.csv")
              if model_key(r) == model_key(e.primary) and r.get("group", "").startswith("rpm_quartile_")]
    chronological = e.csv("results/ablations/earlier_only_sensitivity.csv")
    if not strata or not chronological:
        return ""
    chronological_mse = [fmt(r, "mse", digits=4) for r in chronological]
    return (f"Across ascending measured-RPM quartiles, PI-SRC MSE was {', '.join(fmt(r, 'mse', digits=4) for r in strata)}. "
            f"On the three guarded training holdouts, earlier-only refits produced MSE {', '.join(chronological_mse)}. "
            "Those scores show sensitivity to fitting record length and operating-regime coverage.")


def paired_table(e):
    rows = e.csv("results/statistical_tests/paired_moving_block.csv")
    rows = [r for r in rows if number(r, "block_samples") == 640]
    keys = {model_key(row): name(row) for row in e.comparison_top}
    values = [(keys[model_key(r)], fmt(r, "delta_mse"),
               fmt(r, "interval_low"), fmt(r, "interval_high"))
              for r in rows if model_key(r) in keys]
    return table(["Model", "MSE minus PI-SRC", "Lower 95%", "Upper 95%"], values) if values else (
        "The paired block-interval artifact is unavailable.")


def representation_evidence(e):
    rows = {model_key(r): r for r in e.ablations}
    data = []
    clean = next((r for key, r in rows.items() if "pure" in key and "mse" in key), {})
    scarce = next((r for key, r in rows.items() if key in {"compact-pi-src","pi-src-compact"}
                   or ("compact" in key and "src" in key)), {})
    alternative = number(clean, "mse")
    compact = number(scarce, "mse")
    if alternative is not None:
        data.append(f"The configuration chosen by clean guarded MSE scored {alternative:.6f} on development.")
    if compact is not None:
        data.append(f"The compact fit scored {compact:.6f} with {fmt(scarce, 'parameters', digits=0)} coefficients.")
    for tests in [
        (("no-angle", "no-angular"), "Disabling angular history"),
        (("no-proxy", "no-innovation"), "Disabling proxy innovations"),
        (("12-order", "12-orders", "order12", "full-order", "all-orders", "wide-order"), "The larger order-bank sensitivity"),
    ]:
        sk, detail = tests
        row = next((r for key, r in rows.items() if any(token in key for token in sk)), {})
        v = number(row, "mse")
        if v is not None:
            data.append(f"{detail} scored {v:.6f} on development.")
    if data:
        data.append("The reported feature sensitivities retain their original analysis scope. A favorable development ablation provides a hypothesis for confirmation and leaves the frozen training-only primary choice intact.")
        return " ".join(data)
    return "Measured variants and their scoring scopes are retained in the ablation ledger."



def recorded_timing(e):
    median = number(e.primary, "warm_median_ms")
    if median is None:
        return "A primary timing figure was unavailable at the time of this build."
    base = number(e.sparse_reference, "warm_median_ms")
    detail = (f"The selected PI-SRC required {median:.2f} ms median for one 24,000-sample CPU batch "
              f"({24000/(median/1000):,.0f} samples/s or {median/7500:.3f} times the record duration).")
    if base is not None:
        detail += f" The local sparse LPV–ARMA median was {base:.2f} ms under its recorded benchmark."
    return detail


def fault_interpretation(e):
    row = next((row for row in e.fault if model_key(row) == model_key(e.primary)
                and row.get("metric") == "lecturer_compatible"), {})
    auc = number(row, "conditional_auc_B_high")
    gap = number(row, "median_gap_B_minus_A")
    if auc is None or gap is None:
        return "The candidate-record tables preserve the conditional scope of reporting."
    return (f"The primary PI-SRC median PMI gap B−A was {gap:.5f}, with a bootstrap interval from "
            f"{fmt(row, 'median_gap_ci_low', digits=5)} to {fmt(row, 'median_gap_ci_high', digits=5)}. "
            f"Pairwise B-high ordering was {auc:.3f} within the audited index groups. The minimum-B minus "
            f"maximum-A margin was {fmt(row, 'margin_min_B_minus_max_A', digits=5)}. Several methods tied on "
            "conditional AUC, and gaps and margins reveal differences concealed by that statistic.")


def robustness_evidence(e):
    stress = next((r for r in e.stress if model_key(r) == model_key(e.primary)), {})
    pulse = next((r for r in e.pulses if model_key(r) == model_key(e.primary)
                  and r.get("kind") == "periodic"), {})
    pieces = []
    clean, worst = number(stress, "clean_mse"), number(stress, "worst_stress_mse")
    if clean is not None and worst is not None:
        pieces.append(f"Selected PI-SRC clean MSE {clean:.6f} increased to {worst:.6f} at the worst tested predictor-RPM condition.")
    correlation = number(pulse, "median_pulse_correlation")
    if correlation is not None:
        count = fmt(pulse, "positive_delta_count", digits=0)
        total = fmt(pulse, "cases", digits=0)
        pieces.append(f"Residual PMI rose in {count} of {total} periodic injections. Median paired residual–pulse correlation was {correlation:.3f}.")
        retained = number(pulse, "median_energy_retention")
        if retained is not None and retained > 1:
            pieces.append(f"Median residual change energy was {retained:.2f} times injected energy, consistent with waveform distortion as well as retained pulse energy.")
    layer = e.csv("results/fault/rpm_stress.csv", "results/fault/src_live/rpm_stress.csv")
    nuisance = next((r for r in layer if model_key(r) == model_key(e.primary)
                     and r.get("condition") == "bias_plus_2pct" and r.get("scope") == "predictor"), {})
    delta = number(nuisance, "delta_pmi", "pmi_delta")
    if delta is not None:
        pieces.append(f"The same +2% predictor-only RPM interference changed validation PMI by {delta:+.5f}. The controlled sensor effect should be considered before assigning this score a condition threshold.")
    return " ".join(pieces) or "Stress and pulse results are reported in their saved evidence scopes."



def trials_evidence(e):
    items = e.csv("results/optimization/trials.csv")
    folds = e.csv("results/optimization/fold_scores.csv")
    if not items:
        return "The trial ledger was unavailable at the time of this build."
    done = sum(row.get("status") == "DONE" for row in items)
    return f"The released ledger records {done} completed candidates of {len(items)} attempted and {len(folds)} fold rows."


def causality_evidence(e):
    data = e.json("results/manifests/src_prediction_verification.json")
    tests = data.get("tests", [])
    if not tests:
        return ""
    maximum = max(number(row, "max_through_target_change") or 0 for row in tests)
    return (f"Across {len(tests)} frozen SRC checkpoint tests, perturbing vibration and RPM at and after "
            f"a target changed the forecasts through that target by at most {maximum:.2e} standardized units. "
            "The test evaluates realized numerical causality in addition to the finite-support construction.")



def report(e):
    txt = [
        """---
lang: en-US
---

# COURSE-TITLE-MARKER

MECG 7780 / ENGINEERING DATA ANALYTICS

""" + TITLE + """

Causal one-step prediction, matched benchmarks and residual diagnostics

Final course project report

Submission package includes the editable report, the validated implementation and the corresponding numerical evidence.

Project release / November 2026

WORD_PAGE_BREAK

# Abstract

Variable shaft speed transports mesh-related vibration across frequencies and changes the observed waveform. This study designed physics-informed structured reservoir computing (PI-SRC) to predict the next standardized vibration sample from measurements through the preceding time step. The reservoir combines causal delay coordinates, phase-transported harmonic quadratures, speed context and past innovations. Its regularized readout is fitted after each candidate representation is constructed. Physical settings and the optional residual adapter are chosen under a guarded training-only protocol. The 24,000-sample official development record supplies 22,208 common targets for the reported comparisons.

""" + best_result(e) + """ """ + recorded_timing(e) + """ The study compares conditional residual periodicity across 198 provisionally grouped candidate acquisitions and tests RPM interventions and injected-pulse preservation. The absence of verified condition labels and independent test files limits diagnostic and generalization claims. The release preserves fitted configurations, aligned results and a frozen test interface.

**Keywords**  structured reservoir computing, nonstationary vibration, shaft order, causal prediction, sparse LPV–ARMA, periodic modulation intensity

WORD_PAGE_BREAK

# Contents

WORD_TOC_MARKER

# 1. Introduction

Under a variable speed trajectory, an otherwise stable gear mesh event moves through the frequency axis. A forecasting rule can follow this effect by modelling operating conditions more directly than a stationary filter. The practical task is to improve short-horizon prediction of the supplied gearbox vibration and retain residual structure that may support tooth-crack investigation. The course evaluates accuracy by mean squared error (MSE), computational cost by CPU/GPU time, and diagnostic performance with angular periodicity evidence [1].

The first hypothesis is that order-aware history improves the prediction problem in which the next sample is estimated from measured history. The second is that structured interactions and training-only selection reduce error under a fair mask. The third is that residual correction and modelling error reduction may have distinct magnitudes and diagnostic consequences. Each hypothesis is checked with a measured comparison or a bounded ablation. Results from a published segment and previously fitted archive models are attributed to their own evidence scopes.

The study contributes a finite sensor-driven reservoir, a joint feature and regularization search, a matched comparison across statistical and machine learning families, and a diagnostic investigation that preserves acquisition identity. Figure 1.1 summarizes the separation among fitting, scored forecasting and diagnosis.""",
        figure(e, "Figure 1.1", "Project workflow. Candidate data enter post-fit assessment and share neither selection nor model fitting.",
               "report/workflow.png"),
        """# 2. Physical and methodological background

Gear teeth come into contact every fraction of a shaft revolution. For a monitored $z=37$ tooth gear and shaft speed $n_t$ in revolutions per minute, the nominal mesh frequency is $f_m(t)=z n_t/60$ Hz. Mesh harmonics shift with speed in frequency units and remain fixed multiples of the reference shaft order. Amplitude modulation, transmission through the housing, changes in load, and superposed resonance or noise can change observed power without changing the gear ratio. Such mechanisms motivate a feature library with separately testable speed-locked and fixed-Hz groups.

A sparse linear parameter-varying autoregression lets coefficients change with the measured scheduling signal. A sparse LPV–ARMA additionally describes past error contributions through moving-average terms. Chen et al. use a two-stage identification procedure and evaluate fault indicators on vibration residuals [2]. The first-stage autoregressive residual provides a proxy for unresolved dynamic effects and enters the forecast only at positive lag. The present experiments preserve the distinction among a fresh fit on supplied data, an archive checkpoint replay and the publication’s Table 6.

Reservoir computing separates the map of observed history from the fitted readout [3]. The present model uses an engineered, finite state driven by observed signals and a constrained feature library. Polynomial interactions and linear readout can exploit nonlinear observables. The finite oscillator states implement a physical representation whose candidate frequency and memory settings are tuned in the training folds and frozen at inference. This estimator has no recurrent learned hidden-state update. Related order-tracking and envelope diagnostics provide a way to compare residual periodicity when speed changes [4].

# 3. Experimental data and integrity

The official training and development traces contain 24,000 samples each at 3,200 Hz. The predictor uses supplied standardized vibration and shaft speed. Feature scaling is fitted on permitted training data without per-record re-standardization. The 200 candidate acquisitions were checked for exact matches across vibration and speed. Records 61 and 63 matched the official records and were excluded from A/B summaries. This leaves 198 eligible candidates with provisional index labels. Clip origins and authoritative condition labels remain unavailable.

Table 3.1. Data roles and evaluation status.""",
        split_table(e),
        figure(e, "Figure 3.1", "Representative training vibration and rotational speed. Signal and RPM axes represent their own units.",
               "eda/eda_speed.png", "eda/vibration_speed.png"),
        """# 4. Physics-informed exploratory evidence

Training spectrograms show how speed-locked candidates trace stationary orders while frequency varies. The local ridge near order 148 is close to the fourth mesh harmonic over changing RPM. The independently regenerated EDA measures median absolute ridge discrepancy 0.744 Hz. Its angle-domain peak is about 160 times the local median. Orders 148 and 259 therefore provide directly testable carrier candidates. Among the training diagnostics, the 1.0–1.2 kHz band contains 31.3% of Welch power. A broad concentration in hertz can include crossing harmonics and distributed excitation. It does not establish that a single feature at the center is mechanically resonant.

Figure 4.1 shows the frequency-domain ridge evidence. Figure 4.2 checks whether rotation-consistent components align more clearly in the angle representation. Both figures are calculated from the actual training record [5]. Their preprocessing is diagnostic. It does not introduce complete-record transforms as predictor features.""",
        figure(e, "Figure 4.1", "Training spectrogram with estimated mesh trajectories derived from shaft speed.",
               "eda/eda_spectrogram.png", "eda/spectrogram.png"),
        figure(e, "Figure 4.2", "Training power spectra in elapsed frequency and shaft order, respectively.",
               "eda/eda_orders.png", "eda/order_spectrum.png"),
        """The independently calculated guarded screen quantified the marginal contribution of state choices before the full model fit. LPV–AR24 averaged 0.46153 in that screen. Two order states at 148 and 259 with 128 samples of decay averaged 0.44049. A six-order bank with 32-sample decay averaged 0.43537. A sideband and shaft-aware candidate with orders 148 and 259 averaged 0.42903 in its own feature screen. Two fixed-Hz states at 1080 and 1125 Hz produced mean MSE 0.46152. Selective quadratic and cubic designs gave 0.43819 and 0.43803 under the separate nonlinear screen. Together these results motivate order-conscious states and explicit checks of sideband and nonlinear value under matched pulse and validation protocols. Screening scores on training folds are distinct from the full search and from official development scores.""",
        figure(e, "Figure 4.3", "Actual guarded training feature-screen comparison.",
               "eda/eda_feature_decisions.png"),
        r"""# 5. Physics-Informed Structured Reservoir Computing

## 5.1. Information set and physical state

Let $x_t$ denote standardized vibration, $n_t$ shaft speed in RPM and $f_s$ the sample frequency. The forecast information set is $\mathcal{F}_{t-1}=\sigma(x_s,n_s:s\leq t-1)$. Any finite time-delay bank stores selected $x_{t-j}$ with $j\geq1$. Causal state transformations depend on this information set. For observations, trapezoidal integration updates revolutions using the current and previous available RPM. The target phase advances using the preceding RPM:

$$
u_k=u_{k-1}+\frac{n_k+n_{k-1}}{120f_s},
\qquad
\widehat{u}_t=u_{t-1}+\frac{n_{t-1}}{60f_s}.
$$

An equal-angle feature locates a previous angular offset within known history and interpolates between the available surrounding samples. A delay without valid causal bracketing is caught by the support and evaluation policy. Backward acceleration and causal RMS estimate local speed and vibration context. These channels use measured past inputs and training-only scaling.

## 5.2. Damped harmonic quadratures

On a speed-locked carrier $q$ in cycles per input revolution, use $\phi_{q,k}=2\pi q u_k$. For a fixed-Hz carrier $f$, use $\phi_{f,k}=2\pi f k/f_s$. The implementation represents a finite weighted complex quadrature using real coordinates. Decay $\tau>0$ determines the weight $\rho=\exp(-1/\tau)$ for memory $M$. The state is

$$
h_{q,t}=\frac{2}{Z}\sum_{j=1}^{M}\rho^{j-1}x_{t-j}
\exp\!\left(\mathrm{i}[\widehat{\phi}_{q,t}-\phi_{q,t-j}]\right),
\qquad
Z=\sum_{j=1}^{M}\rho^{j-1}.
$$

The efficient implementation demodulates the input $a_k=x_k\exp(-\mathrm{i}\phi_{q,k})$. Its recurrence is $A_k=a_k-\rho^M a_{k-M}+\rho A_{k-1}$. The outgoing term removes contributions older than $M$ samples. The predicted state remodulates $A_{t-1}$ at the extrapolated phase and normalizes as in Equation (5.2). Separate real and imaginary components preserve phase information in a finite state. Multiple memories encode different persistence. The nonnegative normalized weights imply $|h_{q,t}|\leq 2\max_{1\leq j\leq M}|x_{t-j}|$. This bound describes the feature under measured historical input. Open-loop rollout would require a separate stability analysis. The phases depend on relative measured rotation and are reset for every acquisition. A speed perturbation changes the transported phase and weighted mixture, so the model is tested under RPM bias, delay and other recorded sensor interventions.

## 5.3. Innovation and structured readout

A fitted stage-one LPV–AR produces a causal proxy error. The error becomes available after the observation arrives and can enter a later reservoir state through lagged values. This creates a broader response family than vibration samples alone. It also requires a common guard that covers both stage-one and stage-two history.

The feature vector $z_t$ concatenates the enabled time, angle, speed, harmonic and innovation groups. The nonlinear library $\Phi(z_t)$ may include a restricted set of selected interactions. Group-scaled regularization accounts for differences in feature scaling or dimensionality. Feature means, scales and coefficients are fitted on the allowed blocks. A regularized objective has the form

$$
\min_{\beta,b}\frac{1}{N_{\mathrm{fit}}}\sum_{t\in\mathcal{I}_{\mathrm{fit}}}
L\!\left(x_t-b-\beta^\mathsf{T}\Phi(z_t)\right)
+\lambda\sum_g w_g\left\|\beta_g\right\|_2^2 ,
$$

where $L$ denotes squared or robust loss for the chosen candidate and $w_g$ is the group weight. Changing an allowed physics group changes the feature matrix and therefore requires a fit. Zeroing coefficients after the fit defines a different perturbation. Figure 5.1 shows the separation between state computation, readout fitting and the optional correction.""",
        figure(e, "Figure 5.1", "Finite causal carrier and lag groups feed a structured library and regularized readout.",
               "report/architecture.png"),
        r"""## 5.4. Historical residual correction

The base estimate is $b+\beta^\mathsf{T}\Phi(z_t)$. A clipped, exponentially weighted mean of earlier base errors can correct slowly varying signed bias. Let the past base residual be $r_{t-j}=x_{t-j}-\widehat{x}^{\,0}_{t-j}$. With clipping limit $c$, memory $M_c$, decay $\rho_c$ and gain $g_c$,

$$
\widehat{x}_t=\widehat{x}^{\,0}_t+
g_c\frac{\sum_{j=1}^{M_c}\rho_c^{j-1}\operatorname{clip}(r_{t-j},-c,c)}
{\sum_{j=1}^{M_c}\rho_c^{j-1}}.
$$

This expression makes the causal update explicit. Subtracting the model can remove predictive energy and can also change diagnostically relevant rotational modulation. Matching the correction across eligible alternatives and comparing an adapter-off build separates effects of past-error dynamics from representation and readout effects. A correction is judged jointly on prediction, stress and pulse behaviour.

# 6. Optimization and validation methodology

Three guarded contiguous blocks of the designated training trace provide the selection evidence. Held-out intervals begin at 4,800, 11,200 and 18,400 with 2,400 targets each. The recipe uses 1,792-sample raw-history guards, fits only permitted slices and preserves reset boundaries in the fitted features. If an innovation looks back $L$ samples and the readout uses it a further lag $j$ back, their combined maximum delay is $L+j$. The guard covers this composition and any additional adapter history. Every fold refits transformations and readouts from eligible samples.

Training selection uses clean error $E_{\mathrm{clean}}$, the excess sensor-stress error relative to clean, paired synthetic pulse distortion $D$, and a penalty for the coefficient count $P$:

$$
J=E_{\mathrm{clean}}
+0.14\max(0,E_{\mathrm{stress}}-E_{\mathrm{clean}})
+0.009D+0.000002P.
$$

The finite search compares candidate feature groups, carrier and delay settings, penalty strengths, and adapter parameters. The selected configuration depends on these bounds and the diagnostic proxy. Published MSEs and the provisional candidate labels do not enter selection. The best clean, compact and no-adapter variants use the distinct criteria documented in the registry. Each selected configuration is refit on official training and scored on development.

Table 6.1. Selected configuration and training-fold objective. Values are read from the saved selection record.""",
        selected_table(e),
        trials_evidence(e),
        causality_evidence(e),
        figure(e, "Figure 6.1", "Logged optimization and selection evidence from allowed training folds.",
               "report/search_tradeoff.png", "optimization/search_tradeoff.png", "optimization/src_pareto.png"),
        r"""Predictive errors are paired by actual target index. Paired moving-block estimates are appropriate because adjacent errors are temporally dependent [6]. A block interval on one acquisition quantifies within-segment differences under the chosen resampling rule. It does not measure the distribution across separately acquired machines or experiments. A complete study of cross-machine uncertainty would sample acquisition units at that level.

# 7. Comparator methodology

Classical comparators isolate autoregressive dependence in different ways. AR uses past vibration. ARX adds causal exogenous speed terms. ARMA models serial innovations, and ARIMA uses a differencing transform appropriate to its candidate definition. Sparse LPV–AR and sparse LPV–ARMA extend these models with speed scheduling and sparse support [2]. Fit scope and proxy-innovation implementation affect reproduction. A lower published value on the original paper’s evaluation window cannot resolve the difference in the supplied segment.

Linear Regression, Ridge, Lasso and Elastic Net fit lagged or physics-feature spaces with different shrinkage assumptions [7], [8]. SVR replaces a fixed linear predictor with selected kernel similarities. Random Forest and Gradient Boosting use ensembles of partitions and can express interactions carried by the feature map [9], [10]. NARX uses an exogenous nonlinear lag representation. Training controls include bounded fit and search costs and require a saved configuration and the matched evaluation mask.

MLP, LSTM, GRU, CNN and TCN represent the same finite measured inputs with different nonlinear operations. LSTM and GRU update recurrent hidden summaries on a causal sequence. CNN and TCN use local shared filters, with the TCN using dilated causal operators. The Transformer and Temporal Fusion Transformer introduce attention or variable-selection and gating under their implemented data and compute limits [11]–[14]. Chapter 8 reports all eligible rows. The comparison uses verified checkpoints from the earlier expanded study, applied to the newly audited course inputs and scored again on the same target array. The supervised models were originally fitted using protected training evidence. This run performs the new PI-SRC fit and comparator replay rather than fresh fold searches for all 21 families. Conventional support refits and external paper results are documented separately by their scheduling convention.

""" + baselines_evidence(e) + r"""

# 8. Matched results and critical discussion

## 8.1. Prediction performance

On the supplied official development record use targets with indices from 1,792 inclusive to 24,000 exclusive. Samples in the prefix serve initialization and history. MSE is $N^{-1}\sum_{t}(x_t-\widehat{x}_t)^2$ in standardized vibration squared. RMSE, MAE and signed residual bias describe scale and direction. The baseline and candidate MSE column is calculated on one identical target array.

""" + best_result(e) + """

Table 8.1. Common development prediction metrics. The archived publication’s test result is external to this table.""",
        model_table(e),
        figure(e, "Figure 8.1", "Measured and predicted waveform with aligned development targets.",
               "comparison/validation_trace.png", "comparison/matched_waveforms.png", "comparison/matched_trace.png"),
        """The PI-SRC estimate should be interpreted with the paired intervals and with the stronger archived checkpoint, which has a different stored implementation and fit provenance. A global estimate may average different operating regimes. Error summaries by speed or time block indicate whether a point improvement is concentrated near part of the speed profile. Residual structure indicates remaining systematic effects, even where pointwise error is lower. The waveform and diagnostic figures use the same aligned target indices. The recorded comparison also retains checkpoints and inference runs, so numerical recomputation is distinct from the interpretation of curves.

The publication reports 0.2893 in Table 6 on its own segment [2]. The official record provided for the course has a different speed phase and a distinct target mask. Table 8.2 distinguishes that external benchmark from reproductions with previous-RPM scheduling and with archive current-RPM scheduling. The previous-RPM filter follows the information limit used in the PI-SRC comparison. Archive reproduction also uses its original scoring start at sample 316.

Table 8.2. External paper and local statistical reference MSE under three distinct scopes.""",
        external_table(e),
        """Table 8.3. Selected 95% paired moving-block intervals. Differences are comparator minus PI-SRC MSE using 640-sample blocks.""",
        paired_table(e),
        figure(e, "Figure 8.2", "Errors by observed speed quartile over the same development mask.",
               "comparison/rpm_quartiles.png"),
        regime_evidence(e),
        """## 8.2. Computational efficiency

""" + recorded_timing(e) + """ Benchmarks report measured CPU batch inference on the full scored record under the recorded thread conditions. Fit cost, checkpoint bytes and fitted parameter count measure different quantities. Warm CPU medians and interquartile ranges characterize repeated in-memory calls and include feature production where the benchmark defines this scope. GPU and streaming measurements require observed hardware and deployment workloads.""",
        figure(e, "Figure 8.3", "Development MSE and measured CPU inference cost for eligible common-batch methods.",
               "comparison/accuracy_cost.png", "comparison/cost_accuracy.png", "comparison/mse_latency.png"),
        """Table 8.4. Representative fit, batch inference and model size. The complete ledger is in results/timings/benchmark_table.csv. An em dash marks an unavailable measurement.""",
        timing_table(e),
        r"""# 9. Fault diagnosis, robustness and ablation

## 9.1. Candidate-record periodic modulation

Subtract the prediction from the observation using the same warm-up for each matched method. Resample the eligible vibration or residual onto a uniform revolution grid inferred from measured speed. Use the centered analytic envelope $b_i$ to calculate correlation and PMI at rotation lag $T$.

$$
r(T)=\frac{\sum_{i=1}^{N-T}b_i b_{i+T}}{\sum_{i=1}^{N}b_i^2},
\qquad
\mathrm{PMI}=\frac{r(T)}{1-r(T)}.
$$

The instructor MATLAB reference uses first-position zero lag when indexing its autocorrelation sequence. The literal convention is one sample below the exact one-turn lag, and an alternate mathematical comparison preserves this distinction. Exact end-to-end MATLAB parity requires the missing order-tracking helper and gold outputs. Degenerate envelopes and insufficient rotations must be exposed by the audit. Groups A and B are provisional index groups; an AUC calculated by ranking B above A estimates index-group separation rather than verified crack classification [1], [4].

Table 9.1. Candidate-record PMI for provisional A/B index groups.""",
        compact_fault_table(e),
        fault_interpretation(e),
        figure(e, "Figure 9.1", "Candidate-record PMI by provisional index group. Eligibility and warm-up are matched.",
               "fault/fault_group_distribution.png"),
        figure(e, "Figure 9.2", "Representative envelope/order evidence from measured candidate vibration and residuals.",
               "fault/representative_order.png"),
        """## 9.2. Speed interventions and pulse retention

The stress study treats predictor RPM and angular diagnostic speed as separate interventions. A distorted predictor RPM shifts its constructed phase and may affect MSE. A distorted diagnostic RPM changes the angular resampling even for the same residual. Worse diagnostic stress can arise without a change in the prediction. Both perturbations are evaluated against their own clean reference. Near realistic operating bounds, an angular feature must retain causal support and stable units.

The pulse experiment injects a known disturbance into an otherwise unchanged record and compares the two residuals. The relevant questions are how much pulse energy remains, whether the recovered waveform correlates with the injected pulse, and whether its periodic envelope score changes. Retention and clean MSE can move in different directions. Synthetic results assess the specified waveform and injection model. A real tooth-damage claim requires physical condition metadata and independent records.

Table 9.2. Saved speed-stress measurements by method and scope.""",
        compact_stress_table(e),
        robustness_evidence(e),
        figure(e, "Figure 9.3", "Measured response to prescribed perturbations of rotational speed.",
               "fault/sensor_stress.png"),
        figure(e, "Figure 9.4", "Paired synthetic pulse-preservation evidence and diagnostic trade-offs.",
               "fault/pulse_tradeoff.png"),
        """## 9.3. Matched model ablations

Reselect or refit an ablation within its declared scope. Angular lags, speed context, carrier groups, nonlinear interactions, innovation history and adapter effects have different mechanisms. Compare matched subsets and retain fit and correction policies so an observed change has a testable interpretation. Higher accuracy alone does not determine whether marginal data support the extra cost of the added state.

Table 9.3. Representative SRC controls and development sensitivities. The full matrix is in results/ablations/src_ablations.csv.""",
        compact_ablation(e),
        ablation_interpretation(e),
        representation_evidence(e),
        """Table 9.4. Same-weight comparisons with the selected correction applied identically within each row.""",
        matched_adapter_table(e),
        """# 10. Conclusions

""" + best_result(e) + """ The project establishes an auditable finite reservoir that uses measured history and tests interpretable feature groups in guarded folds. Viable engineering use depends on the measured balance among error, inference cost and fault-sensitive residual information.

Table 10.1. Research hypothesis and qualified outcome.""",
        table(["Research hypothesis", "Evidence", "Supported conclusion"], [
            ("A shaft-angle representation stabilizes some variable-speed spectral components",
             "Training spectrogram, angle spectrum and guarded feature screen",
             "Order-locked peaks and selected guarded gains support order-aware testing."),
            ("A structured reservoir can compete with matched sparse and learned predictors",
             "Common-target errors, guarded search and the model registry",
             e.comparison_text()),
            ("A lower forecast error is sufficient for diagnosis",
             "Candidate PMI, perturbations and paired pulse preservation",
             "Predictive and residual diagnostic outcomes need separate evidence."),
            ("The release establishes independent condition classification",
             "Data availability and exact overlap audit",
             "Blind records, verified labels and acquisition metadata remain unavailable."),
        ]),
        """Independent evaluation uses a frozen manifest after eligible records are supplied. Batch timing is reproducible only relative to the recorded runtime and hardware. The results guide additional experimental design, including condition-confirmed recordings across speed and load regimes.

# References

[1] MECG 7780 Engineering Data Analytic, “Course project: Time serial dependency modeling via advanced machine learning algorithms,” course handout and MATLAB reference script.

[2] Y. Chen, Z. Li, Y. Jiang, D. Gong, and K. Zhou, “Sparse LPV-ARMA model for non-stationary vibration representation and its application on gearbox tooth crack detection under variable speed conditions,” *Mechanical Systems and Signal Processing*, vol. 224, art. 112161, 2025. <https://doi.org/10.1016/j.ymssp.2024.112161>

[3] D. J. Gauthier, E. Bollt, A. Griffith, and W. A. S. Barbosa, “Next generation reservoir computing,” *Nature Communications*, vol. 12, art. 5564, 2021. <https://doi.org/10.1038/s41467-021-25801-2>

[4] Y. Chen, *Vibration-based fault detection and severity assessment for fixed-axis gearboxes*, Ph.D. dissertation, Department of Mechanical Engineering, University of Alberta, 2020.

[5] P. D. Welch, “The use of fast Fourier transform for the estimation of power spectra,” *IEEE Transactions on Audio and Electroacoustics*, vol. 15, pp. 70–73, 1967. <https://doi.org/10.1109/TAU.1967.1161901>

[6] H. R. Künsch, “The jackknife and the bootstrap for general stationary observations,” *Annals of Statistics*, vol. 17, pp. 1217–1241, 1989. <https://doi.org/10.1214/aos/1176347265>

[7] R. Tibshirani, “Regression shrinkage and selection via the lasso,” *Journal of the Royal Statistical Society: Series B*, vol. 58, pp. 267–288, 1996. <https://doi.org/10.1111/j.2517-6161.1996.tb02080.x>

[8] H. Zou and T. Hastie, “Regularization and variable selection via the elastic net,” *Journal of the Royal Statistical Society: Series B*, vol. 67, pp. 301–320, 2005. <https://doi.org/10.1111/j.1467-9868.2005.00503.x>

[9] L. Breiman, “Random forests,” *Machine Learning*, vol. 45, pp. 5–32, 2001. <https://doi.org/10.1023/A:1010933404324>

[10] J. H. Friedman, “Greedy function approximation: A gradient boosting machine,” *Annals of Statistics*, vol. 29, pp. 1189–1232, 2001. <https://doi.org/10.1214/aos/1013203451>

[11] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” *Neural Computation*, vol. 9, pp. 1735–1780, 1997. <https://doi.org/10.1162/neco.1997.9.8.1735>

[12] S. Bai, J. Z. Kolter, and V. Koltun, “An empirical evaluation of generic convolutional and recurrent networks for sequence modeling,” 2018. <https://arxiv.org/abs/1803.01271>

[13] A. Vaswani et al., “Attention is all you need,” *Advances in Neural Information Processing Systems*, vol. 30, 2017. <https://arxiv.org/abs/1706.03762>

[14] B. Lim, S. Ö. Arık, N. Loeff, and T. Pfister, “Temporal Fusion Transformers for interpretable multi-horizon time series forecasting,” *International Journal of Forecasting*, vol. 37, pp. 1748–1764, 2021. <https://doi.org/10.1016/j.ijforecast.2021.03.012>

# Appendix A. Reproducibility and source evidence

The project README and environment constraints identify the tested platform and dependency versions. The data manifest identifies uploaded files and their hashes. Run all commands from the isolated project root. Preserve the released result tables and checkpoints when running new builds. No source attachment or credential is needed to regenerate only the report from its existing project evidence.

```bash
python scripts/build_report.py --pages
python scripts/build_presentations.py --render
```

Table A.1. Scientific evidence map.""",
        table(["Evidence", "Location"], [
            ("Input and dependency evidence", "inputs/input_manifest.csv, README.md and requirements files"),
            ("Training and selected configurations", "results/optimization, results/metrics/src_summary.json"),
            ("Prediction and cost", "results/metrics/all_model_comparison.csv and tracked prediction arrays"),
            ("Candidate and synthetic diagnostics", "results/fault and figures/fault"),
            ("Report validity", "report/validation_report.md and report/qa"),
        ]),
    ]
    return "\n\n".join(txt) + "\n"


def audit(e, directory):
    import fitz

    docx = directory / "SRC_Final_Course_Report.docx"
    pdf = directory / "SRC_Final_Course_Report.pdf"
    with ZipFile(docx) as container:
        xml = ET.fromstring(container.read("word/document.xml"))
    tags = Counter(item.tag.split("}")[-1] for item in xml.iter())
    if tags["oMath"] < 5 or tags["tbl"] < 6 or tags["hyperlink"] < 10:
        raise RuntimeError("Expected native equation, table and navigation structure")
    with fitz.open(pdf) as viewed:
        page_count = len(viewed)
        extracted = " ".join(page.get_text() for page in viewed)
    measured = number(e.primary, "mse")
    if measured is not None and f"{measured:.6f}" not in extracted:
        raise RuntimeError("Saved primary MSE missing in the rendered report")
    absent = [name(r) for r in e.measured if fmt(r, "mse") not in extracted]
    if absent:
        raise RuntimeError(f"Matched MSE missing in rendering: {absent}")
    content = [
        "# Report validation", "",
        f"- PDF rendered {page_count} pages from the delivered Word file.",
        f"- Word contains {tags['oMath']} native mathematical objects, {tags['tbl']} tables and {tags['drawing']} drawings.",
        f"- Found {tags['hyperlink']} document hyperlinks, including generated navigation.",
        f"- Saved primary MSE frequency checked against the rendered text ({fmt(e.primary, 'mse')}).",
        f"- Comparison table regenerated from {len(e.comparison)} saved model rows. Every displayed MSE was checked in the rendered PDF.",
        "- Section navigation and convergence of page references verified in the Word styling step.",
    ]
    (directory / "validation_report.md").write_text("\n".join(content) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", action="store_true")
    args = parser.parse_args()
    e = Evidence()
    with TemporaryDirectory(prefix="src_matplotlib_") as mpl:
        os.environ.setdefault("MPLCONFIGDIR", mpl)
        diagram(ROOT / "figures" / "report", e)
    (REPORT / "SRC_Final_Course_Report.md").write_text(report(e), encoding="utf-8")
    cmd = [sys.executable, str(REPORT / "style_word.py")]
    if args.pages:
        cmd.append("--pages")
    subprocess.run(cmd, cwd=REPORT, check=True)
    audit(e, REPORT)


if __name__ == "__main__":
    main()
