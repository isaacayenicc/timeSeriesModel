"""Build and clean-kernel replay the eleven educational SRC notebooks."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
from textwrap import dedent
from time import perf_counter

import nbformat as nb
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
BASE = r"""
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic("matplotlib", "inline")
from IPython.display import Image, display

hint = os.environ.get("SRC_PROJECT_DIR")
locations = [Path(hint).resolve()] if hint else [Path.cwd(), *Path.cwd().parents]
ROOT = next((p for p in locations if (p / "src" / "src_gearbox").is_dir()), None)
if ROOT is None:
    raise RuntimeError("Open within the project or set SRC_PROJECT_DIR.")
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
DATA = Path(os.environ.get("SRC_DATA_DIR", ROOT / "data" / "raw"))
RESULTS = ROOT / "results"
START = 1792
N = 24000
pd.options.display.max_colwidth = 120
pd.options.display.width = 115
plt.rcParams.update({"figure.dpi": 105, "axes.grid": True, "grid.alpha": .18})

def table(relative):
    file = ROOT / relative
    if not file.is_file():
        print(f"NOT AVAILABLE YET · {relative}")
        return None
    return pd.read_csv(file)

def evidence(relative):
    file = ROOT / relative
    if not file.is_file():
        print(f"NOT AVAILABLE YET · {relative}")
        return {}
    return json.loads(file.read_text(encoding="utf-8"))

def show_fig(relative, width=780):
    file = ROOT / relative
    if file.is_file():
        display(Image(filename=str(file), width=width))
    else:
        print(f"NOT AVAILABLE YET · {relative}")

def forecast(relative):
    file = ROOT / relative
    if not file.is_file():
        print(f"NOT AVAILABLE YET · {relative}")
        return {}
    with np.load(file, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    np.testing.assert_array_equal(arrays["target_indices"], np.arange(START, N))
    assert np.isfinite(arrays["actual"]).all()
    return arrays

def align_forecasts(reference, *others):
    for other in others:
        for key in ("target_indices", "actual", "rpm", "time_s"):
            np.testing.assert_array_equal(reference[key], other[key])

def draw_forecast(bundle, keys, view=slice(5000, 5200)):
    if not bundle:
        return
    y, t = bundle["actual"], bundle["time_s"]
    fig, ax = plt.subplots(2, 1, figsize=(9.0, 4.0), sharex=True, layout="constrained")
    ax[0].plot(t[view], y[view], color="#141a26", lw=1.2, label="Measured")
    for key, label in keys.items():
        if key in bundle:
            pred = bundle[key]
            ax[0].plot(t[view], pred[view], lw=0.9, label=label)
            ax[1].plot(t[view], (y-pred)[view], lw=0.9, label=label)
    ax[0].legend(ncol=3, fontsize=8)
    ax[0].set_ylabel("Std. vibration")
    ax[1].set(ylabel="Std. residual", xlabel="Acquisition time (s)")
    plt.show()

print("Read-only evidence replay.", f"Python {platform.python_version()}",
      f"NumPy {np.__version__}. One-step observed history through t−1.")
"""

def md(value):
    return nb.v4.new_markdown_cell(dedent(value).strip())


def code(value):
    return nb.v4.new_code_cell(dedent(value).strip())


def final_discussion(text):
    return md("### Interpretation and limits\n\n" + dedent(text).strip())


def overview():
    return [
        md(r"""
        ### Course question, units, and integrity

        Predict standardized vibration one sample ahead, then assess prediction accuracy,
        compute, and fault-sensitive residual structure. A 3.2-kHz sample interval is
        0.3125 ms. One prediction row at index $t$ is computed from earlier $x$ and RPM.
        The shared assessment interval is $[1792,24000)$, meaning 22,208 scored targets.

        Development validation provides a held-out segment. Fault records grouped by
        MATLAB index support conditional analysis only. The inventory distinguishes files
        absent now from inputs checked and eligible. Hashes verify identity, not
        scientific representativeness. See Companion Chapters 1, 3, 6 and 13.
        """),
        code(r"""
        availability = evidence("results/data_availability.json")
        display(pd.Series(availability, name="Intake finding").to_frame())
        inputs = table("inputs/input_manifest.csv")
        if inputs is not None:
            print(len(inputs), "source entries")
            display(inputs.head(6))
        """),
        code(r"""
        from src_gearbox.data import find_input, load_record

        path = find_input(DATA, "sigtrain")
        if path:
            record = load_record(path)
            np.testing.assert_allclose(np.diff(record.t), 1/record.fs, rtol=0, atol=2e-12)
            assert len(record.x) == N and record.fs == 3200
            axes = plt.subplots(2, 1, figsize=(9, 4), sharex=True, layout="constrained")[1]
            axes[0].plot(record.t, record.x, color="#256b76", lw=.65)
            axes[0].set_ylabel("Std. vibration")
            axes[1].plot(record.t, record.rpm, color="#a05239")
            axes[1].set(xlabel="Time (s)", ylabel="Input shaft RPM")
            plt.show()
        else:
            show_fig("figures/eda/eda_speed.png")
        """),
        code(r"""
        normal = table("results/normalization_audit.csv")
        if normal is not None:
            display(normal.head(4))
        overlap = table("results/record_overlap_audit.csv")
        if overlap is not None:
            print(len(overlap), "audit rows")
            display(overlap.head(6))
        idx = np.arange(START, N, dtype="<i8")
        print("Scored-index SHA-256", hashlib.sha256(idx.tobytes()).hexdigest())
        assert idx.size == 22208 and idx[0] == 1792 and idx[-1] == 23999
        """),
        final_discussion("""
        A longer diagnostic record may contain a designated excerpt. Exclusion acts on the
        acquisition, so scores on superficially different windows cannot be mistaken for
        independent evidence. Treat the raw and standardized channels as distinct scales.
        *Next*: use the designated training data to derive controlled feature hypotheses.
        """),
    ]


def eda():
    return [
        md(r"""
        ### Time-frequency evidence generates candidate features

        For input shaft speed $n$ rpm, $f_k(t)=37kn(t)/60$ Hz. At constant speed
        the period in samples is $3200/f_1$. Variable speed moves that period,
        motivating past-angle interpolation. A fixed-Hertz component follows a
        separate curve. Short-time Fourier transforms and offline angle spectra
        may use completed samples; predictor features must be left-looking.
        Companion Chapters 2, 9–11, and 14 give the physics and unit derivations.
        """),
        code(r"""
        train_eda = evidence("results/eda/summary.json")
        columns = ["samples","fs_hz","duration_s","rpm_min","rpm_max",
                   "mesh_hz_min","mesh_hz_max","max_mesh_harmonic_at_highest_speed"]
        display(pd.Series({k:train_eda[k] for k in columns if k in train_eda}).to_frame("Training only"))
        if "rpm_max" in train_eda:
            mesh_at_maximum_speed = 37 * train_eda["rpm_max"] / 60
            harmonic_at_12 = 12 * mesh_at_maximum_speed
            assert harmonic_at_12 < train_eda["fs_hz"] / 2
            print(f"Order 444 at maximum observed training speed: {harmonic_at_12:.2f} Hz")
        show_fig("figures/eda/eda_spectrogram.png")
        show_fig("figures/eda/eda_orders.png")
        """),
        md(r"""
        Mesh trajectories turn a spectrogram into a quantitative check. The order
        spectrum tests recurrence in cycles per revolution; the time spectrum uses
        Hertz. Their agreement at one operating point cannot establish a true
        structural frequency. Examine speed strata and along-ridge error.
        """),
        code(r"""
        display(evidence("results/eda/synthetic_phase_checks.json"))
        regimes = table("results/eda/tables/eda_band_power_by_speed_direction.csv")
        if regimes is not None:
            display(regimes.head(8))
        show_fig("figures/eda/eda_pacf.png")
        """),
        code(r"""
        screen = table("results/eda/tables/src_feature_screen.csv")
        if screen is not None:
            display(screen.head(16))
        ideas = table("results/eda_feature_hypotheses.csv")
        if ideas is not None:
            display(ideas.head(20))
        show_fig("figures/eda/eda_feature_decisions.png")
        """),
        final_discussion("""
        Order-aligned and angular features merit controlled tuning. A measured waveform
        can exhibit a fixed-frequency band even when a candidate pair offers little
        incremental prediction value. Cubic expansion should earn its additional columns
        on held-out fold blocks. The synthetic chirp validates units and state indexing;
        it makes no claim about defect sensitivity in the rig.
        """),
    ]


def reference():
    return [
        md(r"""
        ### From AR to sparse LPV-ARMA

        With $\tilde e_j=x_j-\hat x^{(1)}_j$, the strict second stage uses
        $\hat x_t=\sum_i a_i(s_{t-1})x_{t-i}+
        \sum_j b_j(s_{t-1})\tilde e_{t-j}$.
        A design column is a lag multiplied by a speed-basis value. Stage one
        estimates proxy innovations, which must be available before each target.
        ARIMA first predicts a differenced outcome and reconstructs the level
        from earlier observed vibration. See Companion Chapters 4, 5 and 21.

        AIC compares order candidates at the same target indices.
        Sparse evidence requires the objective scale, solver criterion, and
        estimation/selection policy. The local guarded support selection, QR
        refit, archived coefficients, and authors' published Table 6 are
        intentionally different evidence streams.
        """),
        code(r"""
        paper = table("results/metrics/published_paper_reference.csv")
        local = table("results/metrics/local_statistical_reference.csv")
        display(paper)
        display(local)
        if local is not None:
            k = local[(local.convention == "strict_development") & (local.model == "spa_lpv_arma")]
            if len(k):
                assert int(k.iloc[0].n) == 22208
        sparse = evidence("results/metrics/strict_sparse_support_refit.json")
        print("Support-reconciliation fields", list(sparse))
        """),
        code(r"""
        from numpy.polynomial.legendre import legvander

        x = np.array([1., 3., 2., 5., 8.])
        report_idx = np.array([2, 3, 4])
        lag_count = np.array([1, 2])
        lags = x[report_idx[:, None] - lag_count]
        operating_state = np.array([-.8, -.2, .5, .8, 1.])
        sched = legvander(operating_state[report_idx-1], 2)
        design = (lags[:,:,None]*sched[:,None,:]).reshape(3, 6)
        np.testing.assert_array_equal(design[:, ::3], lags)
        display(pd.DataFrame(design).round(3))
        """),
        code(r"""
        paired = forecast("results/predictions/baselines_sigvalid.npz")
        draw_forecast(paired, {"sparse_lpv_ar":"Sparse LPV–AR",
                               "sparse_lpv_arma":"Sparse LPV–ARMA"})
        """),
        final_discussion("""
        Current-speed retrospective and random-row historical conventions describe the
        source reconstruction. They do not enter the strict ranking without the matching
        history and mask. The published test segment has a different provenance. A lower
        value on that segment does not imply direct paired superiority.
        """),
    ]


def theory():
    return [
        md(r"""
        ### From a lag row to structured model state

        A delay vector $[x_{t-1},x_{t-3},x_{t-8}]$ is a feature map. Add a
        past RPM basis, angle-sampled history, oscillator quadratures and selected
        products. A grouped ridge model solves a scaled positive-definite system
        with an unpenalized intercept. Grouping limits width: $p(p+1)/2$ quadratic
        terms expand quickly even before any cubic terms. The regression is linear
        in fitted weights and nonlinear in measured histories.

        A finite oscillator demodulates the past with $\exp(-i2\pi k u_j)$ and rotates
        to the target-clock phase using already observed RPM. For $r=\exp(-1/\tau)$,
        the delayed state obeys $D_{t+1}=rD_t+d_t-r^M d_{t-M}$. Normalized quadratures
        multiply by $2/\sum_{j=0}^{M-1}r^j$. The finite subtraction defines the
        exact history. Companion Chapters 7–12 derive these maps and units.
        """),
        code(r"""
        from src_gearbox.src_model import angular_lags, oscillator_states, phase_turns, solve_readout

        fs = 3200.0
        rpm = np.linspace(125., 165., 760)
        turns, target_turns = phase_turns(rpm, fs)
        order, tau, memory = 148., 22., 96
        x = 0.8*np.cos(2*np.pi*order*turns)
        state = oscillator_states(x, fs, rpm, order=order, tau=tau, memory=memory)
        weights = np.exp(-np.arange(memory)/tau)
        idx = np.arange(memory, len(x))
        prior = idx[:, None] - np.arange(1, memory+1)
        rotation = np.exp(2j*np.pi*order*(target_turns[idx, None]-turns[prior]))
        direct = 2*np.sum(weights*x[prior]*rotation, axis=1)/weights.sum()
        np.testing.assert_allclose(state[idx], direct, rtol=2e-11, atol=2e-11)
        mutated = x.copy()
        mutated[420:] += 50
        mutated_rpm = rpm.copy()
        mutated_rpm[420:] += 5
        probe = oscillator_states(mutated, fs, mutated_rpm, order=order, tau=tau, memory=memory)
        np.testing.assert_array_equal(state[:421], probe[:421])
        print("Finite oscillator direct agreement =", float(np.max(np.abs(state[idx]-direct))))
        """),
        code(r"""
        t = np.arange(1800)/fs
        affine_x = 2*t+1
        fixed_rpm = np.full(len(t), 120.0)
        wanted = np.array([1690,1730])
        observed = angular_lags(affine_x, fixed_rpm, fs, wanted, steps=4, cycles=1)
        shifts = np.arange(1,5)/(37*4)*fs/2
        truth = 2*((wanted[:,None]-shifts)/fs)+1
        np.testing.assert_allclose(observed, truth, atol=2e-7, rtol=0)
        gram_example = np.array([[0.,1.],[1.,1.],[2.,1.],[3.,1.]])
        desired = np.array([.2,1.,1.7,3.])
        penalty = np.array([.25,0.])
        coef, diag = solve_readout(gram_example, desired, penalty)
        reference = np.linalg.solve(gram_example.T@gram_example + len(desired)*np.diag(penalty),
                                    gram_example.T@desired)
        np.testing.assert_allclose(coef, reference)
        display({"slope":coef[0],"unpenalized_intercept":coef[1],**diag})
        """),
        final_discussion("""
        The primitive checks isolate unit, phase and dependency errors without
        using rig data. A stable recurrence does not make the model a trained
        recurrent network. The oscillator settings are chosen in the guarded
        outer search, while the coefficients are fitted by the inner solver.
        """),
    ]


def optimization():
    return [
        md(r"""
        ### Joint search and the guard

        This record has densely overlapping sampled histories. Each test block
        $[a,b)$ has a 1,792-sample fit exclusion before and after the block. Start
        fresh feature states inside the retained fit segments. Stage-specific
        designs screen coherent physical groups, nonlinear width, proxy history,
        solver/penalty, compactness, and adapter choices. Inspect the actual
        configuration and objective weights before interpreting the incumbent.

        A pure-MSE setting, a bounded compact setting, and an objective-ranked
        setting can differ. Compare their clean fold, perturbation and pulse
        evidence. Companion Chapters 6, 15, and 16 explain rank provenance.

        With mean nominal error $C$, mean RPM-perturbed error $S$, mean pulse
        distortion $D$, and learned-parameter count $p$, the recorded utility is
        $J=C+0.14\max(0,S-C)+0.009D+0.000002p$. Here $C$ and $S$ use squared
        standardized vibration units. $D$ is a dimensionless prediction-energy
        ratio to which the specified coefficient assigns an empirical error
        penalty. The positive part keeps an incidental stress improvement from
        paying for added instability elsewhere. All four weights or terms are
        part of the configuration contract. Feature count is distinct from
        parameter count because the innovation fit has its own coefficients.
        No value from the eventual development record appears in $J$.
        """),
        code(r"""
        trials = table("results/optimization/trials.csv")
        folds = table("results/optimization/fold_scores.csv")
        selected = evidence("results/optimization/selected_configs.json")
        if selected:
            print("Recorded optimization protocol")
            display(selected["protocol"])
            display(pd.DataFrame(selected["selection"]).drop(columns="config",errors="ignore"))
        if trials is not None:
            display(trials.groupby(["stage","status"]).size().to_frame("Trials"))
            good = trials.loc[trials.status.eq("DONE")]
            if len(good):
                fields = ["trial_id","stage","fold_mse","fold_stress_mse","fold_pulse_distortion",
                          "utility","parameters"]
                display(good.sort_values("utility")[fields].head(8))
        """),
        code(r"""
        if trials is not None and len(good):
            fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.3), layout="constrained")
            for stage, group in good.groupby("stage"):
                ax[0].scatter(group.fold_mse, group.fold_stress_mse, label=stage, s=25)
                ax[1].scatter(group.parameters, group.utility, label=stage, s=25)
            ax[0].set(xlabel="Guarded clean MSE", ylabel="Guarded stressed MSE")
            ax[1].set(xlabel="Stored coefficients", ylabel="Guarded utility")
            ax[0].legend(ncol=3, fontsize=8)
            plt.show()
        if folds is not None:
            fields = ["trial_id","lo","hi","mse","stress_mse","history","solver","condition","fit_rows"]
            display(folds.loc[folds.trial_id.isin(good.nsmallest(4,"utility").trial_id),fields])
        show_fig("figures/optimization/selection_history.png")
        show_fig("figures/optimization/guarded_frontier.png")
        pareto = table("results/optimization/pareto_front.csv")
        if pareto is not None:
            display(pareto[[k for k in ["trial_id","stage","fold_mse","fold_stress_mse",
                                        "fold_pulse_distortion","parameters"] if k in pareto]].head(10))
        """),
        code(r"""
        if selected and trials is not None:
            protocol = selected["protocol"]
            done = trials.loc[trials.status.eq("DONE")]
            recomputed = (
                done.fold_mse
                + protocol["stress_excess_weight"] * (done.fold_stress_mse-done.fold_mse).clip(lower=0)
                + protocol["pulse_distortion_weight"] * done.fold_pulse_distortion
                + protocol["per_coefficient_penalty"] * done.parameters
            )
            np.testing.assert_allclose(done.utility,recomputed,atol=1e-12,rtol=0)
            for item in selected["selection"]:
                chosen = done.loc[done.trial_id.eq(item["trial_id"])].iloc[0]
                np.testing.assert_allclose(chosen.fold_mse,item["fold_mse"],rtol=0,atol=1e-12)
                print(item["model"], item["trial_id"], f"utility = {chosen.utility:.6f}")
        """),
        md("""
        Conditioning and solver status come from individual training fits.
        A broad oscillator bank can provide useful coverage and still add correlated
        columns. The optimized main and clean-MSE alternatives compare objective
        definitions fixed before development scoring. The Pareto frontier shows
        settings that avoid simultaneous domination across the measured criteria.
        """),
        final_discussion("""
        The best trial is relative to this schedule and its utility. Inspect fold
        variation, columns, and objective terms before treating small ranking
        changes as robust. An alternative with better supplied development MSE is
        a descriptive result. It does not reopen the guarded choice.
        """),
    ]


def training():
    return [
        md(r"""
        ### Configuration, weights, and a new record

        Fixed and optimized banks share the one-step task but may differ in
        physics and capacity. The base prediction is an affine readout of
        normalized selected groups. An optional causal adapter adds a clipped,
        exponentially weighted correction using earlier base errors. This notebook
        reads existing chosen settings and reloads the actual checkpoint with
        frozen feature ordering. For weights estimated on the measured course
        train segment, recover its identity using the saved hash.

        Companion Chapters 8, 12 and 15 distinguish the three residual roles,
        feature-scaling fit rows, and portably serialized runtime state.
        """),
        code(r"""
        registry = evidence("checkpoints/src_registry.json")
        items = registry.get("models", [])
        if items:
            display(pd.DataFrame(items)[["model","display","history","feature_columns","parameters",
                                        "fit_s","train_mse","checkpoint_bytes"]])
        ref = [x for x in items if x["model"] == "primary_pi_src"]
        if ref:
            primary = ref[0]
            display(pd.Series(primary["config"], name="Frozen setting").to_frame())
            checkpoint = ROOT / primary["checkpoint"]
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == primary["sha256"]
        """),
        code(r"""
        from src_gearbox.data import find_input, load_record
        from src_gearbox.src_model import load_model

        bundled = forecast("results/predictions/src_sigvalid.npz")
        path = find_input(DATA, "sigvalid")
        if ref and path and bundled:
            sample = load_record(path)
            final_model = load_model(checkpoint)
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=1):
                yhat = final_model.predict(sample.x, sample.rpm, sample.fs)
            np.testing.assert_allclose(yhat[START:], bundled["primary_pi_src"], atol=1e-8, rtol=0)
            weights = pd.DataFrame(final_model.readout_features, columns=["term","group","weight"])
            display(weights.assign(w2=weights.weight**2).groupby("group").agg(
                columns=("term","count"), squared_norm=("w2","sum")))
        draw_forecast(bundled, {"primary_pi_src":"Primary PI-SRC",
                                 "pi_src_adapter_off":"Primary base only"})
        """),
        md("""
        ### A miniature train and restore example

        Fit a 12-lag readout to an illustrative noisy process before predicting
        a second pseudo-record. Fit statistics come from the first pseudo-record.
        The primary experiment fits the full training record using the
        guarded search selection. This cell exercises the same model entry
        points on synthetic arrays and writes only to a temporary directory.
        """),
        code(r"""
        from tempfile import TemporaryDirectory
        from src_gearbox.src_model import Config, fit
        from threadpoolctl import threadpool_limits

        rng = np.random.default_rng(2026)
        time_axis = np.arange(1800)/3200
        rate = np.full(1800,140.)
        signal = np.sin(2*np.pi*75*time_axis)+rng.normal(scale=.08,size=len(time_axis))
        config = Config(variant="SRC-Linear",lags=12,ridge=.02,adapter_gain=0,
                        pulse_preservation=0,augmentation=0)
        with TemporaryDirectory() as folder, threadpool_limits(limits=1):
            model = fit(config,[(signal[:1100],rate[:1100])],3200)
            state = Path(folder)/"illustrative.npz"
            model.save(state)
            yhat = model.predict(signal[1100:],rate[1100:],3200)
            from_disk = load_model(state).predict(signal[1100:],rate[1100:],3200)
        need = np.isfinite(yhat)
        np.testing.assert_array_equal(yhat[need],from_disk[need])
        print(f"Synthetic holdout MSE: {np.mean((signal[1100:][need]-yhat[need])**2):.5f}")
        """),
        final_discussion("""
        The reload check establishes reproducible numerical identity for the saved
        model. Weight norms describe the fitted coordinate system and can reflect
        collinearity rather than physical causation. For group usefulness, inspect
        the guarded and matched development ablations.
        """),
    ]


def comparison():
    return [
        md(r"""
        ### All model families and exact target alignment

        Statistical models use lag or innovation dynamics. Linear, Ridge, Lasso
        and Elastic Net compare readout regularization. SVR uses a kernel and
        bounded fit support. Forest and boosting ensembles partition a supplied
        feature space. MLP flattens windows; LSTM and GRU use gated temporal
        updates; CNN and TCN operate by shared filters and controlled dilation.
        The Transformer uses historical patch attention, and compact TFT uses
        variable selection and historical gating/attention. NARX explicitly
        conditions nonlinear lag dynamics on past exogenous speed. Sparse LPV
        uses varying coefficients and, in its two-stage variant, proxy residuals.

        The baseline checkpoint lineage is preserved. New inference/mask/causal
        audit uses the same vibration as PI-SRC, and the separate RPM-withheld
        controls determine whether observed speed helps within a matched family.
        Companion Chapter 17 goes through all 21 algorithms and Chapter 23
        relates observed outcomes to their assumptions.

        ### Which statistical or learning assumptions does each baseline test?

        Let $\mathbf{x}_{<t}$ and $\mathbf{n}_{<t}$ denote observed vibration and
        RPM histories. A historical representation $v_t$ contains only permitted
        rows. Below $b$ denotes an intercept and $w$ a fitted coefficient vector.
        Some models operate on the supplied vibration lags, while a tabular family
        can receive richer permitted features. Inspect the configuration table for
        the actual width, fit scope and effective data history of each comparison.

        | Model | How it turns history into a prediction |
        | --- | --- |
        | AR | Predicts the level by a weighted combination of positive vibration lags. It tests stationary linear dependence without an operating input. |
        | ARMA | Adds coefficients on earlier forecast errors to the AR predictor. Initial state and causal innovation recursion determine the early predictions. |
        | ARIMA | Fits the autoregressive and innovation relation on a differenced signal, then reconstructs the prediction at level using observed past vibration. |
        | ARX | Adds measured exogenous history to a linear lag relation. Speed withholding tests the incremental value of that input within the matched fit. |
        | NARX | Uses a parameterized nonlinear map of past vibration and exogenous history. It can express state-dependent lag effects with a short fixed context. |
        | Sparse LPV–AR | Multiplies each vibration lag by speed-basis values and sparsely selects the resulting scheduled columns before coefficient refit. |
        | Sparse LPV–ARMA | Adds scheduled proxy innovations from the first stage. Its benchmark isolates a strict causal refit on protected sparse supports. |
        | Linear Regression | Fits $b+w^Tv_t$ by squared-error minimization without coefficient shrinkage. Its success tests the predictive sufficiency of the chosen inputs. |
        | Ridge | Augments squared error with $\ell_2$ weight penalty. Correlated historical coordinates can yield more stable forecasts after shrinkage. |
        | Lasso | Uses an $\ell_1$ penalty that can drop unnecessary feature columns before evaluation. Coefficient support is a property of this fit. |
        | Elastic Net | Couples $\ell_1$ and $\ell_2$ penalties. This tests the value of retaining groups of related features while encouraging sparsity. |
        | SVR | Predicts with a kernel expansion supported on selected training examples, using an interval-insensitive error loss. Training subsets control cost. |
        | Random Forest | Averages predictors from randomized trees and tests whether threshold interactions in the input features generalize. |
        | Gradient Boosting | Fits successive small tree updates to improve the current loss. Tuning controls tree complexity, shrinkage and number of iterations. |
        | MLP | Learns a nonlinear feedforward transformation of a flattened past window. It has no recurrent state and no explicit mesh phase. |
        | LSTM | Learns input, forget and output gates for hidden and cell state so earlier examples can persist through the context. |
        | GRU | Blends earlier state with a reset-gated candidate state. It uses fewer gate groups than the comparable LSTM architecture. |
        | CNN | Learns local temporal filters across historical samples. Shared filters are a time-structured inductive bias even without angular alignment. |
        | TCN | Uses a stack of dilated causal filters to enlarge its historical receptive field while limiting filter count. |
        | Time Series Transformer | Embeds historical patches and relates them through attention. Learnable attention weights operate within the allowed past context. |
        | Temporal Fusion Transformer | Uses compact variable selection, gating and attention on historical context. The local checkpoint does not implement all modules of the original multihorizon architecture. |

        The rows test families under measured configurations. They cannot isolate
        an architecture alone when feature processing, regularization, training
        width, and guarded tuning budgets differ. The checkpoint lineage and matched
        ablations show which claims survive tighter controls.
        """),
        code(r"""
        coverage = table("results/metrics/model_coverage.csv")
        if coverage is not None:
            display(coverage)
        ranking = table("results/metrics/all_model_comparison.csv")
        base = forecast("results/predictions/baselines_sigvalid.npz")
        src = forecast("results/predictions/src_sigvalid.npz")
        archive = forecast("results/predictions/archived_src_sigvalid.npz")
        if base and src and archive:
            align_forecasts(base, src, archive)
        """),
        code(r"""
        samples = {k:v for bundle in (base,src,archive) for k,v in bundle.items()}
        if ranking is not None and samples:
            cols = ["model","mse","rmse","mae","bias","fit_s","warm_median_ms","parameters",
                    "checkpoint_bytes","scope"]
            display(ranking[[k for k in cols if k in ranking]].sort_values("mse"))
            rebuilt = []
            for row in ranking.itertuples(index=False):
                assert row.model in samples, row.model
                mse = np.mean((samples["actual"]-samples[row.model])**2)
                rebuilt.append((row.model,mse))
                np.testing.assert_allclose(mse,row.mse,atol=1e-11,rtol=0)
            print(len(rebuilt), "reported model scores independently rederived")
        draw_forecast(samples, {"primary_pi_src":"Primary PI-SRC",
                                "sparse_lpv_arma":"Sparse LPV–ARMA",
                                "cnn":"CNN"})
        """),
        code(r"""
        parity = table("results/metrics/vibration_rpm_parity.csv")
        if parity is not None:
            display(parity[["model","measured_mse","vibration_only_mse",
                            "delta_measured_minus_vibration_only"]])
        configs = table("results/metrics/baseline_configs.csv")
        if configs is not None:
            display(configs)
        """),
        code(r"""
        summary = evidence("results/metrics/comparison_summary.json")
        if summary:
            print(f"Checked required families: {summary['required_baselines_checked']}/21")
            print(f"Selected minus archive MSE: {summary['primary_minus_anchor']:+.6f}")
            print(f"Selected improvement over leading required baseline: {summary['relative_improvement_over_required_baseline_pct']:.2f}%")
        paired = table("results/statistical_tests/paired_moving_block.csv")
        if paired is not None:
            display(paired.loc[paired.block_samples.eq(640) & paired.model.isin(
                ["archived_src_corrected","linear","sparse_lpv_arma","cnn","ar"])])
        show_fig("figures/comparison/rpm_quartiles.png")
        show_fig("figures/comparison/residual_structure.png")
        show_fig("figures/comparison/paired_intervals.png")
        """),
        final_discussion("""
        Report the rank of the primary prespecified model even if a correction,
        alternate search outcome, or historical forecast scores lower. Inputs
        and device costs differ across model families. A common target mask is
        necessary for fair evaluation but does not give every model the same
        parameter count or effective number of training examples.
        """),
    ]


def diagnostics():
    return [
        md(r"""
        ### Offline angular diagnosis from causal forecast residuals

        Form $e_t=x_t-\hat x_t$ on the common target range. Integrate measured
        $n/60$ to turns and interpolate the residual to a uniform turn grid.
        The Hilbert-envelope magnitude summarizes modulation. With centered
        envelope $v_q$, define $R(\ell)=\sum_qv_qv_{q+\ell}$, and compute
        $\mathrm{PMI}=R(\ell)/[R(0)-R(\ell)]$. On a grid of 2,048 samples per turn,
        $\ell=2048$ matches one revolution. The historical index convention
        evaluates lag 2047 under semantic comparison. These postrecord steps
        may use the completed residual series.
        """),
        code(r"""
        from src_gearbox.diagnostics.pmi import pmi_from_envelope

        phase = np.arange(8*64)/64
        periodic = 2 + np.cos(2*np.pi*phase)
        measured = pmi_from_envelope(periodic,64)
        display(measured)
        np.testing.assert_allclose(measured.correlation, 7/8, atol=1e-12)
        np.testing.assert_allclose(measured.value, 7.0, atol=1e-12)
        print("Analytical biased one-turn correlation = 7/8; PMI = 7")
        parity = evidence("results/fault/matlab_conformance.json")
        display(parity)
        """),
        code(r"""
        fault = table("results/fault/fault_group_summary.csv")
        if fault is not None:
            display(fault.loc[fault.metric.eq("lecturer_compatible")])
        integrity = evidence("results/fault/availability.json")
        if integrity:
            display(integrity)
        show_fig("figures/fault/representative_order.png")
        show_fig("figures/fault/fault_group_distribution.png")
        """),
        code(r"""
        pulses = table("results/fault/pulse_summary.csv")
        if pulses is not None:
            selected = pulses[pulses.model.isin(["primary_pi_src","pi_src_adapter_off","archived_src_corrected"])]
            display(selected)
        show_fig("figures/fault/pulse_tradeoff.png")
        """),
        final_discussion("""
        Whole-record angle interpolation and the Hilbert transform are diagnostic
        operations. Keep them separate from model inputs. A/B groups are provisional
        indices and overlapping acquisitions are excluded. A diagnostic contrast with
        conditional index groups cannot replace independently verified labels.
        Missing MATLAB helpers or gold outputs restrict numerical parity claims.
        """),
    ]


def ablations():
    return [
        md(r"""
        ### Separate representation, correction, and robustness

        A feature-group ablation generally refits the readout with that group
        removed. Mutating a feature only at inference is a stress intervention.
        A same-weight adapter-off comparison isolates the runtime correction. A
        matched adapter on polynomial SRC and the sparse LPV control determines
        whether earlier errors create a generic advantage independent of
        optimized physical states.

        Perturb RPM in the predictor and in the diagnostic angle axis separately.
        Add a repeatable synthetic pulse pattern into the measured history, then
        compare the changes in residual and pulse geometry. These are controlled
        responses, with a limited connection to physical crack severity.
        """),
        code(r"""
        ablated = table("results/ablations/src_ablations.csv")
        if ablated is not None:
            wanted = ["model","display","mse","mae","train_mse","feature_columns","history",
                      "parameters","warm_median_ms"]
            display(ablated[[k for k in wanted if k in ablated]].sort_values("mse"))
        stress = table("results/fault/rpm_stress_summary.csv")
        pulses = table("results/fault/pulse_summary.csv")
        if stress is not None:
            display(stress.head(12))
        if pulses is not None:
            display(pulses.head(12))
        matched = table("results/ablations/identical_adapter_controls.csv")
        if matched is not None:
            display(matched)
            np.testing.assert_allclose(
                matched.gain_mse, matched.uncorrected_mse-matched.matched_correction_mse)
        earlier = table("results/ablations/earlier_only_sensitivity.csv")
        if earlier is not None:
            display(earlier)
        show_fig("figures/fault/sensor_pmi.png")
        """),
        code(r"""
        candidates = forecast("results/predictions/src_sigvalid.npz")
        if candidates:
            clean = candidates["actual"]
            primary_error = (clean-candidates["primary_pi_src"])**2
            comparison = []
            for key, value in candidates.items():
                if key not in {"target_indices","actual","rpm","time_s","primary_pi_src"}:
                    comparison.append((key,np.mean((clean-value)**2-primary_error)))
            if comparison:
                display(pd.DataFrame(comparison,columns=["variant","MSE minus primary"]).sort_values("MSE minus primary"))
        """),
        final_discussion("""
        Interpret mode and score signs before claiming a feature gain. A bundled
        change in state width, oscillator bank and penalty only attributes the
        result to that complete design. Rotor-angle maps can be more vulnerable
        to RPM corruption than simpler fixed-time models. Diagnose the measured
        sensitivity before treating the nominal winner as deployment default.
        """),
    ]


def efficiency():
    return [
        md(r"""
        ### Cost and uncertainty on a matched record

        Batch real-time factor is median processing seconds divided by record
        duration. It includes feature construction according to the measured call
        scope. An amortized sample cost does not establish the deadline for the
        next sample. Inspect load, cold and warm timings separately when available.
        Parameter count and checkpoint bytes complement but do not replace RAM or
        operation counts. CPU measurements do not imply CUDA availability.

        Adjacent squared errors are correlated. Form the pairwise sequence
        $d_t=e_{A,t}^2-e_{B,t}^2$ and resample blocks of that same sequence.
        Distinct block lengths expose dependence sensitivity, not independent-rig
        transfer. See Companion Chapter 18.
        """),
        code(r"""
        metrics = table("results/metrics/all_model_comparison.csv")
        if metrics is not None:
            cols = ["model","mse","fit_s","warm_median_ms","warm_iqr_ms","warm_p95_ms",
                    "parameters","checkpoint_bytes"]
            selected = metrics[[k for k in cols if k in metrics]].copy()
            if "warm_median_ms" in selected:
                selected["RTF_7p5s"] = selected.warm_median_ms / 7500
                display(selected.sort_values("warm_median_ms").head(22))
                good = selected.dropna(subset=["warm_median_ms","mse"])
                fig, ax = plt.subplots(figsize=(8.2, 3.1), layout="constrained")
                ax.scatter(good.warm_median_ms,good.mse,c="#397881")
                ax.set(xscale="log",xlabel="Median CPU batch time (ms)",ylabel="Development MSE")
                plt.show()
        hardware = evidence("results/environment_intake.json")
        print("Environment fields",list(hardware))
        timings = table("results/timings/baseline_repeats.csv")
        if timings is not None:
            print(f"Recorded comparator timing measurements: {len(timings)}")
        seeds = table("results/metrics/baseline_seed_sensitivity_summary.csv")
        if seeds is not None:
            display(seeds)
        """),
        code(r"""
        from src_gearbox.comparison import blocked

        group = forecast("results/predictions/src_sigvalid.npz")
        benchmark = forecast("results/predictions/baselines_sigvalid.npz")
        if group and benchmark:
            actual = group["actual"]
            align_forecasts(group,benchmark)
            delta = (actual-benchmark["sparse_lpv_arma"])**2 - (actual-group["primary_pi_src"])**2
            recorded = table("results/statistical_tests/paired_moving_block.csv")
            rows = []
            for block in (320,640,1280):
                lower,upper = blocked(delta,block,2000,19)
                if recorded is not None:
                    ref = recorded.loc[recorded.model.eq("sparse_lpv_arma") & recorded.block_samples.eq(block)].iloc[0]
                    np.testing.assert_allclose((delta.mean(),lower,upper),
                        (ref.delta_mse,ref.interval_low,ref.interval_high),atol=1e-12,rtol=0)
                rows.append((block,delta.mean(),lower,upper))
            display(pd.DataFrame(rows,columns=["samples","paired delta","low 95%","high 95%"]))
        """),
        final_discussion("""
        The plotted difference is relative to a strict comparator on the same
        record. Bootstrapping blocks preserves pairing and local dependence within
        each draw. Confidence limits cannot account for differences between machines,
        sensor mountings or new acquisitions absent from the study.
        """),
    ]


def independent():
    return [
        md(r"""
        ### Hashes first, test availability second

        The selected configuration is a result of training-only selection. The
        one-time independent route checks hashes and keeps settings frozen before
        reading any newly supplied test record. If blind files are unavailable,
        the notebook preserves that status and avoids creating scores. This is a
        read-only checkpoint summary. Explicit testing belongs to the independent
        evaluation script after inspecting availability and the frozen manifest.

        Mid-record predictions use measured histories through $t-1$ and a common
        warm-up. A confirmed development improvement demonstrates performance on
        this segment and gives no independent-fault estimate without suitable
        inputs and ground truth. Companion Chapters 20, 22–25 document the checks.
        """),
        code(r"""
        from src_gearbox.data import find_input
        from src_gearbox.data.io_matlab import fault_sources

        blind = {"sigtest":find_input(DATA,"sigtest") is not None,
                 "sigfault2":bool(fault_sources(DATA,"sigfault2"))}
        inventory = evidence("results/data_availability.json")
        display(pd.DataFrame({
           "record":list(blind),
           "local input status":["PRESENT" if value else "NOT AVAILABLE" for value in blind.values()]
        }))
        if not any(blind.values()):
            print("Independent evaluation blocked by missing inputs.")
        print("Recorded acquisition-label status:",inventory.get("label_status","NOT AVAILABLE"))
        """),
        code(r"""
        registry = evidence("checkpoints/src_registry.json")
        if registry.get("models"):
            primary = next(item for item in registry["models"] if item["model"]=="primary_pi_src")
            assert hashlib.sha256((ROOT/primary["checkpoint"]).read_bytes()).hexdigest() == primary["sha256"]
            display({"primary":primary["model"],"checkpoint_sha256":primary["sha256"],
                     "training_sha256":registry["training_sha256"],
                     "mask_sha256":registry["target_mask_sha256"]})
        for record in ("availability.json","fault2_availability.json"):
            data = evidence(f"results/frozen/{record}")
            if data:
                display(data)
        """),
        final_discussion("""
        A freeze fixes the variant, adapter, transforms, and diagnostic policy.
        Loading a checkpoint and displaying availability consumes no test
        targets. The independent route should evaluate once under that contract.
        The course development predictions and provisional detection groups are
        scientifically meaningful only with their stated scope.
        """),
    ]


NOTEBOOKS = [
    ("Project_Overview_and_Data_Audit","Can the proposed benchmark be audited from original observations?",
     "Check input identity, measured units, overlap, and common scoring.",overview),
    ("Physics_Informed_EDA","Which observed structures justify causal physics features?",
     "Connect meshing, order ridges, memory and guarded experiments.",eda),
    ("Statistical_Reference_Reproduction","What part of the published sparse benchmark can be reproduced?",
     "Derive lag designs, proxy innovations and distinct comparison conventions.",reference),
    ("SRC_Theory_and_Feature_Architecture","How does an explicit physical state become a stable predictor?",
     "Derive and verify delayed angular and oscillatory state calculations.",theory),
    ("SRC_Joint_Physics_and_Hyperparameter_Optimization","Which configuration wins the predeclared guarded objective?",
     "Audit selection, feature cost, sensitivity and measured competing objectives.",optimization),
    ("SRC_Training_and_Selection","Can the exact selected model be restored identically?",
     "Read configurations, recreate forecasts and check stored scales and weights.",training),
    ("Full_Model_Comparison","How do all 21 controls compare on identical development targets?",
     "Read models by family, reconcile metrics and assess input parity.",comparison),
    ("Fault_Detection_and_PMI","Which residuals preserve one-turn modulation?",
     "Derive the diagnostic pipeline and interpret provisional record groups.",diagnostics),
    ("SRC_Ablation_and_Robustness","What is attributable to physics, capacity, and correction?",
     "Compare controlled ablations and distinguish prediction from angle stress.",ablations),
    ("Computational_Efficiency_and_Uncertainty","Which methods trade predictive error for CPU cost?",
     "Calculate real-time factor and dependence-aware uncertainty.",efficiency),
    ("Frozen_Independent_Test_and_Synthesis","Is independent evaluation possible under the frozen contract?",
     "Check hashes and score eligibility without selecting on test data.",independent),
]


def make(i, title, question, objective, body):
    notebook = nb.v4.new_notebook()
    notebook.metadata["kernelspec"] = {"display_name":"Python 3","language":"python","name":"python3"}
    notebook.metadata["language_info"] = {"name":"python"}
    notebook.cells = [
        md(f"# {i:02d} {title.replace('_',' ')}\n\n"
           f"**Scientific question.** {question}\n\n"
           f"**Learning objectives.** {objective}\n\n"
           "**Prerequisites.** Install the project requirements. Configure the data directory for raw-record checks. "
           "Portable evidence review also uses the existing run artifacts.\n\n"
           "**Dataset and run.** Course gearbox at 3.2 kHz, the project’s designated training and held-out development "
           "records, and the explicit diagnostic or blind availability files where relevant.\n\n"
           "**Execution.** This clean-kernel replay reads the evidence cited in each code cell. "
           "Use the explicit project pipeline for full fitting and the notebook build script for review replay. "
           "The companion guide contains notation, figures, and worked examples."),
        code(BASE),
        *body(),
        md("Read the [Companion Guide](../docs/companion_guide.md) and the [Project README](../README.md)."),
    ]
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data/raw")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--only", nargs="+", type=int, help="Execute selected notebook numbers")
    args = parser.parse_args()
    destination = ROOT/"notebooks"
    destination.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["SRC_DATA_DIR"] = str(args.data_dir.resolve())
    env["SRC_PROJECT_DIR"] = str(ROOT)
    env["MPLBACKEND"] = "Agg"
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    manifest = []
    for i, (name, question, aim, body) in enumerate(NOTEBOOKS):
        if args.only and i not in args.only:
            continue
        path = destination/f"{i:02d}_{name}.ipynb"
        nt = make(i, name, question, aim, body)
        start = perf_counter()
        if not args.build_only:
            NotebookClient(nt, timeout=120, kernel_name="python3",
                           resources={"metadata":{"path":str(ROOT)}}, allow_errors=False).execute(env=env)
        nb.validate(nt)
        nb.write(nt, path)
        record = {"notebook":path.name,"status":"BUILT" if args.build_only else "EXECUTED",
                  "seconds":perf_counter()-start,"cells":len(nt.cells),
                  "executed_code_cells":sum(cell.cell_type=="code" and cell.execution_count is not None for cell in nt.cells),
                  "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
        manifest.append(record)
        print(path.name, record["status"], f"{record['seconds']:.2f} s", flush=True)
    record_path = ROOT/"results"/"notebook_execution.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.only:
        record_path.write_text(json.dumps(manifest,indent=2)+"\n", encoding="utf-8")


if __name__=="__main__":
    main()
