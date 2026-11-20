"""Offline angular vibration diagnostics and paired robustness checks."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
from scipy import signal, stats
from scipy.integrate import cumulative_trapezoid

if TYPE_CHECKING:
    from .data import Record

SAMPLES_PER_REVOLUTION = 2048
REPRESENTATIVE_IDS = (1, 100, 101, 200)
BOOTSTRAP_SEED = 7780
Predictors = Mapping[str, Callable[["Record"], np.ndarray]]


@dataclass(frozen=True)
class Pmi:
    value: float
    correlation: float
    lag: int
    status: str


def angular_track(
    x: np.ndarray,
    rpm: np.ndarray,
    t: np.ndarray,
    samples_per_rev: int = SAMPLES_PER_REVOLUTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate from elapsed seconds to a uniform grid measured in revolutions.

    This operation is an offline diagnostic. Linear interpolation attenuates near
    the acquisition Nyquist frequency. The finer output grid adds no bandwidth.
    """
    x, rpm, t = (np.asarray(arr, dtype=np.float64) for arr in (x, rpm, t))
    if not (x.ndim == rpm.ndim == t.ndim == 1 and len(x) == len(rpm) == len(t)):
        raise ValueError("x, rpm, and t must be equally sized vectors")
    if samples_per_rev < 2 or len(x) < 3:
        raise ValueError("Insufficient input samples or angular resolution")
    if not all(np.isfinite(a).all() for a in (x, rpm, t)):
        raise ValueError("Nonfinite vibration, RPM or timestamps")
    if np.any(rpm <= 0) or np.any(np.diff(t) <= 0):
        raise ValueError("RPM and timestamp increments must be positive")

    rev = cumulative_trapezoid(rpm / 60.0, t, initial=0)
    count = int(np.floor(rev[-1] * samples_per_rev)) + 1
    grid = np.arange(count, dtype=np.float64) / samples_per_rev
    return grid, np.interp(grid, rev, x)


def analytic_envelope(
    angle_vibration: np.ndarray, *, matlab_envelope: bool
) -> np.ndarray:
    """Apply the FFT Hilbert transform with the documented MATLAB centering option."""
    if matlab_envelope:
        mean = float(np.mean(angle_vibration))
        return np.abs(signal.hilbert(angle_vibration - mean)) + mean
    return np.abs(signal.hilbert(angle_vibration))


def pmi_from_envelope(envelope: np.ndarray, lag: int) -> Pmi:
    """Use a single finite-length zero-lag denominator, as in normalized ACF."""
    a = np.asarray(envelope, dtype=np.float64)
    if lag < 1 or lag >= a.size:
        return Pmi(np.nan, np.nan, lag, "insufficient_angular_history")
    if a.ndim != 1 or not np.isfinite(a).all():
        return Pmi(np.nan, np.nan, lag, "invalid_envelope")
    centered = a - a.mean()
    energy = float(np.dot(centered, centered))
    scale = float(np.dot(a, a))
    if energy <= np.finfo(float).eps * scale * 8 or energy == 0:
        return Pmi(np.nan, np.nan, lag, "zero_envelope_variance")
    cross = float(np.dot(centered[:-lag], centered[lag:]))
    rho = cross / energy
    if energy - cross <= np.finfo(float).eps * energy * 8:
        return Pmi(np.nan, rho, lag, "degenerate_pmi_denominator")
    return Pmi(cross / (energy - cross), rho, lag, "ok")


def correlated_kurtosis(x: np.ndarray, period: int) -> float:
    """Return first-shift correlated kurtosis with zero padding before the record."""
    if period >= len(x):
        return np.nan
    xsq = np.square(x)
    energy = xsq.sum()
    return float(np.dot(xsq[:-period], xsq[period:]) / energy**2) if energy > 0 else np.nan


def order_spectrum(x: np.ndarray, samples_per_rev: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a coherent-gain-corrected, one-sided Hann amplitude spectrum."""
    window = signal.windows.hann(len(x), sym=False)
    centered = np.asarray(x) - np.mean(x)
    freq = np.fft.rfftfreq(len(x), d=1 / samples_per_rev)
    spectrum = np.abs(np.fft.rfft(centered * window)) / window.sum()
    spectrum[1 : None if len(x) % 2 else -1] *= 2
    return freq, spectrum


def diagnostics(
    x: np.ndarray,
    rpm: np.ndarray,
    t: np.ndarray,
    samples_per_rev: int = SAMPLES_PER_REVOLUTION,
    *,
    return_traces: bool = False,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Compute time-domain statistics, literal and corrected order-domain PMI."""
    phase, angular = angular_track(x, rpm, t, samples_per_rev)
    env_matlab = analytic_envelope(angular, matlab_envelope=True)
    env_math = analytic_envelope(angular, matlab_envelope=False)
    course = pmi_from_envelope(env_matlab, samples_per_rev - 1)
    if len(env_matlab) <= int(np.floor(3.1 * samples_per_rev + .5)):
        course = Pmi(np.nan, np.nan, samples_per_rev - 1, "insufficient_reference_acf_span")
    lag_corrected = pmi_from_envelope(env_matlab, samples_per_rev)
    physical = pmi_from_envelope(env_math, samples_per_rev)
    unrelated = pmi_from_envelope(env_matlab, round(0.63 * samples_per_rev))
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    kurtosis = (
        float(stats.kurtosis(x, fisher=False, bias=True))
        if np.std(x) > np.finfo(float).eps * rms
        else np.nan
    )
    row = {
        "samples_per_rev": samples_per_rev,
        "angular_samples": len(phase),
        "rotations": float(phase[-1]),
        "rms": rms,
        "crest_factor": float(np.max(np.abs(x)) / rms) if rms > 0 else np.nan,
        "pearson_kurtosis_biased": kurtosis,
        "correlated_kurtosis_m1": correlated_kurtosis(angular, samples_per_rev),
        "pmi_course": course.value,
        "pmi_course_rho": course.correlation,
        "pmi_course_lag": course.lag,
        "pmi_status": course.status,
        "pmi_lag_corrected": lag_corrected.value,
        "pmi_lag_corrected_status": lag_corrected.status,
        "pmi_math": physical.value,
        "pmi_math_status": physical.status,
        "pmi_unrelated_0p63_rev": unrelated.value,
    }
    if not return_traces:
        return row, {}

    order, vib_spec = order_spectrum(angular, samples_per_rev)
    _, env_spec = order_spectrum(env_math, samples_per_rev)
    return row, {
        "theta_rev": phase, "angular_signal": angular, "envelope": env_math,
        "order": order, "vibration_spectrum": vib_spec, "envelope_spectrum": env_spec,
    }


def _write_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _prediction(
    record: Record, predict: Callable[[Record], np.ndarray], start: int
) -> np.ndarray:
    value = np.asarray(predict(record), dtype=np.float64)
    if value.shape == (len(record.x),):
        value = value[start:]
    if value.shape != (len(record.x) - start,):
        raise ValueError("Prediction violates the full-record or common-mask shape contract")
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite prediction on the required common mask")
    return value


def _threshold_free_statistics(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Measure descriptive provisional-group separation with record resampling."""
    summary = []
    for model in dict.fromkeys(row["model"] for row in rows):
        relevant = [row for row in rows if row["model"] == model and row["eligible"]]
        for column in ("pmi_course", "pmi_lag_corrected", "pmi_math", "pmi_unrelated_0p63_rev"):
            values = {
                name: np.asarray(
                    [row.get(column, np.nan) for row in relevant if row["group"] == name]
                )
                for name in ("A", "B")
            }
            a = values["A"][np.isfinite(values["A"])]
            b = values["B"][np.isfinite(values["B"])]
            if a.size == 0 or b.size == 0:
                continue

            # This matrix gives tie-aware Mann–Whitney pair scoring, with B high.
            wins = (b[:, None] > a[None, :]) + 0.5 * (b[:, None] == a[None, :])
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            wa = rng.multinomial(a.size, np.ones(a.size) / a.size, size=1000) / a.size
            wb = rng.multinomial(b.size, np.ones(b.size) / b.size, size=1000) / b.size
            ac = np.clip(np.sum((wb @ wins) * wa, axis=1), 0, 1)
            gap = np.median(rng.choice(b, (1000, b.size)), axis=1)
            gap -= np.median(rng.choice(a, (1000, a.size)), axis=1)
            summary.append({
                "model": model, "metric": column, "n_A": a.size, "n_B": b.size,
                "median_A": np.median(a), "median_B": np.median(b),
                "median_gap_B_minus_A": np.median(b) - np.median(a),
                "median_gap_ci_low": np.quantile(gap, 0.025),
                "median_gap_ci_high": np.quantile(gap, 0.975),
                "margin_min_B_minus_max_A": b.min() - a.max(),
                "conditional_auc_B_high": wins.mean(),
                "conditional_auc_ci_low": np.quantile(ac, 0.025),
                "conditional_auc_ci_high": np.quantile(ac, 0.975),
                "bootstrap_replicates": len(ac),
                "status": "provisional_index_groups_only",
            })
    return summary


def _exact_reference_overlap(record: Record, references: list[tuple[str, Record]]) -> str:
    """Identify full joint vibration/RPM matches at any contiguous sample offset."""
    for source, reference in references:
        count = len(reference.x)
        if count > len(record.x):
            continue
        offsets = np.flatnonzero(record.x[: len(record.x) - count + 1] == reference.x[0])
        for offset in offsets:
            if np.array_equal(record.x[offset : offset + count], reference.x) and np.array_equal(
                record.rpm[offset : offset + count], reference.rpm
            ):
                return f"exact_{source}_overlap_offset_{offset}"
    return ""


def evaluate_faults(
    data_dir: Path,
    run_dir: Path,
    predictors: Predictors,
    start: int = 1792,
    *,
    dataset: str = "sigfault1",
    excluded_indices: Mapping[int, str] | set[int] | None = None,
    samples_per_rev: int = SAMPLES_PER_REVOLUTION,
) -> list[dict[str, object]]:
    """Evaluate frozen predictors on all fault records, one independent reset each."""
    from .data import find_input, iter_fault_records, load_record

    data_dir, run_dir = Path(data_dir), Path(run_dir)
    metrics = run_dir / "metrics"
    figures = run_dir / "figures"
    predictions = run_dir / "predictions"
    figures.mkdir(parents=True, exist_ok=True)
    predictions.mkdir(parents=True, exist_ok=True)
    references = []
    if dataset == "sigfault1":
        for name in ("sigtrain", "sigvalid"):
            path = find_input(data_dir, f"{name}.mat")
            if path is not None:
                references.append((name, load_record(path)))
    exclusions = excluded_indices or {}
    if isinstance(exclusions, set):
        exclusions = dict.fromkeys(exclusions, "exact_development_overlap")
    if dataset == "sigfault1":
        split_path = run_dir / "tables" / "split_manifest.csv"
        if split_path.exists():
            with split_path.open(newline="", encoding="utf-8") as stream:
                ineligible = {
                    int(row["record_id"].split(":")[1]): row["reason"]
                    for row in csv.DictReader(stream)
                    if row["record_id"].startswith("sigfault1:")
                    and row["eligible_development_diagnostics"].lower() == "false"
                }
            exclusions = {**ineligible, **exclusions}

    rows, resolution = [], []
    trace_arrays = {}
    diagnostic_times: list[float] = []
    for record in iter_fault_records(data_dir, dataset=dataset):
        index = record.index
        group = ("A" if index <= 100 else "B") if dataset == "sigfault1" else "unlabeled"
        exclusion = exclusions.get(index) or _exact_reference_overlap(record, references)
        rep = index in REPRESENTATIVE_IDS
        x, rpm, t = record.x[start:], record.rpm[start:], record.t[start:]

        for model in ("raw", *predictors):
            row: dict[str, object] = {
                "dataset": dataset, "record_id": index, "model": model, "group": group,
                "label_status": "unavailable", "eligible": not exclusion,
                "exclusion_reason": exclusion, "first_sample": start,
                "target_count": len(x), "mse": np.nan,
            }
            try:
                prediction = _prediction(record, predictors[model], start) if model != "raw" else None
                residual = x - prediction if prediction is not None else x
                if prediction is not None:
                    row["mse"] = float(np.mean(np.square(residual)))
                tick = perf_counter()
                features, traces = diagnostics(residual, rpm, t, samples_per_rev, return_traces=rep)
                diagnostic_times.append(1000 * (perf_counter() - tick))
                row.update(features)
            except (ValueError, FloatingPointError) as error:
                row.update({"eligible": False, "pmi_course": np.nan, "pmi_status": str(error)})
                traces = {}
            rows.append(row)

            if rep and traces:
                key = f"{index}:{model}"
                trace_arrays.update({f"{key}:{name}": arr.astype(np.float32) for name, arr in traces.items()})
                trace_arrays[f"{key}:residual"] = residual.astype(np.float32)
                if prediction is not None:
                    trace_arrays[f"{key}:prediction"] = prediction.astype(np.float32)
                if model == "raw":
                    trace_arrays[f"{index}:t"] = t.astype(np.float64)
                    trace_arrays[f"{index}:rpm"] = rpm.astype(np.float64)
                    trace_arrays[f"{index}:index"] = np.arange(start, len(record.x))
                    for spr in (1024, 2048, 4096):
                        sensitivity, _ = diagnostics(residual, rpm, t, spr)
                        resolution.append({
                            "record_id": index, "model": model, **sensitivity,
                            "minimum_native_samples_per_rev": record.fs * 60 / np.max(rpm),
                            "angular_nyquist_alias_risk": 2 * record.fs * 30 / np.min(rpm) > spr,
                        })

    if not rows:
        raise FileNotFoundError(f"No {dataset} records")
    if trace_arrays:
        np.savez_compressed(predictions / "fault_representative_traces.npz", **trace_arrays)

    _write_table(metrics / "fault_per_record.csv", rows)
    _write_table(metrics / "fault_group_summary.csv", _threshold_free_statistics(rows))
    _write_table(metrics / "angular_resolution_sensitivity.csv", resolution)
    _write_json(metrics / "fault_status.json", {
        "dataset": dataset, "records": len({row["record_id"] for row in rows}),
        "models": list(("raw", *predictors)), "warmup_samples": start,
        "diagnostic_median_ms": float(np.median(diagnostic_times)) if diagnostic_times else None,
        "diagnostic_p95_ms": float(np.quantile(diagnostic_times, .95)) if diagnostic_times else None,
        "diagnostic_timing_includes_trace_spectra_on_representatives": True,
        "diagnostic_timing_n": len(diagnostic_times),
        "samples_per_revolution": samples_per_rev,
        "matlab_parity": "PARTIAL: source conventions audited, helper and gold outputs unavailable",
        "label_status": "BLOCKED: verified development or blind fault labels unavailable",
        "calibration_status": "BLOCKED: no independently verified healthy calibration cohort",
        "group_interpretation": "A records 1–100, B records 101–200; descriptive index groups",
        "exclusions": {str(row["record_id"]): row["exclusion_reason"] for row in rows if row["exclusion_reason"]},
        "reference_overlap_checks": [name for name, _ in references],
        "candidate_arrays": f"fixed records {REPRESENTATIVE_IDS}; others reproducible from checkpoints and source data",
    })
    _plot_faults(rows, trace_arrays, figures)
    return rows


def _save_figure(fig, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def _plot_faults(rows: list[dict[str, object]], traces: dict[str, np.ndarray], figures: Path) -> None:
    import matplotlib.pyplot as plt

    names = tuple(dict.fromkeys(row["model"] for row in rows))
    fig, ax = plt.subplots(figsize=(9, max(3.6, 0.56 * len(names))), layout="constrained")
    for i, name in enumerate(names):
        for j, (group, color) in enumerate((("A", "#245d96"), ("B", "#9b3e20"))):
            values = [
                row["pmi_course"] for row in rows
                if row["model"] == name and row["group"] == group and row["eligible"]
                and np.isfinite(row.get("pmi_course", np.nan))
            ]
            if not values:
                continue
            jitter = np.random.default_rng(42 + i + j).uniform(-0.095, 0.095, len(values))
            y = i + (2 * j - 1) * 0.15
            ax.scatter(values, y + jitter, alpha=0.5, s=13, color=color,
                       label=f"Index group {group}" if i == 0 else None)
            ax.scatter(np.median(values), y, marker="|", s=180, linewidth=2, color=color)
    ax.set(yticks=np.arange(len(names)), yticklabels=names,
           xlabel="Instructor-literal PMI (dimensionless)", title="Development PMI by provisional index group")
    ax.grid(axis="x", alpha=0.18)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    _save_figure(fig, figures / "fault_pmi_groups")

    fig, ax = plt.subplots(figsize=(10, 4.4), layout="constrained")
    for i, name in enumerate(names):
        points = [row for row in rows if row["model"] == name and row["eligible"]]
        ax.plot([row["record_id"] for row in points],
                [row.get("pmi_course", np.nan) for row in points], lw=1, alpha=0.75, label=name)
    ax.set(xlabel="One-based record ID", ylabel="Instructor-literal PMI",
           title="Per-record angular-envelope periodicity")
    ax.grid(alpha=0.18)
    ax.legend()
    _save_figure(fig, figures / "fault_pmi_records")

    if "100:raw:theta_rev" not in traces or "101:raw:theta_rev" not in traces:
        return
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.3), layout="constrained")
    for row_i, index in enumerate((100, 101)):
        for name in names[:3]:
            key = f"{index}:{name}"
            if f"{key}:theta_rev" not in traces:
                continue
            theta = traces[f"{key}:theta_rev"]
            order = traces[f"{key}:order"]
            mask = theta < 2
            axes[row_i, 0].plot(theta[mask], traces[f"{key}:angular_signal"][mask], lw=0.75, label=name)
            axes[row_i, 1].plot(order, traces[f"{key}:vibration_spectrum"], lw=0.85, label=name)
            axes[row_i, 2].plot(order, traces[f"{key}:envelope_spectrum"], lw=0.85, label=name)
        axes[row_i, 0].set(xlabel="Revolutions", ylabel=f"Record {index} vibration")
        axes[row_i, 1].set(xlabel="Shaft order", xlim=(0, 550), ylabel="Vibration amplitude")
        axes[row_i, 2].set(xlabel="Shaft order", xlim=(0, 12), ylabel="Envelope amplitude")
        axes[row_i, 2].axvline(1, color="#404040", lw=0.8, ls="--")
    axes[0, 0].legend()
    _save_figure(fig, figures / "fault_representative_spectra")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), layout="constrained")
    for axis, metric, label in zip(axes, ("crest_factor", "pearson_kurtosis_biased"),
                                 ("Crest factor", "Pearson kurtosis")):
        for group, color in (("A", "#245d96"), ("B", "#9b3e20")):
            for model_index, model in enumerate(names):
                values = [row.get(metric, np.nan) for row in rows if row["model"] == model
                          and row["group"] == group and row["eligible"]]
                if values:
                    offset = -0.13 if group == "A" else 0.13
                    axis.scatter(np.full(len(values), model_index + offset), values, s=8,
                                 color=color, alpha=0.35, label=group if model_index == 0 else None)
        axis.set(ylabel=label, xticks=np.arange(len(names)), xticklabels=names)
        axis.tick_params(axis="x", labelrotation=35)
        axis.grid(axis="y", alpha=0.15)
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(title="Index group")
    _save_figure(fig, figures / "fault_time_indicators")


def _speed_perturbations(rpm: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    """Return prespecified archive-style conditions and additional sensor errors."""
    speed = {
        f"bias_{bias:+.3f}": rpm * (1 + bias)
        for bias in (-.020, -.010, -.005, .005, .010, .020)
    }
    white = np.random.default_rng(303).normal(size=len(rpm))
    smooth = signal.lfilter([.035], [1, -.965], white)
    smooth /= smooth.std()
    speed.update({f"jitter_{jitter:.3f}": rpm * (1 + jitter * smooth)
                  for jitter in (.001, .003, .006)})
    step = int(round(fs * 0.04))
    delayed = np.pad(rpm, (step, 0), mode="edge")[:len(rpm)]
    dropouts = rpm.copy()
    for onset in np.arange(int(fs), len(rpm), int(1.3 * fs)):
        dropouts[onset:onset + step] = rpm[onset - 1]
    speed.update({
        "quantized_1rpm": np.rint(rpm),
        "delay_40ms": delayed,
        "dropout_hold_40ms": dropouts,
        "jitter_white_0.003": rpm * (1 + .003 * white),
    })
    return speed


def _changed_record(record: Record, *, x=None, rpm=None) -> Record:
    fields = {}
    if x is not None:
        fields.update(x=x, xraw=None)
    if rpm is not None:
        fields["rpm"] = rpm
        # Preserve the stored scaling convention without fitting altered extrema.
        if record.speed_normalized is not None and record.rpm_min is not None and record.rpm_max is not None:
            spread = record.rpm_max - record.rpm_min
            if spread > 0:
                fields["speed_normalized"] = record.speed_normalized + (rpm - record.rpm) / spread
    return replace(record, **fields)


def _pulse(
    record: Record,
    carrier_hz: float,
    amplitude: float,
    phase_rev: float,
    decay_s: float,
    *,
    kind: str = "periodic",
    modulation: float = 0.0,
    duration_s: float = .025,
) -> np.ndarray:
    """Synthesize damped impulses at actual shaft crossings or random phases."""
    rev = cumulative_trapezoid(record.rpm / 60, record.t, initial=0)
    first_rev = int(np.floor(-phase_rev)) + 1
    last_rev = int(np.ceil(rev[-1] - phase_rev))
    crossing = np.arange(first_rev, last_rev) + phase_rev
    if kind == "aperiodic":
        crossing = np.arange(first_rev, last_rev) + np.random.default_rng(2026).uniform(
            0.02, 0.98, len(crossing)
        )
    crossing = crossing[(crossing >= 0) & (crossing < rev[-1])]
    if crossing.size == 0:
        return np.zeros_like(record.x)
    event_time = np.interp(crossing, rev, record.t)
    event_index = np.searchsorted(event_time, record.t, side="right") - 1
    last_event = np.maximum(event_index, 0)
    dt = np.maximum(record.t - event_time[last_event], 0)
    envelope = np.exp(-dt / decay_s) * (1 + modulation * np.sin(2 * np.pi * crossing[last_event] / 5))
    return np.where(
        (event_index >= 0) & (dt < duration_s), amplitude * envelope * np.sin(2 * np.pi * carrier_hz * dt), 0
    )


def evaluate_robustness(
    record: Record,
    run_dir: Path,
    predictors: Predictors,
    start: int = 1792,
    *,
    samples_per_rev: int = SAMPLES_PER_REVOLUTION,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Apply fixed conditions only after model fitting. The actual target stays measured."""
    run_dir = Path(run_dir)
    y, rpm, t = record.x[start:], record.rpm[start:], record.t[start:]
    perturbations = _speed_perturbations(record.rpm, record.fs)
    noise = np.random.default_rng(2026).normal(0, .05, len(record.x))
    raw_rows, stress, pulses, speed_subgroup, history_rows = [], [], [], [], []
    prediction_traces = {}
    paired = list(product((430, 1100), (.4, 1.25), (.19, .77), (.0028,)))
    paired.extend(((520, .65, .43, .004), (850, 1., .57, .0018),
                   (1250, .6, .11, .002), (300, 1.2, .91, .005)))
    cases = [(car, amp, phase, decay, "periodic", 0., "clean")
             for car, amp, phase, decay in paired]
    cases.extend(((760, .18, .31, .0038, "periodic", .35, "clean"),
                  (760, .18, .31, .0038, "periodic", .35, "bias_+0.010"),
                  (760, .65, .31, .0038, "periodic", .35, "jitter_0.003"),
                  (760, .18, .31, .0038, "aperiodic", 0., "clean"),
                  (760, .65, .31, .0038, "aperiodic", 0., "clean")))

    raw_pmi = diagnostics(y, rpm, t, samples_per_rev)[0]["pmi_course"]
    for case_id, (car, amp, phase, decay, kind, modulation, sensor) in enumerate(cases, 1):
        duration = .018 if case_id <= len(paired) else .025
        injection = _pulse(record, car, amp, phase, decay, kind=kind, modulation=modulation, duration_s=duration)
        raw_new = diagnostics(y + injection[start:], rpm, t, samples_per_rev)[0]["pmi_course"]
        raw_rows.append(
            {"case_id": case_id, "raw_pmi_clean": raw_pmi, "raw_pmi_injected": raw_new,
             "delta_raw_pmi": raw_new - raw_pmi, "pulse_kind": kind, "rpm_condition": sensor}
        )

    for name, predictor in predictors.items():
        base = _prediction(record, predictor, start)
        clean = y - base
        clean_mse = float(np.mean(np.square(clean)))
        clean_pmi = diagnostics(clean, rpm, t, samples_per_rev)[0]["pmi_course"]
        for condition, new_rpm in perturbations.items():
            perturbed = _changed_record(record, rpm=new_rpm)
            disturbed = y - _prediction(perturbed, predictor, start)
            for scope in ("predictor_only", "diagnostic_only", "both"):
                residual = clean if scope == "diagnostic_only" else disturbed
                angle_rpm = rpm if scope == "predictor_only" else new_rpm[start:]
                pmi = diagnostics(residual, angle_rpm, t, samples_per_rev)[0]["pmi_course"]
                mse = float(np.mean(np.square(residual)))
                stress.append({
                    "model": name, "condition": condition, "scope": scope,
                    "mse": mse, "mse_clean": clean_mse, "relative_mse": mse / clean_mse,
                    "pmi_course": pmi, "pmi_clean": clean_pmi, "delta_pmi": pmi - clean_pmi,
                })
        noise_record = _changed_record(record, x=record.x + noise)
        noisy_residual = y - _prediction(noise_record, predictor, start)
        noise_mse = float(np.mean(np.square(noisy_residual)))
        noise_pmi = diagnostics(noisy_residual, rpm, t, samples_per_rev)[0]["pmi_course"]
        stress.append({
            "model": name, "condition": "input_noise_sd_0.05", "scope": "predictor_only",
            "mse": noise_mse, "mse_clean": clean_mse, "relative_mse": noise_mse / clean_mse,
            "pmi_course": noise_pmi, "pmi_clean": clean_pmi, "delta_pmi": noise_pmi - clean_pmi,
        })
        rate = np.diff(record.rpm, prepend=record.rpm[0])[start:] * record.fs
        for label, mask in (("accelerating", rate > 0), ("decelerating", rate < 0),
                            ("steady", rate == 0)):
            if np.count_nonzero(mask):
                speed_subgroup.append({"model": name, "condition": label, "target_count": mask.sum(),
                                       "mse": float(np.mean(np.square(clean[mask])))})
        for anchor in (8000, 16000):
            stop = anchor + 800
            if stop > len(record.x):
                continue
            for history in (512, 1024, 1792):
                sl = slice(anchor - history, stop)
                reset = replace(
                    record, x=record.x[sl], rpm=record.rpm[sl], t=record.t[sl],
                    xraw=None, speed_normalized=(None if record.speed_normalized is None
                                                 else record.speed_normalized[sl]),
                )
                expected = base[anchor - start:stop - start]
                trial = np.asarray(predictor(reset))[history:]
                finite = np.isfinite(trial)
                valid_count = int(finite.sum()) if trial.shape == expected.shape else 0
                history_rows.append({
                    "model": name, "first_target": anchor, "requested_history": history,
                    "valid_target_count": valid_count, "required_target_count": len(expected),
                    "status": "ok" if valid_count == len(expected) else "insufficient_or_partial_history",
                    "max_prediction_delta": (float(np.max(np.abs(expected[finite] - trial[finite])))
                                             if valid_count else np.nan),
                    "mse_untrimmed": (float(np.mean(np.square(record.x[anchor:stop] - trial)))
                                      if valid_count == len(expected) else np.nan),
                })
        for case_id, (car, amp, phase, decay, kind, modulation, sensor) in enumerate(cases, 1):
            duration = .018 if case_id <= len(paired) else .025
            injection = _pulse(record, car, amp, phase, decay, kind=kind, modulation=modulation, duration_s=duration)
            rpm_input = perturbations.get(sensor)
            p_clean_matched = base if rpm_input is None else _prediction(
                _changed_record(record, rpm=rpm_input), predictor, start
            )
            disturbed_record = _changed_record(record, x=record.x + injection, rpm=rpm_input)
            p_injected = _prediction(disturbed_record, predictor, start)
            injected = injection[start:]
            original_error = y - p_clean_matched
            injected_error = y + injected - p_injected
            change = injected_error - original_error
            pulse_energy = float(np.dot(injected, injected))
            not_preserved = injected - change
            before = diagnostics(original_error, rpm, t, samples_per_rev)[0]["pmi_course"]
            after = diagnostics(injected_error, rpm, t, samples_per_rev)[0]["pmi_course"]
            pulses.append({
                "model": name, "case_id": case_id, "carrier_hz": car, "amplitude_std": amp,
                "phase_rev": phase, "decay_s": decay, "window_s": duration, "pulse_kind": kind,
                "amplitude_modulation": modulation, "rpm_condition": sensor,
                "distortion_energy_ratio": float(np.dot(not_preserved, not_preserved) / pulse_energy),
                "residual_energy_ratio": float(np.dot(change, change) / pulse_energy),
                "pulse_corr": float(np.corrcoef(change, injected)[0, 1]),
                "pmi_clean": before, "pmi_injected": after, "delta_pmi": after - before,
                "delta_raw_pmi": raw_rows[case_id - 1]["delta_raw_pmi"],
            })
            if case_id == len(paired) + 1:
                largest = int(np.argmax(np.abs(injected)))
                window = slice(max(0, largest - 80), min(len(t), largest + 140))
                prediction_traces[f"{name}:change"] = change[window]
                prediction_traces["pulse"] = injected[window]
                prediction_traces["elapsed_ms"] = (t[window] - t[largest]) * 1000

    _write_table(run_dir / "metrics" / "rpm_stress.csv", stress)
    _write_table(run_dir / "metrics" / "pulse_preservation.csv", pulses)
    _write_table(run_dir / "metrics" / "pulse_preservation_raw.csv", raw_rows)
    _write_table(run_dir / "metrics" / "diagnostic_speed_subgroups.csv", speed_subgroup)
    _write_table(run_dir / "metrics" / "diagnostic_history_stress.csv", history_rows)
    _write_table(run_dir / "metrics" / "robustness_summary.csv", _robustness_summary(stress, pulses))
    if stress:
        _plot_robustness(stress, pulses, prediction_traces, run_dir / "figures")
    return stress, pulses


def _robustness_summary(stress: list[dict], pulses: list[dict]) -> list[dict]:
    summary = []
    for model in dict.fromkeys(row["model"] for row in stress):
        rpm = [row for row in stress if row["model"] == model and row["scope"] == "predictor_only"
               and row["condition"] != "input_noise_sd_0.05"]
        periodic = [row for row in pulses if row["model"] == model and row["pulse_kind"] == "periodic"]
        aperiodic = [row for row in pulses if row["model"] == model and row["pulse_kind"] == "aperiodic"]
        summary.append({
            "model": model, "rpm_conditions": len(rpm), "mse_clean": rpm[0]["mse_clean"],
            "rpm_mse_mean": float(np.mean([row["mse"] for row in rpm])),
            "rpm_mse_worst": max(row["mse"] for row in rpm),
            "mse_worst_condition": max(rpm, key=lambda row: row["mse"])["condition"],
            "median_periodic_pulse_corr": float(np.median([row["pulse_corr"] for row in periodic])),
            "periodic_pmi_positive": sum(row["delta_pmi"] > 0 for row in periodic),
            "periodic_pmi_cases": len(periodic),
            "median_periodic_delta_pmi": float(np.median([row["delta_pmi"] for row in periodic])),
            "median_aperiodic_delta_pmi": float(np.median([row["delta_pmi"] for row in aperiodic])),
        })
    return summary


def _plot_robustness(stress, pulses, traces, figures: Path) -> None:
    import matplotlib.pyplot as plt

    figures.mkdir(parents=True, exist_ok=True)
    names = tuple(dict.fromkeys(row["model"] for row in stress))
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.7), layout="constrained")
    for name in names:
        bias = [row for row in stress if row["model"] == name and row["scope"] == "predictor_only"
                and row["condition"].startswith("bias_")]
        axes[0, 0].plot([100 * float(row["condition"][5:]) for row in bias],
                        [row["relative_mse"] for row in bias], marker="o", lw=1, label=name)
        pmi = [row for row in stress if row["model"] == name and row["scope"] == "diagnostic_only"
               and row["condition"].startswith("bias_")]
        axes[0, 1].plot([100 * float(row["condition"][5:]) for row in pmi],
                        [row["delta_pmi"] for row in pmi], marker="o", lw=1, label=name)
        periodic = [row for row in pulses if row["model"] == name and row["pulse_kind"] == "periodic"]
        axes[1, 0].scatter([row["pulse_corr"] for row in periodic],
                          [row["delta_pmi"] for row in periodic], alpha=.68, s=18, label=name)
        axes[1, 1].plot(traces["elapsed_ms"], traces[f"{name}:change"], alpha=.8, lw=1, label=name)
    axes[1, 1].plot(traces["elapsed_ms"], traces["pulse"], color="#101010", lw=.95,
                    ls="--", label="Injected pulse")
    axes[0, 0].set(xlabel="Predictor RPM bias (%)", ylabel="MSE / clean MSE")
    axes[0, 1].set(xlabel="Diagnostic RPM bias (%)", ylabel="Change in PMI")
    axes[1, 0].set(xlabel="Correlation with injected impulse", ylabel="Change in residual PMI")
    axes[1, 1].set(xlabel="Time from pulse peak (ms)", ylabel="Paired residual change")
    for ax in axes.flat:
        ax.grid(alpha=0.18)
    axes[0, 0].legend(frameon=False)
    axes[1, 1].legend(frameon=False)
    _save_figure(fig, figures / "rpm_stress_pulses")
