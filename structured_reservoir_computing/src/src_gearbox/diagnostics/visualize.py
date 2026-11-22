"""Rebuild fault figures directly from scored candidate records and paired scenarios."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABEL = {
    "raw": "Raw vibration",
    "primary_pi_src": "Selected PI-SRC",
    "primary_pi_src_same_weights_no_adapter": "Selected PI-SRC, same weights without adapter",
    "pi_src_optimized": "PI-SRC optimized",
    "pi_src_fixed": "PI-SRC fixed",
    "pi_src_adapter_off": "PI-SRC without adapter",
    "src_generic": "Generic SRC",
    "src_polynomial": "Polynomial SRC",
    "sparse_lpv_arma": "Sparse LPV–ARMA",
    "archived_src_corrected": "Archived SRC, correction",
    "cnn": "CNN",
    "ridge": "Ridge",
}


def _label(name):
    return LABEL.get(name, name.replace("_", " "))


def _save(fig, out, stem):
    fig.savefig(out / f"{stem}.png", dpi=300)
    fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)


def _distributions(results, out):
    source = results / "fault_per_record.csv"
    if not source.is_file():
        return
    frame = pd.read_csv(source)
    eligible = frame.loc[frame.eligible & frame.status.eq("ok")]
    # Same eligible records for every box and dot distribution.
    matrix = eligible.pivot(index=["record_id","group"], columns="model",
                            values="lecturer_compatible").dropna().reset_index()
    names = list(frame.model.drop_duplicates())
    groups = ("A", "B")
    if matrix.empty or not set(groups).issubset(matrix.group.unique()):
        return
    height = max(3.6, 1.3 + .64 * len(names))
    fig, ax = plt.subplots(figsize=(8.7, height), layout="constrained")
    for name_index, name in enumerate(names):
        for group_index, (group, color) in enumerate(zip(groups, ("#275f87", "#a45231"))):
            positions = matrix.loc[matrix.group.eq(group), name].to_numpy()
            offset = -.15 if group_index == 0 else .15
            pos = name_index + offset
            jitter = np.random.default_rng(77815 + name_index).uniform(-.075,.075,len(positions))
            ax.scatter(positions, pos + jitter, s=13, alpha=.34, color=color,
                       label=f"Provisional group {group}" if name_index == 0 else None)
            ax.scatter([np.median(positions)], [pos], s=160, marker="|", linewidth=2, color=color)
    ax.set(yticks=np.arange(len(names)), yticklabels=[_label(n) for n in names],
           xlabel="Instructor-literal PMI at 2,047 angular samples",
           title=f"Residual periodicity by record index ({len(matrix)} matched eligible records)")
    ax.invert_yaxis()
    ax.legend(loc="best", frameon=False)
    ax.grid(axis="x", alpha=.20)
    _save(fig, out, "fault_group_distribution")


def _representative(results, out, primary):
    source = results / "fault_traces.npz"
    if not source.is_file():
        return
    colors = {"raw": "#495358", primary: "#236591"}
    with np.load(source, allow_pickle=False) as arrays:
        if f"101:{primary}:vibration" not in arrays:
            return
        fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0), layout="constrained", sharex="col")
        for rep, panel in zip((100, 101), axes, strict=True):
            limit = 0.
            order = arrays[f"{rep}:raw:order"]
            for name in ("raw", primary):
                theta = arrays[f"{rep}:{name}:angle_rev"]
                window = theta < 2
                vibration = arrays[f"{rep}:{name}:vibration"][window]
                limit = max(limit, np.max(np.abs(vibration)))
                panel[0].plot(theta[window], vibration, label=_label(name), color=colors[name],
                              lw=.65, alpha=.65 if name == "raw" else .9)
                spectrum = arrays[f"{rep}:{name}:envelope_spectrum"]
                mask = (order > .2) & (order <= 10)
                panel[1].plot(order[mask], spectrum[mask], label=_label(name), color=colors[name], lw=1.05)
            panel[0].set(ylim=(-limit*1.06, limit*1.06), ylabel=f"Record {rep}\nStandardized vibration")
            panel[1].set(ylabel="Envelope amplitude")
            panel[1].axvline(1, linestyle="--", color="#a45231", lw=.95)
            for axis in panel:
                axis.grid(alpha=.14)
        axes[0,0].set_title("Angular waveforms")
        axes[0,1].set_title("Hilbert-envelope spectra")
        axes[0,0].legend(frameon=False)
        axes[1,0].set_xlabel("Shaft angle (revolutions)")
        axes[1,1].set_xlabel("Shaft order (cycles per revolution)")
    _save(fig, out, "representative_order")


def _stress(results, out):
    source = results / "rpm_stress_summary.csv"
    if not source.is_file():
        return
    frame = pd.read_csv(source)
    if frame.empty:
        return
    frame = frame.loc[~frame.model.eq("pi_src_optimized")]
    positions = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(8.2, max(3.0, 1.25 + .46*len(frame))), layout="constrained")
    ax.hlines(positions, frame.clean_mse, frame.worst_stress_mse, color="#b1bec8", linewidth=1.3)
    ax.scatter(frame.clean_mse, positions, label="Clean", color="#235f86")
    ax.scatter(frame.worst_stress_mse, positions, label="Worst documented RPM case",
               color="#a65331", marker="D")
    ax.set(yticks=positions, yticklabels=[_label(k) for k in frame.model],
           xlabel="Development MSE (standardized vibration squared)", xscale="log")
    ax.invert_yaxis()
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=.18)
    _save(fig, out, "sensor_stress")


def _sensor_pmi(results, out, primary):
    source = results / "rpm_stress.csv"
    if not source.is_file():
        return
    frame = pd.read_csv(source)
    picked = frame.loc[frame.model.eq(primary) & frame.status.eq("ok")
                       & ~frame.condition.isin(("clean","vibration_noise_sd_0p05"))]
    if picked.empty:
        return
    pivot = picked.pivot(index="condition", columns="scope", values="delta_pmi")
    names = {
        "bias_minus_2pct": "−2% sensor bias", "bias_plus_2pct": "+2% sensor bias",
        "correlated_jitter_0p6pct": "0.6% smooth jitter", "independent_jitter_0p3pct": "0.3% white jitter",
        "delay_40ms": "40 ms delay", "quantized_1rpm": "1 rpm quantization",
        "dropout_hold_75ms": "75 ms hold-last-value", "smooth_profile_warp": "Smooth profile warp",
    }
    pivot = pivot.reindex([k for k in names if k in pivot.index])
    y = np.arange(len(pivot))
    fig, axis = plt.subplots(figsize=(8.2, 4.6), layout="constrained")
    for i, (column, label, color) in enumerate(
        (("predictor","Predictor RPM","#275f87"), ("diagnostics","Diagnostic angle","#a45231"),
         ("both","Both","#437556"))
    ):
        axis.barh(y + (i-1)*.22, pivot[column], height=.18, label=label, color=color, alpha=.85)
    axis.axvline(0,color="#354452",lw=.75)
    axis.invert_yaxis()
    axis.set(yticks=y, yticklabels=[names[k] for k in pivot.index],
             xlabel="Change in validation PMI from the unperturbed case",
             title=f"Diagnostic effect of RPM error for {_label(primary)}")
    axis.legend(frameon=False)
    axis.grid(axis="x",alpha=.16)
    _save(fig,out,"sensor_pmi")


def _pulses(results, out, primary):
    source = results / "pulse_preservation.csv"
    if not source.is_file():
        return
    frame = pd.read_csv(source).query("status == 'ok'")
    if frame.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), layout="constrained")
    mats = frame.loc[frame.kind.eq("periodic")]
    for name, rows in mats.groupby("model", sort=False):
        important = name == primary
        axes[0].scatter(rows.residual_pulse_correlation, rows.delta_pmi, label=_label(name),
                        s=27 if important else 19, alpha=.76 if important else .56)
    axes[0].set(xlabel="Correlation with injected waveform", ylabel="Paired PMI change",
                title="Periodic synthetic pulses")
    axes[0].axhline(0, color="#59636b", lw=.8, ls="--")
    axes[0].legend(frameon=False, fontsize=7.1)
    trace = results / "pulse_example.npz"
    if trace.is_file():
        with np.load(trace, allow_pickle=False) as traces:
            time = traces["time_ms"]
            axes[1].plot(time, traces["injection"], label="Injected pulse",
                         color="#30363c", linewidth=1.15, linestyle="--")
            for key in (f"{primary}:response", "sparse_lpv_arma:response",
                        "archived_src_corrected:response"):
                if key in traces:
                    axes[1].plot(time, traces[key], label=_label(key.split(":")[0]), linewidth=.85)
        axes[1].set(xlabel="Elapsed time from pulse peak (ms)", ylabel="Standardized vibration",
                    title="Paired residual increment, case 1")
        axes[1].legend(frameon=False, fontsize=7.1)
    for axis in axes:
        axis.grid(alpha=.15)
    _save(fig, out, "pulse_tradeoff")


def plot(results: Path, out: Path, *, primary="primary_pi_src") -> None:
    results, out = Path(results), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    _distributions(results, out)
    _representative(results, out, primary)
    _stress(results, out)
    _sensor_pmi(results, out, primary)
    _pulses(results, out, primary)
