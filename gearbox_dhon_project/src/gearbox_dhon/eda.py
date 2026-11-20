"""Independent training-record physics description and guarded feature screens."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, stats
from numpy.lib.stride_tricks import sliding_window_view

from .data import Record, _write_csv


FEATURE_FOLDS = ((4800, 7200), (11200, 13600), (18400, 20800))
START = 1792
MESH_ORDER = 37
PINION_ORDER = 37 / 20
SAMPLES_PER_REV = 2048


def causal_turns(rpm: np.ndarray, fs: float) -> np.ndarray:
    """At index t integrate observed speed only through t-1."""
    return np.r_[0.0, np.cumsum(np.asarray(rpm[:-1]) / (60 * fs))]


def angular_grid(v: np.ndarray, turns: np.ndarray, samples_per_rev: int = SAMPLES_PER_REV):
    """Offline linear resampling on a uniform angle grid, excluding extrapolation."""
    step = 1 / samples_per_rev
    if not np.isfinite(turns).all() or len(v) != len(turns) or np.any(np.diff(turns) <= 0):
        raise ValueError("Angular interpolation needs finite, strictly increasing sample phase")
    grid = np.arange(np.ceil(turns[0] / step), np.floor(turns[-1] / step) + 1) * step
    return grid, np.interp(grid, turns, v)


def _save(figure, stem: Path):
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _moving_total(values: np.ndarray, stops: np.ndarray, window: int) -> np.ndarray:
    cumulative = np.r_[0, np.cumsum(values)]
    return cumulative[stops] - cumulative[stops - window]


def _quadrature(x, phase, target, window):
    """Complex-demodulate past samples then rotate the estimate to target phase."""
    summary = _moving_total(x * np.exp(-2j * np.pi * phase), target, window) / window
    transported = 2 * summary * np.exp(2j * np.pi * phase[target])
    return np.column_stack([transported.real, transported.imag])


def _guarded_screen(record: Record, turns: np.ndarray) -> list[dict]:
    x, rpm, fs = record.x, record.rpm, record.fs
    target = np.arange(START, len(x))
    lag = sliding_window_view(x, 24)[target - 24, ::-1].copy()
    z = (rpm[target - 1] - 145) / 25
    available_turns = turns[:-1]
    angles_mesh = turns[target] - 1 / MESH_ORDER
    angles_rev = turns[target] - 1
    # Clock advances less than a mesh period per sample, so both interpolation nodes are in the past.
    if (np.any(angles_rev < turns[target - START])
            or np.any(angles_mesh >= turns[target - 1])):
        raise ValueError("Guard too short or unresolvable mesh lag at observed speed")
    design = {
        "LPV-AR24": np.c_[lag, lag * z[:, None], lag * z[:, None] ** 2, z, z * z],
    }
    additions = {
        "speed acceleration": (rpm[target - 1] - rpm[target - 33]) * fs / (32 * 75),
        "causal RMS64": np.sqrt(_moving_total(x * x, target, 64) / 64),
        "mesh-angle delay": np.interp(angles_mesh, available_turns, x[:-1]),
        "shaft-angle delay": np.interp(angles_rev, available_turns, x[:-1]),
        "148 + 259 order quadrature": np.column_stack([
            _quadrature(x, turns * k, target, 128) for k in (148, 259)
        ]),
        "1100 Hz quadrature": _quadrature(x, np.arange(len(x)) * 1100 / fs, target, 64),
    }
    for label, columns in additions.items():
        design[f"LPV-AR24 + {label}"] = np.column_stack([design["LPV-AR24"], columns])
    rows = []
    for fold, (start, end) in enumerate(FEATURE_FOLDS, start=1):
        fit = (target < start - START) | (target >= end + START)
        score = (target >= start) & (target < end)
        assert not np.any(fit & score)
        for model, features in design.items():
            a, y = features[fit], x[target[fit]]
            center = a.mean(axis=0)
            scale = np.maximum(a.std(axis=0), 1e-8)
            a = (a - center) / scale
            beta = np.linalg.solve(a.T @ a + 10 * np.eye(a.shape[1]), a.T @ (y - y.mean()))
            pred = (features[score] - center) / scale @ beta + y.mean()
            mse = float(np.mean((x[target[score]] - pred) ** 2))
            reference = next((row["mse"] for row in rows if row["fold"] == fold), mse)
            rows.append({
                "fold": fold, "model": model, "mse": mse, "delta_vs_lpv": mse - reference,
                "row_count_fit": int(fit.sum()), "target_start": start, "target_end": end,
                "raw_support_guard_samples": START, "ridge_alpha": 10.0,
            })
    return rows


def _acf(v: np.ndarray, max_lag: int) -> np.ndarray:
    centered = v - v.mean()
    padded = signal.correlate(centered, centered, mode="full", method="fft")
    center = len(v) - 1
    return padded[center : center + 1 + max_lag] / padded[center]


def _conditional_blocks(record: Record):
    n = int(record.fs / 4)
    x, rpm = record.x, record.rpm
    freq, psd = signal.periodogram(x[: len(x) // n * n].reshape(-1, n),
                                    record.fs, window="hann", axis=-1)
    bands = {"0_90": (0, 90), "90_250": (90, 250), "250_550": (250, 550),
             "550_1000": (550, 1000), "1000_1200": (1000, 1200), "1200_1550": (1200, 1550)}
    rows, acfs = [], []
    for i in range(len(x) // n):
        a, b = i * n, (i + 1) * n
        v, speed = x[a:b], rpm[a:b]
        acceleration = (speed[-1] - speed[0]) * record.fs / (n - 1)
        acfs.append(_acf(v, min(200, n - 1)))
        row = {"start": a, "stop": b, "rpm_mean": float(np.mean(speed)),
               "rpm_per_s": float(acceleration), "x_mean": float(np.mean(v)),
               "rms": float(np.sqrt(np.mean(v * v))),
               "excess_kurtosis": float(stats.kurtosis(v)), "peak_abs": float(np.max(np.abs(v)))}
        for label, (low, high) in bands.items():
            row[f"power_{label}_hz"] = float(psd[i, (freq >= low) & (freq < high)].sum() * (freq[1] - freq[0]))
        rows.append(row)
    return rows, np.asarray(acfs)


def _ridge_rows(record: Record, freq, stime, psd):
    center_rpm = np.interp(stime, record.t - record.t[0], record.rpm)
    mesh_hz = MESH_ORDER * center_rpm / 60
    rows = []
    for harmonic in (1, 2, 4, 7, 10, 12, 14):
        theory = harmonic * mesh_hz
        candidate = np.abs(freq[:, None] - theory) < 0.38 * mesh_hz
        peak = np.argmax(np.where(candidate, psd, -np.inf), axis=0)
        # Quadratic peak interpolation on log power; clip shifts to the selected bin neighborhood.
        li = np.maximum(peak - 1, 0)
        hi = np.minimum(peak + 1, len(freq) - 1)
        col = np.arange(len(stime))
        left, mid, right = (np.log(np.maximum(psd[k, col], 1e-18)) for k in (li, peak, hi))
        div = 2 * (left - 2 * mid + right)
        shift = np.divide(left - right, div, out=np.zeros_like(div), where=div != 0)
        actual = freq[peak] + np.clip(shift, -.5, .5) * (freq[1] - freq[0])
        err = actual - theory
        for lo, high in ((120, 140), (140, 155), (155, 170), (120, 170)):
            keep = (center_rpm >= lo) & (center_rpm < high)
            if keep.sum() < 3:
                continue
            rows.append({
                "harmonic": harmonic, "order": harmonic * MESH_ORDER,
                "rpm_low_inclusive": lo, "rpm_high_exclusive": high, "frames": int(keep.sum()),
                "median_abs_hz_error": np.median(np.abs(err[keep])),
                "median_signed_hz_error": np.median(err[keep]),
                "ridge_theory_corr": np.corrcoef(actual[keep], theory[keep])[0, 1],
                "mean_theory_hz": np.mean(theory[keep]),
            })
    return rows


def _turn_bootstrap(amplitudes, mean_rpm, rng, samples=500):
    n = len(amplitudes)
    starts = rng.integers(0, n - 2, size=(samples, int(np.ceil(n / 3))))
    indices = (starts[:, :, None] + np.arange(3)).reshape(samples, -1)[:, :n]
    amp = amplitudes[indices]
    rpm = mean_rpm[indices]
    amp -= amp.mean(axis=1, keepdims=True)
    rpm -= rpm.mean(axis=1, keepdims=True)
    corr = (amp * rpm).sum(axis=1) / np.sqrt((amp ** 2).sum(axis=1) * (rpm ** 2).sum(axis=1))
    return np.quantile(corr[np.isfinite(corr)], [.025, .975])


def run_eda(train: Record, out: Path) -> dict:
    """Build analysis only from the designated training record."""
    matplotlib.rcParams.update({"figure.facecolor": "white", "font.size": 10,
                                "axes.spines.top": False, "axes.spines.right": False})
    out = Path(out)
    tables, figures, folder = [out / name for name in ("tables", "figures", "eda")]
    for target in (tables, figures, folder):
        target.mkdir(exist_ok=True, parents=True)
    x, fs, rpm = train.x, train.fs, train.rpm
    high, low = rpm.max(), rpm.min()
    time = train.t - train.t[0]
    turns = causal_turns(rpm, fs)
    detrended = x - x.mean()
    nyquist_orders = fs / 2 / (high / 60)
    heading = "Designated training record"
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(9, 5.7))
    ax[0].plot(time, rpm, color="#096b72", lw=1.6)
    ax[0].set(ylabel="Speed [rpm]", title=heading)
    ax[1].plot(time, x, color="#304763", lw=.42)
    ax[1].set(ylabel="Vibration [standardized]")
    stops = np.arange(320, len(x))
    mean = _moving_total(x, stops, 320) / 320
    ms = _moving_total(x * x, stops, 320) / 320
    ax[2].plot(time[stops], np.sqrt(ms), label="RMS")
    ax[2].plot(time[stops], mean, label="Mean")
    ax[2].plot(time[stops], ms - mean * mean, label="Variance")
    ax[2].set(ylabel="Moment", xlabel="Elapsed time [s]")
    ax[2].legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    _save(fig, figures / "eda_speed")

    freq, stime, psd = signal.spectrogram(
        x, fs=fs, window="hann", nperseg=1024, noverlap=896, nfft=4096, scaling="density",
    )
    db = 10 * np.log10(np.maximum(psd, 1e-15))
    fig, ax = plt.subplots(figsize=(9, 4.7))
    im = ax.pcolormesh(stime, freq, db, shading="auto", cmap="magma", rasterized=True,
                        vmin=np.quantile(db, .25), vmax=np.quantile(db, .998))
    for k in (1, 2, 4, 7, 10, 12, 14):
        ax.plot(time, k * MESH_ORDER * rpm / 60, color="#7ee5e6", lw=.75,
                label="Expected mesh tracks" if k == 1 else "")
    ax.axhspan(1000, 1200, fc="#ffffd9", alpha=.09, label="1.0–1.2 kHz band")
    ax.set(xlabel="Elapsed time [s]", ylabel="Frequency [Hz]", ylim=(0, 1600),
           title="Training spectrogram with expected mesh trajectories")
    ax.legend(loc="lower right", facecolor="#fff9f2")
    fig.colorbar(im, ax=ax, label="PSD [dB re standardized²/Hz]")
    _save(fig, figures / "eda_spectrogram")
    ridges = _ridge_rows(train, freq, stime, psd)
    _write_csv(tables / "eda_mesh_ridges.csv", ridges)

    angle, xa = angular_grid(x, turns)
    os = SAMPLES_PER_REV
    order, power = signal.welch(xa, os, nperseg=8 * os, noverlap=6 * os, window="hann")
    ef = np.abs(signal.hilbert(xa))
    eo, ep = signal.welch(ef, os, nperseg=8 * os, noverlap=6 * os, window="hann")
    wfreq, wp = signal.welch(x, fs, nperseg=2048, noverlap=1536, window="hann")
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.6))
    ax[0].semilogy(wfreq, wp, color="#254270")
    ax[0].set(xlabel="Frequency [Hz]", ylabel="PSD [std.²/Hz]", title="Training time spectrum")
    ax[1].semilogy(order, power, color="#096b72")
    ax[1].vlines(np.arange(1, 1 + int(nyquist_orders // MESH_ORDER)) * MESH_ORDER,
                  1e-6, power.max(), colors="#de8637", alpha=.49, lw=.8)
    ax[1].set(xlim=(0, nyquist_orders), ylim=(1e-6, None),
              xlabel="Input-shaft order [cycles/revolution]", ylabel="PSD [std.²/order]",
              title="Angle spectrum with expected mesh orders")
    fig.tight_layout()
    _save(fig, figures / "eda_orders")

    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.semilogy(eo, ep, color="#662f66")
    ax.axvline(1, color="#de8637", label="37-tooth shaft order")
    ax.axvline(PINION_ORDER, color="#096b72", label="20-tooth pinion order")
    ax.set(xlim=(0, 15), xlabel="Input-shaft order", ylabel="Envelope PSD [std.²/order]",
           title="Offline Hilbert envelope of training angle signal")
    ax.legend(frameon=False)
    _save(fig, figures / "eda_envelope")

    contrasts = []
    complete = len(xa) // os
    chunks = xa[: complete * os].reshape(-1, os)
    rpm_chunks = np.interp(angle[: complete * os], turns, rpm).reshape(-1, os).mean(axis=1)
    rng = np.random.default_rng(7780)
    turn_rows = []
    for k in np.arange(1, 1 + int(nyquist_orders // MESH_ORDER)):
        carrier = k * MESH_ORDER
        local_peak = float(power[np.abs(order - carrier) <= .25].max())
        background = float(np.median(power[(np.abs(order - carrier) >= 3) & (np.abs(order - carrier) <= 8)]))
        moments = 2 * np.fft.rfft(chunks, axis=1)[:, carrier] / os
        amps = np.abs(moments)
        amp_corr = float(np.corrcoef(rpm_chunks, amps)[0, 1])
        ci = _turn_bootstrap(amps, rpm_chunks, rng)
        contrasts.append({
            "harmonic": k, "order": carrier, "peak_over_local_median": local_peak / background,
            "fraction_power_in_order_pm_0_5": (
                power[np.abs(order - carrier) <= .5].sum() / power[order <= nyquist_orders].sum()
            ),
            "order_minus_1_density": power[np.argmin(np.abs(order - (carrier - 1)))],
            "order_plus_1_density": power[np.argmin(np.abs(order - (carrier + 1)))],
            "order_minus_pinion_density": power[np.argmin(np.abs(order - (carrier - PINION_ORDER)))],
            "order_plus_pinion_density": power[np.argmin(np.abs(order - (carrier + PINION_ORDER)))],
            "n_complete_turns": len(amps), "turn_amplitude_rpm_corr": amp_corr,
            "turn_amplitude_rpm_corr_bootstrap_low": ci[0],
            "turn_amplitude_rpm_corr_bootstrap_high": ci[1],
            "turn_phase_coherence": np.abs(np.mean(moments / np.maximum(amps, 1e-15))),
        })
        for i, (amp, coefficient, speed) in enumerate(zip(amps, moments, rpm_chunks, strict=True)):
            turn_rows.append({"order": carrier, "turn_zero_based": i, "mean_rpm": speed,
                              "amplitude": amp, "phase_rad": np.angle(coefficient)})
    _write_csv(tables / "eda_order_evidence.csv", contrasts)
    _write_csv(tables / "eda_turn_harmonics.csv", turn_rows)

    local, by_block_acf = _conditional_blocks(train)
    _write_csv(tables / "eda_blocks.csv", local)
    rise_fall = []
    conditional = []
    for lo, hi in ((120, 135), (135, 145), (145, 155), (155, 170)):
        both = [i for i, row in enumerate(local) if lo <= row["rpm_mean"] < hi]
        for rising in (False, True):
            rows = [local[i] for i in both if (local[i]["rpm_per_s"] > 0) == rising]
            if rows:
                rise_fall.append({"rpm_low": lo, "rpm_high": hi, "rising": rising, "blocks": len(rows),
                                  "mean_rms": np.mean([r["rms"] for r in rows]),
                                  "mean_rpm": np.mean([r["rpm_mean"] for r in rows])})
        if both:
            conditional.append((lo, hi, by_block_acf[both].mean(axis=0)))
    _write_csv(tables / "eda_rise_fall.csv", rise_fall)
    a_acf = _acf(xa, 2 * os)
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.3))
    for lo, hi, cc in conditional:
        ax[0].plot(np.arange(len(cc)) / fs * 1000, cc, label=f"{lo}–{hi} rpm", lw=1)
    ax[0].set(xlabel="Time lag [ms]", ylabel="ACF", title="Non-overlapping quarter-second blocks")
    ax[0].legend(frameon=False, ncol=4)
    ax[1].plot(np.arange(len(a_acf)) / os, a_acf, color="#096b72", lw=.8)
    for period in (1 / MESH_ORDER, 1):
        ax[1].axvline(period, color="#de8637", lw=.8)
    ax[1].set(xlabel="Angular lag [input turns]", ylabel="ACF")
    fig.tight_layout()
    _save(fig, figures / "eda_explanatory")

    screen = _guarded_screen(train, turns)
    _write_csv(tables / "eda_guarded_feature_screen.csv", screen)
    grouped = {}
    for row in screen:
        grouped.setdefault(row["model"], []).append(row["mse"])
    mean_mse = {name: float(np.mean(errors)) for name, errors in grouped.items()}
    ridge_148 = next(r for r in ridges if r["order"] == 148 and r["rpm_low_inclusive"] == 120
                     and r["rpm_high_exclusive"] == 170)
    order_148 = next(r for r in contrasts if r["order"] == 148)
    summary = {
        "dataset": train.dataset, "samples": len(x), "fs_hz": fs, "duration_s": len(x) / fs,
        "rpm_min": low, "rpm_max": high, "integrated_input_turns": turns[-1],
        "mesh_hz_min": MESH_ORDER * low / 60, "mesh_hz_max": MESH_ORDER * high / 60,
        "samples_per_mesh_min": fs / (MESH_ORDER * high / 60),
        "samples_per_mesh_max": fs / (MESH_ORDER * low / 60),
        "samples_per_input_turn_min": fs / (high / 60),
        "samples_per_input_turn_max": fs / (low / 60),
        "max_order_at_highest_speed": nyquist_orders,
        "max_mesh_harmonic_at_highest_speed": int(nyquist_orders // MESH_ORDER),
        "robust_x_quantiles": np.quantile(x, [.001, .01, .5, .99, .999]).tolist(),
        "x_excess_kurtosis": stats.kurtosis(x).item(),
        "exact_extrema_counts": [int(np.sum(x == x.min())), int(np.sum(x == x.max()))],
        "rms_vs_rpm_block_corr": np.corrcoef([r["rms"] for r in local], [r["rpm_mean"] for r in local])[0, 1],
        "one_mesh_angle_acf": a_acf[round(os / MESH_ORDER)],
        "one_shaft_angle_acf": a_acf[os],
        "time_acf_lag1": _acf(detrended, 1)[1],
        "ridge_148_median_abs_error_hz": ridge_148["median_abs_hz_error"],
        "ridge_148_r": ridge_148["ridge_theory_corr"],
        "order_148_peak_over_local_median": order_148["peak_over_local_median"],
        "band_1000_1200_fraction_of_welch_power": (
            wp[(wfreq >= 1000) & (wfreq < 1200)].sum() / wp.sum()
        ),
        "guarded_screen_mean_mse": mean_mse,
        "angular_resampling": "offline linear interpolation at 2048 points/revolution; report below minimum temporal-Nyquist order",
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_feature_registry(folder, mean_mse)
    return summary


def _write_feature_registry(folder: Path, mean_mse: dict):
    registry = [
        ("TD24", "x[t-j], j=1..L", "standardized vibration", "observed lags", 24,
         "L=24,64,128,256", "recurrence; speed-independent control", "collinearity", "LPV-AR24 comparator",
         "include control"),
        ("LPV", "x[t-j]z[t-1]^k, k=0..2", "dimensionless", "fixed rpm scaling then lagged products", 24,
         "degree 0..3", "different gains across input speed", "confounded load, extrapolation", "LPV-AR24 comparator",
         "include control"),
        ("ANG_MESH", "x(u[t]-1/37)", "standardized vibration", "past-only angle linear interpolation", 43,
         "delay 1,2,4 mesh periods", "period shrinks as rpm rises", "predictable pulses can be absorbed",
         "LPV-AR24 + mesh-angle delay", "test"),
        ("ANG_REV", "x(u[t]-1)", "standardized vibration", "past-only angle linear interpolation", 1549,
         "on/off", "shaft recurrence", "risk to crack-periodic residual signature",
         "LPV-AR24 + shaft-angle delay", "test with pulse preservation"),
        ("QUAD_ORDER", "history of x exp(-i 2 pi k u), transported to u[t]", "standardized vibration",
         "anchor input order and integrate lagged rpm", 128, "k=37,74,111,148,222,259,370; bounded detuning",
         "vary phase frequency with rpm", "harmonics overlap, same-order fault signature",
         "LPV-AR24 + 148 + 259 order quadrature", "test"),
        ("QUAD_HZ", "history of x exp(-i 2 pi f t/fs)", "standardized vibration",
         "known elapsed sample clock", 64, "f=900..1250 Hz; keep below 1600 Hz",
         "separate fixed-Hz and order hypotheses", "broad spectral power is not a modal test",
         "LPV-AR24 + 1100 Hz quadrature", "test"),
        ("ACC", "(n[t-1]-n[t-33])*fs/32", "rpm/s", "trailing speed difference", 33,
         "span 1..128", "condition on rise/fall", "small support at matched speed; noise",
         "LPV-AR24 + speed acceleration", "test"),
        ("ENERGY", "sqrt(mean(x[t-W:t]^2))", "standardized vibration", "trailing energy", 64,
         "W=32..320", "operating amplitude", "adaptation can absorb faults",
         "LPV-AR24 + causal RMS64", "test"),
        ("INNOV", "x[t-j]-first_stage_prediction[t-j]", "standardized vibration",
         "observed residual after earlier target; fit stage inside fold", None,
         "stage-1 horizon plus residual lags", "colored remainder", "cross-fit mismatch and pulse absorption",
         "", "compare matched adapter"),
        ("SIDEBAND", "orders k*37 +/- 1 or +/- 37/20", "cycles/input turn",
         "same phase convention as QUAD_ORDER", 128, "off then grouped test",
         "amplitude modulation", "healthy sidebands; unresolved short windows",
         "", "deprioritize until incremental gain"),
    ]
    rows = []
    yaml = ["basis: designated-training-only EDA", "features:"]
    for label, equation, unit, causal, history, ranges, benefit, risk, comparison, decision in registry:
        evidence = mean_mse.get(comparison)
        rows.append({"feature": label, "equation": equation, "units": unit, "causal_construction": causal,
                     "screen_max_history_samples": history, "candidate_range": ranges,
                     "benefit_hypothesis": benefit, "risks": risk, "guarded_screen_mse": evidence,
                     "decision": decision})
        yaml.extend([f"  {label}:", f"    candidate_range: {json.dumps(ranges)}",
                     f"    decision: {json.dumps(decision)}"])
    _write_csv(folder / "feature_registry.csv", rows)
    (folder / "feature_hypotheses.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
