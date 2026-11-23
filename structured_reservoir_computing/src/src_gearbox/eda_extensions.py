"""Training-only diagnostics for the SRC feature hypotheses."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import linalg, signal, stats

from .data.io_matlab import Record, _write_csv
from .eda import FEATURE_FOLDS, START, _acf, _save, angular_grid, causal_turns


def finite_quadrature(x, phase, tau, length):
    """Past-only finite damped quadrature; phase in cycles and tau in samples."""
    rho = np.exp(-1 / tau)
    carrier = np.exp(2j * np.pi * np.remainder(phase, 1))
    demodulated = x * np.conj(carrier)
    taps = rho ** np.arange(length)
    filtered = signal.lfilter([1.0], [1.0, -rho], demodulated)
    filtered[length:] = filtered[length:] - rho**length * filtered[:-length]
    return 2 * np.r_[0, filtered[:-1]] * carrier / taps.sum()


def phase_fixtures() -> dict:
    """Validate angle scale, quadrature sign and causal cutoff with known signals."""
    fs = 3200
    t = np.arange(24000) / fs
    truth = 148
    outcomes = {}
    for label, s in (
        ("constant_speed", np.full(len(t), 144.)),
        ("speed_chirp", 145 + 20 * np.sin(2 * np.pi * t / 7.5)),
    ):
        u = causal_turns(s, fs)
        expected = np.exp(2j * np.pi * truth * u)
        x = expected.real
        grid, angular = angular_grid(x, u)
        order, p = signal.periodogram(angular, 2048, window="hann")
        peak = float(order[np.argmax(p)])
        estimate = finite_quadrature(x, truth * u, 128, 768)[768:]
        error = float(np.sqrt(np.mean(np.abs(estimate - expected[768:]) ** 2)))
        cut = len(x) // 2
        replacement = s.copy()
        replacement[cut:] += 9
        causal = np.array_equal(causal_turns(replacement, fs)[:cut+1], u[:cut+1])
        replaced_x = x.copy()
        replaced_x[cut:] *= -1
        second = finite_quadrature(replaced_x, truth * u, 128, 768)
        cutoff_error = float(np.max(np.abs(second[:cut+1] - finite_quadrature(x, truth * u, 128, 768)[:cut+1])))
        outcomes[label] = {
            "design_order": truth, "estimated_order": peak,
            "absolute_order_error": abs(peak - truth), "angular_points": len(grid),
            "complex_quadrature_rmse": error, "phase_cutoff_exact": causal,
            "quadrature_suffix_intervention_max_abs_change": cutoff_error,
            "passed": abs(peak-truth) < .08 and error < .03 and causal and cutoff_error < 1e-12,
        }
    assert all(run["passed"] for run in outcomes.values())
    return outcomes


def _frozen_banks(train: Record, t: np.ndarray, tau):
    """Build focused and broader banks fixed to shaft order plus Hz controls."""
    u = causal_turns(train.rpm, train.fs)
    time = np.arange(len(train.x)) / train.fs
    slots = {}
    clocks = {**{f"o{k}": u*k for k in (1, 37, 74, 147, 148, 149, 222, 259, 444)},
              **{f"f{k}": time*k for k in (1080, 1125)}}
    for carrier, phase in clocks.items():
        h = finite_quadrature(train.x, phase, tau, 512)[t]
        slots[carrier] = np.c_[h.real, h.imag]
    return slots


def _columns(bank, keys):
    return np.column_stack([bank[k] for k in keys])


def _feature_trials(train: Record) -> list[dict]:
    x, rpm, fs = train.x, train.rpm, train.fs
    t = np.arange(START, len(x))
    lag = sliding_window_view(x, 24)[t-24, ::-1]
    speed = (rpm[t-1] - 145) / 25
    past_energy = np.mean(lag**2, axis=1)
    rms = np.sqrt(past_energy)
    base = np.c_[lag, lag*speed[:, None], lag*speed[:, None]**2, speed, speed**2]
    dominant = ("o148", "o259")
    working = {}
    for tau in (8, 32, 128):
        bank = _frozen_banks(train, t, tau)
        working[f"two_orders_tau_{tau}"] = np.c_[base, _columns(bank, dominant)]
    bank = _frozen_banks(train, t, 32)
    focus = _columns(bank, dominant)
    geometry = _columns(bank, ("o37", "o74", "o148", "o222", "o259", "o444"))
    working.update({
        "lpv_ar24_control": base,
        "geometry_six_tau_32": np.c_[base, geometry],
        "fixed_hz_pair_tau_32": np.c_[base, _columns(bank, ("f1080", "f1125"))],
        "two_orders_plus_sidebands": np.c_[base, focus, _columns(bank, ("o1", "o147", "o149"))],
        "two_orders_structured_degree_2": np.c_[base, focus, focus*speed[:, None], focus*rms[:, None],
                                                  lag[:, :6]*rms[:, None], lag[:, :6]**2],
        "two_orders_structured_degree_3": np.c_[base, focus, focus*speed[:, None], focus*rms[:, None],
                                                  lag[:, :6]*rms[:, None], lag[:, :6]**2,
                                                  lag[:, :6]**3, focus * speed[:, None]**2],
    })
    for label, lag_samples in (("rpm_lag_32", 32), ("rpm_lead_32_descriptive", -32)):
        # The advance is a deliberate offline diagnostic and is not a candidate for a causal model.
        sample = np.clip(np.arange(len(rpm))-lag_samples, 0, len(rpm)-1)
        u = causal_turns(rpm[sample], fs)
        psi = [finite_quadrature(x, u*k, 32, 512)[t] for k in (148, 259)]
        working[label] = np.c_[base, *(np.c_[p.real, p.imag] for p in psi)]
    rows = []
    for fold, (a, b) in enumerate(FEATURE_FOLDS, start=1):
        train_rows = (t < a-START) | (t >= b+START)
        score_rows = (a <= t) & (t < b)
        y_fit = x[t[train_rows]]
        y_true = x[t[score_rows]]
        for name, features in working.items():
            train_features = features[train_rows]
            center, scale = train_features.mean(axis=0), train_features.std(axis=0)
            scale = np.maximum(scale, 1e-8)
            z = (train_features-center)/scale
            gram = z.T @ z + 10*np.eye(z.shape[1])
            coef = linalg.solve(gram, z.T @ (y_fit-y_fit.mean()), assume_a="pos")
            error = y_true - ((features[score_rows]-center)/scale @ coef + y_fit.mean())
            prev_rms = rms[score_rows]
            rows.append({
                "study": name, "fold": fold, "score_start": a, "score_stop_exclusive": b,
                "max_analysis_history": 544 if name == "rpm_lag_32" else 512, "fit_history_guard": START,
                "speed_lookahead_samples": 32 if name == "rpm_lead_32_descriptive" else 0,
                "columns": z.shape[1], "mse": float(np.mean(error**2)),
                "residual_lag1_acf": float(_acf(error, 1)[1]),
                "squared_residual_rms_rank_corr": float(stats.spearmanr(error**2, prev_rms).statistic),
                "causal_feature": name != "rpm_lead_32_descriptive",
            })
    return rows


def _memory_table(x: np.ndarray, n: int = 96):
    a = _acf(x, max(2000, n))
    pac = np.r_[1.0, [linalg.solve_toeplitz(a[:lag], a[1:lag+1])[-1] for lag in range(1, n+1)]]
    return a, pac


def _band_power(record: Record) -> list[dict]:
    n = int(.25 * record.fs)
    values = record.x[:len(record.x)//n*n].reshape(-1, n)
    rpm = record.rpm[:values.size].reshape(-1, n)
    s = rpm.mean(axis=1)
    f, power = signal.periodogram(values, record.fs, window="hann", axis=1)
    weighted_mean = (power*f).sum(axis=1) / power.sum(axis=1)
    rows = []
    for s0, s1 in ((120, 135), (135, 145), (145, 155), (155, 170)):
        for label, mask in [
            ("rising", rpm[:, -1] > rpm[:, 0]),
            ("falling", rpm[:, -1] < rpm[:, 0]),
            ("pooled", np.ones(len(s), dtype=bool)),
        ]:
            keep = (s0 <= s) & (s < s1) & mask
            if not np.any(keep):
                continue
            for lo, hi in ((0, 250), (250, 550), (550, 1000), (1000, 1200), (1200, 1600)):
                ix = (lo <= f) & (f < hi)
                bands = power[keep][:, ix].sum(axis=1) * (f[1]-f[0])
                rows.append({
                    "rpm_lower": s0, "rpm_upper_exclusive": s1,
                    "direction": label, "frequency_lower_hz": lo,
                    "frequency_upper_hz_exclusive": hi, "blocks": int(keep.sum()),
                    "mean_band_power": float(bands.mean()),
                    "std_band_power": float(bands.std(ddof=1)) if len(bands) > 1 else None,
                    "mean_spectral_centroid_hz": float(weighted_mean[keep].mean()),
                })
    return rows


def run_extensions(train: Record, out: Path, figure_dir: Path) -> dict:
    """Save numerical tests and rank the physics groups linked to SRC candidates."""
    data = Path(out)
    folder = data / "eda"
    tables = folder / "tables"
    figure_dir = Path(figure_dir)
    checks = phase_fixtures()
    (folder / "synthetic_phase_checks.json").write_text(json.dumps(checks, indent=2)+"\n", encoding="utf-8")
    fits = _feature_trials(train)
    _write_csv(tables/"src_feature_screen.csv", fits)
    trials = {name: float(np.mean([r["mse"] for r in fits if r["study"] == name]))
              for name in sorted({r["study"] for r in fits})}
    rows = _band_power(train)
    _write_csv(tables/"eda_band_power_by_speed_direction.csv", rows)
    acf, pac = _memory_table(train.x)
    _write_csv(tables/"eda_acf_pacf.csv", [
        {"lag_samples": i, "lag_ms": 1000*i/train.fs, "acf": value,
         "pacf": pac[i] if i < len(pac) else None}
        for i, value in enumerate(acf)
    ])
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8.2, 4.9))
    lag_ms = 1000*np.arange(len(pac))/train.fs
    for i, (value, label) in enumerate(((acf[:len(pac)], "ACF"), (pac, "PACF"))):
        ax[i].plot(lag_ms, value, color="#156174", lw=1.15, marker="o", ms=2.8)
        ax[i].axhline(0, color="#687b83", lw=.65)
        ax[i].set_ylabel(label)
    ax[1].set_xlabel("Lag [ms]")
    ax[0].set_title("Training vibration memory in sample coordinates")
    _save(fig, figure_dir/"eda_pacf")
    rows = sorted(
        [{"study": name, "mean_guarded_mse": v} for name, v in trials.items()
         if name != "rpm_lead_32_descriptive"],
        key=lambda row: row["mean_guarded_mse"],
    )
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    labels = [r["study"].replace("_", " ") for r in rows]
    y = np.arange(len(rows))
    ax.scatter([r["mean_guarded_mse"] for r in rows], y, c="#126776", s=28)
    ax.set(yticks=y, yticklabels=labels, xlabel="Mean guarded training-fold MSE")
    ax.grid(axis="x", color="#e4eced")
    ax.invert_yaxis()
    _save(fig, figure_dir/"eda_feature_decisions")

    d = np.diff(train.rpm)*train.fs
    summary = {
        "feature_fold_means": trials,
        "oscillator_implementation": "512-sample normalized exponentially weighted complex history, predicting from observations before target",
        "rpm_derivative_quantiles_rpm_per_s": np.quantile(d, [.01,.10,.50,.90,.99]).tolist(),
        "zero_speed_difference_count": int(np.sum(d == 0)),
        "exact_vibration_extrema_count": int(np.sum(train.x == train.x.min())+np.sum(train.x == train.x.max())),
        "skewness": float(stats.skew(train.x)),
        "lag1_pacf": float(pac[1]),
        "largest_absolute_pacf_lag_2_through_96": int(np.argmax(np.abs(pac[2:]))+2),
        "measured_sensor_precision": "unavailable; derivative and zero-difference counts are descriptive only",
        "rpm_lead_diagnostic": "offline deliberate violation excluded from feature recommendations",
        "phase_fixtures": checks,
    }
    (folder/"extension_summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    _write_hypotheses(data, trials)
    return summary


def _write_hypotheses(data, trials):
    base = trials["lpv_ar24_control"]
    rows = [
        ("H1", "speed and conditional RMS", "past RPM schedules gains; local RMS tracks load/speed",
         "RMS/RPM correlation 0.927; causal RMS alone did not improve short LPV screen", "O(L)", 64,
         "degree 1–3; RMS 16–128 samples", "fit-time load/speed confounding", "retain as context", "SRC_SPEED"),
        ("H2", "mesh-aligned lag history", "interpolate at equal past shaft angle, 1/37 turn for one mesh",
         "isolated mesh lag 0.46242 vs short LPV control 0.46153", "O(K log N) batch index", 670,
         "8–24 nodes/mesh, 4–16 cycles", "interpolation bias and pulse absorption", "test grouped history", "SRC_ANGLE"),
        ("H3", "focused speed-locked oscillator", "demodulate by 2π k u and remodulate at target phase",
         f"148 ridge MAE 0.744 Hz; tau32 two-band MSE {trials['two_orders_tau_32']:.5f}", "O(KM) direct FIR or O(K) rolling", 512,
         "orders 148,259; decay 8–128 trials", "underrepresents other harmonics", "recommend focused candidate", "SRC_FOCUSED"),
        ("H4", "broad mesh oscillator", "same rotating history across a first-six, first-twelve or ranked bank",
         f"six-band tau32 MSE {trials['geometry_six_tau_32']:.5f} vs LPV control {base:.5f}", "O(KM) direct FIR or O(K) rolling", 512,
         "order subsets; decay 8–256; cap prior order near 444", "history and dimensionality", "recommend geometry candidate", "SRC_GEOMETRY"),
        ("H5", "fixed-Hz state pair", "phase from known clock around persistent Hz bands",
         f"1080/1125 Hz MSE {trials['fixed_hz_pair_tau_32']:.5f}; 31.3% Welch in 1.0–1.2 kHz", "O(KM) or O(K) rolling", 512,
         "off or 1080/1125 Hz", "mesh frequency overlap; physical mode unconfirmed", "reject as default; retain competing hypothesis", "SRC_FIXED_HZ"),
        ("H6", "shaft and modulation sidebands", "1 and 148 ± 1 orders can encode shaft modulation",
         f"sidebands MSE {trials['two_orders_plus_sidebands']:.5f} vs two-order baseline {trials['two_orders_tau_32']:.5f}", "O(KM) or O(K) rolling", 512,
         "none, order one or small symmetric pairs", "danger of predicting crack periodicity", "require incremental gain and pulse retention", "SRC_SIDEBANDS"),
        ("H7", "structured polynomial interactions", "local gain from RMS×lag and speed×quadrature",
         f"degree2 MSE {trials['two_orders_structured_degree_2']:.5f}; degree3 {trials['two_orders_structured_degree_3']:.5f}", "O(P)", 512,
         "degrees 1–3, screened interactions", "instability outside experienced amplitude range", "compare low degrees with shrinkage", "SRC_LIBRARY"),
        ("H8", "causal innovation memory", "lagged prediction errors of previous observed targets",
         "residual ACF and conditional residual/RMS rank correlation measured by fold", "O(P)", None,
         "proxy AR 64–384 and errors 8–48", "fault dilution and state contamination", "fit both stages within guarded segments", "SRC_INNOVATION"),
    ]
    fields = ["id", "feature", "mechanism", "training_evidence", "cost_note", "history_samples",
              "starting_grid", "failure_mode", "recommendation", "downstream_experiment_id"]
    _write_csv(data/"eda_feature_hypotheses.csv", [dict(zip(fields, row, strict=True)) | {
        "causal_status": "causal with past-only construction and recorded support"
    } for row in rows])
