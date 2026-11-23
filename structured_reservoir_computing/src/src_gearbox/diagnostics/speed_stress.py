"""Predeclared validation RPM interventions and paired pulse experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from .inverse_filtering import START, forecast
from .order_tracking import revolutions
from .pmi import lecturer_score


@dataclass(frozen=True)
class PulseCase:
    carrier_hz: float
    amplitude: float
    phase_turn: float
    decay_seconds: float
    kind: str = "periodic"
    modulation: float = 0.0
    rpm_condition: str = "clean"


PULSE_CASES = (
    PulseCase(390, .12, .13, .0022, modulation=.15),
    PulseCase(620, .26, .61, .0061),
    PulseCase(940, .40, .27, .0042, modulation=.25),
    PulseCase(1260, .88, .84, .0017),
    PulseCase(780, .22, .45, .0038, modulation=.25, rpm_condition="smooth_profile_warp"),
    PulseCase(940, .40, .27, .0042, kind="aperiodic"),
    PulseCase(390, .12, .13, .0022, kind="aperiodic"),
)


def speed_cases(record) -> dict[str, np.ndarray]:
    rpm = record.rpm
    rng = np.random.default_rng(77811)
    z = rng.standard_normal(len(rpm))
    smooth = lfilter([.038], [1, -.962], z)
    smooth /= smooth.std()
    jitter = rng.standard_normal(len(rpm))
    delay = max(1, round(.040 * record.fs))
    dropouts = rpm.copy()
    length = max(1, round(.075 * record.fs))
    for start in range(round(record.fs * .83), len(rpm), round(record.fs * 1.29)):
        dropouts[start:start + length] = rpm[start - 1]
    angle = np.linspace(0, 2 * np.pi, len(rpm))
    return {
        "bias_minus_2pct": rpm * .98,
        "bias_plus_2pct": rpm * 1.02,
        "correlated_jitter_0p6pct": rpm * (1 + .006 * smooth),
        "independent_jitter_0p3pct": rpm * (1 + .003 * jitter),
        "delay_40ms": np.pad(rpm, (delay, 0), mode="edge")[:len(rpm)],
        "quantized_1rpm": np.rint(rpm),
        "dropout_hold_75ms": dropouts,
        "smooth_profile_warp": rpm * (1 + .008 * np.sin(angle) + .004 * np.sin(2 * angle)),
    }


def inject(record, case: PulseCase) -> np.ndarray:
    turn = revolutions(record.rpm, record.t)
    order = np.arange(int(np.ceil(turn[-1])))
    if case.kind == "periodic":
        pulse_turn = order + case.phase_turn
    elif case.kind == "aperiodic":
        pulse_turn = order + np.random.default_rng(77812).uniform(.025, .975, len(order))
    else:
        raise ValueError(f"Unknown pulse timing: {case.kind}")
    pulse_turn = pulse_turn[(pulse_turn >= 0) & (pulse_turn < turn[-1])]
    if not len(pulse_turn):
        return np.zeros_like(record.x)
    event = np.interp(pulse_turn, turn, record.t)
    match = np.searchsorted(event, record.t, side="right") - 1
    age = np.maximum(record.t - event[np.maximum(match, 0)], 0)
    strength = case.amplitude * (1 + case.modulation * np.cos(2 * np.pi * match / 5))
    amplitude = strength * np.exp(-age / case.decay_seconds)
    carrier = np.sin(2 * np.pi * case.carrier_hz * age)
    return np.where((match >= 0) & (age < .025), amplitude * carrier, 0.)


def _pmi(residual, rpm, t) -> float:
    return lecturer_score(residual, rpm, t).value


def evaluate_stress(record, predictions: dict, out: Path, *, start=START, pulse_models=None) -> dict:
    """Run clean, sensor-altered and paired-vibration paths with frozen weights."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pulse_names = set(predictions if pulse_models is None else pulse_models)
    sensor = speed_cases(record)
    target = record.x[start:]
    clean_rpm = record.rpm[start:]
    t = record.t[start:]
    bounds = np.quantile(clean_rpm, [0, .25, .5, .75, 1])
    rpm_bins = np.searchsorted(bounds[1:-1], clean_rpm, side="right")
    acceleration = np.diff(record.rpm, prepend=record.rpm[0])[start:]
    masks = {f"rpm_quartile_{q + 1}": rpm_bins == q for q in range(4)}
    masks.update(accelerating=acceleration > 0, decelerating=acceleration < 0)
    pulse_signals = {case: inject(record, case) for case in PULSE_CASES}
    x_noise = np.random.default_rng(77813).normal(0, .05, len(record.x))
    stress_rows, regime_rows, pulse_rows, traces = [], [], [], {}

    def add_stress(model, condition, mode, err, speed, clean_loss, clean_pmi, status="ok"):
        if err is None:
            mse, pmi = np.nan, np.nan
        else:
            mse = float(np.mean(err**2))
            pmi = _pmi(err, speed, t)
        stress_rows.append({
            "model": model, "condition": condition, "scope": mode,
            "mse": mse, "clean_mse": clean_loss, "relative_mse": mse / clean_loss if clean_loss else np.nan,
            "lecturer_compatible": pmi, "clean_pmi": clean_pmi,
            "delta_pmi": pmi - clean_pmi, "status": status,
        })
        if err is not None and mode == "predictor":
            for key, mask in masks.items():
                if mask.any():
                    regime_rows.append({
                        "model": model, "condition": condition, "group": key,
                        "targets": int(mask.sum()), "mse": float(np.mean(err[mask] ** 2)),
                        "min_true_rpm": float(np.min(clean_rpm[mask])),
                        "max_true_rpm": float(np.max(clean_rpm[mask])),
                    })

    for name, predict in predictions.items():
        base = forecast(predict, record, start=start)
        clean_error = target - base
        clean_loss = float(np.mean(clean_error**2))
        clean_pmi = _pmi(clean_error, clean_rpm, t)
        add_stress(name, "clean", "predictor", clean_error, clean_rpm, clean_loss, clean_pmi)
        prediction_by_sensor = {"clean": base}
        for condition, rpm in sensor.items():
            add_stress(name, condition, "diagnostics", clean_error, rpm[start:], clean_loss, clean_pmi)
            try:
                altered = forecast(predict, record, rpm=rpm, start=start)
            except (ValueError, RuntimeError, FloatingPointError) as error:
                for mode in ("predictor", "both"):
                    add_stress(name, condition, mode, None, clean_rpm, clean_loss, clean_pmi, str(error))
                continue
            prediction_by_sensor[condition] = altered
            for mode, diagnostic_speed in (("predictor", clean_rpm), ("both", rpm[start:])):
                add_stress(name, condition, mode, target - altered, diagnostic_speed, clean_loss, clean_pmi)
        try:
            altered = forecast(predict, record, x=record.x + x_noise, start=start)
            add_stress(name, "vibration_noise_sd_0p05", "predictor", target - altered, clean_rpm,
                       clean_loss, clean_pmi)
        except (ValueError, RuntimeError, FloatingPointError) as error:
            add_stress(name, "vibration_noise_sd_0p05", "predictor", None, clean_rpm,
                       clean_loss, clean_pmi, str(error))
        if name in pulse_names:
            raw_pmi = _pmi(target, clean_rpm, t)
            for case_id, (case, pulse) in enumerate(pulse_signals.items(), 1):
                row = {"model": name, "case": case_id, **vars(case), "status": "ok"}
                try:
                    injected = forecast(predict, record, x=record.x + pulse,
                                        rpm=sensor.get(case.rpm_condition), start=start)
                    compatible_base = prediction_by_sensor[case.rpm_condition]
                    observed = pulse[start:]
                    difference = injected - compatible_base
                    response = observed - difference
                    energy = float(observed @ observed)
                    matched_error = target - compatible_base
                    injected_error = target + observed - injected
                    before = _pmi(matched_error, clean_rpm, t)
                    after = _pmi(injected_error, clean_rpm, t)
                    row.update(
                        fault_distortion=float(difference @ difference) / energy,
                        residual_energy_retention=float(response @ response) / energy,
                        residual_pulse_projection=float(response @ observed) / energy,
                        residual_pulse_correlation=float(np.corrcoef(observed, response)[0, 1]),
                        clean_pmi=before, injected_pmi=after, delta_pmi=after - before,
                        delta_raw_pmi=_pmi(target + observed, clean_rpm, t) - raw_pmi,
                    )
                    if case_id == 1:
                        peak = int(np.argmax(abs(observed)))
                        sl = slice(max(0, peak - 30), min(peak + 120, len(t)))
                        traces["time_ms"] = (t[sl] - t[peak]) * 1000
                        traces["injection"] = observed[sl]
                        traces[f"{name}:response"] = response[sl]
                except (ValueError, RuntimeError, KeyError, FloatingPointError, ZeroDivisionError) as error:
                    row["status"] = str(error)
                pulse_rows.append(row)
        print(f"Fault stress finished: {name}", flush=True)

    stress = pd.DataFrame(stress_rows)
    stress.to_csv(out / "rpm_stress.csv", index=False)
    pd.DataFrame(regime_rows).to_csv(out / "rpm_stress_regimes.csv", index=False)
    selection = stress[(stress.scope == "predictor") & stress.condition.isin(sensor)]
    stress_summary = selection.groupby("model", as_index=False, sort=False).agg(
        scenarios=("condition", "size"), successful=("mse", "count"), clean_mse=("clean_mse", "first"),
        mean_stress_mse=("mse", "mean"), worst_stress_mse=("mse", "max"),
        worst_relative_mse=("relative_mse", "max"), median_delta_pmi=("delta_pmi", "median"),
    )
    stress_summary["failure_rate"] = 1 - stress_summary.successful / stress_summary.scenarios
    worst = selection.dropna(subset=["mse"]).sort_values("mse").groupby("model").tail(1)
    stress_summary = stress_summary.merge(worst[["model", "condition"]].rename(
        columns={"condition": "worst_condition"}), on="model", how="left")
    stress_summary.to_csv(out / "rpm_stress_summary.csv", index=False)
    if pulse_rows:
        pulses = pd.DataFrame(pulse_rows)
        pulses.to_csv(out / "pulse_preservation.csv", index=False)
        okay = pulses.loc[pulses.status.eq("ok")].copy()
        if not okay.empty:
            okay["increased_pmi"] = okay.delta_pmi > 0
            summary = okay.groupby(["model", "kind"], as_index=False, sort=False).agg(
                cases=("case", "size"), positive_delta_count=("increased_pmi", "sum"),
                median_pmi_delta=("delta_pmi", "median"),
                max_pmi_delta=("delta_pmi", "max"),
                median_distortion=("fault_distortion", "median"),
                median_energy_retention=("residual_energy_retention", "median"),
                median_pulse_correlation=("residual_pulse_correlation", "median"),
            )
            summary.to_csv(out / "pulse_summary.csv", index=False)
    if traces:
        np.savez_compressed(out / "pulse_example.npz", **traces)
    return {"rpm_rows": len(stress_rows), "pulse_rows": len(pulse_rows)}
