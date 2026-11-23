"""Instructor-compatible and direct mathematical PMI with explicit invalid cases."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .envelope import amplitude_spectrum, analytic_envelope
from .order_tracking import SAMPLES_PER_REV, resample

MODE_KEYS = ("lecturer_compatible", "lag_corrected", "mathematically_specified", "pmi_unrelated")


@dataclass(frozen=True)
class PMI:
    value: float
    correlation: float
    lag: int
    status: str


def pmi_from_envelope(envelope, lag: int) -> PMI:
    a = np.asarray(envelope, dtype=np.float64)
    if a.ndim != 1 or not np.isfinite(a).all():
        return PMI(np.nan, np.nan, lag, "invalid_envelope")
    if not 0 < lag < len(a):
        return PMI(np.nan, np.nan, lag, "insufficient_angular_history")
    b = a - a.mean()
    energy = float(b @ b)
    amplitude_energy = float(a @ a)
    if energy == 0 or energy <= 8 * np.finfo(float).eps * amplitude_energy:
        return PMI(np.nan, np.nan, lag, "zero_envelope_variance")
    cross = float(b[:-lag] @ b[lag:])
    if energy - cross <= 8 * np.finfo(float).eps * energy:
        return PMI(np.nan, cross / energy, lag, "degenerate_pmi_denominator")
    return PMI(cross / (energy - cross), cross / energy, lag, "ok")


def lecturer_score(x, rpm, t, samples_per_rev=SAMPLES_PER_REV) -> PMI:
    theta, angular = resample(x, rpm, t, samples_per_rev)
    lag = samples_per_rev - 1
    if len(theta) <= int(np.floor(3.1 * samples_per_rev + .5)):
        return PMI(np.nan, np.nan, lag, "insufficient_lecturer_autocorr_span")
    return pmi_from_envelope(analytic_envelope(angular, lecturer_mean_treatment=True), lag)


def measure(x, rpm, t, samples_per_rev=SAMPLES_PER_REV, *, traces=False) -> dict:
    """Measure frozen residuals on the common window, with no extra edge trimming."""
    theta, angular = resample(x, rpm, t, samples_per_rev)
    instructor_env = analytic_envelope(angular, lecturer_mean_treatment=True)
    direct_env = analytic_envelope(angular)
    full_turn = samples_per_rev
    literal = pmi_from_envelope(instructor_env, full_turn - 1)
    if len(theta) <= int(np.floor(3.1 * full_turn + .5)):
        literal = PMI(np.nan, np.nan, full_turn - 1, "insufficient_lecturer_autocorr_span")
    modes = (
        literal, pmi_from_envelope(instructor_env, full_turn),
        pmi_from_envelope(direct_env, full_turn),
        pmi_from_envelope(instructor_env, int(np.floor(.63 * full_turn + .5))),
    )
    x = np.asarray(x, dtype=np.float64)
    rms = float(np.sqrt(np.mean(x**2)))
    order, angular_spectrum = amplitude_spectrum(angular, full_turn)
    _, envelope_spectrum = amplitude_spectrum(direct_env, full_turn)
    power = envelope_spectrum**2
    whole_energy = power[1:].sum()
    squares = angular**2
    angular_energy = squares.sum()
    ck = float(squares[:-full_turn] @ squares[full_turn:] / angular_energy**2) if angular_energy else np.nan
    result = {
        "samples_per_rev": full_turn, "angular_samples": len(theta), "shaft_turns": float(theta[-1]),
        "rms": rms, "crest_factor": float(np.max(np.abs(x)) / rms) if rms else np.nan,
        "kurtosis_pearson_biased": float(stats.kurtosis(x, fisher=False)) if np.std(x) > 0 else np.nan,
        "correlated_kurtosis_shift_1rev": ck if len(theta) > full_turn else np.nan,
        "envelope_order1_amplitude": float(envelope_spectrum[np.argmin(abs(order - 1))]),
        "vibration_order37_amplitude": float(angular_spectrum[np.argmin(abs(order - 37))]),
        "envelope_near_order1_energy_fraction":
            float(power[(order >= .8) & (order <= 1.2)].sum() / whole_energy) if whole_energy else np.nan,
    }
    for name, value in zip(MODE_KEYS, modes, strict=True):
        result[name] = value.value
        result[f"{name}_lag"] = value.lag
        result[f"{name}_status"] = value.status
        result[f"{name}_rho"] = value.correlation
    if traces:
        result["traces"] = {
            "angle_rev": theta, "vibration": angular, "envelope": direct_env, "order": order,
            "vibration_spectrum": angular_spectrum, "envelope_spectrum": envelope_spectrum,
        }
    return result
