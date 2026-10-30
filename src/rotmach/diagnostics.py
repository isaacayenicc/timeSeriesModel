"""Residual diagnostics with explicit numerical conventions.

These are descriptive statistics, not physical-fault classifications.  Hilbert
envelopes and ACFs are whole-record diagnostic transforms; they must never be
used as causal predictor inputs without a separate causal implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from time import perf_counter
from typing import Any

import numpy as np
from scipy.signal import correlate, hilbert
from scipy.stats import kurtosis


def _vector(y: np.ndarray, minimum: int = 1) -> np.ndarray:
    """Validate a finite real one-dimensional signal without flattening records."""
    a = np.asarray(y, dtype=float)
    if a.ndim != 1 or len(a) < minimum or not np.all(np.isfinite(a)):
        raise ValueError(f"Expected finite 1D signal with >= {minimum} samples")
    return a


def signal_features(y: np.ndarray) -> dict[str, float]:
    """Return RMS, maximum absolute amplitude/RMS, and biased Pearson kurtosis.

    Input: finite shape (N,) in any consistent amplitude unit. RMS retains this
    unit; crest factor and kurtosis are dimensionless. Constant/zero signals
    have undefined kurtosis; a zero signal also has undefined crest factor.
    They are represented by NaN, rather than fabricated zeros. Multiplicative
    scaling avoids intermediate overflow in the statistics.
    """
    a = _vector(y)
    scale = float(np.max(np.abs(a)))
    if scale == 0:
        return {"rms": 0.0, "crest_factor": np.nan, "kurtosis": np.nan}
    z = a / scale
    rms_z = float(np.sqrt(np.mean(z * z)))
    variance = float(np.var(z))
    return {
        "rms": scale * rms_z,
        "crest_factor": 1.0 / rms_z,
        "kurtosis": float(kurtosis(z, fisher=False, bias=True))
        if variance > 0 else np.nan,
    }


def correlated_kurtosis(
    y: np.ndarray, period_samples: int, order: int = 1
) -> float:
    r"""Port the reference's zero-padded correlated-kurtosis statistic.

    For a finite vector y of length N, integer T=period_samples >=1 and integer
    M=order >=0, return

    sum_n [prod_{m=0}^M y[n-mT]]^2 / (sum_n y[n]^2)^(M+1),

    with y[k]=0 for k<0. The reference dynamically grows its N-by-M array to
    N-by-(M+1); this implementation uses the formula directly. CK depends on
    record length and padding, so only equal-length segments should be compared.
    Zero energy gives NaN. Scale normalization preserves the exact statistic.
    """
    a = _vector(y)
    if isinstance(period_samples, bool) or not isinstance(
        period_samples, (int, np.integer)
    ) or period_samples < 1:
        raise ValueError("period_samples must be a positive integer sample lag")
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 0:
        raise ValueError("order must be a nonnegative integer")
    scale = float(np.max(np.abs(a)))
    if scale == 0:
        return np.nan
    z = a / scale
    product = z.copy()
    for m in range(1, order + 1):
        lag = m * period_samples
        shifted = np.zeros_like(z)
        if lag < len(z):
            shifted[lag:] = z[:-lag]
        product *= shifted
    return float(np.sum(product**2) / np.sum(z**2)**(order + 1))


def normalized_acf(y: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Centered, biased ACF, rho[k]=sum(z[k:]*z[:-k])/sum(z*z).

    Lag zero is element zero. The denominator is common to all lags; this is
    not the lag-wise N-k/unbiased convention. Constant signals return all NaNs.
    All requested lags must be between zero and N-1 inclusive. A relative range
    <=1e-12 of maximum absolute amplitude is treated as numerically constant;
    this prevents meaningless normalized correlations of Hilbert roundoff.
    """
    a = _vector(y)
    if max_lag is None:
        max_lag = len(a) - 1
    if not isinstance(max_lag, (int, np.integer)) or not 0 <= max_lag < len(a):
        raise ValueError("max_lag must be an integer in [0, N-1]")
    scale = float(np.max(np.abs(a)))
    z = a / scale if scale > 0 else a.copy()
    z = z - np.mean(z)
    denominator = float(z @ z)
    if denominator <= np.finfo(float).tiny or np.ptp(z) <= 1e-12:
        return np.full(max_lag + 1, np.nan)
    acf = correlate(z, z, mode="full", method="fft")[len(z)-1:len(z)+max_lag]
    return np.asarray(acf / denominator)


def analytic_envelope(
    y: np.ndarray, convention: str = "hilbert"
) -> np.ndarray:
    """Return abs(scipy.signal.hilbert(y)); endpoint effects remain explicit.

    MATLAB envelope's default analytic-envelope construction is not guaranteed
    bitwise identical, particularly at boundaries. Live MATLAB parity has not
    been tested. This FFT/Hilbert definition is the declared Python convention.
    """
    a = _vector(y, minimum=2)
    if convention == "hilbert":
        return np.abs(hilbert(a))
    if convention == "matlab_mean_restored_candidate":
        mean = float(np.mean(a))
        return np.abs(hilbert(a - mean)) + mean
    raise ValueError("Unknown envelope convention")


def pmi(
    y: np.ndarray,
    samples_per_unit: float,
    fault_order: float = 1.0,
    variant: str = "scientifically_corrected",
    envelope_convention: str = "matlab_mean_restored_candidate",
) -> dict[str, Any]:
    """Envelope ACF PMI at one assumed period, with explicit MATLAB indexing.

    samples_per_unit is samples/cycle for angular data or samples/second for
    time data; fault_order has the reciprocal coordinate unit. It must not be
    inferred from a speed vector with unknown units. No defect order is implied
    by the default value 1. The corrected branch selects lag round(fs/f_fault),
    using MATLAB-style positive half-up rounding. faithful_reference mimics the
    original coarse branch's acf(round(fs/f_fault)) one-based indexing: selected
    zero-based lag is round(fs/f_fault)-1. The returned historic timeindex is
    period+1, another offset in the source, and is not a physical lag.

    Returns rho, PMI=rho/(1-rho), selected_lag_samples, expected_period_samples,
    source_timeindex_matlab and status. Near rho=1 is explicitly undefined
    (NaN); negative rho is retained, never absolutized or clipped. Refined
    source timeindex search is not implemented and is not claimed reproduced.
    """
    a = _vector(y, minimum=2)
    if not np.isfinite(samples_per_unit) or samples_per_unit <= 0:
        raise ValueError("samples_per_unit must be finite and positive")
    if not np.isfinite(fault_order) or fault_order <= 0:
        raise ValueError("fault_order must be finite and positive")
    if variant not in ("faithful_reference", "scientifically_corrected"):
        raise ValueError("Unknown PMI variant")
    period = int(np.floor(samples_per_unit / fault_order + 0.5))
    if period < 1:
        raise ValueError("Expected period must round to at least one sample")
    lag = period - 1 if variant == "faithful_reference" else period
    if lag >= len(a):
        raise ValueError("Expected PMI lag exceeds available signal length")
    envelope = analytic_envelope(a, convention=envelope_convention)
    rho = float(normalized_acf(envelope, lag)[lag])
    status = "ok"
    if not np.isfinite(rho):
        value = np.nan
        status = "undefined_zero_envelope_variance"
    elif abs(1.0 - rho) <= 32 * np.finfo(float).eps:
        value = np.nan
        status = "undefined_near_unit_acf"
    else:
        value = rho / (1.0 - rho)
    return {
        "pmi": float(value), "rho": rho,
        "selected_lag_samples": lag,
        "expected_period_samples": period,
        "source_timeindex_matlab": period + 1
        if variant == "faithful_reference" else None,
        "variant": variant, "status": status,
        "envelope_convention": envelope_convention,
    }


def analyze_records(
    records: Iterable[Mapping[str, Any]],
    predictors: Mapping[str, Callable[[Mapping[str, Any]], np.ndarray]] | None = None,
    burnin: int = 256,
    speed_unit: str | None = None,
    samples_per_cycle: int = 128,
) -> list[dict[str, Any]]:
    """Analyze independent records and same-segment raw/residual diagnostics.

    Every record must expose x, t, spd shape (N,). Predictor(record) must return
    full shape (N,) predictions (NaNs allowed before burnin), and is called anew
    per record, so stateful callers must reset themselves inside that callback.
    No model is fit here and no record is concatenated. IDs are provenance IDs,
    never health labels. Unknown speed units omit angular CK/PMI entirely.
    """
    from .angle_resample import angle_resample

    if not isinstance(burnin, int) or burnin < 0:
        raise ValueError("burnin must be a nonnegative integer")
    results: list[dict[str, Any]] = []
    predictors = {} if predictors is None else predictors
    for record_number, record in enumerate(records, start=1):
        x = _vector(record["x"], minimum=burnin + 3)
        time = _vector(record["t"])
        speed = _vector(record["spd"])
        if len(time) != len(x) or len(speed) != len(x):
            raise ValueError("Record arrays x/t/spd must have equal length")
        if np.any(np.diff(time) <= 0):
            raise ValueError("Record timestamps must increase strictly")
        series = {"raw": x[burnin:]}
        for model_id, predictor in predictors.items():
            prediction = np.asarray(predictor(record), dtype=float)
            if prediction.shape != x.shape:
                raise ValueError("Predictor must return full-record shape (N,)")
            series[model_id] = _vector(x[burnin:] - prediction[burnin:])
        for model_id, y in series.items():
            start = perf_counter()
            row: dict[str, Any] = {
                "record_id": record.get("record_id", f"fault_candidate_record_{record_number:03d}"),
                "model_id": model_id,
                "n_evaluated": len(y), "burnin_samples": burnin,
                "mean_speed_source_units": float(np.mean(speed[burnin:])),
                "speed_unit_assumption": speed_unit or "unknown",
                **signal_features(y),
            }
            if speed_unit is None:
                row.update({"angular_status": "blocked_unknown_speed_unit"})
            else:
                angular = angle_resample(
                    y, time[burnin:], speed[burnin:], unit=speed_unit,
                    samples_per_cycle=samples_per_cycle,
                )
                row["angular_status"] = "assumption_based_not_verified_physical_fault_order"
                row["angular_n"] = len(angular["signal"])
                row["ck_order1"] = correlated_kurtosis(
                    angular["signal"], samples_per_cycle, 1
                )
                for variant in ("faithful_reference", "scientifically_corrected"):
                    output = pmi(angular["signal"], samples_per_cycle, variant=variant)
                    row[f"pmi_{variant}"] = output["pmi"]
                    row[f"pmi_status_{variant}"] = output["status"]
            row["feature_seconds"] = perf_counter() - start
            results.append(row)
    return results
