"""Offline rotational resampling. Prediction uses a separate causal angle path."""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid

SAMPLES_PER_REV = 2048


def revolutions(rpm: np.ndarray, t: np.ndarray) -> np.ndarray:
    rpm = np.asarray(rpm, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if rpm.ndim != 1 or rpm.shape != t.shape or len(rpm) < 3:
        raise ValueError("Expected a corresponding RPM value for every timestamp")
    if not np.isfinite(rpm).all() or not np.isfinite(t).all():
        raise ValueError("RPM or timestamp is nonfinite")
    if np.any(rpm <= 0) or np.any(np.diff(t) <= 0):
        raise ValueError("RPM and timestamp increments must be positive")
    phase = cumulative_trapezoid(rpm / 60, t, initial=0)
    if np.any(np.diff(phase) <= 0):
        raise ValueError("Integrated shaft angle is nonmonotonic")
    return phase


def resample(x, rpm, t, samples_per_rev=SAMPLES_PER_REV) -> tuple[np.ndarray, np.ndarray]:
    phase = revolutions(rpm, t)
    x = np.asarray(x, dtype=np.float64)
    if x.shape != phase.shape or not np.isfinite(x).all():
        raise ValueError("Vibration must be finite and aligned to the RPM vector")
    if not isinstance(samples_per_rev, (int, np.integer)) or samples_per_rev < 2:
        raise ValueError("Angular samples per revolution must be an integer >= 2")
    count = int(np.floor(phase[-1] * samples_per_rev)) + 1
    if count < 3:
        raise ValueError("Acquisition is too short for angular interpolation")
    grid = np.arange(count, dtype=np.float64) / samples_per_rev
    return grid, np.interp(grid, phase, x)
