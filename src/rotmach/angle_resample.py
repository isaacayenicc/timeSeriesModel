"""Explicit-unit angular resampling; not a verified MATLAB-toolbox clone."""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid


def angle_resample(
    y: np.ndarray,
    t: np.ndarray,
    speed: np.ndarray,
    unit: str,
    samples_per_cycle: int = 128,
) -> dict[str, np.ndarray | float | int | str]:
    """Integrate positive speed and linearly interpolate onto uniform cycles.

    y, t and speed are finite 1D vectors with identical length >=3. t is seconds;
    unit must be 'rpm', 'rps', or 'rad_s'. No default unit is guessed. Integrated
    cycles start at zero at t[0], so no rotation before the first sample is
    invented. samples_per_cycle is an integer output density, not Hz.

    The endpoint grid stops within the measured cycle interval; no extrapolation
    is permitted. Linear interpolation is neither bandlimited resampling nor
    anti-alias filtering. For precision order analysis the caller must justify
    samples/cycle using signal bandwidth and minimum shaft speed; this routine
    warns through metadata when requested density can downsample the record.
    A round-trip error is a sensitivity diagnostic, not a bound on interpolation
    error against the unknown continuous signal. Whole-record angular features
    are diagnostic transforms, not causal prediction inputs.
    """
    arrays = [np.asarray(a, dtype=float) for a in (y, t, speed)]
    if any(a.ndim != 1 or len(a) < 3 or not np.all(np.isfinite(a)) for a in arrays):
        raise ValueError("y/t/speed must be finite 1D vectors with >=3 samples")
    if len({len(a) for a in arrays}) != 1:
        raise ValueError("y/t/speed must have identical lengths")
    signal, time, s = arrays
    if np.any(np.diff(time) <= 0):
        raise ValueError("t must increase strictly")
    if np.any(s <= 0):
        raise ValueError("Angular mapping requires strictly positive speed")
    if isinstance(samples_per_cycle, bool) or not isinstance(
        samples_per_cycle, (int, np.integer)
    ) or samples_per_cycle < 2:
        raise ValueError("samples_per_cycle must be an integer >=2")
    conversion = {"rpm": 1 / 60, "rps": 1.0, "rad_s": 1 / (2 * np.pi)}
    if unit not in conversion:
        raise ValueError("Explicit speed unit must be rpm, rps, or rad_s")
    cycles_per_second = s * conversion[unit]
    # Extended accumulator precision reduces long positive-increment summation
    # drift; np.interp uses ordinary double precision for its coordinate arrays.
    cycles = np.asarray(cumulative_trapezoid(
        cycles_per_second.astype(np.longdouble), time.astype(np.longdouble),
        initial=0.0,
    ), dtype=float)
    total = float(cycles[-1])
    count = int(np.floor(total * samples_per_cycle + 1e-10)) + 1
    if count < 3:
        raise ValueError("Record covers too little angle for requested density")
    grid = np.arange(count, dtype=float) / samples_per_cycle
    # Floating point cannot authorize extrapolation beyond the final observation.
    grid = grid[grid <= total]
    interpolated = np.interp(grid, cycles, signal)
    output_time = np.interp(grid, cycles, time)
    inside = cycles <= grid[-1]
    reconstructed = np.interp(cycles[inside], grid, interpolated)
    error = float(np.sqrt(np.mean((reconstructed - signal[inside]) ** 2)))
    input_samples_per_cycle_max = float(np.max(1 / (np.diff(time) * cycles_per_second[:-1])))
    return {
        "signal": interpolated, "cycles": grid, "time": output_time,
        "samples_per_cycle": samples_per_cycle, "speed_unit_assumption": unit,
        "total_measured_cycles": total,
        "edge_loss_cycles": float(total - grid[-1]),
        "roundtrip_rmse": error,
        "downsampling_warning": int(samples_per_cycle < input_samples_per_cycle_max),
    }
