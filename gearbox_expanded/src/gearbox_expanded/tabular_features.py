"""Bounded, strict-prior-sample features for tabular gearbox predictors."""

from __future__ import annotations

import numpy as np

START = 1792
GUARD = 1792
FOLDS = ((4800, 7200), (11200, 13600), (18400, 20800))
ANGLE_HISTORY = 768
COMPACT_LAGS = np.r_[1:17, [19, 22, 26, 31, 37, 43, 51, 61, 73, 87, 101, 128, 160, 192, 240]]
ANGLE_DELAYS = np.r_[np.array([1, 2, 3, 4, 6, 8]) / 37, 0.25, 0.375]
ORDER_BANK = (37, 74, 111, 148, 259, 370)
FEATURES = {
    "temporal_compact": (COMPACT_LAGS, 0, False, False),
    "temporal": (np.arange(1, 161), 0, False, False),
    "rpm": (np.arange(1, 161), 2, False, False),
    "physics_compact": (COMPACT_LAGS, 1, True, False),
    "physics": (np.arange(1, 161), 2, True, True),
    "physics_wide": (np.arange(1, 241), 3, True, True),
}


def history_for(kind: str) -> int:
    delays, speed_degree, physics, _ = FEATURES[kind]
    return max(int(delays[-1]), ANGLE_HISTORY if physics else 1, 65 if speed_degree else 1)


def _angular_lags(x, rpm, fs, targets):
    position = np.r_[0.0, np.cumsum(rpm[:-1])] / (60.0 * fs)
    # Clamp each interpolation to the finite context of the requested target.
    lower = targets[:, None] - ANGLE_HISTORY
    phase = np.maximum(position[targets, None] - ANGLE_DELAYS,
                       position[lower])
    right = np.minimum(np.searchsorted(position, phase, side="right"), targets[:, None] - 1)
    left = np.maximum(lower, right - 1)
    frac = np.clip((phase - position[left]) / (position[right] - position[left]), 0.0, 1.0)
    return x[left] + frac * (x[right] - x[left])


def _quadratures(x, rpm, fs, targets):
    recent = np.arange(1, 161)
    ix = targets[:, None] - recent
    # Only the phase difference is required, so the dot product has no long-term state.
    phase = 2 * np.pi * np.cumsum(rpm[ix], axis=1) / (60 * fs)
    clk = 2 * np.pi * recent / fs
    weight = np.exp(-(recent - 1) / 45)
    weighted = x[ix] * weight / weight.sum()
    outputs = []
    for n in ORDER_BANK:
        rotation = n * phase
        outputs.extend([np.sum(weighted * np.cos(rotation), axis=1),
                        np.sum(weighted * np.sin(rotation), axis=1)])
    outputs.extend([weighted @ np.cos(1100 * clk), weighted @ np.sin(1100 * clk)])
    return np.column_stack(outputs)


def matrix(x, rpm, fs, targets, kind: str) -> np.ndarray:
    """Return exact requested rows with strict past-only vibration and RPM."""
    x = np.asarray(x, dtype=np.float64).ravel()
    rpm = np.asarray(rpm, dtype=np.float64).ravel()
    ix = np.asarray(targets, dtype=int)
    delays, degree, physics, expanded = FEATURES[kind]
    if len(x) != len(rpm):
        raise ValueError("Vibration and RPM must have equal lengths.")
    if ix.ndim != 1 or (len(ix) and (np.min(ix) < history_for(kind) or np.max(ix) >= len(x))):
        raise ValueError(f"Unsupported targets for {kind} with history {history_for(kind)}.")
    lag = x[ix[:, None] - delays]
    chunks = [lag]
    if degree:
        q = (rpm[ix - 1] - 145.0) / 25.0
        speed = [q, (rpm[ix - 1] - rpm[ix - 17]) / 25, (rpm[ix - 1] - rpm[ix - 65]) / 25]
        chunks.append(np.column_stack(speed))
        if kind != "physics_compact":
            chunks.append(lag * q[:, None])
            if degree >= 2:
                chunks.append(lag * (2 * q[:, None] ** 2 - 1))
            if degree >= 3:
                chunks.append(lag * (4 * q[:, None] ** 3 - 3 * q[:, None]))
    if physics:
        angular = _angular_lags(x, rpm, fs, ix)
        quad = _quadratures(x, rpm, fs, ix)
        energy = np.column_stack((np.mean(lag[:, :16] ** 2, axis=1),
                                  np.mean(lag[:, :31] ** 2, axis=1)))
        chunks.extend((angular, quad, energy))
        if expanded:
            chunks.extend((angular * q[:, None], quad * q[:, None]))
    return np.column_stack(chunks)


def training_matrix(x, rpm, fs, kind, slices=None):
    """Build features separately inside each allowed raw fit slice."""
    if slices is None:
        targets = np.arange(START, len(x))
        return matrix(x, rpm, fs, targets, kind), x[targets]
    features, labels = [], []
    for segment in slices:
        sub_x, sub_rpm = x[segment], rpm[segment]
        targets = np.arange(START, len(sub_x))
        if len(targets):
            features.append(matrix(sub_x, sub_rpm, fs, targets, kind))
            labels.append(sub_x[targets])
    if not features:
        raise ValueError("Fold contains no permitted targets after causal warm-up.")
    return np.concatenate(features), np.concatenate(labels)


def prepare_folds(x, rpm, fs, kind):
    """Return fit/evaluation features and labels for the three guarded folds."""
    return [
        (*training_matrix(x, rpm, fs, kind, (slice(None, a - GUARD), slice(b + GUARD, None))),
         matrix(x[:b], rpm[:b], fs, np.arange(a, b), kind), x[a:b])
        for a, b in FOLDS
    ]
