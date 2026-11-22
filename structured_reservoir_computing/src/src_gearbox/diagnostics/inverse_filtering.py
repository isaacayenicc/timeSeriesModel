"""Required common-mask, whole-record prediction adapter."""

from __future__ import annotations

import numpy as np

START = 1792


def forecast(predictor, record, *, x=None, rpm=None, start=START) -> np.ndarray:
    vibration = record.x if x is None else x
    speed = record.rpm if rpm is None else rpm
    result = np.asarray(predictor(vibration, speed, record.fs), dtype=np.float64)
    if result.shape != record.x.shape:
        raise ValueError("Frozen model must return one prediction sample per input sample")
    score = result[start:]
    if len(score) == 0 or not np.isfinite(score).all():
        raise ValueError("Frozen predictor unavailable on the required common mask")
    return score
