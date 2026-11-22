"""Analytic envelopes and coherent-gain corrected order spectra."""

from __future__ import annotations

import numpy as np
from scipy import signal


def analytic_envelope(x, *, lecturer_mean_treatment=False) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if lecturer_mean_treatment:
        mean = float(x.mean())
        return np.abs(signal.hilbert(x - mean)) + mean
    return np.abs(signal.hilbert(x))


def amplitude_spectrum(x, samples_per_rev) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    window = signal.windows.hann(len(x), sym=False)
    order = np.fft.rfftfreq(len(x), d=1 / samples_per_rev)
    amplitude = np.abs(np.fft.rfft((x - x.mean()) * window)) / window.sum()
    amplitude[1 : None if len(x) % 2 else -1] *= 2
    return order, amplitude
