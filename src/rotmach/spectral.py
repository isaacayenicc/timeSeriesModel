"""One-sided FFT amplitudes, with an explicitly labeled reference variant."""

from __future__ import annotations

import numpy as np


def amplitude_spectrum(
    y: np.ndarray, fs: float, variant: str = "scientifically_corrected"
) -> tuple[np.ndarray, np.ndarray]:
    """Return frequency in Hz and rectangular-window one-sided amplitudes.

    For even N the real Nyquist bin is not doubled. For odd N every positive
    bin has a negative-frequency partner and must be doubled. The MATLAB
    fftspecturm reference always excludes its final returned bin from doubling;
    faithful_reference exposes its factor-of-two error at the final odd-N bin.
    This is amplitude (same unit as y), not PSD. No detrending is applied.
    """
    a = np.asarray(y, dtype=float)
    if a.ndim != 1 or len(a) < 2 or not np.all(np.isfinite(a)):
        raise ValueError("y must be finite 1D with >=2 samples")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be finite and positive")
    if variant not in ("faithful_reference", "scientifically_corrected"):
        raise ValueError("Unknown FFT variant")
    amplitude = np.abs(np.fft.rfft(a)) / len(a)
    if len(a) % 2 == 0 or variant == "faithful_reference":
        amplitude[1:-1] *= 2
    else:
        amplitude[1:] *= 2
    return np.fft.rfftfreq(len(a), d=1 / fs), amplitude
