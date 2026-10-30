"""Strictly past-only, teacher-forced windows."""
import numpy as np


def make_windows(record, lag, kind="ar"):
    """Return X,y,indices; first target x[lag] sees x[0:lag].

    Neural input is [window,channel,time], channels x and past spdnorm.
    LPV uses speed at t-1, never at or after t.
    """
    x, s = record["x"], record["spdnorm"]
    if not isinstance(lag, int) or not 1 <= lag < len(x):
        raise ValueError("lag must be an integer in [1,N)")
    w = np.lib.stride_tricks.sliding_window_view(x, lag)[:-1]
    sw = np.lib.stride_tricks.sliding_window_view(s, lag)[:-1]
    if kind == "arx":
        a = np.column_stack((w, sw[:, -1], sw[:, -1]**2))
    elif kind == "lpv":
        a = np.column_stack((w, w*sw[:, -1, None], w*sw[:, -1, None]**2))
    elif kind in ("mlp", "gru", "tcn", "speed_tcn", "late_tcn", "shift_tcn", "capacity_tcn"):
        a = np.stack((w, sw), axis=1).astype(np.float32)
    else:
        a = w
    return a, x[lag:], np.arange(lag, len(x))
