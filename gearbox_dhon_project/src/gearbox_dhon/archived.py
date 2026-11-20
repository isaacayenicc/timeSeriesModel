"""Standalone replay of the improved archive's frozen inference checkpoints."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import integrate, signal
from scipy.special import expit, softmax

from gearbox_dhon.baselines import START, BaselineRecord, lag, load_arrays

ARCHIVE_NAMES = ("DHON_linear", "DHON_optimized", "NGRC_optimized", "DHON_local", "NGRC_local", "DHON_seed29")
CORRECTION_MEMORY = 1024


def load_archive_checkpoint(archive_dir: Path, name: str):
    """Load arrays and readable configuration from the supplied trusted checkpoint."""
    return json.loads((archive_dir / f"{name}.json").read_text(encoding="utf-8")), load_arrays(archive_dir / f"{name}.npz")


def _past_speed(rpm, indices):
    return np.clip((rpm[indices - 1] - 145) / 22, -1.12, 1.12).astype(np.float32)


def _stage_residual(x, rpm, spec, metadata, arrays):
    indices = np.arange(1, len(x))
    z = _past_speed(rpm, indices)
    p = np.full(len(x), metadata["sb"])
    for power, taps in enumerate(arrays["sw"].reshape(3, spec["stageL"])):
        p[1:] += signal.lfilter(taps.astype(float), [1.], x)[:-1] * z ** power
    e = (x - p).astype(np.float32)
    e[:spec["stageL"]] = 0.
    return e


def _base_features(record, spec, metadata, arrays, indices):
    x = record.x.astype(np.float32)
    rpm, t, fs = record.rpm, record.t, record.fs
    u = integrate.cumulative_trapezoid(rpm / 60, t, initial=0)
    phase = u[indices - 1] + rpm[indices - 1] / (60 * fs)
    z = _past_speed(rpm, indices)
    h = lag(x, spec["L"], indices)
    a = lag(np.clip(_stage_residual(x, rpm, spec, metadata, arrays), -spec["clip"], spec["clip"]), spec["ma"], indices)
    energy = np.sqrt(np.mean(h[:, :64] ** 2, axis=1))
    m = phase[:, None] - np.arange(1, spec["points"] * spec["cycles"] + 1) / (37 * spec["points"])
    if np.any(m[:, 0] > u[indices - 1] + 1e-10) or np.any(m[:, -1] < u[0]):
        raise ValueError("Archived angular history would extrapolate or use unavailable current vibration")
    angular = np.interp(m.ravel(), u, x).reshape(m.shape).astype(np.float32)

    oscillators = [(order, u, phase) for order in spec["orders"]]
    oscillators += [(hz, t, t[indices - 1] + 1 / fs) for hz in spec["hz"]]
    p = np.empty((len(indices), len(oscillators), len(spec["taus"]), 2), np.float32)
    for k, (frequency, reference, target) in enumerate(oscillators):
        v = x * np.exp(-2j * np.pi * frequency * reference)
        rotate = np.exp(2j * np.pi * frequency * target)
        for j, tau in enumerate(spec["taus"]):
            decay = np.exp(-1 / tau)
            memory = spec["mem"]
            v0 = v.copy()
            v0[memory:] -= decay ** memory * v[:-memory]
            gain = 2 * (1 - decay) / (1 - decay ** memory)
            q = gain * signal.lfilter([1.], [1., -decay], v0)[indices - 1] * rotate
            p[:, k, j, 0], p[:, k, j, 1] = q.real, q.imag
    return h, angular, a, z, energy, p


def _nonlinear_parts(h, m, a, p, spec):
    angular_lags = np.minimum((np.array([.5, 1, 2, 4]) * spec["points"] - 1).astype(int), m.shape[1] - 1)
    q = np.column_stack((h[:, [0, 1, 3, 7, 15, 31]], m[:, angular_lags], a[:, [0, 1, 3, 7]]))
    harmonics = p[:, :, 1, :]
    eh = np.sum(harmonics ** 2, axis=-1)
    eh = eh / (1 + eh)
    order_count = len(spec["orders"])
    e1 = eh[:, :order_count].mean(axis=1)
    e2 = eh[:, order_count:].mean(axis=1) if spec["hz"] else e1
    return q, harmonics, eh, e1, e2


def _affine(x, arrays, layer):
    return x @ arrays[f"net_{layer}.weight"].T + arrays[f"net_{layer}.bias"]


def _head(p, q, arrays):
    n, o, r, _ = p.shape
    q = np.ascontiguousarray(np.clip((q - arrays["mu"]) / arrays["sd"], -5, 5), dtype=np.float32)
    mix = arrays["net_mix"][None] + .5 * np.tanh(q[:, -5:-2] @ arrays["net_adapt.weight"].T).reshape(n, o, r)
    pooled = np.sum(p * softmax(mix, axis=2)[..., None], axis=2)
    context = np.column_stack((q, pooled.reshape(n, -1)))
    gain = _affine(context, arrays, "gain.0")
    gain = _affine(gain * expit(gain), arrays, "gain.3")
    residual = _affine(context, arrays, "res.0")
    residual = _affine(residual * expit(residual), arrays, "res.3")
    return np.sum(.25 * np.tanh(gain).reshape(n, o, 2) * pooled, axis=(1, 2)) + residual[:, 0]


def _predict(record, indices, metadata, arrays):
    spec = metadata["spec"]
    family = metadata["family"]
    gate = metadata["gate"]
    h, m, a, z, energy, p = _base_features(record, spec, metadata, arrays, indices)
    g = np.column_stack((h, m, a, p.reshape(len(indices), -1)))
    w, b = arrays["bw"], metadata["bb"]
    if family == "NGRC":
        w = (1 - gate) * w + gate * arrays["nw"][:len(w)]
        b = (1 - gate) * b + gate * metadata["nb"]
    vals = g @ w[:-2].reshape(spec["degree"] + 1, g.shape[1]).T
    result = vals[:, 0] + w[-2] * z + w[-1] * energy + b
    for power in range(1, spec["degree"] + 1):
        result += vals[:, power] * z ** power

    if family in ("DHON", "NGRC"):
        q, harmonics, eh, e1, e2 = _nonlinear_parts(h, m, a, p, spec)
    if family == "NGRC":
        lib = np.column_stack((q ** 3, q * energy[:, None] ** 2, q * e1[:, None], q * e2[:, None],
                               (harmonics * eh[..., None]).reshape(len(indices), -1),
                               q * z[:, None] * energy[:, None]))
        result += lib @ (gate * arrays["nw"][len(w):])
    if family == "DHON":
        context = np.column_stack((q, z, np.log1p(energy), np.log1p(np.sqrt(np.mean(a * a, axis=1))), e1, e2))
        result += gate * _head(p, context, arrays)
    return result


def archived_forecast(record: BaselineRecord, metadata, arrays, corrected=True, start=START):
    n = len(record.x)
    first = start - CORRECTION_MEMORY if corrected else start
    indices = np.arange(first, n)
    p = _predict(record, indices, metadata, arrays)
    result = np.full(n, np.nan)
    if corrected:
        e = np.clip(record.x[indices].astype(np.float32).astype(float) - p, -1.75, 1.75)
        decay = np.exp(-1 / 640)
        finite = e.copy()
        finite[CORRECTION_MEMORY:] -= decay ** CORRECTION_MEMORY * e[:-CORRECTION_MEMORY]
        previous = np.r_[0., signal.lfilter([1 - decay], [1., -decay], finite)[:-1]] / (1 - decay ** CORRECTION_MEMORY)
        result[start:] = p[CORRECTION_MEMORY:] + .5 * previous[CORRECTION_MEMORY:]
    else:
        result[start:] = p
    return result
