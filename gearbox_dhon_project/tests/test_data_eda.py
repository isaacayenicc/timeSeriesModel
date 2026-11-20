from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat
from scipy.signal import periodogram

from gearbox_dhon.data import (
    _partial_overlap_scan, find_input, iter_fault_records, load_record, record_stats,
)
from gearbox_dhon.eda import angular_grid, causal_turns


def _values(x, rpm=None):
    t = (np.arange(len(x)) + 1) / 3200
    if rpm is None:
        rpm = 125 + 10 * t
    return dict(x=np.asarray(x)[:, None], spd=rpm[:, None], t=t[:, None], fs=3200,
                N=len(x), xraw=(.4 * x + .01)[:, None], xmean=[.01, 0, 0], xstd=[.4, 0, 0])


def _write_part(path, values):
    block = np.empty((1, len(values)), dtype=[(key, "O") for key in values[0]])
    for i, row in enumerate(values):
        for key, value in row.items():
            block[0, i][key] = value
    savemat(path, {"sigfault1": block})


def test_find_duplicate_suffix_and_reject_conflict(tmp_path):
    x = np.random.default_rng(75).normal(size=800)
    path = tmp_path / "SIGTRAIN (9).MAT"
    savemat(path, {"sigtrain": _values(x)}, appendmat=False)
    assert find_input(tmp_path, "sigtrain") == path
    record = load_record(path)
    assert np.array_equal(record.x, x)
    assert record_stats(record, record)["raw_reconstruction_max_abs_error"] < 1e-14
    savemat(tmp_path / "sigtrain.mat", {"sigtrain": _values(x + 1)})
    with pytest.raises(ValueError, match="Conflicting"):
        find_input(tmp_path, "sigtrain")


def test_mat_part_order_and_shifted_exact_match(tmp_path):
    rng = np.random.default_rng(15)
    x = rng.normal(size=2500)
    rpm = 125 + rng.uniform(size=len(x))
    _write_part(tmp_path / "sigfault1_part_002.mat", [
        _values(x[1200:], rpm[1200:]),
        _values(rng.normal(size=1300)),
    ])
    _write_part(tmp_path / "sigfault1_part_001.mat", [
        _values(rng.normal(size=1800)), _values(x[:1800], rpm[:1800]),
    ])
    records = list(iter_fault_records(tmp_path))
    assert [record.index for record in records] == [1, 2, 3, 4]
    assert [record.part_index for record in records] == [1, 2, 1, 2]
    pairs = _partial_overlap_scan([], tmp_path)
    assert len(pairs) == 1
    assert pairs[0]["record_id_first"] == "sigfault1:002"
    assert pairs[0]["record_id_second"] == "sigfault1:003"
    assert pairs[0]["second_minus_first_offset_samples"] == -1200


def test_causal_phase_and_order_peak():
    rpm = np.linspace(125, 167, 24000)
    u = causal_turns(rpm, 3200)
    perturbed = rpm.copy()
    perturbed[700:] += 30
    u2 = causal_turns(perturbed, 3200)
    np.testing.assert_array_equal(u[:701], u2[:701])
    assert u[701] != u2[701]
    order = 148
    _, xa = angular_grid(np.cos(2 * np.pi * order * u), u)
    f, psd = periodogram(xa, fs=2048, window="hann")
    assert abs(f[np.argmax(psd)] - order) < 0.1
