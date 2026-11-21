from dataclasses import dataclass

import numpy as np
import pytest

from gearbox_expanded.protocol import FOLDS, START, guarded_segments, mse
from gearbox_expanded.study import paired_rows, verify_causality


@dataclass
class Signal:
    x: np.ndarray
    rpm: np.ndarray
    rpm_min: float | None = None
    rpm_max: float | None = None
    speed_normalized: np.ndarray | None = None


def test_guards_disjoint():
    sample = np.arange(24_000)
    record = Signal(sample, sample)
    for a,b in FOLDS:
        left,right = guarded_segments(record,a,b)
        assert left[0][-1] == a-START-1
        assert right[0][0] == b+START


def test_mse_finite_matched():
    assert mse(np.array([2.,3.]), np.array([1.,4.])) == 1
    with pytest.raises(ValueError):
        mse(np.array([2.,3.]), np.array([1.,np.nan]))


def test_paired_interval_zero_when_same():
    y = np.sin(np.arange(1500))
    rows = paired_rows(y, {"primary_dhon":y.copy(),"ar":y.copy()})
    assert len(rows)==3
    assert all(row["delta_mse"]==row["low95"]==row["high95"]==0 for row in rows)


def test_future_intervention_detects_target_leakage():
    n = 14_000
    item = Signal(np.arange(n,dtype=float), np.full(n,140.))
    past = lambda rec: np.r_[0.,rec.x[:-1]]
    assert verify_causality(past,item,past(item)) == 0
    with pytest.raises(RuntimeError):
        verify_causality(lambda rec: rec.x,item,item.x)
