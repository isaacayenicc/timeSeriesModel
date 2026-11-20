"""Bootstrap, exact overlap and prediction metrics with analytic examples."""
from types import SimpleNamespace

import numpy as np

from gearbox_dhon.freeze import exact_overlap
from gearbox_dhon.results import block_difference, scores


def test_scores_exact_constant_offset():
    y = np.linspace(-4, 4, 100)
    result = scores(y, y-1)
    assert result["count"] == 100
    np.testing.assert_allclose([result["mse"], result["rmse"], result["mae"], result["bias"]], 1)


def test_moving_block_constant_difference():
    np.testing.assert_allclose(block_difference(np.full(77, -.25), block=16, draws=50), [-.25, -.25])


def test_joint_exact_shift_check():
    rng = np.random.default_rng(78)
    ref = SimpleNamespace(x=rng.normal(size=256), rpm=rng.uniform(120, 160, 256), fs=3200.)
    other = SimpleNamespace(x=np.r_[rng.normal(size=8), ref.x, rng.normal(size=31)],
                            rpm=np.r_[rng.uniform(120, 160, 8), ref.rpm, rng.uniform(120, 160, 31)], fs=3200.)
    assert exact_overlap(other, ref, window=32)
    other.rpm += .1
    assert not exact_overlap(other, ref, window=32)
