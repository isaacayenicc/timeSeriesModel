"""Independent numerical and causal checks for the comparison suite."""

import numpy as np
import pytest
import torch

from gearbox_dhon.comparators import (
    ANGLE_HISTORY, Config, FOLD_RANGES, START, _bounded_angular_lags, feature_matrix,
    fit, fold_targets, load_comparator,
)


def _record():
    rng = np.random.default_rng(7780)
    n = 2100
    t = np.arange(n) / 3200.0
    return rng.normal(size=n), 145 + 15 * np.cos(2 * np.pi * t), 3200.0


def test_guarded_rows_have_disjoint_raw_input_support():
    n = 24000
    for (start, stop), (training, heldout) in zip(FOLD_RANGES, fold_targets(n), strict=True):
        np.testing.assert_array_equal(heldout, np.arange(start, stop))
        assert np.all((training < start - START) | (training >= stop + START))
        assert np.all((training < start) | (training - ANGLE_HISTORY >= stop))


def test_constant_rpm_angular_delay_matches_linear_phase():
    fs = 3200.0
    rpm = np.full(2100, 150.0)
    target = np.arange(1792, 1830)
    azimuth = np.arange(len(rpm)) * rpm[0] / (60.0 * fs)
    x = 3.0 * azimuth + 12.0
    delays = np.array([1 / 37, 2 / 37, 3 / 37, 4 / 37, 6 / 37, 0.25, 0.375])
    expected = 12.0 + 3.0 * (azimuth[target, None] - delays)
    np.testing.assert_allclose(_bounded_angular_lags(x, rpm, fs, target), expected,
                               atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("config", [
    Config("persistence"),
    Config("ridge", lag_count=48),
    Config("ridge", lag_count=48, rpm_degree=2, angular=True, orders=True, polynomial_degree=3),
    Config("mlp", lag_count=48, rpm_degree=1, epochs=1, width=4),
    Config("tcn", rpm_degree=1, epochs=1, width=4),
])
def test_checkpoint_causality_and_record_reset(tmp_path, config):
    torch.set_num_threads(1)
    x, rpm, fs = _record()
    training = np.arange(900, 1200)
    targets = np.arange(START, 1850)
    test_x, test_rpm = x.copy(), rpm.copy()
    test_x[1850:], test_rpm[1850:] = -24, 190
    estimator, _ = fit(x, rpm, fs, config, training)
    full = estimator.predict(x, rpm, fs, targets)
    np.testing.assert_allclose(estimator.predict(test_x, test_rpm, fs, targets),
                               full, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(estimator.predict(x[:1850], rpm[:1850], fs, targets),
                               full, atol=1e-6, rtol=0.0)
    observed_t = targets[-1]
    same_time_change_x, same_time_change_rpm = x.copy(), rpm.copy()
    same_time_change_x[observed_t], same_time_change_rpm[observed_t] = -50, 195
    np.testing.assert_allclose(
        estimator.predict(same_time_change_x, same_time_change_rpm, fs, np.array([observed_t])),
        full[-1:], atol=1e-6, rtol=0.0)
    checkpoint = tmp_path / "weights.npz"
    estimator.save(checkpoint)
    np.testing.assert_array_equal(load_comparator(checkpoint).predict(x, rpm, fs, targets), full)


def test_polynomial_library_size_and_history():
    x, rpm, fs = _record()
    target = np.array([1792, 1880, 1900])
    config = Config("ridge", lag_count=48, rpm_degree=2,
                    angular=True, orders=True, polynomial_degree=3)
    columns = 48 * 3 + 2 + 7 * 2 + 8 * 2 + 36 + 8 + 8 + 4 + 4
    features = feature_matrix(x, rpm, fs, target, config)
    assert features.shape == (3, columns)
    np.testing.assert_array_equal(features[:, :48], x[target[:, None] - np.arange(1, 49)])


def test_heldout_blocks_cannot_change_angular_fit_rows():
    rng = np.random.default_rng(813)
    n = 24000
    rpm = rng.uniform(123.0, 167.0, n)
    x = rng.normal(size=n)
    fit_rows, eval_rows = fold_targets(n)[1]
    # Widely spaced probes cover both sides of the guarded fold.
    probes = fit_rows[::113]
    config = Config("ridge", lag_count=48, rpm_degree=2, angular=True, orders=True)
    clean = feature_matrix(x, rpm, 3200.0, probes, config)
    x[eval_rows] = 100.0
    rpm[eval_rows] = 220.0
    perturbed = feature_matrix(x, rpm, 3200.0, probes, config)
    np.testing.assert_allclose(perturbed, clean, atol=1e-10, rtol=0.0)
