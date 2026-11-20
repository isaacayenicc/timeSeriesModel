import numpy as np
from numpy.testing import assert_allclose
from sklearn.linear_model import Lasso

from gearbox_dhon.baselines import BaselineRecord, admm_path, apply_lpv, design, fold_rows


def test_lag_major_polynomial_and_one_step_alignment():
    x = np.arange(1, 9, dtype=float)
    w = np.linspace(0, 1, len(x))
    r = BaselineRecord(x, 100 + w, w, np.arange(len(x)), 1.)
    targets = np.arange(2, len(x))
    beta = np.array([2., .1, -.1, 1.])
    x_reg = design(r, targets, 2, 2, basis="monomial")
    expected = [beta[0] * x[t - 1] + beta[1] * x[t - 1] * w[t - 1] +
                beta[2] * x[t - 2] + beta[3] * x[t - 2] * w[t - 1] for t in targets]
    assert_allclose(x_reg @ beta, expected)
    assert_allclose(apply_lpv(r, beta, 2, 2, basis="monomial")[targets], expected)


def test_admm_converges_to_independent_lasso_solution():
    rng = np.random.default_rng(174)
    x = rng.normal(size=(150, 9))
    y = x[:, 0] - 2 * x[:, 2] + .4 * rng.normal(size=len(x))
    path = list(admm_path(x, y, (.13, .041, .013), rtol=1e-7))
    for fitted, info in path:
        reference = Lasso(alpha=info["lambda_normalized"], fit_intercept=False, tol=1e-10).fit(x, y)
        assert info["converged"]
        assert_allclose(fitted, reference.coef_, rtol=1e-4, atol=1e-5)


def test_fold_training_raw_support_excludes_held_out_targets():
    a, b = 11200, 13600
    train, val = fold_rows(24000, a, b)
    assert val[0] == a and val[-1] == b - 1
    assert np.all((train < a) | (train - 1792 >= b))
