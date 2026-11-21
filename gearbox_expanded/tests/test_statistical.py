import numpy as np
from numpy.testing import assert_allclose

from gearbox_expanded.statistical import (
    SCORE_START, FitSpec, SparseLPV, StatisticalModel, _exact_recursive, _fir,
    fit_model, load,
)


def test_arma_fir_matches_exact_innovation_recursion():
    x = np.random.default_rng(123).normal(size=4200)
    speed = np.full(len(x), 145.0)
    for family in ("arma", "arima"):
        model = StatisticalModel(FitSpec(family, 2, 2), np.array([.5, -.25]), np.array([.29, -.11]))
        actual = model.predict(x, speed, 3200)[SCORE_START:]
        assert_allclose(actual, _exact_recursive(x, model)[SCORE_START:], rtol=0, atol=2e-15)
        assert _fir(model.ar, model.ma, model.spec.d, SCORE_START)[0] == 0


def test_additive_arx_alignment_causality_and_checkpoint(tmp_path):
    rng = np.random.default_rng(124)
    x = rng.normal(size=4000)
    rpm = 145 + rng.normal(size=len(x))
    model = StatisticalModel(FitSpec("arx", 2, speed_lags=(1, 17)), np.array([.2, -.1]),
                             exogenous=np.array([.05, .07]), intercept=-17.4)
    path = tmp_path / "arx.npz"
    model.save(path)
    predicted = load(path).predict(x, rpm, 3200)
    t = 2100
    expected = .2*x[t-1] - .1*x[t-2] + .05*rpm[t-1] + .07*rpm[t-17] - 17.4
    assert_allclose(predicted[t], expected, atol=1e-14)
    future_x, future_rpm = x.copy(), rpm.copy()
    future_x[t:] += 4.
    future_rpm[t:] -= 7.
    mutated = model.predict(future_x, future_rpm, 3200)
    assert_allclose(predicted[SCORE_START:t+1], mutated[SCORE_START:t+1], rtol=0, atol=0)


def test_sparse_stage2_uses_prior_stage1_error():
    x = np.linspace(-1, 1, 4000)
    rpm = np.linspace(125, 160, len(x))
    model = SparseLPV(np.array([120., 170.]), 2,
                      np.array([.4, .2]), np.array([.25, 0., -.3, .1]), 1, 1, 1)
    t = 2200
    w1 = 2 * (rpm[t-2] - 120) / 50 - 1
    w2 = 2 * (rpm[t-1] - 120) / 50 - 1
    error = x[t-1] - (.4 + .2*w1) * x[t-2]
    expected = .25*x[t-1] + (-.3 + .1*w2)*error
    assert_allclose(model.predict(x, rpm, 3200)[t], expected, atol=1e-15)


def test_fitted_arx_recovers_additive_causal_speed():
    rng = np.random.default_rng(130)
    rpm = 145 + rng.normal(size=6000)
    x = np.zeros(len(rpm))
    for t in range(1, len(x)):
        x[t] = .63*x[t-1] + .2*(rpm[t-1]-145) + .015*rng.normal()
    fitted = fit_model([(x, rpm)], FitSpec("arx", 1, speed_lags=(1,), ridge=.0002))
    assert_allclose(fitted.ar, [.63], atol=.015)
    assert_allclose(fitted.exogenous, [.2], atol=.015)
