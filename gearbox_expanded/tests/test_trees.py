"""Regression checks for native/scientific tree checkpoint parity."""
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from gearbox_expanded.trees import candidates, export_estimator, fit_tree, load_ensemble


@pytest.mark.parametrize("family", ["random_forest", "gradient_boosting"])
def test_numeric_tree_export_matches_sklearn(tmp_path, family):
    rng = np.random.default_rng(92)
    x = rng.normal(size=(1300, 5))
    x[::41, 2] = np.nan
    y = 0.5 * x[:, 0] - np.sin(x[:, 1]) + x[:, 0] * x[:, 3]
    _, params = candidates(smoke=True)[family][0]
    with threadpool_limits(limits=1):
        sklearn_estimator = fit_tree(family, params, x[:900], y[:900])
        model = export_estimator(family, "physics", sklearn_estimator)
        path = tmp_path / "trees.npz"
        model.save(path, params)
        replay = load_ensemble(path)
        np.testing.assert_allclose(replay.predict_matrix(x[900:]),
                                   sklearn_estimator.predict(x[900:]), atol=2e-13, rtol=0)
        assert replay.feature_set == "physics"
        assert len(replay.roots) > 1
        with pytest.raises(ValueError, match="input features"):
            replay.predict_matrix(x[900:, :2])


def test_tree_uses_strictly_prior_features():
    from gearbox_expanded.tabular_features import matrix

    rng = np.random.default_rng(7780)
    t = np.arange(2150)
    x = rng.normal(size=t.shape)
    rpm = 145 + 20 * np.sin(t / 400)
    train = np.arange(800, 1600)
    config = candidates(smoke=True)["gradient_boosting"][0][1]
    z = matrix(x, rpm, 3200, train, "physics_compact")
    with threadpool_limits(limits=1):
        native = fit_tree("gradient_boosting", config, z, x[train])
    trained = export_estimator("gradient_boosting", "physics_compact", native)
    expected = trained.predict(x, rpm, 3200)
    change = 1950
    x[change:] = 12.4
    rpm[change:] = 185.8
    actual = trained.predict(x, rpm, 3200)
    np.testing.assert_array_equal(actual[:change + 1], expected[:change + 1])
    np.testing.assert_array_equal(trained.predict(x[:change + 1], rpm[:change + 1], 3200),
                                  actual[:change + 1])
