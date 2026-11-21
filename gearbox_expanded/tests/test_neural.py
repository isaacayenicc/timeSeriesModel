import numpy as np
import pytest
import torch

from gearbox_expanded.neural import (
    Config, DilatedConvolutions, FAMILIES, Model, NeuralForecaster, _windows, load,
)


def test_window_is_ordered_and_strictly_prior():
    x = np.arange(20, dtype=float)
    targets = np.array([8, 14, 19])
    windows = _windows(x, x + 100, targets, 8, np.zeros(2), np.ones(2)).numpy()
    for index, target in enumerate(targets):
        np.testing.assert_array_equal(windows[index, :, 0], x[target - 8:target])
        np.testing.assert_array_equal(windows[index, :, 1], x[target - 8:target] + 100)


@pytest.mark.parametrize("family", FAMILIES)
def test_neural_checkpoint_and_future_invariance(family, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(481)
    rng = np.random.default_rng(73)
    x = rng.normal(size=210)
    rpm = rng.uniform(120, 160, size=x.size)
    target = 170
    config = Config(family, 128, 8, 1, 0.002)
    model = Model(config, np.array([0.0, 140.0]), np.array([1.0, 12.0]), 3200,
                  NeuralForecaster(config).eval())
    targets = np.arange(target - 20, target + 1)
    original = model.predict(x, rpm, 3200, targets)
    x[target:] += 100
    rpm[target:] *= 3
    np.testing.assert_array_equal(model.predict(x, rpm, 3200, targets), original)
    path = tmp_path / "saved.npz"
    model.save(path)
    restored = load(path)
    np.testing.assert_array_equal(restored.predict(x, rpm, 3200, targets), original)


def test_tcn_last_patch_can_use_full_24_patch_history():
    torch.set_num_threads(1)
    network = DilatedConvolutions(24, 2)
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.fill_(0.2)
    series = torch.ones(1, 24, 16, requires_grad=True)
    network(series).sum().backward()
    assert torch.all(series.grad > 0)
