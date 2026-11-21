"""Verify causal window boundaries and attention structure."""

from dataclasses import asdict

import numpy as np
import torch

from gearbox_expanded.attention import Config, Predictor, _windows, fit, load


def test_causal_window_target_alignment():
    x = np.arange(13, dtype=np.float64)
    rpm = 130 + x
    windows = _windows(x, rpm, np.zeros(2), np.ones(2), history=4)
    assert windows.shape == (9, 4, 2)
    for target in range(4, len(x)):
        np.testing.assert_array_equal(windows[target-4, :, 0], x[target-4:target])
        np.testing.assert_array_equal(windows[target-4, :, 1], rpm[target-4:target])


def test_transformer_internal_mask_blocks_suffix_patches():
    torch.set_num_threads(1)
    config = Config("transformer", 24, 8, 2, 2, 1, .001, patch=4, dropout=0.)
    model = config.build().eval()
    x = torch.randn(2, 6, 8)
    with torch.inference_mode():
        initial = model.encoder(x, mask=model.mask, is_causal=True)
        change = x.clone()
        change[:,3:] += torch.randn_like(change[:,3:]) * 10
        altered = model.encoder(change, mask=model.mask, is_causal=True)
    torch.testing.assert_close(initial[:,:3], altered[:,:3], atol=2e-7, rtol=0)
    assert not torch.equal(initial[:,3:], altered[:,3:])


def test_tft_weights_are_normalized_and_gradients_reach_attention():
    torch.set_num_threads(1)
    config = Config("tft", 12, 8, 2, 1, 1, .001, dropout=0.)
    model = config.build()
    target, weights, attention = model.inspect(torch.randn(4, 12, 2))
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(4,12))
    torch.testing.assert_close(attention.sum(dim=-1), torch.ones(4))
    target.square().mean().backward()
    assert model.attention.query.weight.grad.norm() > 0
    assert model.selection.weights[-1].weight.grad.norm() > 0
    assert model.recurrent.weight_ih_l0.grad.norm() > 0


def test_checkpoint_and_future_perturbation_for_both_architectures(tmp_path):
    torch.set_num_threads(1)
    t = np.arange(2100, dtype=np.float64)/3200.
    x = np.sin(500*t)
    rpm = 144 + np.sin(40*t)
    for family in ("transformer", "tft"):
        config = Config(family, 12, 8, 2, 1, 1, .002,
                        patch=3 if family=="transformer" else 1, dropout=0.)
        model = fit(asdict(config), [(x[:900],rpm[:900])],3200)
        path = tmp_path/f"{family}.npz"
        assert isinstance(model,Predictor)
        model.save(path)
        expected = model.predict(x,rpm,3200)
        restored = load(path)
        np.testing.assert_array_equal(expected,restored.predict(x,rpm,3200))
        change_x, change_rpm = x.copy(),rpm.copy()
        change_x[1950:] += 4
        change_rpm[1950:] *= 1.04
        got = restored.predict(change_x,change_rpm,3200)
        np.testing.assert_array_equal(expected[:1951],got[:1951])
        assert np.isfinite(got[1792:]).all()
