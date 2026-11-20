"""Mechanistic checks for finite oscillator state, training and causal deployment."""

from dataclasses import replace

import numpy as np
import torch

from gearbox_dhon.dhon import (
    BASE_HISTORY, SCORE_START, Config, angular_lags, fit, load_model,
    oscillator_recurrence, oscillator_states, shaft_phase, slow_adapter,
)
from gearbox_dhon.dhon_search import FOLDS, fold_segments

torch.set_num_threads(1)


def direct_sum(x, phase, taus, memory):
    """Deliberately independent, slow summation with explicit target indexing."""
    result = np.zeros((len(x), *taus.shape), dtype=np.complex128)
    kernel = np.exp(-np.arange(memory) / taus[..., None])
    kernel *= 2 / kernel.sum(axis=-1, keepdims=True)
    for t in range(len(x)):
        j = np.arange(1, min(t, memory) + 1)
        change = phase[:, t, None] - phase[:, t-j]
        result[t] = (kernel[..., :len(j)] * x[t-j] * np.exp(1j * change[:, None])).sum(-1)
    return np.stack([result.real, result.imag], axis=-1)


def test_fft_and_recurrence_match_chirp_harmonic_impulse_and_zero():
    fs = 1000
    time = np.arange(160) / fs
    rpm = np.linspace(118, 175, len(time))
    order = 37 * 2 * np.pi * shaft_phase(rpm, fs)
    phase = np.stack([order, 2 * np.pi * 210 * time])
    taus = np.array([[11., 47.], [31., 79.]])
    memory = 49
    impulse = np.zeros(len(time))
    impulse[51] = 1
    for x in (np.cos(order) + .4 * np.sin(phase[1]), impulse, impulse * 0):
        direct = direct_sum(x, phase, taus, memory)
        fft = oscillator_states(torch.tensor(x), torch.tensor(phase),
                                torch.tensor(taus), memory).numpy()
        recurrence = oscillator_recurrence(x, phase, taus, memory)
        np.testing.assert_allclose(fft, direct, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(recurrence, direct, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(oscillator_recurrence(x, phase + 4.72, taus, memory),
                                   direct, rtol=2e-12, atol=2e-12)
        assert np.max(np.linalg.norm(fft, axis=-1)) <= 2 * np.max(np.abs(x)) + 1e-12
    assert np.linalg.norm(direct_sum(impulse, phase, taus, memory)[100]) > 0
    assert np.count_nonzero(direct_sum(impulse, phase, taus, memory)[101:]) == 0


def test_gradcheck_for_order_and_decay():
    x = torch.linspace(-1, 1, 31, dtype=torch.float64).cos()
    turns = torch.linspace(0, .025, len(x), dtype=torch.float64)
    order = torch.tensor([37.4], dtype=torch.float64, requires_grad=True)
    decay = torch.tensor([[24., 70.]], dtype=torch.float64, requires_grad=True)

    def state(frequency, memory_constant):
        return oscillator_states(x, 2 * torch.pi * frequency[:, None] * turns, memory_constant, 14)

    assert torch.autograd.gradcheck(state, (order, decay), eps=1e-5, atol=1e-7, rtol=1e-4)
    phase_gradient, decay_gradient = torch.autograd.grad(state(order, decay)[-1].square().sum(),
                                                         (order, decay))
    assert 1e-7 < phase_gradient.norm() < 10
    assert 1e-7 < decay_gradient.norm() < 1


def test_uniform_mesh_interpolation_queries_never_reach_target():
    rpm = np.linspace(130, 160, 2000)
    turns = shaft_phase(rpm, 3200)
    x = 2 * turns + 1
    target = np.arange(400, 1000)
    got = angular_lags(x, turns, target, 12, 4)
    grid = turns[target - 1, None] - np.arange(1, 49) / (37 * 12)
    np.testing.assert_allclose(got, 2*grid + 1, atol=2e-14, rtol=0)
    assert np.all(grid[:, 0] < turns[target - 1])


def test_guarded_fit_segments_have_disjoint_contiguous_raw_support():
    sample = np.arange(24000)
    for a, b in FOLDS:
        segments = fold_segments(sample, sample, (a, b))
        assert segments[0][0][-1] < a - SCORE_START
        assert segments[1][0][0] >= b + SCORE_START
        for signals, _ in segments:
            assert np.all(np.diff(signals) == 1)
        assert len(fold_segments(sample, sample, (a, b), chronological=True)) == 1


def test_adapter_clips_and_never_predicts_from_current_error():
    p = np.zeros(24)
    x = p.copy()
    x[10] = 7
    actual = slow_adapter(x, p, gain=.5, tau=3, memory=5, clip=1.75)
    kernel = np.exp(-np.arange(5)/3)
    expected = p.copy()
    expected[11:16] = .5 * 1.75 * kernel/kernel.sum()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
    assert actual[10] < 1e-14


def test_training_changes_physics_and_checkpoint_causal_reset(tmp_path):
    rpm = np.linspace(122., 166., 4100)
    turns = shaft_phase(rpm, 3200)
    x = np.cos(2 * np.pi * 36.7 * turns) + .4 * np.sin(2 * np.pi * 77. * turns)
    x += .35 * np.random.default_rng(18).standard_normal(len(x))
    c = replace(Config(), lags=24, angular_steps=8, mesh_cycles=2, harmonics=(1, 2),
                memory=96, taus=(8., 32.), innovation_order=24, innovation_lags=4,
                epochs=5, augment=0., preserve=0.)
    model, trace = fit(c, [(x, rpm)], 3200, 4, "cpu")
    fixed, _ = fit(replace(c, joint="fixed", epochs=0), [(x, rpm)], 3200, 4, "cpu")
    trainable, _ = fit(replace(c, epochs=0), [(x, rpm)], 3200, 4, "cpu")
    np.testing.assert_array_equal(trainable.predict(x, rpm, 3200), fixed.predict(x, rpm, 3200))
    assert trace[-1]["carrier_gradient_norm"] > 1e-8
    assert trace[-1]["decay_gradient_norm"] > 1e-8
    learned, decay = model.physical()
    assert torch.any((learned - model.carrier_prior).abs() > 1e-5)
    assert torch.any((decay - model.tau_prior).abs() > 1e-5)
    assert torch.all((decay > 4) & (decay < 384))
    state = tmp_path / "checkpoint.npz"
    model.save(state)
    expected = model.predict(x, rpm, 3200)
    restored = load_model(state)
    np.testing.assert_array_equal(restored.predict(x, rpm, 3200), expected)
    boundary = 2500
    altered = x.copy()
    altered[boundary:] += 2.5
    r = rpm.copy()
    r[boundary:] *= 1.04
    np.testing.assert_allclose(model.predict(altered, r, 3200)[SCORE_START:boundary+1],
                               expected[SCORE_START:boundary+1], atol=3e-5, rtol=0)
    model.predict(altered, r, 3200)
    np.testing.assert_array_equal(model.predict(x, rpm, 3200), expected)
    assert np.isnan(expected[:BASE_HISTORY]).all()
    assert np.isfinite(expected[SCORE_START:]).all()
