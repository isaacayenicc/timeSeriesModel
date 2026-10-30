"""Analytic and hand-computed diagnostics checks; no manufactured fault labels."""

import numpy as np
import pytest
from scipy.signal import hilbert

from rotmach.angle_resample import angle_resample
from rotmach.diagnostics import (
    analyze_records, analytic_envelope, correlated_kurtosis, normalized_acf,
    pmi, signal_features,
)
from rotmach.spectral import amplitude_spectrum


def test_statistics_hand_computed():
    y = np.array([-1.0, 1.0, -1.0, 1.0])
    f = signal_features(y)
    assert f == {"rms": 1.0, "crest_factor": 1.0, "kurtosis": 1.0}
    f = signal_features(np.array([0.0, 0.0, 2.0, 0.0]))
    assert f["rms"] == 1
    assert f["crest_factor"] == 2
    assert f["kurtosis"] == pytest.approx(7 / 3)


def test_zero_and_constant_statistics_are_undefined_where_required():
    f = signal_features(np.zeros(10))
    assert f["rms"] == 0
    assert np.isnan(f["crest_factor"]) and np.isnan(f["kurtosis"])
    assert np.isnan(signal_features(np.ones(10))["kurtosis"])


def test_statistics_are_scale_robust():
    x = np.array([-2.0, -1.0, 0, 1, 7])
    a, b = signal_features(x), signal_features(1e150 * x)
    assert b["rms"] / 1e150 == pytest.approx(a["rms"])
    assert b["kurtosis"] == pytest.approx(a["kurtosis"])


def test_ck_hand_computed_and_exact_padding():
    y = np.array([1.0, 2.0, 3.0])
    assert correlated_kurtosis(y, 1, 1) == pytest.approx(40 / 196)
    assert correlated_kurtosis(y, 2, 1) == pytest.approx(9 / 196)
    assert correlated_kurtosis(y, 1, 2) == pytest.approx(36 / 14**3)
    assert correlated_kurtosis(y, 4, 1) == 0
    assert correlated_kurtosis(y, 1, 0) == pytest.approx(1)


def test_ck_scale_and_zero_energy():
    y = np.arange(1, 9, dtype=float)
    assert correlated_kurtosis(y, 2, 2) == pytest.approx(
        correlated_kurtosis(y * 1e100, 2, 2)
    )
    assert np.isnan(correlated_kurtosis(np.zeros(20), 2))


@pytest.mark.parametrize("lag", [0, -2, 1.5, True])
def test_ck_rejects_nonpositive_or_noninteger_lag(lag):
    with pytest.raises(ValueError):
        correlated_kurtosis(np.ones(10), lag)


def test_acf_known_centered_sequence_and_biased_normalization():
    # z = [-1,0,1]; denominator 2; lag 1 numerator 0, lag 2 -1.
    assert normalized_acf(np.array([0.0, 1.0, 2.0])) == pytest.approx([1, 0, -0.5])
    assert np.isnan(normalized_acf(np.ones(10))).all()


@pytest.mark.parametrize("n", [63, 64])
def test_fft_unit_amplitude_sinusoid_odd_even(n):
    k = (n - 1) // 2
    y = np.cos(2 * np.pi * k * np.arange(n) / n)
    freq, a = amplitude_spectrum(y, fs=n)
    assert freq[k] == k
    assert a[k] == pytest.approx(1, abs=1e-13)
    _, reference = amplitude_spectrum(y, fs=n, variant="faithful_reference")
    assert reference[k] == pytest.approx(0.5 if n % 2 else 1, abs=1e-13)


def test_fft_even_nyquist_not_doubled_and_dc():
    y = (-1.0) ** np.arange(64)
    freq, amplitude = amplitude_spectrum(y, 64)
    assert freq[-1] == 32
    assert amplitude[-1] == pytest.approx(1)
    assert amplitude_spectrum(np.ones(20) * 3, 20)[1][0] == 3


def test_hilbert_envelope_amplitude_modulation():
    n = np.arange(4000)
    expected = 1 + 0.5 * np.cos(2 * np.pi * 10 * n / 1000)
    y = expected * np.cos(2 * np.pi * 200 * n / 1000)
    assert analytic_envelope(y) == pytest.approx(expected, abs=1e-12)
    shifted = y + 2
    assert analytic_envelope(shifted, "matlab_mean_restored_candidate") == pytest.approx(
        np.abs(hilbert(shifted - shifted.mean())) + shifted.mean()
    )


def test_pmi_literal_matlab_lag_offset_and_negative_values():
    n = np.arange(4000)
    y = (1 + 0.5 * np.cos(2 * np.pi * n / 100)) * np.cos(2 * np.pi * 0.2 * n)
    corrected = pmi(y, 100)
    reference = pmi(y, 100, variant="faithful_reference")
    assert corrected["selected_lag_samples"] == 100
    assert reference["selected_lag_samples"] == 99
    assert reference["source_timeindex_matlab"] == 101
    assert corrected["rho"] == pytest.approx(0.975, abs=1e-12)
    assert corrected["pmi"] == pytest.approx(39, abs=1e-9)
    negative = pmi(y, 50)
    assert negative["rho"] < 0 and negative["pmi"] < 0


def test_pmi_no_infinite_fabrication_on_constant_envelope():
    assert np.isnan(pmi(np.zeros(100), 10)["pmi"])
    n = np.arange(1000)
    carrier = np.cos(2 * np.pi * 0.2 * n)
    assert np.isnan(pmi(carrier, 10)["pmi"])
    # The historic lag-zero bug when period rounds to one is also guarded.
    rng = np.random.default_rng(1)
    output = pmi(rng.normal(size=100), 1, variant="faithful_reference")
    assert output["status"] == "undefined_near_unit_acf"


def test_rounding_is_matlab_positive_half_up_not_python_bankers():
    assert pmi(np.random.default_rng(1).normal(size=100), 2.5)[
        "expected_period_samples"
    ] == 3


@pytest.mark.parametrize("unit,speed", [("rps", 10), ("rpm", 600), ("rad_s", 20*np.pi)])
def test_angle_constant_speed_unit_equivalence(unit, speed):
    time = np.arange(1001) / 1000
    y = np.sin(2 * np.pi * 10 * time)
    output = angle_resample(y, time, np.ones_like(time)*speed, unit, 100)
    assert output["total_measured_cycles"] == pytest.approx(10)
    assert output["signal"] == pytest.approx(
        np.sin(2 * np.pi * output["cycles"]), abs=1e-12
    )
    assert output["roundtrip_rmse"] < 1e-10
    assert output["edge_loss_cycles"] <= 1 / 100 + 1e-12


def test_angle_variable_speed_chirp_known_order():
    time = np.arange(20001) / 10000
    speed = 5 + 2 * time
    cycles = 5 * time + time**2
    y = np.sin(2 * np.pi * 3 * cycles)
    output = angle_resample(y, time, speed, "rps", 256)
    expected = np.sin(2 * np.pi * 3 * output["cycles"])
    # Linear interpolation error <= max|d2x/dt2| * dt^2 / 8. For a phase chirp,
    # |d2x/dt2| <= (2*pi*order*max(speed))^2 + 2*pi*order*max(speed derivative).
    bound = ((2*np.pi*3*speed.max())**2 + 2*np.pi*3*2) * (1/10000)**2 / 8
    assert np.max(np.abs(output["signal"] - expected)) < bound
    assert np.max(output["time"]) <= time[-1]
    assert output["cycles"][0] == 0
    assert np.diff(output["cycles"]) == pytest.approx(np.ones(len(expected)-1)/256)


def test_periodic_impulse_envelope_prefers_matching_period():
    # Periodic amplitude pulses on a high-frequency carrier validate a known
    # synthetic order. This is not a real gear-tooth crack detection test.
    n = np.arange(8192)
    phase = n % 128
    pulses = np.exp(-0.5 * (np.minimum(phase,128-phase)/5)**2)
    y = pulses * np.cos(2 * np.pi * 0.25 * n)
    assert pmi(y, 128)["rho"] > pmi(y, 64)["rho"] + 0.5


@pytest.mark.parametrize("failure", ["unknown", "negative_speed", "bad_time", "shape"])
def test_angle_rejects_invalid_physics_and_shapes(failure):
    t = np.arange(10) / 100
    speed = np.ones(10) * 10
    unit = "unknown" if failure == "unknown" else "rps"
    if failure == "negative_speed": speed[2] = -1
    if failure == "bad_time": t[3] = t[2]
    y = np.ones((2, 5)) if failure == "shape" else np.ones(10)
    with pytest.raises(ValueError):
        angle_resample(y, t, speed, unit)


def test_record_boundary_isolation_common_burnin_and_unknown_units():
    records = [
        {"x": np.full(20, value), "t": np.arange(20)/10, "spd": np.ones(20)}
        for value in (1.0, 10.0)
    ]
    def persistence(record):
        return np.r_[np.nan, record["x"][:-1]]
    rows = analyze_records(records, {"persistence": persistence}, burnin=2)
    assert len(rows) == 4
    assert [r["rms"] for r in rows] == [1, 0, 10, 0]
    assert all(r["n_evaluated"] == 18 for r in rows)
    assert len({r["record_id"] for r in rows}) == 2
    assert all(r["angular_status"] == "blocked_unknown_speed_unit" for r in rows)


def test_nonfinite_and_matrix_input_rejected():
    for y in (np.array([1, np.nan]), np.ones((2, 2))):
        with pytest.raises(ValueError): signal_features(y)
