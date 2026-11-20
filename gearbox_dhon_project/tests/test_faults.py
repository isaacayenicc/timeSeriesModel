"""Numerical references for angular spectra, PMI and diagnostic isolation."""

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

from gearbox_dhon.data import Record
from gearbox_dhon.faults import (
    _changed_record,
    _exact_reference_overlap,
    _pulse,
    analytic_envelope,
    angular_track,
    correlated_kurtosis,
    diagnostics,
    order_spectrum,
    pmi_from_envelope,
)


def test_centered_acf_finite_record_denominator_and_lag():
    points = np.array([5., 3., 5., 3.])
    lag1 = pmi_from_envelope(points, 1)
    lag2 = pmi_from_envelope(points, 2)
    assert lag1.correlation == -3 / 4
    assert lag2.correlation == 1 / 2
    assert lag1.value == -3 / 7
    assert lag2.value == 1
    assert pmi_from_envelope(points + 20, 2) == lag2
    assert pmi_from_envelope(np.ones(10), 1).status == "zero_envelope_variance"
    assert pmi_from_envelope(points, 4).status == "insufficient_angular_history"


def test_fft_hilbert_known_amplitude_and_matlab_dc_handling():
    theta = np.arange(2048) / 2048
    amplitude = 1 + 0.4 * np.cos(2 * np.pi * 3 * theta)
    carrier = amplitude * np.cos(2 * np.pi * 37 * theta)
    assert np.allclose(analytic_envelope(carrier + 7, matlab_envelope=True), amplitude + 7, atol=1e-12)
    assert np.allclose(analytic_envelope(carrier, matlab_envelope=False), amplitude, atol=1e-12)


def test_order_amplitude_and_variable_speed_peak():
    t = np.arange(64000) / 3200
    rpm = 125.0 + 40.0 * t / t[-1]
    phase = cumulative_trapezoid(rpm / 60.0, t, initial=0)
    x = 1.8 * np.cos(2 * np.pi * 37 * phase)
    theta, angular = angular_track(x, rpm, t, samples_per_rev=2048)
    order, amplitude = order_spectrum(angular, 2048)
    peak = np.argmax(amplitude)
    assert np.allclose(np.diff(theta), 1 / 2048)
    assert 0 <= phase[-1] - theta[-1] < 1 / 2048
    assert abs(order[peak] - 37) < 1 / theta[-1]
    assert 1.6 < amplitude[peak] < 1.81
    with pytest.raises(ValueError, match="positive"):
        angular_track(x, -rpm, t)


def test_lag_modes_and_reference_autocorr_span():
    rate, spr, rpm = 3200, 1600, 120
    t = np.arange(8 * spr) / rate
    theta = t * rpm / 60
    x = (1 + .4 * np.cos(2 * np.pi * theta)) * np.cos(2 * np.pi * 37 * theta)
    rows, _ = diagnostics(x, np.full(t.shape, rpm), t, spr)
    assert rows["pmi_course_lag"] == spr - 1
    assert rows["pmi_course"] == pytest.approx(6.99, abs=.025)
    assert rows["pmi_lag_corrected"] == pytest.approx(7.0, abs=1e-10)
    assert rows["pmi_math"] == pytest.approx(7.0, abs=1e-10)
    short, _ = diagnostics(x[:2 * spr], np.full(2 * spr, rpm), t[:2 * spr], spr)
    assert short["pmi_status"] == "insufficient_reference_acf_span"
    assert short["pmi_math_status"] == "ok"


def test_correlated_kurtosis_zero_padding():
    actual = correlated_kurtosis(np.array([1, 2, 3, 4.]), 2)
    assert actual == (1 * 9 + 4 * 16) / (1 + 4 + 9 + 16)**2


def test_real_time_pulses_lock_to_angle_and_rpm_perturbation_isolated():
    t = np.arange(32000) / 3200
    rpm = 145 + 22 * np.sin(2 * np.pi * t / 7.5)
    normalized = (rpm - 123) / 44
    rec = Record(np.zeros(len(rpm)), rpm, t, 3200, "test", 1, "fixture", 1, (),
                 speed_normalized=normalized, rpm_min=123, rpm_max=167)
    fork = _changed_record(rec, rpm=rpm * 1.01)
    assert np.array_equal(fork.x, rec.x)
    assert np.array_equal(fork.t, rec.t)
    assert np.allclose(fork.speed_normalized, (fork.rpm - 123) / 44)
    periodic = _pulse(rec, 760, 1, .31, .0038)
    random = _pulse(rec, 760, 1, .31, .0038, kind="aperiodic")
    periodic_pmi = diagnostics(periodic, rpm, t)[0]["pmi_course"]
    random_pmi = diagnostics(random, rpm, t)[0]["pmi_course"]
    assert periodic_pmi > 5 * max(random_pmi, 0.1)


def test_exact_joint_overlap_detects_shift_and_checks_rpm():
    y = np.arange(12) ** 2
    rpm = np.arange(12) / 10 + 125
    t = np.arange(12) / 3200
    reference = Record(y[3:10], rpm[3:10], t[:7], 3200, "sigtrain", 1, "ref", 1, ())
    record = Record(y, rpm, t, 3200, "sigfault1", 1, "fault", 1, ())
    assert _exact_reference_overlap(record, [("sigtrain", reference)]) == "exact_sigtrain_overlap_offset_3"
    fork = _changed_record(record, rpm=rpm + .0001)
    assert _exact_reference_overlap(fork, [("sigtrain", reference)]) == ""
