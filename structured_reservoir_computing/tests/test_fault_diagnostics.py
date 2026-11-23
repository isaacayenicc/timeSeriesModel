"""Numerical oracles for label-free angular and transient diagnostics."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src_gearbox.diagnostics.envelope import amplitude_spectrum, analytic_envelope
from src_gearbox.diagnostics.fault_metrics import _exact_reference_overlap, summarize
from src_gearbox.diagnostics.inverse_filtering import forecast
from src_gearbox.diagnostics.order_tracking import resample, revolutions
from src_gearbox.diagnostics.pmi import measure, pmi_from_envelope
from src_gearbox.diagnostics.speed_stress import PulseCase, inject, speed_cases


def test_finite_normalized_autocorrelation():
    envelope = [5., 3., 5., 3.]
    first = pmi_from_envelope(envelope, 1)
    second = pmi_from_envelope(envelope, 2)
    assert first.correlation == -.75
    assert second.value == 1.
    assert first.value == -3 / 7
    assert pmi_from_envelope(envelope, 4).status == "insufficient_angular_history"
    assert pmi_from_envelope([1, 1, 1], 1).status == "zero_envelope_variance"
    assert pmi_from_envelope([1, np.nan, 1], 1).status == "invalid_envelope"


def test_lecturer_envelope_center_restores_mean():
    angle = np.arange(2048) / 2048
    magnitude = 1 + .4 * np.cos(2 * np.pi * 3 * angle)
    sampled = magnitude * np.cos(2 * np.pi * 37 * angle) + 7
    observed = analytic_envelope(sampled, lecturer_mean_treatment=True)
    np.testing.assert_allclose(observed, magnitude + 7, rtol=0, atol=1e-12)


def test_one_turn_lag_analytical_reference_and_short_span():
    t = np.arange(8 * 1600) / 3200
    phase = 2 * t
    x = (1 + .4 * np.cos(2 * np.pi * phase)) * np.cos(2 * np.pi * 37 * phase)
    speed = np.full_like(x, 120)
    result = measure(x, speed, t, 1600)
    assert result["lecturer_compatible_lag"] == 1599
    assert result["lag_corrected_lag"] == result["mathematically_specified_lag"] == 1600
    assert result["lag_corrected"] == pytest.approx(7, abs=1e-10)
    assert result["mathematically_specified"] == pytest.approx(7, abs=1e-10)
    assert abs(result["lecturer_compatible"] - 7) > .001
    short = measure(x[:3200], speed[:3200], t[:3200], 1600)
    assert short["lecturer_compatible_status"] == "insufficient_lecturer_autocorr_span"
    assert short["mathematically_specified_status"] == "ok"


def test_variable_speed_carrier_and_sample_bounds():
    t = np.arange(32000) / 3200
    n = np.linspace(120, 165, len(t))
    turns = revolutions(n, t)
    x = 1.3 * np.cos(2 * np.pi * 37 * turns)
    grid, angular = resample(x, n, t)
    assert grid[0] == 0
    assert 0 <= turns[-1] - grid[-1] < 1 / 2048
    order, amp = amplitude_spectrum(angular, 2048)
    assert abs(order[np.argmax(amp)] - 37) < 1 / grid[-1]
    for wrong in (-n, n[:-1], np.zeros_like(n)):
        with pytest.raises(ValueError):
            resample(x, wrong, t)
    with pytest.raises(ValueError, match="timestamp"):
        resample(x, n, -t)


def test_paired_periodic_control_and_sensor_alignment():
    t = np.arange(32000) / 3200
    n = 145 + 18 * np.sin(2*np.pi*t / 7.5)
    rec = SimpleNamespace(x=np.zeros_like(t), rpm=n, fs=3200, t=t)
    p = inject(rec, PulseCase(760,.6,.25,.004))
    a = inject(rec, PulseCase(760,.6,.25,.004,kind="aperiodic"))
    periodic = measure(p, n, t)["lecturer_compatible"]
    aperiodic = measure(a, n, t)["lecturer_compatible"]
    assert periodic > 5 * max(.1, aperiodic)
    speed_variants = speed_cases(rec)
    np.testing.assert_array_equal(speed_variants["delay_40ms"][:128], np.repeat(n[0],128))
    np.testing.assert_allclose(speed_variants["bias_plus_2pct"], n*1.02)
    assert all(s.shape == n.shape and np.all(s > 0) for s in speed_variants.values())


def test_exact_overlap_requires_joint_match_and_forecast_alignment():
    x = np.arange(30)**2
    n = np.arange(30) / 10 + 120
    record = SimpleNamespace(x=x, rpm=n, fs=3200)
    refs = {"sigtrain": SimpleNamespace(x=x[9:22], rpm=n[9:22])}
    assert _exact_reference_overlap(record, refs) == "exact_sigtrain_overlap_offset_9"
    refs["sigtrain"].rpm = refs["sigtrain"].rpm + 1
    assert _exact_reference_overlap(record, refs) == ""
    with pytest.raises(ValueError, match="per input sample"):
        forecast(lambda x, n, fs: x[1:], record, start=1)
    np.testing.assert_array_equal(forecast(lambda x,n,fs:x, record, start=3), x[3:])


def test_summary_uses_matched_eligible_acquisitions_and_ties(tmp_path):
    records = []
    for name, values in (("x", (0.,1.,1.,9.)), ("y", (0.,1.,1.,np.nan))):
        for index, score in enumerate(values, 1):
            records.append(dict(model=name,record_id=index,group="A" if index<3 else "B",
                                eligible=True,lecturer_compatible=score))
    result = summarize(pd.DataFrame(records), tmp_path, primary="x", n_boot=50)
    table = pd.read_csv(tmp_path / "fault_group_summary.csv")
    assert result["summary_rows"] == 2
    assert result["paired_comparisons"] == 1
    assert table.n_A.tolist() == [2,2]
    assert table.n_B.tolist() == [1,1]
    assert table.conditional_auc_B_high.tolist() == [.75,.75]
