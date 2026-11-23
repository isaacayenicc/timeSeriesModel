"""Test finite support, causal feature parity and numeric SRC checkpoints."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from src_gearbox.src_model import (
    Config, angular_lags, fit, load_model, oscillator_states, phase_turns,
    receptive_fields, slow_adapter, solve_readout, time_lags,
)


def samples(n):
    rng = np.random.default_rng(42)
    rpm = np.linspace(123.,166.,n)
    turns, _ = phase_turns(rpm,3200)
    x = np.cos(2*np.pi*37*turns) + .25*np.sin(2*np.pi*74*turns)
    x += .1*rng.normal(size=n)
    return x,rpm


def test_stable_phase_on_constant_speed_and_chirp():
    fs = 3200
    constant = np.full(520,144.)
    turns, goal = phase_turns(constant,fs)
    np.testing.assert_allclose(turns,np.arange(len(turns))*144/60/fs,atol=1e-14)
    np.testing.assert_allclose(goal,turns,atol=1e-14)
    x,rpm = samples(1000)
    observed,target = phase_turns(rpm,fs)
    grids = angular_lags(1+2*observed,rpm,fs,np.arange(250,1000),8,2)
    expected = 1+2*(target[250:,None]-np.arange(1,17)/(37*8))
    np.testing.assert_allclose(grids,expected,atol=2e-7,rtol=0)


@pytest.mark.parametrize("carrier",[{"order":37},{"hz":1100.}])
def test_oscillator_direct_sum_bounded_reset_and_truncation(carrier):
    n = 270
    wave,rpm = samples(n)
    m,tau,fs = 51,42.,3200
    if "order" in carrier:
        history_phase,target_phase = [2*np.pi*37*values for values in phase_turns(rpm,fs)]
    else:
        history_phase = target_phase = 2*np.pi*1100*np.arange(n)/fs
    kernel = np.exp(-np.arange(m)/tau)
    kernel *= 2./kernel.sum()
    pulse = np.zeros(n)
    pulse[47] = 1.
    for x in (wave,pulse,np.zeros(n)):
        got = oscillator_states(x,fs,rpm,tau=tau,memory=m,**carrier)
        direct = np.zeros(n,dtype=complex)
        for t in range(n):
            j = np.arange(1,min(t,m)+1)
            direct[t] = np.sum(kernel[:len(j)]*x[t-j]*
                               np.exp(1j*(target_phase[t]-history_phase[t-j])))
        np.testing.assert_allclose(got,direct,atol=3e-13,rtol=0)
        assert np.max(np.abs(got))<=2*np.max(np.abs(x))+1e-12
    decay = oscillator_states(pulse,fs,rpm,tau=tau,memory=m,**carrier)
    assert abs(decay[47])==0
    assert abs(decay[47+m])>0
    assert np.max(np.abs(decay[48+m:]))<1e-14


def test_delay_choice_and_exact_nested_history_bound():
    c = replace(Config(),lags=32,lag_history=256,lag_policy="multiresolution",
                mesh_cycles=0,harmonics=(),innovation_order=100,innovation_lags=24)
    chosen = time_lags(c)
    assert len(chosen)==32 and chosen[0]==1 and chosen[-1]==256
    assert np.all(np.diff(chosen)>0)
    assert receptive_fields(c,3200)==(256,1280)
    assert receptive_fields(replace(c,lag_policy="consecutive"),3200)==(124,1148)


def test_reference_readout_weighted_ridge_and_rank_deficiency():
    rng=np.random.default_rng(84)
    x=np.column_stack([np.ones(130),rng.normal(size=(130,7))])
    y=.4*x[:,2]+.3*x[:,5]
    p=np.r_[0.,np.full(7,.012)]
    target=np.linalg.solve(x.T@x+len(y)*np.diag(p),x.T@y)
    for height in (25,300):
        b,detail=solve_readout(x,y,p,block_size=height)
        np.testing.assert_allclose(b,target,atol=5e-15,rtol=0)
        assert detail["backend"]=="cholesky"
    deficient=np.column_stack([x,x[:,2]])
    b,diag=solve_readout(deficient,y,0.,solver="ols")
    assert diag["backend"]=="pivoted_qr"
    np.testing.assert_allclose(deficient@b,y,atol=2e-14,rtol=0)
    contaminated=y.copy()
    contaminated[4]+=100
    robust,_=solve_readout(x,contaminated,p,solver="huber",huber_delta=.5)
    plain,_=solve_readout(x,contaminated,p)
    assert np.linalg.norm(robust-target)<np.linalg.norm(plain-target)


def test_adapter_delays_impulse_and_finite_decay():
    y=np.zeros(90)
    y[12]=15
    got=slow_adapter(y,np.zeros(90),gain=.6,tau=11,memory=20,clip=1.75)
    kernel=np.exp(-np.arange(20)/11)
    expected=np.zeros(90)
    expected[13:33]=.6*1.75*kernel/kernel.sum()
    np.testing.assert_allclose(got,expected,atol=1e-14,rtol=0)


@pytest.mark.parametrize("variant",["SRC-Linear","SRC-Polynomial","SRC-Speed","SRC-Physics-Linear",
                                    "SRC-Generic","PI-SRC-Fixed","PI-SRC-Optimized",
                                    "PI-SRC-Corrected","PI-SRC-Compact"])
def test_model_variant_causal_checkpoint_and_record_reset(tmp_path,variant):
    x,rpm=samples(950)
    c=replace(Config(),variant=variant,lags=16,mesh_cycles=2,angular_steps=8,
              harmonics=(37,74),fixed_hz=(1070.,),taus=(7.,31.),memory=100,
              innovation_order=14,innovation_lags=6,nonlinear_cap=15,
              augmentation=0.,pulse_preservation=0.,adapter_memory=76,rank=5)
    model=fit(c,[(x[:320],rpm[:320]),(x[500:],rpm[500:])],3200)
    names,groups,_=zip(*model.readout_features)
    assert len(names)==len(model.coefficients)
    assert len(set(names))==len(names)
    physics="oscillator" in groups
    nonlinear=any(g.startswith("nonlinear") for g in groups)
    if variant in {"SRC-Linear","SRC-Speed","SRC-Physics-Linear"}:
        assert not nonlinear
    else:
        assert nonlinear
    assert physics == (variant=="SRC-Physics-Linear" or variant.startswith("PI-"))
    y=model.predict(x,rpm,3200)
    assert np.isnan(y[:model.history]).all()
    assert np.isfinite(y[model.history:]).all()
    assert np.isnan(model.predict(x[:50],rpm[:50],3200)).all()
    x_future=x.copy()
    r_future=rpm.copy()
    t=640
    x_future[t:]+=2.5
    r_future[t:]*=1.04
    altered=model.predict(x_future,r_future,3200)
    np.testing.assert_allclose(y[model.history:t+1],altered[model.history:t+1],atol=1e-11,rtol=0)
    checkpoint=tmp_path/"m.npz"
    model.save(checkpoint)
    restored=load_model(checkpoint)
    np.testing.assert_array_equal(restored.predict(x,rpm,3200),y)
    record=SimpleNamespace(x=x,rpm=rpm,t=np.arange(len(x))/3200,fs=3200)
    result=restored.predict_one_step(record,np.arange(600,620))
    np.testing.assert_array_equal(result["prediction"],y[600:620])
    assert result["mask"].sum()==20


def test_crossfit_innovations_retains_replay_behavior():
    x,rpm=samples(960)
    c=replace(Config(),lags=16,mesh_cycles=0,angular_steps=8,
              harmonics=(37,),taus=(7.,31.),memory=48,innovation_order=12,
              innovation_lags=4,innovation_folds=3,adapter_gain=0.,nonlinear_cap=8)
    crossfit=fit(c,[(x[:450],rpm[:450]),(x[500:],rpm[500:])],3200)
    direct=fit(replace(c,innovation_folds=0),[(x[:450],rpm[:450]),(x[500:],rpm[500:])],3200)
    assert crossfit.solver_info["innovation_crossfit"]==3
    np.testing.assert_array_equal(crossfit.proxy,direct.proxy)
    assert not np.array_equal(crossfit.coefficients,direct.coefficients)
    assert np.isfinite(crossfit.predict(x,rpm,3200)[crossfit.history:]).all()


def test_segmented_inference_matches_full_record_features():
    from src_gearbox.src_model import _library
    x,rpm=samples(6100)
    c=replace(Config(),lags=32,mesh_cycles=4,angular_steps=12,harmonics=(37,74,259),
              fixed_hz=(1100.,),taus=(6.,30.,83.),memory=256,innovation_order=48,
              innovation_lags=12,nonlinear_cap=32,adapter_gain=0.,pulse_preservation=0.,augmentation=0.)
    model=fit(c,[(x[:1500],rpm[:1500])],3200)
    idx=np.arange(model.history,len(x))
    full,_,_=_library(x,rpm,3200,idx,c,model.speed,model.proxy)
    direct=model.coefficients[0]+model._select(full)@model.coefficients[1:]
    measured=model.predict_base(x,rpm,3200)[idx]
    np.testing.assert_allclose(measured,direct,atol=4e-7,rtol=0)
