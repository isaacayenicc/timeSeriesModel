"""Frozen-model fault inference equivalence and angular convention checks."""
from pathlib import Path
import json

import numpy as np
import pytest
import torch

from rotmach.angle_resample import angle_resample
from rotmach.diagnostics import pmi
from rotmach.experiment import load_model_and_predict
from rotmach.fault_study import CachedPredictor, angular_features, select_models
from rotmach.io_matlab import load_records
from rotmach.models import build_network

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("run", ["lpv_L256_s17_a1", "ar_L256_s17_a1", "arx_L256_s17_a1", "ols_L256_s17"])
def test_fir_linear_predictor_matches_original_design(run):
    record=load_records(ROOT/"data/raw/sigvalid.mat")[0]
    fast=CachedPredictor(ROOT,run).predict(record)
    reference=load_model_and_predict(ROOT,run,record)
    np.testing.assert_allclose(fast[256:],reference[256:],atol=1e-11,rtol=1e-11)


@pytest.mark.parametrize("kind", ["gru", "tcn", "speed_tcn"])
def test_cached_neural_prediction_resets_between_records(tmp_path,kind):
    run="test_model";folder=tmp_path/"experiments/runs"/run;folder.mkdir(parents=True)
    (tmp_path/"models").mkdir()
    cfg=dict(kind=kind,lag=8,channels=4,family="neural")
    (folder/"config.json").write_text(json.dumps(cfg))
    torch.manual_seed(42);model=build_network(kind,8,4)
    torch.save(model.state_dict(),tmp_path/"models"/(run+".pt"))
    predictor=CachedPredictor(tmp_path,run)
    rng=np.random.default_rng(44)
    a=dict(x=rng.normal(size=40),spdnorm=rng.normal(size=40),N=40)
    b=dict(x=rng.normal(size=40),spdnorm=rng.normal(size=40),N=40)
    before=predictor.predict(a);predictor.predict(b);after=predictor.predict(a)
    np.testing.assert_array_equal(before,after)
    np.testing.assert_allclose(before[8:],load_model_and_predict(tmp_path,run,a)[8:],atol=1e-7)


@pytest.mark.parametrize("unit", ["rpm", "rps", "rad_s"])
def test_shared_angular_transform_matches_direct_pmi_variants(unit):
    t=np.arange(6400)/3200;speed=np.full(len(t),150.)
    x=(1+.3*np.cos(2*np.pi*5*t))*np.cos(2*np.pi*300*t)
    result=angular_features(x,t,speed,3200,unit)
    angle=angle_resample(x,t,speed,unit,result["samples_per_cycle"])
    for variant in ("faithful_reference","scientifically_corrected"):
        reference=pmi(angle["signal"],result["samples_per_cycle"],variant=variant)
        assert result["pmi_"+variant]==pytest.approx(reference["pmi"],abs=1e-12)


def test_validation_selected_models_are_causal_and_one_per_family():
    selected=select_models(ROOT)
    assert len({row["kind"] for row in selected})==len(selected)==13
    assert all(row["kind"]!="shift_tcn" for row in selected)


def test_noncausal_checkpoint_rejected():
    with pytest.raises(ValueError,match="Noncausal"):
        CachedPredictor(ROOT,"shift_tcn_L64_s17")
