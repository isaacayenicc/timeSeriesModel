"""Tests for matched masks, temporal folds and evidence-preserving serialization."""
import json
from pathlib import Path

import numpy as np
import pytest

from src_gearbox.comparison import blocked
from src_gearbox.optimization import candidate_grid, pulse_pattern
from src_gearbox.study import scores
from src_gearbox.study_protocol import MASK_HASH, START, STOP, TARGETS, atomic_json, fold_segments, settings


def test_training_folds_exclude_the_complete_guard():
    train = np.arange(STOP)
    cv = settings("optimization")
    for lo,hi in cv["folds"]:
        first,second = fold_segments(train,train,lo,hi,cv["guard_samples"])
        assert first[0][-1] == lo-START-1
        assert second[0][0] == hi+START
        pulse = pulse_pattern(train.astype(float),lo,hi,.6)
        assert np.all(pulse[:lo]==0) and np.all(pulse[hi:]==0)
    assert len(TARGETS)==22208
    assert MASK_HASH=="1522c0a1b184b971b59763df3d827515370162751cc61209f025765039a89fb1"


def test_trial_space_is_bounded_and_uses_no_development_record():
    standard=candidate_grid("standard")
    keys=[(stage,name) for stage,name,cfg in standard]
    assert len(set(keys))==len(standard)
    assert {s for s,n in keys}=={"A","B","C","D"}
    assert len(candidate_grid("quick"))<len(standard)<len(candidate_grid("full"))
    assert all("sigvalid" not in json.dumps(cfg).lower() for _,_,cfg in standard)


def test_json_numpy_type_recovery(tmp_path: Path):
    p=tmp_path/"trial.json"
    atomic_json(p,{"counts":np.int64(7),"eligible":np.bool_(True)})
    assert json.loads(p.read_text())=={"counts":7,"eligible":True}
    with pytest.raises(ValueError):
        atomic_json(p,{"value":float("nan")})
    assert json.loads(p.read_text())=={"counts":7,"eligible":True}


def test_mse_and_bootstrap_pairing():
    truth=np.arange(100,dtype=float)
    result=scores(truth,truth+.5)
    assert result["mse"]==.25 and result["mae"]==.5
    np.testing.assert_allclose(blocked(np.full(243,.023),40,100,19),[.023,.023],rtol=1e-12)


def test_frozen_protocol_is_inference_only_and_locks_scored_files(tmp_path):
    from shutil import copy2
    from scipy.io import savemat
    from src_gearbox.study import frozen
    from src_gearbox.study_protocol import PROJECT, fingerprint

    model_rel=Path("checkpoints/src/primary_pi_src.npz")
    checkpoint=PROJECT/model_rel
    if not checkpoint.is_file():
        pytest.skip("The final SRC checkpoint is required for the frozen integration test")
    copy=(tmp_path/"study")/model_rel
    copy.parent.mkdir(parents=True)
    copy2(checkpoint,copy)
    atomic_json(tmp_path/"study/checkpoints/src_registry.json",
        dict(target_mask_sha256=MASK_HASH,models=[dict(model="primary_pi_src",
             checkpoint=model_rel.as_posix(),sha256=fingerprint(copy))]))
    data=tmp_path/"inputs"
    data.mkdir()
    n=9000
    t=np.arange(n)/3200
    rpm=146+6*np.sin(2*np.pi*.7*t)
    turns=np.r_[0,np.cumsum(rpm[:-1])/(60*3200)]
    vibration=np.sin(2*np.pi*37*turns)+.1*np.sin(2*np.pi*415*t)
    record=dict(x=vibration,spd=rpm,t=t,fs=3200,N=n)
    savemat(data/"sigtest.mat",dict(sigtest=record))
    savemat(data/"sigfault2.mat",dict(sigfault2=record))
    frozen(data,tmp_path/"study")
    results=tmp_path/"study/results/frozen"
    assert json.loads((results/"availability.json").read_text())["status"]=="SCORED"
    assert json.loads((results/"fault2_availability.json").read_text())["records"]==1
    assert "residual_pmi" in (results/"sigfault2_pmi.csv").read_text()
    original=fingerprint(copy)
    frozen(data,tmp_path/"study")
    record["x"]=vibration+.02
    savemat(data/"sigtest.mat",dict(sigtest=record))
    with pytest.raises(RuntimeError,match="different independent"):
        frozen(data,tmp_path/"study")
    assert fingerprint(copy)==original
