"""Scientific invariants and real-input model artifact roundtrips."""
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from scipy.io import savemat
from sklearn.linear_model import Ridge
from rotmach.io_matlab import load_records
from rotmach.windows import make_windows
from rotmach.models import build_network, CausalBlock
from rotmach.experiment import metrics, load_model_and_predict

ROOT=Path(__file__).resolve().parents[1]


def record(n=400):
    return dict(N=n,fs=100.,t=np.arange(1,n+1)/100,x=np.arange(n,dtype=float),
                spd=np.ones(n)*100,spdnorm=np.ones(n)*.3)


def test_manual_window_alignment():
    x,y,idx=make_windows(record(),3)
    np.testing.assert_array_equal(x[0],[0,1,2])
    assert y[0]==3 and idx[0]==3 and x[-1,-1]==398


@pytest.mark.parametrize('kind',['ar','arx','lpv','mlp','gru','tcn','speed_tcn'])
def test_future_perturbation(kind):
    a=record(); b=record(); b['x'][200:]+=1000; b['spdnorm'][200:]-=.1
    xa,_,idx=make_windows(a,32,kind); xb,_,_=make_windows(b,32,kind)
    np.testing.assert_array_equal(xa[idx<=200],xb[idx<=200])
    if kind in ['mlp','gru','tcn','speed_tcn']:
        torch.manual_seed(1); model=build_network(kind,32).eval()
        with torch.no_grad():
            # CPU GRU kernels can differ by ~1e-6 between identical float32
            # calls. Inputs above must match exactly; outputs use the same
            # numerical scale as checkpoint parity, rather than zero atol.
            np.testing.assert_allclose(model(torch.tensor(xa[idx<=200])).numpy(),
                                       model(torch.tensor(xb[idx<=200])).numpy(),
                                       rtol=1e-5,atol=2e-6)


def test_causal_block_future_invariance():
    b=CausalBlock(4,2).eval(); x=torch.randn(2,4,30); z=x.clone(); z[:,:,20:]+=100
    with torch.no_grad():
        torch.testing.assert_close(b(x)[:,:,:20],b(z)[:,:,:20])


def test_struct_record_boundaries(tmp_path):
    p=tmp_path/'multi.mat'; records=np.empty(2,dtype=object); records[0]=record(10); records[1]=record(12)
    savemat(p,{'records':records}); result=load_records(p)
    assert len(result)==2 and [r['N'] for r in result]==[10,12]


def test_bad_time_fails(tmp_path):
    r=record(); r['t'][50]=r['t'][49]; p=tmp_path/'bad.mat'; savemat(p,{'r':r})
    with pytest.raises(ValueError,match='timestamp'):
        load_records(p)


def test_normalization():
    for filename in ['sigtrain.mat','sigvalid.mat']:
        r=load_records(ROOT/'data/raw'/filename)[0]
        np.testing.assert_allclose((r['xraw']-r['xmean'][0])/r['xstd'][0],r['x'],atol=1e-13)


def test_known_ar_process():
    rng=np.random.default_rng(11); x=np.zeros(30000)
    for i in range(2,len(x)):
        x[i]=.7*x[i-1]-.2*x[i-2]+rng.normal(scale=.1)
    r={'x':x,'spdnorm':np.zeros(len(x))}; a,y,_=make_windows(r,2)
    model=Ridge(alpha=.00001).fit(a,y)
    np.testing.assert_allclose(model.coef_,[-.2,.7],atol=.02)


def test_metric_definitions():
    assert metrics([1,2],[1,2])['mse']==0
    assert metrics([1,2],[0,0])['mse']==2.5


@pytest.mark.parametrize('kind',['mlp','gru','tcn','speed_tcn','late_tcn','capacity_tcn'])
def test_network_shapes_and_receptive_field(kind):
    model=build_network(kind,64)
    assert model(torch.randn(3,2,64)).shape==(3,)
    if hasattr(model,'receptive_field'):
        assert model.receptive_field>=64


def test_actual_best_artifact():
    info=ROOT/'experiments/best_model.json'
    if not info.exists():
        pytest.skip('No trained artifact present; run training first')
    run=json.loads(info.read_text())['run_id']
    record=load_records(ROOT/'data/raw/sigvalid.mat')[0]
    p=load_model_and_predict(ROOT,run,record)
    cache=ROOT/'experiments/runs'/run/'predictions.npz'
    if cache.exists():
        expected=np.load(cache)
        np.testing.assert_allclose(p[expected['target_indices']],expected['y_pred'],atol=1e-5)
    else:
        info=json.loads(info.read_text())
        assert metrics(record['x'][256:],p[256:])['mse']==pytest.approx(info['mse'],rel=1e-6)


def test_missing_prediction_cache_rebuilds_from_frozen_checkpoint(tmp_path):
    import shutil
    from rotmach.evaluate import ensure_prediction_cache
    run='ar_L256_s17_a1'
    source=ROOT/'data/raw/sigvalid.mat'
    record=load_records(source)[0]
    prediction=load_model_and_predict(ROOT,run,record)
    folder=tmp_path/'experiments/runs'/run;folder.mkdir(parents=True)
    for name in ('config.json','metrics.json'):
        shutil.copy2(ROOT/'experiments/runs'/run/name,folder/name)
    data=tmp_path/'data/raw';data.mkdir(parents=True);shutil.copy2(source,data/'sigvalid.mat')
    cache=ensure_prediction_cache(tmp_path,run,record,prediction)
    saved=np.load(cache)
    np.testing.assert_array_equal(saved['target_indices'],np.arange(256,24000))
    np.testing.assert_allclose(saved['y_pred'],prediction[256:])
    cache.unlink();data.joinpath('sigvalid.mat').write_bytes(b'wrong source file')
    with pytest.raises(ValueError,match='Validation data differ'):
        ensure_prediction_cache(tmp_path,run,record,prediction)
