"""Checkpoint rescore, within-trace uncertainty and descriptive residual statistics."""
import json
import platform
try:
    import resource
except ImportError:  # Windows has no resource module.
    resource = None
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from .io_matlab import load_records, sha256
from .experiment import collect, load_model_and_predict, metrics
from .diagnostics import signal_features


def ensure_prediction_cache(root, run_id, record, prediction):
    """Restore a Git-excluded signal cache from supplied data and a saved model.

    The validation input digest and recorded MSE must match the original run.
    This is checkpoint inference, never a training or model-selection operation.
    """
    folder=Path(root)/'experiments/runs'/run_id
    cache=folder/'predictions.npz'
    if cache.exists():
        return cache
    cfg=json.loads((folder/'config.json').read_text())
    source=Path(root)/'data/raw/sigvalid.mat'
    if sha256(source)!=cfg['valid_sha256']:
        raise ValueError('Validation data differ from the recorded model input')
    idx=np.arange(cfg['burnin'],record['N'])
    y=record['x'][idx]; pred=np.asarray(prediction)[idx]
    expected=json.loads((folder/'metrics.json').read_text())['mse']
    if not np.isclose(metrics(y,pred)['mse'],expected,rtol=1e-6):
        raise AssertionError('Checkpoint score differs; refusing to rebuild prediction cache')
    np.savez_compressed(cache,target_indices=idx,time=record['t'][idx],
                        y_true=y,y_pred=pred,residual=y-pred,
                        record_id=record['record_id'],run_id=run_id)
    return cache


def evaluate_saved(root):
    """Rescore every saved run; uncertainty is conditional on one validation trace.

    Nonoverlapping fixed blocks retain local dependence; bootstrap block means,
    not overlapping windows. These intervals are not post-selection corrected.
    """
    root=Path(root); torch.set_num_threads(2)
    r=load_records(root/'data/raw/sigvalid.mat')[0]
    df=collect(root); rng=np.random.default_rng(2026)
    uncertainty=[]; features=[]; bins=[]; rescore=[]
    edges=np.linspace(r['spdnorm'].min(),r['spdnorm'].max()+1e-10,6)
    for row in df.to_dict('records'):
        run=row['run_id']; p=load_model_and_predict(root,run,r)
        saved=np.load(ensure_prediction_cache(root,run,r,p))
        idx=saved['target_indices']; y=r['x'][idx]; pred=p[idx]; e=y-pred
        np.testing.assert_allclose(pred,saved['y_pred'],atol=1e-5)
        score=metrics(y,pred)['mse']
        if not np.isclose(score,row['mse'],rtol=1e-6):
            raise AssertionError('Rescored MSE differs from registry')
        rescore.append(dict(run_id=run,mse_recomputed=score,max_prediction_error=float(np.max(abs(pred-saved['y_pred'])))))
        features.append(dict(run_id=run,source='validation residual',n=len(e),**signal_features(e)))
        for block in (256,512,1024):
            means=np.array([np.mean(e[i:i+block]**2) for i in range(0,len(e)-block+1,block)])
            boot=rng.choice(means,size=(1000,len(means)),replace=True).mean(axis=1)
            uncertainty.append(dict(run_id=run,block_samples=block,n_complete_blocks=len(means),
                                    excluded_tail_samples=len(e)-len(means)*block,
                                    block_mean_mse=float(means.mean()),lower=float(np.quantile(boot,.025)),upper=float(np.quantile(boot,.975))))
        for b in range(5):
            keep=(r['spdnorm'][idx]>=edges[b])&(r['spdnorm'][idx]<edges[b+1])
            bins.append(dict(run_id=run,bin=b,left=edges[b],right=edges[b+1],n=int(keep.sum()),mse=float(np.mean(e[keep]**2))))
    features.append(dict(run_id='raw_validation',source='validation supplied x',n=len(r['x'])-256,**signal_features(r['x'][256:])))
    pd.DataFrame(features).to_csv(root/'reports/validation_residual_features.csv',index=False)
    pd.DataFrame(uncertainty).to_csv(root/'reports/within_trace_uncertainty.csv',index=False)
    pd.DataFrame(bins).to_csv(root/'reports/speed_bin_errors.csv',index=False)
    pd.DataFrame(rescore).to_csv(root/'reports/checkpoint_rescore.csv',index=False)
    seeds=df[df.kind.isin(['tcn','speed_tcn']) & (df.lag==64)].groupby('kind').mse.agg(['mean','std','min','max','count'])
    seeds.to_csv(root/'reports/seed_variability.csv')
    if not (root/'data/raw/sigfault1.mat').exists() and not (root/'data/raw/sigfault1_parts/manifest.json').exists():
        if not (root/'reports/fault_record_features.csv').exists():
            (root/'reports/fault_record_features.csv').write_text('record_id,model_id,status\n')
        (root/'reports/fault_analysis_status.json').write_text(json.dumps(dict(status='blocked',
            reason='sigfault1.mat upload exceeds 32 MiB executor transfer limit; no records inspected',
            real_fault_records_processed=0,labels_verified=False,speed_unit_verified=False),indent=2))
    try:
        cpu=next(x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name'))
    except (OSError,StopIteration): cpu=platform.processor()
    hardware=dict(cpu=cpu,platform=platform.platform(),python=platform.python_version(),numpy=np.__version__,
                  scipy=scipy.__version__,sklearn=sklearn.__version__,torch=torch.__version__,
                  cuda_available=torch.cuda.is_available(),torch_threads=torch.get_num_threads(),
                  precision='linear float64; neural float32',training_wall_s_sum=float(df.train_wall_s.sum()),
                  superseded_control_training_wall_s=33.496943697000006,
                  all_attempts_training_wall_s_sum=float(df.train_wall_s.sum())+33.496943697000006,
                  process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource else None)
    (root/'reports/hardware.json').write_text(json.dumps(hardware,indent=2))
    (root/'reports/timing_and_hardware.md').write_text('# Timing and hardware\n\n```json\n'+json.dumps(hardware,indent=2)+'\n```\n\nCPU only; no GPU timing was measured. 2 PyTorch and 2 BLAS threads during original training. Preprocessing is separate. Training wall time includes fit/epochs, within-epoch validation, saving and a final forward pass; it excludes later reload/timing repetitions. Inference is the public load_model_and_predict full-record path: checkpoint loading + window preparation + scoring all N−L targets of the 24,000-sample validation record, one warmup then 3 repetitions, median/min/max. This is not device-only latency or sustained single-window streaming latency. Training counts differ with context. RSS is process-level Linux high-water memory for rescoring, not per-model incremental peak RAM. Timing varies with concurrent machine load and storage cache.\n')
    return df
