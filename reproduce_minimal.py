"""Audit inputs and independently rescore the packaged ridge AR checkpoint."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
import numpy as np
from rotmach.audit import audit
from rotmach.cli import stage_data
from rotmach.io_matlab import load_records
from rotmach.experiment import load_model_and_predict, metrics

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parent
    stage_data(root,a.data_dir); audit(root)
    r=load_records(root/'data/raw/sigvalid.mat')[0]
    pred=load_model_and_predict(root,'ar_L256_s17_a1',r)
    score=metrics(r['x'][256:],pred[256:])
    expected=json.loads((root/'experiments/runs/ar_L256_s17_a1/metrics.json').read_text())
    assert np.isclose(score['mse'],expected['mse'],rtol=1e-6)
    print(json.dumps(dict(status='PASS',run_id=expected['run_id'],**score),indent=2))
