"""Report training-design conditioning without changing the fitted models."""
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from rotmach.io_matlab import load_records
from rotmach.windows import make_windows


def diagnose(root):
    root=Path(root); r=load_records(root/'data/raw/sigtrain.mat')[0]
    result=[]
    for p in sorted((root/'experiments/runs').glob('*/config.json')):
        cfg=json.loads(p.read_text())
        if cfg['kind'] not in ('ar','arx','lpv','ols'):continue
        x,_,_=make_windows(r,cfg['lag'],cfg['kind'])
        # Intercept fitted separately by scikit-learn; analyze the centered design.
        x=x-x.mean(axis=0)
        gram=x.T@x
        ev=np.linalg.eigvalsh(gram)
        alpha=0. if cfg['kind']=='ols' else cfg.get('alpha',1.)
        result.append(dict(run_id=cfg['run_id'],n_rows=len(x),n_columns=x.shape[1],
                           centered_gram_min_eigenvalue=ev[0],centered_gram_max_eigenvalue=ev[-1],
                           centered_design_condition_sqrt_eigen_ratio=float(np.sqrt(ev[-1]/ev[0])) if ev[0]>0 else float('inf'),
                           regularized_gram_condition=float((ev[-1]+alpha)/(ev[0]+alpha)),alpha=alpha))
    pd.DataFrame(result).to_csv(root/'reports/linear_design_diagnostics.csv',index=False)
    return result


if __name__=='__main__':
    diagnose(Path(__file__).resolve().parents[1])
