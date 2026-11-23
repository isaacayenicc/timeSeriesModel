"""Isolate matched residual correction and earlier-only fit sensitivity."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src_gearbox.study import scores
from src_gearbox.study_protocol import PROJECT, START, TARGETS, atomic_npz, read_vector_record, resolve_record, settings
from src_gearbox.src_model import Config, fit, load_model, slow_adapter


def run(root,data):
    base_root=root/"checkpoints"
    main_source=base_root/"src/primary_pi_src.npz"
    chosen=load_model(main_source)
    c=chosen.config
    summary=read_vector_record(resolve_record(data,"sigvalid"))
    x,rpm,_,fs=summary
    from src_gearbox.models.baselines.replay import load_predictor
    baseline=load_predictor("sparse_lpv_arma",base_root/"baselines")
    with threadpool_limits(limits=1):
        nominal=baseline.predict(x,rpm,fs)
        corrected=slow_adapter(x,nominal,c.adapter_gain,c.adapter_tau,c.adapter_memory,c.adapter_clip)
        primary=chosen.predict(x,rpm,fs)
        raw=chosen.predict_base(x,rpm,fs)
        ordinary=load_model(base_root/"src/src_polynomial.npz")
        ordinary_matched=load_model(base_root/"src/src_polynomial_matched_adapter.npz")
        matched = {
            "strict_sparse_lpv_arma": (nominal,corrected),
            "selected_pi_src":(raw,primary),
            "polynomial_src": (ordinary.predict(x,rpm,fs),ordinary_matched.predict(x,rpm,fs)),
        }
        train=read_vector_record(resolve_record(data,"sigtrain"))
        xt,wt,_,fst=train
        chronological=[]
        for lo,hi in settings("optimization")["folds"]:
            learner=fit(c,[(xt[:lo-START],wt[:lo-START])],fst)
            estimation=learner.predict(xt,wt,fst)[lo:hi]
            chronological.append(dict(train_start=0,train_stop=lo-START,hold_start=lo,hold_stop=hi,
                                      mse=float(np.mean((xt[lo:hi]-estimation)**2)),
                                      note="earlier-only refit; never used for selection"))
    out=root/"results/ablations"
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    arrays={"target_indices":TARGETS,"actual":x[TARGETS]}
    for name,(before,after) in matched.items():
        baseline_mse=scores(x[TARGETS],before[TARGETS])["mse"]
        adapted_mse=scores(x[TARGETS],after[TARGETS])["mse"]
        rows.append(dict(model=name,uncorrected_mse=baseline_mse,matched_correction_mse=adapted_mse,
                         gain_mse=baseline_mse-adapted_mse,gain=c.adapter_gain,tau=c.adapter_tau,
                         correction_memory=c.adapter_memory,clip=c.adapter_clip))
        arrays[name+"_base"] = before[TARGETS]
        arrays[name+"_matched"] = after[TARGETS]
    pd.DataFrame(rows).to_csv(out/"identical_adapter_controls.csv",index=False)
    pd.DataFrame(chronological).to_csv(out/"earlier_only_sensitivity.csv",index=False)
    atomic_npz(out/"identical_adapter_controls.npz",**arrays)
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(chronological).to_string(index=False))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-dir",type=Path,default=PROJECT/"data/raw")
    cli.add_argument("--project-dir",type=Path,default=PROJECT)
    args=cli.parse_args()
    run(args.project_dir,args.data_dir)
