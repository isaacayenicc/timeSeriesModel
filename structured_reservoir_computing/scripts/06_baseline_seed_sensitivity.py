"""Measure fixed-configuration seed sensitivity for representative neural baselines."""

import argparse
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from src_gearbox.models.baselines import attention, neural
from src_gearbox.models.baselines.protocol import SEED, START, load_data, mse, write_table
from src_gearbox.models.baselines.replay import CHECKPOINT_DIR, load_predictor


def run(root, data_dir, models):
    torch.set_num_threads(1)
    train,valid=load_data(data_dir)
    checkpoint=root/CHECKPOINT_DIR/"seed_sensitivity"
    checkpoint.mkdir(parents=True,exist_ok=True)
    all_rows, summary = [], []
    with threadpool_limits(1):
        for slug in models:
            original=load_predictor(slug,root/CHECKPOINT_DIR)
            loss_rows = []
            base=original.predict(valid.x,valid.rpm,valid.fs)
            loss_rows.append(dict(model=slug,seed=SEED,fit_s=None,evidence="prior_selected_checkpoint",
                                  mse=mse(valid.x[START:],base[START:])))
            for seed in (SEED+1,SEED+2):
                tick=perf_counter()
                if slug in ("transformer","tft"):
                    fitted=attention.fit(asdict(original.config),[(train.x,train.rpm)],train.fs,seed=seed)
                else:
                    fitted,*_=neural._fit([(train.x,train.rpm)],train.fs,original.config,seed=seed)
                fit_s=perf_counter()-tick
                stored=checkpoint/f"{slug}_{seed}.npz"
                fitted.save(stored)
                model=(attention if slug in ("transformer","tft") else neural).load(stored)
                estimate=model.predict(valid.x,valid.rpm,valid.fs)
                loss_rows.append(dict(model=slug,seed=seed,fit_s=fit_s,evidence="new_fixed_config_sigtrain_refit",
                                      mse=mse(valid.x[START:],estimate[START:])))
            all_rows.extend(loss_rows)
            scores=[r["mse"] for r in loss_rows]
            summary.append(dict(model=slug,seeds=len(scores),mean_mse=np.mean(scores),
                                sd_mse=np.std(scores,ddof=1),min_mse=min(scores),max_mse=max(scores)))
            write_table(root/"results/metrics/baseline_seed_sensitivity.csv",all_rows)
            write_table(root/"results/metrics/baseline_seed_sensitivity_summary.csv",summary)
            print(summary[-1],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--models",nargs="+",default=("mlp","lstm","cnn","transformer"),
                        choices=("mlp","lstm","gru","cnn","tcn","transformer","tft"))
    args=parser.parse_args()
    run(args.root,args.data_dir,args.models)
