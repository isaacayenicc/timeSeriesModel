"""Fit one explicit SRC configuration or replay one numeric checkpoint."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits
import yaml

from src_gearbox.src_model import Config, fit, load_model
from src_gearbox.study_protocol import PROJECT, START, atomic_npz, read_vector_record, resolve_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=PROJECT/"configs"/"experiments"/"src_single_starter.yaml")
    parser.add_argument("--data-dir","--input-dir",type=Path,default=PROJECT/"data"/"raw")
    parser.add_argument("--mode",choices=["quick","full"],default="full")
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--replay",type=Path)
    parser.add_argument("--record",type=Path)
    parser.add_argument("--predictions",type=Path)
    parser.add_argument("--threads",type=int,default=1)
    args=parser.parse_args()
    if args.threads<1:
        parser.error("Thread count must be positive.")
    if bool(args.replay)==bool(args.checkpoint):
        parser.error("Use either --checkpoint for fitting or --replay for an existing model.")
    if args.replay and not args.record:
        parser.error("Replay requires a record path.")
    with threadpool_limits(args.threads):
        if args.replay:
            model=load_model(args.replay)
        else:
            options=yaml.safe_load(args.config.read_text(encoding="utf-8"))
            config=Config(**options.get("config",options))
            x,rpm,_,fs=read_vector_record(resolve_record(args.data_dir,"sigtrain"))
            if args.mode=="quick":
                x,rpm=x[:8000],rpm[:8000]
            started=perf_counter()
            model=fit(config,[(x,rpm)],fs)
            model.save(args.checkpoint)
            print(json.dumps({"checkpoint":str(args.checkpoint),"training_mode":args.mode,
                              "training_samples":len(x),"fit_seconds":perf_counter()-started,
                              "parameters":model.parameters,"history":model.history,
                              "solver":model.solver_info},indent=2))
        if args.record:
            x,rpm,t,fs=read_vector_record(args.record)
            forecast=model.predict(x,rpm,fs)
            idx=np.arange(max(START,model.history),len(x))
            if not len(idx):
                raise ValueError("Record has no eligible targets under its warm-up.")
            print(json.dumps({"record":str(args.record),"n_scored":len(idx),
                              "target_start":int(idx[0]),
                              "mse":float(np.mean((forecast[idx]-x[idx])**2))},indent=2))
            if args.predictions:
                atomic_npz(args.predictions,prediction=forecast,target=x,rpm=rpm,time=t,indices=idx)


if __name__=="__main__":
    main()
