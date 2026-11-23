"""Compare full design materialization with bounded-batch feature evaluation."""
import argparse
from pathlib import Path
import statistics
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src_gearbox.src_model import _library, load_model, receptive_fields, solve_readout
from src_gearbox.study_protocol import PROJECT, TARGETS, read_vector_record, resolve_record


def validate(data):
    record=read_vector_record(resolve_record(data,"sigvalid"))
    x,rpm,_,fs=record
    model=load_model(PROJECT/"checkpoints/src/primary_pi_src.npz")
    start,_=receptive_fields(model.config,fs)

    def explicit():
        idx = np.arange(start,len(x))
        lib,_,_ = _library(x,rpm,fs,idx,model.config,model.speed,model.proxy)
        dense = model._select(lib)
        forecast=model.coefficients[0]+dense@model.coefficients[1:]
        return forecast[TARGETS-start],lib.nbytes+dense.nbytes

    def bounded():
        predicted=model.predict_base(x,rpm,fs)[TARGETS]
        idx=np.arange(start,start+4096)
        lib,_,_ = _library(x[:idx[-1]+1],rpm[:idx[-1]+1],fs,idx,model.config,model.speed,model.proxy)
        return predicted,lib.nbytes+model._select(lib).nbytes

    with threadpool_limits(limits=1):
        ref,_=explicit()
        records=[]
        for name,action in [("full_explicit",explicit),("bounded_batches_4096",bounded)]:
            # Memory is the two principal feature arrays during evaluation, not
            # process peak. Measure only each predictor on repeated timed calls.
            check,size=action()
            dif=float(np.max(np.abs(ref-check)))
            assert dif<2e-6, (name,dif)
            times=[]
            for _ in range(5):
                t=perf_counter()
                if name=="bounded_batches_4096":
                    model.predict_base(x,rpm,fs)
                else:
                    explicit()
                times.append((perf_counter()-t)*1000)
            records.append(dict(method=name,median_ms=statistics.median(times),repeats=len(times),
                                scored_targets=len(TARGETS),principal_feature_bytes=size,
                                max_abs_delta_vs_explicit=dif,threads=1))
        # Two verified chunk sizes for sufficient-statistics assembly, with
        # identical readable objective on a compact true-data slice.
        indices=np.arange(max(start,3000),6500)
        features=_library(x,rpm,fs,indices,model.config,model.speed,model.proxy)[0]
        design=np.column_stack([np.ones(len(indices)),model._select(features)[:,:128]])
        penalty=np.r_[0.0,np.full(design.shape[1]-1,model.config.ridge)]
        outcomes=[]
        ref=None
        for block in (len(design),256):
            t=perf_counter()
            coef,stats=solve_readout(design,x[indices],penalty,block_size=block)
            elapsed=(perf_counter()-t)*1000
            if ref is None:
                ref=coef.copy()
            err=float(np.max(np.abs(coef-ref)))
            assert err<1e-9
            outcomes.append(dict(accumulation_rows=block,elapsed_ms=elapsed,coefficient_max_delta=err,
                                 rows=len(indices),columns=design.shape[1],**stats))
    target=PROJECT/"results/timings"
    pd.DataFrame(records).to_csv(target/"feature_materialization.csv",index=False)
    pd.DataFrame(outcomes).to_csv(target/"gram_block_benchmark.csv",index=False)
    print(pd.DataFrame(records).to_string(index=False))
    print(pd.DataFrame(outcomes).to_string(index=False))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",type=Path,default=PROJECT/"data/raw")
    validate(p.parse_args().data_dir)
