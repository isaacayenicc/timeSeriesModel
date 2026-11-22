"""Recompute local and archived lecturer reference estimates on declared windows."""

from pathlib import Path
import json

import numpy as np
from threadpoolctl import threadpool_limits

from . import reference as ref
from .protocol import digest, write_json, write_table
from .statistical import _refit_sparse


def run(root: Path, data_dir: Path, *, fit_paper=False):
    root=Path(root)
    data_dir=Path(data_dir)
    references=root/"references/baselines"
    metrics=root/"results/metrics"
    train=ref.mat_record(ref.find_record(data_dir,"sigtrain"),"sigtrain")
    valid=ref.mat_record(ref.find_record(data_dir,"sigvalid"),"sigvalid")
    rows, predictions = [], {}
    modes=(("strict_development", "conventional.npz", False, False, 1792),
           ("preserved_archive_current_rpm", "archived_lpv_coefficients.npz", True, True, 316),
           ("preserved_archive_previous_rpm", "archived_lpv_coefficients.npz", False, True, 1792))
    with threadpool_limits(1):
        for case,n,current,archive,first in modes:
            arrays=ref.load_arrays(references/n)
            models=ref.conventional_forecasts(valid,arrays,current=current,archived=archive)
            train_models=ref.conventional_forecasts(train,arrays,current=current,archived=archive)
            _record(rows,predictions,case,models,train_models,train,valid,first,current)
        for basis in ("legendre","monomial"):
            if fit_paper:
                fit_run=root/"results/experiments"/f"paper_current_rpm_{basis}_refit"
                (fit_run/"checkpoints").mkdir(parents=True,exist_ok=True)
                ref.paper_row_diagnostic(train,valid,fit_run,basis=basis)
                path=fit_run/"checkpoints"/f"paper_row_cv_{basis}.npz"
            else:
                path=references/f"paper_row_cv_{basis}.npz"
            arrays=ref.load_arrays(path)
            kw=dict(current=True,archived=basis=="monomial",selected=("spa_lpv_ar","spa_lpv_arma"))
            models=ref.conventional_forecasts(valid,arrays,**kw)
            train_models=ref.conventional_forecasts(train,arrays,**kw)
            _record(rows,predictions,f"current_rpm_row_cv_{basis}",models,train_models,train,valid,316,True)
        strict_path=references/"conventional.npz"
        _,fit_s,deltas=_refit_sparse(train.x,train.rpm,strict_path)
    write_table(metrics/"local_statistical_reference.csv",rows)
    write_table(metrics/"published_paper_reference.csv",[
        dict(model=k,train_mse=v[0],authors_test_segment_mse=v[1],authors_runs_p=v[2],
             doi="10.1016/j.ymssp.2024.112161",status="external_different_segment")
        for k,v in ref.PUBLISHED.items()])
    write_json(metrics/"strict_sparse_support_refit.json",dict(
        source_checkpoint_sha256=digest(strict_path),coefficient_max_abs_delta=deltas,
        stage1_fit_s=fit_s[0],two_stage_fit_s=fit_s[1],status="new_qr_refit_of_preserved_guarded_support",
        note="Source Table 6 describes the authors’ test segment. The current-RPM diagnostic is distinct from strict lagged-RPM fair evaluation. Chen et al. §3.2 uses one standard deviation." ))
    np.savez_compressed(root/"results/predictions/local_statistical_reference.npz",
                        actual=valid.x, target_indices=np.arange(len(valid.x)),**predictions)
    return rows


def _record(rows,predictions,case,models,train_models,train,valid,first,current):
    for key, model in models.items():
        residual=valid.x[first:]-model[first:]
        ts=train.x[first:]-train_models[key][first:]
        rows.append(dict(model=key,convention=case,scheduling="current" if current else "previous",
                         start=first,n=len(residual),train_mse=float(np.mean(ts*ts)),
                         development_mse=float(np.mean(residual*residual)),
                         common_mask_mse=float(np.mean((valid.x[1792:]-model[1792:])**2)),
                         runs_p=ref.runs_test(residual)))
        predictions[f"{case}__{key}"]=model
