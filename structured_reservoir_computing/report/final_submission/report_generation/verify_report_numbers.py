"""Rebuild the PI-SRC report evidence directly from released artifacts; no training."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.io import loadmat
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits


GEN = Path(__file__).resolve().parent
ROOT = GEN.parents[2]
PREDICTED = "development: sigvalid, teacher-forced one step"
PRIMARY = "primary_pi_src"
BASELINES = (
    "ar", "arma", "arima", "arx", "sparse_lpv_ar", "sparse_lpv_arma", "linear", "ridge",
    "lasso", "elastic_net", "svr", "random_forest", "gradient_boosting", "narx",
    "mlp", "lstm", "gru", "cnn", "tcn", "transformer", "tft",
)
AXES = ("target_indices", "actual", "time_s", "rpm")
METRIC_UNITS = {"mse":"standardized vibration squared","rmse":"standardized vibration",
                "mae":"standardized vibration","bias":"standardized vibration",
                "train_mse":"standardized vibration squared","nrmse":"dimensionless",
                "fold_mse":"standardized vibration squared","fold_mse_sd":"standardized vibration squared",
                "fold_stress_mse":"standardized vibration squared","stress_mse":"standardized vibration squared",
                "fold_pulse_distortion":"dimensionless","pulse_distortion":"dimensionless",
                "utility":"weighted objective; comparable only within search","fit_s":"seconds",
                "warm_median_ms":"milliseconds","warm_iqr_ms":"milliseconds",
                "warm_p95_ms":"milliseconds","parameters":"stored coefficient/parameter count",
                "checkpoint_bytes":"bytes","feature_columns":"columns","history":"samples",
                "relative_mse":"ratio","fold_mse_prior":"standardized vibration squared",
                "relative_improvement_pct":"percent of comparator MSE","rpm_min":"rpm","rpm_max":"rpm",
                "duration_s":"seconds","fs_hz":"hertz","N":"samples","n":"samples",
                "median_ms":"milliseconds","elapsed_ms":"milliseconds","block_samples":"samples",
                "samples_per_second":"samples per second","principal_feature_bytes":"bytes",
                "sample_rate_hz":"hertz","checkpoint_load_s":"seconds","prediction_s":"seconds"}


def check_close(a, b, *, atol=1e-12):
    np.testing.assert_allclose(a,b,rtol=0,atol=atol)


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing to create empty evidence {path}")
    keys=list(rows[0])
    with path.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=keys,lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def json_read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mstats(actual, pred):
    assert actual.shape == pred.shape and np.isfinite(pred).all()
    err=actual-pred
    mse=float(np.mean(err*err))
    return {"mse":mse,"rmse":mse**.5,"mae":float(np.mean(np.abs(err))),
            "bias":float(np.mean(err))}


def record_metrics(ledger, row, metrics, *, model, split, source, n="", targets="", mask="",
                   qualifier="", run="", source_selector="", units_override=None):
    for metric in metrics:
        value=row.get(metric)
        if value is None or (not isinstance(value,str) and pd.isna(value)):
            continue
        if isinstance(value,(int,float,np.integer,np.floating)) and not np.isfinite(value):
            raise ValueError(f"nonfinite {metric} for {model}")
        units=units_override or METRIC_UNITS.get(metric,("standardized vibration squared" if
                               ("mse" in metric and not metric.startswith("relative") and "_relative_" not in metric)
                               else "dimensionless"))
        ledger.append({"model":model,"run_or_variant":run,"split":split,
                       "metric":metric,"value":value,"units":units,
                       "sample_count":n,"target_indices_zero_based":targets,"target_mask_sha256":mask,
                       "source":source,"source_selector":source_selector,"qualification":qualifier})


def moving_interval(delta, width, draws, seed):
    total=len(delta)
    count, remainder=divmod(total,width)
    rng=np.random.default_rng(seed)
    cum=np.r_[0.,np.cumsum(delta)]
    start=rng.integers(0,total-width+1,(draws,count))
    sums=(cum[start+width]-cum[start]).sum(axis=1)
    if remainder:
        start=rng.integers(0,total-remainder+1,draws)
        sums+=cum[start+remainder]-cum[start]
    return np.quantile(sums/total,[.025,.975])


def predictions(root, ledger):
    res=root/"results"
    mask=json_read(res/"metrics/evaluation_mask.json")
    ids=np.arange(mask["start"],mask["stop_exclusive"],dtype="<i8")
    mask_hash=hashlib.sha256(ids.tobytes()).hexdigest()
    assert mask_hash==mask["target_indices_sha256"] and len(ids)==mask["count"]==22208
    axes={}
    all_forecasts={}
    for bundle in ("src_sigvalid","baselines_sigvalid","archived_src_sigvalid"):
        with np.load(res/"predictions"/f"{bundle}.npz",allow_pickle=False) as f:
            for key in AXES:
                if key in axes:
                    np.testing.assert_array_equal(axes[key],f[key],err_msg=f"{bundle}/{key}")
                else:
                    axes[key]=f[key]
            for key in set(f.files)-set(AXES):
                assert key not in all_forecasts
                all_forecasts[key]=f[key]
    np.testing.assert_array_equal(ids,axes["target_indices"])
    source="results/metrics/all_model_comparison.csv"
    head=pd.read_csv(root/source)
    assert len(head)==27 and len(set(head.model)&set(BASELINES))==21
    assert len(set(head.model))==27
    assert set(head.n)=={len(ids)} and set(head.mask_sha256)=={mask_hash}
    selected=json_read(res/"metrics/src_summary.json")
    assert set(head.source_validation_sha256)=={selected["development_sha256"]}
    idx=f"[{ids[0]}, {ids[-1]+1})"
    largest_comparison_error=0.
    estimates={}
    for row in head.to_dict("records"):
        model=row["model"]
        estimate=mstats(axes["actual"],all_forecasts[model])
        for key,value in estimate.items():
            check_close(value,row[key])
            largest_comparison_error=max(largest_comparison_error,abs(value-row[key]))
        estimates[model]=estimate
        record_metrics(ledger,row,(*estimate,"nrmse","residual_acf_lag1","two_sided_runs_p","ljung_box_p64"),
                       model=model,split=PREDICTED,source=source,n=len(ids),targets=idx,mask=mask_hash,
                       qualifier=row["scope"],source_selector=f"model={model}")
        fit_split="training: sigtrain, stored in-sample estimate"
        record_metrics(ledger,row,("train_mse",),model=model,split=fit_split,source=source,
                       qualifier="common training scoring slice where recorded",n=len(ids),
                       targets=idx, mask=mask_hash,source_selector=f"model={model}")
        record_metrics(ledger,row,("warm_median_ms","warm_iqr_ms","fit_s","checkpoint_bytes","parameters"),
                       model=model,split="runtime or model inventory",source=source,
                       qualifier="Warm CPU inference consumes a full 24,000-sample in-memory batch; historical fit times retain original scope",
                       source_selector=f"model={model}")
    for metric,key in (("mse","validation_mse"),("rmse","validation_rmse"),("mae","validation_mae")):
        check_close(selected[key],estimates[PRIMARY][metric])
    scores=head.set_index("model").mse
    best_required=scores.reindex(BASELINES).idxmin()
    best_neural=scores.reindex(("mlp","lstm","gru","cnn","tcn","transformer","tft")).idxmin()
    assert (best_required,best_neural)==("linear","cnn")
    summary=json_read(res/"metrics/comparison_summary.json")
    check_close(summary["relative_improvement_over_required_baseline_pct"],
                100*(scores.linear-scores[PRIMARY])/scores.linear)
    check_close(summary["relative_improvement_over_archive_pct"],
                100*(scores.archived_src_corrected-scores[PRIMARY])/scores.archived_src_corrected)
    paired=pd.read_csv(res/"statistical_tests/paired_moving_block.csv")
    cfg=yaml.safe_load((root/"configs/evaluation.yaml").read_text())
    primary_sq=(axes["actual"]-all_forecasts[PRIMARY])**2
    assert len(paired)==78
    for row in paired.to_dict("records"):
        model=row["model"]
        assert row["reference"]==PRIMARY
        delta=(axes["actual"]-all_forecasts[model])**2-primary_sq
        check_close(np.mean(delta),row["delta_mse"])
        low,high=moving_interval(delta,int(row["block_samples"]),int(row["draws"]),cfg["random_seed"])
        check_close([low,high],[row["interval_low"],row["interval_high"]])
        record_metrics(ledger,row,("delta_mse","interval_low","interval_high","block_samples","draws"),
                       model=model,split=PREDICTED,source="results/statistical_tests/paired_moving_block.csv",
                       n=len(ids),targets=idx,mask=mask_hash,
                       qualifier="comparator minus selected PI-SRC; common contiguous-block resample within one signal",
                       source_selector=f"model={model}; block_samples={row['block_samples']}")
    for model in ("linear","sparse_lpv_arma","cnn","archived_src_corrected"):
        row={"delta_mse":float(scores[model]-scores[PRIMARY]),
             "relative_improvement_pct":float(100*(scores[model]-scores[PRIMARY])/scores[model])}
        record_metrics(ledger,row,row,model=model,split=PREDICTED,source=source,n=len(ids),targets=idx,
                       mask=mask_hash,qualifier="baseline minus primary; relative percentage denominator is named comparator",
                       source_selector=f"model={model}; paired with {PRIMARY}")
    return axes,all_forecasts,selected,head,estimates,largest_comparison_error


def audit_variants(root, axes, forecasts, ledger):
    res=root/"results"
    mhash=hashlib.sha256(axes["target_indices"].astype("<i8").tobytes()).hexdigest()
    span="[1792, 24000)"
    source="results/ablations/src_ablations.csv"
    f=pd.read_csv(root/source)
    for row in f.to_dict("records"):
        model=row["model"]
        assert row["n"]==22208 and row["mask_sha256"]==mhash
        for key,v in mstats(axes["actual"],forecasts[model]).items():
            check_close(v,row[key])
        control="post-selection feature intervention/refit; excluded from primary selection"
        if model in ("pi_src_adapter_off","pi_src_adapter_on"):
            control="adapter inference toggle, same fitted base coefficients"
        elif model in ("pi_src_pure_mse","pi_src_optimized","pi_src_compact",PRIMARY):
            control="training-fold selected finalist or protocol comparator"
        record_metrics(ledger,row,("mse","rmse","mae","bias","nrmse","feature_columns","history",
                                   "warm_median_ms","warm_iqr_ms","warm_p95_ms","parameters"),
                       model=model,split=PREDICTED,source=source,n=22208,targets=span,mask=mhash,
                       qualifier=control,source_selector=f"model={model}")
        record_metrics(ledger,row,("train_mse",),model=model,
                       split="training: sigtrain, stored in-sample estimate",source=source,n=22208,
                       targets=span,mask=mhash,qualifier=control,source_selector=f"model={model}")
    with np.load(res/"ablations/identical_adapter_controls.npz",allow_pickle=False) as a:
        np.testing.assert_array_equal(a["actual"],axes["actual"])
        np.testing.assert_array_equal(a["target_indices"],axes["target_indices"])
        for row in pd.read_csv(res/"ablations/identical_adapter_controls.csv").to_dict("records"):
            model=row["model"]
            mt=mstats(a["actual"],a[model+"_base"])["mse"]
            mc=mstats(a["actual"],a[model+"_matched"])["mse"]
            check_close([mt,mc,mt-mc],[row["uncorrected_mse"],row["matched_correction_mse"],row["gain_mse"]])
            record_metrics(ledger,row,("uncorrected_mse","matched_correction_mse","gain_mse","gain","tau","correction_memory","clip"),
                           model=model,split=PREDICTED,source="results/ablations/identical_adapter_controls.csv",
                           n=22208,targets=span,mask=mhash,qualifier="same frozen base weights; common historical correction, off versus on",
                           source_selector=f"model={model}")
    with np.load(res/"predictions/local_statistical_reference.npz",allow_pickle=False) as a:
        np.testing.assert_array_equal(a["actual"][1792:],axes["actual"])
        for row in pd.read_csv(res/"metrics/local_statistical_reference.csv").to_dict("records"):
            conv,model=row["convention"],row["model"]
            pred=a[conv+"__"+model]
            for lo,name in ((int(row["start"]),"development_mse"),(1792,"common_mask_mse")):
                got=mstats(a["actual"][lo:],pred[lo:])["mse"]
                check_close(got,row[name])
            record_metrics(ledger,row,("train_mse",),model=model,split="historical local training convention",run=conv,
                           source="results/metrics/local_statistical_reference.csv",
                           qualifier=f"{row['scheduling']}-RPM schedule",source_selector=conv)
            for metric,lo,n in (("development_mse",int(row["start"]),int(row["n"])),("common_mask_mse",1792,22208)):
                record_metrics(ledger,row,(metric,),model=model,split="local development, scope as named",run=conv,
                               source="results/metrics/local_statistical_reference.csv",n=n,targets=f"[{lo}, 24000)",
                               mask=mhash if lo==1792 else "",qualifier=f"{row['scheduling']}-RPM schedule",
                               source_selector=conv)
    paper=pd.read_csv(res/"metrics/published_paper_reference.csv")
    assert paper.train_mse.tolist()==[.4049,.4034,.3001,.3099]
    assert paper.authors_test_segment_mse.tolist()==[.3658,.3671,.2911,.2893]
    for row in paper.to_dict("records"):
        for metric,table,split in (("train_mse",5,"authors' fitted training signal"),
                                   ("authors_test_segment_mse",6,"authors' separate test segment")):
            record_metrics(ledger,row,(metric,),model=row["model"],split=split,
                           source="Chen et al. (2025), DOI 10.1016/j.ymssp.2024.112161",
                           source_selector=f"Table {table}; model={row['model']}",
                           units_override="published MSE scale, as labelled by source",
                           qualifier="source table MSE as printed; target mask unavailable; not a local head-to-head comparison")
    return len(f)


def search(root, selected, ledger):
    res=root/"results/optimization"
    trials=pd.read_csv(res/"trials.csv")
    folds=pd.read_csv(res/"fold_scores.csv")
    opt=yaml.safe_load((root/"configs/optimization.yaml").read_text())
    fspec=json_read(res/"selected_configs.json")
    assert len(trials)==41 and set(trials.status)=={"DONE"} and len(folds)==123
    assert fspec["selection"][0]["trial_id"]==selected["trial_id"] and fspec["selection"][0]["config"]==selected["config"]
    spans=dict.fromkeys(map(tuple,opt["folds"]))
    for row in trials.to_dict("records"):
        run=row["trial_id"]
        part=folds[folds.trial_id.eq(run)]
        assert len(part)==3 and set(zip(part.lo,part.hi))==set(spans)
        check_close(part.mse.mean(),row["fold_mse"])
        check_close(part.stress_mse.mean(),row["fold_stress_mse"])
        check_close(part.pulse_distortion.mean(),row["fold_pulse_distortion"])
        u=row["fold_mse"] + opt["stress_excess_weight"]*max(0,row["fold_stress_mse"]-row["fold_mse"])+opt["pulse_distortion_weight"]*row["fold_pulse_distortion"]+opt["per_coefficient_penalty"]*row["parameters"]
        check_close(u,row["utility"])
        record_metrics(ledger,row,("fold_mse","fold_mse_sd","fold_stress_mse","fold_pulse_distortion","utility","parameters","fit_s"),
                       model=row["variant"],run=run,split="training-only guarded three-block search",
                       source="results/optimization/trials.csv",n=7200,
                       targets="[4800,7200); [11200,13600); [18400,20800)",
                       qualifier="trial mean across three equal 2400-sample holdouts; 1792-sample guard",source_selector=f"trial_id={run}")
    return len(trials),len(folds)


def diagnostic(root, ledger):
    folder=root/"results/fault"
    eligibility=pd.read_csv(folder/"fault_eligibility.csv")
    assert len(eligibility)==200 and eligibility.eligible.sum()==198
    assert set(eligibility.loc[~eligibility.eligible,"record_id"])=={61,63}
    assert eligibility.eligible.eq(True).groupby(eligibility.group).sum().to_dict()=={"A":98,"B":100}
    per=pd.read_csv(folder/"fault_per_record.csv")
    metrics=pd.read_csv(folder/"fault_group_summary.csv")
    assert len(per)==1200 and len(metrics)==24
    saved_gap={}
    for row in metrics.to_dict("records"):
        model,mode=row["model"],row["metric"]
        q=per[per.eligible & per.model.eq(model)].sort_values("record_id")
        a=q.loc[q.group.eq("A"),mode].to_numpy()
        b=q.loc[q.group.eq("B"),mode].to_numpy()
        med_a,med_b=float(np.median(a)),float(np.median(b))
        rank_b=rankdata(np.r_[a,b])[len(a):]
        auc=(rank_b.sum()-len(b)*(len(b)+1)/2)/(len(a)*len(b))
        check_close([med_a,med_b,med_b-med_a,auc,b.min()-a.max()],
                    [row[k] for k in ("median_A","median_B","median_gap_B_minus_A",
                                       "conditional_auc_B_high","margin_min_B_minus_max_A")])
        rng=np.random.default_rng(77814)
        aa=rng.integers(len(a),size=(1000,len(a)))
        bb=rng.integers(len(b),size=(1000,len(b)))
        dboot=np.median(b[bb],axis=1)-np.median(a[aa],axis=1)
        check_close(np.quantile(dboot,[.025,.975]),[row["median_gap_ci_low"],row["median_gap_ci_high"]])
        saved_gap[model,mode]=(med_b-med_a,dboot)
        record_metrics(ledger,row,("median_A","median_B","median_gap_B_minus_A","median_gap_ci_low","median_gap_ci_high",
                                   "margin_min_B_minus_max_A","central_margin_p05_B_minus_p95_A","conditional_auc_B_high"),
                       model=model,split="candidate diagnostics, index-defined provisional groups",run=mode,
                       source="results/fault/fault_group_summary.csv",n=len(a)+len(b),targets="per candidate [1792, 32000)",
                       qualifier="not verified condition labels; matched 98 A + 100 B; 61/63 excluded; empirical within-cohort interval",
                       source_selector=f"model={model}; metric={mode}")
    pair=pd.read_csv(folder/"paired_pmi_bootstrap.csv")
    assert len(pair)==20
    for row in pair.to_dict("records"):
        p,p_boot=saved_gap[row["primary"],row["metric"]]
        r,r_boot=saved_gap[row["reference"],row["metric"]]
        check_close(p-r,row["delta_median_gap"])
        check_close(np.quantile(p_boot-r_boot,[.025,.975]),[row["paired_ci_low"],row["paired_ci_high"]])
    pulses=pd.read_csv(folder/"pulse_preservation.csv")
    for row in pd.read_csv(folder/"pulse_summary.csv").to_dict("records"):
        model,kind=row["model"],row["kind"]
        part=pulses[(pulses.model==model)&(pulses.kind==kind)]
        assert len(part)==row["cases"] and sum(part.delta_pmi>0)==row["positive_delta_count"]
        check_close(part.delta_pmi.median(),row["median_pmi_delta"])
        record_metrics(ledger,row,("cases","positive_delta_count","median_pmi_delta","max_pmi_delta","median_distortion",
                                   "median_energy_retention","median_pulse_correlation"),
                       model=model,split="synthetic injection on sigvalid",run=kind,
                       source="results/fault/pulse_summary.csv",n=row["cases"],targets="per case [1792, 24000)",
                       qualifier="designed pulse family; no physical crack-condition inference",
                       source_selector=f"model={model}; kind={kind}")
    for row in pd.read_csv(folder/"rpm_stress.csv").to_dict("records"):
        model,cond,scope=row["model"],row["condition"],row["scope"]
        if row["status"]=="ok":
            check_close(row["mse"]/row["clean_mse"],row["relative_mse"])
            check_close(row["lecturer_compatible"]-row["clean_pmi"],row["delta_pmi"])
        record_metrics(ledger,row,("mse","clean_mse","relative_mse","lecturer_compatible","clean_pmi","delta_pmi"),
                       model=model,split="sensor RPM intervention on sigvalid",run=f"{scope}/{cond}",n=22208,
                       targets="[1792, 24000)",source="results/fault/rpm_stress.csv",
                       qualifier="synthetic perturbation; separated predictor and diagnostic pathways",
                       source_selector=f"model={model}; scope={scope}; condition={cond}")
    return len(eligibility),int(eligibility.eligible.sum()),len(metrics)


def raw_records(root, raw, axes, ledger):
    """Verify file and record hashes and independently check exact joint overlap."""
    records=pd.read_csv(root/"results/record_inventory.csv")
    assert len(records)==202 and set(records.fs_hz)=={3200.0}
    rec=records.set_index("record_id")
    official_keys=("sigtrain","sigvalid")
    for key in official_keys:
        d=rec.loc[f"{key}:001"]
        assert d.N==24000
        check_close(d.duration_s,7.5)
    assert set(records[records.record_id.str.startswith("sigfault1")].N)=={32000}
    saved=pd.read_csv(root/"results/record_overlap_audit.csv")
    exact=saved[saved.exact_full_length]
    assert set(zip(exact.record_id,exact.reference,exact.best_start_zero_based))=={
        ("sigfault1:061","sigtrain",0),("sigfault1:063","sigvalid",0)}
    for row in records.to_dict("records"):
        vals={f:row[f] for f in ("N","fs_hz","duration_s","rpm_min","rpm_max","x_mean","x_std_population",
                                "stored_x_mean_channel1","stored_x_std_channel1","raw_reconstruction_max_abs_error")}
        record_metrics(ledger,vals,vals,model=row["record_id"],split="input record inventory",n=row["N"],
                       source="results/record_inventory.csv",targets=f"[0, {row['N']})",source_selector=row["record_id"])
    if raw is None:
        return []
    manifest=pd.read_csv(root/"inputs/input_manifest.csv").query("access_status=='READABLE'").set_index("filename")
    train={}
    vals={}
    paths=[raw/f"{name}.mat" for name in official_keys]+sorted(raw.glob("sigfault1_part_*.mat"))
    assert len(paths)==22
    for path in paths:
        assert sha(path)==manifest.at[path.name,"sha256"],path.name
        blob=loadmat(path,simplify_cells=True)
        if path.stem in official_keys:
            vals[path.stem]=blob[path.stem]
        else:
            part=int(path.stem.rsplit("_",1)[1])
            for j, item in enumerate(blob["sigfault1"],1):
                vals[f"sigfault1:{10*(part-1)+j:03}"]=item
    overlaps=[]
    for key,d in vals.items():
        x,rpm=np.asarray(d["x"]).reshape(-1),np.asarray(d["spd"]).reshape(-1)
        name=f"{key}:001" if key in official_keys else key
        row=rec.loc[name]
        assert len(x)==len(rpm)==int(row.N) and np.isfinite(x).all() and np.isfinite(rpm).all()
        assert hashlib.sha256(x.astype("<f8").tobytes()).hexdigest()==row.sha256_x_le_f64
        assert hashlib.sha256(rpm.astype("<f8").tobytes()).hexdigest()==row.sha256_rpm_le_f64
        check_close([rpm.min(),rpm.max(),float(d["fs"])],[row.rpm_min,row.rpm_max,row.fs_hz])
        if key in official_keys:
            train[key]=(x,rpm)
            if key=="sigvalid":
                for k,v in (("actual",x),("rpm",rpm),("time_s",d["t"])):
                    np.testing.assert_array_equal(np.asarray(v).reshape(-1)[axes["target_indices"]],axes[k])
            mu=np.asarray(d["xmean"]).flat[0]
            scale=np.asarray(d["xstd"]).flat[0]
            raw_channel=np.asarray(d["xraw"])
            if raw_channel.ndim==2:
                raw_channel=raw_channel[0] if raw_channel.shape[1]==x.size else raw_channel[:,0]
            check_close(mu+x*scale,raw_channel.flatten())
        else:
            for official,(a,b) in train.items():
                for lo in np.flatnonzero((x[:x.size-a.size+1]==a[0])&(rpm[:rpm.size-b.size+1]==b[0])):
                    if np.array_equal(x[lo:lo+a.size],a) and np.array_equal(rpm[lo:lo+b.size],b):
                        overlaps.append((key,official,int(lo)))
    assert overlaps==[("sigfault1:061","sigtrain",0),("sigfault1:063","sigvalid",0)]
    return overlaps


def checkpoint_registry(root, selected, head, evidence, ledger):
    reg=json_read(root/"checkpoints/registry.json")
    entries=reg["models"]
    cp=entries[PRIMARY]
    assert cp["config"]==selected["config"]
    checkpoints={}
    for model,obj in entries.items():
        for file, digest in obj["files"].items():
            if file not in checkpoints:
                checkpoints[file]=sha(root/file)
            assert checkpoints[file]==digest,file
    assert len(set(BASELINES)&set(entries))==21
    main=head.set_index("model")
    bfiles=pd.read_csv(root/"results/metrics/baseline_comparison.csv").set_index("model")
    scfg=pd.read_csv(root/"results/metrics/all_model_configs.csv").set_index("model")
    arch=pd.read_csv(root/"results/metrics/archived_src_metrics.csv").set_index("model")
    selected_refs=json_read(root/"results/optimization/selected_configs.json")["selection"]
    members={v["model"]:v["trial_id"] for v in selected_refs}
    trials=pd.read_csv(root/"results/optimization/trials.csv")
    stage_a = dict(zip(trials.trial_id,trials.stage))
    rows=[]
    replay_cert=json_read(root/"results/experiments/baseline_refit/fresh_refit_reconciliation.json")
    assert replay_cert["validated_models"]==21 and replay_cert["max_forecast_difference"]==0
    for model, obj in entries.items():
        if model in bfiles.index:
            b=bfiles.loc[model]
            family=b.family
            variant=b.evaluated_speed
            training=b.fit_evidence
            seed="7780 (neural guarded folds use 7780 + fold index)" if family in ("attention","neural") or model in ("narx","random_forest","gradient_boosting") else "deterministic / see config"
            features=scfg.loc[model,"config_json"]
            state="audited fresh fit and frozen checkpoint replay"
        elif model in arch.index:
            family="historical SRC"
            variant="legacy configuration or preserved forecast"
            training=arch.loc[model,"provenance"]
            seed=""
            features=json.dumps(obj.get("config",{}),sort_keys=True,separators=(",",":"))
            state=training
        else:
            family="SRC"
            variant=obj.get("config",{}).get("variant","")
            training=members.get(model,"registered control variant")
            if training in stage_a:
                training=f"{stage_a[training]}/{training}"
            seed="19 search/intervention seed; fitted readout deterministic"
            features=json.dumps(obj.get("config",{}),sort_keys=True,separators=(",",":"))
            state="saved model; replay audited"
        rows.append({"model_id":model,"display":main.loc[model,"display"] if model in main.index else "",
                     "family":family,"input_variant":variant,"checkpoint_paths_and_sha256":json.dumps(obj["files"],sort_keys=True),
                     "config_and_feature_set":features,"training_run":training,"seed":seed,
                     "split":PREDICTED,"warm_up":1792,"target_indices":"[1792, 24000)",
                     "mask_sha256":reg["target_mask_sha256"],"preprocessing":"as stored in fit/forecast; shared standardized target",
                     "evaluation_status":state})
    write_csv(evidence/"model_registry.csv",rows)
    cp_path=root/next(iter(cp["files"]))
    assert cp_path.stat().st_size==selected["checkpoint_bytes"]
    with np.load(cp_path,allow_pickle=False) as n, np.load(root/"checkpoints/src/pi_src_adapter_off.npz",allow_pickle=False) as off:
        groups=collections.Counter(map(str,n["groups"]))
        assert sum(groups.values())==len(n["kept"])==n["names"].size==selected["feature_columns"]
        n_fit=n["coefficients"].size+n["proxy"].size+n["projection"].size
        assert n_fit==selected["parameters"]==1456
        assert json.loads(str(n["config"]))==selected["config"]
        solver=json.loads(str(n["solver_info"]))
        assert solver["backend"]=="cholesky" and solver["innovation_crossfit"]==0
        for key in ("speed","proxy","center","scale","projection","coefficients","groups","names","kept"):
            np.testing.assert_array_equal(n[key],off[key],err_msg=f"Same-weight adapter control differs in {key}")
    timings=json_read(root/"results/manifests/src_prediction_verification.json")["tests"]
    primary_timing=next(row for row in timings if row["model"]==PRIMARY)
    repeat=np.asarray(primary_timing["warm_repeat_seconds"])
    assert len(repeat)==7
    check_close(1e3*np.median(repeat),selected["warm_median_ms"])
    check_close(1e3*np.diff(np.quantile(repeat,[.25,.75]))[0],selected["warm_iqr_ms"])
    common={"model":PRIMARY,"split":"selected frozen checkpoint",
            "source":"checkpoints/src/primary_pi_src.npz","qualifier":"stored matrices and attributes"}
    record_metrics(ledger,dict(groups),groups,**common,source_selector="groups; count by group")
    record_metrics(ledger,solver,(x for x in solver if x!="backend"),**common,source_selector="solver_info")
    record_metrics(ledger,selected,("feature_columns","parameters","history","checkpoint_bytes"),**common,source_selector="frozen config/model dimensions")
    return groups, solver, len(rows)


def support_evidence(root, ledger):
    specs=[
        ("results/metrics/speed_and_time_strata.csv",("n","mse","rpm_min","rpm_max"),"sigvalid development strata",("model","group")),
        ("results/metrics/vibration_rpm_parity.csv",("measured_mse","vibration_only_mse","delta_measured_minus_vibration_only","fit_s"),"sigvalid paired RPM-input variants",("model",)),
        ("results/metrics/baseline_seed_sensitivity_summary.csv",("seeds","mean_mse","sd_mse","min_mse","max_mse"),"sigvalid repeated comparator seeds",("model",)),
        ("results/fault/rpm_stress_summary.csv",("scenarios","successful","clean_mse","mean_stress_mse","worst_stress_mse","worst_relative_mse","median_delta_pmi","failure_rate"),"sigvalid sensor stress aggregate",("model",)),
        ("results/fault/paired_pmi_bootstrap.csv",("delta_median_gap","paired_ci_low","paired_ci_high"),"candidate diagnostics conditional paired bootstrap",("primary","reference","metric")),
        ("results/ablations/earlier_only_sensitivity.csv",("mse",),"train chronology stress refit",("hold_start","hold_stop")),
        ("results/timings/benchmark_table.csv",("batch_size","samples_per_second","batch_real_time_factor"),"full 24,000-sample CPU batch",("model",)),
        ("results/timings/feature_materialization.csv",("median_ms","repeats","scored_targets","principal_feature_bytes","max_abs_delta_vs_explicit","threads"),"feature-storage timing microbenchmark",("method",)),
        ("results/timings/gram_block_benchmark.csv",("elapsed_ms","coefficient_max_delta","rows","columns","gram_condition_estimate","iterations"),"small Gram accumulation timing microbenchmark",("accumulation_rows",)),
    ]
    for path, metrics, split, ids in specs:
        for row in pd.read_csv(root/path).to_dict("records"):
            n=row.get("n",row.get("batch_size",row.get("rows","")))
            record_metrics(ledger,row,metrics,model=row.get("model",row.get("primary",PRIMARY)),split=split,source=path,n=n,
                           source_selector="; ".join(f"{k}={row[k]}" for k in ids))
    for suffix in ("summary","extension_summary","synthetic_phase_checks"):
        path=f"results/eda/{suffix}.json"
        def visit(obj, ancestry=""):
            for k,v in obj.items():
                if isinstance(v,dict):
                    visit(v,ancestry+k+"/")
                elif isinstance(v,(int,float)) and not isinstance(v,bool):
                    record_metrics(ledger,{k:v},(k,),model="training EDA",split="sigtrain descriptive or guarded feature screen",
                                   source=path,source_selector=ancestry+k)
        visit(json_read(root/path))
    for name in ("sigtest","sigfault2"):
        path=f"results/frozen/{'availability' if name=='sigtest' else 'fault2_availability'}.json"
        assert json_read(root/path)["status"]=="NOT AVAILABLE"
        record_metrics(ledger,{"availability":"NOT AVAILABLE"},("availability",),model=PRIMARY,split=name,
                       source=path,qualifier="no input file supplied for frozen independent evaluation")


def source_type(relative):
    filename=str(relative)
    prefix=relative.parts[0]
    purposes={"src":"implementation used to interpret algorithms and audit numerical claims",
              "scripts":"reproduction and evidence build code", "configs":"frozen or evaluated settings",
              "results":"measured outputs, arrays or evidence scope", "figures":"release scientific visualization",
              "notebooks":"executed and reviewed evidence walkthrough", "checkpoints":"frozen fitted state",
              "docs":"release methodology, lineage or configuration guide", "report":"older report retained for conflict checks",
              "inputs":"input/archive provenance", "data":"split metadata", "tests":"validation code"}
    return purposes.get(prefix,"project environment or documentation")


def inventory(root, evidence, raw, reference_dir, attachment_dir, brief):
    root_items=("README.md","requirements.txt","requirements-lock.txt","pyproject.toml","CITATION.cff","LICENSE_NOTICE.md")
    admitted=("src","scripts","configs","results","figures","notebooks","checkpoints","docs","report","inputs","data","tests")
    tracked = subprocess.run(["git","-C",str(root),"ls-files","-z"],capture_output=True,check=False)
    paths = sorted(set(Path(s) for s in tracked.stdout.decode().split("\0") if s))
    candidates = []
    for f in paths:
        if f.is_absolute() or len(f.parts)>1 and f.parts[:2]==("report","final_submission"):
            continue
        if (f.parts[0] in admitted or str(f) in root_items) and (root/f).is_file():
            candidates.append(f)
    for f in sorted((root/"results/predictions").glob("*.npz")):
        if f.relative_to(root) not in candidates:
            candidates.append(f.relative_to(root))
    evidence_rows=[]
    for f in sorted(candidates):
        path=root/f
        evidence_rows.append({"path":f.as_posix(),"file_type":f.suffix.lstrip("."),
                              "purpose":source_type(f),"bytes":path.stat().st_size,
                              "sha256":sha(path),"provenance":"SRC release artifact",
                              "verification_status":"file bytes and identity checked"})
    originals=pd.read_csv(root/"inputs/input_manifest.csv").fillna("")
    for row in originals.to_dict("records"):
        name=row["filename"]
        path=None
        if raw and (raw/name).is_file():
            path=raw/name
        elif reference_dir and (reference_dir/name).is_file():
            path=reference_dir/name
        elif attachment_dir:
            supplied=Path(row["original_path"]).parts
            if len(supplied)==3 and supplied[0].startswith("source_") and supplied[-1]==name:
                candidate=attachment_dir/supplied[1]/supplied[2]
                if candidate.is_file():
                    path=candidate
        checked=False
        if path:
            assert sha(path)==row["sha256"],name
            checked=True
        evidence_rows.append({"path":f"original inputs/{name}","file_type":Path(name).suffix.lstrip("."),
                              "purpose":row["role"],"bytes":int(row["bytes"]) if row["bytes"] else "",
                              "sha256":row["sha256"],"provenance":"provided/course input manifest",
                              "verification_status":"supplied bytes rehashed" if checked else row["access_status"]})
    if brief:
        evidence_rows.append({"path":"original inputs/"+brief.name,"file_type":"md","purpose":"final report requirements",
                             "bytes":brief.stat().st_size,"sha256":sha(brief),"provenance":"uploaded report brief",
                             "verification_status":"supplied bytes rehashed"})
    write_csv(evidence/"source_inventory.csv",evidence_rows)
    return len(evidence_rows)


def replay_primary(root, raw, axes, forecasts, selected):
    """One bounded frozen-model replay. Fits and historical outputs remain untouched."""
    sys.path.insert(0,str(root/"src"))
    from src_gearbox.src_model import load_model
    model=load_model(root/"checkpoints/src/primary_pi_src.npz")
    ids=axes["target_indices"]
    with threadpool_limits(limits=1):
        for name in ("sigtrain","sigvalid"):
            d=loadmat(raw/f"{name}.mat",simplify_cells=True)[name]
            pred=model.predict(d["x"],d["spd"],float(d["fs"]))[ids]
            if name=="sigtrain":
                check_close(mstats(np.asarray(d["x"])[ids],pred)["mse"],selected["train_mse"])
            else:
                np.testing.assert_array_equal(pred,forecasts[PRIMARY])
    return True


def manuscript_gate(markdown, selected, estimates):
    text=markdown.read_text(encoding="utf-8")
    hits=list(re.finditer(r"^#{1,2}[ \t]+(?:\d+\.?[ \t]+)?(Abstract|Conclusions?)\b[^\n]*$",text,re.M|re.I))
    if not hits:
        raise ValueError("No standalone Abstract or Conclusion heading located")
    primary = estimates[PRIMARY]["mse"]
    baselines = {name: estimates[name]["mse"] for name in ("sparse_lpv_arma", "linear")}
    expected = {
        "training MSE": (selected["train_mse"], 6, ""),
        "development MSE": (primary, 6, ""),
        "archive replay MSE": (estimates["archived_src_corrected"]["mse"], 6, ""),
        **{f"gain over {name}": (100*(baseline-primary)/baseline, 2, "%")
           for name, baseline in baselines.items()},
    }
    checks={}
    for h in hits:
        following=re.search(r"^#{1,2} +",text[h.end():],re.M)
        end=h.end()+following.start() if following else len(text)
        part=text[h.end():end]
        for label, (value, digits, suffix) in expected.items():
            token = f"{value:.{digits}f}{suffix}"
            if not re.search(r"(?<![\d.])"+re.escape(token)+r"(?!\d)", part):
                raise ValueError(f"{h[1]} is missing the verified {label} ({token})")
        checks[h[1].lower()] = list(expected)
    if not any(k.startswith("abstract") for k in checks) or not any(k.startswith("conclusion") for k in checks):
        raise ValueError("Both an abstract and a conclusion are required for this gate")
    return checks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root",type=Path,default=ROOT)
    p.add_argument("--destination",type=Path,default=GEN.parent,help="Existing final_submission directory or new evidence staging directory")
    p.add_argument("--raw-dir",type=Path,help="Optional source MAT folder to rescan all 202 records")
    p.add_argument("--reference-dir",type=Path,help="Optional original source folder to rehash available PDFs/archives")
    p.add_argument("--attachment-dir",type=Path,help="Optional root of the original attachment copies identified by the input manifest")
    p.add_argument("--brief",type=Path,help="Optional original final Word report brief for source inventory")
    p.add_argument("--markdown",type=Path,help="Optionally verify the final Markdown abstract/conclusion")
    p.add_argument("--replay-primary",action="store_true",help="Check one frozen checkpoint on the two supplied records; requires --raw-dir")
    opt=p.parse_args()
    evidence=opt.destination/"evidence"
    qa=opt.destination/"qa"
    evidence.mkdir(parents=True,exist_ok=True); qa.mkdir(parents=True,exist_ok=True)
    root=opt.project_root.resolve()
    ledger=[]
    axes,forecasts,selected,head,estimates,maxerr=predictions(root,ledger)
    variants=audit_variants(root,axes,forecasts,ledger)
    trials,folds=search(root,selected,ledger)
    candidate,eligible,ds=diagnostic(root,ledger)
    overlaps=raw_records(root,opt.raw_dir,axes,ledger)
    groups,solver,models=checkpoint_registry(root,selected,head,evidence,ledger)
    support_evidence(root,ledger)
    sources=inventory(root,evidence,opt.raw_dir,opt.reference_dir,opt.attachment_dir,opt.brief)
    if opt.replay_primary and opt.raw_dir is None:
        p.error("--replay-primary requires --raw-dir")
    replayed=replay_primary(root,opt.raw_dir,axes,forecasts,selected) if opt.replay_primary else False
    gate=manuscript_gate(opt.markdown,selected,estimates) if opt.markdown else {}
    write_csv(evidence/"results_ledger.csv",ledger)
    dev=estimates[PRIMARY]["mse"]
    refs=[]
    for name in ("linear","sparse_lpv_arma","cnn","archived_src_corrected"):
        m=estimates[name]["mse"]
        delta=m-dev
        refs.append(f"| {name} | {m:.12f} | {delta:+.12f} | {100*delta/m:+.6f}% |")
    lines=[
        "# Numerical consistency audit","",
        "## Executed verification","",
        f"- Verified {len(head)} common-mask prediction rows, including all {len(BASELINES)} required comparators, from three frozen NPZ releases. Recalculated MSE, RMSE, MAE and bias. Largest absolute tabulation difference {maxerr:.3g}.",
        f"- Verified all {variants} SRC variants, all local statistical-reference forecast windows, and matched adapter contrasts from saved arrays. The original training estimates remain sourced to the stored in-sample evidence.",
        "- Reproduced all 78 common-window paired moving-block intervals with the recorded seed, block lengths and replicate counts. The intervals describe dependence within the same development record.",
        f"- Checked {trials} executed optimization candidates and {folds} guarded fold rows. Recalculated the mean fold components and objective for every candidate.",
        f"- Recalculated all {ds} provisional group medians, differences, margins and ranking AUCs from the actual per-record CSV. Reproduced each provisional median-gap interval and its 20 paired gap contrasts from 1,000 subject-ID replicates. {eligible} of {candidate} subjects were eligible. Checked paired intervention ratios and synthetic pulse summary medians.",
        f"- Verified underlying file/record hashes and exact complete joint overlaps for all 202 source records. The only full matches were {overlaps}." if opt.raw_dir else "- Full raw-MAT rehashing was not requested for this run. The record inventory, exclusion metadata and three canonical archived array bundles were checked.",
        f"- Rehashed the frozen artifacts for {models} model registry rows and indexed {sources} source files/manifest entries.",
        f"- Checked {sum(groups.values())} stored feature columns by group, the intercept and stage-one coefficient inventory, exact base parameter equality for adapter on/off, and the primary timing median and IQR from seven repeats. The selected solver record is `{solver['backend']}` with Gram condition estimate {solver['gram_condition_estimate']:.6f}.",
        f"- Checked {len(ledger)} exact/traceable scalar records in the generated results ledger.",
        "- Frozen primary replay recovered training MSE and all 22,208 development predictions. This was one inference-only checkpoint check." if replayed else "- Primary checkpoint SHA was checked. The full-record replay check can be requested with `--replay-primary --raw-dir`.",
        f"- Abstract and conclusion in `{opt.markdown.name}` checked against source metrics." if gate else "- The optional final-manuscript gate requires `--markdown`. Preflight covered the authoritative numerical sources.",
        "",
        "## Preserved scoped findings","",
        f"Primary development MSE {dev:.12f}. RMSE {estimates[PRIMARY]['rmse']:.12f}. MAE {estimates[PRIMARY]['mae']:.12f}. Training in-sample MSE {selected['train_mse']:.12f}. Target indices [1792, 24000), mask `{selected['mask_sha256']}`.","",
        "| Comparator | MSE | Comparator minus selected | Improvement, comparator denominator |",
        "| --- | ---: | ---: | ---: |",*refs,"",
        "Recovered train, development and archive timings retain their saved scope. No GPU, complete independently classified fault cohort or independent prediction result entered the comparison.","",
        "The paper values were checked against Tables 5 and 6 of the original PDF separately from local rows. The DOI and original scope remain attached to those rows.","",
    ]
    (qa/"numeric_consistency_audit.md").write_text("\n".join(lines),encoding="utf-8")
    print(f"Verified {len(head)} prediction rows, {trials} trial configurations, {folds} folds, {models} registry members, {len(ledger)} scalar entries.")


if __name__=="__main__":
    main()
