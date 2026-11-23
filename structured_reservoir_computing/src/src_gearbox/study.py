"""Freeze training-selected PI-SRC fits, score aligned targets, and replay blind inputs."""
from dataclasses import asdict, replace
import argparse
import json
import platform
from pathlib import Path
import statistics
import time

import numpy as np
import pandas as pd
from scipy import signal, stats
from threadpoolctl import threadpool_info, threadpool_limits

from .optimization import source_digest
from .src_model import Config, fit, load_model
from .study_protocol import (
    MASK_HASH, PROJECT, START, STOP, TARGETS, atomic_json, atomic_npz, fingerprint,
    read_vector_record, resolve_record,
)

NAMES = {
    "primary_pi_src":"PI-SRC (selected)", "pi_src_pure_mse":"PI-SRC (clean-MSE selection)",
    "pi_src_compact":"PI-SRC-Compact", "pi_src_optimized":"PI-SRC-Optimized (uncorrected)",
    "src_linear":"SRC-Linear", "src_polynomial":"SRC-Polynomial", "src_speed":"SRC-Speed",
    "src_physics_linear":"SRC-Physics-Linear", "pi_src_fixed":"PI-SRC-Fixed",
}


def scores(x, forecast):
    error = np.asarray(x)-np.asarray(forecast)
    if error.ndim != 1 or not np.isfinite(error).all():
        raise ValueError("Scored forecast must be a complete finite one-dimensional vector")
    mse = float(np.mean(error**2))
    centered = error-error.mean()
    denom = float(centered@centered)
    acf = signal.correlate(centered,centered,method="fft")[len(error):len(error)+64]/max(denom,np.finfo(float).tiny)
    lags = np.arange(1,len(acf)+1)
    q = len(error)*(len(error)+2)*np.sum(acf**2/(len(error)-lags))
    signed = error[error!=0]>=0
    n_pos,n_neg = int(signed.sum()), int((~signed).sum())
    total = n_pos+n_neg
    runs = 1+int((signed[1:]!=signed[:-1]).sum())
    expectation = 1+2*n_pos*n_neg/max(1,total)
    variance = 2*n_pos*n_neg*(2*n_pos*n_neg-total)/max(1,total**2*(total-1))
    return dict(mse=mse,rmse=mse**.5,mae=float(np.mean(np.abs(error))),bias=float(error.mean()),
                n=len(error),nrmse=float(mse**.5/max(np.std(x),.00000001)),
                residual_acf_lag1=float(acf[0]) if len(acf) else 0,
                ljung_box_p64=float(stats.chi2.sf(q,len(acf))) if len(acf) else None,
                two_sided_runs_p=float(2*stats.norm.sf(abs((runs-expectation)/variance**.5))) if variance>0 else None)


def optional_record(data, name):
    try:
        return resolve_record(data,name)
    except FileNotFoundError:
        return None


def exact_targets(record,name):
    x,rpm,t,fs = record
    if len(x)!=STOP or fs!=3200:
        raise ValueError(f"{name} differs from expected {STOP} samples at 3200 Hz")
    return x,rpm,t,fs


def expanded_specs(selection, trials):
    spec = {entry["model"]:Config(**entry["config"]) for entry in selection}
    trial = {item.trial_id:Config(**json.loads(item.config_json)) for item in trials.itertuples() if item.status=="DONE"}
    standard = dict(src_linear="delays",src_polynomial="polynomial",src_speed="speed",
                    src_physics_linear="linear_physics",pi_src_fixed="fixed_four",src_generic="equal_cap_generic")
    spec.update({slug:trial[ref] for slug,ref in standard.items() if ref in trial})
    c = spec["primary_pi_src"]
    common = dict(adapter_gain=c.adapter_gain or .5,adapter_tau=c.adapter_tau,
                  adapter_memory=c.adapter_memory,adapter_clip=c.adapter_clip)
    spec.update({
        "pi_src_adapter_off":replace(c,adapter_gain=0.),
        "pi_src_adapter_on":replace(c,**common),
        "pi_src_no_angle":replace(c,mesh_cycles=0),
        "pi_src_no_carrier":replace(c,harmonics=(),fixed_hz=()),
        "pi_src_no_angle_no_carrier":replace(c,mesh_cycles=0,harmonics=(),fixed_hz=()),
        "pi_src_no_innovation":replace(c,innovation_lags=0,innovation_order=0),
        "pi_src_no_acceleration":replace(c,acceleration_window=0),
        "pi_src_no_augmentation":replace(c,augmentation=0.,pulse_preservation=0.),
        "pi_src_low_capacity":replace(c,nonlinear_cap=30),
        "pi_src_high_capacity":replace(c,nonlinear_cap=122),
        "pi_src_degree2":replace(c,polynomial_degree=2),
        "pi_src_order6":replace(c,harmonics=(37,74,111,148,185,222)),
        "pi_src_order12":replace(c,harmonics=tuple(37*i for i in range(1,13))),
        "pi_src_fixed_hz_only":replace(c,harmonics=(),fixed_hz=(1080.,1125.)),
    })
    if "src_polynomial" in spec:
        spec["src_polynomial_matched_adapter"] = replace(spec["src_polynomial"],**common)
    return spec


def train(data_dir, destination, selected_path, only_primary=False):
    doc = json.loads(selected_path.read_text())
    trial_log = pd.read_csv(selected_path.with_name("trials.csv"))
    train_path = resolve_record(data_dir,"sigtrain")
    data_signature = fingerprint(train_path)
    if data_signature != doc["training_file_sha256"]:
        raise ValueError("Training file differs from the frozen guarded selection.")
    if source_digest()!=doc["source_sha256"]:
        raise ValueError("Implementation/optimization sources changed after selection. Rerun or resume search.")
    record = exact_targets(read_vector_record(train_path),"sigtrain")
    x,rpm,t,fs = record
    full = expanded_specs(doc["selection"],trial_log)
    spec = {k:full[k] for k in ("primary_pi_src","pi_src_optimized","pi_src_adapter_off")} if only_primary else full
    model_dir = destination/"checkpoints/src"
    model_dir.mkdir(parents=True,exist_ok=True)
    rows, cached = [], {}
    previous = destination/"checkpoints/src_registry.json"
    saved = json.loads(previous.read_text()) if previous.is_file() else {}
    with threadpool_limits(limits=1):
        for slug,c in spec.items():
            path = model_dir/f"{slug}.npz"
            key = json.dumps(asdict(c),sort_keys=True)
            old = next((m for m in saved.get("models",[]) if m["model"]==slug),None)
            if (old and saved.get("training_sha256")==data_signature and saved.get("source_sha256")==doc["source_sha256"]
                    and path.is_file() and old["sha256"]==fingerprint(path)
                    and json.dumps(old["config"],sort_keys=True)==key):
                rows.append(old)
                print(f"Verified existing {slug}",flush=True)
                continue
            if key not in cached:
                start = time.perf_counter()
                m = fit(c,[(x,rpm)],fs)
                elapsed = time.perf_counter()-start
                cached[key] = m,elapsed
            m,elapsed = cached[key]
            if m.history>START:
                raise ValueError(f"{slug} receptive field exceeds common mask")
            m.save(path)
            rec = scores(x[TARGETS],m.predict(x,rpm,fs)[TARGETS])
            row = dict(model=slug,display=NAMES.get(slug,slug.replace("_"," ")),
                       checkpoint=path.relative_to(destination).as_posix(),sha256=fingerprint(path),
                       config=asdict(c),parameters=int(m.parameters),feature_columns=int(m.feature_count),
                       history=int(m.history),fit_s=elapsed,train_mse=rec["mse"],checkpoint_bytes=path.stat().st_size,
                       solver_info=m.solver_info)
            rows.append(row)
            print(f"Frozen {slug}: train MSE {rec['mse']:.7f}, history {m.history}",flush=True)
            if slug=="primary_pi_src":
                partial = dict(training_sha256=data_signature,source_sha256=doc["source_sha256"],
                               source_commit=doc["source_commit"],target_mask_sha256=MASK_HASH,
                               model_selection=doc["selection"],models=rows)
                atomic_json(previous,partial)
    output = dict(training_sha256=data_signature,source_sha256=doc["source_sha256"],
                  source_commit=doc["source_commit"],target_mask_sha256=MASK_HASH,
                  model_selection=doc["selection"],models=rows)
    atomic_json(previous,output)
    selected_yaml = PROJECT/"configs/physics_src.yaml"
    if destination == PROJECT and not only_primary:
        import yaml
        selected_yaml.write_text(yaml.safe_dump(asdict(spec["primary_pi_src"]),sort_keys=False),encoding="utf-8")
    return output


def load_registered(destination, registry, row):
    target = destination/row["checkpoint"]
    if registry["target_mask_sha256"]!=MASK_HASH or fingerprint(target)!=row["sha256"]:
        raise ValueError(f"Frozen checkpoint or mask hash failed: {target}")
    return load_model(target)


def perturbation(m, x,rpm,fs, reference):
    return max(float(np.max(np.abs(m.predict(
                   np.r_[x[:stop],x[stop:]+1.25*np.cos(np.arange(len(x)-stop))],
                   np.r_[rpm[:stop],rpm[stop:]*(1+.006*np.sin(np.arange(len(x)-stop)))],fs
               )[START:stop+1]-reference[START:stop+1]))) for stop in [START,4567,12375,21600])


def validate(data_dir, destination, primary_only=False):
    import psutil
    registry = json.loads((destination/"checkpoints/src_registry.json").read_text())
    path = resolve_record(data_dir,"sigvalid")
    x,rpm,t,fs = exact_targets(read_vector_record(path),"sigvalid")
    rows = []
    arrays = dict(target_indices=TARGETS,actual=x[TARGETS],rpm=rpm[TARGETS],time_s=t[TARGETS])
    validations = []
    with threadpool_limits(limits=1):
        pools = threadpool_info()
        for row in registry["models"]:
            slug = row["model"]
            if primary_only and slug!="primary_pi_src":
                continue
            a = time.perf_counter()
            m = load_registered(destination,registry,row)
            loaded_s = time.perf_counter()-a
            def predict():
                begin = time.perf_counter()
                estimate = m.predict(x,rpm,fs)
                return estimate,time.perf_counter()-begin
            full,cold_s = predict()
            repeated = [predict()[1] for _ in range(7 if slug=="primary_pi_src" else 3)]
            reloaded = load_registered(destination,registry,row).predict(x,rpm,fs)
            np.testing.assert_array_equal(full[TARGETS],reloaded[TARGETS])
            causality = perturbation(m,x,rpm,fs,full)
            if causality>1e-8:
                raise AssertionError(f"Future perturbation changes {slug} by {causality:.3g}")
            result = scores(arrays["actual"], full[TARGETS])
            arrays[slug] = full[TARGETS]
            per_model = destination/"results/predictions"/slug/"sigvalid.npz"
            atomic_npz(per_model,target_indices=TARGETS,prediction=full[TARGETS],target=x[TARGETS],
                       residual=x[TARGETS]-full[TARGETS],rpm=rpm[TARGETS],time_s=t[TARGETS],
                       checkpoint_sha256=row["sha256"],target_mask_sha256=MASK_HASH)
            rows.append(result|dict(model=slug,display=row["display"],train_mse=row["train_mse"],
                fit_s=row["fit_s"],warm_median_ms=statistics.median(repeated)*1000,
                warm_iqr_ms=float(np.percentile(repeated,75)-np.percentile(repeated,25))*1000,
                warm_p95_ms=float(np.percentile(repeated,95))*1000,load_ms=loaded_s*1000,
                first_predict_ms=cold_s*1000,warm_repeats=len(repeated),parameters=m.parameters,
                checkpoint_bytes=row["checkpoint_bytes"],feature_columns=m.feature_count,
                history=m.history,scope="newly fit on sigtrain",target_start=START,target_stop=STOP,
                mask_sha256=MASK_HASH,split="held-out development",device="CPU",
                config_json=json.dumps(row["config"],sort_keys=True)))
            validations.append(dict(model=slug,checkpoint_sha256=row["sha256"],replay_error=0.,
                                   max_through_target_change=causality,
                                   guard=int(m.history),warm_repeat_seconds=repeated,
                                   rss_after_wall_mib=psutil.Process().memory_info().rss/2**20))
            print(f"Scored {slug}: MSE {result['mse']:.7f}, Δfuture={causality:.3g}",flush=True)
    out = destination/"results"
    atomic_npz(out/"predictions/src_sigvalid.npz",**arrays)
    (out/"ablations").mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/"ablations/src_ablations.csv",index=False)
    atomic_json(out/"manifests/src_prediction_verification.json",dict(mask=MASK_HASH,era="development",
        record_sha256=fingerprint(path),tests=validations,threads=1,pools=pools,
        python=platform.python_version(),system=platform.platform(),warmup_range=[0,START]))
    p = rows[0]
    selected = registry["model_selection"][0]
    summary = selected|dict(validation_mse=p["mse"],validation_rmse=p["rmse"],validation_mae=p["mae"],
        train_mse=p["train_mse"],target_start=START,target_stop=STOP,targets=len(TARGETS),mask_sha256=MASK_HASH,
        parameters=p["parameters"],history=p["history"],feature_columns=p["feature_columns"],
        fit_s=p["fit_s"],warm_median_ms=p["warm_median_ms"],warm_iqr_ms=p["warm_iqr_ms"],
        checkpoint_bytes=p["checkpoint_bytes"],checkpoint_sha256=registry["models"][0]["sha256"],
        development_sha256=fingerprint(path),source_commit=registry["source_commit"],
        source_sha256=registry["source_sha256"])
    atomic_json(out/"metrics/src_summary.json",summary)
    atomic_json(out/"metrics/evaluation_mask.json",dict(start=START,stop_exclusive=STOP,count=len(TARGETS),
             dtype="<i8",target_indices_sha256=MASK_HASH,fs=fs,policy="one-step measured history through t-1"))
    return summary


def known_overlap(x, rpm, known):
    """Find exact joint vibration/RPM containment against known course signals."""
    for name, baseline, speed in known:
        (short,short_rpm),(long,long_rpm)=sorted([(x,rpm),(baseline,speed)],key=lambda pair:len(pair[0]))
        lead=np.flatnonzero((long==short[0]) & (long_rpm==short_rpm[0]))
        for start in lead:
            stop=start+len(short)
            if stop<=len(long) and np.array_equal(long[start:stop],short) and np.array_equal(long_rpm[start:stop],short_rpm):
                return name
    return None


def frozen_fault_records(data_dir, paths, target, model, checkpoint_hash, known):
    from .data.io_matlab import iter_fault_records
    from .diagnostics.pmi import lecturer_score
    state = target/"fault2_availability.json"
    identity = {p.name:fingerprint(p) for p in paths}
    prior = json.loads(state.read_text()) if state.is_file() else {}
    if not identity:
        if prior.get("status")!="SCORED":
            atomic_json(state,dict(status="NOT AVAILABLE",dataset="sigfault2",reason="No blind fault records supplied"))
        return
    if prior.get("status")=="SCORED":
        if prior["input_sha256"]!=identity or prior["checkpoint_sha256"]!=checkpoint_hash:
            raise RuntimeError("Changing the blind fault input or model requires a separate registered output")
        return
    rows = []
    with threadpool_limits(limits=1):
        for rec in iter_fault_records(data_dir,"sigfault2"):
            overlap=known_overlap(rec.x,rec.rpm,known)
            if overlap is not None:
                rows.append(dict(record_id=rec.index,n=0,mse=None,residual_pmi=None,verified_label=None,
                                 status=f"exact overlap with {overlap}"))
                continue
            if len(rec.x)<=START:
                raise ValueError(f"Insufficient warm-up in blind fault record {rec.index}")
            interval = slice(START,None)
            base = model.predict(rec.x,rec.rpm,rec.fs)[interval]
            error = rec.x[interval]-base
            pmi = lecturer_score(error,rec.rpm[interval],rec.t[interval])
            raw = pmi.value
            rows.append(dict(record_id=rec.index,n=len(error),mse=float(np.mean(error**2)),
                             residual_pmi=float(raw) if np.isfinite(raw) else None,verified_label=None,status="scored"))
    target.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(target/"sigfault2_pmi.csv",index=False)
    atomic_json(state,dict(status="SCORED",dataset="sigfault2",input_sha256=identity,
                          checkpoint_sha256=checkpoint_hash,verified_labels_supplied=False,
                          records=len(rows),excluded_exact_overlaps=sum(row["n"]==0 for row in rows),
                          interpretation="descriptive PMI without authoritative labels"))


def frozen(data_dir, destination):
    from .data.io_matlab import fault_sources
    registry = json.loads((destination/"checkpoints/src_registry.json").read_text())
    selected = next(row for row in registry["models"] if row["model"]=="primary_pi_src")
    target = destination/"results/frozen"
    test = optional_record(data_dir,"sigtest")
    fault_paths=fault_sources(data_dir,"sigfault2")
    m = load_registered(destination,registry,selected)
    known=[]
    if test is not None or fault_paths:
        for name in ("sigtrain","sigvalid"):
            routine=optional_record(data_dir,name)
            if routine is not None:
                x,rpm,_,_=read_vector_record(routine)
                known.append((name,x,rpm))
    frozen_fault_records(data_dir,fault_paths,target,m,selected["sha256"],known)
    data = dict(status="NOT AVAILABLE" if test is None else "SCORED",
                target="sigtest",checkpoint_sha256=selected["sha256"],split="frozen independent input",
                fault2_available=bool(fault_paths),
                verified_labels_supplied=False,normalizers_refit=False)
    availability = target/"availability.json"
    old = json.loads(availability.read_text()) if availability.is_file() else {}
    if test is None:
        data["reason"] = "No independent vibration test was supplied."
        if old.get("status")!="SCORED":
            atomic_json(availability,data)
        return
    if old.get("status")=="SCORED":
        if old.get("data_sha256")!=fingerprint(test) or old.get("checkpoint_sha256")!=selected["sha256"]:
            raise RuntimeError("A different independent test or checkpoint requires its own frozen output directory")
        return
    x,rpm,t,fs = read_vector_record(test)
    overlap=known_overlap(x,rpm,known)
    if overlap is not None:
        raise ValueError(f"Proposed independent vibration signal exactly overlaps {overlap}")
    idx = np.arange(START,x.size,dtype="<i8")
    if not len(idx):
        raise ValueError("Blind record is shorter than common warm-up")
    with threadpool_limits(limits=1):
        pred = m.predict(x,rpm,fs)[idx]
    result = scores(x[idx],pred)
    atomic_npz(target/"sigtest.npz",target_indices=idx,actual=x[idx],prediction=pred,residual=x[idx]-pred,time_s=t[idx],rpm=rpm[idx])
    atomic_json(target/"availability.json",data|dict(metrics=result,data_sha256=fingerprint(test),
                target_start=START,target_stop=len(x),prediction_sha256=fingerprint(target/"sigtest.npz")))


def main(action=None):
    ap = argparse.ArgumentParser(description=__doc__)
    if action is None:
        ap.add_argument("action",choices=["fit","score","frozen"])
    ap.add_argument("--data-dir",type=Path,default=PROJECT/"data/raw")
    ap.add_argument("--project-dir",type=Path,default=PROJECT)
    ap.add_argument("--selection",type=Path,default=None)
    ap.add_argument("--primary-only",action="store_true")
    args = ap.parse_args()
    action = action or args.action
    if action=="fit":
        train(args.data_dir,args.project_dir,args.selection or args.project_dir/"results/optimization/selected_configs.json",
              only_primary=args.primary_only)
    elif action=="score":
        validate(args.data_dir,args.project_dir,primary_only=args.primary_only)
    else:
        frozen(args.data_dir,args.project_dir)


if __name__=="__main__":
    main()
