"""Versioned guarded-fold search for physical and readout choices."""
from dataclasses import asdict, replace
from hashlib import sha256
from itertools import product
from pathlib import Path
import argparse
import json
import subprocess
import time
import traceback

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .study_protocol import (
    PROJECT, atomic_json, atomic_npz, fingerprint, fold_segments,
    read_vector_record, resolve_record, settings,
)


def candidate_grid(profile):
    """Build fixed stages A–D without access to the development record."""
    bank4 = (37, 74, 148, 259)
    bank6 = tuple(37*k for k in range(1, 7))
    bank12 = tuple(37*k for k in range(1, 13))
    base = dict(variant="PI-SRC-Optimized", lags=96, angular_steps=12, mesh_cycles=6,
                harmonics=bank4, fixed_hz=(), taus=(8,32,128), memory=512,
                innovation_order=96, innovation_lags=16, speed_degree=2,
                nonlinear_cap=96, ridge=.025, augmentation=0., pulse_preservation=0.,
                adapter_gain=0.)
    stages = {}
    stages["A"] = [
        ("delays",dict(variant="SRC-Linear")),
        ("polynomial",dict(variant="SRC-Polynomial")),
        ("speed",dict(variant="SRC-Speed")),
        ("linear_physics",dict(variant="SRC-Physics-Linear")),
        ("fixed_four",dict(variant="PI-SRC-Fixed",innovation_order=0,innovation_lags=0)),
        ("no_carrier",dict(harmonics=())),
        ("fixed_hz",dict(harmonics=(),fixed_hz=(1100.,))),
        ("combined_hz",dict(fixed_hz=(1100.,))),
        ("order6",dict(harmonics=bank6)),
        ("order12",dict(harmonics=bank12)),
        ("order2",dict(harmonics=(148,259))),
        ("order_sidebands",dict(harmonics=(1,147,148,149,259),taus=(8,32,128))),
        ("short_angular",dict(angular_steps=8,mesh_cycles=4)),
    ]
    stages["B"] = [
        ("low_capacity",dict(nonlinear_cap=32,ridge=.014)),
        ("high_capacity",dict(nonlinear_cap=160,ridge=.04)),
        ("no_innovations",dict(innovation_order=0,innovation_lags=0)),
        ("innovation32",dict(innovation_order=128,innovation_lags=32)),
        ("time_geometry",dict(mesh_cycles=0,lags=128)),
        ("lag_degree1",dict(lags=128,speed_degree=1,nonlinear_cap=64)),
        ("no_acceleration",dict(acceleration_window=0)),
        ("equal_cap_generic",dict(variant="SRC-Generic",interaction_policy="generic")),
        ("ols",dict(solver="ols",ridge=0.)),
        ("huber",dict(solver="huber",huber_delta=1.1,ridge=.018)),
        ("reduced_rank_degree2",dict(rank=28,polynomial_degree=2,ridge=.017,nonlinear_ridge_multiplier=3.)),
        ("sparse_lag_coherent",dict(lags=80,lag_history=256,lag_policy="multiresolution",
            interaction_policy="coherent",physics_ridge_multiplier=.55,ridge=.018)),
        ("least_ridge",dict(ridge=.003)),
    ]
    stages["C"] = [
        ("short",dict(taus=(6,20,72),memory=256,ridge=.015,angular_steps=16,mesh_cycles=4,lags=128)),
        ("long",dict(taus=(12,48,190),memory=720,ridge=.05,angular_steps=16,mesh_cycles=8,lags=128)),
        ("compact",dict(taus=(8,32),memory=256,angular_steps=8,mesh_cycles=4,lags=48,
                        harmonics=(148,259),nonlinear_cap=35,innovation_order=64,innovation_lags=8,ridge=.02)),
        ("sidebands_mixed",dict(harmonics=(37,74,147,148,149,259),fixed_hz=(1080.,1125.),
                             taus=(12,40,160),memory=512,ridge=.032,nonlinear_cap=75)),
        ("six_wide",dict(taus=(12,40,140),memory=512,harmonics=bank6,angular_steps=16,mesh_cycles=8,lags=128,
                         ridge=.018,nonlinear_cap=128,innovation_order=128,innovation_lags=24)),
    ]
    stages["D"] = [
        ("matched_corr",dict(adapter_gain=.5,adapter_tau=320.,adapter_memory=640,adapter_clip=1.75)),
        ("corr_light",dict(adapter_gain=.25,adapter_tau=160.,adapter_memory=320,adapter_clip=1.25)),
        ("corr_wide",dict(adapter_gain=.65,adapter_tau=512.,adapter_memory=960,adapter_clip=2.)),
        ("corr_long",dict(adapter_gain=.5,adapter_tau=512.,adapter_memory=1024,adapter_clip=1.75,
                          harmonics=bank6,taus=(12,40,140),innovation_lags=24,angular_steps=12)),
        ("protected_four",dict(variant="PI-SRC-Corrected",adapter_gain=.5,adapter_tau=640.,
            adapter_memory=1024,angular_steps=16,mesh_cycles=8,ridge=.022,nonlinear_cap=60,
            innovation_order=192,innovation_lags=24,pulse_preservation=.2,augmentation=.35)),
        ("protected_crossfit",dict(variant="PI-SRC-Corrected",adapter_gain=.5,adapter_tau=640.,
            adapter_memory=768,angular_steps=12,mesh_cycles=6,ridge=.023,nonlinear_cap=65,
            innovation_order=96,innovation_lags=16,innovation_folds=3,pulse_preservation=.16,augmentation=.25)),
    ]
    if profile == "quick":
        stages = {k: stages[k][:2] for k in stages}
    if profile == "full":
        stages["C"].extend((f"mesh{len(bank)}_memory{mem}_cap{cap}",
                            dict(harmonics=bank,memory=mem,nonlinear_cap=cap,taus=(8,30,100) if mem==320 else (16,56,140),
                                 lags=96 if mem==320 else 128,ridge=.012 if cap==60 else .040))
                           for bank,mem,cap in product((bank4,bank6,bank12),(320,640),(60,140)))
    return [(stage, name, dict(base,**changes)) for stage, items in stages.items() for name, changes in items]


def physical_candidates(trials):
    return trials.loc[trials.status.eq("DONE") &
               trials.variant.isin(["PI-SRC-Optimized","PI-SRC-Corrected"]) & trials.has_physics.astype(bool)]


def pareto_rows(trials):
    good = trials.loc[trials.status.eq("DONE")]
    labels = ["fold_mse","fold_stress_mse","fold_pulse_distortion","parameters"]
    metrics = good[labels].to_numpy()
    nondominated = [not np.any(np.all(metrics <= r,axis=1) & np.any(metrics < r,axis=1)) for r in metrics]
    return good.loc[nondominated].sort_values("utility")


def source_revision():
    try:
        return subprocess.check_output(["git","rev-parse","HEAD"],cwd=PROJECT,text=True,stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def source_digest():
    hasher = sha256()
    for p in [Path(__file__), Path(__file__).with_name("src_model.py"), PROJECT/"configs/optimization.yaml"]:
        hasher.update(p.name.encode("utf-8"))
        hasher.update(p.read_bytes())
    return hasher.hexdigest()


def pulse_pattern(x, lo, hi, amplitude):
    """Known damped transients separated in time; only within a held fold."""
    pulse = np.zeros_like(x)
    relative = np.arange(20)
    decay = np.exp(-relative / 6) * np.cos(2*np.pi*.19*relative)
    for index in np.linspace(lo+80,hi-100,8,dtype=int):
        pulse[index:index+len(decay)] += amplitude * decay
    return pulse


def config_model(proposed):
    from .src_model import Config
    return Config(**proposed)


def config_history(model, config):
    history = getattr(model,"history",None)
    if isinstance(history,(int,float,np.integer,np.floating)):
        return int(history)
    if config.adapter_gain:
        return max(max(getattr(config,"lags",0),config.memory)+config.adapter_memory,
                   config.innovation_order+config.innovation_lags+config.adapter_memory)
    return max(getattr(config,"lags",0), config.memory,config.innovation_order+config.innovation_lags)


def score_trial(stage, name, proposed, training, protocol, source_hash, revision):
    from .src_model import fit
    from psutil import Process
    xt, wt, _, fs = training
    config = config_model(proposed)
    max_guard = protocol["guard_samples"]
    fold_rows = []
    collected = []
    whole = time.perf_counter()
    proc = Process()
    for lo,hi in protocol["folds"]:
        test = slice(lo,hi)
        start = time.perf_counter()
        trained = fit(config,fold_segments(xt,wt,lo,hi,max_guard),fs)
        fit_time = time.perf_counter()-start
        history = config_history(trained,config)
        if history > max_guard:
            raise RuntimeError(f"Fitted history {history} exceeded the predeclared {max_guard}-sample guard")
        start = time.perf_counter()
        clean = trained.predict(xt,wt,fs)[test]
        predict_time = time.perf_counter()-start
        delta = protocol["speed_bias_fraction"]
        speed = np.arange(xt.size)/fs
        stress_signal = wt * (1 + delta*np.cos(2*np.pi*.45*speed))
        stress = trained.predict(xt,stress_signal,fs)[test]
        pulse = pulse_pattern(xt,lo,hi,protocol["pulse_amplitude"])
        injected = trained.predict(xt+pulse,wt,fs)[test]
        if not all(np.isfinite(f).all() for f in (clean,stress,injected)):
            raise ValueError("A fold prediction is nonfinite inside the common held block")
        distortion = np.sum((injected-clean)**2) / np.sum(pulse[test]**2)
        fold_rows.append(dict(lo=lo,hi=hi,n=hi-lo,mse=float(np.mean((xt[test]-clean)**2)),
            stress_mse=float(np.mean((xt[test]-stress)**2)),pulse_distortion=float(distortion),
            fit_s=fit_time,prediction_s=predict_time,parameters=int(trained.parameters),
            run_rss_mib=proc.memory_info().rss/2**20,history=history,
            solver=trained.solver_info.get("backend"), condition=trained.solver_info.get("gram_condition_estimate"),
            raw_features=trained.solver_info.get("raw_features"), fit_rows=trained.solver_info.get("training_rows")))
        collected.append((np.arange(lo,hi),clean,stress,injected,pulse[test]))
    mean = {k:float(np.mean([f[k] for f in fold_rows])) for k in ("mse","stress_mse","pulse_distortion","fit_s","prediction_s")}
    count = int(max(f["parameters"] for f in fold_rows))
    utility = mean["mse"]+protocol["stress_excess_weight"]*max(0,mean["stress_mse"]-mean["mse"])+              protocol["pulse_distortion_weight"]*mean["pulse_distortion"]+protocol["per_coefficient_penalty"]*count
    stack = np.concatenate(collected,axis=1)
    row = dict(stage=stage, trial_id=name, status="DONE", variant=config.variant,
        has_physics=bool(config.harmonics or config.fixed_hz or config.mesh_cycles), config=asdict(config),
        fold_mse=mean["mse"], fold_mse_sd=float(np.std([f["mse"] for f in fold_rows],ddof=1)),
        fold_stress_mse=mean["stress_mse"], fold_pulse_distortion=mean["pulse_distortion"],
        parameters=count, utility=utility, fit_s=mean["fit_s"], prediction_s=mean["prediction_s"],
        runtime_s=time.perf_counter()-whole, seed=protocol["seed"], code_sha256=source_hash, git_tree_base=revision,
        fold_scores=fold_rows)
    return row,dict(target_indices=stack[0].astype("<i8"), actual=xt[stack[0].astype(int)],
                    clean=stack[1], speed_stress=stack[2], pulse_prediction=stack[3], pulse=stack[4])


def summarize(folder, rows):
    df = pd.DataFrame(rows)
    df["config_json"] = df.config.map(lambda x:json.dumps(x,sort_keys=True))
    df.drop(columns=["config","fold_scores"],errors="ignore").to_csv(folder/"trials.csv",index=False)
    folds = [dict(stage=row["stage"],trial_id=row["trial_id"],**f) for row in rows for f in row.get("fold_scores",[])]
    pd.DataFrame(folds).to_csv(folder/"fold_scores.csv",index=False)
    return df


def run(directory, folder, profile, resume):
    raw = resolve_record(directory,"sigtrain")
    data_sig = fingerprint(raw)
    protocol = settings("optimization")
    training = read_vector_record(raw)
    source = source_digest()
    revision = source_revision()
    folder.mkdir(parents=True,exist_ok=True)
    candidates = folder/"trial_records"
    candidates.mkdir(exist_ok=True)
    rows = []

    def evaluate(stage, name, proposed):
        record = candidates/f"{stage}__{name}.json"
        prediction = formula_folder / f"{stage}__{name}.npz"
        job = sha256(json.dumps([proposed,protocol,data_sig,source],sort_keys=True).encode()).hexdigest()
        if resume and record.is_file():
            old = json.loads(record.read_text())
            if old["job_sha256"]==job and old["status"]=="DONE" and prediction.is_file():
                rows.append(old)
                return
        try:
            row, arrays = score_trial(stage,name,proposed,training,protocol,source,revision)
            atomic_npz(prediction, **arrays)
            print(f"{stage:1} {name:24} {row['fold_mse']:.7f} J={row['utility']:.7f}",flush=True)
        except (ValueError, RuntimeError, ArithmeticError, TypeError) as exc:
            row = dict(stage=stage,trial_id=name,status="FAILED",variant=proposed["variant"],config=proposed,
                       has_physics=False,error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc(),
                       code_sha256=source,git_tree_base=revision)
            print(f"{stage:1} {name:24} FAILED {row['error']}",flush=True)
        row["job_sha256"] = job
        row["training_file_sha256"] = data_sig
        atomic_json(record,row)
        rows.append(row)

    formula_folder = folder/"fold_predictions"
    with threadpool_limits(limits=1):
        for stage, name, proposed in candidate_grid(profile):
            evaluate(stage,name,proposed)
        trials = summarize(folder,rows)
        if "utility" not in trials or physical_candidates(trials).empty:
            raise RuntimeError(f"No successful PI-SRC trials. Inspect {folder/'trials.csv'}")
        initial = physical_candidates(trials).sort_values(["utility","trial_id"]).iloc[0]
        for label, scale in [("local_ridge_down",.45),("local_ridge_up",2.2)]:
            evaluate("E",label,dict(initial.config,ridge=initial.config["ridge"]*scale))
        if profile!="quick":
            evaluate("E","selected_uncorrected",dict(initial.config,adapter_gain=0.))
            evaluate("E","selected_correction",dict(initial.config,adapter_gain=.5,adapter_tau=512.,
                                                   adapter_memory=768,adapter_clip=1.75))
    all_trials = summarize(folder,rows)
    viable = physical_candidates(all_trials)
    primary = viable.sort_values(["utility","trial_id"]).iloc[0]
    pure = viable.sort_values(["fold_mse","trial_id"]).iloc[0]
    limit = max(viable.parameters.min(), protocol["compact_fraction"]*primary.parameters)
    compact = viable.loc[viable.parameters <= limit].sort_values(["utility","trial_id"]).iloc[0]
    neutral = viable.loc[viable.config.map(lambda x:x.get("adapter_gain",0)==0)].sort_values(["utility","trial_id"]).iloc[0]
    chosen = dict(primary_pi_src=primary,pi_src_pure_mse=pure,pi_src_compact=compact,pi_src_optimized=neutral)
    entries = []
    for name, row in chosen.items():
        entries.append({v:row[v] for v in ["trial_id","stage","utility","fold_mse","fold_stress_mse","fold_pulse_distortion","parameters","config"]}|{"model":name})
    atomic_json(folder/"selected_configs.json",dict(profile=profile,source_sha256=source,training_file_sha256=data_sig,
                                                     source_commit=revision,selection=entries,protocol=protocol))
    pd.DataFrame(entries).to_csv(folder/"selected_configs.csv",index=False)
    pareto_rows(all_trials).drop(columns=["config","fold_scores"],errors="ignore").to_csv(folder/"pareto_front.csv",index=False)
    print(f"Training-only selection: {primary.trial_id} (J={primary.utility:.7f}); {sum(all_trials.status.eq('DONE'))}/{len(all_trials)} evaluated",flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir",type=Path,default=PROJECT/"data/raw")
    ap.add_argument("--output-dir",type=Path,default=PROJECT/"results/optimization")
    ap.add_argument("--profile",choices=["quick","standard","full"],default="standard")
    ap.add_argument("--resume",action="store_true")
    args = ap.parse_args()
    run(args.data_dir,args.output_dir,args.profile,args.resume)


if __name__ == "__main__":
    main()
