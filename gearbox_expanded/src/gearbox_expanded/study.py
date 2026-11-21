"""Execute and independently audit the expanded gearbox comparison."""
from __future__ import annotations

import argparse
from dataclasses import replace
from importlib import import_module
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pandas as pd
from scipy import signal
from threadpoolctl import threadpool_limits

from gearbox_dhon.data import find_input
from gearbox_dhon.dhon import load_model
from gearbox_dhon.inference import development_specs, load_predictor
from gearbox_dhon.results import block_difference, scores
from .protocol import FAMILIES, FOLDS, START, digest, guarded_segments, load_data, mse, write_json

ANCHORS = ("primary_dhon", "archive_dhon_optimized_corrected")
ANCESTOR = "archive_dhon_tau_context_extended_gated_corrected"
MODEL_NAMES = {
    "ar": "AR", "arma": "ARMA", "arima": "ARIMA", "arx": "ARX", "narx": "NARX",
    "sparse_lpv_ar": "Sparse LPV-AR", "sparse_lpv_arma": "Sparse LPV-ARMA",
    "linear": "Linear Regression", "ridge": "Ridge Regression", "lasso": "Lasso Regression",
    "elastic_net": "Elastic Net", "svr": "Support Vector Regression", "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting", "mlp": "MLP", "lstm": "LSTM", "gru": "GRU",
    "cnn": "CNN", "tcn": "TCN", "transformer": "Time Series Transformer", "tft": "Temporal Fusion Transformer",
    "primary_dhon": "Selected DHON", "archive_dhon_optimized_corrected": "Archived DHON checkpoint",
    ANCESTOR: "Archived DHON forecasts",
}


def get_module(family):
    return import_module(f"{__package__}.{family}")


def model_registry(out, prior):
    """Load one fixed predictor per available requested family and the DHON anchors."""
    predictors, metadata = {}, {}
    for family, slugs in FAMILIES.items():
        candidates = [(slug, out / "models" / f"{slug}.npz") for slug in slugs]
        if not all(path.is_file() for _, path in candidates):
            raise FileNotFoundError(f"Incomplete {family} checkpoints")
        module = get_module(family)
        for slug, path in candidates:
            model = module.load(path)
            predictors[slug] = lambda rec, model=model: model.predict(rec.x, rec.rpm, rec.fs)
            metadata[slug] = {"family": family, "relative_checkpoint": f"models/{path.name}",
                              "checkpoint_sha256": digest(path)}
    prior_run = prior / "outputs" / "runs" / "development"
    for slug, (kind, path) in development_specs(prior_run, prior).items():
        if slug in ANCHORS:
            predictors[slug] = load_predictor(kind, path, "cpu")
            metadata[slug] = {"family": "prior_dhon", "relative_checkpoint": path.relative_to(prior).as_posix(),
                              "checkpoint_sha256": digest(path)}
    if not set(ANCHORS).issubset(predictors):
        raise ValueError("Existing DHON checkpoint anchors are required")
    return predictors, metadata


def verify_causality(predict, record, pred):
    """Intervene at a fixed target; strict models use earlier observations only."""
    cut = 12_375
    x, rpm = record.x.copy(), record.rpm.copy()
    rng = np.random.default_rng(417)
    x[cut:] += rng.normal(0, .8, len(x)-cut)
    rpm[cut:] *= 1.002
    if record.rpm_min is not None and record.rpm_max is not None:
        norm = (rpm-record.rpm_min)/(record.rpm_max-record.rpm_min)
    else:
        norm = record.speed_normalized
    perturbed = replace(record, x=x, rpm=rpm, speed_normalized=norm)
    altered = predict(perturbed)
    maximum = float(np.max(np.abs(pred[START:cut+1]-altered[START:cut+1])))
    if not np.isfinite(maximum) or maximum > 3e-5:
        raise RuntimeError(f"Future intervention changed causal predictions by {maximum:g}")
    return maximum


def measure(predict, record, count=7):
    times = []
    arr = None
    for _ in range(count):
        start = perf_counter()
        arr = np.asarray(predict(record))
        times.append(perf_counter()-start)
    median = np.median(times)
    return arr, dict(batch_median_s=median, batch_p95_s=np.quantile(times,.95),
                     batch_iqr_s=np.ptp(np.quantile(times,[.25,.75])),
                     batch_rtf=median / (len(record.x)/record.fs),
                     batch_repeats=count, samples_per_s=len(record.x)/median)


def paired_rows(y, models):
    rows = []
    references = [s for s in ("primary_dhon","sparse_lpv_arma", "arma", ANCESTOR) if s in models]
    squared = {slug: (y - values) ** 2 for slug, values in models.items()}
    for slug, loss in squared.items():
        for ref in references:
            if slug == ref:
                continue
            delta = loss - squared[ref]
            for block in (320,640,1280):
                low, high = block_difference(delta, block=block, draws=2000)
                rows.append(dict(model=slug, reference=ref, block_samples=block,
                                 delta_mse=delta.mean(), low95=low, high95=high))
    return rows


def subgroup_rows(record, pred, slug):
    x, rpm = record.x[START:], record.rpm[START:]
    e = x - pred
    quarters = np.quantile(rpm, [.25, .50, .75])
    bins = np.digitize(rpm, quarters)
    up = (record.rpm[START-1:-1] - record.rpm[START-65:-65]) > 0
    rows = []
    groups = [(f"rpm_quartile_{k+1}", bins==k) for k in range(4)]
    groups += [("rising_rpm", up), ("nonrising_rpm", ~up)]
    for k, ids in enumerate(np.array_split(np.arange(len(x)), 6),1):
        mask = np.zeros(len(x), dtype=bool)
        mask[ids] = True
        groups.append((f"chronological_block_{k}", mask))
    for label, mask in groups:
        rows.append(dict(model=slug, group=label, targets=mask.sum(),
                         mse=np.mean(e[mask]**2), mae=np.mean(abs(e[mask]))))
    f, p = signal.welch(e, fs=record.fs, nperseg=1024)
    freq_rows = []
    for low,high in [(0,150), (150,500), (500,1000), (1000,1200), (1200,1601)]:
        selected = (f>=low)&(f<high)
        freq_rows.append(dict(model=slug, low_hz=low, high_hz=min(high,1600),
                              power=float(p[selected].sum()*(f[1]-f[0])),
                              resolution_hz=float(f[1]-f[0])))
    return rows, freq_rows


def summarize(data_dir, out, prior):
    """Re-score model arrays, replay portable checkpoints, and freeze research evidence."""
    train, valid = load_data(data_dir)
    predictors, metadata = model_registry(out, prior)
    store = {}
    workers = {}
    trials = []
    for family in FAMILIES:
        frame = pd.read_csv(out / f"{family}_metrics.csv")
        workers.update(frame.set_index("model").to_dict("index"))
        path = out / f"{family}_trials.csv"
        if path.exists():
            trials.append(pd.read_csv(path).assign(family=family))
        with np.load(out / f"{family}_predictions.npz", allow_pickle=False) as bundle:
            store.update({key: bundle[key] for key in bundle})
    if trials:
        pd.concat(trials, ignore_index=True).to_csv(out / "guarded_trials.csv", index=False)
    x = valid.x[START:]
    records, groups, spectral = [], [], []
    forecasts = {}
    verification = {}
    for slug, predictor in predictors.items():
        record_pred, timing = measure(predictor, valid)
        if record_pred.shape != valid.x.shape:
            raise ValueError(f"Misaligned {slug}")
        pred = record_pred[START:].astype(np.float64)
        tr = np.asarray(predictor(train))[START:].astype(np.float64)
        error = float(np.max(np.abs(store[slug][START:]-pred))) if slug in store else 0
        if error > 1e-5:
            raise RuntimeError(f"Replay discrepancy {slug}: {error:g}")
        shift = verify_causality(predictor, valid, record_pred)
        verification[slug] = dict(replay_max_abs=error, future_change_max_abs=shift, tolerance=3e-5)
        model_score = scores(x, pred)
        if slug in ANCHORS:
            fit_evidence = "Checkpoint replay of existing run"
        elif slug.startswith("sparse_lpv_"):
            fit_evidence = "New QR refit of prior guarded support"
        else:
            fit_evidence = "New guarded selection and fit"
        worker = workers.get(slug, {})
        records.append({**worker, **model_score, "model":slug, "display":MODEL_NAMES[slug],
                        "train_mse":mse(train.x[START:],tr), **timing, "fit_evidence":fit_evidence})
        forecasts[slug] = pred
        speed_group, bands = subgroup_rows(valid, pred, slug)
        groups.extend(speed_group)
        spectral.extend(bands)
        print(f"{slug}: MSE={model_score['mse']:.8f}; checkpointΔ={error:.2g}; futureΔ={shift:.2g}", flush=True)
    old = prior / "outputs" / "runs" / "development" / "predictions" / "baseline.npz"
    with np.load(old, allow_pickle=False) as archive:
        forecasts[ANCESTOR] = archive[ANCESTOR][START:].astype(np.float64)
    records.append({"model": ANCESTOR, "display": MODEL_NAMES[ANCESTOR], **scores(x, forecasts[ANCESTOR]),
                    "fit_evidence": "Metric recomputed from archived saved forecasts only"})
    metrics = pd.DataFrame(records).sort_values("mse", kind="stable")
    metrics.to_csv(out / "comparison_metrics.csv", index=False)
    pd.DataFrame(groups).to_csv(out / "by_operating_region.csv", index=False)
    pd.DataFrame(spectral).to_csv(out / "residual_spectral_bands.csv", index=False)
    pd.DataFrame(paired_rows(x, forecasts)).to_csv(out / "paired_mse_intervals.csv", index=False)
    np.savez_compressed(out / "matched_validation.npz", target_indices=np.arange(START,len(valid.x)),
                        time_s=valid.t[START:], rpm=valid.rpm[START:], actual=x, **forecasts)
    write_json(out / "forecast_verification.json", verification)
    code = {p.name:digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    write_json(out / "frozen_comparison.json", dict(
        schema=1, train={"name":"sigtrain.mat", "sha256":digest(find_input(data_dir,"sigtrain"))},
        validation={"name":"sigvalid.mat", "sha256":digest(find_input(data_dir,"sigvalid"))},
        target_mask={"start":START,"end_exclusive":len(valid.x)}, folds=FOLDS,
        speed_cutoff="previous sample", calibration=None, source_sha256=code,
        model_metadata=metadata, blind_status={key:("present_unevaluated" if find_input(data_dir,key) else "absent")
                                               for key in ("sigtest","sigfault2")},
        cpu_only=True, seven_repeats=True, platform=platform.platform(),
        hardware=(next((l.split(":",1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                        if l.startswith("model name")), platform.processor())
                  if Path("/proc/cpuinfo").exists() else platform.processor())))
    return metrics


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=[*FAMILIES, "fit", "compare", "diagnostics"])
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--prior", type=Path, default=Path("prior/gearbox_dhon_project"))
    p.add_argument("--out", type=Path, default=Path("outputs/expanded"))
    args = p.parse_args(argv)
    import torch
    torch.set_num_threads(1)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(1):
        if args.stage in FAMILIES:
            get_module(args.stage).run(args.data_dir, out, args.prior)
        elif args.stage == "fit":
            for family in FAMILIES:
                get_module(family).run(args.data_dir, out, args.prior)
        elif args.stage == "compare":
            summarize(args.data_dir, out, args.prior)
        elif args.stage == "diagnostics":
            from .diagnostics import run
            predictors, _ = model_registry(out, args.prior)
            run(args.data_dir, out, args.prior, predictors,
                full_names=("primary_dhon", "archive_dhon_optimized_corrected", "sparse_lpv_arma", "lasso"))


if __name__ == "__main__":
    main()
