"""Re-run 21 independently executable baseline checkpoints and the archived SRC anchors."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
from importlib import import_module
import json
import platform
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy import stats
from threadpoolctl import threadpool_limits

from .protocol import START, FAMILIES, digest, load_data, mse, write_json, write_table
from .reference import runs_test

BASELINE_SLUGS = tuple(model for models in FAMILIES.values() for model in models)
ARCHIVE_EXECUTABLE = ("archived_src_corrected", "archived_src_base", "archived_src_local_corrected")
ARCHIVE_FORECASTS = ("archived_src_earlier_corrected", "archived_src_earlier")
ARCHIVE_LIMIT = 2e-5
CAUSAL_TOLERANCE = 3e-5
CHECKPOINT_DIR = Path("checkpoints/baselines")
REPEAT_COUNT = 7


def load_predictor(slug, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    if slug in ARCHIVE_EXECUTABLE:
        from .archived_src import load
        return load(checkpoint_dir, slug)
    family = next((family for family, keys in FAMILIES.items() if slug in keys), None)
    if family is None:
        raise KeyError(f"Unknown fitted comparator {slug}")
    module = import_module(".trees" if slug in ("random_forest", "gradient_boosting")
                           else "." + family, package=__package__)
    return module.load(checkpoint_dir / f"{slug}.npz")


def _timed(model, record, repeats):
    calls = []
    for _ in range(repeats + 1):
        started = perf_counter()
        result = np.asarray(model.predict(record.x, record.rpm, record.fs), dtype=np.float64)
        calls.append(perf_counter() - started)
    if result.shape != record.x.shape:
        raise ValueError("One full acquisition-aligned prediction vector is required")
    warm = calls[1:]
    return result, warm, dict(cold_s=calls[0], median_s=float(np.median(warm)),
                             p95_s=float(np.quantile(warm, .95)),
                             iqr_s=float(np.ptp(np.quantile(warm, [.25,.75]))))


def _verify(model, record, baseline, tolerance=CAUSAL_TOLERANCE):
    cut = 12375
    x, rpm = record.x.copy(), record.rpm.copy()
    x[cut:] += 3
    rpm[cut:] *= 1.01
    altered = model.predict(x, rpm, record.fs)
    before = float(np.max(np.abs(baseline[START:cut + 1] - altered[START:cut + 1])))
    repeated = model.predict(record.x, record.rpm, record.fs)
    reset = float(np.max(np.abs(baseline[START:] - repeated[START:])))
    if max(before, reset) > tolerance:
        raise ArithmeticError(f"Causality or record reset failed: {before:g}, {reset:g}")
    return before, reset


def _error_metrics(actual, pred):
    e = actual - pred
    err = mse(actual, pred)
    residual = e - e.mean()
    den = float(residual @ residual)
    acf = (np.array([residual[k:] @ residual[:-k] for k in range(1,65)]) / den
           if den > 0 else np.zeros(64))
    n = len(e)
    return dict(mse=err, rmse=err**.5, mae=float(np.mean(np.abs(e))),
                nrmse_sd=float(err**.5/np.std(actual)), bias=float(np.mean(e)),
                r2=1-err/float(np.var(actual)), residual_acf_lag1=float(acf[0]),
                runs_p=runs_test(e),
                ljung_box_64_p=float(stats.chi2.sf(n*(n+2)*np.sum(acf**2/(n-np.arange(1,65))),64)))


def _scope_statistics(actual, predicted, speed, direction, model):
    quartiles = np.quantile(speed, [.25,.5,.75])
    bands = np.searchsorted(quartiles, speed, side="right")
    groups = [(f"rpm_quartile_{k+1}", bands == k) for k in range(4)]
    groups.extend([("decelerating", direction < 0), ("accelerating_or_steady", direction >= 0)])
    for group, keep in groups:
        if keep.any():
            yield dict(model=model, group=group, count=int(keep.sum()),
                       mse=mse(actual[keep], predicted[keep]))


def _rss():
    """Linux process high-water mark. Cumulative across models in this run."""
    if platform.system() != "Linux":
        return None
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)


def _memory(model):
    if hasattr(model, "maximum_raw_lag"):
        return model.maximum_raw_lag
    if hasattr(model, "stage1"):
        return max(model.a1+model.b2,model.a2)
    if hasattr(model, "spec"):
        return START if len(model.ma) else max((model.spec.p, *model.spec.speed_lags))
    cfg = getattr(model, "config", None)
    if hasattr(cfg, "history"):
        return cfg.history
    from .tabular_features import history_for
    feature_set = getattr(model, "feature_set", getattr(cfg,"feature_set",None))
    return history_for(feature_set) if feature_set is not None else None


def _write_predictions(predictions, split, arrays):
    path = predictions / "baselines_sigvalid.npz"
    np.savez_compressed(path, **arrays)
    shared = {k:arrays[k] for k in ("target_indices","actual","time_s","rpm")}
    for name in split:
        model_dir = predictions / "baselines" / name
        model_dir.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(model_dir/"sigvalid.npz", prediction=arrays[name],
                            residual=shared["actual"]-arrays[name], **shared)


def _archived(root, checkpoint_dir, record, axes, repeats):
    source = root/"references/baselines/archived_src_reference_vectors.npz"
    lineage = json.loads((root/"configs/baselines/archived_src_lineage.json").read_text(encoding="utf-8"))
    if digest(source) != lineage["saved_vector_reference_sha256"]:
        raise ValueError("Archived vector reference digest differs from its manifest")
    for name, formats in lineage["original_payloads"].items():
        for ext, expected_digest in formats.items():
            if digest(checkpoint_dir/f"{name}.{ext}") != expected_digest:
                raise ValueError(f"Archived checkpoint digest differs for {name}.{ext}")
    expected = np.load(source, allow_pickle=False)
    if not np.array_equal(expected["target_indices"], axes["target_indices"]):
        raise ValueError("Archive score mask differs from the designated validation mask")
    maskhash=hashlib.sha256(axes["target_indices"].tobytes()).hexdigest()
    rows, timing, predictions = [], [], dict(axes)
    for slug in ARCHIVE_EXECUTABLE+ARCHIVE_FORECASTS:
        if slug in ARCHIVE_EXECUTABLE:
            model = load_predictor(slug,checkpoint_dir)
            full, times, measure = _timed(model,record,repeats)
            prediction = full[START:]
            error = float(np.max(np.abs(prediction-expected[slug])))
            if error > ARCHIVE_LIMIT:
                raise ArithmeticError(f"{slug} did not replay its frozen prediction ({error:g})")
            causal, reset = _verify(model,record,full)
            timing.extend(dict(model=slug,repetition=i+1,batch_s=v) for i,v in enumerate(times))
            proof = dict(provenance="archived_checkpoint_replay", replay_max_abs=error,
                         causal_max_abs=causal, repeat_record_max_abs=reset, parameters=model.parameters, **measure)
        else:
            prediction = expected[slug]
            proof = dict(provenance="archived_forecast_only", replay_max_abs=None)
        rows.append(dict(model=slug,n=len(prediction),start=START,end_exclusive=len(record.x),mask_sha256=maskhash,
                         **_error_metrics(axes["actual"],prediction),**proof))
        predictions[slug]=prediction
        print(f"{slug}: MSE {rows[-1]['mse']:.9f}",flush=True)
    expected.close()
    write_table(root/"results/metrics/archived_src_metrics.csv", rows)
    write_table(root/"results/timings/archived_src_repeats.csv", timing)
    np.savez_compressed(root/"results/predictions/archived_src_sigvalid.npz",**predictions)
    return rows


def run(root: Path,data_dir: Path, *, repeats=REPEAT_COUNT):
    """Verify checkpoints, then execute and score each baseline on exact course data."""
    import torch

    root=Path(root)
    torch.set_num_threads(1)
    meta=json.loads((root/"configs/baselines/selection.json").read_text(encoding="utf-8"))
    train, valid = load_data(data_dir)
    for name, record in (("train",train),("validation",valid)):
        if digest(Path(data_dir)/record.source)!=meta[name]["sha256"]:
            raise ValueError(f"{name} MAT does not match the verified course dataset")
    indices=np.arange(START,len(valid.x),dtype="<i8")
    maskhash=hashlib.sha256(indices.tobytes()).hexdigest()
    axes=dict(target_indices=indices,actual=valid.x[indices],time_s=valid.t[indices],rpm=valid.rpm[indices])
    predictions=dict(axes)
    expected=np.load(root/"references/baselines/prior_baseline_predictions.npz",allow_pickle=False)
    if not np.array_equal(expected["target_indices"],indices):
        raise ValueError("Prior reference forecasts do not have the matched scoring indices")
    with (root/"results/metrics/baseline_prior_trials.csv").open(newline="",encoding="utf-8") as stream:
        counts=Counter(row["model"] for row in csv.DictReader(stream))
    rows, coverage, timing, grouped, config_rows = [], [], [], [], []
    checkpoint_dir=root/CHECKPOINT_DIR
    use_speed = set(BASELINE_SLUGS)-{"ar","arma","arima"}
    with threadpool_limits(1):
        for slug in BASELINE_SLUGS:
            prior=meta["models"][slug]
            path=checkpoint_dir/f"{slug}.npz"
            origin="protected_guarded_support_refit" if slug.startswith("sparse_lpv") else "protected_guarded_training"
            try:
                if digest(path)!=prior["prior_checkpoint_sha256"]:
                    raise ValueError("Frozen checkpoint SHA-256 does not match registry")
                start=perf_counter()
                model=load_predictor(slug,checkpoint_dir)
                load_s=perf_counter()-start
                full,repeat,stats_row=_timed(model,valid,repeats)
                causal,reset=_verify(model,valid,full)
                prediction=full[indices]
                old_delta=float(np.max(np.abs(prediction-expected[slug])))
                if old_delta > CAUSAL_TOLERANCE:
                    raise ArithmeticError(f"Prior independent replay forecast mismatch {old_delta:g}")
                train_pred=np.asarray(model.predict(train.x,train.rpm,train.fs))[START:]
                train_mse=mse(train.x[START:],train_pred)
                errors=_error_metrics(axes["actual"],prediction)
                rows.append(dict(model=slug,family=prior["family"],status="DONE",n=len(indices),start=START,
                                 end_exclusive=len(valid.x),mask_sha256=maskhash,fit_evidence=origin,
                                 evaluated_speed="past_only" if slug in use_speed else "vibration_only",
                                 fit_s_prior=prior.get("fit_s"), fit_rows_prior=prior.get("fit_rows",prior.get("fit_targets")),
                                 fit_state_prior=prior.get("fit_state"),
                                 guarded_fold_rows=counts[slug],fold_mse_prior=prior.get("fold_mean_mse"),
                                 train_mse=train_mse,generalization_gap=errors["mse"]-train_mse,**errors,
                                 checkpoint_load_s=load_s,**stats_row,
                                 samples_per_second=len(valid.x)/stats_row["median_s"],
                                 real_time_factor=stats_row["median_s"]/(len(valid.x)/valid.fs),
                                 checkpoint_bytes=path.stat().st_size,parameters=prior["parameters"],
                                 maximum_raw_lag=_memory(model),process_peak_rss_bytes_cumulative=_rss(),
                                 causal_max_abs=causal, repeated_record_max_abs=reset, prior_replay_max_abs=old_delta,
                                 device="cpu",threads=1,gpu_status="not_available_in_cpu_runtime"))
                predictions[slug]=prediction
                grouped.extend(_scope_statistics(axes["actual"],prediction,valid.rpm[indices-1],
                                                 valid.rpm[indices-1]-valid.rpm[indices-65],slug))
                timing.extend(dict(model=slug,repetition=i+1,batch_s=v,device="cpu",threads=1)
                              for i,v in enumerate(repeat))
                coverage.append(dict(model=slug,status="DONE",fit_evidence=origin,
                                     evaluation="independent_checkpoint_replay",
                                     limitation="Protected single selected seed availability where stochastic"))
                print(f"{slug}: {errors['mse']:.9f}, {stats_row['median_s']:.4f} s",flush=True)
            except (OSError,ValueError,ArithmeticError,RuntimeError) as error:
                coverage.append(dict(model=slug,status="NOT EVALUATED",fit_evidence=origin,
                                     evaluation="failed",limitation=str(error)))
                print(f"{slug}: NOT EVALUATED: {error}",flush=True)
            config_rows.append(dict(model=slug,family=prior["family"],checkpoint_sha256=prior["prior_checkpoint_sha256"],
                                    selected_config=json.dumps(prior["selection"],sort_keys=True),
                                    guarded_fold_rows=counts[slug],fit_scope=prior.get("fit_scope","full_training_refit")))
    expected.close()
    metrics=root/"results/metrics"
    write_table(metrics/"baseline_comparison.csv",sorted(rows,key=lambda row:row["mse"]))
    write_json(metrics/"baseline_comparison.json", rows)
    write_table(metrics/"model_coverage.csv",coverage)
    write_table(metrics/"baseline_configs.csv",config_rows)
    write_table(metrics/"baseline_speed_regimes.csv",grouped)
    write_table(root/"results/timings/baseline_repeats.csv",timing)
    _write_predictions(root/"results/predictions",[r["model"] for r in rows],predictions)
    write_json(metrics/"baseline_replay_provenance.json",dict(
        input_files={name:meta[name] for name in ("train","validation")},mask_sha256=maskhash,
        mask_encoding="little-endian signed int64",scored_slice=[START,len(valid.x)],targets=len(indices),
        protocol="one_step_previous_measurements",threads=1,repeats=repeats,python=sys.version.split()[0],
        torch=torch.__version__,hardware=platform.platform(),cuda_available=torch.cuda.is_available(),
        process_peak_rss_scope="cumulative process resident high-water; not isolated per-model allocation",
        timing_scope="full batch in-memory feature-to-output"))
    with threadpool_limits(1):
        _archived(root,checkpoint_dir,valid,axes,repeats)
    return rows
