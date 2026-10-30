"""Frozen-model, record-isolated descriptive analysis of fault-candidate data.

No classes, damage severity, physical speed unit or fault order are inferred.
Model representatives are selected using validation scores before fault loading.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
import torch

from .angle_resample import angle_resample
from .diagnostics import analytic_envelope, correlated_kurtosis, normalized_acf, signal_features
from .experiment import predict_network
from .io_matlab import sha256
from .models import build_network
from .windows import make_windows


CAUSAL_FAMILIES = ("zero", "mean", "persistence", "ols", "ar", "arx", "lpv",
                   "mlp", "gru", "tcn", "speed_tcn", "late_tcn", "capacity_tcn")
NEUTRAL_RECORDS = (1, 50, 100, 150, 200)


def select_models(root: Path, model_ids=None) -> list[dict]:
    """Freeze best validation run within each causal family, without fault tuning."""
    frame = pd.read_csv(root / "reports/results_summary.csv")
    if model_ids is None:
        selected = frame[frame.kind.isin(CAUSAL_FAMILIES)].sort_values(
            ["mse", "run_id"]
        ).groupby("kind", sort=False).head(1)
    else:
        selected = frame[frame.run_id.isin(model_ids)]
        if set(selected.run_id) != set(model_ids):
            raise ValueError("A requested model ID has no completed validation run")
    if len(selected) == 0 or (selected.kind == "shift_tcn").any():
        raise ValueError("Select completed causal models; shift control is excluded")
    return selected.to_dict("records")


class CachedPredictor:
    """Load one trusted local artifact once; reset each finite-window prediction.

    Only internally generated checkpoints may be loaded. The finite-window GRU
    implementation creates a fresh hidden state for every window, so no state
    crosses recordings. Return full N samples with NaN before its own lag.
    """
    def __init__(self, root: Path, run_id: str):
        self.cfg = json.loads((root / "experiments/runs" / run_id / "config.json").read_text())
        self.kind, self.lag = self.cfg["kind"], self.cfg["lag"]
        if self.kind == "shift_tcn":
            raise ValueError("Noncausal shift control cannot enter primary fault study")
        if self.kind in ("zero", "mean", "persistence"):
            self.value = float(np.load(root / "models" / (run_id + ".npz"))["value"])
        elif self.cfg["family"] == "linear":
            with (root / "models" / (run_id + ".pkl")).open("rb") as stream:
                self.model = pickle.load(stream)
        else:
            self.model = build_network(self.kind, self.lag, self.cfg["channels"])
            self.model.load_state_dict(torch.load(root / "models" / (run_id + ".pt"), map_location="cpu", weights_only=True))
            self.model.eval()

    def predict(self, record: dict, neural_windows: dict | None = None) -> np.ndarray:
        """Compute causal teacher-forced predictions; reuse same-lag neural arrays."""
        output = np.full(record["N"], np.nan)
        indices = np.arange(self.lag, record["N"])
        if self.kind in ("zero", "mean", "persistence"):
            output[indices] = record["x"][indices - 1] if self.kind == "persistence" else self.value
        elif self.cfg["family"] == "linear":
            # Exact fitted linear algebra without allocating an N-by-3L LPV
            # design. lfilter taps reverse the oldest-to-newest lag coefficients.
            from scipy.signal import lfilter
            coef = self.model.coef_
            wdot = lambda c: lfilter(c[::-1], [1.0], record["x"])[self.lag-1:-1]
            speed = record["spdnorm"][indices-1]
            pred = wdot(coef[:self.lag]) + self.model.intercept_
            if self.kind == "lpv":
                pred += speed*wdot(coef[self.lag:2*self.lag])
                pred += speed**2*wdot(coef[2*self.lag:])
            elif self.kind == "arx":
                pred += speed*coef[-2] + speed**2*coef[-1]
            output[indices] = pred
        else:
            cache = neural_windows if neural_windows is not None else {}
            if self.lag not in cache:
                cache[self.lag] = make_windows(record, self.lag, self.kind)[0]
            output[indices] = predict_network(self.model, cache[self.lag])
        return output


def angular_features(y: np.ndarray, t: np.ndarray, speed: np.ndarray, fs: float,
                     unit: str) -> dict:
    """Compute both coarse PMI variants from one shared Hilbert/ACF transform.

    Every branch is an explicit unit assumption and one-cycle modulation order.
    Output density avoids nominal coordinate downsampling; interpolation loss
    remains measured and cannot be interpreted as physical damage evidence.
    """
    factor = {"rpm": 1/60, "rad_s": 1/(2*np.pi), "rps": 1.0}[unit]
    density = int(2 ** np.ceil(np.log2(max(2, fs / np.min(speed * factor)))))
    transformed = angle_resample(y, t, speed, unit, density)
    z = transformed["signal"]
    envelope = analytic_envelope(z, "matlab_mean_restored_candidate")
    acf = normalized_acf(envelope, density)
    result = dict(speed_unit_assumption=unit, samples_per_cycle=density,
                  angular_n=len(z), angular_edge_loss_cycles=transformed["edge_loss_cycles"],
                  roundtrip_rmse=transformed["roundtrip_rmse"],
                  roundtrip_rmse_fraction_of_source_rms=transformed["roundtrip_rmse"]/signal_features(y)["rms"],
                  ck_order1=correlated_kurtosis(z, density),
                  assumed_modulation_order=1, physical_speed_unit_verified=False,
                  fault_order_verified=False, envelope_convention="matlab_mean_restored_candidate")
    for variant, lag in (("faithful_reference", density-1), ("scientifically_corrected", density)):
        rho = float(acf[lag])
        result[f"rho_{variant}"] = rho
        result[f"pmi_{variant}"] = rho/(1-rho) if np.isfinite(rho) and abs(1-rho)>32*np.finfo(float).eps else np.nan
        result[f"selected_lag_{variant}"] = lag
    return result


def run_fault_study(root, model_ids=None, max_records=None, resume=True,
                    parts_dir=None, output_tag=None) -> dict:
    """Analyze all independent records with frozen models and three assumptions.

    A JSON checkpoint per record makes a long run restartable. Cache identity
    includes input manifest, model hashes, selected IDs and diagnostic source.
    No record-wise health class is created and no model is retrained.
    max_records is a pilot only and cannot produce a complete status.
    """
    from .fault_io import iter_fault_records, verify_fault_parts

    root = Path(root)
    if max_records is not None and (not isinstance(max_records, int) or max_records < 1):
        raise ValueError("max_records must be a positive integer pilot limit")
    parts = Path(parts_dir) if parts_dir is not None else root / "data/raw/sigfault1_parts"
    report_dir = root / "reports"
    if max_records is not None or output_tag is not None:
        report_dir = report_dir / (output_tag or "fault_smoke")
    report_dir.mkdir(parents=True, exist_ok=True)
    verification = verify_fault_parts(parts)
    selected = select_models(root, model_ids)
    expected_records = int(json.loads((parts / "manifest.json").read_text())["record_total"])
    plan = dict(selection_rule="lowest original validation MSE within each causal family; no fault tuning",
                selected_models=[{k:r[k] for k in ("run_id", "kind", "mse", "artifact_sha256")} for r in selected],
                manifest_sha256=sha256(parts / "manifest.json"), burnin=256,
                speed_unit_assumptions=["rpm", "rad_s", "rps"], assumed_modulation_order=1,
                neutral_trace_record_ids=list(NEUTRAL_RECORDS), expected_records=expected_records,
                diagnostic_sources={name:sha256(Path(__file__).with_name(name)) for name in
                                    ("fault_study.py", "diagnostics.py", "angle_resample.py")})
    fingerprint = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    plan["fingerprint"] = fingerprint
    (report_dir / "fault_study_plan.json").write_text(json.dumps(plan, indent=2))
    cache_dir = root / "experiments/fault_study" / fingerprint[:12]
    cache_dir.mkdir(parents=True, exist_ok=True)
    traces = root / "experiments/fault_traces"
    traces.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    predictors = {r["run_id"]: CachedPredictor(root, r["run_id"]) for r in selected}
    kind_by_id = {r["run_id"]:r["kind"] for r in selected}
    time_rows, angle_rows, timings = [], [], []
    started = perf_counter()
    with threadpool_limits(limits=2):
        for sequence, record in enumerate(iter_fault_records(parts, verify=False), start=1):
            if max_records is not None and sequence > max_records:
                break
            record_id = int(record.get("global_record_id", record.get("record_index", sequence)))
            cache_path = cache_dir / f"record_{record_id:03d}.json"
            if resume and cache_path.exists():
                cached = json.loads(cache_path.read_text())
                time_rows.extend(cached["time_rows"]); angle_rows.extend(cached["angular_rows"])
                timings.append(cached["wall_seconds"])
                continue
            begin = perf_counter()
            burnin = plan["burnin"]
            if record["N"] <= burnin:
                raise ValueError("Fault record shorter than shared burn-in")
            base = dict(record_id=record_id, record_index=record_id,
                        source_record_id=str(record.get("record_id", record_id)),
                        part_filename=record.get("source_part", record.get("part_filename", "")),
                        input_sha256=record.get("source_part_sha256", record.get("part_sha256", "")),
                        n_evaluated=record["N"]-burnin, burnin_samples=burnin,
                        first_target_index=burnin, last_target_index=record["N"]-1,
                        fs=record["fs"], mean_speed_source_units=float(np.mean(record["spd"][burnin:])),
                        speed_std=float(np.std(record["spd"][burnin:])),
                        spdnorm_min=float(np.min(record["spdnorm"])),
                        spdnorm_max=float(np.max(record["spdnorm"])),
                        normalized_speed_outside_0_1_fraction=float(np.mean((record["spdnorm"]<0)|(record["spdnorm"]>1))))
            neural_windows = {}
            series = {"raw": record["x"][burnin:]}
            prediction_times = {"raw": 0.0}
            for run_id, predictor in predictors.items():
                stamp = perf_counter()
                pred = predictor.predict(record, neural_windows)
                prediction_times[run_id] = perf_counter() - stamp
                series[run_id] = record["x"][burnin:] - pred[burnin:]
            one_time, one_angle = [], []
            for model_id, y in series.items():
                stamp = perf_counter()
                feature = signal_features(y)
                one_time.append(dict(base, model_id=model_id, kind=kind_by_id.get(model_id,"raw"),
                                     mse=float(np.mean(y*y)), **feature,
                                     prediction_seconds=prediction_times[model_id],
                                     feature_seconds=perf_counter()-stamp))
                for unit in plan["speed_unit_assumptions"]:
                    stamp = perf_counter()
                    result = angular_features(y, record["t"][burnin:], record["spd"][burnin:], record["fs"], unit)
                    one_angle.append(dict(base, model_id=model_id, kind=kind_by_id.get(model_id,"raw"),
                                          **result, feature_seconds=perf_counter()-stamp))
            if record_id in NEUTRAL_RECORDS:
                np.savez_compressed(traces / f"record_{record_id:03d}.npz",
                                    record_id=record_id, target_indices=np.arange(burnin, record["N"]),
                                    time=record["t"][burnin:],speed=record["spd"][burnin:],
                                    **{name: values for name, values in series.items()})
            wall = perf_counter()-begin
            cache_path.write_text(json.dumps(dict(time_rows=one_time, angular_rows=one_angle, wall_seconds=wall)))
            time_rows.extend(one_time); angle_rows.extend(one_angle); timings.append(wall)
            print(json.dumps(dict(fault_record=record_id, n_targets=record["N"]-burnin,
                                  seconds=wall, models=len(predictors))), flush=True)
            status=dict(status="running", real_fault_records_processed=len(timings),expected_records=expected_records,
                        labels_verified=False,speed_unit_verified=False,plan_fingerprint=fingerprint)
            (report_dir/"fault_analysis_status.json").write_text(json.dumps(status,indent=2))
    frame = pd.DataFrame(time_rows)
    angular = pd.DataFrame(angle_rows)
    frame.to_csv(report_dir / "fault_record_features.csv", index=False)
    angular.to_csv(report_dir / "fault_angular_features.csv", index=False)
    complete = frame.record_id.nunique() == expected_records and max_records is None
    summary = summarize_fault_study(root, frame, angular, plan, report_dir)
    summary.update(status="complete_descriptive_unlabeled" if complete else "partial_pilot",
                   current_call_wall_seconds=perf_counter()-started,
                   total_record_compute_seconds=float(np.sum(timings)),
                   median_record_compute_seconds=float(np.median(timings)),
                   verification=verification)
    if complete:
        render_fault_figures(root, frame, angular, summary)
    # The summary is the completion marker: write after complete CSVs/figures.
    (report_dir / "fault_study_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (report_dir / "fault_analysis_status.json").write_text(json.dumps({k:summary[k] for k in
        ("status", "real_fault_records_processed", "labels_verified", "speed_unit_verified")}, indent=2))
    return summary


def summarize_fault_study(root: Path, frame: pd.DataFrame, angular: pd.DataFrame, plan: dict,
                          report_dir: Path | None = None) -> dict:
    """Write descriptive summaries/correlations, never fault-class metrics."""
    report_dir = root/"reports" if report_dir is None else report_dir
    scalar = ["mse", "rms", "crest_factor", "kurtosis", "prediction_seconds"]
    model_summary = frame.groupby(["model_id", "kind"])[scalar].agg(["mean","median","std","min","max"])
    model_summary.columns = ["_".join(c) for c in model_summary.columns]
    model_summary = model_summary.reset_index()
    model_summary.to_csv(report_dir / "fault_model_summary.csv",index=False)
    angular_summary=angular.groupby(["model_id","kind","speed_unit_assumption"])[
        ["ck_order1", "pmi_faithful_reference", "pmi_scientifically_corrected", "roundtrip_rmse"]].agg(["mean","median","std","min","max"])
    angular_summary.columns=["_".join(c) for c in angular_summary.columns]
    angular_summary.reset_index().to_csv(report_dir / "fault_angular_summary.csv",index=False)
    correlations=[]
    for name, group in frame.groupby("model_id"):
        for feature in ("rms", "crest_factor", "kurtosis"):
            for covariate in ("record_index", "mean_speed_source_units", "speed_std"):
                value=float(spearmanr(group[feature],group[covariate]).statistic) if len(group)>2 else np.nan
                correlations.append(dict(model_id=name,feature=feature,covariate=covariate,spearman_rho=value,
                                         interpretation="descriptive association; record order is not fault severity"))
    pd.DataFrame(correlations).to_csv(report_dir/"fault_feature_correlations.csv",index=False)
    recommended=json.loads((root/"experiments/best_model.json").read_text())["run_id"]
    baseline=frame[frame.model_id=="raw"]
    recommended_frame=frame[frame.model_id==recommended]
    compare=dict(raw={k:float(baseline[k].median()) for k in ("rms","crest_factor","kurtosis")})
    if len(recommended_frame):
        compare[recommended]={k:float(recommended_frame[k].median()) for k in ("rms","crest_factor","kurtosis")}
    return dict(real_fault_records_processed=int(frame.record_id.nunique()),
                n_records=int(frame.record_id.nunique()),
                selected_models=plan["selected_models"],n_models=len(plan["selected_models"]),
                n_time_feature_rows=len(frame),n_angular_feature_rows=len(angular),
                n_samples_each_evaluated=int(frame.n_evaluated.iloc[0]),
                record_duration_s=float((frame.last_target_index.iloc[0]+1)/frame.fs.iloc[0]),
                fs=float(frame.fs.iloc[0]),labels_verified=False,speed_unit_verified=False,
                assumed_modulation_order=1,reference_state_health_verified=False,
                raw_and_recommended_medians=compare,recommended_model_id=recommended,
                unit_assumptions=plan["speed_unit_assumptions"],
                normalized_speed_outside_0_1_fraction_min=float(baseline.normalized_speed_outside_0_1_fraction.min()),
                normalized_speed_outside_0_1_fraction_max=float(baseline.normalized_speed_outside_0_1_fraction.max()),
                prediction_mse_mean_ranking=model_summary[model_summary.model_id!="raw"].sort_values("mse_mean")[
                    ["model_id","mse_mean"]].to_dict("records"),
                physical_detection_metrics="not estimable without validated fault labels",
                record_order_semantics="unknown; no onset/severity/class labels inferred",
                plan_fingerprint=plan["fingerprint"])


def render_fault_figures(root: Path, frame: pd.DataFrame, angular: pd.DataFrame, summary: dict) -> None:
    """Create genuine-data plots and a provenance sidecar for manifest merging."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory=root/"reports/figures"; directory.mkdir(exist_ok=True)
    best=summary["recommended_model_id"]
    labels={"raw":"Supplied x",best:"LPV-AR residual"}
    rows=[]
    def save(fig,name,title):
        fig.tight_layout(); path=directory/(name+".png");fig.savefig(path,dpi=180);plt.close(fig)
        rows.append(dict(figure_id=name,path=str(path.relative_to(root)),title=title,
                         script="src/rotmach/fault_study.py:render_fault_figures",
                         runs=";".join(frame.model_id.unique()),
                         input_hashes=sha256(root/"data/raw/sigfault1_parts/manifest.json"),
                         sha256=sha256(path),destination="fault report/notebook/slides"))
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    for model in labels:
        data=frame[frame.model_id==model].sort_values("record_id")
        axes[0].plot(data.record_id,data.rms,label=labels[model],lw=1.2)
        axes[1].plot(data.record_id,data['kurtosis'],label=labels[model],lw=1.2)
    axes[0].set_ylabel("RMS (dimensionless)");axes[1].set_ylabel("Pearson kurtosis")
    axes[1].set_xlabel("Record provenance index; physical meaning unknown")
    axes[0].legend();axes[0].set_title("200 separate recordings; equal targets 256–31,999")
    save(fig,"fault_rms_trends","Record-wise raw/residual RMS and kurtosis; no inferred classes")
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,column in zip(axes,("rms","crest_factor","kurtosis")):
        ax.boxplot([frame.loc[frame.model_id==m,column] for m in labels],tick_labels=list(labels.values()),showfliers=False)
        ax.set_title(column.replace("_"," "));ax.tick_params(axis="x",rotation=15)
    fig.suptitle("Descriptive record distributions; not fault-detection accuracy")
    save(fig,"fault_indicator_distributions","Distributions across all200independent record IDs")
    fig,axes=plt.subplots(1,2,figsize=(13,7))
    models=frame[frame.model_id!="raw"].groupby("model_id").mse.mean().sort_values()
    axes[0].barh(models.index,models.values);axes[0].set_xlabel("Mean candidate-record residual MSE")
    pmis=angular[angular.speed_unit_assumption=="rad_s"].groupby("model_id").pmi_scientifically_corrected.median()
    axes[1].barh(models.index,[pmis[m] for m in models.index]);axes[1].set_xlabel("Median PMI; rad/s ASSUMPTION")
    axes[1].set_title("Different objective; larger is not proven better detection")
    save(fig,"fault_prediction_diagnostic_tradeoff","Candidate prediction error and assumption-based descriptive PMI")
    fig,ax=plt.subplots(figsize=(8,4))
    for model in labels:
        data=angular[angular.model_id==model].groupby("speed_unit_assumption").pmi_scientifically_corrected.median()
        order=["rpm","rad_s","rps"]
        ax.plot(order,[data[x] for x in order],"o-",label=labels[model])
    ax.set_ylabel("Median corrected PMI");ax.set_xlabel("Unverified source speed-unit assumption")
    ax.set_title("One-cycle modulation assumed; no verified defect order");ax.legend()
    save(fig,"fault_unit_sensitivity","PMI sensitivity to threeunverifiedspeedunit interpretations")
    trace=np.load(root/"experiments/fault_traces/record_001.npz")
    fig,axes=plt.subplots(2,1,figsize=(10,5),sharex=True)
    keep=np.arange(min(800,len(trace["time"])))
    for ax,model in zip(axes,("raw",best)):
        ax.plot(trace["time"][keep],trace[model][keep],lw=.8)
        ax.set_ylabel("Dimensionless amplitude");ax.set_title(labels[model])
    axes[-1].set_xlabel("Time (s), neutral provenance record1, first800evaluatedsamples")
    save(fig,"fault_neutral_record_overlay","Neutral record1 raw/residual; equal evaluated samples")
    pd.DataFrame(rows).to_csv(root/"reports/fault_figure_manifest.csv",index=False)
