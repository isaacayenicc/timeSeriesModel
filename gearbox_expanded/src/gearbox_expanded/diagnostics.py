"""Frozen prediction diagnostics for the expanded model suite."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from gearbox_dhon.data import Record, find_input, iter_fault_records, load_record, sha256
from gearbox_dhon.faults import (
    SAMPLES_PER_REVOLUTION,
    _changed_record,
    _exact_reference_overlap,
    _pulse,
    _speed_perturbations,
    _threshold_free_statistics,
    diagnostics as measure,
)

START = 1792
SCREEN_IDS = tuple(sorted({*range(1, 201, 5), 63}))
FEATURES = ("pmi_course", "pmi_lag_corrected", "pmi_math", "pmi_unrelated_0p63_rev")
SENSOR_CASES = ("bias_-0.020", "bias_+0.020", "jitter_0.006", "quantized_1rpm",
                "delay_40ms", "dropout_hold_40ms")
PULSE_CASES = (
    (430, .4, .19, .0028, "periodic", 0.0),
    (1100, 1.25, .77, .0028, "periodic", 0.0),
    (850, 1.0, .57, .0018, "periodic", 0.2),
    (760, .18, .31, .0038, "periodic", 0.35),
    (760, .18, .31, .0038, "aperiodic", 0.0),
    (760, .65, .31, .0038, "aperiodic", 0.0),
)
Predictors = Mapping[str, Callable[[Record], np.ndarray]]


def _reference(data: Path, name: str) -> Record:
    path = find_input(data, name)
    if path is None:
        raise FileNotFoundError(f"Required reference {name} absent from {data}")
    return load_record(path)


def _target_prediction(model: Callable, record: Record, start: int = START) -> np.ndarray:
    pred = np.asarray(model(record), dtype=np.float64)
    if pred.shape != record.x.shape:
        raise ValueError("Expected a whole-record prediction aligned to raw sample indices")
    scored = pred[start:]
    if scored.size == 0 or not np.isfinite(scored).all():
        raise ValueError("Model has unavailable or nonfinite common-mask predictions")
    return scored


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".csv.tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_scope(path: Path, scope: str, rows: list[dict]) -> dict:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"record_count": 0, "model_count": 0, "prediction_failures": 0}
    _atomic_csv(frame, path / f"fault_{scope}_per_record.csv")
    # Use one eligible record intersection per score before comparing models.
    # The per-record sheet records failures and all available diagnostic values.
    summary_rows = []
    model_count = frame["model"].nunique()
    features_by_mask: dict[frozenset[int], list[str]] = {}
    for feature in FEATURES:
        if feature not in frame:
            continue
        finite = np.isfinite(frame[feature])
        common = frame.loc[frame["eligible"] & finite].groupby("record_id")["model"].nunique()
        common = frozenset(common[common == model_count].index)
        features_by_mask.setdefault(common, []).append(feature)
    for common, features in features_by_mask.items():
        comparable = [{**row, "eligible": row["eligible"] and row["record_id"] in common}
                      for row in rows]
        summary_rows.extend({**stat, "scope": scope, "comparison_record_count": len(common)}
                            for stat in _threshold_free_statistics(comparable) if stat["metric"] in features)
    if summary_rows:
        _atomic_csv(pd.DataFrame(summary_rows), path / f"fault_{scope}_group_summary.csv")
    return {
        "record_count": int(frame["record_id"].nunique()),
        "model_count": int(model_count),
        "predictions_evaluated": int((frame["model"] != "raw").sum()),
        "prediction_failures": int(frame["prediction_status"].ne("ok").sum()),
    }


def _fault_corpus(
    data: Path, path: Path, predictors: Predictors, full_names: tuple[str, ...],
    subset_ids: tuple[int, ...],
) -> tuple[dict[str, dict], list[dict]]:
    refs = [(name, _reference(data, name)) for name in ("sigtrain", "sigvalid")]
    checked = []
    by_fingerprint: dict[bytes, int] = {}
    screen_ids = frozenset(subset_ids)
    screen_rows: list[dict] = []
    full_rows: list[dict] = []
    for record in iter_fault_records(data):
        key = hashlib.sha256(np.ascontiguousarray(record.x, dtype="<f8"))
        key.update(np.ascontiguousarray(record.rpm, dtype="<f8"))
        same_record = by_fingerprint.setdefault(key.digest(), record.index)
        exclusion = _exact_reference_overlap(record, refs)
        if not exclusion and record.index != same_record:
            exclusion = f"full_vibration_rpm_duplicate_of_{same_record}"
        group = "A" if record.index <= 100 else "B" if record.index <= 200 else "unknown"
        checked.append({"record_id": record.index, "group": group, "eligible": not exclusion,
                        "source_part": record.source, "exclusion_reason": exclusion})

        screen = record.index in screen_ids
        pool = ("raw", *predictors) if screen else ("raw", *full_names) if full_names else ()
        target = record.x[START:]
        for name in pool:
            row: dict[str, Any] = {**checked[-1], "model": name, "target_count": len(target),
                                   "first_sample": START, "label_provenance": "unconfirmed_record_index",
                                   "prediction_status": "ok", "error": "", "mse": np.nan,
                                   "prediction_seconds": 0., "diagnostic_seconds": np.nan}
            try:
                if name != "raw":
                    tic = perf_counter()
                    prediction = _target_prediction(predictors[name], record)
                    row["prediction_seconds"] = perf_counter() - tic
                    residual = target - prediction
                    row["mse"] = float(np.mean(residual**2))
                else:
                    residual = target
                tic = perf_counter()
                values, _ = measure(residual, record.rpm[START:], record.t[START:])
                row.update(values)
                row["diagnostic_seconds"] = perf_counter() - tic
            except (ValueError, FloatingPointError) as exc:
                row.update(prediction_status="invalid", error=str(exc))
            if screen:
                screen_rows.append(row)
            if full_names and (name == "raw" or name in full_names):
                full_rows.append(row)

        if record.index % 20 == 0:
            print(f"Fault records scanned {record.index}; screen rows {len(screen_rows)},"
                  f" whole-corpus rows {len(full_rows)}", flush=True)
            _atomic_csv(pd.DataFrame(screen_rows), path / "fault_screen_per_record.csv")
            if full_rows:
                _atomic_csv(pd.DataFrame(full_rows), path / "fault_full_per_record.csv")

    if checked:
        _atomic_csv(pd.DataFrame(checked), path / "fault_eligibility.csv")
    return {
        "screen": _write_scope(path, "screen", screen_rows),
        "full": _write_scope(path, "full", full_rows),
    }, checked


def _pmi(error: np.ndarray, rpm: np.ndarray, record: Record) -> float:
    result, _ = measure(error, rpm[START:], record.t[START:])
    return float(result["pmi_course"])


def _robustness(
    record: Record, path: Path, predictors: Predictors, stress_names: tuple[str, ...],
    pulse_names: tuple[str, ...],
) -> dict:
    sensor = _speed_perturbations(record.rpm, record.fs)
    target = record.x[START:]
    pulse_signals = [
        _pulse(record, car, amp, phase, decay, kind=kind, modulation=mod)
        for car, amp, phase, decay, kind, mod in PULSE_CASES
    ]
    noise = np.random.default_rng(2026).normal(0, .05, len(record.x))
    stress_rows = []
    pulse_rows = []
    raw_before = _pmi(target, record.rpm, record)
    delta_raw = [_pmi(target + pulse[START:], record.rpm, record) - raw_before
                 for pulse in pulse_signals]

    for name in dict.fromkeys((*stress_names, *pulse_names)):
        model = predictors[name]
        base = _target_prediction(model, record)
        error = target - base
        clean_loss = float(np.mean(error**2))
        clean_pmi = _pmi(error, record.rpm, record)
        if name in stress_names:
            for condition in SENSOR_CASES:
                rpm = sensor[condition]
                disturbed = _changed_record(record, rpm=rpm)
                try:
                    altered = target - _target_prediction(model, disturbed)
                except (ValueError, FloatingPointError) as exc:
                    stress_rows.append({"model": name, "condition": condition, "scope": "predictor",
                                        "status": "invalid", "error": str(exc), "mse": np.nan,
                                        "pmi": np.nan, "clean_mse": clean_loss, "clean_pmi": clean_pmi})
                    continue
                for scope, residual, diagnostic_rpm in (
                    ("predictor", altered, record.rpm),
                    ("diagnostics", error, rpm),
                    ("both", altered, rpm),
                ):
                    stress_rows.append({"model": name, "condition": condition, "scope": scope,
                                        "mse": float(np.mean(residual**2)), "clean_mse": clean_loss,
                                        "pmi": _pmi(residual, diagnostic_rpm, record),
                                        "clean_pmi": clean_pmi, "status": "ok", "error": ""})
            noisy_error = target - _target_prediction(model, _changed_record(record, x=record.x + noise))
            stress_rows.append({"model": name, "condition": "input_noise_sd_0.05",
                                "scope": "predictor", "mse": float(np.mean(noisy_error**2)),
                                "clean_mse": clean_loss, "pmi": _pmi(noisy_error, record.rpm, record),
                                "clean_pmi": clean_pmi, "status": "ok", "error": ""})
        if name in pulse_names:
            for index, ((car, amp, phase, decay, kind, mod), injection) in enumerate(
                zip(PULSE_CASES, pulse_signals, strict=True)
            ):
                sample = injection[START:]
                synthetic = _changed_record(record, x=record.x + injection)
                response = target + sample - _target_prediction(model, synthetic)
                delta = response - error
                energy = float(sample @ sample)
                change_pmi = _pmi(response, record.rpm, record) - clean_pmi
                pulse_rows.append({
                    "model": name, "case": index + 1, "carrier_hz": car, "amplitude": amp,
                    "phase_revolutions": phase, "decay_seconds": decay, "kind": kind,
                    "amplitude_modulation": mod, "delta_pmi": change_pmi, "delta_raw_pmi": delta_raw[index],
                    "energy_retention": float(delta @ delta) / energy,
                    "distortion": float((delta - sample) @ (delta - sample)) / energy,
                    "pulse_correlation": float(np.corrcoef(delta, sample)[0, 1]),
                })
        print(f"Diagnostic stress complete for {name}", flush=True)

    stress = pd.DataFrame(stress_rows)
    pulses = pd.DataFrame(pulse_rows)
    if len(stress):
        stress["delta_pmi"] = stress["pmi"] - stress["clean_pmi"]
        stress["mse_ratio"] = stress["mse"] / stress["clean_mse"]
        _atomic_csv(stress, path / "rpm_stress.csv")
        predictors_only = stress[(stress.scope == "predictor") & stress.condition.isin(SENSOR_CASES)]
        summary = predictors_only.groupby("model", sort=False).agg(
            scenarios=("status", "size"), successful=("mse", "count"), mse_clean=("clean_mse", "first"),
            mse_mean=("mse", "mean"), mse_worst=("mse", "max"), median_pmi_delta=("delta_pmi", "median"),
        )
        _atomic_csv(summary.reset_index(), path / "rpm_stress_summary.csv")
    if len(pulses):
        _atomic_csv(pulses, path / "pulse_preservation.csv")
        table = pulses.assign(pmi_increased=pulses.delta_pmi > 0)
        table = table.groupby(["model", "kind"], sort=False).agg(
            cases=("case", "size"), positive_pmi=("pmi_increased", "sum"),
            median_correlation=("pulse_correlation", "median"),
            median_pmi_delta=("delta_pmi", "median"),
            median_energy_retention=("energy_retention", "median"),
        )
        _atomic_csv(table.reset_index(), path / "pulse_preservation_summary.csv")
    return {"rpm_rows": len(stress), "pulse_rows": len(pulses)}


def _plots(path: Path) -> None:
    import matplotlib.pyplot as plt

    for scope in ("screen", "full"):
        source = path / f"fault_{scope}_group_summary.csv"
        if not source.is_file():
            continue
        frame = pd.read_csv(source).query("metric == 'pmi_course'")
        if frame.empty:
            continue
        gap = frame["median_gap_B_minus_A"]
        low = np.maximum(0, gap - frame["median_gap_ci_low"])
        high = np.maximum(0, frame["median_gap_ci_high"] - gap)
        fig, ax = plt.subplots(figsize=(9, max(3, 1.25 + .37 * len(frame))), layout="constrained")
        positions = np.arange(len(frame))
        ax.errorbar(gap, positions, xerr=[low, high], fmt="o", lw=1.1, color="#275f88")
        ax.axvline(0, color="#4d5160", lw=.7, ls="--")
        ax.set(yticks=positions, yticklabels=frame["model"],
               xlabel="Group B minus group A median PMI",
               title=f"Provisional index-group separation on {int(frame.comparison_record_count.iloc[0])} records")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.23)
        fig.savefig(path / f"fault_{scope}_pmi_gap.svg")
        fig.savefig(path / f"fault_{scope}_pmi_gap.png", dpi=300)
        plt.close(fig)


def audit(path: Path, *, faults: bool = True, stress: bool = True) -> dict:
    """Independently recompute group rank AUC and check perturbation invariants."""
    output = {"status": "pass", "verified_scopes": {}, "checked_stress_rows": 0}
    for scope in ("screen", "full") if faults else ():
        per_record = path / f"fault_{scope}_per_record.csv"
        summary_file = path / f"fault_{scope}_group_summary.csv"
        if not per_record.is_file() or not summary_file.is_file():
            continue
        frame = pd.read_csv(per_record)
        summary = pd.read_csv(summary_file)
        if frame.duplicated(["record_id", "model"]).any():
            raise ValueError("Duplicate model/record rows")
        grouped = frame.groupby("record_id", sort=False)
        if (grouped[["target_count", "first_sample", "eligible"]].nunique() != 1).any().any():
            raise ValueError("Record eligibility or support differs across prediction methods")
        for feature, table in summary.groupby("metric", sort=False):
            kept = frame.pivot(index=["record_id", "group", "eligible"], columns="model", values=feature)
            kept = kept.dropna().reset_index()
            kept = kept[kept["eligible"]]
            for row in table.itertuples():
                a = kept.loc[kept.group == "A", row.model].to_numpy()
                b = kept.loc[kept.group == "B", row.model].to_numpy()
                if (len(a), len(b)) != (row.n_A, row.n_B):
                    raise ValueError("Published conditional cohort sizes do not match record scores")
                ranks = rankdata(np.r_[a, b])[len(a):]
                auc = (ranks.sum() - len(b) * (len(b) + 1) / 2) / (len(a) * len(b))
                np.testing.assert_allclose(
                    [auc, np.median(a), np.median(b), b.min() - a.max()],
                    [row.conditional_auc_B_high, row.median_A, row.median_B,
                     row.margin_min_B_minus_max_A], atol=1e-12, rtol=0,
                )
        output["verified_scopes"][scope] = int(len(summary))
    stress_path = path / "rpm_stress.csv"
    if stress and stress_path.is_file():
        frame = pd.read_csv(stress_path)
        pure = frame[(frame["scope"] == "diagnostics") & (frame["status"] == "ok")]
        np.testing.assert_allclose(pure["mse"], pure["clean_mse"], atol=1e-12, rtol=0)
        successful = frame[frame["status"] == "ok"]
        np.testing.assert_allclose(successful["delta_pmi"], successful["pmi"] - successful["clean_pmi"],
                                   atol=1e-12, rtol=0)
        output["checked_stress_rows"] = int(len(frame))
    _atomic_json(path / "audit.json", output)
    return output


def run(
    data_dir: Path, out: Path, prior: Path, predictors: Predictors, *,
    full_names: Sequence[str] = (), subset_ids: Sequence[int] = SCREEN_IDS,
    stress_names: Sequence[str] | None = None, pulse_names: Sequence[str] | None = None,
    include_stress: bool = True, include_faults: bool = True,
) -> dict:
    """Evaluate frozen predictors. Callers pass all selected names explicitly."""
    data, path, prior = Path(data_dir), Path(out) / "diagnostics", Path(prior)
    path.mkdir(parents=True, exist_ok=True)
    selected = tuple(dict.fromkeys(full_names))
    screen_ids = tuple(sorted(set(subset_ids)))
    sensor_names = tuple(predictors if stress_names is None else stress_names)
    impulse_names = tuple(selected if pulse_names is None else pulse_names)
    for name in (*selected, *sensor_names, *impulse_names):
        if name not in predictors:
            raise KeyError(f"No frozen predictor for {name}")
    tic = perf_counter()
    source = find_input(data, "fdt_featcal.m")
    prior_freeze = prior / "outputs" / "runs" / "development" / "frozen_manifest.json"
    status = {
        "status": "running", "first_zero_based_target": START,
        "angular_samples_per_revolution": SAMPLES_PER_REVOLUTION,
        "screen_ids": screen_ids, "full_models": selected,
        "predictor_names": tuple(predictors), "stress_models": sensor_names, "pulse_models": impulse_names,
        "label_status": "unconfirmed index groups, neither health labels nor calibration are available",
        "matlab_status": "literal-source semantics; missing original order-tracking helper and numerical gold output",
        "instructor_script_sha256": sha256(source) if source else None,
        "prior_freeze_sha256": sha256(prior_freeze) if prior_freeze.is_file() else None,
        "fault_inputs_reference_mask": "actual 1792:N samples of each recording",
        "phase_policy": "trapezoidal RPM integral and linear interpolation for offline diagnostics only",
    }
    _atomic_json(path / "diagnostic_status.json", status)
    print(f"Diagnostics use mask {START}:N, grid {SAMPLES_PER_REVOLUTION} samples/revolution,"
          f" and {len(screen_ids)} prespecified screening record IDs", flush=True)
    if include_stress:
        status.update(_robustness(_reference(data, "sigvalid"), path, predictors, sensor_names, impulse_names))
        _atomic_json(path / "diagnostic_status.json", status)
    if include_faults:
        status["scopes"], candidates = _fault_corpus(data, path, predictors, selected, screen_ids)
        status["record_count_audited"] = len(candidates)
        status["exclusions"] = {row["record_id"]: row["exclusion_reason"] for row in candidates
                                if not row["eligible"]}
        _plots(path)
    status["audit"] = audit(path, faults=include_faults, stress=include_stress)
    status["elapsed_seconds"] = perf_counter() - tic
    status["status"] = "complete"
    _atomic_json(path / "diagnostic_status.json", status)
    return status
