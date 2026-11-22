"""All-record inference, transient checks and provisional index-group summaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .inverse_filtering import START, forecast
from .pmi import MODE_KEYS, measure

REPRESENTATIVE_IDS = (1, 100, 101, 200)
BOOTSTRAP_SEED = 77814


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    pending = path.with_suffix(".csv.tmp")
    pd.DataFrame(rows).to_csv(pending, index=False)
    pending.replace(path)


def _index_group(index: int) -> str:
    return "A" if index <= 100 else "B" if index <= 200 else "unclassified"


def _exact_reference_overlap(record, references: dict) -> str:
    for name, ref in references.items():
        n = len(ref.x)
        candidates = np.flatnonzero(record.x[:max(0, len(record.x) - n + 1)] == ref.x[0])
        for start in candidates:
            if (np.array_equal(record.x[start:start+n], ref.x)
                    and np.array_equal(record.rpm[start:start+n], ref.rpm)):
                return f"exact_{name}_overlap_offset_{start}"
    return ""


def _finite_common(frame: pd.DataFrame, feature: str) -> pd.DataFrame:
    pivot = frame.pivot(index=["record_id", "group", "eligible"], columns="model", values=feature)
    pivot = pivot.dropna().reset_index()
    return pivot.loc[pivot.eligible].drop(columns="eligible")


def _auc(a, b) -> float:
    ranks = rankdata(np.r_[a, b])[len(a):]
    return float((ranks.sum() - b.size * (b.size + 1) / 2) / (a.size * b.size))


def summarize(frame: pd.DataFrame, out: Path, *, primary=None, n_boot=1000) -> dict:
    """Recompute matched, conditional summaries from real per-record scores."""
    summary, contrasts = [], []
    outputs = {"summary_rows": 0, "paired_comparisons": 0, "bootstrap_replicates": n_boot}
    for metric in MODE_KEYS:
        if metric not in frame:
            continue
        common = _finite_common(frame, metric)
        aa = common.loc[common.group.eq("A")]
        bb = common.loc[common.group.eq("B")]
        if aa.empty or bb.empty:
            continue
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        a_ix = rng.integers(len(aa), size=(n_boot, len(aa)))
        b_ix = rng.integers(len(bb), size=(n_boot, len(bb)))
        gap_distribution = {}
        median_gap = {}
        for name in frame.model.drop_duplicates():
            a, b = aa[name].to_numpy(), bb[name].to_numpy()
            bands = np.median(b[b_ix], axis=1) - np.median(a[a_ix], axis=1)
            wins = (b[:, None] > a[None, :]) + .5 * (b[:, None] == a[None, :])
            auc = _auc(a, b)
            np.testing.assert_allclose(auc, wins.mean(), rtol=0, atol=1e-13)
            a_weight = np.array([np.bincount(ix, minlength=len(a)) for ix in a_ix]) / len(a)
            b_weight = np.array([np.bincount(ix, minlength=len(b)) for ix in b_ix]) / len(b)
            aucs = np.clip(np.sum((b_weight @ wins) * a_weight, axis=1), 0, 1)
            q5 = np.quantile(b, .05)
            q95 = np.quantile(a, .95)
            median_gap[name] = float(np.median(b) - np.median(a))
            gap_distribution[name] = bands
            lower, upper = np.quantile(bands, [.025, .975])
            auc_lower, auc_upper = np.quantile(aucs, [.025, .975])
            summary.append({
                "model": name, "metric": metric, "n_A": len(a), "n_B": len(b),
                "median_A": float(np.median(a)), "median_B": float(np.median(b)),
                "median_gap_B_minus_A": median_gap[name], "median_gap_ci_low": lower,
                "median_gap_ci_high": upper, "margin_min_B_minus_max_A": float(b.min() - a.max()),
                "central_margin_p05_B_minus_p95_A": float(q5 - q95),
                "conditional_auc_B_high": auc, "conditional_auc_ci_low": auc_lower,
                "conditional_auc_ci_high": auc_upper, "label_status": "provisional_index_group_only",
            })
        if primary in gap_distribution:
            for name in gap_distribution:
                if name == primary:
                    continue
                gap = median_gap[primary] - median_gap[name]
                dist = gap_distribution[primary] - gap_distribution[name]
                lo, hi = np.quantile(dist, [.025, .975])
                contrasts.append({
                    "primary": primary, "reference": name, "metric": metric, "n_A": len(aa),
                    "n_B": len(bb), "delta_median_gap": gap, "paired_ci_low": lo, "paired_ci_high": hi,
                    "interpretation": "paired acquisition IDs; lineage independence unverified",
                })
    if summary:
        _write_csv(out / "fault_group_summary.csv", summary)
    if contrasts:
        _write_csv(out / "paired_pmi_bootstrap.csv", contrasts)
    outputs.update(summary_rows=len(summary), paired_comparisons=len(contrasts))
    return outputs


def _source_status(data: Path, out: Path) -> dict:
    from ..data import find_input

    script = find_input(data, "fdt_featcal.m")
    contents = Path(script).read_text(encoding="utf-8", errors="replace") if script else ""
    code = "".join(contents.split())
    literal = "acf((round(fs/f_fault)-ran):(round(fs/f_fault)+ran))" in code
    fixture_samples = 1600
    t = np.arange(8 * fixture_samples) / 3200
    theta = t * 120 / 60
    x = (1 + .4 * np.cos(2 * np.pi * theta)) * np.cos(2 * np.pi * 37 * theta)
    fixture = measure(x, np.full(t.shape, 120.), t, fixture_samples)
    assert fixture["lecturer_compatible_lag"] == fixture_samples - 1
    np.testing.assert_allclose(fixture["lag_corrected"], 7, rtol=0, atol=1e-10)
    np.testing.assert_allclose(fixture["mathematically_specified"], 7, rtol=0, atol=1e-10)
    details = {
        "reference_script": Path(script).name if script else None,
        "reference_sha256": hashlib.sha256(Path(script).read_bytes()).hexdigest() if script else None,
        "literal_branch_confirmed_in_reference": literal,
        "exact_matlab_parity": "unverified",
        "exact_parity_reason": "original angular resampling helper and MATLAB numerical gold outputs not supplied",
        "course_mode": "lecturer_compatible: analytic envelope of mean-centered signal, zero-lag energy denominator, lag S−1",
        "lag_corrected": "lecturer envelope convention at exact lag S",
        "mathematically_specified": "direct Hilbert magnitude then center envelope at exact lag S",
        "analysis_convention": "offline; no padding, bandpass, prewhitening, or extra trimming beyond common warm-up",
        "known_fixture": {
            "rotations": 8, "shaft_rpm": 120, "sample_rate_hz": 3200, "points_per_revolution": fixture_samples,
            "exact_lag_analytical_pmi": 7, "observed_lecturer": fixture["lecturer_compatible"],
            "observed_corrected_lag": fixture["lag_corrected"],
            "observed_mathematical": fixture["mathematically_specified"],
        },
    }
    _write_json(out / "matlab_conformance.json", details)
    return details


def availability(data: Path, out: Path) -> dict:
    from ..data import find_input

    blind_fault = bool(find_input(data, "sigfault2") or
                       any(p.name.lower().startswith("sigfault2_part_") for p in data.iterdir()))
    blind_pred = bool(find_input(data, "sigtest"))
    result = {
        "sigtest_present": blind_pred,
        "sigfault2_present": blind_fault,
        "blind_execution": "requires explicit frozen blind command" if blind_fault or blind_pred
                           else "blocked: sigtest and sigfault2 missing",
        "verified_condition_labels": False,
        "verified_calibration": False,
        "classification_metrics": "unavailable; index-group AUC is conditional only",
        "pmi_standardization": "withheld without separate verified healthy calibration",
    }
    _write_json(out / "availability.json", result)
    return result


def analyze(data_dir: Path, out: Path, predictors: dict, *, start=START,
            primary=None, dataset="sigfault1", excluded_indices=None) -> dict:
    """Reset each frozen model for every candidate. Raw and residual masks match."""
    from ..data import find_input, iter_fault_records, load_record

    data, out = Path(data_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    _source_status(data, out)
    availability(data, out)
    refs = {}
    if dataset == "sigfault1":
        for name in ("sigtrain", "sigvalid"):
            path = find_input(data, name)
            if path:
                refs[name] = load_record(path)
    excluded_indices = excluded_indices or {}
    traces, rows, subjects = {}, [], []
    fingerprints = {}
    for record in iter_fault_records(data, dataset=dataset):
        index = int(record.index)
        source = str(Path(record.source).name) if hasattr(record, "source") else ""
        group = _index_group(index) if dataset == "sigfault1" else "unclassified"
        hash_x = hashlib.sha256(np.ascontiguousarray(record.x, dtype="<f8"))
        hash_x.update(np.ascontiguousarray(record.rpm, dtype="<f8"))
        first = fingerprints.setdefault(hash_x.digest(), index)
        exclusion = excluded_indices.get(index) or _exact_reference_overlap(record, refs)
        if first != index and not exclusion:
            exclusion = f"full_joint_duplicate_of_{first}"
        subject = {
            "record_id": index, "dataset": dataset, "group": group, "eligible": not exclusion,
            "exclusion_reason": exclusion, "label_source": "unverified_record_index",
            "source_file": source, "first_target": start, "target_count": len(record.x) - start,
            "min_rpm": float(record.rpm.min()), "max_rpm": float(record.rpm.max()),
        }
        subjects.append(subject)
        keep_trace = index in REPRESENTATIVE_IDS
        true = record.x[start:]
        if keep_trace:
            traces[f"{index}:index"] = np.arange(start, len(record.x))
            traces[f"{index}:rpm"] = record.rpm[start:]
            traces[f"{index}:time"] = record.t[start:]
        for name in ("raw", *predictors):
            row = {**subject, "model": name, "mse": np.nan, "prediction_seconds": 0.,
                   "diagnostic_seconds": np.nan, "pmi_standardized": np.nan,
                   "standardization_status": "no_verified_health_calibration", "status": "ok"}
            try:
                tic = perf_counter()
                estimate = None if name == "raw" else forecast(predictors[name], record, start=start)
                row["prediction_seconds"] = perf_counter() - tic if estimate is not None else 0.
                values = true if estimate is None else true - estimate
                if estimate is not None:
                    row["mse"] = float(np.mean(values**2))
                tic = perf_counter()
                features = measure(values, record.rpm[start:], record.t[start:], traces=keep_trace)
                row["diagnostic_seconds"] = perf_counter() - tic
                diagnostic_traces = features.pop("traces", {})
                if keep_trace:
                    for key, array in diagnostic_traces.items():
                        traces[f"{index}:{name}:{key}"] = array.astype(np.float32)
                    traces[f"{index}:{name}:residual"] = values.astype(np.float32)
                    if estimate is not None:
                        traces[f"{index}:{name}:prediction"] = estimate.astype(np.float32)
                row.update(features)
                row["status"] = features["lecturer_compatible_status"]
            except (ValueError, RuntimeError, FloatingPointError) as error:
                row.update(status=str(error), lecturer_compatible=np.nan)
            rows.append(row)
        if len(subjects) % 20 == 0:
            _write_csv(out / "fault_per_record.csv", rows)
            _write_csv(out / "fault_eligibility.csv", subjects)
            print(f"Fault candidates processed {len(subjects)}, model-record rows {len(rows)}", flush=True)

    if not subjects:
        raise FileNotFoundError(f"No {dataset} candidate records found under {data}")
    _write_csv(out / "fault_per_record.csv", rows)
    _write_csv(out / "fault_eligibility.csv", subjects)
    if traces:
        np.savez_compressed(out / "fault_traces.npz", **traces)
    frame = pd.DataFrame(rows)
    summary = summarize(frame, out, primary=primary)
    result = {
        "candidates": len(subjects), "models": ["raw", *predictors], "model_record_rows": len(rows),
        "common_first_target": start, "warm_up_seconds_at_3200_hz": start / 3200,
        "invalid_rows": int(frame.status.ne("ok").sum()),
        "eligible_candidates": int(sum(s["eligible"] for s in subjects)),
        "excluded_candidates": {str(s["record_id"]): s["exclusion_reason"]
                                for s in subjects if not s["eligible"]},
        "group_status": "conditional on unverified index grouping, no classification threshold",
        "record_84_status": "evaluated descriptively; previous retrospective inspection cannot establish untouched independence",
        "representative_record_ids_one_based": list(REPRESENTATIVE_IDS), **summary,
        "elapsed_seconds": perf_counter() - started,
    }
    _write_json(out / "fault_status.json", result)
    return result
