"""Reconcile standalone comparator refits to the protected common-mask release."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from src_gearbox.models.baselines.protocol import FAMILIES, FOLDS, load_data, mse, predict, write_table
from src_gearbox.models.baselines.replay import CAUSAL_TOLERANCE, load_predictor
from src_gearbox.study_protocol import MASK_HASH, PROJECT, TARGETS, atomic_json, fingerprint


METRIC_TOLERANCE = 1e-12
REPLAY_TOLERANCE = 1e-7


def read_unique_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    indexed = {r["model"]: r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate model identifiers in {path}")
    return indexed


def validate_axes(release, record):
    axes = {"target_indices": TARGETS, "actual": record.x[TARGETS],
            "rpm": record.rpm[TARGETS], "time_s": record.t[TARGETS]}
    for name, expected in axes.items():
        np.testing.assert_array_equal(release[name], expected, err_msg=f"Protected {name} axis differs")


def saved_mse(value, expected, what):
    if abs(value - float(expected)) > METRIC_TOLERANCE:
        raise ValueError(f"{what} MSE differs from its metric row: {value:.17g} vs {expected}")


def audit_folds(slug, row, trial_rows):
    selected = [item for item in trial_rows if item["model"] == slug and item["selected"] == "True"]
    selected.sort(key=lambda item: int(item["fold"]))
    if slug.startswith("sparse_lpv_"):
        if selected:
            raise ValueError(f"Unexpected new model selection for frozen support refit {slug}")
        return 0
    if [int(item["fold"]) for item in selected] != list(range(1, len(FOLDS) + 1)):
        raise ValueError(f"Expected every selected guarded fold for {slug}")
    for item, (start, stop) in zip(selected, FOLDS, strict=True):
        if item.get("start") and (int(item["start"]), int(item["stop"])) != (start, stop):
            raise ValueError(f"Selected guarded fold timing differs for {slug}")
    saved_mse(float(np.mean([float(item["mse"]) for item in selected])), row["fold_mean_mse"],
              f"{slug} selected guarded folds")
    return len(selected)


def audited_model(slug, family, checkpoint_dir, stage, train, dev, data, main, row, main_row, frozen, trials):
    path = checkpoint_dir / f"{slug}.npz"
    digest = fingerprint(path)
    if digest != stage["models"][slug]:
        raise ValueError(f"Fresh checkpoint digest differs from completed-stage marker: {slug}")
    if any(data[name].shape != dev.x.shape for name in (slug, f"{slug}__train")):
        raise ValueError(f"Expected full acquisition-length fit predictions: {slug}")
    observed = data[slug][TARGETS]
    observed_train = data[f"{slug}__train"][TARGETS]
    reference = main[slug]
    loss = mse(dev.x[TARGETS], observed)
    loss_train = mse(train.x[TARGETS], observed_train)
    reference_loss = mse(dev.x[TARGETS], reference)
    for value, expected, scope in ((loss, row["validation_mse"], "fresh development"),
                                   (loss_train, row["train_mse"], "fresh training"),
                                   (reference_loss, main_row["mse"], "protected development")):
        saved_mse(value, expected, f"{slug} {scope}")
    if main_row["mask_sha256"] != MASK_HASH:
        raise ValueError(f"Protected metric mask identity differs: {slug}")
    model = load_predictor(slug, checkpoint_dir)
    replay_dev = float(np.max(np.abs(predict(model, dev)[TARGETS] - observed)))
    replay_train = float(np.max(np.abs(predict(model, train)[TARGETS] - observed_train)))
    if max(replay_dev, replay_train) > REPLAY_TOLERANCE:
        raise ValueError(f"Fresh checkpoint does not reproduce stored {slug} forecasts")
    delta = float(np.max(np.abs(observed - reference)))
    support_refit = slug.startswith("sparse_lpv_")
    if support_refit and row.get("fit_scope") != "full_training_support_refit":
        raise ValueError(f"Expected prior-support coefficient refit for {slug}")
    current_selection = json.loads(row["selection"])
    previous_selection = frozen["selection"]
    same_config = current_selection.get("config", current_selection) == previous_selection.get("config", previous_selection)
    return dict(model=slug, family=family, status="REPRODUCED" if delta <= CAUSAL_TOLERANCE else "DIFFERENT_REFIT",
                fresh_fit_scope="prior_guarded_support_new_coefficients" if support_refit else "guarded_search_full_refit",
                scored_targets=len(TARGETS), mask_sha256=MASK_HASH, checkpoint_sha256=digest,
                checkpoint_bytes_identical=digest == frozen["prior_checkpoint_sha256"],
                selected_configuration_identical=same_config,
                fresh_fit_mse=loss_train, fresh_development_mse=loss, protected_development_mse=reference_loss,
                mse_delta=loss-reference_loss, fresh_vs_protected_prediction_max_abs=delta,
                fresh_reload_development_max_abs=replay_dev, fresh_reload_training_max_abs=replay_train,
                selected_guarded_folds_verified=audit_folds(slug, row, trials),
                fresh_guarded_fold_mse=row.get("fold_mean_mse", ""),
                protected_guarded_fold_mse=main_row.get("fold_mse_prior", ""),
                fresh_selection=row.get("selection", ""))


def audit_sparse_support(root, out):
    evidence = json.loads((out / "statistical_sparse_provenance.json").read_text(encoding="utf-8"))
    prior = root / "references/baselines/conventional.npz"
    if fingerprint(prior) != evidence["prior_checkpoint_sha256"]:
        raise ValueError("Sparse prior-support hash discrepancy")
    with np.load(prior, allow_pickle=False) as frozen, \
         np.load(out / "models/sparse_lpv_ar.npz", allow_pickle=False) as fresh_ar, \
         np.load(out / "models/sparse_lpv_arma.npz", allow_pickle=False) as fresh_arma:
        for fresh in (fresh_ar, fresh_arma):
            np.testing.assert_array_equal(fresh["rpm_range"], frozen["rpm_range"])
        np.testing.assert_array_equal(fresh_ar["stage1"], fresh_arma["stage1"])
        frozen_coefficients = (frozen["lpv_ar"], frozen["lpv_arma"])
        fresh_coefficients = (fresh_ar["stage1"], fresh_arma["stage2"])
        sizes, differences = [], []
        for before, after in zip(frozen_coefficients, fresh_coefficients, strict=True):
            np.testing.assert_array_equal(before != 0, after != 0)
            sizes.append(int(np.count_nonzero(before)))
            differences.append(float(np.max(np.abs(before - after))))
    if sizes != [stage["support"] for stage in evidence["stages"]]:
        raise ValueError("Sparse prior-support counts differ from fit provenance")
    np.testing.assert_array_equal(differences, evidence["refitted_coefficient_max_abs"])
    return dict(frozen_prior_sha256=evidence["prior_checkpoint_sha256"],
                supports_identical=True, rpm_calibration_identical=True,
                nonzero_support_sizes=sizes, refitted_coefficient_max_abs=differences,
                note=evidence["note"])


def run(root, out, data_dir, allow_partial):
    selection = root / "configs/baselines/selection.json"
    release = root / "results/predictions/baselines_sigvalid.npz"
    frozen_metrics = read_unique_rows(root / "results/metrics/baseline_comparison.csv")
    train, dev = load_data(data_dir)
    inputs = {"training_sha256": fingerprint(data_dir / train.source),
              "development_sha256": fingerprint(data_dir / dev.source),
              "frozen_selection_sha256": fingerprint(selection)}
    registered = json.loads(selection.read_text(encoding="utf-8"))
    if inputs["training_sha256"] != registered["train"]["sha256"] or inputs["development_sha256"] != registered["validation"]["sha256"]:
        raise ValueError("Refit audit expects the exact protected course input recordings")
    rows, missing = [], []
    with np.load(release, allow_pickle=False) as main, threadpool_limits(1):
        validate_axes(main, dev)
        for family, slugs in FAMILIES.items():
            marker = out / f"{family}_stage_complete.json"
            if not marker.is_file():
                missing.append(family)
                continue
            stage = json.loads(marker.read_text(encoding="utf-8"))
            if stage["stage"] != family or stage["status"] != "DONE" or stage["identity"] != inputs:
                raise ValueError(f"Refit stage marker does not identify the protected inputs: {family}")
            metrics = read_unique_rows(out / f"{family}_metrics.csv")
            if set(metrics) != set(slugs) or set(stage["models"]) != set(slugs):
                raise ValueError(f"Unexpected or missing refit model identifiers for {family}")
            with (out / f"{family}_trials.csv").open(newline="", encoding="utf-8") as stream:
                trial_rows = list(csv.DictReader(stream))
            with np.load(out / f"{family}_predictions.npz", allow_pickle=False) as data:
                for slug in slugs:
                    rows.append(audited_model(slug, family, out / "models", stage, train, dev, data, main,
                                              metrics[slug], frozen_metrics[slug], registered["models"][slug], trial_rows))
    if missing and not allow_partial:
        raise ValueError(f"Incomplete refit stages: {', '.join(missing)}. Use --allow-partial to audit complete stages.")
    if not rows:
        raise ValueError("No completed refit stages to audit")
    sparse = audit_sparse_support(root, out) if "statistical" not in missing else None
    write_table(out / "fresh_refit_reconciliation.csv", rows)
    meta = dict(status="PARTIAL" if missing else "COMPLETE", validated_models=len(rows),
                expected_models=sum(map(len, FAMILIES.values())), missing_stages=missing,
                selected_guarded_fold_rows_verified=sum(r["selected_guarded_folds_verified"] for r in rows),
                different_refits=[r["model"] for r in rows if r["status"] == "DIFFERENT_REFIT"],
                different_checkpoint_bytes=[r["model"] for r in rows if not r["checkpoint_bytes_identical"]],
                different_selected_configurations=[r["model"] for r in rows if not r["selected_configuration_identical"]],
                max_abs_mse_change=max(abs(r["mse_delta"]) for r in rows),
                max_forecast_difference=max(r["fresh_vs_protected_prediction_max_abs"] for r in rows),
                fresh_checkpoint_reload_max_abs=max(max(r["fresh_reload_training_max_abs"],
                                                        r["fresh_reload_development_max_abs"]) for r in rows),
                axis_mask_sha256=MASK_HASH, protected_forecasts_sha256=fingerprint(release), inputs=inputs,
                selected_on_development=False, sparse_support_refit=sparse)
    atomic_json(out / "fresh_refit_reconciliation.json", meta)
    print(json.dumps(meta, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    run(args.root, args.out or args.root / "results/experiments/baseline_refit",
        args.data_dir or args.root / "data/raw", args.allow_partial)


if __name__ == "__main__":
    main()
