"""Hash-checked freeze and overlap evidence for inference-only evaluation."""
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .results import START, scores


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def create_manifest(run, data_dir, root):
    from .inference import model_specs
    source = sorted([*root.glob("src/gearbox_dhon/*.py"), *root.glob("configs/*.yaml")])
    specs = model_specs(run)
    if "primary_dhon" not in specs:
        raise ValueError("Cannot freeze without the primary DHON checkpoint")
    freeze = run / "frozen_manifest.json"
    manifest = {
        "schema": 1, "run_id": run.name, "scoring_start_zero_based": START, "speed_cutoff": "t-1",
        "reset_policy": "independent record; warm-up excluded uniformly",
        "diagnostic_policy": "development angular grid and both predeclared PMI lags; no fitted threshold",
        "calibration": None, "label_source": None,
        "development_inputs": {p.name: sha256(p) for p in sorted(data_dir.iterdir())
                               if p.is_file() and p.name.lower().startswith(("sigtrain", "sigvalid", "sigfault1"))},
        "code": {p.relative_to(root).as_posix(): sha256(p) for p in source},
        "run_evidence": {p.relative_to(run).as_posix(): sha256(p)
                         for group in ("search", "checkpoints", "metrics")
                         for p in sorted((run / group).glob("*")) if p.is_file() and p.suffix != ".log"},
        "models": {k: {"loader": kind, "checkpoint": path.relative_to(run).as_posix(),
                       "sha256": sha256(path)} for k, (kind, path) in specs.items()}
    }
    if freeze.exists():
        if json.loads(freeze.read_text(encoding="utf-8")) != manifest:
            raise FileExistsError("Frozen manifest already exists with different immutable content")
    else:
        write_json(freeze, manifest)
    return freeze


def verify_manifest(file, root, data):
    manifest = json.loads(file.read_text(encoding="utf-8"))
    if manifest["schema"] != 1 or manifest["scoring_start_zero_based"] != START:
        raise ValueError("Incompatible freeze protocol")
    if manifest["speed_cutoff"] != "t-1" or manifest["calibration"] is not None:
        raise ValueError("Unexpected frozen prediction or calibration policy")
    for path, digest in manifest["code"].items():
        if sha256(root / path) != digest:
            raise ValueError(f"Frozen code hash changed: {path}")
    for path, digest in manifest["development_inputs"].items():
        if sha256(data / path) != digest:
            raise ValueError(f"Development inputs unavailable or changed: {path}")
    for path, digest in manifest["run_evidence"].items():
        if sha256(file.parent / path) != digest:
            raise ValueError(f"Frozen run evidence changed: {path}")
    for model in manifest["models"].values():
        if sha256(file.parent / model["checkpoint"]) != model["sha256"]:
            raise ValueError(f"Frozen checkpoint changed: {model['checkpoint']}")
    return manifest


def exact_overlap(candidate, reference, window=256):
    """Search for exact x and RPM windows, including a shifted overlap covering ≥2 windows."""
    if not np.isclose(candidate.fs, reference.fs):
        return False
    first_x = np.ascontiguousarray(candidate.x, dtype="<f8")
    first_r = np.ascontiguousarray(candidate.rpm, dtype="<f8")
    other_x = np.ascontiguousarray(reference.x, dtype="<f8")
    other_r = np.ascontiguousarray(reference.rpm, dtype="<f8")
    if min(len(first_x), len(other_x)) < window:
        return False

    def scan(a_x, a_r, b_x, b_r):
        data = b_x.tobytes()
        offsets = sorted({*range(0, len(a_x)-window+1, window), len(a_x)-window})
        for pos in offsets:
            needle = a_x[pos:pos+window].tobytes()
            found = data.find(needle)
            while found != -1:
                if found % 8 == 0 and np.array_equal(a_r[pos:pos+window], b_r[found//8:found//8+window]):
                    return True
                found = data.find(needle, found+1)
        return False

    return scan(first_x, first_r, other_x, other_r) or scan(other_x, other_r, first_x, first_r)


def run_blind(data, manifest_path, output, root, device="cpu"):
    """Evaluate new records only after a freeze. No fit, threshold selection, or inferred labels."""
    from .data import find_input, iter_fault_records, load_record
    from .faults import evaluate_faults
    from .inference import load_predictor
    manifest = verify_manifest(manifest_path, root, data)
    if output.exists():
        raise FileExistsError(f"Choose a new blind output directory: {output}")
    status = {}
    healthy = find_input(data, "sigtest")
    fault_available = (data / "sigfault2.mat").exists() or bool(list(data.glob("sigfault2_part_*.mat")))
    if healthy is None and not fault_available:
        status = {"sigtest": "BLOCKED: absent", "sigfault2": "BLOCKED: absent"}
        write_json(output / "data_availability.json", status)
        return status

    predictors = {k: load_predictor(kind=v["loader"], path=manifest_path.parent / v["checkpoint"], device=device)
                  for k, v in manifest["models"].items()}
    development = [load_record(find_input(data, f"sig{name}")) for name in ("train", "valid")]
    if any(p.startswith("sigfault1") for p in manifest["development_inputs"]):
        development.extend(iter_fault_records(data, dataset="sigfault1"))
    if healthy is None:
        status["sigtest"] = "BLOCKED: absent"
    else:
        record = load_record(healthy)
        if any(exact_overlap(record, ref) for ref in development):
            status["sigtest"] = "EXCLUDED: exact overlapping vibration and RPM"
        else:
            rows, predictions = [], {}
            for name, predict in predictors.items():
                tic = perf_counter()
                values = predict(record)
                rows.append(dict(model=name, elapsed_s=perf_counter()-tic,
                                 **scores(record.x[START:], values[START:])))
                predictions[name] = values
            output.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(output / "metrics_blind.csv", index=False)
            np.savez_compressed(output / "predictions_blind.npz", **predictions)
            status["sigtest"] = "COMPUTED: numerical overlap screened; acquisition identity unconfirmed"
    if not fault_available:
        status["sigfault2"] = "BLOCKED: absent"
    else:
        excluded = {}
        for record in iter_fault_records(data, dataset="sigfault2"):
            if any(exact_overlap(record, ref) for ref in development):
                excluded[record.index] = "exact_development_vibration_and_rpm_overlap"
        evaluate_faults(data, output, predictors, start=START, dataset="sigfault2", excluded_indices=excluded)
        status["sigfault2"] = f"COMPUTED: unlabelled diagnostics, {len(excluded)} exact overlaps excluded"
    write_json(output / "data_availability.json", status)
    return status
