"""Portable inference registry. Array loads disallow pickle."""
from pathlib import Path
from shutil import copyfile

import numpy as np

from .baselines import BaselineRecord, conventional_forecasts, load_arrays

ARCHIVES = {
    "archive_dhon_optimized": "DHON_optimized",
    "archive_dhon_optimized_corrected": "DHON_optimized",
    "archive_ngrc_optimized_corrected": "NGRC_optimized",
}
COMPARATORS = ("ngrc", "physics_linear")


def model_specs(run, *, include_alternatives=True):
    checkpoints = run / "checkpoints"
    result = {
        "primary_dhon": ("dhon", checkpoints / "primary_dhon.npz"),
        "spa_lpv_arma": ("spa_lpv_arma", checkpoints / "conventional.npz"),
    }
    if include_alternatives:
        for key in ("alternative_dhon",):
            result[key] = ("dhon", checkpoints / f"{key}.npz")
        for key in COMPARATORS:
            result[key] = ("comparator", checkpoints / f"comparator_{key}.npz")
    return {name: spec for name, spec in result.items() if spec[1].is_file()}


def development_specs(run, root):
    result = model_specs(run, include_alternatives=False)
    result.update({key: ("comparator", run / "checkpoints" / f"comparator_{key}.npz")
                   for key in COMPARATORS if (run / "checkpoints" / f"comparator_{key}.npz").exists()})
    archive = root / "references" / "frozen_archive"
    for key, stem in ARCHIVES.items():
        path = archive / f"{stem}.npz"
        if path.exists() and (run / "metrics" / "archive_replay.csv").exists():
            result[key] = ("archive_corrected" if key.endswith("corrected") else "archive", path)
    return result


def load_predictor(kind: str, path: Path, device="cpu"):
    if kind == "dhon":
        from .dhon import load_model
        model = load_model(path, device)
        return lambda record: model.predict(record.x, record.rpm, record.fs)
    if kind == "comparator":
        from .comparators import load_comparator
        model = load_comparator(path, device)
        return lambda record: model.predict(record.x, record.rpm, record.fs)
    if kind == "spa_lpv_arma":
        arrays = load_arrays(path)

        def predict(record):
            # RPM normalization is frozen to the training extrema stored in the NPZ.
            low, high = arrays.get("rpm_range", (None, None))
            if low is None:
                normalized = record.speed_normalized
                if normalized is None:
                    raise ValueError("Missing frozen RPM normalization")
            else:
                normalized = (record.rpm-low)/(high-low)
            ref = BaselineRecord(record.x, record.rpm, normalized, record.t, record.fs)
            return conventional_forecasts(ref, arrays, selected=(kind,))[kind]
        return predict
    if kind in {"archive", "archive_corrected"}:
        from .archived import archived_forecast, load_archive_checkpoint
        meta, arrays = load_archive_checkpoint(path.parent, path.stem)

        def predict(record):
            normalized = record.speed_normalized
            if normalized is None:
                normalized = (record.rpm - record.rpm_min)/(record.rpm_max - record.rpm_min)
            ref = BaselineRecord(record.x, record.rpm, normalized, record.t, record.fs)
            return archived_forecast(ref, meta, arrays, corrected=kind == "archive_corrected")
        return predict
    raise ValueError(f"Unknown frozen model loader: {kind}")
