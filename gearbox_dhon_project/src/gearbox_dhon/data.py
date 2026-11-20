"""Load course MAT records without refitting supplied transformations."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Iterator

import numpy as np
from scipy.io import loadmat
from scipy.signal import fftconvolve


@dataclass(frozen=True)
class Record:
    """An acquisition record with MATLAB-order ID and sample-aligned vectors."""

    x: np.ndarray
    rpm: np.ndarray
    t: np.ndarray
    fs: float
    dataset: str
    index: int
    source: str
    part_index: int
    fields: tuple[str, ...]
    speed_normalized: np.ndarray | None = None
    xraw: np.ndarray | None = None
    mean: float | None = None
    scale: float | None = None
    rpm_min: float | None = None
    rpm_max: float | None = None

    @property
    def record_id(self) -> str:
        return f"{self.dataset}:{self.index:03d}"


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _logical_stem(path: Path) -> str:
    return re.sub(r"\s*\(\d+\)$", "", path.stem.lower()).strip()


def _verified_file(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    if len(paths) > 1 and len({sha256(path) for path in paths}) != 1:
        raise ValueError(f"Conflicting files for {_logical_stem(paths[0])}: {[p.name for p in paths]}")
    return sorted(paths, key=lambda path: (len(path.name), path.name.casefold()))[0]


def find_input(directory: Path, name: str) -> Path | None:
    """Resolve case and duplicate-download suffixes within the specified raw directory."""
    directory = Path(directory)
    expected = Path(name) if Path(name).suffix else Path(f"{name}.mat")
    return _verified_file([
        path for path in directory.glob("*") if path.is_file()
        and path.suffix.lower() == expected.suffix.lower()
        and _logical_stem(path) == _logical_stem(expected)
    ])


def fault_sources(directory: Path, dataset: str = "sigfault1") -> list[Path]:
    directory = Path(directory)
    whole = find_input(directory, f"{dataset}.mat")
    part_paths: dict[int, list[Path]] = {}
    for path in directory.glob("*"):
        match = re.fullmatch(rf"{re.escape(dataset.lower())}_part_(\d+)", _logical_stem(path))
        if path.is_file() and path.suffix.lower() == ".mat" and match:
            part_paths.setdefault(int(match[1]), []).append(path)
    if whole and part_paths:
        raise ValueError(f"Both whole file and numbered parts present for {dataset}; specify one source")
    if not part_paths:
        return [whole] if whole else []
    if sorted(part_paths) != list(range(1, max(part_paths) + 1)):
        raise ValueError(f"Non-contiguous {dataset} parts: {sorted(part_paths)}")
    paths = [_verified_file(part_paths[number]) for number in sorted(part_paths)]
    if len({sha256(path) for path in paths}) != len(paths):
        raise ValueError(f"Identical numbered {dataset} part files")
    return paths


def _hdf5_structs(path: Path, variable: str):
    """Read standard MATLAB v7.3 struct fields stored as reference arrays."""
    import h5py

    with h5py.File(path, "r") as handle:
        struct = handle[variable]
        if not isinstance(struct, h5py.Group):
            raise ValueError(f"{variable} is not a MATLAB v7.3 struct in {path.name}")
        fields = {
            name: value for name, value in struct.items() if isinstance(value, h5py.Dataset)
        }
        refs = {name for name, value in fields.items() if h5py.check_dtype(ref=value.dtype)}
        count = fields[next(iter(refs))].size if refs else 1
        if any(fields[name].size != count for name in refs):
            raise ValueError(f"Inconsistent MATLAB v7.3 struct fields in {path.name}")
        if count > 1 and len(refs) != len(fields):
            raise ValueError(f"Ambiguous direct field in MATLAB v7.3 struct array in {path.name}")
        records = []
        for index in range(count):
            record = {}
            for name, value in fields.items():
                record[name] = (
                    np.asarray(handle[np.asarray(value).reshape(-1)[index]]).T
                    if name in refs else np.asarray(value).T
                )
            records.append(record)
        return records


def _mat_structs(path: Path, variable: str):
    try:
        data = loadmat(path, variable_names=[variable], struct_as_record=False, squeeze_me=True)
    except NotImplementedError:
        return _hdf5_structs(path, variable)
    if variable not in data:
        raise ValueError(f"Missing MAT struct {variable} in {path.name}")
    elements = np.asarray(data[variable]).ravel(order="F")
    return [{name: getattr(value, name) for name in value._fieldnames} for value in elements]


def _vector(values: dict, key: str, *, optional: bool = False) -> np.ndarray | None:
    if optional and key not in values:
        return None
    array = np.asarray(values[key], dtype=np.float64).squeeze()
    if array.ndim != 1:
        raise ValueError(f"{key} must be a MATLAB vector")
    return array


def _scalar(values: dict, key: str, *, optional: bool = False, first_channel: bool = False):
    if optional and key not in values:
        return None
    array = np.asarray(values[key], dtype=np.float64).reshape(-1)
    if not first_channel and array.size != 1:
        raise ValueError(f"{key} must be a MATLAB scalar")
    return float(array[0])


def _as_record(values: dict, path: Path, dataset: str, index: int, part_index: int) -> Record:
    x, rpm, t = (_vector(values, name) for name in ("x", "spd", "t"))
    speed_normalized = _vector(values, "spdnorm", optional=True)
    xraw = _vector(values, "xraw", optional=True)
    fs = _scalar(values, "fs")
    n = _scalar(values, "N", optional=True)
    if (len(x) < 2 or len(x) != len(rpm) or len(x) != len(t)
            or (n is not None and len(x) != n)):
        raise ValueError(f"Signal/metadata length mismatch in {path.name}, record {index}")
    for array in (x, rpm, t, speed_normalized, xraw):
        if array is not None and (len(array) != len(x) or not np.isfinite(array).all()):
            raise ValueError(f"Non-finite or malformed signal in {path.name}, record {index}")
    if (not np.isfinite(fs) or fs <= 0 or np.any(rpm <= 0)
            or not np.allclose(np.diff(t), 1 / fs, rtol=1e-5, atol=1e-10)):
        raise ValueError(f"Invalid speed or time grid in {path.name}, record {index}")
    return Record(
        x, rpm, t, fs, dataset, index, path.name, part_index, tuple(sorted(values)),
        speed_normalized, xraw, _scalar(values, "xmean", optional=True, first_channel=True),
        _scalar(values, "xstd", optional=True, first_channel=True),
        _scalar(values, "spdmin", optional=True), _scalar(values, "spdmax", optional=True),
    )


def load_record(path: Path) -> Record:
    """Load a single MATLAB record named after the logical filename."""
    path = Path(path)
    dataset = _logical_stem(path)
    records = _mat_structs(path, dataset)
    if len(records) != 1:
        raise ValueError(f"{path.name} contains {len(records)} records; use iter_fault_records")
    return _as_record(records[0], path, dataset, 1, 1)


def iter_fault_records(directory: Path, dataset: str = "sigfault1") -> Iterator[Record]:
    """Stream one numbered MAT part at a time, preserving MATLAB linear order."""
    paths = fault_sources(directory, dataset)
    index = 1
    part_size = None
    for path in paths:
        block = _mat_structs(path, dataset)
        if part_size is None:
            part_size = len(block)
        if not block or (len(paths) > 1 and len(block) != part_size):
            raise ValueError(f"Inconsistent struct counts across {dataset} parts")
        for part_index, values in enumerate(block, start=1):
            yield _as_record(values, path, dataset, index, part_index)
            index += 1


def record_stats(record: Record, train: Record | None) -> dict:
    v = record.x
    physical_error = None
    normalized_error = None
    if record.xraw is not None and record.scale is not None and record.mean is not None:
        physical_error = float(np.max(np.abs(record.xraw - (record.mean + record.scale * v))))
    if (record.speed_normalized is not None and train is not None
            and train.rpm_min is not None and train.rpm_max > train.rpm_min):
        normalized_error = float(np.max(np.abs(
            record.speed_normalized - (record.rpm - train.rpm_min) / (train.rpm_max - train.rpm_min)
        )))
    return {
        "record_id": record.record_id,
        "source_filename": record.source,
        "part_struct_index_1based": record.part_index,
        "fields": " ".join(record.fields),
        "N": len(v), "fs_hz": record.fs, "duration_s": len(v) / record.fs,
        "t_first_s": record.t[0], "t_last_s": record.t[-1],
        "max_grid_error_s": np.max(np.abs(np.diff(record.t) - 1 / record.fs)),
        "finite": True,
        "rpm_min": np.min(record.rpm), "rpm_max": np.max(record.rpm),
        "stored_rpm_min": record.rpm_min, "stored_rpm_max": record.rpm_max,
        "rpm_unit": "rpm", "vibration_unit": "dimensionless archived x",
        "observed_rotations": np.trapezoid(record.rpm, record.t) / 60,
        "x_mean": np.mean(v), "x_std_population": np.std(v),
        "x_min": np.min(v), "x_max": np.max(v),
        "stored_x_mean_channel1": record.mean, "stored_x_std_channel1": record.scale,
        "raw_reconstruction_max_abs_error": physical_error,
        "speed_norm_max_error_vs_train_bounds": normalized_error,
        "sha256_x_le_f64": hashlib.sha256(v.astype("<f8", copy=False).tobytes()).hexdigest(),
        "sha256_rpm_le_f64": hashlib.sha256(record.rpm.astype("<f8", copy=False).tobytes()).hexdigest(),
    }


def _scan_reference(record: Record, reference: Record) -> dict:
    """Scan all full-length offsets with normalized correlation and direct error checks."""
    n = len(reference.x)
    if len(record.x) < n:
        raise ValueError("Reference scan requires the candidate to be at least as long as reference")
    reference_centered = reference.x - reference.x.mean()
    num = fftconvolve(record.x, reference_centered[::-1], mode="valid")
    sums = np.r_[0.0, np.cumsum(record.x)]
    squares = np.r_[0.0, np.cumsum(record.x ** 2)]
    var = np.maximum(squares[n:] - squares[:-n] - (sums[n:] - sums[:-n]) ** 2 / n, 0)
    den = np.sqrt(np.sum(reference_centered ** 2) * var)
    corr = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    start = int(np.argmax(corr))
    # Directly check any exact vibration-anchor match; a correlation tie can select a different offset.
    anchors = np.flatnonzero(record.x[: len(corr)] == reference.x[0])
    exact = [int(i) for i in anchors if (
        np.array_equal(record.x[i : i + n], reference.x)
        and np.array_equal(record.rpm[i : i + n], reference.rpm)
    )]
    if exact:
        start = exact[0]
    x = record.x[start : start + n]
    rpm = record.rpm[start : start + n]
    dx = float(np.max(np.abs(x - reference.x)))
    drpm = float(np.max(np.abs(rpm - reference.rpm)))
    return {
        "record_id": record.record_id, "reference": reference.dataset,
        "windows_scanned": len(corr), "best_start_zero_based": start,
        "best_end_exclusive": start + n,
        "x_correlation": float(np.clip(corr[start], -1, 1)),
        "max_x_abs_difference": dx, "max_rpm_abs_difference": drpm,
        "near_by_eps_1e_8": bool(dx <= 1e-8 and drpm <= 1e-8),
        "exact_full_length": bool(exact), "exact_offset_count": len(exact),
    }


def _partial_overlap_scan(refs: list[Record], data_dir: Path) -> list[dict]:
    """Find exact shifted overlaps by matching 64 consecutive x/RPM pairs.

    One side of each comparison is subsampled at 512-sample anchor intervals.
    Any exact overlap of at least 575 samples necessarily contains an anchor
    with 64 subsequent coincident pairs. Record inferred relative offsets.
    """
    width, spacing = 64, 512
    samples = []
    for rec in chain(refs, iter_fault_records(data_dir)):
        for start in range(0, len(rec.x) - width + 1, spacing):
            samples.append((
                rec.record_id, start, rec.x[start:start + width].copy(), rec.rpm[start:start + width].copy(),
            ))
    keys = np.array([section[2][0] for section in samples])
    order = np.argsort(keys)
    ordered_keys = keys[order]
    offsets: dict[tuple[str, str, int], set[tuple[str, int]]] = {}
    for rec in chain(refs, iter_fault_records(data_dir)):
        candidates = rec.x[:len(rec.x) - width + 1]
        low = np.searchsorted(ordered_keys, candidates, side="left")
        high = np.searchsorted(ordered_keys, candidates, side="right")
        for start in np.flatnonzero(low != high):
            for anchor_position in order[low[start]:high[start]]:
                other_id, other_start, other_x, other_rpm = samples[anchor_position]
                if other_id == rec.record_id:
                    continue
                stop = start + width
                if (not np.array_equal(other_x, rec.x[start:stop])
                        or not np.array_equal(other_rpm, rec.rpm[start:stop])):
                    continue
                if other_id < rec.record_id:
                    key = (other_id, rec.record_id, int(start - other_start))
                else:
                    key = (rec.record_id, other_id, int(other_start - start))
                offsets.setdefault(key, set()).add((other_id, other_start))
    return [{
        "record_id_first": a, "record_id_second": b, "second_minus_first_offset_samples": lag,
        "matched_markers": len(markers), "marker_length_samples": width,
    } for (a, b, lag), markers in sorted(offsets.items())]


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _file_inventory(directories: list[tuple[str, Path]]) -> list[dict]:
    inventory = []
    for role, directory in directories:
        for path in sorted(directory.glob("*")):
            if path.is_file():
                # Blind input contents are intentionally deferred until explicit frozen evaluation.
                deferred = any(_logical_stem(path).startswith(name) for name in ("sigtest", "sigfault2"))
                inventory.append({
                    "role": role, "filename": path.name, "bytes": path.stat().st_size,
                    "sha256": "" if deferred else sha256(path),
                    "status": "DEFERRED_UNTIL_FREEZE" if deferred else "READABLE",
                })
    return inventory


def _environment() -> dict:
    installed = {}
    for package in ("numpy", "scipy", "pandas", "matplotlib", "torch", "scikit-learn",
                    "python-pptx", "nbclient", "h5py"):
        try:
            installed[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            installed[package] = None
    git = subprocess.run(["git", "status", "--short", "--branch"], capture_output=True, text=True) if shutil.which("git") else None
    gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True) if shutil.which("nvidia-smi") else None
    memory = Path("/proc/meminfo")
    memory = memory.read_text().splitlines()[0] if memory.is_file() else None
    return {
        "platform": platform.platform(), "python": platform.python_version(),
        "packages": installed, "logical_cpu_count": os.cpu_count(),
        "process_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "host_memory": memory, "free_bytes": shutil.disk_usage(Path.cwd()).free,
        "git_status": git.stdout.strip() if git and git.returncode == 0 else None,
        "gpu": gpu.stdout.strip() if gpu and gpu.returncode == 0 else None,
    }


def run_audit(data_dir: Path, out: Path, source_dir: Path | None = None) -> dict:
    """Save provenance, development duplicate checks and an explicit blind-data gate."""
    data_dir, out = Path(data_dir), Path(out)
    train_path, valid_path = (find_input(data_dir, f"{stem}.mat") for stem in ("sigtrain", "sigvalid"))
    train = load_record(train_path) if train_path else None
    valid = load_record(valid_path) if valid_path else None
    refs = [record for record in (train, valid) if record is not None]
    inventory, splits, duplicates = [], [], []
    full_fingerprints: dict[tuple[str, str], str] = {}
    fault_count = 0
    for record in (*refs,):
        inventory.append(record_stats(record, train))
        splits.append({
            "record_id": record.record_id, "group_inferred": record.record_id,
            "offset_zero_based": 0, "end_exclusive": len(record.x), "provisional_group": "",
            "eligible_development_diagnostics": False,
            "reason": "official fit" if record.dataset == "sigtrain" else "official development validation",
        })
    if train is not None and valid is not None:
        duplicates.append(_scan_reference(valid, train))
    exact_overlap_ids = []
    for record in iter_fault_records(data_dir):
        fault_count += 1
        stats = record_stats(record, train)
        inventory.append(stats)
        scans = [_scan_reference(record, ref) for ref in refs if len(record.x) >= len(ref.x)]
        duplicates.extend(scans)
        near_overlap = next((scan for scan in scans if scan["near_by_eps_1e_8"]), None)
        full_key = (stats["sha256_x_le_f64"], stats["sha256_rpm_le_f64"])
        same_record = full_fingerprints.get(full_key)
        if same_record is None:
            full_fingerprints[full_key] = record.record_id
        if near_overlap:
            exact_overlap_ids.append(record.index)
        splits.append({
            "record_id": record.record_id,
            "group_inferred": (
                f"{near_overlap['reference']}:001" if near_overlap else same_record or record.record_id
            ),
            "offset_zero_based": 0, "end_exclusive": len(record.x),
            "provisional_group": "A" if record.index <= 100 else "B",
            "eligible_development_diagnostics": near_overlap is None and same_record is None,
            "reason": (
                "overlapping official acquisition" if near_overlap else
                "duplicate full-record fingerprint" if same_record else
                "index-based exploratory group; labels and acquisition metadata absent"
            ),
        })
    shifted = _partial_overlap_scan(refs, data_dir)
    parents = {row["record_id"]: row["record_id"] for row in splits}

    def root(record_id):
        while record_id != parents[record_id]:
            record_id = parents[record_id]
        return record_id

    for match in shifted:
        a = root(match["record_id_first"])
        b = root(match["record_id_second"])
        parents[max(a, b)] = min(a, b)
    groups: dict[str, list[str]] = {}
    for record_id in parents:
        groups.setdefault(root(record_id), []).append(record_id)
    for row in splits:
        record_id = row["record_id"]
        members = groups[root(record_id)]
        if len(members) <= 1 or not record_id.startswith("sigfault1:"):
            continue
        # Numbered course candidates group with the official comparator whenever linked.
        official = next((value for value in members if not value.startswith("sigfault1:")), None)
        row["group_inferred"] = official or min(members)
        row["eligible_development_diagnostics"] = False
        row["reason"] = (
            "overlapping official acquisition" if official
            else "exact shared vibration and speed marker with another fault record"
        )
        index = int(record_id.split(":")[-1])
        if official and index not in exact_overlap_ids:
            exact_overlap_ids.append(index)

    blind_pred = any(_logical_stem(path) == "sigtest" and path.is_file()
                     for path in data_dir.glob("*") if path.suffix.lower() == ".mat")
    blind_fault = any(_logical_stem(path).startswith("sigfault2") and path.is_file()
                      for path in data_dir.glob("*") if path.suffix.lower() == ".mat")
    status = {
        "sigtrain": train_path is not None, "sigvalid": valid_path is not None,
        "sigfault1_count": fault_count, "sigfault1_excluded_overlap_indices_1based": exact_overlap_ids,
        "sigtest": blind_pred, "sigfault2": blind_fault,
        "fdt_featcal": find_input(data_dir, "fdt_featcal.m") is not None,
        "confirmed_labels": False, "acquisition_start_metadata": False,
        "label_status": "A/B are index groups only; no authoritative health labels supplied",
        "blind_status": "DEFERRED_UNTIL_FREEZE" if blind_pred or blind_fault else "BLOCKED_MISSING_INPUTS",
        "duplicate_search_scope": "all complete contiguous training/validation-length offsets in each development fault record, and full-record exact fault fingerprints",
        "partial_window_overlap": "all-pairs exact shared 64-sample x/RPM markers on a 512-sample anchor grid; additional arbitrary approximate partial overlap not assessed",
    }
    _write_csv(out / "tables" / "record_inventory.csv", inventory)
    _write_csv(out / "tables" / "split_manifest.csv", splits)
    _write_csv(out / "tables" / "duplicate_scan.csv", duplicates)
    _write_csv(out / "tables" / "exact_shifted_overlap.csv", shifted)
    sources = [("raw_input", data_dir)]
    if source_dir is not None:
        sources.append(("reference", Path(source_dir)))
    _write_csv(out / "tables" / "source_inventory.csv", _file_inventory(sources))
    (out / "data_availability.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    (out / "environment_intake.json").write_text(json.dumps(_environment(), indent=2) + "\n", encoding="utf-8")
    return status
