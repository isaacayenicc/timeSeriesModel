"""Shared target alignment, provenance and training-fold rules."""
from hashlib import sha256
from pathlib import Path
import json
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.io import loadmat
import yaml


PROJECT = Path(__file__).resolve().parents[2]
START = 1792
STOP = 24000
TARGETS = np.arange(START, STOP, dtype="<i8")
SEED = 19
MASK_HASH = sha256(TARGETS.tobytes()).hexdigest()


def settings(name):
    return yaml.safe_load((PROJECT / "configs" / f"{name}.yaml").read_text(encoding="utf-8"))


def fingerprint(path):
    with path.open("rb") as source:
        return sha256_stream(source)


def sha256_stream(stream):
    digest = sha256()
    for chunk in iter(lambda: stream.read(1 << 20), b""):
        digest.update(chunk)
    return digest.hexdigest()


def resolve_record(directory, name):
    directory = Path(directory)
    candidates = [directory / f"{name}.mat", *sorted(directory.glob(f"{name} (*).mat"))]
    files = [f for f in candidates if f.is_file()]
    if not files:
        raise FileNotFoundError(f"Place {name}.mat in {directory}")
    digests = {fingerprint(f) for f in files}
    if len(digests) != 1:
        raise ValueError(f"Ambiguous copies of {name}: {files}")
    return files[0]


def read_vector_record(path):
    """Read a single MATLAB course struct; fall back to the main loader for HDF5."""
    try:
        values = loadmat(path, simplify_cells=True)
    except NotImplementedError:
        from .data import load_record
        item = load_record(path)
        return item.x, item.rpm, item.t, item.fs
    structures = [v for k, v in values.items() if not k.startswith("__") and isinstance(v, dict)]
    if len(structures) != 1:
        raise ValueError(f"Expected one MATLAB recording structure in {path}")
    source = structures[0]
    x, rpm, t = [np.asarray(source[k], dtype=np.float64).reshape(-1) for k in ["x", "spd", "t"]]
    fs = float(source["fs"])
    stagger = np.diff(t)
    if x.size != rpm.size or x.size != t.size or x.size < 2 or fs <= 0:
        raise ValueError(f"Inconsistent record dimensions in {path}")
    if not all(np.isfinite(a).all() for a in (x,rpm,t)) or (rpm <= 0).any() or (stagger <= 0).any():
        raise ValueError(f"Nonfinite or invalid vibration/speed/time values in {path}")
    if not np.allclose(stagger,1 / fs,rtol=.01,atol=1e-8):
        raise ValueError(f"Time increments disagree with supplied sample rate in {path}")
    return x,rpm,t,fs


def fold_segments(x, rpm, lo, hi, guard):
    """Reset every fit segment and omit targets within guard of a held block."""
    if not 0 < guard < lo < hi < len(x) - guard:
        raise ValueError("The held block and guard must leave two fit segments")
    return [(x[:lo-guard], rpm[:lo-guard]), (x[hi+guard:], rpm[hi+guard:])]


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    def encode_scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Cannot JSON encode {type(value).__name__}")
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False, encoding="utf-8", mode="w") as f:
        tmp = Path(f.name)
        json.dump(data, f, indent=2, allow_nan=False, default=encode_scalar)
        f.write("\n")
    tmp.replace(path)


def atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as f:
        tmp = Path(f.name)
        np.savez_compressed(f, **arrays)
    tmp.replace(path)
