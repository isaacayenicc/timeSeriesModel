"""Load MATLAB v5 structs while preserving recording boundaries."""
from pathlib import Path
import hashlib
import numpy as np
from scipy.io import loadmat


def sha256(path):
    """Return streaming SHA256 of an unmodified input file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_records(path):
    """Return validated dict records, never concatenate MATLAB structs.

    Vectors x/t/spd/spdnorm must be finite, length N, with uniform 1/fs time.
    HDF5 MATLAB files fail explicitly; no unsupported format is misread.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing user-supplied input: {path}")
    try:
        content = loadmat(path, simplify_cells=True)
    except NotImplementedError as exc:
        raise ValueError("MATLAB v7.3/HDF5 requires a separate verified loader; not supported") from exc
    records = []

    def visit(value):
        if isinstance(value, dict):
            if "x" in value:
                records.append(dict(value))
            else:
                for child in value.values():
                    visit(child)
        elif isinstance(value, (list, tuple, np.ndarray)):
            for child in np.asarray(value, dtype=object).flat:
                if isinstance(child, (dict, list, tuple, np.ndarray)):
                    visit(child)
    for key, value in content.items():
        if not key.startswith("__"):
            visit(value)
    if not records:
        raise ValueError(f"No vibration record structs in {path.name}")
    for i, record in enumerate(records):
        for key in ("N", "fs", "x", "t", "spd", "spdnorm"):
            if key not in record:
                raise ValueError(f"{path.name} record {i}: missing {key}")
        n = int(record["N"])
        fs = float(record["fs"])
        if n != record["N"] or n < 2 or not np.isfinite(fs) or fs <= 0:
            raise ValueError("Invalid N or sampling frequency")
        for key in ("x", "t", "spd", "spdnorm", "xraw"):
            if key in record:
                a = np.asarray(record[key], dtype=float)
                if a.ndim != 1 or len(a) != n or not np.isfinite(a).all():
                    raise ValueError(f"{path.name} record {i}: invalid vector {key}")
                record[key] = a
        if not np.allclose(np.diff(record["t"]), 1/fs, rtol=1e-7, atol=1e-10):
            raise ValueError(f"{path.name} record {i}: invalid timestamp increments")
        if np.any(record["spd"] <= 0):
            raise ValueError("Speed must be positive for this workflow")
        record.update(record_id=f"{path.stem}_{i+1:03d}", N=n, fs=fs)
    return records
