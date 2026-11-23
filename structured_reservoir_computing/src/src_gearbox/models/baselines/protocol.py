"""Common data, fold, timing, and model selection contracts."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ...data.io_matlab import Record, find_input, load_record

START = 1792
FOLDS = ((4800, 7200), (11200, 13600), (18400, 20800))
FAMILIES = {
    "statistical": ("ar", "arma", "arima", "arx", "sparse_lpv_ar", "sparse_lpv_arma"),
    "tabular": ("linear", "ridge", "lasso", "elastic_net", "svr", "random_forest",
                "gradient_boosting", "narx"),
    "neural": ("mlp", "lstm", "gru", "cnn", "tcn"),
    "attention": ("transformer", "tft"),
}
SEED = 7780


def load_data(data_dir: Path) -> tuple[Record, Record]:
    """Read common inputs and enforce the instructor's exact course grid."""
    files = [find_input(Path(data_dir), name) for name in ("sigtrain", "sigvalid")]
    if any(path is None for path in files):
        raise FileNotFoundError("Both sigtrain and sigvalid are required")
    train, valid = (load_record(path) for path in files)
    if any(len(record.x) != 24_000 or record.fs != 3200 for record in (train, valid)):
        raise ValueError("Expected 24,000 samples per course record at 3,200 Hz")
    return train, valid


def guarded_segments(record: Record, a: int, b: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """State-free fitting blocks separated from the held-out interval by 1,792 samples."""
    if not START <= a < b <= len(record.x) - START:
        raise ValueError("Guard falls outside the training acquisition")
    return [(record.x[:a-START], record.rpm[:a-START]),
            (record.x[b+START:], record.rpm[b+START:])]


def mse(y, pred) -> float:
    x = np.asarray(y, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    if x.shape != p.shape or x.ndim != 1 or not (np.isfinite(x).all() and np.isfinite(p).all()):
        raise ValueError("Targets and predictions must be aligned finite vectors")
    return float(np.mean((x-p)**2))


def predict(model, record: Record, stop: int | None = None):
    expected = len(record.x) if stop is None else stop
    pred = np.asarray(model.predict(record.x[:stop], record.rpm[:stop], record.fs), dtype=np.float64)
    if pred.shape != (expected,):
        raise ValueError(f"Model returned {pred.shape}; expected {(expected,)}")
    return pred


def timed_predict(model, record: Record, repeats: int = 5):
    """Return first result and timing statistics for full in-memory record calls."""
    start = perf_counter()
    first = predict(model, record)
    cold = perf_counter() - start
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        predict(model, record)
        samples.append(perf_counter()-start)
    return first, {
        "cold_s": cold, "median_s": float(np.median(samples)),
        "p95_s": float(np.quantile(samples, .95)),
        "iqr_s": float(np.ptp(np.quantile(samples, [.25, .75]))),
    }


def write_table(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False)+"\n", encoding="utf-8")


def digest(path: Path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_candidates(slug, candidates, train, valid, fit, out, loader=None):
    """Choose lowest equally weighted fold MSE then refit on full designated training."""
    table = []
    ranking = []
    for cid, config in enumerate(candidates):
        errors = []
        for fid, (a,b) in enumerate(FOLDS, start=1):
            start = perf_counter()
            estimator = fit(config, guarded_segments(train, a, b), train.fs)
            fit_seconds = perf_counter()-start
            values = predict(estimator, train, stop=b)
            loss = mse(train.x[a:b], values[a:b])
            errors.append(loss)
            table.append(dict(model=slug, candidate=cid, fold=fid, start=a, stop=b,
                              mse=loss, fit_s=fit_seconds, config=json.dumps(config, sort_keys=True)))
        ranking.append((float(np.mean(errors)), cid, float(np.std(errors, ddof=1))))
    cv_mean, cid, cv_sd = min(ranking)
    config = candidates[cid]
    start = perf_counter()
    selected = fit(config, [(train.x, train.rpm)], train.fs)
    fit_seconds = perf_counter()-start
    models = Path(out) / "models"
    models.mkdir(parents=True, exist_ok=True)
    checkpoint = models / f"{slug}.npz"
    selected.save(checkpoint)
    clean, timings = timed_predict(selected, valid)
    replay_max_abs = None
    if loader is not None:
        loaded = predict(loader(checkpoint), valid)
        replay_max_abs = float(np.max(np.abs(loaded[START:]-clean[START:])))
        if replay_max_abs > 1e-6:
            raise RuntimeError(f"{slug} checkpoint mismatch {replay_max_abs:g}")
    training = predict(selected, train)
    summary = dict(model=slug, fit_s=fit_seconds, **timings,
                   parameters=selected.parameters, checkpoint_bytes=checkpoint.stat().st_size,
                   selection=json.dumps(config, sort_keys=True), selected_candidate=cid,
                   fold_mean_mse=cv_mean, fold_sd=cv_sd, validation_mse=mse(valid.x[START:],clean[START:]),
                   train_mse=mse(train.x[START:], training[START:]), checkpoint_replay_max_abs=replay_max_abs)
    for row in table:
        row["selected"] = row["candidate"] == cid
    return summary, table, clean, training
