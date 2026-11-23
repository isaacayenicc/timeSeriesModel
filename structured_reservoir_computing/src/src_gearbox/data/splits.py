"""Stable data roles and common historical-speed evaluation protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .io_matlab import Record, find_input, load_record

START = 1792
FOLDS = ((4800, 7200), (11200, 13600), (18400, 20800))


def prepare_protocol(train: Record, development: Record, destination: Path) -> dict:
    """Describe the fixed split without training, modifying data, or loading blind targets."""
    if len(train.x) < FOLDS[-1][1] + START:
        raise ValueError("Training record is too short for the prespecified course folds")
    if len(development.x) <= START:
        raise ValueError("Development record has no common target after warm-up")
    target = np.arange(START, len(development.x), dtype="<i8")
    data = {
        "indexing": "zero-based, half-open",
        "prediction_horizon_samples": 1,
        "latest_vibration_index_relative_to_target": -1,
        "latest_rpm_index_relative_to_target": -1,
        "common_score_start": START,
        "common_score_stop_exclusive": len(development.x),
        "common_target_count": len(target),
        "common_target_indices_sha256_le_i64": hashlib.sha256(target.tobytes()).hexdigest(),
        "train_record": train.record_id,
        "development_record": development.record_id,
        "fold_guard_either_side": START,
        "folds": [
            {
                "fold": j, "score_span": [a,b],
                "permitted_raw_fit_spans": [[0,a-START], [b+START,len(train.x)]],
                "fit_segment_boundary_rule": "reinitialize windows and model state inside each fit segment",
            }
            for j, (a,b) in enumerate(FOLDS,1)
        ],
        "evidence_scope": "training folds share one acquisition; development is not a blind test",
    }
    destination = Path(destination)
    destination.mkdir(exist_ok=True, parents=True)
    (destination/"protocol.json").write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8")
    np.save(destination/"development_indices.npy", target, allow_pickle=False)
    return data


def prepare_from_inputs(data_dir: Path, destination: Path) -> dict:
    p, q = (find_input(data_dir, name) for name in ("sigtrain", "sigvalid"))
    if p is None or q is None:
        raise FileNotFoundError("The protocol requires both sigtrain and sigvalid")
    return prepare_protocol(load_record(p), load_record(q), destination)
