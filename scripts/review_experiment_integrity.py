"""Independently check saved arrays against inputs, metrics and winner selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rotmach.io_matlab import load_records, sha256  # noqa: E402


def review(root: Path) -> dict:
    """Check all finished runs; this rescores arrays, not loaded checkpoints."""
    record = load_records(root / "data/raw/sigvalid.mat")[0]
    hashes = {split: sha256(root / "data/raw" / f"sig{split}.mat")
              for split in ("train", "valid")}
    rows = []
    shared_indices = None
    for path in sorted((root / "experiments/runs").glob("*/metrics.json")):
        row = json.loads(path.read_text())
        config = json.loads((path.parent / "config.json").read_text())
        saved = np.load(path.parent / "predictions.npz", allow_pickle=False)
        expected_indices = np.arange(config["burnin"], record["N"])
        np.testing.assert_array_equal(saved["target_indices"], expected_indices)
        if shared_indices is None:
            shared_indices = expected_indices
        np.testing.assert_array_equal(saved["target_indices"], shared_indices)
        np.testing.assert_array_equal(saved["y_true"], record["x"][shared_indices])
        np.testing.assert_allclose(saved["residual"], saved["y_true"] - saved["y_pred"], rtol=0, atol=0)
        mse = float(np.mean(saved["residual"] ** 2))
        if abs(mse - row["mse"]) >= 1e-14:
            raise AssertionError(f"MSE mismatch in {row['run_id']}")
        for split in ("train", "valid"):
            if config[f"{split}_sha256"] != hashes[split]:
                raise AssertionError(f"Data checksum mismatch in {row['run_id']}")
        rows.append(row)
    if not rows:
        raise RuntimeError("No finished runs to review")
    best = json.loads((root / "experiments/best_model.json").read_text())
    if best["kind"] == "shift_tcn" or best["mse"] != min(
        row["mse"] for row in rows if row["kind"] != "shift_tcn"
    ):
        raise AssertionError("Recommended model selection is inconsistent")
    result = {
        "status": "PASS",
        "runs_checked": len(rows),
        "first_index_inclusive": int(shared_indices[0]),
        "last_index_inclusive": int(shared_indices[-1]),
        "targets_per_run": len(shared_indices),
        "identical_y_true": True,
        "metrics_recomputed_from_saved_predictions": True,
        "data_hashes_match": True,
        "noncausal_control_excluded_from_winner": True,
        "best_run": best["run_id"],
        "scope": "Saved-array integrity; checkpoint reload checked separately",
    }
    (root / "reports/independent_integrity_review.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    arguments = parser.parse_args()
    print(json.dumps(review(arguments.root.resolve()), indent=2))
