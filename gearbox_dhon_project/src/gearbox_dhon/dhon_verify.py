"""Check strict causal and checkpoint behavior on the supplied validation record."""

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from .dhon import SCORE_START, load_model
from .dhon_search import _read


def verify(data_dir: Path, run_dir: Path) -> dict:
    record = _read(data_dir,"sigvalid")
    x, rpm, fs = record["x"], record["spd"], float(record["fs"])
    path = run_dir/"checkpoints"/"primary_dhon.npz"
    model = load_model(path)
    baseline = model.predict(x,rpm,fs)
    bounds = (SCORE_START, 12000, 22000)
    records=[]
    for t in bounds:
        xa = x.copy()
        ra = rpm.copy()
        xa[t:] += 3
        ra[t:] *= 1.008
        indices = slice(SCORE_START,t+1)
        err = float(np.max(np.abs(model.predict(xa,ra,fs)[indices] - baseline[indices])))
        truncate = float(np.max(np.abs(model.predict(x[:t+1],rpm[:t+1],fs)[indices] - baseline[indices])))
        records.append({"boundary_index":t,"perturbed_suffix_max_abs":err,
                        "truncated_prefix_max_abs":truncate})
    repeat = float(np.max(np.abs(model.predict(x,rpm,fs)[SCORE_START:]-baseline[SCORE_START:])))
    replay = float(np.max(np.abs(load_model(path).predict(x,rpm,fs)[SCORE_START:]-baseline[SCORE_START:])))
    bounds_met = all(max(row["perturbed_suffix_max_abs"],row["truncated_prefix_max_abs"]) < 3e-5
                     for row in records)
    result = {"causal_checks":records, "float32_tolerance":3e-5, "repeated_record_max_abs":repeat,
              "checkpoint_reload_max_abs":replay,"status":"PASS" if bounds_met and replay == repeat == 0 else "FAIL"}
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--run-dir",type=Path,required=True)
    options=parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(1):
        result=verify(options.data_dir,options.run_dir)
    serialized=json.dumps(result,indent=2)+"\n"
    (options.run_dir/"metrics"/"dhon_development_verification.json").write_text(serialized,encoding="utf-8")
    print(serialized)
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
