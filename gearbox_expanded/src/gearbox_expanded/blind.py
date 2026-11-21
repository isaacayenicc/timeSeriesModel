"""Inference-only blind evaluation from the frozen expanded study."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from gearbox_dhon.data import fault_sources, find_input, iter_fault_records, load_record
from gearbox_dhon.faults import diagnostics
from gearbox_dhon.freeze import exact_overlap
from gearbox_dhon.results import scores
from .protocol import START, digest, load_data, write_json
from .study import model_registry


def run(data, evidence, prior, out):
    if out.exists():
        raise FileExistsError("Choose a new directory for blind evidence")
    manifest = json.loads((evidence / "frozen_comparison.json").read_text(encoding="utf-8"))
    if (manifest["target_mask"]["start"], manifest["speed_cutoff"]) != (START, "previous sample"):
        raise ValueError("Unexpected scoring or measurement protocol")
    for name, score in manifest["source_sha256"].items():
        if digest(Path(__file__).parent / name) != score:
            raise ValueError(f"Changed expanded source: {name}")
    for dataset, field in [("sigtrain", "train"), ("sigvalid", "validation")]:
        if digest(find_input(data, dataset)) != manifest[field]["sha256"]:
            raise ValueError(f"Changed dataset: {dataset}")
    predictors, metadata = model_registry(evidence, prior)
    if metadata != manifest["model_metadata"]:
        raise ValueError("Changed checkpoint registry")
    path = find_input(data, "sigtest")
    faults = fault_sources(data, "sigfault2")
    status = {"sigtest": "BLOCKED: absent" if path is None else "PRESENT",
              "sigfault2": "BLOCKED: absent" if not faults else "PRESENT"}
    if path is None and not faults:
        write_json(out / "availability.json", status)
        return status

    out.mkdir(parents=True)
    refs = [*load_data(data), *iter_fault_records(data)]
    if path is not None:
        item = load_record(path)
        if any(exact_overlap(item, ref) for ref in refs):
            status["sigtest"] = "EXCLUDED: joint exact overlap with development"
        else:
            vectors = {key: np.asarray(predict(item))[START:] for key,predict in predictors.items()}
            pd.DataFrame([dict(model=key, **scores(item.x[START:], v)) for key,v in vectors.items()]).to_csv(
                out / "blind_vibration_scores.csv", index=False)
            np.savez_compressed(out / "blind_vibration_forecasts.npz", **vectors)
            status["sigtest"] = "SCORED: exact joint overlap screen passed; acquisition metadata unconfirmed"
        refs.append(item)
    if faults:
        rows = []
        for item in iter_fault_records(data,dataset="sigfault2"):
            if any(exact_overlap(item, ref) for ref in refs):
                rows.append(dict(record=item.index, scope="excluded_overlap"))
                continue
            refs.append(item)
            for key, predict in predictors.items():
                try:
                    value = np.asarray(predict(item))[START:]
                    numeric, _ = diagnostics(item.x[START:]-value, item.rpm[START:], item.t[START:])
                    rows.append(dict(record=item.index, model=key, scope="unlabelled", **numeric))
                except ValueError as exc:
                    rows.append(dict(record=item.index, model=key, scope="invalid_input", reason=str(exc)))
        pd.DataFrame(rows).to_csv(out / "blind_diagnostics.csv", index=False)
        status["sigfault2"] = "SCORED: unlabelled diagnostics; threshold-free"
    write_json(out / "availability.json", status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--prior", type=Path, default=Path("prior/gearbox_dhon_project"))
    parser.add_argument("--evidence", type=Path, default=Path("outputs/expanded"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    with threadpool_limits(1):
        print(run(args.data_dir, args.evidence, args.prior, args.out))


if __name__ == "__main__":
    main()
