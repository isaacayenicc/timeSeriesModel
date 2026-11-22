"""Evaluate fitted SRC checkpoints and stored baseline controls on fault data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from src_gearbox.data import find_input, load_record
from src_gearbox.diagnostics.fault_metrics import analyze, availability
from src_gearbox.diagnostics.speed_stress import evaluate_stress


def registry(root, learner_specs, baseline_slugs):
    handlers, origins = {}, {}
    if learner_specs:
        from src_gearbox.src_model import load_model
        for spec in learner_specs:
            name, filename = spec.split("=", maxsplit=1)
            file = Path(filename)
            use_base = name.endswith("_same_weights_no_adapter")
            learner = load_model(file)
            handlers[name] = learner.predict_base if use_base else learner.predict
            origins[name] = {
                "origin": str(file.relative_to(root)) if file.is_relative_to(root) else file.name,
                "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                "prediction_method": "predict_base" if use_base else "predict",
            }
    if baseline_slugs:
        from src_gearbox.models.baselines.replay import load_predictor
        reference = root / "checkpoints" / "baselines"
        for name in baseline_slugs:
            if name in handlers:
                continue
            handlers[name] = load_predictor(name, reference).predict
            paths = sorted(reference.glob(name + ".*"))
            if name.startswith("archived_src") and not paths:
                stem = "archived_src_local" if "local" in name else "archived_src_optimized"
                paths = sorted(reference.glob(stem + ".*"))
            origins[name] = {
                "origin": "frozen_local_checkpoint_replay",
                "checkpoint_files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                "prediction_method": "predict",
            }
    return handlers, origins


def input_identity(path: Path, include_faults: bool) -> dict:
    files = []
    for name in ("sigtrain", "sigvalid", "fdt_featcal.m"):
        p = find_input(path, name)
        if p:
            files.append(p)
    if include_faults:
        from src_gearbox.data.io_matlab import fault_sources
        files.extend(fault_sources(path, "sigfault1"))
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--out", type=Path, default=ROOT / "results/fault")
    parser.add_argument("--figures", type=Path)
    parser.add_argument("--src", action="append", default=[], metavar="NAME=CHECKPOINT")
    parser.add_argument("--baseline", action="append", default=[], metavar="SLUG")
    parser.add_argument("--cohort", nargs="*", help="Names to score on all candidate acquisitions")
    parser.add_argument("--pulse", nargs="*", help="Names eligible for validation pulse experiments")
    parser.add_argument("--primary", default="primary_pi_src")
    parser.add_argument("--skip-cohort", action="store_true")
    parser.add_argument("--skip-stress", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.src and not args.baseline:
        automatic = ROOT / "checkpoints" / "src" / "primary_pi_src.npz"
        if not automatic.is_file():
            parser.error("The selected SRC checkpoint is unavailable. Supply --src or --baseline.")
        args.src.append(f"primary_pi_src={automatic}")
        args.baseline.append("sparse_lpv_arma")
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    with threadpool_limits(1):
        models, origins = registry(ROOT, args.src, args.baseline)
        cohort = models if args.cohort is None else {name: models[name] for name in args.cohort}
        manifest = {
            "input_sha256": input_identity(args.data_dir, not args.skip_cohort),
            "checkpoint_origins": origins, "cohort_models": list(cohort) if not args.skip_cohort else [],
            "stress_models": list(models) if not args.skip_stress else [],
            "primary": args.primary, "target_mask": "1792:N", "cpu_threads": 1,
        }
        (args.out / "fault_run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if not args.skip_stress:
            path = find_input(args.data_dir, "sigvalid")
            if not path:
                raise FileNotFoundError("The designated course validation file is required for stress testing")
            evaluate_stress(load_record(path), models, args.out, pulse_models=args.pulse)
        if not args.skip_cohort:
            analyze(args.data_dir, args.out, cohort, primary=args.primary)
        else:
            availability(args.data_dir, args.out)
        if args.figures:
            from src_gearbox.diagnostics.visualize import plot
            plot(args.out, args.figures, primary=args.primary)


if __name__ == "__main__":
    main()
