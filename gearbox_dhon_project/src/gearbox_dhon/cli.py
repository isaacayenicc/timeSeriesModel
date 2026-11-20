"""Run or replay the project; blind evaluation never enters a fitting path."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from .data import find_input, iter_fault_records, load_record, run_audit
from .freeze import create_manifest, run_blind, sha256, write_json
from .inference import development_specs, load_predictor, model_specs
from .results import START, analyze, read_predictions

ROOT = Path(__file__).resolve().parents[2]


def select_device(device):
    import torch
    selected = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if selected == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA explicitly requested but unavailable")
    return selected


def replay(root, data, run, device="cpu"):
    valid_path = find_input(data, "sigvalid")
    if not valid_path:
        raise FileNotFoundError("Original sigvalid is required for independent checkpoint replay")
    valid = load_record(valid_path)
    arrays = read_predictions(run, len(valid.x))
    result = {}
    for name, (kind, path) in development_specs(run, root).items():
        if name in arrays:
            y = load_predictor(kind, path, device)(valid)
            err = float(np.max(np.abs(arrays[name][START:] - y[START:])))
            result[name] = {"max_abs": err, "tolerance": 1e-5, "pass": err <= 1e-5}
    if not result or not all(row["pass"] for row in result.values()):
        raise RuntimeError(f"Replay mismatch: {result}")
    print(json.dumps(result, indent=2))
    return result


def fingerprint(root, data, profile, device, threads):
    inputs = {p.name: sha256(p) for p in sorted(data.iterdir()) if p.is_file()
              and p.name.lower().startswith(("sigtrain", "sigvalid", "sigfault1", "fdt_featcal"))}
    sources = {str(p.relative_to(root)): sha256(p)
               for base, pattern in (("src", "*.py"), ("configs", "*.yaml"), ("references/frozen_archive", "*"))
               for p in sorted((root / base).rglob(pattern)) if p.is_file()}
    state = dict(inputs=inputs, sources=sources, profile=profile, device=device, threads=threads)
    state["sha256"] = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
    return state


def run_pipeline(root, data, run_id, profile, device, threads=1, resume=False, skip_diagnostics=False):
    from . import baselines, comparators, dhon_search
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("Run IDs are single directory names")
    device = select_device(device)
    run = root / "outputs" / "runs" / run_id
    if run.exists() and not resume:
        raise FileExistsError(f"{run} exists. Use a new run ID or --resume.")
    run.mkdir(parents=True, exist_ok=True)
    state = dict(fingerprint(root, data, profile, device, threads), skip_diagnostics=skip_diagnostics)
    manifest = run / "run_fingerprint.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != state:
        raise ValueError("Resume rejected: inputs, implementation, device or configuration changed")
    if not manifest.exists() and any(run.iterdir()):
        raise ValueError("Resume rejected: existing run has no pipeline fingerprint")
    write_json(manifest, state)
    stages = run / "pipeline_stages.json"
    progress = json.loads(stages.read_text(encoding="utf-8")) if stages.exists() else {}

    def stage(name, operation):
        if progress.get(name, {}).get("status") == "PASS":
            return
        print(f"[{run_id}] {name}", flush=True)
        start = perf_counter()
        try:
            operation()
        except Exception as exc:
            progress[name] = {"status": "ERROR", "seconds": perf_counter()-start, "error": str(exc)}
            write_json(stages, progress)
            raise
        progress[name] = {"status": "PASS", "seconds": perf_counter()-start}
        write_json(stages, progress)

    with threadpool_limits(limits=threads):
        stage("audit", lambda: run_audit(data, run))
        train_path, valid_path = [find_input(data, f"sig{stem}") for stem in ("train", "valid")]
        if train_path is None or valid_path is None:
            print("BLOCKED: the requested fit requires sigtrain and sigvalid.")
            return run
        train, valid = load_record(train_path), load_record(valid_path)
        from .eda import run_eda
        stage("eda", lambda: run_eda(train, run))
        stage("baselines", lambda: baselines.run_baselines(data, run, profile, threads=threads, timing=False))
        stage("dhon", lambda: dhon_search.run(data, run, profile, device, resume=resume))
        stage("comparators", lambda: comparators.run(train, valid, run, profile, device, threads))
        stage("matched_metrics", lambda: analyze(run, valid))
        stage("checkpoint_replay", lambda: write_json(run / "metrics" / "checkpoint_replay.json",
                                                       replay(root, data, run, device)))
        if profile == "full" and not skip_diagnostics:
            stage("faults_and_stress", lambda: run_faults(root, data, run, device))
        stage("timing", lambda: run_timing(root, data, run, device, threads, repeats=3 if profile == "smoke" else 7))
        stage("freeze", lambda: create_manifest(run, data, root))
    return run


def run_faults(root, data, run, device):
    from .faults import evaluate_faults, evaluate_robustness
    valid = load_record(find_input(data, "sigvalid"))
    predictors = {key: load_predictor(*spec, device)
                  for key, spec in development_specs(run, root).items()
                  if key in {"primary_dhon", "spa_lpv_arma", "archive_dhon_optimized_corrected"}}
    if not predictors:
        raise ValueError("No deployable diagnostic predictors")
    evaluate_robustness(valid, run, predictors, start=START)
    status = json.loads((run / "data_availability.json").read_text(encoding="utf-8"))
    if status.get("sigfault1_count"):
        evaluate_faults(data, run, predictors, start=START)


def run_timing(root, data, run, device, threads, repeats=7):
    from .benchmark import benchmark
    specs = development_specs(run, root)
    if device == "cuda":
        specs = {key: value for key, value in specs.items() if value[0] in {"dhon", "comparator"}}
    predictors = {name: load_predictor(*spec, device) for name, spec in specs.items()}
    records = {"development_7p5s": load_record(find_input(data, "sigvalid"))}
    try:
        representative = next(iter_fault_records(data, dataset="sigfault1"))
        records["candidate_10s"] = representative
    except (FileNotFoundError, StopIteration):
        pass
    benchmark(predictors, records, run / "metrics", device, threads, repeats)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "replay", "audit", "analyze", "faults", "benchmark", "freeze", "blind"])
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--run-id", default="development")
    parser.add_argument("--profile", choices=["full", "smoke"], default="full")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-diagnostics", action="store_true")
    parser.add_argument("--frozen-manifest", type=Path)
    parser.add_argument("--blind-output", type=Path, default=Path("outputs/runs/blind"))
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    root, data = ROOT, args.data_dir.resolve()
    run = root / "outputs" / "runs" / args.run_id
    if args.command == "run":
        run_pipeline(root, data, args.run_id, args.profile, args.device, args.threads, args.resume, args.skip_diagnostics)
    elif args.command == "audit":
        print(run_audit(data, run, args.reference_dir))
    elif args.command == "freeze":
        print(create_manifest(run, data, root))
    else:
        device = select_device(args.device)
        with threadpool_limits(limits=args.threads):
            if args.command == "replay":
                replay(root, data, run, device)
            elif args.command == "analyze":
                print(analyze(run, load_record(find_input(data, "sigvalid"))).to_string(index=False))
            elif args.command == "faults":
                run_faults(root, data, run, device)
            elif args.command == "benchmark":
                run_timing(root, data, run, device, args.threads)
            elif args.command == "blind":
                if args.frozen_manifest is None:
                    parser.error("blind requires --frozen-manifest")
                print(run_blind(data, args.frozen_manifest.resolve(), args.blind_output.resolve(), root, device))


if __name__ == "__main__":
    main()
