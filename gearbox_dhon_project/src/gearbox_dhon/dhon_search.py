"""Guarded training-only design search and saved development checkpoints."""

from dataclasses import asdict, replace
import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.io import loadmat
from scipy.signal import lfilter
from threadpoolctl import threadpool_limits
import torch

from .dhon import BASE_HISTORY, SCORE_START, Config, fit, injected_pulses, load_model, slow_adapter

FOLDS = ((4800, 7200), (11200, 13600), (18400, 20800))
SEED = 19
FINAL_SEEDS = (19, 29, 41)


def stages() -> list[tuple[str, str, Config]]:
    """Frozen deterministic screening and interaction design."""
    start = Config(epochs=7)
    expanded = replace(start, lags=96, angular_steps=16, mesh_cycles=8,
                       harmonics=tuple(range(1, 13)), taus=(24, 80, 256),
                       memory=512, innovation_order=256, innovation_lags=24,
                       ridge=.025, width=16, gate=.6)
    compact = replace(start, lags=64, angular_steps=8, mesh_cycles=4, harmonics=(1, 2, 4, 7, 12),
                      taus=(8, 32, 128), memory=256, innovation_order=128, innovation_lags=16,
                      ridge=.025, width=8, gate=.4)
    focused = replace(start, lags=128, harmonics=(1, 2, 4, 7, 12), hz=(1080., 1125.),
                      taus=(12, 40, 128), memory=384, innovation_order=192, innovation_lags=24,
                      ridge=.018, width=16, gate=.5)
    temporal = replace(start, harmonics=(), mesh_cycles=0, acceleration=False,
                       innovation_order=0, innovation_lags=0, width=0, epochs=0,
                       joint="fixed", use_speed=False, speed_degree=0)
    return [
        ("A", "raw_history", temporal),
        ("A", "with_rpm", replace(temporal, use_speed=True, speed_degree=2)),
        ("A", "with_angle", replace(temporal, use_speed=True, speed_degree=2, mesh_cycles=6)),
        ("A", "with_oscs_linear", replace(start, width=0, joint="fixed", epochs=0)),
        ("B", "compact_fixed", replace(compact, joint="fixed")),
        ("B", "expanded_fixed", replace(expanded, joint="fixed")),
        ("B", "focused_fixed", replace(focused, joint="fixed")),
        ("C", "compact_joint", compact),
        ("C", "expanded_joint", expanded),
        ("C", "focused_joint", focused),
        ("C", "focused_decay", replace(focused, joint="decay")),
        ("C", "focused_frequency", replace(focused, joint="frequency")),
        ("C", "focused_fixed_hz_only", replace(focused, harmonics=())),
        ("C", "expanded_short_tau", replace(expanded, taus=(8, 32, 128), width=8, ridge=.012)),
        ("C", "focused_no_innov", replace(focused, innovation_order=0, innovation_lags=0)),
        ("C", "focused_no_acc", replace(focused, acceleration=False)),
        ("C", "compact_long_lags", replace(compact, lags=192, ridge=.008, lr=.00015)),
        ("C", "expanded_weak_ridge", replace(expanded, ridge=.008, gate=.35, lr=.0004)),
        ("D", "focused_no_preserve", replace(focused, preserve=0., augment=0.)),
        ("D", "compact_quick_adapter", replace(compact, adapter_gain=.4, adapter_tau=320, adapter_memory=768)),
        ("D", "expanded_no_adapter", replace(expanded, adapter_gain=0.)),
    ]


def fold_segments(x: np.ndarray, rpm: np.ndarray, bounds: tuple[int, int],
                  chronological: bool = False) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return separately reset raw fit histories outside target+guard intervals."""
    a, b = bounds
    before = [(x[:a - SCORE_START], rpm[:a - SCORE_START])]
    after = [(x[b + SCORE_START:], rpm[b + SCORE_START:])]
    return before if chronological else before + after


def rpm_stress(rpm: np.ndarray, bias: float, seed: int) -> np.ndarray:
    """Signed 1.2% calibration bias and 0.2% correlated jitter use separate seed."""
    noise = lfilter([.035], [1, -.965], np.random.default_rng(seed).standard_normal(len(rpm)))
    return rpm * (1 + bias + .002 * noise / max(np.std(noise), 1e-12))


def _metrics(x, rpm, fs, model, indices) -> tuple[dict, np.ndarray]:
    base = model.predict_base(x, rpm, fs)
    c = model.config
    p = slow_adapter(x, base, c.adapter_gain, c.adapter_tau, c.adapter_memory, c.adapter_clip)[indices]
    y = x[indices]
    stress = [model.predict(x, rpm_stress(rpm, bias, 201 + j), fs)[indices]
              for j, bias in enumerate((-.012, .012))]
    pulses = injected_pulses(rpm, fs)
    pulse_pred = model.predict(x + pulses, rpm, fs)[indices]
    delta = pulse_pred - p
    pulse_resid = pulses[indices] - delta
    pulse_energy = max(np.sum(pulses[indices]**2), 1e-12)
    distortion = float(np.sum(delta**2) / pulse_energy)
    stress_mse = float(np.mean([(q - y) ** 2 for q in stress]))
    mse = float(np.mean((p - y)**2))
    return {
        "mse": mse, "base_mse": float(np.mean((base[indices] - y)**2)),
        "mae": float(np.mean(np.abs(p-y))), "stress_mse": stress_mse,
        "pulse_distortion": distortion,
        "residual_pulse_projection": float(np.dot(pulse_resid, pulses[indices]) / pulse_energy),
        "utility": mse + .12 * stress_mse + .004 * distortion, "n": len(indices),
    }, p


def _atomic_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _csv(path: Path, rows: list[dict]) -> None:
    if rows:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(path)


def _read(path: Path, key: str) -> dict:
    candidate = path / f"{key}.mat"
    if not candidate.exists():
        matches = [p for p in path.glob("*.mat") if p.stem.lower().replace(" ", "").startswith(key)]
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected a unique {key}.mat in {path}")
        candidate, = matches
    return loadmat(candidate, simplify_cells=True)[key]


def run(data_dir: Path, run_dir: Path, profile: str = "full", device: str = "cpu",
        resume: bool = False) -> None:
    """Search guarded folds, freeze choices, refit and score one official validation record."""
    paths = {key: run_dir / key for key in ("metrics", "predictions", "search", "checkpoints")}
    for path in paths.values():
        path.mkdir(exist_ok=True, parents=True)
    search = paths["search"]
    train = _read(data_dir, "sigtrain")
    x, rpm, fs = train["x"], train["spd"], float(train["fs"])
    trials = stages()
    if profile == "smoke":
        trials = [("smoke", "compact_joint", replace(trials[7][2], epochs=3, augment=0., preserve=0.)),
                  ("smoke", "focused_joint", replace(trials[9][2], epochs=3, augment=0., preserve=0.))]
    digest = hashlib.sha256()
    for array in (x, rpm):
        digest.update(np.asarray(array, dtype="<f8").tobytes())
    digest.update(json.dumps({"trials": [(a, b, asdict(c)) for a,b,c in trials], "profile": profile,
                              "device": device, "folds": FOLDS, "fs": fs}, sort_keys=True).encode())
    digest.update(Path(__file__).read_bytes())
    digest.update(Path(__file__).with_name("dhon.py").read_bytes())
    signature = digest.hexdigest()
    manifest = search / "dhon_search_manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if not resume or old["signature"] != signature:
            raise ValueError("Use a new run directory, or --resume with exact input/code/design match")
    _atomic_json(manifest, {"signature": signature, "device": device, "seed": SEED, "profile": profile,
                            "fold_bounds": FOLDS, "fold_guard": SCORE_START, "base_reset": BASE_HISTORY,
                            "n_scheduled": len(trials), "torch": torch.__version__, "threads": 1,
                            "expected_fold_fits": len(trials) * len(FOLDS) + (3 if profile == "full" else 0),
                            "selection_objective": "mean(mse + .12 * stress_mse + .004 * pulse_distortion)",
                            "trials": {name: asdict(conf) for _, name, conf in trials}})
    rows = []
    ranked = []
    folds_dir = search / "folds"
    folds_dir.mkdir(exist_ok=True)
    for stage, label, config in trials:
        per_trial = []
        print(f"DHON {stage} {label}", flush=True)
        for k, bounds in enumerate(FOLDS):
            start = perf_counter()
            fold_file = folds_dir / f"{label}_{k}.json"
            if fold_file.exists() and resume:
                row = json.loads(fold_file.read_text(encoding="utf-8"))
            else:
                a, b = bounds
                segments = fold_segments(x, rpm, bounds)
                idx = np.arange(a, b)
                row = {"stage": stage, "trial": label, "fold": k + 1, "seed": SEED,
                       "test_start": a, "test_end": b, "status": "complete",
                       "n_train_fit": sum(max(len(v) - BASE_HISTORY, 0) for v, _ in segments)}
                row.update(asdict(config))
                try:
                    model, trace = fit(config, segments, fs, SEED, device)
                    fit_s = perf_counter() - start
                    result, predictions = _metrics(x[:b], rpm[:b], fs, model, idx)
                    np.savez_compressed(folds_dir / f"{label}_{k}.npz", indices=idx, actual=x[idx],
                                        prediction=predictions)
                    _atomic_json(folds_dir / f"{label}_{k}_learning.json", trace)
                    row.update(result, fit_seconds=fit_s,
                               trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as err:
                    row.update(status="failed", error=str(err))
                row["seconds"] = perf_counter() - start
                _atomic_json(fold_file, row)
            rows.append(row)
            per_trial.append(row)
        _csv(search / "dhon_trials.csv", rows)
        if all(r["status"] == "complete" for r in per_trial):
            ranked.append({"trial": label, "stage": stage, "config": config,
                           "trainable": per_trial[0]["trainable"],
                           **{name: float(np.mean([r[name] for r in per_trial]))
                              for name in ("mse", "stress_mse", "pulse_distortion", "utility")},
                           "mse_fold_std": float(np.std([r["mse"] for r in per_trial], ddof=1))})
        _csv(search / "dhon_ranking.csv", [{k:v for k,v in row.items() if k != "config"}
                                           for row in sorted(ranked, key=lambda r:r["utility"])])
    ranking = sorted(ranked, key=lambda r:r["utility"])
    eligible = [r for r in ranking if r["config"].joint == "both" and r["config"].width]
    if len(eligible) < 2:
        raise RuntimeError("Fewer than two completed trainable-state choices")
    primary = eligible[0]
    compact = [r for r in eligible[1:] if r["trainable"] <= primary["trainable"] and
               r["config"].adapter_gain == primary["config"].adapter_gain]
    alternative = (compact or eligible[1:])[0]
    chosen = {"primary_dhon": primary["config"], "alternative_dhon": alternative["config"]}
    if profile == "full":
        chosen.update({
            "dhon_matched_fixed": replace(primary["config"], joint="fixed"),
            "dhon_matched_decay_only": replace(primary["config"], joint="decay"),
            "dhon_matched_frequency_only": replace(primary["config"], joint="frequency"),
            "dhon_matched_linear": replace(primary["config"], joint="fixed", width=0, epochs=0),
        })
    _atomic_json(search / "dhon_selection.json", {
        "profile": profile, "primary_trial": primary["trial"], "primary_fold_utility": primary["utility"],
        "alternative_trial": alternative["trial"], "alternative_fold_utility": alternative["utility"],
        "rule": "Best joint utility; next best no-larger at the same gain when available",
        "selected_before_official_validation": True,
        "configurations": {name: asdict(config) for name, config in chosen.items()},
    })
    # Official validation is opened only after the fold ranking and configuration selection exist.
    validation = _read(data_dir, "sigvalid")
    vx, vrpm = validation["x"], validation["spd"]
    if float(validation["fs"]) != fs:
        raise ValueError("Training and validation sample rates differ")
    idx = np.arange(SCORE_START, len(vx))
    y = vx[idx]
    series = {"indices": idx, "actual": vx, "rpm": vrpm, "time": validation["t"]}
    metrics = []
    validations = []
    for slug, config in chosen.items():
        print(f"DHON final {slug}", flush=True)
        seeds = FINAL_SEEDS if slug == "primary_dhon" and profile == "full" else (SEED,)
        for seed in seeds:
            name = slug if seed == SEED else f"{slug}_seed{seed}"
            start = perf_counter()
            model, trace = fit(config, [(x,rpm)], fs, seed, device)
            fit_s = perf_counter() - start
            path = paths["checkpoints"] / f"{name}.npz"
            model.save(path)
            _atomic_json(search / f"{name}_learning.json", trace)
            result, pred = _metrics(vx, vrpm, fs, model, idx)
            restored = load_model(path, device)
            reloaded = restored.predict(vx, vrpm, fs)
            reload_error = float(np.max(np.abs(reloaded[idx] - pred)))
            if reload_error > 1e-6:
                raise RuntimeError(f"Checkpoint mismatch {name}: {reload_error:g}")
            series[name] = reloaded
            base = restored.predict_base(vx, vrpm, fs)
            series[f"{name}_base"] = base
            train = restored.predict(x, rpm, fs)
            trials_s = []
            for _ in range(5):
                t0 = perf_counter()
                restored.predict(vx,vrpm,fs)
                if device == "cuda":
                    torch.cuda.synchronize()
                trials_s.append(perf_counter() - t0)
            fitted = sum(p.numel() for p in model.parameters() if p.requires_grad)
            deployed = sum(p.numel() for p in model.parameters()) + model.innovation.numel()
            metrics.append({"model": name, "mse": result["mse"], "mae": result["mae"],
                            "base_mse": result["base_mse"], "n": len(idx), "seed": seed,
                            "train_mse": float(np.mean((x[SCORE_START:] - train[SCORE_START:])**2)),
                            "trainable": fitted, "parameters": deployed, "checkpoint_bytes": path.stat().st_size,
                            "fit_seconds": fit_s, "median_predict_s": float(np.median(trials_s)),
                            "p95_predict_s": float(np.quantile(trials_s,.95)), "reload_error": reload_error})
            validations.append({"model": name, **result})
    if profile == "full":
        chrono = []
        for a,b in FOLDS:
            model, _ = fit(primary["config"], fold_segments(x,rpm,(a,b),True), fs, SEED, device)
            row,_ = _metrics(x[:b],rpm[:b],fs,model,np.arange(a,b))
            chrono.append({"start":a,"end":b,**row})
        _csv(search / "dhon_chronological.csv", chrono)
    _csv(paths["metrics"] / "dhon_metrics.csv", metrics)
    _csv(paths["metrics"] / "dhon_validation_robustness.csv", validations)
    np.savez_compressed(paths["predictions"] / "dhon.npz", **series)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=["full","smoke"], default="full")
    parser.add_argument("--device", choices=["cpu","cuda","auto"], default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this PyTorch runtime")
    torch.set_num_threads(1)
    with threadpool_limits(1):
        run(args.data_dir, args.run_dir, args.profile, device, args.resume)


if __name__ == "__main__":
    main()
