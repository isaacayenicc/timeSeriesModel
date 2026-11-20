"""Causal learned comparison models for the common one-step gearbox task."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.special import eval_chebyt
from threadpoolctl import threadpool_limits
import torch
from torch import nn


START = 1792
FOLD_RANGES = ((4800, 7200), (11200, 13600), (18400, 20800))
ANGLE_HISTORY = 768
SEED = 7780
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Config:
    family: str
    lag_count: int = 128
    rpm_degree: int = 0
    angular: bool = False
    orders: bool = False
    polynomial_degree: int = 0
    alpha: float = 1.0
    nonlinear_penalty: float = 1.0
    width: int = 24
    epochs: int = 14
    learning_rate: float = 0.002
    dilation_base: int = 3
    kernel_size: int = 3


def fold_targets(n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Indices for the archived guarded, blocked, training-only selection folds."""
    if n < FOLD_RANGES[-1][1] + START:
        raise ValueError("At least 22592 training samples are needed for the fixed folds.")
    all_targets = np.arange(START, n)
    return [
        (all_targets[(all_targets < first - START) | (all_targets >= stop + START)],
         np.arange(first, stop))
        for first, stop in FOLD_RANGES
    ]


def _bounded_angular_lags(x: FloatArray, rpm: FloatArray, fs: float,
                          targets: np.ndarray) -> FloatArray:
    """Interpolate past x at fixed angular delays, with a 768-sample support."""
    angles = np.r_[0.0, np.cumsum(rpm[:-1])] / (60.0 * fs)
    delays = np.array([1 / 37, 2 / 37, 3 / 37, 4 / 37, 6 / 37, 0.25, 0.375])
    left = targets - ANGLE_HISTORY
    wanted = angles[targets, None] - delays
    wanted = np.maximum(wanted, angles[left, None])
    right = np.searchsorted(angles, wanted, side="right")
    right = np.minimum(right, targets[:, None] - 1)
    left = np.maximum(right - 1, targets[:, None] - ANGLE_HISTORY)
    fraction = (wanted - angles[left]) / (angles[right] - angles[left])
    return x[left] + np.clip(fraction, 0.0, 1.0) * (x[right] - x[left])


def _order_features(x: FloatArray, rpm: FloatArray, fs: float,
                    targets: np.ndarray) -> FloatArray:
    """Short, fixed causal order quadratures with finite ring-down support."""
    recent = np.arange(1, 97)
    prior = targets[:, None] - recent
    angle = np.cumsum(rpm[prior], axis=1) / (60.0 * fs)
    weighted_x = x[prior] * np.exp(-recent / 28.0)
    norm = np.exp(-recent / 28.0).sum()
    columns = []
    for order in (37.0, 74.0, 111.0, 148.0):
        phase = 2 * np.pi * order * angle
        columns.extend([(weighted_x * np.cos(phase)).sum(1) / norm,
                        (weighted_x * np.sin(phase)).sum(1) / norm])
    return np.column_stack(columns)


def feature_matrix(x: FloatArray, rpm: FloatArray, fs: float, targets: np.ndarray,
                   config: Config) -> FloatArray:
    """Create finite-history regressors. Row t only accesses observations through t-1."""
    lags = np.arange(1, config.lag_count + 1)
    history = x[targets[:, None] - lags]
    columns = [history]
    if config.rpm_degree:
        speed = rpm[targets - 1]
        # Fixed engineering units determine the basis coordinates. All feature scaling is fitted.
        q = (speed - 145.0) / 25.0
        accel = (speed - rpm[targets - 33]) / 25.0
        columns.extend([q[:, None], accel[:, None]])
        basis = [eval_chebyt(degree, q)[:, None] for degree in range(1, config.rpm_degree + 1)]
        if config.family != "mlp":
            columns.extend(history * polynomial for polynomial in basis)
    if config.angular:
        angular = _bounded_angular_lags(x, rpm, fs, targets)
        columns.extend([angular, angular * q[:, None]])
    if config.orders:
        quadrature = _order_features(x, rpm, fs, targets)
        columns.extend([quadrature, quadrature * q[:, None]])
    if config.polynomial_degree:
        # Restricted NG-RC library: quadratic short delays and selected
        # cubic energy-by-lag and order energy-by-quadrature terms.
        delayed = history[:, np.array([0, 2, 5, 10, 17, 24, 33, 47])]
        energy = np.mean(history[:, :32] ** 2, axis=1, keepdims=True)
        columns.append(np.column_stack([delayed[:, i] * delayed[:, j]
                                        for i in range(delayed.shape[1])
                                        for j in range(i, delayed.shape[1])]))
        if config.polynomial_degree >= 3:
            columns.append(delayed * energy)
            if config.orders:
                columns.append(quadrature * energy)
                order_energy = quadrature[:, ::2] ** 2 + quadrature[:, 1::2] ** 2
                columns.extend([order_energy, order_energy * delayed[:, :4]])
    return np.column_stack(columns)


class FeedForward(nn.Module):
    """Linear skip and tanh hidden readout for the same delay vector."""

    def __init__(self, input_size: int, width: int):
        super().__init__()
        self.skip = nn.Linear(input_size, 1)
        self.nonlinear = nn.Sequential(nn.Linear(input_size, width), nn.Tanh(), nn.Linear(width, 1))
        nn.init.zeros_(self.nonlinear[-1].weight)
        nn.init.zeros_(self.nonlinear[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.skip(x) + self.nonlinear(x)).squeeze(-1)


class CausalTCN(nn.Module):
    """Finite convolution stack evaluated with left-only padding and explicit input shift."""

    def __init__(self, input_size: int, config: Config):
        super().__init__()
        self.input_layer = nn.Conv1d(input_size, config.width, 1)
        self.pads = [config.dilation_base ** i * (config.kernel_size - 1) for i in range(4)]
        self.layers = nn.ModuleList(
            nn.Conv1d(config.width, config.width, config.kernel_size, dilation=config.dilation_base ** i)
            for i in range(4)
        )
        self.readout = nn.Conv1d(config.width, 1, 1)

    @property
    def history(self) -> int:
        return sum(self.pads) + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_layer(x)
        for conv, pad in zip(self.layers, self.pads):
            z = z + torch.tanh(conv(nn.functional.pad(z, (pad, 0))))
        return self.readout(z).flatten()


def _linear_parameters(z: FloatArray, target: FloatArray,
                       alpha: float | FloatArray) -> tuple[FloatArray, float]:
    """Centered positive-definite ridge solve with an unpenalized intercept."""
    center = float(target.mean())
    gram = z.T @ z
    gram.flat[::len(gram) + 1] += alpha
    weights = cho_solve(cho_factor(gram, check_finite=False),
                        z.T @ (target - center), check_finite=False)
    return weights, center


def _speed_scale(x: FloatArray, rpm: FloatArray, targets: np.ndarray) -> tuple[FloatArray, FloatArray]:
    values = np.vstack((x[targets - 1], rpm[targets - 1]))
    return values.mean(1), np.maximum(values.std(1), 1e-7)


def _tcn_input(x: FloatArray, rpm: FloatArray, location: FloatArray,
               scale: FloatArray, use_speed: bool, device: str) -> torch.Tensor:
    shifted = np.column_stack((np.r_[0.0, x[:-1]], np.r_[location[1], rpm[:-1]]))
    scaled = (shifted - location) / scale
    return torch.as_tensor(scaled.T[None, :1 + use_speed, :], device=device, dtype=torch.float32)


@dataclass
class Comparator:
    config: Config
    location: FloatArray
    scale: FloatArray
    fs: float
    coefficient: FloatArray | None = None
    intercept: float = 0.0
    module: nn.Module | None = None
    device: str = "cpu"

    def predict(self, x: FloatArray, rpm: FloatArray, fs: float,
                target_indices: np.ndarray | None = None) -> FloatArray:
        """Return predictions ordered by target_indices, or length-N with an unscored prefix."""
        if len(x) != len(rpm) or not np.isclose(fs, self.fs):
            raise ValueError("Record length and sampling rate must match the trained input convention.")
        if target_indices is None:
            y = np.full(len(x), np.nan)
            idx = np.arange(START, len(x))
            y[idx] = self.predict(x, rpm, fs, idx)
            return y
        idx = np.asarray(target_indices, dtype=int)
        minimum = (1 if self.config.family == "persistence" else
                   ANGLE_HISTORY if self.config.angular else
                   self.module.history if self.config.family == "tcn" else
                   max(self.config.lag_count, 96 if self.config.orders else 33))
        if np.any(idx < minimum) or np.any(idx >= len(x)):
            raise ValueError(f"Target indices require at least {minimum} history samples.")
        if self.config.family == "persistence":
            return x[idx - 1].copy()
        if self.config.family == "tcn":
            with torch.inference_mode():
                values = _tcn_input(x, rpm, self.location, self.scale, bool(self.config.rpm_degree), self.device)
                return self.module(values).cpu().numpy()[idx].astype(np.float64)
        matrix = feature_matrix(x, rpm, fs, idx, self.config)
        matrix = (matrix - self.location) / self.scale
        if self.config.family == "mlp":
            with torch.inference_mode():
                inputs = torch.as_tensor(matrix, device=self.device, dtype=torch.float32)
                return self.module(inputs).cpu().numpy().astype(np.float64)
        return matrix @ self.coefficient + self.intercept

    def save(self, path: Path) -> None:
        """Save portable non-pickle arrays. Neural tensors are stored by state key."""
        arrays = dict(config=np.array(json.dumps(asdict(self.config))),
                      location=self.location, scale=self.scale, fs=np.array(self.fs),
                      intercept=np.array(self.intercept))
        if self.coefficient is not None:
            arrays["coefficient"] = self.coefficient
        if self.module is not None:
            arrays.update({f"tensor.{key}": value.detach().cpu().numpy()
                           for key, value in self.module.state_dict().items()})
        np.savez_compressed(path, **arrays)


def load_comparator(path: Path, device: str = "cpu") -> Comparator:
    """Restore a frozen comparator with no pickled objects."""
    with np.load(path, allow_pickle=False) as arrays:
        config = Config(**json.loads(arrays["config"].item()))
        location, scale = arrays["location"], arrays["scale"]
        model = Comparator(config, location, scale, float(arrays["fs"]),
                           arrays["coefficient"] if "coefficient" in arrays else None,
                           float(arrays["intercept"]), device=device)
        if config.family == "mlp":
            model.module = FeedForward(len(location), config.width)
        elif config.family == "tcn":
            model.module = CausalTCN(1 + bool(config.rpm_degree), config)
        if model.module is not None:
            state = {name[7:]: torch.from_numpy(arrays[name]) for name in arrays if name.startswith("tensor.")}
            model.module.load_state_dict(state)
            model.module.to(device).eval()
    return model


def _fit_mlp(model: Comparator, z: FloatArray, y: FloatArray, seed: int) -> list[float]:
    config, device = model.config, model.device
    torch.manual_seed(seed)
    module = FeedForward(z.shape[1], config.width).to(device)
    coef, intercept = _linear_parameters(z, y, config.alpha)
    with torch.no_grad():
        module.skip.weight.copy_(torch.as_tensor(coef[None, :]))
        module.skip.bias.fill_(intercept)
    module.skip.requires_grad_(False)
    optimizer = torch.optim.AdamW(module.nonlinear.parameters(), lr=config.learning_rate, weight_decay=0.015)
    values = torch.as_tensor(z, device=device, dtype=torch.float32)
    desired = torch.as_tensor(y, device=device, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    trace = []
    for _ in range(config.epochs):
        minibatches = np.array_split(rng.permutation(len(y)), max(1, len(y) // 256))
        total_loss = 0.0
        for batch in minibatches:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(module(values[batch]), desired[batch])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        trace.append(total_loss / len(y))
    model.module = module.eval()
    return trace


def _fit_tcn(model: Comparator, x: FloatArray, rpm: FloatArray, targets: np.ndarray,
             seed: int) -> list[float]:
    config, device = model.config, model.device
    torch.manual_seed(seed)
    module = CausalTCN(1 + bool(config.rpm_degree), config).to(device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=config.learning_rate, weight_decay=0.012)
    values = _tcn_input(x, rpm, model.location, model.scale, bool(config.rpm_degree), device)
    desired = torch.as_tensor(x, device=device, dtype=torch.float32)
    runs = np.split(targets, np.flatnonzero(np.diff(targets) != 1) + 1)
    contexts = [(int(chunk[0]) - module.history + 1, int(chunk[0]), int(chunk[-1]) + 1)
                for run in runs for chunk in np.array_split(run, max(1, (len(run) + 1023) // 1024)) if len(chunk)]
    rng = np.random.default_rng(seed)
    trace = []
    for _ in range(config.epochs):
        total_loss = 0.0
        for row in rng.permutation(len(contexts)):
            context, first, stop = contexts[row]
            optimizer.zero_grad(set_to_none=True)
            yhat = module(values[:, :, context:stop])[first - context:]
            loss = nn.functional.mse_loss(yhat, desired[first:stop])
            loss.backward()
            nn.utils.clip_grad_norm_(module.parameters(), 2.0)
            optimizer.step()
            total_loss += loss.item() * (stop - first)
        trace.append(total_loss / len(targets))
    model.module = module.eval()
    return trace


def fit(x: FloatArray, rpm: FloatArray, fs: float, config: Config,
        targets: np.ndarray, *, device: str = "cpu", seed: int = SEED,
        matrix: FloatArray | None = None) -> tuple[Comparator, list[float]]:
    """Fit each scaler and all coefficients on the permitted training targets."""
    model = Comparator(config, np.empty(0), np.empty(0), fs, device=device)
    if config.family == "persistence":
        return model, []
    if config.family == "tcn":
        model.location, model.scale = _speed_scale(x, rpm, targets)
        trace = _fit_tcn(model, x, rpm, targets, seed)
    else:
        a = feature_matrix(x, rpm, fs, targets, config) if matrix is None else matrix
        model.location = a.mean(0)
        model.scale = np.maximum(a.std(0), 1e-7)
        z = (a - model.location) / model.scale
        if config.family == "mlp":
            trace = _fit_mlp(model, z, x[targets], seed)
        else:
            penalties = np.full(z.shape[1], config.alpha)
            if config.polynomial_degree:
                nonlinear_columns = 36
                if config.polynomial_degree >= 3:
                    nonlinear_columns += 8 + (16 if config.orders else 0)
                penalties[-nonlinear_columns:] *= config.nonlinear_penalty
            model.coefficient, model.intercept = _linear_parameters(z, x[targets], penalties)
            trace = []
    return model, trace


def candidates(profile: str) -> dict[str, list[Config]]:
    """Finite, declared comparison searches without accessing development validation."""
    grid = {
        "persistence": [Config("persistence")],
        "ridge_ar": [Config("ridge", lag_count=lag, alpha=alpha)
                     for lag in (48, 128, 240) for alpha in (1.0, 30.0)],
        "ridge_lpv": [Config("ridge", lag_count=lag, rpm_degree=degree, alpha=alpha)
                      for lag in (64, 160, 240) for degree in (1, 2, 4) for alpha in (1.0, 30.0)],
        "physics_linear": [Config("ridge", lag_count=lag, rpm_degree=2, angular=True,
                                  orders=orders, alpha=alpha)
                           for lag in (80, 160) for orders in (False, True) for alpha in (2.0, 40.0)],
        "ngrc": [Config("ridge", lag_count=lag, rpm_degree=2, angular=angle,
                        orders=angle, polynomial_degree=degree, alpha=alpha, nonlinear_penalty=penalty)
                 for lag in (96, 160) for angle in (False, True)
                 for degree in (2, 3) for alpha, penalty in ((5.0, 1.0), (60.0, 1.0), (20.0, 5.0))],
        "mlp_vibration": [Config("mlp", lag_count=lag, width=width, epochs=epochs)
                          for lag, width, epochs in ((64, 24, 10), (128, 32, 14), (240, 32, 14))],
        "mlp_rpm": [Config("mlp", lag_count=lag, rpm_degree=1, width=width, epochs=epochs)
                    for lag, width, epochs in ((64, 24, 10), (128, 32, 14), (240, 32, 14))],
        "tcn_vibration": [Config("tcn", kernel_size=kernel, width=width, epochs=epochs)
                          for kernel, width, epochs in ((3, 16, 18), (5, 16, 22), (3, 24, 22))],
        "tcn_rpm": [Config("tcn", rpm_degree=1, kernel_size=kernel, width=width, epochs=epochs)
                    for kernel, width, epochs in ((3, 16, 18), (5, 16, 22), (3, 24, 22))],
    }
    if profile == "smoke":
        return {name: [replace(configs[0], epochs=2)] for name, configs in grid.items()}
    return grid


def _mse(x: FloatArray, y: FloatArray) -> float:
    return float(np.mean((x - y) ** 2))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def run(train, valid, run_dir: Path, profile: str = "full", device: str = "cpu",
        threads: int = 1) -> list[dict]:
    """Run the finite comparator search, refit on sigtrain, and emit audit artifacts."""
    torch.set_num_threads(threads)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch reports it unavailable.")
    for subdir in ("metrics", "predictions", "checkpoints", "search", "tables"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    folds = fold_targets(len(train.x))
    train_idx = np.arange(START, len(train.x))
    valid_idx = np.arange(START, len(valid.x))
    all_scores, trials, curves, timing, predictions = [], [], [], [], {}
    search_start = perf_counter()
    with threadpool_limits(limits=threads):
        cached_key = None
        for slug, configs in candidates(profile).items():
            means = []
            for trial, config in enumerate(configs):
                key = replace(config, alpha=0.0, nonlinear_penalty=0.0)
                if config.family not in ("tcn", "persistence") and cached_key != key:
                    matrix = feature_matrix(train.x, train.rpm, train.fs, train_idx, config)
                    cached_key = key
                losses = []
                for fold, (fit_rows, eval_rows) in enumerate(folds, 1):
                    start = perf_counter()
                    fold_matrix = None if config.family in ("tcn", "persistence") else matrix[fit_rows - START]
                    fit_model, trace = fit(train.x, train.rpm, train.fs, config,
                                           fit_rows, device=device, matrix=fold_matrix)
                    loss = _mse(train.x[eval_rows], fit_model.predict(train.x, train.rpm, train.fs, eval_rows))
                    elapsed = perf_counter() - start
                    trials.append(dict(model=slug, trial=trial, fold=fold,
                                       train_targets=len(fit_rows), evaluation_targets=len(eval_rows),
                                       mse=loss, seconds=elapsed, seed=SEED,
                                       config=json.dumps(asdict(config), sort_keys=True)))
                    for epoch, online_loss in enumerate(trace, 1):
                        curves.append(dict(model=slug, trial=trial, fold=fold,
                                           epoch=epoch, online_training_mse=online_loss))
                    losses.append(loss)
                means.append(float(np.mean(losses)))
            best = int(np.argmin(means))
            chosen = configs[best]
            _sync(device)
            start = perf_counter()
            model, trace = fit(train.x, train.rpm, train.fs, chosen, train_idx, device=device)
            _sync(device)
            fit_seconds = perf_counter() - start
            for epoch, online_loss in enumerate(trace, 1):
                curves.append(dict(model=slug, trial=best, fold=0, epoch=epoch,
                                   online_training_mse=online_loss))
            neural_runtime = device if model.module is not None else "cpu"
            checkpoint = run_dir / "checkpoints" / f"comparator_{slug}.npz"
            model.save(checkpoint)
            training_prediction = model.predict(train.x, train.rpm, train.fs, train_idx)
            valid_prediction = model.predict(valid.x, valid.rpm, valid.fs)
            replay_model = load_comparator(checkpoint, device=device)
            replay = replay_model.predict(valid.x, valid.rpm, valid.fs, valid_idx)
            difference = float(np.max(np.abs(replay - valid_prediction[valid_idx])))
            if difference > 1e-6:
                raise RuntimeError(f"{slug} replay maximum absolute deviation was {difference:g}")
            predictions[slug] = valid_prediction
            for repeat in range(7):
                _sync(device)
                start = perf_counter()
                replay_model.predict(valid.x, valid.rpm, valid.fs, valid_idx)
                _sync(device)
                timing.append(dict(model=slug, repeat=repeat + 1, inference_seconds=perf_counter() - start,
                                   n_targets=len(valid_idx), device=neural_runtime, threads=threads,
                                   precision="float32" if model.module is not None else "float64"))
            elapsed_list = [r["inference_seconds"] for r in timing if r["model"] == slug]
            param_count = (sum(p.numel() for p in model.module.parameters()) if model.module is not None
                           else 0 if model.coefficient is None else len(model.coefficient) + 1)
            error = valid.x[valid_idx] - valid_prediction[valid_idx]
            all_scores.append(dict(model=slug, validation_mse=float(np.mean(error ** 2)),
                                   train_mse=_mse(train.x[train_idx], training_prediction),
                                   fold_mean_mse=means[best], validation_rmse=float(np.sqrt(np.mean(error ** 2))),
                                   validation_mae=float(np.mean(np.abs(error))), validation_bias=float(np.mean(error)),
                                   selected_trial=best, n_targets=len(valid_idx), first_target=int(valid_idx[0]),
                                   fit_seconds=fit_seconds, median_inference_seconds=float(np.median(elapsed_list)),
                                   inference_iqr_seconds=float(np.percentile(elapsed_list, 75) - np.percentile(elapsed_list, 25)),
                                   inference_p95_seconds=float(np.percentile(elapsed_list, 95)),
                                   checkpoint_bytes=checkpoint.stat().st_size, parameters=param_count,
                                   replay_max_abs_difference=difference, device=neural_runtime,
                                   config=json.dumps(asdict(chosen), sort_keys=True)))
            _write_csv(run_dir / "search" / "comparators_folds.csv", trials)
            if curves:
                _write_csv(run_dir / "search" / "comparators_learning_curves.csv", curves)
            _write_csv(run_dir / "metrics" / "comparators.csv", all_scores)
            _write_csv(run_dir / "metrics" / "comparator_timing.csv", timing)
            np.savez_compressed(run_dir / "predictions" / "comparators.npz", **predictions)
            print(f"{slug}  fold MSE {means[best]:.6f}  development MSE {all_scores[-1]['validation_mse']:.6f}", flush=True)
    (run_dir / "search" / "comparator_run.json").write_text(json.dumps(dict(
        profile=profile, duration_seconds=perf_counter() - search_start, trials=sum(len(x) for x in candidates(profile).values()),
        fold_fits=len(trials), threads=threads, device=device, torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(), seed=SEED,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        folds=FOLD_RANGES, target_start=START, guard=START,
    ), indent=2))
    return all_scores


def main() -> None:
    from .data import load_record

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    run(load_record(args.data_dir / "sigtrain.mat"), load_record(args.data_dir / "sigvalid.mat"),
        args.run_dir, args.profile, args.device, args.threads)


if __name__ == "__main__":
    main()
