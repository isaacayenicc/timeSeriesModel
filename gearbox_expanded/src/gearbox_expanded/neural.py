"""Guarded neural comparators with finite, strict-past forecasting windows."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from threadpoolctl import threadpool_limits
from torch import nn
from torch.nn.functional import pad

from .protocol import (
    FAMILIES as PROTOCOL_FAMILIES, FOLDS, SEED, START, guarded_segments,
    load_data, mse, predict, timed_predict, write_json, write_table,
)

FAMILIES = PROTOCOL_FAMILIES["neural"]
PATCH = 8
BATCH = 512
PREDICT_BATCH = 1024
TIMING_REPEATS = 5


@dataclass(frozen=True)
class Config:
    family: str
    history: int
    width: int
    epochs: int
    rate: float

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"Invalid neural family {self.family!r}")
        if not PATCH <= self.history <= START or self.history % PATCH:
            raise ValueError("Neural history must be a multiple of the temporal patch size")
        if self.width < 1 or self.epochs < 1 or not 0 < self.rate <= 0.1:
            raise ValueError("Invalid network width, epoch count, or learning rate")


def candidates(family: str) -> tuple[Config, ...]:
    return (
        Config(family, 64, 16, 8, 0.002),
        Config(family, 128, 24, 12, 0.002),
        Config(family, 192, 32, 16, 0.0015),
    )


class LocalConvolutions(nn.Module):
    """Ordinary CNN over consecutive patches of the finished historical window."""

    def __init__(self, steps: int, width: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(2 * PATCH, width, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(width * steps, 1),
        )

    def forward(self, patches):
        return self.network(patches.transpose(1, 2))


class DilatedConvolutions(nn.Module):
    """Left-padded temporal residual blocks ending at the latest prior patch."""

    def __init__(self, steps: int, width: int):
        super().__init__()
        self.projection = nn.Conv1d(2 * PATCH, width, 1)
        layer_count = (steps - 1).bit_length()
        self.dilations = tuple(2 ** i for i in range(layer_count))
        self.blocks = nn.ModuleList(
            nn.Conv1d(width, width, 2, dilation=dilation) for dilation in self.dilations
        )
        self.activation = nn.GELU()
        self.output = nn.Linear(width, 1)

    def forward(self, patches):
        encoded = self.projection(patches.transpose(1, 2))
        for dilation, block in zip(self.dilations, self.blocks, strict=True):
            encoded = encoded + self.activation(block(pad(encoded, (dilation, 0))))
        return self.output(encoded[..., -1])


class NeuralForecaster(nn.Module):
    """Shared affine shortcut plus one of five distinct nonlinear readouts."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.steps = config.history // PATCH
        self.shortcut = nn.Linear(config.history * 2, 1)
        if config.family == "mlp":
            self.network = nn.Sequential(nn.Linear(config.history * 2, config.width), nn.GELU(),
                                         nn.Linear(config.width, config.width), nn.GELU(),
                                         nn.Linear(config.width, 1))
        elif config.family == "lstm":
            self.network = nn.LSTM(2 * PATCH, config.width, batch_first=True)
            self.head = nn.Linear(config.width, 1)
        elif config.family == "gru":
            self.network = nn.GRU(2 * PATCH, config.width, batch_first=True)
            self.head = nn.Linear(config.width, 1)
        elif config.family == "cnn":
            self.network = LocalConvolutions(self.steps, config.width)
        else:
            self.network = DilatedConvolutions(self.steps, config.width)

    def forward(self, windows):
        flat = windows.flatten(1)
        if self.config.family == "mlp":
            nonlinear = self.network(flat)
        else:
            patches = windows.reshape(windows.shape[0], self.steps, 2 * PATCH)
            if self.config.family in {"lstm", "gru"}:
                encoded, _ = self.network(patches)
                nonlinear = self.head(encoded[:, -1, :])
            else:
                nonlinear = self.network(patches)
        return (self.shortcut(flat) + nonlinear).squeeze(-1)


def _windows(x, rpm, targets, history, center, scale):
    channels = (np.column_stack((x, rpm)) - center) / scale
    lag_view = sliding_window_view(channels, history, axis=0).transpose(0, 2, 1)
    return torch.from_numpy(np.ascontiguousarray(lag_view[targets - history], dtype=np.float32))


@dataclass
class Model:
    config: Config
    center: np.ndarray
    scale: np.ndarray
    fs: float
    network: NeuralForecaster

    def predict(self, x, rpm, fs, targets=None):
        x = np.asarray(x)
        rpm = np.asarray(rpm)
        if x.ndim != 1 or x.shape != rpm.shape or not np.isclose(self.fs, fs):
            raise ValueError("Incompatible record channels or sample rate")
        if targets is None:
            forecasts = np.full(x.size, np.nan)
            indices = np.arange(START, x.size)
            forecasts[indices] = self.predict(x, rpm, fs, indices)
            return forecasts
        indices = np.asarray(targets, dtype=int)
        if np.any(indices < self.config.history) or np.any(indices >= x.size):
            raise ValueError("Target falls outside available history")
        if not indices.size:
            return np.empty(0, dtype=np.float64)
        inputs = _windows(x, rpm, indices, self.config.history, self.center, self.scale)
        with torch.inference_mode():
            prediction = torch.cat([self.network(group) for group in inputs.split(PREDICT_BATCH)])
        return prediction.numpy().astype(np.float64)

    @property
    def parameters(self):
        return sum(parameter.numel() for parameter in self.network.parameters())

    def save(self, path: Path):
        weight_arrays = {f"weights.{name}": value.detach().cpu().numpy()
                         for name, value in self.network.state_dict().items()}
        np.savez_compressed(path, **weight_arrays, config=np.array(json.dumps(asdict(self.config))),
                            center=self.center, scale=self.scale, fs=np.array(self.fs))


def load(path: Path) -> Model:
    """Restore a CPU one-step model from typed arrays with safe NumPy loading."""
    with np.load(path, allow_pickle=False) as archive:
        config = Config(**json.loads(archive["config"].item()))
        network = NeuralForecaster(config)
        restored = {name.removeprefix("weights."): torch.from_numpy(archive[name])
                    for name in archive.files if name.startswith("weights.")}
        network.load_state_dict(restored)
        return Model(config, archive["center"], archive["scale"], float(archive["fs"].item()),
                     network.eval())


def _fit(segments, fs: float, config: Config, seed: int):
    """Fit on independent input slices, with reset windows and fit-only scaling."""
    history = config.history
    valid = [(x, rpm) for x, rpm in segments if len(x) > history]
    if not valid:
        raise ValueError("Training segments have no targets after context warm-up")
    pooled = np.concatenate([np.column_stack((x, rpm)) for x, rpm in valid])
    center = pooled.mean(axis=0)
    scale = np.maximum(pooled.std(axis=0), np.finfo(float).eps)
    inputs = torch.cat([_windows(x, rpm, np.arange(history, len(x)),
                                history, center, scale) for x, rpm in valid])
    labels = torch.from_numpy(np.concatenate([x[history:] for x, _ in valid]).astype(np.float32))
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    network = NeuralForecaster(config)
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.rate, weight_decay=0.003)
    trace = []
    for epoch in range(config.epochs):
        randomized = torch.randperm(len(labels), generator=generator)
        cumulative = 0.0
        for batch in randomized.split(BATCH):
            optimizer.zero_grad()
            loss = (network(inputs[batch]) - labels[batch]).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            cumulative += float(loss.detach()) * len(batch)
        trace.append({"epoch": epoch + 1, "online_mse": cumulative / len(labels)})
    return Model(config, center, scale, fs, network.eval()), trace, len(labels)


def _causality(model, x, rpm, fs, base):
    cutoff = len(x) // 2
    changed_x, changed_rpm = x.copy(), rpm.copy()
    changed_x[cutoff:] += 11.0
    changed_rpm[cutoff:] *= 0.7
    alternate = model.predict(changed_x, changed_rpm, fs)
    before = slice(START, cutoff + 1)
    difference = float(np.max(np.abs(base[before] - alternate[before])))
    if difference > 1e-7:
        raise RuntimeError(f"Current or future sample affects neural prediction ({difference:.3g})")
    return difference


def run(data_dir: Path, out: Path, prior: Path):
    """Run all five families. The common entry point accepts an unused prior path."""
    out = Path(out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    train, validation = load_data(data_dir)
    torch.set_num_threads(1)
    rows, curves, selection, metrics, prediction_arrays = [], [], {}, [], {}
    with threadpool_limits(limits=1):
        for family in FAMILIES:
            scores = []
            options = candidates(family)
            for trial, config in enumerate(options, 1):
                fold_scores = []
                for fold, (start, stop) in enumerate(FOLDS, 1):
                    begin = perf_counter()
                    model, trace, n_fit = _fit(guarded_segments(train, start, stop),
                                               train.fs, config, seed=SEED + fold)
                    predictions = predict(model, train, stop=stop)
                    fold_mse = mse(train.x[start:stop], predictions[start:stop])
                    rows.append(dict(model=family, trial=trial, fold=fold,
                                     start=start, stop=stop, guard=START, n_fit=n_fit,
                                     count=stop - start, mse=fold_mse, seed=SEED + fold,
                                     seconds=perf_counter()-begin, **asdict(config)))
                    curves.extend(dict(model=family, trial=trial, fold=fold, **row) for row in trace)
                    fold_scores.append(fold_mse)
                scores.append(float(np.mean(fold_scores)))
                print(f"Neural {family} trial {trial} mean guarded MSE {scores[-1]:.6f}", flush=True)
                write_table(out / "neural_trials.csv", rows)
            best = min(range(len(scores)), key=scores.__getitem__)
            selection[family] = dict(trial=best + 1, config=asdict(options[best]), fold_mean_mse=scores[best])
        write_json(out / "neural_selection.json", selection)
        for row in rows:
            row["selected"] = selection[row["model"]]["trial"] == row["trial"]
        write_table(out / "neural_trials.csv", rows)
        for family, result in selection.items():
            config = Config(**result["config"])
            begin = perf_counter()
            model, trace, fit_rows = _fit([(train.x, train.rpm)], train.fs, config, seed=SEED)
            fit_s = perf_counter() - begin
            curves.extend(dict(model=family, trial=result["trial"], fold=0, **row) for row in trace)
            path = out / "models" / f"{family}.npz"
            model.save(path)
            fitted = predict(model, validation)
            replay = load(path)
            final, timing = timed_predict(replay, validation, repeats=TIMING_REPEATS)
            replay_delta = float(np.max(np.abs(final[START:] - fitted[START:])))
            if replay_delta > 1e-7:
                raise RuntimeError(f"Checkpoint replay mismatch for {family}: {replay_delta}")
            causal_delta = _causality(replay, validation.x, validation.rpm, validation.fs, final)
            train_prediction = predict(replay, train)
            prediction_arrays[family] = final
            prediction_arrays[f"{family}__train"] = train_prediction
            metrics.append(dict(
                model=family, fit_s=fit_s, **timing,
                parameters=replay.parameters,
                checkpoint_bytes=path.stat().st_size, selection=json.dumps(result, sort_keys=True),
                device="cpu", threads=1, repeats=TIMING_REPEATS, raw_history=config.history,
                train_mse=mse(train.x[START:], train_prediction[START:]),
                validation_mse=mse(validation.x[START:], final[START:]),
                fit_targets=fit_rows, scored_targets=len(final) - START,
                checkpoint_replay_max_abs=replay_delta, causal_delta=causal_delta,
                selected_candidate=result["trial"], fold_mean_mse=result["fold_mean_mse"],
            ))
            print(f"Neural {family} score {metrics[-1]['validation_mse']:.6f}", flush=True)
        write_table(out / "neural_metrics.csv", metrics)
        write_table(out / "neural_learning.csv", curves)
        np.savez_compressed(out / "neural_predictions.npz", **prediction_arrays)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--prior", required=True, type=Path)
    args = parser.parse_args()
    run(args.data_dir, args.out, args.prior)


if __name__ == "__main__":
    main()
