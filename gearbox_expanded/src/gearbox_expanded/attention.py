"""Causal patch Transformer and single-horizon Temporal Fusion Transformer."""

from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
import torch
from torch import nn

from .protocol import SEED, START, load_data, run_candidates, write_table
MODEL_IDS = ("transformer", "tft")


class TimeSeriesTransformer(nn.Module):
    """Causal forecasting encoder with non-overlapping historical patches.

    Input shape is [batch, history, 2], ordered oldest first. The final encoded
    patch contains the last observed measurement at t-1.
    """

    def __init__(self, history: int, width: int, heads: int, patch: int, layers: int,
                 dropout: float):
        super().__init__()
        if history % patch or width % heads or width % 2:
            raise ValueError("History must be divisible by patch and width by heads and two")
        self.history = history
        self.patch = patch
        self.projection = nn.Linear(2 * patch, width)
        count = history // patch
        positions = torch.arange(count, dtype=torch.float32)[:, None]
        rates = torch.exp(-math.log(1e4) * torch.arange(0, width, 2) / width)
        position = torch.empty(count, width)
        position[:, 0::2] = torch.sin(positions * rates)
        position[:, 1::2] = torch.cos(positions * rates)
        self.register_buffer("position", position, persistent=False)
        self.register_buffer("mask", torch.ones(count, count, dtype=torch.bool).triu(1),
                             persistent=False)
        block = nn.TransformerEncoderLayer(width, heads, dim_feedforward=2 * width,
                                            dropout=dropout, activation="gelu",
                                            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.reshape(x.shape[0], self.history // self.patch, 2 * self.patch)
        token = self.projection(patches) + self.position
        encoded = self.encoder(token, mask=self.mask, is_causal=True)
        return self.output(encoded[:, -1, :]).squeeze(-1)


class GatedAddNorm(nn.Module):
    """GLU over the main branch, residual addition and per-vector normalization."""

    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(width, 2 * width)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(width)

    def forward(self, data: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        value, gate = self.linear(self.dropout(data)).chunk(2, dim=-1)
        return self.norm(residual + value * torch.sigmoid(gate))


class GatedResidualNetwork(nn.Module):
    """ELU transform with GLU, skip projection and layer normalization."""

    def __init__(self, width_in: int, width_hidden: int, width_out: int, dropout: float):
        super().__init__()
        self.main = nn.Sequential(nn.Linear(width_in, width_hidden), nn.ELU(),
                                  nn.Linear(width_hidden, width_out))
        self.skip = nn.Identity() if width_in == width_out else nn.Linear(width_in, width_out)
        self.gate = GatedAddNorm(width_out, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(self.main(x), self.skip(x))


class VariableSelection(nn.Module):
    """Variable-wise transformations with context-dependent normalized selection weights."""

    def __init__(self, width: int, dropout: float, channels: int = 2):
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Linear(1, width) for _ in range(channels))
        self.transforms = nn.ModuleList(
            GatedResidualNetwork(width, width, width, dropout) for _ in range(channels)
        )
        self.weights = nn.Sequential(GatedResidualNetwork(channels * width, width, width, dropout),
                                     nn.Linear(width, channels))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = [layer(x[..., j:j+1]) for j, layer in enumerate(self.embeddings)]
        weights = self.weights(torch.cat(embeddings, dim=-1)).softmax(dim=-1)
        values = torch.stack([layer(z) for layer, z in zip(self.transforms, embeddings)], dim=-2)
        selected = (weights[..., None] * values).sum(dim=-2)
        return selected, weights


class SharedValueAttention(nn.Module):
    """Interpretable TFT attention with one final query and shared values."""

    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        if width % heads:
            raise ValueError("Width must be divisible by the number of heads")
        self.heads = heads
        self.depth = width // heads
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, self.depth, bias=False)
        self.project = nn.Linear(self.depth, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, _ = x.shape
        query = self.query(x[:, -1]).reshape(batch, self.heads, self.depth)
        keys = self.key(x).reshape(batch, length, self.heads, self.depth).transpose(1, 2)
        score = (keys * query[:, :, None, :]).sum(dim=-1) / math.sqrt(self.depth)
        averaged_weights = score.softmax(dim=-1).mean(dim=1)
        weights = self.dropout(averaged_weights)
        values = self.value(x)
        attended = self.project((weights[..., None] * values).sum(dim=1))
        return attended, averaged_weights


class TemporalFusionTransformer(nn.Module):
    """History-only point forecast with variable selection and gated temporal fusion.

    Static covariates, future covariates and a decoder are absent in this
    adaptation. Recurrent state is reset for each bounded window.
    """

    def __init__(self, history: int, width: int, heads: int, layers: int, dropout: float):
        super().__init__()
        self.history = history
        self.selection = VariableSelection(width, dropout)
        self.recurrent = nn.LSTM(width, width, num_layers=layers, batch_first=True,
                                 dropout=dropout if layers > 1 else 0.)
        self.local_fusion = GatedAddNorm(width, dropout)
        self.enrichment = GatedResidualNetwork(width, width, width, dropout)
        self.attention = SharedValueAttention(width, heads, dropout)
        self.attention_fusion = GatedAddNorm(width, dropout)
        self.after_attention = GatedResidualNetwork(width, width, width, dropout)
        self.output_fusion = GatedAddNorm(width, dropout)
        self.readout = nn.Linear(width, 1)

    def inspect(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return prediction, variable weights [B,T,2] and averaged attention [B,T]."""
        selected, variable_weights = self.selection(x)
        recurrent, _ = self.recurrent(selected)
        local = self.local_fusion(recurrent, selected)
        enriched = self.enrichment(local)
        attended, attention_weights = self.attention(enriched)
        fused = self.attention_fusion(attended, enriched[:, -1])
        hidden = self.output_fusion(self.after_attention(fused), local[:, -1])
        output = self.readout(hidden).squeeze(-1)
        return output, variable_weights, attention_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prediction, _, _ = self.inspect(x)
        return prediction


@dataclass(frozen=True)
class Config:
    """Fixed training budget and encoder geometry declared per candidate."""

    family: str
    history: int
    width: int
    heads: int
    layers: int
    epochs: int
    learning_rate: float
    patch: int = 1
    dropout: float = .04
    batch: int = 256
    stride: int = 2

    def build(self) -> nn.Module:
        if self.family == "transformer":
            return TimeSeriesTransformer(self.history, self.width, self.heads,
                                         self.patch, self.layers, self.dropout)
        if self.family == "tft":
            return TemporalFusionTransformer(self.history, self.width, self.heads,
                                             self.layers, self.dropout)
        raise ValueError(f"Unknown attention family {self.family!r}")


def _windows(x: np.ndarray, rpm: np.ndarray, center: np.ndarray,
             scale: np.ndarray, history: int) -> np.ndarray:
    channels = ((np.column_stack([x, rpm]) - center) / scale).astype(np.float32)
    # A window ending at index t-1 predicts target t. The last window would
    # predict beyond the available target array and is dropped.
    return np.lib.stride_tricks.sliding_window_view(channels, history, axis=0).transpose(0, 2, 1)[:-1]


@dataclass
class Predictor:
    config: Config
    model: nn.Module
    center: np.ndarray
    scale: np.ndarray
    fs: float
    learning: list[float]

    @property
    def parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @torch.inference_mode()
    def predict(self, x: np.ndarray, rpm: np.ndarray, fs: float) -> np.ndarray:
        if np.shape(x) != np.shape(rpm) or not np.isclose(fs, self.fs):
            raise ValueError("Expected aligned vibration and RPM at the fitted sampling rate")
        output = np.full(len(x), np.nan)
        if len(x) <= START:
            return output
        z = _windows(x, rpm, self.center, self.scale, self.config.history)
        for a in range(START, len(x), self.config.batch):
            b = min(a + self.config.batch, len(x))
            chunk = z[a-self.config.history:b-self.config.history]
            inputs = torch.from_numpy(np.array(chunk, copy=True, order="C"))
            output[a:b] = self.model(inputs).numpy()
        return output

    @torch.inference_mode()
    def tft_weights(self, x: np.ndarray, rpm: np.ndarray, target_indices: np.ndarray) -> tuple:
        """Summarize two variable weights and history attention over specified TFT windows."""
        if not isinstance(self.model, TemporalFusionTransformer):
            raise ValueError("Selection and shared-value weights apply to TFT")
        windows = _windows(x, rpm, self.center, self.scale, self.config.history)
        examples = windows[target_indices-self.config.history]
        _, variable, attention = self.model.inspect(torch.from_numpy(np.ascontiguousarray(examples)))
        return variable.numpy(), attention.numpy()

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path, config=json.dumps(asdict(self.config)), center=self.center,
            scale=self.scale, fs=self.fs, learning=np.asarray(self.learning),
            **{f"state.{key}": val.detach().cpu().numpy() for key, val in self.model.state_dict().items()}
        )


def load(path: Path) -> Predictor:
    """Use only serialized numeric arrays and JSON to reconstruct a CPU model."""
    with np.load(path, allow_pickle=False) as arrays:
        config = Config(**json.loads(arrays["config"].item()))
        model = config.build()
        model.load_state_dict({key[6:]: torch.from_numpy(arrays[key])
                               for key in arrays if key.startswith("state.")})
        return Predictor(config, model.eval(), arrays["center"], arrays["scale"],
                         float(arrays["fs"]), arrays["learning"].tolist())


def fit(spec: dict, segments: list[tuple[np.ndarray, np.ndarray]], fs: float) -> Predictor:
    """Create windows separately inside each fit block and train with next-sample MSE."""
    config = Config(**spec)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = config.build()
    usable = [(x, n) for x, n in segments if len(x) > config.history]
    joined = np.concatenate([np.column_stack([x, rpm]) for x, rpm in usable], axis=0)
    center = joined.mean(axis=0)
    scale = np.maximum(joined.std(axis=0), 1e-7)
    # Materialize only the fixed-stride training rows inside each permitted block.
    X = torch.from_numpy(np.concatenate([
        _windows(x, rpm, center, scale, config.history)[::config.stride] for x, rpm in usable
    ], axis=0))
    y = torch.from_numpy(np.concatenate([
        x[config.history::config.stride] for x, _ in usable
    ]).astype(np.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.learning_rate * .15
    )
    history = []
    for _ in range(config.epochs):
        total = 0.
        permutation = rng.permutation(len(y))
        for a in range(0, len(y), config.batch):
            batch = permutation[a:a+config.batch]
            estimate = model(X[batch])
            error = nn.functional.mse_loss(estimate, y[batch])
            optimizer.zero_grad(set_to_none=True)
            error.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += error.item() * len(batch)
        scheduler.step()
        history.append(total / len(y))
    return Predictor(config, model.eval(), center, scale, fs, history)


def candidates(slug: str) -> list[dict]:
    """Two alternatives for each family on the official guarded training folds."""
    plans = {
        "transformer": [
            Config("transformer", 96, 16, 2, 1, 8, .003, patch=8),
            Config("transformer", 144, 24, 4, 2, 9, .002, patch=8),
        ],
        "tft": [
            Config("tft", 32, 10, 2, 1, 7, .003, stride=3),
            Config("tft", 56, 12, 3, 1, 8, .003, stride=3),
        ],
    }
    return [asdict(config) for config in plans[slug]]


def verify(model: Predictor, record) -> dict:
    x, rpm = record.x, record.rpm
    cutoff = 12000
    base = model.predict(x, rpm, record.fs)
    altered_x, altered_rpm = x.copy(), rpm.copy()
    altered_x[cutoff:] += 3
    altered_rpm[cutoff:] *= 1.04
    changed = model.predict(altered_x, altered_rpm, record.fs)
    repeat = model.predict(x, rpm, record.fs)
    errors = {
        "future_suffix_max_abs":float(np.max(np.abs(base[START:cutoff+1] - changed[START:cutoff+1]))),
        "repeated_record_max_abs":float(np.max(np.abs(base[START:] - repeat[START:]))),
    }
    if max(errors.values()) > 1e-6:
        raise RuntimeError(f"{model.config.family} causality check failed: {errors}")
    return errors


def run(data_dir: Path, out: Path, prior: Path) -> None:
    """Fit the attention families using the common run/metrics/checkpoint interface."""
    train, valid = load_data(data_dir)
    metrics, trials, verification = [], [], []
    predictions = {}
    torch.set_num_threads(1)
    with threadpool_limits(1):
        for slug in MODEL_IDS:
            print(f"Fitting {slug}", flush=True)
            row, folds, clean, training = run_candidates(
                slug, candidates(slug), train, valid, fit, out, loader=load
            )
            metrics.append(row)
            trials.extend(folds)
            predictions[slug] = clean
            predictions[f"{slug}__train"] = training
            model = load(out/"models"/f"{slug}.npz")
            verification.append({"model":slug, **verify(model, valid)})
            write_table(out/"attention_metrics.csv",metrics)
            write_table(out/"attention_trials.csv",trials)
            np.savez_compressed(out/"attention_predictions.npz",**predictions)
            if slug == "tft":
                idx = np.arange(START,len(valid.x),32)
                weights, attention = model.tft_weights(valid.x, valid.rpm, idx)
                if max(np.max(np.abs(weights.sum(-1)-1)),
                       np.max(np.abs(attention.sum(-1)-1))) > 1e-6:
                    raise ArithmeticError("TFT explanation weights do not sum to one")
                np.savez_compressed(out/"attention_tft_weights.npz", target_indices=idx,
                                    relative_lags=np.arange(-model.config.history,0),
                                    variable_selection=weights, attention=attention,
                                    variable_names=np.asarray(["vibration","rpm"]))
        write_table(out/"attention_verification.csv",verification)
