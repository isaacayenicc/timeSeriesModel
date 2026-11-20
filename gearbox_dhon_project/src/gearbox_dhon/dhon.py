"""Finite-context differentiable gearbox prediction with a strict lag-1 information set."""

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np
from scipy.linalg import solve
from scipy.signal import fftconvolve, lfilter
import torch
from torch import nn

BASE_HISTORY = 768
SCORE_START = 1792
MIN_RPM = 100.0


@dataclass(frozen=True)
class Config:
    """Discrete encoder choices and bounds for the physical parameters."""

    lags: int = 96
    angular_steps: int = 12
    mesh_cycles: int = 6
    harmonics: tuple = (1, 2, 4, 7)
    hz: tuple = ()
    taus: tuple = (16.0, 64.0, 192.0)
    memory: int = 384
    width: int = 12
    speed_degree: int = 2
    use_speed: bool = True
    preserve: float = 0.20
    augment: float = 0.35
    acceleration: bool = True
    innovation_order: int = 128
    innovation_lags: int = 16
    innovation_clip: float = 3.0
    ridge: float = 0.012
    epochs: int = 10
    lr: float = 0.00025
    physics_lr: float = 0.012
    joint: str = "both"
    prior_penalty: float = 0.002
    gate: float = 0.35
    adapter_gain: float = 0.5
    adapter_tau: float = 640.0
    adapter_memory: int = 1024
    adapter_clip: float = 1.75

    def validate(self, fs: float, rpm: np.ndarray) -> None:
        if fs <= 0 or np.any(rpm < MIN_RPM) or not np.isfinite(rpm).all():
            raise ValueError(f"DHON requires finite RPM >= {MIN_RPM:g} and positive fs")
        if self.joint not in {"fixed", "decay", "frequency", "both"}:
            raise ValueError(f"Unknown oscillator training mode: {self.joint}")
        if not (1 <= self.memory <= BASE_HISTORY and 1 <= self.lags <= BASE_HISTORY
                and self.innovation_order + self.innovation_lags <= BASE_HISTORY
                and self.adapter_memory <= SCORE_START - BASE_HISTORY):
            raise ValueError("DHON configuration exceeds finite-context policy")
        if self.mesh_cycles and (self.angular_steps <= 0 or
                self.mesh_cycles * 60 * fs / (37 * MIN_RPM) + 2 > BASE_HISTORY):
            raise ValueError("Angular history exceeds its minimum-RPM support bound")
        if (self.harmonics and (37 * max(self.harmonics) + 0.75) * rpm.max() / 60 >= .475 * fs
                or self.hz and max(self.hz) + 24.0 >= .475 * fs):
            raise ValueError("Oscillator bank violates the declared Nyquist margin")
        if len(self.taus) == 0 or not all(4 < tau < 384 for tau in self.taus):
            raise ValueError("Decay initializers must be in (4,384) samples")


def shaft_phase(rpm: np.ndarray, fs: float) -> np.ndarray:
    """Left-step integral in turns. Target phase depends on speed through t-1."""
    return np.r_[0., np.cumsum(rpm[:-1] / (60 * fs), dtype=np.float64)]


def _lags(x: np.ndarray, indices: np.ndarray, lags: int) -> np.ndarray:
    return x[indices[:, None] - np.arange(1, lags + 1)]


def angular_lags(x: np.ndarray, phase: np.ndarray, indices: np.ndarray,
                 angular_steps: int, mesh_cycles: int) -> np.ndarray:
    """Uniform mesh-cycle history ending before phase at the last observed sample."""
    offsets = np.arange(1, angular_steps * mesh_cycles + 1) / (37 * angular_steps)
    queries = phase[indices - 1, None] - offsets
    right = np.searchsorted(phase, queries, side="right")
    if right.size and (np.any(right >= indices[:, None]) or np.any(right < 1)):
        raise ValueError("Angular interpolation exceeded observed positive-time support")
    left = right - 1
    fraction = (queries - phase[left]) / (phase[right] - phase[left])
    return x[left] + fraction * (x[right] - x[left])


def ridge(x: np.ndarray, y: np.ndarray, penalty: float) -> np.ndarray:
    """Stable ridge solve with penalty relative to the number of training rows."""
    x = np.asarray(x, dtype=np.float64)
    matrix = x.T @ x
    matrix.flat[::matrix.shape[0] + 1] += penalty * len(y)
    return solve(matrix, x.T @ y.astype(np.float64), assume_a="pos")


def oscillator_states(x: torch.Tensor, phase: torch.Tensor, taus: torch.Tensor,
                      memory: int) -> torch.Tensor:
    """Quadrature by linear convolution, returning [time, oscillator, decay, real/imag].

    phase has shape [oscillator, time] in radians. Delaying the convolved envelope
    one sample is essential: the zero-delay coefficient weights x[t-1].
    """
    count = x.numel()
    nfft = 1 << (count + memory - 2).bit_length()
    j = torch.arange(memory, dtype=x.dtype, device=x.device)
    weights = torch.exp(-j / taus[..., None])
    weights = 2 * weights / weights.sum(dim=-1, keepdim=True)
    cosine = phase.cos().to(x.dtype)
    sine = phase.sin().to(x.dtype)
    carrier = torch.complex(cosine, sine)
    demodulated = x * carrier.conj()
    spectrum = torch.fft.fft(demodulated, n=nfft)
    filtered = torch.fft.ifft(
        spectrum[:, None] * torch.fft.fft(weights, n=nfft), n=nfft
    )[..., :count - 1]
    delayed = torch.cat([torch.zeros_like(filtered[..., :1]), filtered], dim=-1)
    return torch.view_as_real((delayed * carrier[:, None]).permute(2, 0, 1).contiguous())


def oscillator_recurrence(x: np.ndarray, phase: np.ndarray, taus: np.ndarray,
                          memory: int) -> np.ndarray:
    """Equivalent finite-memory recurrence in the fixed demodulation frame."""
    z = np.asarray(x)[None, None, :] * np.exp(-1j * phase[:, None, :])
    z = np.broadcast_to(z, (*taus.shape, len(x)))
    rho = np.exp(-1 / taus)
    e = np.zeros(z.shape[:-1], dtype=np.complex128)
    h = np.empty_like(z)
    normalization = np.exp(-np.arange(memory) / taus[..., None]).sum(axis=-1)
    for t in range(len(x)):
        h[..., t] = 2 * e * np.exp(1j * phase[:, t, None]) / normalization
        e = rho * e + z[..., t]
        if t >= memory:
            e -= rho ** memory * z[..., t - memory]
    return np.stack([h.real, h.imag], axis=-1).transpose(2, 0, 1, 3)


def injected_pulses(rpm: np.ndarray, fs: float, hz: float = 930.,
                    offset_turn: float = .61, amplitude: float = 1.) -> np.ndarray:
    """Synthetic once-per-revolution decaying burst for paired pulse checks."""
    turns = shaft_phase(rpm, fs)
    seconds = np.arange(len(rpm)) / fs
    event_turns = np.arange(np.floor(turns[0] - offset_turn) + 1,
                            np.ceil(turns[-1] - offset_turn)) + offset_turn
    events = np.interp(event_turns, turns, seconds)
    previous = np.searchsorted(events, seconds, side="right") - 1
    if not len(events):
        return np.zeros(len(rpm))
    dt = seconds - events[np.maximum(previous, 0)]
    positive_dt = np.maximum(dt, 0)
    return np.where((previous >= 0) & (dt < .018),
                    amplitude * np.exp(-positive_dt / .0035) * np.sin(2 * np.pi * hz * positive_dt), 0.)


def slow_adapter(x: np.ndarray, base: np.ndarray, gain: float = .5, tau: float = 640,
                 memory: int = 1024, clip: float = 1.75) -> np.ndarray:
    """Apply a delayed clipped mean of base errors. Missing warm-up errors enter as zero."""
    if gain == 0:
        return base.copy()
    error = np.nan_to_num(x - base, nan=0.0)
    w = np.exp(-np.arange(memory) / tau)
    corrected = fftconvolve(np.clip(error, -clip, clip), w / w.sum(), mode="full")
    return base + gain * np.r_[0., corrected[:len(base) - 1]]


class DHON(nn.Module):
    """Trainable order, damping and readout from short past vibration and speed."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        bank = len(config.harmonics) + len(config.hz)
        n_tau = len(config.taus)
        self.register_buffer("carrier_prior", torch.tensor(
            [37.0 * h for h in config.harmonics] + list(config.hz), dtype=torch.float64))
        self.register_buffer("carrier_bound", torch.tensor(
            [0.75] * len(config.harmonics) + [24.] * len(config.hz), dtype=torch.float64))
        tau = torch.tensor(config.taus)
        self.register_buffer("tau_prior", tau.repeat(bank, 1))
        self.carrier_raw = nn.Parameter(torch.zeros(bank), requires_grad=config.joint in {"frequency", "both"})
        self.tau_raw = nn.Parameter(torch.logit((self.tau_prior - 4) / 380),
                                    requires_grad=config.joint in {"decay", "both"})
        self.register_buffer("rpm_center", torch.tensor(0.))
        self.register_buffer("rpm_scale", torch.tensor(1.))
        self.register_buffer("innovation", torch.zeros(3, config.innovation_order))
        self.context_width = 2 + int(config.acceleration)
        linear_dim = (1 + config.lags + config.angular_steps * config.mesh_cycles
                      + self.context_width + config.innovation_lags + bank * n_tau * 2)
        self.linear = nn.Linear(linear_dim * (1 + config.speed_degree), 1, bias=False)
        self.register_buffer("feature_scale", torch.ones(self.linear.in_features))
        if config.width:
            self.mix = nn.Linear(self.context_width, bank * n_tau) if bank else None
            self.gain = nn.Linear(self.context_width, bank) if bank else None
            self.head = nn.Sequential(nn.Linear(self.context_width + bank * 2, config.width),
                                      nn.SiLU(), nn.Linear(config.width, 1), nn.Tanh())
            nn.init.zeros_(self.head[2].weight)
            nn.init.zeros_(self.head[2].bias)

    def physical(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.carrier_prior + self.carrier_bound * self.carrier_raw.tanh(), 4 + 380 * self.tau_raw.sigmoid()

    def prepare(self, x: np.ndarray, rpm: np.ndarray, fs: float) -> tuple:
        """Build deterministic measured-history features. No targets enter their own rows."""
        c = self.config
        c.validate(fs, rpm)
        indices = np.arange(BASE_HISTORY, len(x))
        phase = shaft_phase(rpm, fs)
        center = float(self.rpm_center)
        scale = float(self.rpm_scale)
        speed = np.clip((rpm[indices - 1] - center) / scale, -1.2, 1.2)
        if not c.use_speed:
            speed = np.zeros_like(speed)
        rms = np.sqrt(np.mean(_lags(x, indices, 48) ** 2, axis=1))
        context = [speed, rms]
        if c.acceleration:
            context.append((rpm[indices - 1] - rpm[indices - 33]) / scale)
        context = np.column_stack(context)
        features = [np.ones(len(indices)), _lags(x, indices, c.lags), context]
        if c.mesh_cycles:
            features.append(angular_lags(x, phase, indices, c.angular_steps, c.mesh_cycles))
        if c.innovation_lags:
            coefficients = self.innovation.detach().cpu().numpy()
            z = np.clip((rpm[:-1] - center) / scale, -1.2, 1.2)
            estimate = sum(lfilter(weight, [1.], x)[:-1] * z ** degree
                           for degree, weight in enumerate(coefficients))
            innovation = x - np.r_[0., estimate]
            innovation[:c.innovation_order] = 0.
            features.append(_lags(np.clip(innovation, -c.innovation_clip, c.innovation_clip),
                                  indices, c.innovation_lags))
        basis = [np.ones_like(speed), speed, (3 * speed**2 - 1) / 2, (5 * speed**3 - 3 * speed) / 2]
        device = self.linear.weight.device
        return (torch.as_tensor(x, dtype=torch.float32, device=device),
                torch.as_tensor(phase, dtype=torch.float64, device=device),
                torch.as_tensor(np.arange(len(x)) / fs, dtype=torch.float64, device=device),
                torch.as_tensor(np.column_stack(features), dtype=torch.float32, device=device),
                torch.as_tensor(context, dtype=torch.float32, device=device),
                torch.as_tensor(np.column_stack(basis[:1 + c.speed_degree]), dtype=torch.float32, device=device))

    def design(self, data: tuple, *, cached_states: torch.Tensor | None = None) -> tuple:
        x, turns, seconds, fixed, context, basis = data
        c = self.config
        bank = self.carrier_prior.numel()
        if bank:
            carrier, taus = self.physical()
            n_order = len(c.harmonics)
            phase = 2 * torch.pi * torch.cat(
                [carrier[:n_order, None] * turns, carrier[n_order:, None] * seconds]
            )
            state = oscillator_states(x, phase, taus, c.memory)[BASE_HISTORY:] if cached_states is None else cached_states
            state_flat = state.flatten(1)
            linear = torch.cat([fixed, state_flat], dim=1)
        else:
            state = None
            linear = fixed
        return (basis[:, :, None] * linear[:, None, :]).flatten(1), state, context

    def forward_design(self, matrix: torch.Tensor, state: torch.Tensor | None,
                       context: torch.Tensor) -> torch.Tensor:
        base = self.linear(matrix / self.feature_scale).squeeze(-1)
        if not self.config.width:
            return base
        if state is not None:
            bank, n_tau = state.shape[1:3]
            mix = self.mix(context).view(-1, bank, n_tau).softmax(dim=2)
            gain = 1 + .25 * self.gain(context).tanh()
            mixed = ((state * mix[..., None]).sum(dim=2) * gain[..., None]).flatten(1)
            context = torch.cat([context, mixed], dim=1)
        return base + self.config.gate * self.head(context).squeeze(-1)

    @torch.no_grad()
    def predict_base(self, x: np.ndarray, rpm: np.ndarray, fs: float) -> np.ndarray:
        self.eval()
        if len(x) <= BASE_HISTORY:
            return np.full(len(x), np.nan)
        values = self.forward_design(*self.design(self.prepare(x, rpm, fs))).cpu().numpy()
        return np.r_[np.full(BASE_HISTORY, np.nan), values]

    def predict(self, x: np.ndarray, rpm: np.ndarray, fs: float) -> np.ndarray:
        c = self.config
        return slow_adapter(x, self.predict_base(x, rpm, fs), c.adapter_gain,
                            c.adapter_tau, c.adapter_memory, c.adapter_clip)

    def save(self, path: Path) -> None:
        np.savez_compressed(path, configuration=json.dumps(asdict(self.config)), **{
            key: value.detach().cpu().numpy() for key, value in self.state_dict().items()
        })


def load_model(path: Path, device: str = "cpu") -> DHON:
    """Restore a checkpoint from numeric arrays and JSON metadata without pickle."""
    with np.load(path, allow_pickle=False) as archive:
        model = DHON(Config(**json.loads(str(archive["configuration"])))).to(device)
        model.load_state_dict({name: torch.from_numpy(archive[name]) for name in model.state_dict()})
    return model.eval()


def fit(config: Config, segments: list[tuple[np.ndarray, np.ndarray]], fs: float,
        seed: int, device: str) -> tuple[DHON, list[dict]]:
    """Fit a ridge initialization, then update the readout, mixing head and physical parameters."""
    torch.manual_seed(seed)
    model = DHON(config).to(device)
    segments = [(x, rpm) for x, rpm in segments if len(x) > BASE_HISTORY]
    if not segments:
        raise ValueError("No fitted block is longer than its reset context")
    train_rpm = np.concatenate([rpm for _, rpm in segments])
    with torch.no_grad():
        model.rpm_center.fill_(float((train_rpm.max() + train_rpm.min()) / 2))
        model.rpm_scale.fill_(float(max((train_rpm.max() - train_rpm.min()) / 2, 1.0)))
        if config.innovation_lags and config.innovation_order:
            a = config.innovation_order
            matrices = []
            for x, rpm in segments:
                history = _lags(x, np.arange(a, len(x)), a)
                speed = np.clip((rpm[a-1:-1] - float(model.rpm_center)) / float(model.rpm_scale), -1.2, 1.2)
                matrices.append(np.column_stack([history * speed[:, None] ** j for j in range(3)]))
            stage_matrix = np.concatenate(matrices)
            stage_targets = np.concatenate([x[a:] for x, _ in segments])
            model.innovation.copy_(torch.from_numpy(ridge(stage_matrix, stage_targets, 0.0003).reshape(3, a)))
    data = [model.prepare(x, rpm, fs) for x, rpm in segments]
    y_np = np.concatenate([x[BASE_HISTORY:] for x, _ in segments])
    y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        designs = [model.design(block) for block in data]
        matrices = [design[0].cpu().numpy() for design in designs]
        base = np.concatenate(matrices)
        scale = np.maximum(base.std(axis=0), 0.05)
        # The constant intercept is represented as a column; preserve its unit scale.
        scale[0] = 1.
        model.feature_scale.copy_(torch.from_numpy(scale))
        ridge_data = [base / scale]
        targets = [y_np]
        # Paired RPM variants preserve the targets. Paired pulses penalize changes
        # in the prediction so that impulsive content remains in the residual.
        for x, rpm in segments:
            subset = slice(None, None, 5)
            pristine = model.design(model.prepare(x, rpm, fs))[0].cpu().numpy()[subset] / scale
            if config.augment > 0:
                for bias in (-.0075, .0075):
                    variant = model.design(model.prepare(x, rpm * (1 + bias), fs))[0].cpu().numpy()[subset]
                    factor = config.augment ** .5
                    ridge_data.append(factor * variant / scale)
                    targets.append(factor * x[BASE_HISTORY:][subset])
            if config.preserve > 0:
                for hz, turns in ((520., .23), (700., .71)):
                    pulse = injected_pulses(rpm, fs, hz, turns, amplitude=.9)
                    variant = model.design(model.prepare(x + pulse, rpm, fs))[0].cpu().numpy()[subset]
                    ridge_data.append((config.preserve / 2) ** .5 * (variant / scale - pristine))
                    targets.append(np.zeros(len(variant)))
        weights = ridge(np.concatenate(ridge_data), np.concatenate(targets), config.ridge)
        model.linear.weight.copy_(torch.from_numpy(weights[None]))
    traces = []
    physical = [p for p in (model.carrier_raw, model.tau_raw) if p.requires_grad and p.numel()]
    readout = [p for name, p in model.named_parameters() if name not in ("carrier_raw", "tau_raw")]
    optimizer = torch.optim.AdamW([{"params": readout, "lr": config.lr},
                                   {"params": physical, "lr": config.physics_lr}], weight_decay=1e-4)
    # Full-batch updates share the contiguous differentiable convolutions within each reset block.
    # Two fixed-physics epochs let the residual head initialize before physical updates.
    for epoch in range(config.epochs):
        optimizer.zero_grad()
        active = bool(physical) and epoch >= 2
        if active:
            values = [model.forward_design(*model.design(block)) for block in data]
        else:
            values = [model.forward_design(*design) for design in designs]
        pred = torch.cat(values)
        carrier, tau = model.physical()
        prior = torch.zeros((), device=device)
        if active:
            prior = (model.carrier_raw.square().mean() + (tau / model.tau_prior).log().square().mean())
        mse = (pred - y).square().mean()
        loss = mse + config.prior_penalty * prior
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        gradients = {
            name: float(p.grad.norm()) if p.grad is not None else 0.
            for name, p in (("carrier", model.carrier_raw), ("decay", model.tau_raw))
        }
        optimizer.step()
        carrier, tau = model.physical()
        traces.append({"epoch": epoch + 1, "train_mse_before_step": float(mse.detach()),
                       "carrier_gradient_norm": gradients["carrier"], "decay_gradient_norm": gradients["decay"],
                       "carriers": carrier.detach().cpu().tolist(), "taus": tau.detach().cpu().tolist()})
    return model.eval(), traces
