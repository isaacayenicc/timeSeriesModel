"""Physics-informed structured reservoir features and a regularized linear readout."""

from dataclasses import asdict, dataclass, field
from itertools import combinations_with_replacement
import json
import math
from pathlib import Path

import numpy as np
from scipy import linalg, signal


@dataclass(frozen=True)
class Config:
    """Searchable causal feature architecture and deterministic solver settings."""

    variant: str = "optimized"
    lags: int = 96
    lag_history: int = 192
    lag_policy: str = "consecutive"
    angular_steps: int = 16
    mesh_cycles: int = 8
    harmonics: tuple[float, ...] = (37., 74., 148., 259.)
    fixed_hz: tuple[float, ...] = ()
    taus: tuple[float, ...] = (8., 32., 128.)
    memory: int = 512
    innovation_order: int = 192
    innovation_lags: int = 24
    innovation_degree: int = 2
    innovation_folds: int = 0
    innovation_clip: float = 3.
    innovation_ridge: float = .0003
    speed_degree: int = 2
    energy_window: int = 64
    acceleration_window: int = 32
    min_rpm: float = 100.
    polynomial_degree: int = 3
    interaction_policy: str = "structured"
    nonlinear_cap: int = 60
    rank: int = 0
    solver: str = "ridge"
    huber_delta: float = 1.5
    ridge: float = .022
    nonlinear_ridge_multiplier: float = 6.
    physics_ridge_multiplier: float = .9
    adapter_gain: float = .5
    adapter_tau: float = 640.
    adapter_memory: int = 1024
    adapter_clip: float = 1.75
    pulse_preservation: float = .2
    augmentation: float = .35


def _variant(c: Config) -> tuple[bool, bool, bool]:
    name = c.variant.lower()
    if name.startswith("pi-src-"):
        name = name[7:]
    elif name.startswith("src-"):
        name = name[4:]
    if name not in {"linear", "polynomial", "speed", "physics_linear", "physics-linear",
                    "generic", "fixed", "optimized", "compact", "corrected"}:
        raise ValueError(f"Unknown SRC variant {c.variant!r}")
    physics = name not in {"linear", "polynomial", "speed", "generic"}
    speed = name not in {"linear", "polynomial"}
    nonlinear = (name not in {"linear", "speed", "physics_linear", "physics-linear"}
                 and c.polynomial_degree >= 2 and c.nonlinear_cap > 0)
    return physics, speed, nonlinear


def _validate(c: Config, fs: float) -> None:
    _variant(c)
    if not (fs > 0 and c.min_rpm > 0 and c.lags >= 1 and c.lag_history >= 1
            and 0 <= c.speed_degree <= 3 and 0 <= c.innovation_degree <= 3
            and c.memory >= 1 and all(tau > 0 for tau in c.taus) and len(c.taus)
            and c.ridge >= 0 and c.innovation_ridge > 0
            and c.angular_steps > 0 and c.mesh_cycles >= 0
            and c.adapter_tau > 0 and c.adapter_memory > 0 and c.adapter_clip > 0
            and c.nonlinear_cap >= 0 and c.rank >= 0
            and c.augmentation >= 0 and c.pulse_preservation >= 0
            and c.huber_delta > 0):
        raise ValueError("Invalid nonpositive dimension, memory, penalty or rate")
    if c.innovation_lags and not c.innovation_order:
        raise ValueError("A proxy innovation history requires a nonzero first-stage order")
    if c.solver not in {"ridge", "huber", "ols"}:
        raise ValueError("solver must be ridge, huber or ols")


def time_lags(c: Config) -> np.ndarray:
    """Deterministic consecutive or dense-recent/multiresolution delay geometry."""
    if c.lag_policy == "consecutive":
        return np.arange(1, c.lags + 1)
    if c.lag_policy != "multiresolution":
        raise ValueError("lag_policy must be consecutive or multiresolution")
    split = min(16, c.lags)
    early = np.arange(1, split + 1)
    if split == c.lags:
        return early
    if c.lag_history < c.lags:
        raise ValueError("Multiresolution lag history must be no smaller than lag count")
    sparse = np.round(np.geomspace(split + 1, c.lag_history, c.lags - split)).astype(int)
    chosen = np.unique(np.r_[early, sparse])
    if len(chosen) < c.lags:
        unused = np.setdiff1d(np.arange(1, c.lag_history + 1), chosen)
        chosen = np.sort(np.r_[chosen, unused[:c.lags-len(chosen)]])
    return chosen


def receptive_fields(c: Config, fs: float) -> tuple[int, int]:
    """Return exact conservative base and full support in raw samples."""
    physics, speed, nonlinear = _variant(c)
    history = max(int(time_lags(c)[-1]), c.energy_window, 1)
    if speed and c.acceleration_window:
        history = max(history, c.acceleration_window + 1)
    if nonlinear:
        history = max(history, 64)
    if physics:
        if c.mesh_cycles:
            history = max(history, math.ceil(c.mesh_cycles * fs * 60 / (37 * c.min_rpm)) + 2)
        if c.harmonics or c.fixed_hz:
            history = max(history,c.memory)
        if c.innovation_lags:
            history = max(history, c.innovation_order + c.innovation_lags)
    return history, history + (c.adapter_memory if c.adapter_gain else 0)


def phase_turns(rpm: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Observed trapezoidal rotations and strict previous-speed target rotations."""
    rpm = np.asarray(rpm, dtype=np.float64)
    if len(rpm) == 0:
        return rpm.copy(), rpm.copy()
    turns = np.r_[0., np.cumsum((rpm[:-1] + rpm[1:]) / (120. * fs))]
    target = np.r_[0., turns[:-1] + rpm[:-1] / (60. * fs)]
    return turns, target


def angular_lags(x: np.ndarray, rpm: np.ndarray, fs: float, idx: np.ndarray,
                 steps: int, cycles: int) -> np.ndarray:
    """Interpolate only between observations earlier than the target index."""
    turns, goal = phase_turns(rpm,fs)
    offsets = np.arange(1, steps * cycles + 1) / (37. * steps)
    query = goal[idx, None] - offsets
    # The cap matters for very high angle resolution at the upper RPM edge.
    query = np.minimum(query, turns[idx-1,None])
    right = np.searchsorted(turns, query, side="left")
    if np.any(right < 1) or np.any(right >= idx[:,None]):
        raise ValueError("Angular lag exceeded its pre-target support")
    left = right-1
    frac = (query-turns[left]) / (turns[right]-turns[left])
    return (x[left]*(1-frac) + x[right]*frac).astype(np.float32)


def _state(x, source_phase, target_phase, tau, memory):
    carrier = x * np.exp(-1j * source_phase)
    rho = math.exp(-1. / tau)
    finite = carrier.copy()
    finite[memory:] -= (rho ** memory) * carrier[:-memory]
    envelope = signal.lfilter([1.], [1., -rho], finite)
    scale = 2. * (-math.expm1(-1. / tau)) / (-math.expm1(-memory / tau))
    return scale * np.r_[0j, envelope[:-1]] * np.exp(1j * target_phase)


def oscillator_states(x: np.ndarray, fs: float, rpm: np.ndarray, *, tau: float,
                      memory: int, order: float | None = None,
                      hz: float | None = None) -> np.ndarray:
    """A normalized finite quadrature state at each strict one-step target."""
    if (order is None) == (hz is None):
        raise ValueError("Specify exactly one of order and hz")
    if not len(x):
        return np.empty(0, dtype=np.complex128)
    if order is not None:
        u, goal = phase_turns(rpm, fs)
        source_phase, target_phase = (2*np.pi*order*u, 2*np.pi*order*goal)
    else:
        source_phase = target_phase = 2*np.pi*hz*np.arange(len(x))/fs
    return _state(x,source_phase,target_phase,tau,memory)


def slow_adapter(x, base, gain=.5, tau=640., memory=1024, clip=1.75):
    """Use only clipped past base model errors; no recurrent parameter fitting."""
    if gain == 0:
        return base.copy()
    if not len(x):
        return base.copy()
    e = np.clip(np.nan_to_num(x-base, nan=0.),-clip,clip)
    weights = np.exp(-np.arange(memory)/tau)
    scored = signal.oaconvolve(e,weights/weights.sum(),mode="full")
    return base + gain * np.r_[0.,scored[:len(x)-1]]


def injected_pulses(rpm, fs, hz=930., phase=.61, amplitude=1., tau=.0035):
    """Offline synthetic once-per-shaft-turn decaying burst for paired tests."""
    turns, _ = phase_turns(rpm,fs)
    if not len(turns):
        return turns
    time = np.arange(len(turns)) / fs
    crossings = np.arange(math.floor(turns[0]-phase)+1,math.ceil(turns[-1]-phase))+phase
    if not len(crossings):
        return np.zeros(len(turns))
    event_t = np.interp(crossings,turns,time)
    last = np.searchsorted(event_t,time,side="right") - 1
    dt = np.maximum(time-event_t[np.maximum(last,0)],0.)
    return np.where((last >= 0) & (dt < .018),
                    amplitude*np.exp(-dt/tau)*np.sin(2*np.pi*hz*dt),0.)


def _lag(x, idx, offsets):
    return x[idx[:,None]-offsets]


def _scaled_speed(rpm, speed):
    return np.clip((rpm-speed[0])/speed[1],-1.2,1.2)


def _legendre(z, degree):
    return [np.ones_like(z), z, (3*z*z-1)/2, (5*z*z*z-3*z)/2][:degree+1]


def _proxy_design(x,rpm,idx,c,speed):
    lags = _lag(x,idx,np.arange(1,c.innovation_order+1)).astype(np.float32)
    z = _scaled_speed(rpm[idx-1],speed)
    return np.column_stack([lags*z[:,None]**degree for degree in range(c.innovation_degree+1)])


def _proxy_errors(x,rpm,c,speed,w):
    if not len(w):
        return np.zeros(len(x))
    z = _scaled_speed(rpm[:-1],speed)
    pred = np.zeros(len(x)-1)
    for degree, row in enumerate(w):
        pred += signal.lfilter(row,[1.],x)[:-1] * z**degree
    e = x - np.r_[0.,pred]
    e[:c.innovation_order] = 0.
    return np.clip(e,-c.innovation_clip,c.innovation_clip)


def _normal_eq(x,y,penalty,obs_weight,block_size):
    p = x.shape[1]
    gram = np.zeros((p,p),dtype=np.float64)
    rhs = np.zeros(p,dtype=np.float64)
    for a in range(0,len(y),block_size):
        b = min(a+block_size,len(y))
        chunk = x[a:b].astype(np.float64)
        w = obs_weight[a:b]
        gram += chunk.T @ (w[:,None]*chunk)
        rhs += chunk.T @ (w*y[a:b])
    gram.flat[::p+1] += len(y)*penalty
    return gram,rhs


def solve_readout(x,y,penalty,*,solver="ridge",huber_delta=1.5,
                  residual_rows=None,block_size=4096):
    """Blockwise Gram assembly, stable solves and optional Huber IRLS."""
    x = np.asarray(x)
    y = np.asarray(y,dtype=np.float64)
    penalty = np.broadcast_to(np.asarray(penalty,dtype=np.float64),(x.shape[1],)).copy()
    if solver == "ols":
        penalty.fill(0.)
    active = np.ones(len(y),dtype=bool) if residual_rows is None else np.asarray(residual_rows)
    weights = np.ones(len(y))
    machine = np.finfo(np.float64).eps
    backend = "cholesky"
    stability = None
    for epoch in range(5 if solver == "huber" else 1):
        gram,rhs = _normal_eq(x,y,penalty,weights,block_size)
        gram_norm = np.linalg.norm(gram,ord=1)
        try:
            triangular = linalg.cholesky(gram,lower=True,check_finite=False)
            rcond,info = linalg.lapack.dpocon(triangular,gram_norm,uplo="L")
            if info or rcond < 32*machine:
                raise linalg.LinAlgError("Near-singular normal equations")
            stability = float(1./rcond)
            coef = linalg.cho_solve((triangular,True),rhs,check_finite=False)
        except linalg.LinAlgError:
            backend = "pivoted_qr"
            x_weighted = x.astype(np.float64) * np.sqrt(weights[:,None])
            design = np.vstack([x_weighted,np.diag(np.sqrt(len(y)*penalty))])
            desired = np.r_[np.sqrt(weights)*y,np.zeros(x.shape[1])]
            coef,_,_,_ = linalg.lstsq(design,desired,lapack_driver="gelsy",cond=1e-10)
        if solver == "huber":
            residual = np.abs(y[active] - x[active]@coef)
            weights[active] = np.minimum(1.,huber_delta/np.maximum(residual,machine))
    return coef, {"backend":backend,"gram_condition_estimate":stability,"iterations":epoch+1}


def _proxy_fits(segments,c,speed):
    size = (c.innovation_degree+1)*c.innovation_order
    no_innovation = not (_variant(c)[0] and c.innovation_lags)
    if no_innovation:
        return np.empty((0,0)), [None]*len(segments)
    rows = []
    for x,rpm in segments:
        idx = np.arange(c.innovation_order,len(x))
        rows.append((_proxy_design(x,rpm,idx,c,speed),x[idx]))
    matrix = np.concatenate([a for a,_ in rows])
    target = np.concatenate([b for _,b in rows])
    w,_ = solve_readout(matrix,target,c.innovation_ridge)
    w = w.reshape(-1,c.innovation_order)
    if c.innovation_folds < 2:
        return w,[None]*len(segments)
    # Cross-fit innovations in whole temporal blocks. The first-stage fit
    # excludes its own receptive field around each held innovation interval.
    fit_errors = []
    for current,(x,rpm) in enumerate(segments):
        errors = np.zeros(len(x))
        for fold in np.array_split(np.arange(c.innovation_order,len(x)),c.innovation_folds):
            if not len(fold):
                continue
            training = [(a,b) if j!=current else
                        (np.concatenate([a[:max(fold[0]-2*c.innovation_order,0)],
                                         a[fold[-1]+1:]]),
                         np.r_[b[:max(fold[0]-2*c.innovation_order,0)],b[fold[-1]+1:]])
                        for j,(a,b) in enumerate(rows)]
            a = np.concatenate([part for part,_ in training])
            b = np.concatenate([part for _,part in training])
            if len(b) < size:
                raise ValueError("Insufficient inner training support for cross-fitted innovations")
            held_w,_ = solve_readout(a,b,c.innovation_ridge)
            errors[fold] = _proxy_errors(x,rpm,c,speed,held_w.reshape(-1,c.innovation_order))[fold]
        fit_errors.append(errors)
    return w,fit_errors


def _library(x,rpm,fs,idx,c,speed,proxy,override_errors=None):
    physics, use_speed, nonlinear = _variant(c)
    outputs, names, groups = [], [], []

    def append(values,names_here,group):
        values = np.asarray(values,dtype=np.float32).reshape(len(idx),-1)
        outputs.append(values)
        names.extend(names_here)
        groups.extend([group]*values.shape[1])

    offsets = time_lags(c)
    raw = _lag(x,idx,offsets)
    lag_groups = [(raw,[f"x[-{i}]" for i in offsets],"lag")]
    z = _scaled_speed(rpm[idx-1],speed) if use_speed else np.zeros(len(idx))
    energy = np.zeros(len(idx))
    if c.energy_window:
        energy = (_lag(x,idx,np.arange(1,c.energy_window+1))**2).mean(axis=1)
    er = np.empty((len(idx),0))
    states = []
    if physics:
        if c.mesh_cycles:
            angle = angular_lags(x,rpm,fs,idx,c.angular_steps,c.mesh_cycles)
            lag_groups.append((angle,[f"mesh_delay[{j+1}/{c.angular_steps}]"
                                     for j in range(angle.shape[1])],"angle"))
        if c.innovation_lags:
            innovation = _proxy_errors(x,rpm,c,speed,proxy) if override_errors is None else override_errors
            er = _lag(innovation,idx,np.arange(1,c.innovation_lags+1))
            lag_groups.append((er,[f"proxy_error[-{j+1}]" for j in range(er.shape[1])],"innovation"))
        carriers = [(float(k),"order") for k in c.harmonics] + [(float(f),"Hz") for f in c.fixed_hz]
        if carriers:
            u, goal = phase_turns(rpm,fs)
            seconds = np.arange(len(x))/fs
            for frequency,unit in carriers:
                source = 2*np.pi*frequency*(u if unit=="order" else seconds)
                target = 2*np.pi*frequency*(goal if unit=="order" else seconds)
                quadratures = [_state(x,source,target,tau,c.memory)[idx] for tau in c.taus]
                osc_name = f"{frequency:g}{unit}"
                states.append((osc_name,quadratures[len(c.taus)//2]))
                columns = [col for h in quadratures for col in (h.real,h.imag)]
                labels = [f"{quad}H[{osc_name},{tau:g}]" for tau in c.taus for quad in ("Re","Im")]
                lag_groups.append((np.column_stack(columns),labels,"oscillator"))

    for degree, basis in enumerate(_legendre(z,c.speed_degree if use_speed else 0)):
        for matrix,labels,group in lag_groups:
            if degree == 0:
                append(matrix,labels,group)
            else:
                append(matrix*basis[:,None],[f"P{degree}(rpm)*{label}" for label in labels],f"speed_{group}")
    if use_speed:
        append(z,["rpm"],"operating")
        if c.acceleration_window:
            delta = (rpm[idx-1] - rpm[idx-1-c.acceleration_window])/speed[1]
            append(delta,[f"rpm_delta[{c.acceleration_window}]"],"operating")
    if c.energy_window:
        append(np.sqrt(energy),[f"RMS[{c.energy_window}]"],"energy")
    if nonlinear:
        taps = np.array([1,2,4,8,16,32,64])
        q = _lag(x,idx,taps)
        if c.variant.lower().endswith("generic") or c.interaction_policy == "generic":
            q_all = np.column_stack([q,z]) if use_speed else q
            par = [f"x[-{i}]" for i in taps] + (["rpm"] if use_speed else [])
            pairs = list(combinations_with_replacement(range(len(par)),2))
            append(np.column_stack([q_all[:,i]*q_all[:,j] for i,j in pairs]),
                   [f"{par[i]}*{par[j]}" for i,j in pairs],"nonlinear_generic")
            if c.polynomial_degree >= 3:
                append(q_all**3,[f"{name}^3" for name in par],"nonlinear_generic")
        else:
            append(q*q,[f"x[-{i}]^2" for i in taps],"nonlinear_delay")
            append(q[:,:-1]*q[:,1:],[f"x[-{i}]*x[-{j}]" for i,j in zip(taps[:-1],taps[1:])],
                   "nonlinear_delay")
            if c.polynomial_degree >= 3:
                append(q**3,[f"x[-{i}]^3" for i in taps],"nonlinear_delay")
            if c.energy_window:
                append(q*energy[:,None],[f"x[-{i}]*energy" for i in taps],"nonlinear_energy")
                if use_speed:
                    append(q*(energy*z)[:,None],[f"x[-{i}]*energy*rpm" for i in taps],
                           "nonlinear_energy")
            if er.shape[1]:
                e_cols = [i for i in (0,1,3,7) if i < er.shape[1]]
                append(q[:,:len(e_cols)]*er[:,e_cols],
                       [f"x[-{taps[j]}]*proxy_error[-{i+1}]" for j,i in enumerate(e_cols)],
                       "nonlinear_innovation")
                append(np.mean(er**2,axis=1),["proxy_error_energy"],"nonlinear_innovation")
            if states:
                energies = []
                for label,h in states:
                    raw_energy = h.real*h.real+h.imag*h.imag
                    h_energy = raw_energy/(1+raw_energy)
                    energies.append(h_energy)
                    append(h_energy,[f"bounded_energy[{label}]"],"nonlinear_oscillator")
                    if use_speed:
                        append(z*h_energy,[f"rpm*bounded_energy[{label}]"],"nonlinear_oscillator")
                    if c.polynomial_degree>=3:
                        append(np.column_stack([h.real*h_energy,h.imag*h_energy]),
                               [f"ReH[{label}]*energy",f"ImH[{label}]*energy"],"nonlinear_oscillator")
                modulation = np.mean(energies,axis=0)
                append(q*modulation[:,None],[f"x[-{i}]*mean_osc_energy" for i in taps],
                       "nonlinear_oscillator")
                if c.interaction_policy=="coherent":
                    for (name_a,a),(name_b,b) in zip(states[:-1],states[1:]):
                        if name_a.endswith("order") and name_b.endswith("order"):
                            cross = a*np.conj(b)
                            append(np.column_stack([cross.real,cross.imag]),
                                   [f"{p}H[{name_a}]*conjH[{name_b}]" for p in ("Re","Im")],
                                   "nonlinear_coherent")
    return np.column_stack(outputs), names, groups


@dataclass
class SRCModel:
    """Fitted preprocessing, selected library and model coefficients."""

    config: Config
    fs: float
    speed: np.ndarray
    proxy: np.ndarray
    kept: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    names: tuple[str,...]
    groups: tuple[str,...]
    projection: np.ndarray = field(default_factory=lambda: np.empty((0,0)))
    solver_info: dict = field(default_factory=dict)

    @property
    def history(self) -> int:
        return receptive_fields(self.config,self.fs)[1]

    @property
    def parameters(self) -> int:
        return self.coefficients.size + self.proxy.size + self.projection.size

    @property
    def feature_count(self) -> int:
        return self.coefficients.size-1

    def _select(self,matrix):
        scaled = (matrix[:,self.kept]-self.center)/self.scale
        if self.projection.size:
            count = self.projection.shape[0]
            return np.column_stack([scaled[:,:-count],scaled[:,-count:]@self.projection])
        return scaled

    @property
    def readout_features(self):
        """Ordered coefficients and their group membership after optional PCA."""
        names = self.names
        groups = self.groups
        if self.projection.size:
            count,rank = self.projection.shape
            names = names[:-count] + tuple(f"nonlinear_component_{j+1}" for j in range(rank))
            groups = groups[:-count] + ("nonlinear_projected",)*rank
        return list(zip(("intercept",)+names,("intercept",)+groups,self.coefficients.tolist()))

    def predict_base(self,x,rpm,fs):
        x, rpm = np.asarray(x,dtype=np.float64), np.asarray(rpm,dtype=np.float64)
        _check_record(x,rpm,fs,self.fs,self.config)
        start,_ = receptive_fields(self.config,self.fs)
        output = np.full(len(x),np.nan)
        for a in range(start,len(x),4096):
            idx = np.arange(a,min(a+4096,len(x)))
            # Each batch sees only the required bounded past, not another
            # acquisition or prior call. Phase origin cancels in all states.
            origin = a-start
            stop = idx[-1]+1
            lib,_,_ = _library(x[origin:stop],rpm[origin:stop],fs,idx-origin,self.config,self.speed,self.proxy)
            output[idx] = self.coefficients[0] + self._select(lib)@self.coefficients[1:]
        return output

    def predict(self,x,rpm,fs):
        base = self.predict_base(x,rpm,fs)
        c = self.config
        score = slow_adapter(np.asarray(x),base,c.adapter_gain,c.adapter_tau,c.adapter_memory,c.adapter_clip)
        score[:self.history] = np.nan
        return score

    def predict_one_step(self,record,target_indices=None,state_policy="reset"):
        if state_policy!="reset":
            raise ValueError("Only independent record reset is supported")
        forecast = self.predict(record.x,record.rpm,record.fs)
        idx = np.arange(self.history,len(forecast)) if target_indices is None else np.asarray(target_indices)
        if np.any(idx < self.history) or np.any(idx >= len(forecast)):
            raise ValueError("Requested indices exceed the model's scorable range")
        mask = np.zeros(len(forecast), dtype=bool)
        mask[idx] = True
        return {"indices":idx,"mask":mask,"prediction":forecast[idx],"target":record.x[idx],
                "residual":record.x[idx]-forecast[idx],"time":record.t[idx],"state_policy":state_policy,
                "history":self.history,"model":self.config.variant,"model_version":"1"}

    def save(self,path):
        path = Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary = path.with_suffix(path.suffix+".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle,config=json.dumps(asdict(self.config)),fs=self.fs,
                                speed=self.speed,proxy=self.proxy,kept=self.kept,center=self.center,
                                scale=self.scale,coefficients=self.coefficients,names=np.asarray(self.names),
                                groups=np.asarray(self.groups),projection=self.projection,
                                solver_info=json.dumps(self.solver_info))
        temporary.replace(path)


def load_model(path) -> SRCModel:
    """Read numeric arrays and configuration without executable pickle objects."""
    with np.load(path,allow_pickle=False) as a:
        return SRCModel(Config(**json.loads(a["config"].item())),float(a["fs"]),a["speed"],
                        a["proxy"],a["kept"],a["center"],a["scale"],a["coefficients"],
                        tuple(a["names"].tolist()),tuple(a["groups"].tolist()),a["projection"],
                        json.loads(a["solver_info"].item()))


def _check_record(x,rpm,fs,expected_fs,c):
    if x.ndim!=1 or rpm.shape!=x.shape or not np.isclose(fs,expected_fs):
        raise ValueError("Expected aligned vector inputs at the fitted sample rate")
    if not np.isfinite(x).all() or not np.isfinite(rpm).all():
        raise ValueError("Nonfinite record input")
    if _variant(c)[0] and np.any(rpm<c.min_rpm):
        raise ValueError(f"RPM below the declared angular support of {c.min_rpm:g}")


def fit(config: Config | dict, segments: list[tuple[np.ndarray,np.ndarray]],fs: float) -> SRCModel:
    """Fit all data-dependent transformations and the readout on explicit raw fit blocks."""
    c = Config(**config) if isinstance(config,dict) else config
    _validate(c,fs)
    history,_ = receptive_fields(c,fs)
    segments = [(np.asarray(x,dtype=np.float64),np.asarray(rpm,dtype=np.float64))
                for x,rpm in segments if len(x)>history]
    if not segments:
        raise ValueError("No fit segment contains enough causal history")
    for x,rpm in segments:
        _check_record(x,rpm,fs,fs,c)
    lo = min(float(rpm.min()) for _,rpm in segments)
    hi = max(float(rpm.max()) for _,rpm in segments)
    if _variant(c)[0] and ((c.harmonics and max(c.harmonics)*hi*1.03/60 >= .49*fs)
                          or (c.fixed_hz and max(c.fixed_hz)>=.49*fs)):
        raise ValueError("Oscillator bank violates training Nyquist margin under RPM stress")
    speed = np.array([(lo+hi)/2,max((hi-lo)/2,1.)])
    proxy, overrides = _proxy_fits(segments,c,speed)
    matrices, indices = [], []
    for (x,rpm),errors in zip(segments,overrides):
        idx = np.arange(history,len(x))
        lib,names,groups = _library(x,rpm,fs,idx,c,speed,proxy,errors)
        indices.append(idx)
        matrices.append(lib)
    design = np.concatenate(matrices)
    y = np.concatenate([x[idx] for (x,_),idx in zip(segments,indices)])
    center = design.mean(axis=0,dtype=np.float64)
    scale = design.std(axis=0,dtype=np.float64)
    practical = np.flatnonzero(scale>1e-7)
    nl = np.array([str(g).startswith("nonlinear_") for g in groups])
    linear_cols = practical[~nl[practical]]
    nonlinear_cols = practical[nl[practical]]
    if len(nonlinear_cols)>c.nonlinear_cap:
        values = design[:,nonlinear_cols]
        corr = np.abs((y-y.mean())@values) / scale[nonlinear_cols]
        shortlist = np.argsort(-corr,kind="stable")[:c.nonlinear_cap]
        nonlinear_cols = np.sort(nonlinear_cols[shortlist])
    kept = np.r_[linear_cols,nonlinear_cols]
    model = SRCModel(c,fs,speed,proxy,kept,center[kept],np.maximum(scale[kept],.005),
                     np.empty(0),tuple(names[j] for j in kept),tuple(groups[j] for j in kept))
    if c.rank and len(nonlinear_cols)>c.rank:
        small = model._select(design)[:,-len(nonlinear_cols):]
        _,vectors = linalg.eigh(small.T@small,subset_by_index=(len(nonlinear_cols)-c.rank,len(nonlinear_cols)-1))
        model.projection = vectors[:,::-1]
    scaled = model._select(design)
    inputs = [np.column_stack([np.ones(len(y)),scaled])]
    targets = [y]
    validation = [np.ones(len(y),dtype=bool)]
    for (x,rpm),idx,pristine,errors in zip(segments,indices,matrices,overrides):
        idx = idx[::5]
        # Paired comparisons use the same full-fit proxy on both
        # sides when the clean training matrix uses inner cross-fitting.
        paired_base = (pristine[::5] if errors is None
                       else _library(x,rpm,fs,idx,c,speed,proxy)[0])
        original = model._select(paired_base)
        if c.augmentation and _variant(c)[1]:
            for bias in (-.0075,.0075):
                biased = _library(x,rpm*(1+bias),fs,idx,c,speed,proxy)[0]
                w = math.sqrt(c.augmentation)
                inputs.append(w*np.column_stack([np.ones(len(idx)),model._select(biased)]))
                targets.append(w*x[idx])
                validation.append(np.zeros(len(idx),dtype=bool))
        if c.pulse_preservation:
            for frequency,phase in ((520.,.23),(700.,.71)):
                pulse = injected_pulses(rpm,fs,frequency,phase,.9)
                injected = _library(x+pulse,rpm,fs,idx,c,speed,proxy)[0]
                w = math.sqrt(c.pulse_preservation/2)
                delta = model._select(injected)-original
                inputs.append(w*np.column_stack([np.zeros(len(idx)),delta]))
                targets.append(np.zeros(len(idx)))
                validation.append(np.zeros(len(idx),dtype=bool))
    post_groups = list(model.groups)
    if model.projection.size:
        post_groups = post_groups[:-model.projection.shape[0]]+["nonlinear_projected"]*model.projection.shape[1]
    multipliers = [c.nonlinear_ridge_multiplier if g.startswith("nonlinear")
                   else c.physics_ridge_multiplier if "oscillator" in g else 1.
                   for g in post_groups]
    penalty = c.ridge*np.r_[0.,multipliers]
    model.coefficients,model.solver_info = solve_readout(
        np.concatenate(inputs),np.concatenate(targets),penalty,solver=c.solver,
        huber_delta=c.huber_delta,residual_rows=np.concatenate(validation)
    )
    model.solver_info.update(raw_features=design.shape[1],selected_features=len(kept),
                             fitted_features=model.feature_count,near_constant=design.shape[1]-len(practical),
                             training_rows=len(y),augmented_rows=sum(map(len,targets))-len(y),
                             innovation_crossfit=c.innovation_folds)
    return model
