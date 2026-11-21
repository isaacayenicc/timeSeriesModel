"""Classical causal one-step vibration predictors and strict sparse LPV replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from time import perf_counter
import json

import numpy as np
from scipy import linalg, signal

from .protocol import (
    FOLDS, START as SCORE_START, digest, guarded_segments, load_data, mse,
    predict, timed_predict, write_json, write_table,
)
FIT_WARMUP = 336
PROXY_ORDER = 320
MA_ROOT_LIMIT = .985
FAMILY_KEYS = ("ar", "arma", "arima", "arx")
LPV_KEYS = ("sparse_lpv_ar", "sparse_lpv_arma")
EMPTY = np.empty(0)


def _array_checkpoint(path):
    with np.load(path, allow_pickle=False) as file:
        return {key: file[key] for key in file.files}


def _lag(x, indices, offsets):
    return x[indices[:, None] - np.asarray(offsets)]


def _lstsq(x, y):
    return linalg.lstsq(x, y, cond=1e-10, lapack_driver="gelsy", check_finite=False)[0]


def _lagrange(terms):
    return np.arange(1, terms + 1)


def _fir(ar, ma, d, n_lags):
    """Generate transfer function 1-A/M on observed past levels or differences."""
    length = max(len(ar), len(ma)) + 1
    numerator = np.zeros(length)
    numerator[1:len(ar) + 1] += ar
    numerator[1:len(ma) + 1] += ma
    delta = np.zeros(n_lags + 1)
    delta[0] = 1
    response = signal.lfilter(numerator, np.r_[1., ma], delta)
    if d:
        return np.r_[0., 1 + response[1], response[2:] - response[1:-1]]
    return response


@dataclass(frozen=True)
class FitSpec:
    family: str
    p: int
    q: int = 0
    speed_lags: tuple = ()
    ridge: float = 0.

    @property
    def d(self):
        return int(self.family == "arima")


@dataclass(frozen=True)
class StatisticalModel:
    spec: FitSpec
    ar: np.ndarray
    ma: np.ndarray = field(default_factory=lambda: EMPTY)
    exogenous: np.ndarray = field(default_factory=lambda: EMPTY)
    intercept: float = 0.
    ma_radius: float = 0.
    proxy_order: int = 0

    def predict(self, x, rpm, fs):
        x = np.asarray(x, dtype=float)
        if len(self.ma):
            taps = _fir(self.ar, self.ma, self.spec.d, SCORE_START)
            predicted = signal.lfilter(taps, [1.], x)
        else:
            predicted = signal.lfilter(np.r_[0., self.ar], [1.], x)
        if len(self.exogenous):
            rpm = np.asarray(rpm, dtype=float)
            coefficients = np.zeros(max(self.spec.speed_lags) + 1)
            coefficients[list(self.spec.speed_lags)] = self.exogenous
            predicted += signal.lfilter(coefficients, [1.], rpm)
        predicted += self.intercept
        predicted[:SCORE_START] = np.nan
        return predicted

    def parameter_count(self):
        return len(self.ar) + len(self.ma) + len(self.exogenous) + int(self.spec.family == "arx")

    def save(self, path):
        np.savez_compressed(path, family=self.spec.family, ar=self.ar, ma=self.ma,
                            exogenous=self.exogenous, intercept=self.intercept,
                            p=self.spec.p, q=self.spec.q, speed_lags=np.array(self.spec.speed_lags, dtype=int),
                            ridge=self.spec.ridge, ma_radius=self.ma_radius, proxy_order=self.proxy_order)


@dataclass(frozen=True)
class SparseLPV:
    speed_range: np.ndarray
    p: int
    stage1: np.ndarray
    stage2: np.ndarray
    a1: int
    a2: int
    b2: int

    def predict(self, x, rpm, fs):
        x, rpm = np.asarray(x, dtype=float), np.asarray(rpm, dtype=float)
        lo, hi = self.speed_range
        w = (rpm - lo) / (hi - lo)
        stage1 = _lpv_predict(x, w, self.a1, self.p, self.stage1)
        if len(self.stage2):
            proxy = np.nan_to_num(x - stage1)
            predicted = _lpv_predict(x, w, self.a2, self.p, self.stage2, proxy, self.b2, self.a1)
        else:
            predicted = stage1
        predicted[:SCORE_START] = np.nan
        return predicted

    def parameter_count(self):
        return int(np.count_nonzero(self.stage1) + np.count_nonzero(self.stage2))

    def save(self, path):
        np.savez_compressed(path, family="sparse_lpv_arma" if len(self.stage2) else "sparse_lpv_ar",
                            rpm_range=self.speed_range, terms=self.p, stage1=self.stage1,
                            stage2=self.stage2, orders=np.array([self.a1, self.a2, self.b2]))


def load(path):
    arrays = _array_checkpoint(path)
    family = str(arrays["family"])
    if family in LPV_KEYS:
        a1, a2, b2 = map(int, arrays["orders"])
        return SparseLPV(arrays["rpm_range"], int(arrays["terms"]), arrays["stage1"],
                         arrays["stage2"], a1, a2, b2)
    spec = FitSpec(family, int(arrays["p"]), int(arrays["q"]), tuple(map(int, arrays["speed_lags"])),
                   float(arrays["ridge"]))
    return StatisticalModel(spec, arrays["ar"], arrays["ma"], arrays["exogenous"],
                            float(arrays["intercept"]), float(arrays["ma_radius"]), int(arrays["proxy_order"]))


def _fit_arrays(segments, difference=0):
    """Keep initialization and lag construction within each raw training slice."""
    return [
        (np.r_[0., np.diff(x)] if difference else x, rpm, np.arange(FIT_WARMUP, len(x)))
        for x, rpm in segments if len(x) > FIT_WARMUP
    ]


def _proxy_innovations(segments):
    regressors = np.concatenate([_lag(x, i, _lagrange(PROXY_ORDER)) for x, _, i in segments])
    target = np.concatenate([x[i] for x, _, i in segments])
    beta = _lstsq(regressors, target)
    return [signal.lfilter(np.r_[1., -beta], [1.], x) for x, _, _ in segments]


def _fit_ar(target, design, spec):
    if not np.any(spec.speed_lags) and not spec.ridge:
        return StatisticalModel(spec, _lstsq(design, target))

    # Centred and scaled ARX variables avoid a speed-unit-dependent ridge penalty.
    mean = np.mean(design, axis=0)
    scale = np.maximum(np.std(design, axis=0), 1e-8)
    normalized = (design - mean) / scale
    yc = target - target.mean()
    gram = normalized.T @ normalized
    gram.flat[::len(gram) + 1] += spec.ridge * len(normalized)
    beta = linalg.solve(gram, normalized.T @ yc, assume_a="pos", check_finite=False) / scale
    return StatisticalModel(spec, beta[:spec.p], exogenous=beta[spec.p:],
                            intercept=float(target.mean() - mean @ beta))


def fit_model(segments, spec, cached=None):
    views = _fit_arrays(segments, spec.d) if cached is None else cached[0]
    y = np.concatenate([x[i] for x, _, i in views])
    speed_parts = [np.asarray(spec.speed_lags)] if spec.speed_lags else []
    if spec.q:
        proxy = _proxy_innovations(views) if cached is None else cached[1]
        all_regressors = [
            np.column_stack((_lag(x, idx, _lagrange(spec.p)), _lag(e, idx, _lagrange(spec.q))))
            for (x, _, idx), e in zip(views, proxy, strict=True)
        ]
        fit = _lstsq(np.concatenate(all_regressors), y)
        ar, ma = fit[:spec.p], fit[spec.p:]
        radius = float(np.max(np.abs(np.roots(np.r_[1., ma]))))
        return StatisticalModel(spec, ar, ma, ma_radius=radius, proxy_order=PROXY_ORDER)

    if speed_parts:
        design = np.concatenate([np.column_stack((_lag(x, i, _lagrange(spec.p)), _lag(rpm, i, speed_parts[0])))
                                 for x, rpm, i in views])
    else:
        design = np.concatenate([_lag(x, i, _lagrange(spec.p)) for x, _, i in views])
    return _fit_ar(y, design, spec)


def _lpv_design(x, w, i, ar, p, innovations=None, ma=0):
    lags = _lag(x, i, _lagrange(ar))
    if ma:
        lags = np.column_stack((lags, _lag(innovations, i, _lagrange(ma))))
    basis = np.polynomial.legendre.legvander(2 * w[i - 1] - 1, p - 1)
    return basis, lags


def _lpv_predict(x, w, ar, p, beta, innovations=None, ma=0, stage1_ar=0):
    start = max(ar, stage1_ar + ma)
    indices = np.arange(start, len(x))
    basis, lags = _lpv_design(x, w, indices, ar, p, innovations, ma)
    pred = np.full(len(x), np.nan)
    pred[start:] = np.einsum("ti,ij,tj->t", lags, beta.reshape(ar + ma, p), basis, optimize=True)
    return pred


def _lpv_fixed_support_fit(x, w, a, b, p, support, e=None):
    indices = np.arange(SCORE_START, len(x))
    basis, lags = _lpv_design(x, w, indices, a, p, e, b)
    columns = np.flatnonzero(support != 0)
    design = lags[:, columns // p] * basis[:, columns % p]
    beta = np.zeros_like(support)
    beta[columns] = _lstsq(design, x[indices])
    return beta


def _refit_sparse(x, rpm, prior_path):
    arrays = _array_checkpoint(prior_path)
    a1, a2, b2, p, first = map(int, arrays["orders"])
    if first != SCORE_START:
        raise ValueError("Prior LPV support was selected under a different scoring protocol")
    low, high = arrays["rpm_range"]
    w = (rpm - low) / (high - low)
    tick = perf_counter()
    b1 = _lpv_fixed_support_fit(x, w, a1, 0, p, arrays["lpv_ar"])
    elapsed_first = perf_counter() - tick
    e1 = np.nan_to_num(x - _lpv_predict(x, w, a1, p, b1))
    b2_fit = _lpv_fixed_support_fit(x, w, a2, b2, p, arrays["lpv_arma"], e1)
    elapsed_second = perf_counter() - tick
    differences = (float(np.max(np.abs(b1 - arrays["lpv_ar"]))),
                   float(np.max(np.abs(b2_fit - arrays["lpv_arma"]))))
    if max(differences) > 1e-8:
        raise RuntimeError(f"Frozen sparse support refit did not reproduce the prior coefficients: {differences}")
    models = (SparseLPV(arrays["rpm_range"], p, b1, EMPTY, a1, 0, 0),
              SparseLPV(arrays["rpm_range"], p, b1, b2_fit, a1, a2, b2))
    return models, (elapsed_first, elapsed_second), differences


def _specifications():
    yield from (FitSpec("ar", p) for p in (16, 32, 64, 96, 160, 256, 320))
    for family in ("arma", "arima"):
        yield from (FitSpec(family, p, q) for p, q in product((32, 96, 160, 256), (2, 5, 8)))
    speed_sets = ((1,), (1, 17, 65), (1, 17, 65, 257))
    for p, steps, ridge in product((32, 96, 256), speed_sets, (.0002, .002)):
        yield FitSpec("arx", p, speed_lags=steps, ridge=ridge)


def _exact_recursive(x, model):
    d = np.r_[0., np.diff(x)] if model.spec.d else x
    er = signal.lfilter(np.r_[1., -model.ar], np.r_[1., model.ma], d)
    fit = d - er
    if model.spec.d:
        fit[1:] += x[:-1]
    return fit


def _causal_change(model, x, rpm, fs, prediction):
    cut = min(12000, len(x) - 1)
    context = x.copy()
    exog = rpm.copy()
    context[cut:] += 3
    exog[cut:] *= 1.01
    alternative = model.predict(context, exog, fs)
    return float(np.max(np.abs(prediction[SCORE_START:cut + 1] - alternative[SCORE_START:cut + 1])))



def _batch_search(train):
    candidates = tuple(_specifications())
    all_trials = []
    for fold, (a, b) in enumerate(FOLDS, start=1):
        slices = guarded_segments(train, a, b)
        prepared = {}
        for d in (0, 1):
            views = _fit_arrays(slices, d)
            prepared[d] = views, _proxy_innovations(views)
        for trial, spec in enumerate(candidates):
            start = perf_counter()
            fitted = fit_model(slices, spec, prepared[spec.d])
            fit_seconds = perf_counter() - start
            eligible = fitted.ma_radius < MA_ROOT_LIMIT
            value = mse(train.x[a:b], predict(fitted, train, stop=b)[a:b]) if eligible else None
            all_trials.append(dict(model=spec.family, candidate=trial, fold=fold, start=a, stop=b,
                                   mse=value, fit_s=fit_seconds, eligible=eligible, ma_radius=fitted.ma_radius,
                                   rejected_reason="" if eligible else "MA root magnitude outside allowed inverse filter margin",
                                   config=json.dumps(asdict(spec), sort_keys=True)))
        print(f"Statistical fold {fold}: {len(candidates)} candidates processed", flush=True)

    rankings = {family: [] for family in FAMILY_KEYS}
    for trial, spec in enumerate(candidates):
        rows = [row for row in all_trials if row["candidate"] == trial]
        if all(row["eligible"] for row in rows):
            loss = [row["mse"] for row in rows]
            rankings[spec.family].append((float(np.mean(loss)), trial, float(np.std(loss, ddof=1)), spec))
    for family in rankings:
        rankings[family].sort(key=lambda row: (row[0], row[1]))
    return rankings, all_trials


def _model_outputs(slug, model, train, valid, out, fit_seconds, selection):
    path = out / "models" / f"{slug}.npz"
    model.save(path)
    clean, timings = timed_predict(model, valid, repeats=7)
    train_prediction = predict(model, train)
    restored = predict(load(path), valid)
    replay_error = float(np.max(np.abs(clean[SCORE_START:] - restored[SCORE_START:])))
    causal_delta = _causal_change(model, valid.x, valid.rpm, valid.fs, clean)
    if replay_error > 1e-10 or causal_delta > 1e-10:
        raise RuntimeError(f"{slug} replay {replay_error:g} or causality {causal_delta:g} failed")
    row = dict(model=slug, fit_s=fit_seconds, **timings, parameters=model.parameter_count(),
               checkpoint_bytes=path.stat().st_size, selection=json.dumps(selection, sort_keys=True),
               train_mse=mse(train.x[SCORE_START:], train_prediction[SCORE_START:]),
               validation_mse=mse(valid.x[SCORE_START:], clean[SCORE_START:]), replay_max_abs=replay_error,
               causal_perturbation_max_abs=causal_delta)
    if isinstance(model, StatisticalModel) and len(model.ma):
        recursive = _exact_recursive(valid.x, model)
        delta = float(np.max(np.abs(recursive[SCORE_START:] - clean[SCORE_START:])))
        tail = float(np.sum(np.abs(_fir(model.ar, model.ma, model.spec.d, SCORE_START*4)[SCORE_START+1:])))
        if delta > 1e-10 or tail > 1e-10:
            raise RuntimeError(f"{slug} finite-history replay differs from its exact recursion")
        row.update(ma_radius=model.ma_radius, recursive_truncation_max_abs=delta,
                   checked_tail_l1=tail, maximum_raw_lag=SCORE_START)
    elif isinstance(model, StatisticalModel):
        row["maximum_raw_lag"] = max(model.spec.p, *model.spec.speed_lags) if model.spec.speed_lags else model.spec.p
    else:
        row["maximum_raw_lag"] = max(model.a1 + model.b2, model.a2)
    return row, clean, train_prediction


def run(data_dir: Path, out: Path, prior: Path):
    """Execute guarded candidate fitting, strict sparse support refit and replay."""
    out = Path(out)
    prior_run = Path(prior) / "outputs" / "runs" / "development"
    prior_checkpoint = prior_run / "checkpoints" / "conventional.npz"
    out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    train, valid = load_data(data_dir)
    rankings, trials = _batch_search(train)
    predictions, metrics, chosen = {}, [], {}
    for family, candidates in rankings.items():
        for mean, trial, sd, spec in candidates:
            start = perf_counter()
            model = fit_model([(train.x, train.rpm)], spec)
            elapsed = perf_counter() - start
            if model.ma_radius < MA_ROOT_LIMIT:
                break
        else:
            raise RuntimeError(f"No eligible full {family} refit")
        selection = asdict(spec)
        row, val_pred, train_pred = _model_outputs(family, model, train, valid, out, elapsed, selection)
        row.update(selected_candidate=trial, fold_mean_mse=mean, fold_sd=sd,
                   fit_scope="full_training_refit", ar_pole_radius=float(np.max(np.abs(np.roots(np.r_[1., -model.ar])))))
        predictions.update({family: val_pred, f"{family}__train": train_pred})
        metrics.append(row)
        chosen[family] = trial
        print(f"{family} {selection}: MSE {row['validation_mse']:.8f}", flush=True)
    for row in trials:
        row["selected"] = chosen[row["model"]] == row["candidate"]

    models, elapsed, coeff_deltas = _refit_sparse(train.x, train.rpm, prior_checkpoint)
    prior_predictions = _array_checkpoint(prior_run / "predictions" / "baseline.npz")
    replay = {}
    for slug, model, seconds, previous in zip(LPV_KEYS, models, elapsed, ("spa_lpv_ar", "spa_lpv_arma"), strict=True):
        selection = {"stage1_ar": model.a1, "stage2_ar": model.a2, "stage2_ma": model.b2,
                     "basis": "shifted_legendre", "basis_terms": model.p, "schedule_lag": 1,
                     "support_source": "prior_guarded_admm_checkpoint"}
        row, val_pred, train_pred = _model_outputs(slug, model, train, valid, out, seconds, selection)
        delta = float(np.max(np.abs(val_pred[SCORE_START:] - prior_predictions[previous][SCORE_START:])))
        if delta > 1e-8:
            raise RuntimeError(f"{slug} prediction differs from prior strict replay by {delta}")
        row.update(fit_scope="full_training_support_refit", protected_prior_prediction_max_abs=delta)
        metrics.append(row)
        replay[slug] = delta
        predictions.update({slug: val_pred, f"{slug}__train": train_pred})
        print(f"{slug} {selection}: MSE {row['validation_mse']:.8f}", flush=True)
    provenance = json.loads((prior_run / "metrics" / "conventional_fit.json").read_text(encoding="utf-8"))
    write_json(out / "statistical_sparse_provenance.json", dict(
        prior_checkpoint_sha256=digest(prior_checkpoint), stages=[provenance["stage1"], provenance["stage2"]],
        refitted_coefficient_max_abs=coeff_deltas, replay_prediction_max_abs=replay,
        note="Supports frozen by prior guarded tuning. New QR refit and timing include no new support selection."))
    write_table(out / "statistical_trials.csv", trials)
    write_table(out / "statistical_metrics.csv", metrics)
    np.savez_compressed(out / "statistical_predictions.npz", **predictions)
    return metrics


if __name__ == "__main__":
    import argparse
    from threadpoolctl import threadpool_limits

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prior", required=True, type=Path)
    args = parser.parse_args()
    with threadpool_limits(1):
        run(args.data_dir, args.output, args.prior)
