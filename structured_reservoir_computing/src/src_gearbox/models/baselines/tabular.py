"""Guarded search and portable prediction for the expanded regression families."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
from scipy.linalg import cho_factor, cho_solve, lstsq
from scipy.spatial.distance import cdist
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits

from .protocol import load_data, mse, timed_predict, write_json, write_table
from .tabular_features import FOLDS, START, matrix, prepare_folds, training_matrix


@dataclass(frozen=True)
class Config:
    """Training settings; only the method-relevant fields affect each estimator."""

    family: str
    feature_set: str
    alpha: float = 1.0
    l1_ratio: float = 0.5
    c: float = 3.0
    epsilon: float = 0.1
    gamma_scale: float = 0.4
    max_rows: int = 3000
    width: int = 20
    epochs: int = 28
    neural_alpha: float = 0.02


def candidate_grid():
    return {
        "linear": [Config("linear", kind) for kind in ("temporal", "rpm", "physics")],
        "ridge": [Config("ridge", kind, alpha=alpha)
                  for kind in ("temporal", "rpm", "physics", "physics_wide")
                  for alpha in (0.6, 15.0, 80.0)],
        "lasso": [Config("lasso", kind, alpha=alpha)
                  for kind in ("temporal", "physics", "physics_wide")
                  for alpha in (0.0015, 0.0045, 0.012)],
        "elastic_net": [Config("elastic_net", kind, alpha=alpha, l1_ratio=ratio)
                        for kind in ("temporal", "physics", "physics_wide")
                        for alpha in (0.0025, 0.01) for ratio in (0.25, 0.75)],
        "svr": [Config("svr", kind, c=c, gamma_scale=gamma, epsilon=epsilon, max_rows=rows)
                for kind in ("temporal_compact", "physics_compact")
                for c, gamma, epsilon, rows in ((2., .25, .08, 2200), (6., .25, .08, 3000),
                                               (4., .5, .10, 3000), (10., .14, .10, 4500))],
        "narx": [Config("narx", kind, alpha=alpha, width=width, epochs=epochs, neural_alpha=penalty)
                 for kind in ("physics", "physics_wide")
                 for alpha, width, epochs, penalty in ((15., 12, 18, .04), (15., 24, 28, .02), (80., 32, 36, .04))],
    }


def _standardize(x):
    mean = np.mean(x, axis=0)
    scale = np.maximum(np.std(x, axis=0), 1e-7)
    return (x - mean) / scale, mean, scale


def _ridge(z, y, penalty):
    gram = z.T @ z
    gram.flat[::len(gram) + 1] += penalty
    return cho_solve(cho_factor(gram, check_finite=False), z.T @ y, check_finite=False)


def _linear(z, y):
    # Rank-revealing QR avoids exploding coefficients on redundant physical features.
    w, _, rank, _ = lstsq(z, y, cond=1e-8, lapack_driver="gelsy", check_finite=False)
    return w, int(rank)


@dataclass
class Regressor:
    """Runtime-only predictor with inspectable, non-pickled parameters."""

    config: Config
    fs: float
    location: np.ndarray
    scale: np.ndarray
    intercept: float
    arrays: dict[str, np.ndarray]

    @property
    def parameters(self) -> int:
        names = {
            "linear": ("coef",), "ridge": ("coef",), "lasso": ("coef",),
            "elastic_net": ("coef",), "svr": ("support", "dual"),
            "narx": ("coef", "hidden_weight", "hidden_bias", "output_weight", "output_bias"),
        }[self.config.family]
        return 1 + sum(int(np.count_nonzero(self.arrays[k])) if self.config.family in ("lasso", "elastic_net")
                       else self.arrays[k].size for k in names)

    def _predict_matrix(self, main, compact=None):
        z = (main - self.location) / self.scale
        a = self.arrays
        if self.config.family == "svr":
            values = np.empty(len(z))
            for start in range(0, len(z), 512):
                destination = slice(start, start + 512)
                squared = cdist(z[destination], a["support"], "sqeuclidean")
                np.multiply(squared, -float(a["gamma"]), out=squared)
                np.exp(squared, out=squared)
                values[destination] = squared @ a["dual"]
            return values + self.intercept
        prediction = z @ a["coef"] + self.intercept
        if self.config.family == "narx":
            if compact is None:
                raise ValueError("NARX requires its exogenous nonlinear inputs.")
            h = (compact - a["compact_location"]) / a["compact_scale"]
            prediction += np.tanh(h @ a["hidden_weight"] + a["hidden_bias"]) @ a["output_weight"]
            prediction += float(a["output_bias"])
        return prediction

    def predict(self, x, rpm, fs):
        """Return an aligned per-record array, with the common warm-up marked NaN."""
        if not np.isclose(self.fs, fs):
            raise ValueError(f"Expected {self.fs} Hz, received {fs} Hz.")
        x, rpm = np.asarray(x), np.asarray(rpm)
        out = np.full(len(x), np.nan)
        indices = np.arange(START, len(x))
        if not len(indices):
            return out
        main = matrix(x, rpm, fs, indices, self.config.feature_set)
        compact = matrix(x, rpm, fs, indices, "physics_compact") if self.config.family == "narx" else None
        out[indices] = self._predict_matrix(main, compact)
        return out

    def save(self, path):
        np.savez_compressed(path, config=np.array(json.dumps(asdict(self.config))),
                            fs=np.array(self.fs), location=self.location, scale=self.scale,
                            intercept=np.array(self.intercept), **self.arrays)


def load(path: Path):
    """Restore a frozen model, dispatching numeric tree checkpoints when needed."""
    if Path(path).stem in {"random_forest", "gradient_boosting"}:
        from .trees import load as load_tree
        return load_tree(path)
    with np.load(path, allow_pickle=False) as artifact:
        config = Config(**json.loads(artifact["config"].item()))
        arrays = {k: artifact[k] for k in artifact.files
                  if k not in {"config", "fs", "location", "scale", "intercept"}}
        return Regressor(config, float(artifact["fs"]), artifact["location"],
                         artifact["scale"], float(artifact["intercept"]), arrays)


def _fit(config, fs, x, y, compact=None):
    status = dict(state="completed", n_fit=len(y), iterations=None, dual_gap=None)
    center = float(y.mean())
    desired = y - center
    z, mean, scale = _standardize(x)
    model = Regressor(config, fs, mean, scale, center, {})
    if config.family == "linear":
        coefficients, rank = _linear(z, desired)
        model.arrays["coef"] = coefficients
        status["state"] = f"rank={rank}"
    elif config.family == "ridge":
        model.arrays["coef"] = _ridge(z, desired, config.alpha)
    elif config.family in {"lasso", "elastic_net"}:
        common = dict(alpha=config.alpha, fit_intercept=False, max_iter=1250, tol=2e-4,
                      selection="cyclic", precompute=True)
        estimator = (Lasso(**common) if config.family == "lasso" else
                     ElasticNet(l1_ratio=config.l1_ratio, **common))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(z, desired)
        model.arrays["coef"] = estimator.coef_.copy()
        status.update(state="iteration_cap" if caught else "converged",
                      iterations=int(estimator.n_iter_), dual_gap=float(estimator.dual_gap_))
    elif config.family == "svr":
        # Fit an exact RBF model on a deterministic time-stratified subset.
        rows = np.linspace(0, len(y) - 1, min(len(y), config.max_rows), dtype=int)
        gamma = config.gamma_scale / z.shape[1]
        estimator = SVR(C=config.c, epsilon=config.epsilon, gamma=gamma, cache_size=120)
        estimator.fit(z[rows], y[rows])
        model.intercept = float(estimator.intercept_[0])
        model.arrays.update(support=estimator.support_vectors_,
                            dual=estimator.dual_coef_[0], gamma=np.array(gamma))
        status.update(state="converged" if estimator.fit_status_ == 0 else "solver_limit",
                      n_fit=len(rows), iterations=int(estimator.n_iter_))
    elif config.family == "narx":
        # Previous measured y is the autoregressive feedback. Fit only its
        # nonlinear readout on the residual of a physics-conditioned ARX fit.
        coef = _ridge(z, desired, config.alpha)
        z_compact, compact_location, compact_scale = _standardize(compact)
        error = desired - z @ coef
        estimator = MLPRegressor(hidden_layer_sizes=(config.width,), activation="tanh", solver="adam",
                                 alpha=config.neural_alpha, learning_rate_init=0.001, batch_size=256,
                                 max_iter=config.epochs, random_state=7780, n_iter_no_change=12,
                                 early_stopping=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(z_compact, error)
        model.arrays.update(coef=coef, compact_location=compact_location, compact_scale=compact_scale,
                            hidden_weight=estimator.coefs_[0], hidden_bias=estimator.intercepts_[0],
                            output_weight=estimator.coefs_[1].ravel(), output_bias=estimator.intercepts_[1][0])
        status.update(state="epoch_budget_reached" if caught else "training_plateau",
                      iterations=int(estimator.n_iter_))
    return model, status


def _selection_rank(config, rows):
    """Prefer all-fold convergence for coordinate-descent candidates."""
    failed = any(row["state"] == "iteration_cap" for row in rows)
    loss = float(np.mean([row["mse"] for row in rows]))
    return (failed, loss) if config.family in {"lasso", "elastic_net"} else (False, loss)


def _run_six(data_dir, out):
    train, valid = load_data(data_dir)
    models_dir = out / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics, trials, predictions = [], [], {}
    start = perf_counter()
    for model, configs in candidate_grid().items():
        ranks = []
        cache_key, prepare, compact = None, None, None
        for candidate, config in enumerate(configs):
            if cache_key != config.feature_set:
                prepare = prepare_folds(train.x, train.rpm, train.fs, config.feature_set)
                cache_key = config.feature_set
                if config.family == "narx" and compact is None:
                    compact = prepare_folds(train.x, train.rpm, train.fs, "physics_compact")
            candidate_rows = []
            for fold, ((a, b), (x, y, x_valid, y_valid)) in enumerate(zip(FOLDS, prepare), 1):
                t0 = perf_counter()
                x_extra = compact[fold-1][0] if compact is not None else None
                x_valid_extra = compact[fold-1][2] if compact is not None else None
                estimator, status = _fit(config, train.fs, x, y, x_extra)
                fit_s = perf_counter() - t0
                yhat = estimator._predict_matrix(x_valid, x_valid_extra)
                loss = mse(y_valid, yhat)
                candidate_rows.append(dict(model=model, candidate=candidate, fold=fold,
                                           start=a, stop=b, mse=loss, fit_s=fit_s, n_targets=len(y),
                                           config=json.dumps(asdict(config), sort_keys=True), **status))
            trials.extend(candidate_rows)
            ranks.append((*_selection_rank(config, candidate_rows), candidate,
                          float(np.std([row["mse"] for row in candidate_rows], ddof=1))))
        failed, fold_mse, selected_id, fold_sd = min(ranks)
        selected = configs[selected_id]
        t0 = perf_counter()
        x_train, target = training_matrix(train.x, train.rpm, train.fs, selected.feature_set)
        x_extra = training_matrix(train.x, train.rpm, train.fs, "physics_compact")[0] if model == "narx" else None
        estimator, info = _fit(selected, train.fs, x_train, target, x_extra)
        if info["state"] == "iteration_cap":
            raise RuntimeError(f"Selected {model} did not converge in final fit.")
        fit_s = perf_counter() - t0
        checkpoint = models_dir / f"{model}.npz"
        estimator.save(checkpoint)
        dev_pred, timings = timed_predict(estimator, valid)
        train_pred = estimator.predict(train.x, train.rpm, train.fs)
        restored = load(checkpoint)
        replayed = restored.predict(valid.x, valid.rpm, valid.fs)
        error = float(np.max(np.abs(replayed[START:] - dev_pred[START:])))
        if error > 1e-10:
            raise ValueError(f"{model} frozen weights differ from fitted predictions by {error}.")
        metrics.append(dict(model=model, fit_s=fit_s, **timings, parameters=estimator.parameters,
                            checkpoint_bytes=checkpoint.stat().st_size,
                            selection=json.dumps(asdict(selected), sort_keys=True),
                            selected_candidate=selected_id, fold_mean_mse=fold_mse, fold_sd=fold_sd,
                            n_targets=len(dev_pred)-START, train_mse=mse(train.x[START:], train_pred[START:]),
                            validation_mse=mse(valid.x[START:], dev_pred[START:]),
                            checkpoint_replay_max_abs=error, fit_state=info["state"],
                            fit_rows=info["n_fit"], fit_iterations=info["iterations"],
                            fold_selection_has_cap=failed))
        for row in trials:
            if row["model"] == model:
                row["selected"] = row["candidate"] == selected_id
        predictions[model] = dev_pred
        predictions[f"{model}__train"] = train_pred
        write_table(out / "tabular_metrics.csv", metrics)
        write_table(out / "tabular_trials.csv", trials)
        np.savez_compressed(out / "tabular_predictions.npz", **predictions)
        print(f"{model:16} fold MSE {fold_mse:.6f}  development MSE {metrics[-1]['validation_mse']:.6f}", flush=True)
    write_json(out / "tabular_six_run.json", dict(duration_s=perf_counter()-start, models=len(metrics),
                                                 fold_fits=len(trials), candidates=len(trials)//3,
                                                 threads=1, scoring_start=START, folds=FOLDS))


def merge_tree_results(out):
    """Merge disjoint six-model and tree evidence; allow repeated orchestration."""
    for stem in ("metrics", "trials"):
        left = out / f"tabular_{stem}.csv"
        with left.open(encoding="utf-8", newline="") as stream:
            core = [row for row in csv.DictReader(stream)
                    if row["model"] not in {"random_forest", "gradient_boosting"}]
        with (out / f"trees_{stem}.csv").open(encoding="utf-8", newline="") as stream:
            core.extend(csv.DictReader(stream))
        write_table(left, core)
    predictions = {}
    for stem in ("tabular", "trees"):
        with np.load(out / f"{stem}_predictions.npz", allow_pickle=False) as stored:
            predictions.update({key: stored[key] for key in stored.files})
    np.savez_compressed(out / "tabular_predictions.npz", **predictions)


def run(data_dir: Path, out: Path, prior: Path, part: str = "all"):
    """Run the finite tabular search and merge portable tree model evidence."""
    from . import trees

    out.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        if part != "merge":
            _run_six(data_dir, out)
        if part == "all":
            trees.run(data_dir, out, prior)
        if part != "six":
            merge_tree_results(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--prior", required=True, type=Path)
    parser.add_argument("--part", choices=("all", "six", "merge"), default="all")
    args = parser.parse_args()
    run(args.data_dir, args.out, args.prior, args.part)
