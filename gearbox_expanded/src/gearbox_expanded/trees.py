"""Tree ensemble comparisons and pickle-free numerical inference."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from threadpoolctl import threadpool_limits


FAMILIES = ("random_forest", "gradient_boosting")
SEED = 7780
START = 1792


def fit_tree(family: str, params: dict, x: np.ndarray, y: np.ndarray):
    """Fit a single-output ensemble; matrix rows must already be causal."""
    if family == "random_forest":
        return RandomForestRegressor(n_jobs=1, random_state=SEED, **params).fit(x, y)
    if family == "gradient_boosting":
        return HistGradientBoostingRegressor(early_stopping=False, categorical_features=None,
                                             random_state=SEED, **params).fit(x, y)
    raise ValueError(f"Unknown tree family {family}")


def candidates(smoke: bool = False) -> dict[str, list[tuple[str, dict]]]:
    """Declare a finite configuration grid with CPU caps for forest fits."""
    forest = [
        ("temporal_compact", dict(n_estimators=32, max_depth=7, min_samples_leaf=12,
                       max_features=0.5, max_samples=0.7)),
        ("physics_compact", dict(n_estimators=32, max_depth=7, min_samples_leaf=12,
                         max_features=0.5, max_samples=0.7)),
        ("physics_compact", dict(n_estimators=48, max_depth=12, min_samples_leaf=8,
                         max_features=0.35, max_samples=0.7)),
        ("temporal_compact", dict(n_estimators=32, max_depth=10, min_samples_leaf=14,
                        max_features=0.35, max_samples=0.7)),
    ]
    gradient = [
        ("temporal_compact", dict(max_iter=100, learning_rate=0.1, max_leaf_nodes=15,
                       min_samples_leaf=30, l2_regularization=1.0)),
        ("physics_compact", dict(max_iter=100, learning_rate=0.1, max_leaf_nodes=15,
                         min_samples_leaf=30, l2_regularization=1.0)),
        ("physics_compact", dict(max_iter=170, learning_rate=0.075, max_leaf_nodes=31,
                         min_samples_leaf=24, l2_regularization=2.0)),
        ("temporal_compact", dict(max_iter=120, learning_rate=0.1, max_leaf_nodes=31,
                        min_samples_leaf=35, l2_regularization=3.0)),
    ]
    if smoke:
        forest = [("physics_compact", {**forest[0][1], "n_estimators": 6})]
        gradient = [("physics_compact", {**gradient[0][1], "max_iter": 12})]
    return {"random_forest": forest, "gradient_boosting": gradient}


@dataclass
class TreeEnsemble:
    """Compact shared node-table representation of a fitted forest or booster."""

    family: str
    feature_set: str
    column_count: int
    base_value: float
    weight: float
    roots: np.ndarray
    left: np.ndarray
    right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    value: np.ndarray
    missing_left: np.ndarray

    def predict_matrix(self, features: np.ndarray) -> np.ndarray:
        if features.ndim != 2 or features.shape[1] != self.column_count:
            raise ValueError(f"Tree ensemble expects {self.column_count} input features")
        # sklearn's classic tree image casts RF input to float32 before the split.
        if self.family == "random_forest":
            features = np.asarray(features, dtype=np.float32)
        result = np.full(features.shape[0], self.base_value)
        row = np.arange(len(features))
        for root in self.roots:
            node = np.full(len(features), root, dtype=np.intp)
            active = self.left[node] >= 0
            while active.any():
                p = node[active]
                q = features[row[active], self.feature[p]]
                takes_left = np.where(np.isnan(q), self.missing_left[p], q <= self.threshold[p])
                node[active] = np.where(takes_left, self.left[p], self.right[p])
                active = active & (self.left[node] >= 0)
            result += self.value[node]
        return result * self.weight

    def predict(self, x: np.ndarray, rpm: np.ndarray, fs: float) -> np.ndarray:
        """Return the full target signal with the common warm-up prefix missing."""
        from .tabular_features import matrix

        forecast = np.full(len(x), np.nan)
        if len(x) > START:
            targets = np.arange(START, len(x))
            forecast[targets] = self.predict_matrix(matrix(x, rpm, fs, targets, self.feature_set))
        return forecast

    def save(self, path: Path, config: dict) -> None:
        meta = {
            "schema": 1,
            "family": self.family,
            "feature_set": self.feature_set,
            "column_count": self.column_count,
            "base_value": self.base_value,
            "weight": self.weight,
            "config": config,
        }
        np.savez_compressed(path, meta=np.array(json.dumps(meta, sort_keys=True)),
                            roots=self.roots, left=self.left, right=self.right,
                            feature=self.feature, threshold=self.threshold,
                            value=self.value, missing_left=self.missing_left)


def export_estimator(family: str, feature_set: str, fitted) -> TreeEnsemble:
    """Convert trained estimators to numerical integer and floating point node arrays."""
    parts: dict[str, list[np.ndarray]] = {k: [] for k in
        ("left", "right", "feature", "threshold", "value", "missing_left")}
    roots = []
    offset = 0
    if family == "random_forest":
        trees = [estimator.tree_ for estimator in fitted.estimators_]
        base_value, weight = 0.0, 1.0 / len(trees)
    elif family == "gradient_boosting":
        # HistGradientBoosting stores numeric thresholds and already weighted leaf values.
        # The pinned sklearn version and native inference parity test guard this export.
        trees = [stage[0].nodes for stage in fitted._predictors]
        base_value, weight = float(fitted._baseline_prediction.item()), 1.0
    else:
        raise ValueError(f"Unknown tree family {family}")
    for tree in trees:
        roots.append(offset)
        if family == "random_forest":
            left, right = tree.children_left, tree.children_right
            columns, split = tree.feature, tree.threshold
            vals, missing = tree.value[:, 0, 0], tree.missing_go_to_left
            leaf = left < 0
        else:
            if np.any(tree["is_categorical"]):
                raise ValueError("Portable numeric trees cannot store categorical split nodes")
            left, right = tree["left"], tree["right"]
            columns, split = tree["feature_idx"], tree["num_threshold"]
            vals, missing = tree["value"], tree["missing_go_to_left"]
            leaf = tree["is_leaf"].astype(bool)
        for key, data in {
            "left": np.where(leaf, -1, left.astype(np.int64) + offset),
            "right": np.where(leaf, -1, right.astype(np.int64) + offset),
            "feature": columns, "threshold": split, "value": vals,
            "missing_left": missing.astype(bool),
        }.items():
            parts[key].append(data)
        offset += len(left)
    return TreeEnsemble(family, feature_set, int(fitted.n_features_in_),
                        base_value, weight, np.asarray(roots, dtype=np.int64),
                        **{key: np.concatenate(value) for key, value in parts.items()})


def load_ensemble(path: Path) -> TreeEnsemble:
    """Load trusted numeric arrays from a frozen experiment checkpoint."""
    with np.load(path, allow_pickle=False) as arrays:
        meta = json.loads(arrays["meta"].item())
        if meta["schema"] != 1 or meta["family"] not in FAMILIES:
            raise ValueError("Unexpected tree checkpoint schema")
        keys = ("roots", "left", "right", "feature", "threshold", "value", "missing_left")
        return TreeEnsemble(meta["family"], meta["feature_set"], meta["column_count"],
                            meta["base_value"], meta["weight"],
                            **{key: arrays[key] for key in keys})


def load(path: Path) -> TreeEnsemble:
    """Load a frozen tree predictor with record-level inference."""
    return load_ensemble(path)


def _loss(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.square(y - yhat)))


def _csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(data_dir: Path, out: Path, prior: Path, *, smoke: bool = False) -> None:
    """Search only sigtrain folds, refit winners, and report common-mask development scores."""
    from .protocol import load_data
    from .tabular_features import matrix, prepare_folds

    train, valid = load_data(data_dir)
    X, speed, fs = train.x, train.rpm, train.fs
    Y, speed_valid = valid.x, valid.rpm
    models = out / "models"
    models.mkdir(parents=True, exist_ok=True)
    train_indices = np.arange(START, len(X))
    valid_indices = np.arange(START, len(Y))
    predictions, trials, metrics = {}, [], []
    cache = {}
    with threadpool_limits(limits=1):
        for family, grid in candidates(smoke).items():
            means, sds = [], []
            for trial, (feature_set, params) in enumerate(grid):
                if feature_set not in cache:
                    cache[feature_set] = prepare_folds(X, speed, fs, feature_set)
                scores = []
                for fold, (z_fit, y_fit, z_eval, y_eval) in enumerate(cache[feature_set], 1):
                    tick = perf_counter()
                    trained = fit_tree(family, params, z_fit, y_fit)
                    score = _loss(y_eval, trained.predict(z_eval))
                    scores.append(score)
                    trials.append(dict(model=family, trial=trial, fold=fold, seed=SEED,
                                       feature_set=feature_set,
                                       params=json.dumps(params, sort_keys=True),
                                       n_fit=len(y_fit), n_eval=len(y_eval),
                                       mse=score, seconds=perf_counter() - tick))
                means.append(float(np.mean(scores)))
                sds.append(float(np.std(scores, ddof=1)))
            winning = int(np.argmin(means))
            feature_set, params = grid[winning]
            cfg = dict(trial=winning, feature_set=feature_set, seed=SEED, **params)
            t0 = perf_counter()
            z_fit = matrix(X, speed, fs, train_indices, feature_set)
            trained = fit_tree(family, params, z_fit, X[train_indices])
            model = export_estimator(family, feature_set, trained)
            fit_elapsed = perf_counter() - t0
            checkpoint = models / f"{family}.npz"
            model.save(checkpoint, cfg)
            artifact = load(checkpoint)
            cold_tick = perf_counter()
            valid_prediction = artifact.predict(Y, speed_valid, fs)
            cold_s = perf_counter() - cold_tick
            official = trained.predict(matrix(Y, speed_valid, fs, valid_indices, feature_set))
            discrepancy = float(np.max(np.abs(valid_prediction[START:] - official)))
            if discrepancy > 2e-12:
                raise RuntimeError(f"{family} native and portable predictions differ by {discrepancy:g}")
            fit_prediction = artifact.predict(X, speed, fs)
            durations = []
            for repeat in range(7):
                tick = perf_counter()
                artifact.predict(Y, speed_valid, fs)
                durations.append(perf_counter() - tick)
            metrics.append(dict(model=family, fit_s=fit_elapsed, cold_s=cold_s,
                                median_s=float(np.median(durations)), p95_s=float(np.quantile(durations, 0.95)),
                                iqr_s=float(np.ptp(np.quantile(durations, (0.25, 0.75)))),
                                parameters=len(artifact.left),
                                total_nodes=len(artifact.left),
                                leaf_count=int(np.count_nonzero(artifact.left < 0)),
                                checkpoint_bytes=checkpoint.stat().st_size,
                                selection=json.dumps(cfg, sort_keys=True), selected_candidate=winning,
                                train_mse=_loss(X[START:], fit_prediction[START:]),
                                fold_mean_mse=means[winning], fold_sd=sds[winning],
                                validation_mse=_loss(Y[START:], valid_prediction[START:]),
                                cpu_threads=1, checkpoint_replay_max_abs=discrepancy,
                                timings_s=json.dumps(durations)))
            predictions[family] = valid_prediction
            predictions[f"{family}__train"] = fit_prediction
            for row in trials:
                if row["model"] == family:
                    row["selected"] = row["trial"] == winning
            _csv(out / "trees_trials.csv", trials)
            _csv(out / "trees_metrics.csv", metrics)
            np.savez_compressed(out / "trees_predictions.npz", **predictions)
            print(f"{family}: fold {means[winning]:.6f}, development {metrics[-1]['validation_mse']:.6f}, "
                  f"refit {fit_elapsed:.3f}s", flush=True)
