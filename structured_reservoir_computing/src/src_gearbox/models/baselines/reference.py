"""Conventional forecasts for standardized vibration with frozen speed scheduling."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
import csv

import numpy as np
from scipy import linalg, signal, stats
from ...data.io_matlab import find_input, load_record

START = 1792
FOLDS = ((4800, 7200), (11200, 13600), (18400, 20800))
FRACTIONS = np.geomspace(.1, .002, 6)
STRICT_MODELS = ("ari", "arima", "lpv_ar_dense", "lpv_arma_dense", "spa_lpv_ar", "spa_lpv_arma")


@dataclass(frozen=True)
class BaselineRecord:
    x: np.ndarray
    rpm: np.ndarray
    rpm_normalized: np.ndarray
    t: np.ndarray
    fs: float
    rpm_range: np.ndarray | None = None


def mat_record(path: Path, variable: str) -> BaselineRecord:
    raw = load_record(path)
    if raw.dataset != variable or raw.speed_normalized is None:
        raise ValueError("Expected the corresponding course record with normalized speed")
    rpm_range = (np.array([raw.rpm_min,raw.rpm_max]) if raw.rpm_min is not None and raw.rpm_max is not None else None)
    return BaselineRecord(raw.x,raw.rpm,raw.speed_normalized,raw.t,raw.fs,rpm_range)


def find_record(root: Path, stem: str) -> Path:
    match = find_input(root,stem)
    if match is None:
        raise FileNotFoundError(f"No course input for {stem}")
    return match


def lag(x: np.ndarray, order: int, indices: np.ndarray) -> np.ndarray:
    """Lag-major design with column zero equal to x[t-1]."""
    if np.min(indices, initial=len(x)) < order:
        raise ValueError("Target has insufficient lag history")
    return x[indices[:, None] - np.arange(1, order + 1)]


def speed_basis(w: np.ndarray, p: int, basis: str) -> np.ndarray:
    if basis == "legendre":
        return np.polynomial.legendre.legvander(2 * w - 1, p - 1)
    if basis == "monomial":
        return w[:, None] ** np.arange(p)
    raise ValueError(f"Unknown basis: {basis}")


def design(record, indices, a, p, e=None, b=0, basis="legendre", current=False):
    history = lag(record.x, a, indices)
    if b:
        history = np.column_stack((history, lag(e, b, indices)))
    w = speed_basis(record.rpm_normalized[indices - (not current)], p, basis)
    return (history[:, :, None] * w[:, None, :]).reshape(len(indices), (a + b) * p)


def linear_fit(x, y):
    beta, _, rank, _ = linalg.lstsq(x, y, cond=1e-10, lapack_driver="gelsy", check_finite=False)
    return beta, int(rank)


def aic_scan(record, indices, ar_grid, ma_grid=(0,), innovations=None, p=8):
    """Use shared, stable Legendre coordinates and ranked fallback for order search."""
    amax, bmax = max(ar_grid), max(ma_grid)
    x = design(record, indices, amax, p, innovations, bmax)
    y = record.x[indices]
    gram, rhs = x.T @ x, x.T @ y
    n, tss = len(y), y @ y
    rows = []
    for a in ar_grid:
        for b in ma_grid:
            ix = np.r_[np.arange(a * p), np.arange(amax * p, (amax + b) * p)]
            g, h = gram[np.ix_(ix, ix)], rhs[ix]
            method = "cholesky_legendre"
            try:
                r = linalg.cholesky(g, lower=True, check_finite=False)
                diag = np.diag(r)
                if diag.min() < 1e-5 * diag.max():
                    raise linalg.LinAlgError("Rank or conditioning diagnostic")
                z = linalg.solve_triangular(r, h, lower=True, check_finite=False)
                mse, rank = (tss - z @ z) / n, len(ix)
                factor_diag_ratio = float(diag.min() / diag.max())
            except linalg.LinAlgError:
                method = "rank_aware_qr"
                xi = x[:, ix]
                beta, rank = linear_fit(xi, y)
                mse = np.mean((y - xi @ beta) ** 2)
                factor_diag_ratio = None
            rows.append(dict(ar=a, ma=b, aic=float(n * np.log(max(mse, 1e-30)) + 2 * rank),
                             mse=float(mse), rank=rank, n=n, method=method,
                             factor_diag_ratio=factor_diag_ratio))
    return min(rows, key=lambda row: row["aic"]), rows


def admm_path(x, y, fractions, max_iter=1000, rtol=1e-4, initial_rho=.03):
    """Minimize ||Xb-y||²/(2N)+λ||b||₁ and verify dual feasibility and gap."""
    n, d = x.shape
    gram, rhs = x.T @ x / n, x.T @ y / n
    y2 = float(y @ y / n)
    identity = np.eye(d)
    rho = initial_rho
    factor = linalg.cho_factor(gram + rho * identity, lower=True, check_finite=False)
    z, u = np.zeros(d), np.zeros(d)
    for fraction in fractions:
        lam = np.max(np.abs(rhs)) * fraction
        for iteration in range(1, max_iter + 1):
            beta = linalg.cho_solve(factor, rhs + rho * (z - u), check_finite=False)
            old = z
            v = beta + u
            z = np.sign(v) * np.maximum(np.abs(v) - lam / rho, 0)
            u = u + beta - z
            primal, dual = np.linalg.norm(beta - z), rho * np.linalg.norm(z - old)
            e_primal = rtol * (.1 * np.sqrt(d) + max(np.linalg.norm(beta), np.linalg.norm(z)))
            e_dual = rtol * (.1 * np.sqrt(d) + rho * np.linalg.norm(u))
            if iteration == 1 or iteration % 10 == 0 or iteration == max_iter:
                gz = gram @ z
                residual2 = y2 - 2 * (rhs @ z) + z @ gz
                scale = min(1., lam / max(np.max(np.abs(rhs - gz)), 1e-30))
                pobj = .5 * residual2 + lam * np.abs(z).sum()
                dobj = scale * (y2 - rhs @ z) - .5 * scale ** 2 * residual2
                gap = max(float(pobj - dobj), 0.)
                if primal <= e_primal and dual <= e_dual and gap <= rtol * max(.05, pobj):
                    break
            if iteration % 25 == 0:
                new_rho = min(1e3, 2 * rho) if primal > 8 * dual else max(.00001, rho / 2) if dual > 8 * primal else rho
                if new_rho != rho:
                    u *= rho / new_rho
                    rho = new_rho
                    factor = linalg.cho_factor(gram + rho * identity, lower=True, check_finite=False)
        converged = primal <= e_primal and dual <= e_dual and gap <= rtol * max(.05, pobj)
        yield z.copy(), dict(fraction=float(fraction), lambda_normalized=float(lam), iterations=iteration,
                             primal=float(primal), dual=float(dual), gap=gap, objective=float(pobj),
                             support=int(np.count_nonzero(np.abs(z) > 1e-7)), converged=bool(converged))


def refit_support(x, y, beta):
    support = np.abs(beta) > 1e-7
    result = np.zeros_like(beta)
    if np.any(support):
        result[support], rank = linear_fit(x[:, support], y)
    else:
        rank = 0
    return result, rank


def fold_rows(n, a, b, start=START):
    targets = np.arange(start, n)
    train = targets[(targets < a - start) | (targets >= b + start)]
    return train, np.arange(a, b)


def choose_one_sd(rows):
    """Largest eligible penalty within one sample SD of the minimum mean."""
    viable = [(float(fraction), np.array([row["mse"] for row in group]))
              for fraction in sorted({r["fraction"] for r in rows}, reverse=True)
              if (group := [r for r in rows if r["fraction"] == fraction])
              and all(row["converged"] for row in group)]
    if not viable:
        raise RuntimeError("No candidate ADMM penalty converged on all folds")
    best = min(viable, key=lambda p: p[1].mean())[1]
    return next(p[0] for p in viable if p[1].mean() <= best.mean() + best.std(ddof=1))


def apply_lpv(record, beta, a, p, innovations=None, b=0, basis="legendre", current=False, e_start=0):
    """Evaluate polynomial coefficient curves, using only measured past vibration."""
    n = len(record.x)
    first = max(a, e_start + b if b else 0)
    targets = np.arange(first, n)
    coeffs = beta.reshape(a + b, p)
    w = speed_basis(record.rpm_normalized[targets - (not current)], p, basis)
    history = lag(record.x, a, targets)
    if b:
        history = np.column_stack((history, lag(innovations, b, targets)))
    pr = np.full(n, np.nan)
    pr[targets] = np.einsum("ti,ij,tj->t", history, coeffs, w, optimize=True)
    return pr


def fit_stage(record, train, ar, ma, proxy=None, p=8, fractions=FRACTIONS):
    model, aic = aic_scan(record, train, ar, ma, proxy, p)
    x = design(record, train, model["ar"], p, proxy, model["ma"])
    y = record.x[train]
    return model, aic, x, y, list(admm_path(x, y, fractions))


def stage_predict(record, model, beta, p, proxy=None, proxy_start=0):
    return apply_lpv(record, beta, model["ar"], p, proxy, model["ma"], e_start=proxy_start)


def add_fold(rows, fold_id, stage, record, heldout, model, path, p, proxy=None):
    vh = design(record, heldout, model["ar"], p, proxy, model["ma"])
    for beta, diag in path:
        score = float(np.mean((record.x[heldout] - vh @ beta) ** 2))
        rows.append(dict(fold=fold_id, stage=stage, ar=model["ar"], ma=model["ma"], mse=score, **diag))


def constant_fit(record, train, ari_order, arima_order, proxy_order, q):
    d = np.r_[0., np.diff(record.x)]
    a, _ = linear_fit(lag(d, ari_order, train), d[train])
    hp, _ = linear_fit(lag(d, proxy_order, train), d[train])
    e = signal.lfilter(np.r_[1., -hp], [1.], d)
    x = np.column_stack((lag(d, arima_order, train), lag(e, q, train)))
    coefs, rank = linear_fit(x, d[train])
    return dict(ari=a, arima_ar=coefs[:arima_order], arima_ma=coefs[arima_order:], arima_proxy=hp,
                constant_rank=rank)


def conventional_forecasts(record, arrays, current=False, selected=None, archived=None):
    """Replay native or historical fits, optionally evaluating just one branch.

    Set archived=True and current=False for the scheduling-only intervention
    on unchanged historical coefficients.
    """
    archived = current if archived is None else archived
    if "rpm_range" in arrays:
        lo, hi = arrays["rpm_range"]
        record = replace(record, rpm_normalized=(record.rpm - lo) / (hi - lo))
    selected = set(selected) if selected is not None else set(STRICT_MODELS)
    result = {}
    if {"ari", "arima"} & selected:
        d = np.r_[0., np.diff(record.x)]
        live = np.r_[np.nan, record.x[:-1]]
        if "ari" in selected:
            ar = signal.lfilter(np.r_[0., arrays["ari"]], [1.], d)
            result["ari"] = live + ar
        if "arima" in selected:
            if archived:
                # Archived ARIMA, initialized at the first observed difference.
                innovation = signal.lfilter(np.r_[1., -arrays["arima_ar"]],
                                            np.r_[1., arrays["arima_ma"]], d[1:])
                arima = np.r_[0., d[1:] - innovation]
            else:
                arima = signal.lfilter(np.r_[0., arrays["arima_ar"]], [1.], d)
                proxy = signal.lfilter(np.r_[1., -arrays["arima_proxy"]], [1.], d)
                arima += signal.lfilter(np.r_[0., arrays["arima_ma"]], [1.], proxy)
            result["arima"] = live + arima

    a, a2, b2, p = map(int, arrays["orders"][:4])
    if {"spa_lpv_ar", "spa_lpv_arma"} & selected:
        basis = "monomial" if archived else "legendre"
        p1 = apply_lpv(record, arrays["lpv_ar"], a, p, basis=basis, current=current)
        if "spa_lpv_ar" in selected:
            result["spa_lpv_ar"] = p1
        if "spa_lpv_arma" in selected:
            e = np.nan_to_num(record.x - p1)
            result["spa_lpv_arma"] = apply_lpv(record, arrays["lpv_arma"], a2, p, e, b2,
                                              basis=basis, current=current, e_start=a)
    if "dense_orders" in arrays and {"lpv_ar_dense", "lpv_arma_dense"} & selected:
        da, da2, db2 = map(int, arrays["dense_orders"])
        dp1 = apply_lpv(record, arrays["dense_ar"], da, p)
        if "lpv_ar_dense" in selected:
            result["lpv_ar_dense"] = dp1
        if "lpv_arma_dense" in selected:
            de = np.nan_to_num(record.x - dp1)
            result["lpv_arma_dense"] = apply_lpv(record, arrays["dense_arma"], da2, p, de, db2, e_start=da)
    return result


def fit_conventional(record, profile, run):
    ar_grid, arma_grid, ma_grid, p = ((24, 48), (12, 24), (8, 16), 4) if profile == "smoke" else ((96, 160, 220, 240, 260), (55, 80, 110, 135), (30, 50, 65, 75), 8)
    fractions = FRACTIONS[:3] if profile == "smoke" else FRACTIONS
    full = np.arange(START, len(record.x))
    rows, ic, fold_states = [], [], []
    for i, (a, b) in enumerate(FOLDS):
        tr, va = fold_rows(len(record.x), a, b)
        m, grid, x, y, path = fit_stage(record, tr, ar_grid, (0,), p=p, fractions=fractions)
        add_fold(rows, i, 1, record, va, m, path, p)
        ic.extend(dict(fold=i, stage=1, **r) for r in grid)
        fold_states.append((tr, va, m, path))
        print(f"LPV Stage I completed fold {i + 1}, AR {m['ar']}", flush=True)
    alpha1 = choose_one_sd(rows)
    for i, (tr, va, m, path) in enumerate(fold_states):
        x1 = design(record, tr, m["ar"], p)
        beta, _ = refit_support(x1, record.x[tr], next(v for v, diag in path if diag["fraction"] == alpha1))
        e = np.nan_to_num(record.x - stage_predict(record, m, beta, p))
        del x1
        m2, grid2, x2, y2, path2 = fit_stage(record, tr, arma_grid, ma_grid, e, p, fractions)
        add_fold(rows, i, 2, record, va, m2, path2, p, e)
        ic.extend(dict(fold=i, stage=2, **r) for r in grid2)
        print(f"LPV Stage II completed fold {i + 1}, ARMA {m2['ar'], m2['ma']}", flush=True)
    del x, y, x2, y2
    alpha2 = choose_one_sd([r for r in rows if r["stage"] == 2])
    write_rows(run / "search/lpv_guarded_admm.csv", rows)
    started = perf_counter()
    m1, grid, x, y, path = fit_stage(record, full, ar_grid, (0,), p=p, fractions=[alpha1])
    beta1, diag1 = path[0]
    if not diag1["converged"]:
        raise RuntimeError("Full-record Stage I did not converge")
    beta1, rank1 = refit_support(x, y, beta1)
    d1, dr1 = linear_fit(x, y)
    del x, y
    e = np.nan_to_num(record.x - stage_predict(record, m1, beta1, p))
    de = np.nan_to_num(record.x - stage_predict(record, m1, d1, p))
    stage1_seconds = perf_counter() - started
    started = perf_counter()
    m2, grid2, x, y, path = fit_stage(record, full, arma_grid, ma_grid, e, p, [alpha2])
    beta2, diag2 = path[0]
    if not diag2["converged"]:
        raise RuntimeError("Full-record Stage II did not converge")
    beta2, rank2 = refit_support(x, y, beta2)
    del x, y
    dm2, dense_grid = aic_scan(record, full, arma_grid, ma_grid, de, p)
    d2, dr2 = linear_fit(design(record, full, dm2["ar"], p, de, dm2["ma"]), record.x[full])
    stage2_seconds = perf_counter() - started
    ic.extend(dict(fold="full", stage=1, **r) for r in grid)
    ic.extend(dict(fold="full", stage=2, **r) for r in grid2)
    ic.extend(dict(fold="full_dense_proxy", stage=2, **r) for r in dense_grid)
    write_rows(run / "search/lpv_aic.csv", ic)
    started = perf_counter()
    orders = (32, 33, 40, 3) if profile == "smoke" else (275, 285, 310, 5)
    arrays = constant_fit(record, full, *orders)
    constant_seconds = perf_counter() - started
    if record.rpm_range is not None:
        lo, hi = record.rpm_range
        if not (hi > lo and np.allclose(record.rpm_normalized, (record.rpm - lo) / (hi - lo), rtol=0, atol=1e-12)):
            raise ValueError("Stored speed normalization is inconsistent with its calibration extrema")
        arrays["rpm_range"] = record.rpm_range
    arrays.update(lpv_ar=beta1, lpv_arma=beta2, dense_ar=d1, dense_arma=d2,
                  orders=np.array([m1["ar"], m2["ar"], m2["ma"], p, START]),
                  dense_orders=np.array([m1["ar"], dm2["ar"], dm2["ma"]]))
    fit = dict(profile=profile, stage1={**diag1, **m1, "refit_rank":rank1},
               stage2={**diag2, **m2, "refit_rank":rank2},
               stage1_full_seconds=stage1_seconds, stage2_full_seconds=stage2_seconds,
               arima_invertibility_root_max=float(np.max(np.abs(np.roots(np.r_[1., arrays["arima_ma"]])))),
               dense_ranks=[dr1, dr2], constant_full_seconds=constant_seconds,
               proxy_orders=orders, fold_common_start=START,
               note="Stage II folds use fold-specific Stage I fits at the selected training-only penalty fraction")
    return arrays, fit


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def load_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def runs_test(e):
    sign = e[np.isfinite(e) & (e != 0)] > 0
    n = len(sign)
    if n < 2:
        return np.nan
    n1 = np.count_nonzero(sign)
    expected = 1 + 2 * n1 * (n - n1) / n
    variance = (expected - 1) * (expected - 2) / (n - 1)
    actual = 1 + np.count_nonzero(sign[1:] != sign[:-1])
    return float(2 * stats.norm.sf(abs(actual - expected) / np.sqrt(variance))) if variance > 0 else np.nan


PUBLISHED = {
    "ari": (.4049, .3658, 0.),
    "arima": (.4034, .3671, 0.),
    "spa_lpv_ar": (.3001, .2911, .4356),
    "spa_lpv_arma": (.3099, .2893, .6233),
}


def _model_rows(tr, va, fit_pred, val_pred, current, orig_start=START):
    rows = []
    for name, forecast in val_pred.items():
        y = va.x[START:] - forecast[START:]
        old = va.x[orig_start:] - forecast[orig_start:]
        tr_diff = tr.x[orig_start:] - fit_pred[name][orig_start:]
        pub = PUBLISHED.get(name, (np.nan, np.nan, np.nan))
        rows.append(dict(model="archived_" + name if current else name, scheduling="current" if current and "lpv" in name else "strict",
                         first_index=START, n=len(y), mse=float(np.mean(y * y)), mae=float(np.mean(np.abs(y))),
                         runs_p=runs_test(y), historical_start=orig_start, historical_mse=float(np.mean(old * old)),
                         historical_train_mse=float(np.mean(tr_diff * tr_diff)), historical_runs_p=runs_test(old),
                         paper_train_mse=pub[0], paper_test_mse=pub[1], paper_runs_p=pub[2]))
    return rows



def paper_row_diagnostic(train, valid, run, basis="legendre"):
    """Current-RPM, five shuffled row folds at the verified archived orders."""
    from sklearn.model_selection import KFold

    ar, ar2, ma, p = 240, 110, 65, 8
    start = 316
    train_i = np.arange(start, len(train.x))
    valid_i = np.arange(start, len(valid.x))
    history = []
    proxy = None
    forecasts = {}
    coef_arrays = {}
    fractions = np.geomspace(.027, .001, 5)
    for stage, order, ma_order in ((1, ar, 0), (2, ar2, ma)):
        x = design(train, train_i, order, p, proxy, ma_order, basis, current=True)
        y = train.x[train_i]
        rows = []
        for k, (t, v) in enumerate(KFold(5, shuffle=True, random_state=20261120).split(x)):
            validation_design = x[v]
            for beta, diag in admm_path(x[t], y[t], fractions, max_iter=800):
                err = y[v] - validation_design @ beta
                rows.append(dict(stage=stage, fold=k + 1, ar=order, ma=ma_order, basis=basis,
                                 mse=float(np.mean(err * err)), **diag))
        history.extend(rows)
        write_rows(run / f"search/lpv_paper_row_cv_{basis}.csv", history)
        fraction = choose_one_sd(rows)
        result, info = next(admm_path(x, y, [fraction], max_iter=1000))
        if not info["converged"]:
            raise RuntimeError(f"Current-speed stage {stage} did not converge")
        coef, rank = refit_support(x, y, result)
        fit = apply_lpv(train, coef, order, p, proxy, ma_order, basis, current=True, e_start=ar)
        if stage == 1:
            proxy = np.nan_to_num(train.x - fit)
        vf = apply_lpv(valid, coef, order, p, forecasts.get("validation_proxy"), ma_order, basis, current=True, e_start=ar)
        if stage == 1:
            forecasts["validation_proxy"] = np.nan_to_num(valid.x - vf)
        forecasts[str(stage)] = vf
        coef_arrays[str(stage)] = coef
        history.append(dict(stage=stage, fold="full", ar=order, ma=ma_order, basis=basis,
                            mse=float(np.mean((train.x[train_i] - fit[train_i]) ** 2)), **info))
        print(f"Historical row-CV stage {stage} selected {info['support']} terms, rank {rank}", flush=True)
    write_rows(run / f"search/lpv_paper_row_cv_{basis}.csv", history)
    rows = []
    for stage, m in ((1, "spa_lpv_ar"), (2, "spa_lpv_arma")):
        e = valid.x - forecasts[str(stage)]
        rows.append(dict(model=m, basis=basis, scheduling="current", cv_policy="random_rows_diagnostic",
                         start=start, n=len(valid_i), mse=float(np.mean(e[start:] ** 2)),
                         start1792_mse=float(np.mean(e[START:] ** 2)), runs_p=runs_test(e[start:])))
    write_rows(run / f"metrics/paper_convention_{basis}.csv", rows)
    checkpoint = dict(lpv_ar=coef_arrays["1"], lpv_arma=coef_arrays["2"],
                      orders=np.array([ar, ar2, ma, p, start]), basis=np.array(basis))
    if train.rpm_range is not None:
        checkpoint["rpm_range"] = train.rpm_range
    np.savez_compressed(run / f"checkpoints/paper_row_cv_{basis}.npz", **checkpoint)
    return rows
