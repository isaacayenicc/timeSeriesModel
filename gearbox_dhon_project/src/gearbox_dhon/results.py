"""Matched-target development metrics and paired time-block uncertainty."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats

START = 1792
PUBLISHED = {"ARI": (0.4049, 0.3658), "ARIMA": (0.4034, 0.3671),
             "Spa LPV-AR": (0.3001, 0.2911), "Spa LPV-ARMA": (0.3099, 0.2893)}


def scores(y, pred):
    """Score aligned finite arrays; the caller defines the uniform mask."""
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if y.ndim != 1 or y.shape != pred.shape or not np.all(np.isfinite(y - pred)):
        raise ValueError("Scoring requires aligned finite one-dimensional arrays")
    e = y - pred
    mse = float(np.mean(e ** 2))
    signs = e[e != 0] > 0
    npos = float(signs.sum())
    nneg = len(signs) - npos
    if npos and nneg and len(signs) > 1:
        runs = 1 + int(np.count_nonzero(signs[1:] != signs[:-1]))
        znum = runs - 1 - 2 * npos * nneg / len(signs)
        zvar = (2*npos*nneg*(2*npos*nneg-len(signs))) / (len(signs)**2*(len(signs)-1))
        p_runs = float(2 * stats.norm.sf(abs(znum) / np.sqrt(zvar))) if zvar > 0 else np.nan
    else:
        p_runs = np.nan
    centered = e - e.mean()
    acf = signal.correlate(centered, centered, method="fft")[len(e)-1:len(e)+64]
    acf = acf / acf[0] if acf[0] > 0 else np.full(65, np.nan)
    lags = np.arange(1, len(acf))
    q = len(e) * (len(e) + 2) * np.sum(acf[1:] ** 2 / (len(e) - lags))
    variance = np.var(y)
    return dict(count=len(e), mse=mse, rmse=np.sqrt(mse), mae=np.mean(abs(e)),
                r2=1-mse/variance if variance > 0 else np.nan, bias=np.mean(e),
                acf_lag1=acf[1], runs_p=p_runs, ljung_box_64_p=stats.chi2.sf(q, len(lags)))


def block_difference(difference, block=640, draws=2000, seed=7780):
    """Moving-block bootstrap of the mean of paired squared-error differences."""
    difference = np.asarray(difference, dtype=np.float64)
    if not 1 <= block <= len(difference):
        raise ValueError("Invalid block length")
    rng = np.random.default_rng(seed)
    n_full, remainder = divmod(len(difference), block)
    starts = rng.integers(0, len(difference)-block + 1, (draws, n_full + bool(remainder)))
    cumulative = np.r_[0, np.cumsum(difference)]
    sums = np.sum(cumulative[starts[:, :n_full]+block] - cumulative[starts[:, :n_full]], axis=1)
    if remainder:
        starts_partial = starts[:, -1]
        sums += cumulative[starts_partial+remainder] - cumulative[starts_partial]
    return np.quantile(sums/len(difference), [.025, .975])


def read_predictions(run, n, start=START):
    """Reject silently changed scoring support and duplicate method identifiers."""
    arrays = {}
    for name in ("baseline", "baselines", "comparators", "dhon"):
        source = run / "predictions" / f"{name}.npz"
        if not source.exists():
            continue
        with np.load(source, allow_pickle=False) as data:
            for key in data.files:
                yhat = data[key]
                if yhat.shape != (n,) or key in {"x", "t", "time", "actual", "rpm", "target", "fs", "indices", "score_start"}:
                    continue
                if key in arrays or not np.isfinite(yhat[start:]).all():
                    raise ValueError(f"{source.name}/{key}: duplicate identifier or missing scored targets")
                arrays[key] = yhat
    if not arrays:
        raise FileNotFoundError("No complete development prediction bundles")
    return arrays


def save_figure(fig, destination):
    for suffix in (".png", ".pdf"):
        fig.savefig(Path(destination).with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def analyze(run, record):
    """Regenerate tables and figures directly from prediction arrays."""
    metrics = run / "metrics"
    figures = run / "figures"
    for path in (metrics, figures):
        path.mkdir(parents=True, exist_ok=True)
    arrays = read_predictions(run, len(record.x))
    y, rpm, time = (np.asarray(a)[START:] for a in (record.x, record.rpm, record.t))
    rows, grouped, paired = [], [], []
    spa_key = next((k for k in ("spa_lpv_arma", "spa_lpv_arma_refit", "refit_spa_lpv_arma", "strict_sparse_lpv_arma",
                                  "sparse_lpv_arma", "archive_spa_lpv_arma") if k in arrays), None)
    prior_key = next((k for k in arrays if ("original" in k or "tau_context_extended" in k) and "dhon" in k and ("correct" in k or "slow" in k)), None)
    bin_edges = np.quantile(rpm, [0, .25, .5, .75, 1])
    quartile = np.digitize(rpm, bin_edges[1:-1])
    direction = np.diff(np.asarray(record.rpm), prepend=record.rpm[0])[START:] >= 0
    blocks = np.array_split(np.arange(len(y)), 6)
    for key, values in arrays.items():
        pred = values[START:]
        rows.append(dict(model=key, **scores(y, pred)))
        e2 = (y-pred)**2
        masks = [(f"rpm_q{k+1}", quartile == k) for k in range(4)]
        masks += [("nondecreasing_rpm", direction), ("decreasing_rpm", ~direction)]
        for group, mask in masks:
            if mask.any():
                grouped.append(dict(model=key, group=group, count=mask.sum(), mse=e2[mask].mean()))
        for k, ix in enumerate(blocks):
            grouped.append(dict(model=key, group=f"time_block_{k+1}", count=len(ix), mse=e2[ix].mean()))
        for ref in filter(None, (spa_key, prior_key)):
            if ref != key:
                ref_e2 = (y-arrays[ref][START:])**2
                diff = e2-ref_e2
                low, high = block_difference(diff)
                paired.append(dict(model=key, reference=ref, delta_mse=diff.mean(),
                                   improvement_pct=-100*diff.mean()/ref_e2.mean(), low_95=low,
                                   high_95=high, block_samples=640))
    summary = pd.DataFrame(rows).sort_values("mse")
    summary.to_csv(metrics / "summary.csv", index=False)
    pd.DataFrame(grouped).to_csv(metrics / "groups.csv", index=False)
    pd.DataFrame(paired).to_csv(metrics / "paired_block_bootstrap.csv", index=False)
    pd.DataFrame([(k, *v) for k, v in PUBLISHED.items()],
                 columns=["model", "published_train_mse", "published_test_mse"]).to_csv(
        metrics / "published_external.csv", index=False)
    np.savez_compressed(run / "predictions" / "development_targets.npz",
                        zero_based_indices=np.arange(START, len(record.x)), target=y, rpm=rpm, time_s=time)

    shortlist = list(dict.fromkeys([*summary.head(10).model, *filter(None, [spa_key, prior_key])]))
    displayed = summary.set_index("model").loc[shortlist].sort_values("mse", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#167a77" if "dhon" in k else "#64748b" for k in displayed.index]
    ax.barh([k.replace("_", " ") for k in displayed.index], displayed.mse, color=colors)
    ax.set_xlabel("Development MSE (standardized vibration²)")
    ax.grid(axis="x", alpha=.22)
    save_figure(fig, figures / "mseloss")

    first = next((k for k in summary.model if "dhon" in k and
                  not any(word in k for word in ("archive", "original", "local", "expanded"))), summary.model.iloc[0])
    view = slice(min(3200, len(y)//2), min(3600, len(y)))
    trace_keys = list(dict.fromkeys(filter(None, (first, spa_key, prior_key))))
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True, layout="constrained")
    ax[0].plot(time[view], y[view], color="black", lw=1, label="Measured")
    for key in trace_keys:
        ax[0].plot(time[view], arrays[key][START:][view], lw=.9, label=key)
        ax[1].plot(time[view], (y-arrays[key][START:])[view], lw=.9, label=key)
    ax[0].legend(fontsize=7)
    ax[0].set_ylabel("Vibration (standardized)")
    ax[1].set_ylabel("Residual (standardized)")
    ax[1].set_xlabel("Acquisition time (s)")
    save_figure(fig, figures / "trace")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for key in trace_keys:
        residual = y-arrays[key][START:]
        acf = signal.correlate(residual-residual.mean(), residual-residual.mean(), method="fft")[len(y)-1:len(y)+128]
        ax[0].plot(np.arange(len(acf)), acf/acf[0], label=key)
        freq, p = signal.welch(residual, fs=record.fs, nperseg=1024)
        ax[1].semilogy(freq, p)
    ax[0].set(xlabel="Lag (samples)", ylabel="Residual autocorrelation")
    ax[0].legend(fontsize=7)
    ax[1].set(xlabel="Frequency (Hz)", ylabel="Residual PSD (standardized²/Hz)")
    save_figure(fig, figures / "residual_structure")
    return summary
