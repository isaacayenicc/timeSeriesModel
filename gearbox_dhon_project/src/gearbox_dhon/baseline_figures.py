"""Figures generated from conventional fit and development replay artifacts."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate

from gearbox_dhon.baselines import START, BaselineRecord, load_arrays


def conventional_figures(run: Path, valid: BaselineRecord):
    with (run / "search/lpv_aic.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    axes_ar = [r for r in rows if (r["fold"], r["stage"]) == ("full", "1")]
    axes_arma = [r for r in rows if (r["fold"], r["stage"]) == ("full", "2")]
    dest = run / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8), layout="constrained")
    axs[0].plot([int(r["ar"]) for r in axes_ar], [float(r["aic"]) for r in axes_ar], "o-", color="#17657d")
    axs[0].set(xlabel="AR order", ylabel="AIC", title="Stage I | LPV-AR")
    for ma in sorted({int(r["ma"]) for r in axes_arma}):
        selected = [r for r in axes_arma if int(r["ma"]) == ma]
        axs[1].plot([int(r["ar"]) for r in selected], [float(r["aic"]) for r in selected], "o-", label=f"MA {ma}")
    axs[1].set(xlabel="AR order", ylabel="AIC", title="Stage II | Sparse proxy")
    axs[1].legend(frameon=False)
    fig.savefig(dest / "lpv_order_selection.png", dpi=170)
    plt.close(fig)

    forecasts = load_arrays(run / "predictions/baseline.npz")
    fig, axs = plt.subplots(3, 1, figsize=(9.5, 6.0), sharex=False, layout="constrained")
    sel = slice(8000, 8120)
    axs[0].plot(valid.t[sel], valid.x[sel], color="#1b263b", label="Measured")
    for name, label, color in (("spa_lpv_arma", "Sparse LPV-ARMA", "#d17047"),
                               ("lpv_arma_dense", "Dense LPV-ARMA", "#17657d")):
        axs[0].plot(valid.t[sel], forecasts[name][sel], color=color, label=label)
        e = valid.x[START:] - forecasts[name][START:]
        acf = correlate(e - e.mean(), e - e.mean(), mode="full", method="fft")[len(e)-1:]
        axs[1].plot(np.arange(1, 161), acf[1:161] / acf[0], color=color, label=label)
        rpm = valid.rpm[START-1:-1]
        bins = np.quantile(rpm, np.linspace(0, 1, 5))
        groups = np.searchsorted(bins[1:-1], rpm, side="right")
        mse = [np.mean(e[groups == k] ** 2) for k in range(4)]
        axs[2].plot(np.arange(1, 5), mse, "o-", color=color, label=label)
    axs[0].set(xlabel="Time (s)", ylabel="Standardized vibration")
    axs[1].set(xlabel="Lag (samples)", ylabel="Residual ACF")
    axs[2].set(xlabel="Prior RPM quartile", ylabel="MSE", xticks=np.arange(1, 5))
    axs[0].legend(frameon=False, ncol=3, fontsize=8, loc="upper right")
    axs[1].axhline(0., color="#c0c0c0", lw=1, zorder=0)
    fig.savefig(dest / "lpv_development_comparison.png", dpi=170)
    plt.close(fig)


    arrays = load_arrays(run / "checkpoints/conventional.npz")
    a1, a2, b2, terms, _ = arrays["orders"]
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 4.6), layout="constrained", sharey=False)
    for ax, key, count, title in ((axs[0], "lpv_ar", a1, "Stage I coefficients"),
                                   (axs[1], "lpv_arma", a2 + b2, "Stage II coefficients")):
        weights = arrays[key].reshape(count, terms)
        nonzero = np.abs(weights[weights != 0])
        cap = np.quantile(nonzero, .97) if len(nonzero) else 1.
        im = ax.imshow(weights, aspect="auto", cmap="RdBu_r", vmin=-cap, vmax=cap, interpolation="nearest")
        ax.set(xlabel="Legendre degree", ylabel="Design lag row", title=title, xticks=np.arange(terms))
        fig.colorbar(im, ax=ax, label="Coefficient", shrink=.8)
    axs[1].axhline(a2 - .5, linestyle="--", color="#000000", linewidth=1)
    fig.savefig(dest / "lpv_coefficient_map.png", dpi=170)
    plt.close(fig)
