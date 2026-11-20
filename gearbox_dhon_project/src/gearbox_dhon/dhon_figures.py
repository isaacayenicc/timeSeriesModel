"""Regenerate figures for the oscillator derivation and training-only search."""

from pathlib import Path
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from .dhon import oscillator_recurrence, shaft_phase


def save(fig, folder: Path, name: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(folder / f"{name}.{suffix}", dpi=320, bbox_inches="tight")
    plt.close(fig)


def oscillator_diagram(folder: Path) -> None:
    """Draw the finite-state encoder and separate residual feedback path."""
    fig, ax = plt.subplots(figsize=(10.1, 4))
    ax.set(xlim=(0, 11), ylim=(0, 5.3))
    ax.axis("off")
    boxes = [
        (0.2, 2.14, 1.45, 1., "Past vibration\nand lagged RPM", "#e5effb"),
        (2.25, 3.68, 2.1, .95, "Lag / angle / LPV\nproxy innovations", "#e5effb"),
        (2.25, 1.92, 2.1, .95, "Finite oscillator\norder and decay", "#d8f1e7"),
        (4.95, 3.68, 2.1, .95, "Ridge-initialized\nLPV readout", "#e5effb"),
        (4.95, 1.92, 2.1, .95, "Adaptive mixture\nand bounded head", "#d8f1e7"),
        (7.65, 3.01, 1.5, 1., "Base\nprediction", "#edecf8"),
        (7.65, .25, 1.5, .95, "Past clipped\nbase errors", "#fce9dc"),
        (9.57, 3.01, 1.2, 1., "Adapted\nprediction", "#edecf8"),
    ]
    for x,y,w,h,label,color in boxes:
        box = FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.05,rounding_size=.15",
                             edgecolor="#334c60", facecolor=color, linewidth=1)
        ax.add_patch(box)
        ax.text(x + w/2,y + h/2,label,ha="center",va="center",fontsize=10,color="#152841")
    for start,end in [
        ((1.65,2.7),(2.22,4.1)),((1.65,2.6),(2.22,2.38)),
        ((4.35,4.15),(4.92,4.15)),((4.35,2.65),(4.92,3.77)),((4.35,2.38),(4.92,2.38)),
        ((7.05,4.15),(7.62,3.72)),((7.05,2.38),(7.62,3.3)),
        ((9.17,3.53),(9.53,3.53)),((8.4,3),(8.4,1.24)),((9.18,.8),(10.18,2.97))
    ]:
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=12,
                                     linewidth=1.15,color="#4d6478"))
    ax.text(8.5,2.05,"after target\narrival",fontsize=8,color="#4d6478",ha="left")
    ax.set_title("Strict lag-1 DHON inference",loc="left",fontweight="bold",fontsize=15)
    save(fig,folder,"dhon_architecture")


def oscillator_response(folder: Path) -> None:
    fs = 3200
    n = np.linspace(124., 166., 900)
    t = np.arange(len(n)) / fs
    phi = 2*np.pi*37*shaft_phase(n, fs)
    harmonic = np.cos(phi)
    pulse = np.zeros_like(n)
    pulse[225] = 1
    taus = np.array([[8., 32., 128.]])
    h = oscillator_recurrence(harmonic, phi[None,:], taus, 320)
    hp = oscillator_recurrence(pulse, phi[None,:], taus, 320)
    colors = ("#267984", "#0053A5", "#AA5323")
    fig, axs = plt.subplots(1, 2, figsize=(9.8, 3.5), constrained_layout=True)
    axs[0].plot(t*1000, n*37/60, color=colors[1])
    axs[0].set(xlabel="Time (ms)",ylabel="Mesh frequency (Hz)",title="Order 37 follows changing RPM")
    ax2 = axs[0].twinx()
    ax2.plot(t*1000, np.linalg.norm(h[:,0,2,:],axis=-1), color=colors[2], alpha=.8)
    ax2.set_ylabel("Quadrature magnitude",color=colors[2])
    for j,tau in enumerate(taus[0]):
        axs[1].plot((t-t[225])*1000, np.linalg.norm(hp[:,0,j,:],axis=-1),color=colors[j],
                    label=f"τ = {tau:g} samples")
    axs[1].set(xlim=(-4,115),xlabel="Time from impulse (ms)",ylabel="Quadrature magnitude",
               title="One-step delay and finite-memory decay")
    axs[1].legend(frameon=False,fontsize=8)
    save(fig,folder,"dhon_oscillator_response")


def run(run_dir: Path) -> None:
    folder=run_dir/"figures"
    folder.mkdir(exist_ok=True,parents=True)
    oscillator_diagram(folder)
    oscillator_response(folder)
    search=run_dir/"search"
    with (search/"dhon_ranking.csv").open(newline="") as file:
        ranking=list(csv.DictReader(file))
    palette={"A":"#607388","B":"#a55727","C":"#197785","D":"#5a539f","smoke":"#197785"}
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    compact=ranking[:8]
    axs[0].errorbar([float(row["mse"]) for row in compact],range(len(compact)),
                    xerr=[float(row["mse_fold_std"]) for row in compact],fmt="o",color="#197785",
                    ecolor="#6798a0",capsize=3)
    axs[0].set(yticks=range(len(compact)),yticklabels=[row["trial"].replace("_"," ") for row in compact],
               xlabel="Clean MSE ± sample SD across three folds",title="Eight lowest composite utilities")
    axs[0].invert_yaxis()
    for stage in set(row["stage"] for row in ranking):
        rows=[r for r in ranking if r["stage"]==stage]
        axs[1].scatter([int(r["trainable"]) for r in rows],[float(r["utility"]) for r in rows],
                       color=palette[stage],label=stage,s=34,alpha=.85)
    axs[1].set(xlabel="Trainable coefficients",ylabel="Composite objective",title="Trade-off across the screened design")
    axs[1].legend(frameon=False,title="Stage",fontsize=8)
    for ax in axs:
        ax.grid(axis="x",alpha=.16)
    save(fig,folder,"dhon_search_tradeoffs")
    trace=json.loads((search/"primary_dhon_learning.json").read_text())
    if trace:
        epoch=[row["epoch"] for row in trace]
        carrier=np.asarray([row["carriers"] for row in trace])
        taus=np.asarray([row["taus"] for row in trace])
        fig,axs=plt.subplots(1,3,figsize=(11,3.5),constrained_layout=True)
        for k in range(carrier.shape[1]):
            axs[0].plot(epoch,carrier[:,k]-carrier[0,k],linewidth=1.3)
        for d in range(taus.shape[2]):
            changes=taus[:,:,d]-taus[0,:,d]
            axs[1].plot(epoch,changes.mean(axis=1),label=f"{taus[0,0,d]:g} samples")
            axs[1].fill_between(epoch,changes.min(axis=1),changes.max(axis=1),alpha=.10)
        axs[0].set(xlabel="Epoch",ylabel="Change from initialized frequency",
                   title="Carrier shifts by oscillator")
        axs[1].set(xlabel="Epoch",ylabel="Change in τ (samples)",title="Decay shifts across carriers")
        axs[1].legend(frameon=False,fontsize=8)
        for key,label in (("carrier_gradient_norm","Carrier"),("decay_gradient_norm","Decay")):
            vals=np.asarray([row[key] for row in trace])
            axs[2].plot(np.asarray(epoch)[vals>0], vals[vals>0],"o-",label=label)
        axs[2].set(xlabel="Epoch",ylabel="Gradient norm",title="Automatic differentiation")
        axs[2].legend(frameon=False,fontsize=8)
        save(fig,folder,"dhon_parameter_trajectory")


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    run(parser.parse_args().run_dir)


if __name__ == "__main__":
    main()
