"""Refresh frozen comparison and training optimization figures."""
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src_gearbox.comparison import run
from src_gearbox.study_protocol import PROJECT


def search_figures(root):
    data = pd.read_csv(root/"results/optimization/trials.csv")
    valid=data.loc[data.status.eq("DONE")].copy()
    front=pd.read_csv(root/"results/optimization/pareto_front.csv")
    path=root/"figures/optimization"
    path.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(8.5,4.1),layout="constrained")
    for stage, rows in valid.groupby("stage"):
        ax.scatter(rows.parameters,rows.fold_mse,s=33,alpha=.75,label=f"Stage {stage}")
    ax.scatter(front.parameters,front.fold_mse,facecolor="none",edgecolor="#202d43",s=70,label="Pareto trial")
    ax.set(xlabel="Stored coefficients",ylabel="Guarded fold MSE")
    ax.legend(ncol=3,fontsize=9,frameon=False)
    ax.grid(alpha=.2)
    fig.savefig(path/"guarded_frontier.png",dpi=320)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8.5,4.1),layout="constrained")
    history=valid.utility.rolling(1).min().cummin()
    axes[0].scatter(valid.index+1,valid.utility,s=13,alpha=.42,c="#476b7a")
    axes[0].step(valid.index+1,history,where="post",color="#17675c")
    axes[0].set(xlabel="Trial sequence",ylabel="Guarded utility")
    folds=pd.read_csv(root/"results/optimization/fold_scores.csv")
    stable=folds.dropna(subset=["condition"])
    axes[1].scatter(stable.condition,stable.mse,s=11,alpha=.46,c="#6c8e86")
    axes[1].set(xscale="log",xlabel="Regularized Gram condition estimate",ylabel="Held-block MSE")
    for axis in axes:
        axis.grid(alpha=.2)
    fig.savefig(path/"selection_history.png",dpi=320)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir",type=Path,default=PROJECT)
    parser.add_argument("--only-search",action="store_true")
    args=parser.parse_args()
    search_figures(args.project_dir)
    if not args.only_search:
        run(args.project_dir)
