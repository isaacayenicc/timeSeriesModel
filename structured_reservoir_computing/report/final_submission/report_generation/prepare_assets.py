"""Stage quantitative evidence and make a small number of original figures."""
from argparse import ArgumentParser
import csv
from hashlib import sha256
from pathlib import Path
from shutil import copyfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat

ROOT=Path(__file__).resolve().parents[1]

# File names describe the published exhibit. Descriptions identify the
# originating data scope, which is also printed in the actual manuscript.
FIGURES={
 "3.1":("rig.png","original rig schematic; Chen et al., 2025, sec. 5.1"),
 "3.2":("workflow.png","original information-flow diagram; data/splits/protocol.json"),
 "3.3":("records.png","sigtrain.mat and sigvalid.mat, checked against input manifest"),
 "4.1":("spectrogram.png","figures/eda/eda_spectrogram.png; training only; 148-order ridge"),
 "4.2":("orders.png","figures/eda/eda_orders.png; training only; Welch spectra"),
 "5.1":("architecture.png","figures/report/architecture.png; reviewed with selected checkpoint"),
 "6.1":("selection.png","figures/optimization/selection_history.png; 41 configurations"),
 "8.1":("ranking.png","figures/comparison/mse_ranking.png; common-mask ranking"),
 "8.2":("trace.png","figures/comparison/validation_trace.png; fixed comparison excerpt"),
 "8.3":("residuals.png","figures/comparison/residual_structure.png; official development residuals"),
 "8.4":("rpm_quartiles.png","figures/comparison/rpm_quartiles.png; measured development RPM quartiles"),
 "8.5":("cost.png","results/metrics/all_model_comparison.csv; finite forecasts and recorded CPU batch"),
 "9.1":("fault_order.png","figures/fault/representative_order.png; descriptive pilot records"),
 "9.2":("fault_distribution.png","figures/fault/fault_group_distribution.png; eligible provisional cohort"),
 "9.3":("pulse.png","figures/fault/pulse_tradeoff.png; controlled development signal injections"),
 "10.1":("ablations.png","results/ablations/src_ablations.csv and identical_adapter_controls.csv"),
 "10.2":("rpm_stress.png","figures/fault/sensor_stress.png; eight development RPM interventions"),
}
TABLES=[
 "results/metrics/src_summary.json", "results/metrics/comparison_summary.json",
 "results/metrics/all_model_comparison.csv", "results/metrics/model_coverage.csv",
 "results/metrics/evaluation_mask.json", "results/metrics/published_paper_reference.csv",
 "results/metrics/local_statistical_reference.csv", "results/metrics/vibration_rpm_parity.csv",
 "results/metrics/speed_and_time_strata.csv", "results/metrics/baseline_seed_sensitivity_summary.csv",
 "results/optimization/selected_configs.json", "results/optimization/trials.csv",
 "results/optimization/fold_scores.csv", "results/statistical_tests/paired_moving_block.csv",
 "results/experiments/baseline_refit/fresh_refit_reconciliation.json",
 "results/eda/summary.json", "results/eda/extension_summary.json",
 "results/eda/tables/eda_rise_fall.csv", "results/data_inventory.json",
 "results/fault/fault_group_summary.csv", "results/fault/paired_pmi_bootstrap.csv",
 "results/fault/matlab_conformance.json", "results/fault/pulse_summary.csv",
 "results/fault/rpm_stress.csv", "results/fault/rpm_stress_summary.csv",
 "results/fault/fault_status.json", "results/ablations/src_ablations.csv",
 "results/ablations/identical_adapter_controls.csv", "results/ablations/earlier_only_sensitivity.csv",
 "results/timings/feature_materialization.csv", "results/manifests/runtime.json",
 "results/manifests/src_prediction_verification.json",
]
PALETTE={"blue":"#496A87","navy":"#203947","teal":"#21675C","amber":"#8C4D32","cell":"#EDF3F5"}


def save(fig,path):
    fig.savefig(path,dpi=350,bbox_inches="tight",pad_inches=.13)
    plt.close(fig)


def rig(path):
    fig,ax=plt.subplots(figsize=(7.8,2.7),layout="constrained")
    ax.set(xlim=(0,12),ylim=(0,4.1))
    ax.axis("off")
    def node(x,y,w,label,c):
        p=FancyBboxPatch((x,y),w,.92,boxstyle="round,pad=.06,rounding_size=.09",facecolor=c,
                         edgecolor=PALETTE["blue"],linewidth=1.05)
        ax.add_patch(p)
        ax.text(x+w/2,y+.46,label,ha="center",va="center",color=PALETTE["navy"],fontsize=9.4)
    def arrow(a,b,style="-"):
        ax.annotate("",b,a,arrowprops=dict(arrowstyle="->",color="#405767",lw=1.25,linestyle=style))
    ax.text(.14,3.82,"Mechanical path",size=11,weight="bold",color=PALETTE["navy"])
    for x,w,label,c in [(.15,1.7,"Drive motor",PALETTE["cell"]),(2.15,2.12,"Speed reduction",PALETTE["cell"]),
        (4.6,2.65,"Test gearbox\n37-tooth gear\n20-tooth pinion","#E2EEEA"),(7.58,1.83,"Speed increase",PALETTE["cell"]),
        (9.72,2.12,"Alternator load",PALETTE["cell"])]:
        node(x,2.55,w,label,c)
    for a,b in [(1.87,2.15),(4.31,4.6),(7.28,7.58),(9.45,9.72)]:
        arrow((a,3.02),(b,3.02))
    node(.4,.28,3.6,"Input-bearing acceleration\nx-direction vibration",PALETTE["cell"])
    node(7.9,.28,3.6,"Optical encoder\nMeasured input-shaft RPM",PALETTE["cell"])
    arrow((5.25,2.50),(3.95,1.26),"--")
    arrow((6.4,2.50),(7.98,1.26),"--")
    save(fig,path)


def workflow(path):
    fig,ax=plt.subplots(figsize=(9.6,2.4),layout="constrained")
    ax.set(xlim=(0,12),ylim=(0,3))
    ax.axis("off")
    def node(x,y,w,label,fill):
        box=FancyBboxPatch((x,y),w,.83,boxstyle="round,pad=.055,rounding_size=.09",
                           facecolor=fill,edgecolor=PALETTE["blue"],linewidth=1.1)
        ax.add_patch(box)
        ax.text(x+w/2,y+.42,label,ha="center",va="center",fontsize=10,color=PALETTE["navy"])
    def arrow(a,b):
        ax.annotate("",b,a,arrowprops=dict(arrowstyle="->",lw=1.2,color=PALETTE["navy"]))
    node(.12,1.5,2.05,"sigtrain\nTraining only",PALETTE["cell"])
    node(2.64,1.5,2.05,"Guarded folds\nFit and select",PALETTE["cell"])
    node(5.14,1.5,2.05,"Frozen PI-SRC\nReset per record","#E2EEEA")
    node(8.02,2.06,3.7,"sigvalid\nMatched development scoring",PALETTE["cell"])
    node(8.02,.48,3.7,"Eligible candidate records\nOffline residual diagnostics",PALETTE["cell"])
    arrow((2.21,1.92),(2.60,1.92))
    arrow((4.73,1.92),(5.09,1.92))
    arrow((7.23,1.97),(7.99,2.45))
    arrow((7.23,1.84),(7.99,.90))
    save(fig,path)


def cost(project,path):
    frame=pd.read_csv(project/"results/metrics/all_model_comparison.csv").set_index("model").dropna(subset=["warm_median_ms"])
    fig,ax=plt.subplots(figsize=(8.0,4.35),layout="constrained")
    ax.scatter(frame.warm_median_ms,frame.mse,s=32,alpha=.68,color="#77939C")
    labels=[
        ("ar","AR",(.42,.493)),
        ("sparse_lpv_arma","Sparse LPV–ARMA",(13,.54)),
        ("archived_src_corrected","Archived SRC",(28,.285)),
        ("primary_pi_src","Selected PI-SRC",(230,.435)),
        ("linear","Linear",(1550,.421)),
        ("tft","TFT",(1700,.92)),
    ]
    for key,label,point in labels:
        row=frame.loc[key]
        col=PALETTE["teal"] if key=="primary_pi_src" else PALETTE["blue"]
        ax.scatter([row.warm_median_ms],[row.mse],s=43,color=col,zorder=4)
        ax.annotate(label,xy=(row.warm_median_ms,row.mse),xytext=point,
                    arrowprops=dict(arrowstyle="-",lw=.8,color=col),color=PALETTE["navy"],fontsize=9)
    ax.set(xscale="log",xlim=(.20,6300),ylim=(.255,1.018),
           xlabel="Warm CPU batch median (ms, logarithmic axis)",ylabel="Development MSE")
    ax.grid(alpha=.22)
    save(fig,path)


def records(raw,path):
    fig,ax=plt.subplots(2,2,figsize=(8.1,4.25),sharex=True,layout="constrained")
    for i,name in enumerate(("sigtrain","sigvalid")):
        z=loadmat(raw/f"{name}.mat",simplify_cells=True)[name]
        t,x,n=(np.asarray(z[k]) for k in ("t","x","spd"))
        assert t.size==24000 and float(z["fs"])==3200.
        ax[0,i].plot(t,x,color=PALETTE["blue"],lw=.27,alpha=.85)
        ax[1,i].plot(t,n,color=PALETTE["teal"],lw=1.3)
        ax[0,i].set(title=f"{'Training' if i==0 else 'Development'}  ·  {name}",ylim=(-5,5))
        ax[1,i].set(xlabel="Elapsed time (s)",ylim=(119,173))
        for a in ax[:,i]:
            a.grid(alpha=.19)
    ax[0,0].set_ylabel("Standardized vibration")
    ax[1,0].set_ylabel("Input speed (RPM)")
    save(fig,path)


def ablations(project,path):
    t=pd.read_csv(project/"results/ablations/src_ablations.csv").set_index("model")
    primary=t.at["primary_pi_src","mse"]
    items=[("Remove adapter, same weights","pi_src_adapter_off"),
           ("Remove angular delays","pi_src_no_angle"),("Remove angular delays and carriers","pi_src_no_angle_no_carrier"),
           ("Remove proxy innovations","pi_src_no_innovation"),("Remove carrier states","pi_src_no_carrier"),
           ("Fixed-Hz states only","pi_src_fixed_hz_only"),("Remove acceleration","pi_src_no_acceleration"),
           ("Remove both training penalties","pi_src_no_augmentation"),("Use twelve mesh orders","pi_src_order12")]
    labels=list(reversed([label for label,_ in items]))
    vals=np.asarray(list(reversed([t.at[key,"mse"]-primary for _,key in items])))
    fig,a=plt.subplots(figsize=(8.5,4.15),layout="constrained")
    a.barh(labels,vals,color=[PALETTE["teal"] if n<0 else PALETTE["blue"] for n in vals])
    a.axvline(0,color=PALETTE["navy"],lw=.8)
    a.set(xlabel="Development MSE minus selected PI-SRC MSE")
    a.grid(axis="x",alpha=.22)
    save(fig,path)


def run(project,raw):
    images=ROOT/"assets/figures"
    data=ROOT/"assets/tables"
    for p in (images,data):
        p.mkdir(parents=True,exist_ok=True)
    for rel in TABLES:
        copyfile(project/rel,data/Path(rel).name)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10.5,"axes.spines.top":False,
                         "axes.spines.right":False,"legend.frameon":False})
    rig(images/"rig.png")
    workflow(images/"workflow.png")
    cost(project,images/"cost.png")
    records(raw,images/"records.png")
    ablations(project,images/"ablations.png")
    entries=[]
    for number,(name,source) in FIGURES.items():
        if source.startswith("figures/"):
            copyfile(project/source.split(";")[0],images/name)
        src=images/name
        with Image.open(src) as im:
            w,h=im.size
        entries.append([number,"assets/figures/"+name,source,w,h,sha256(src.read_bytes()).hexdigest()])
    with (ROOT/"evidence/figure_manifest.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f)
        w.writerow(["figure","asset","source","width_px","height_px","sha256"])
        w.writerows(entries)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--project-root",type=Path,default=ROOT.parents[1])
    p.add_argument("--raw-dir",type=Path,required=True)
    a=p.parse_args()
    run(a.project_root.resolve(),a.raw_dir.resolve())
