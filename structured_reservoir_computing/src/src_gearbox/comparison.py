"""Reconcile matched arrays, paired uncertainty and figures without fitting."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from scipy import signal
from threadpoolctl import threadpool_limits

from .study import scores
from .study_protocol import PROJECT, MASK_HASH, START, STOP, TARGETS, atomic_json, atomic_npz, settings

BASE_NAMES = dict(ar="AR",arma="ARMA",arima="ARIMA",arx="ARX",narx="NARX",sparse_lpv_ar="Sparse LPV-AR",
    sparse_lpv_arma="Sparse LPV-ARMA",linear="Linear Regression",ridge="Ridge",lasso="Lasso",elastic_net="Elastic Net",
    svr="SVR",random_forest="Random Forest",gradient_boosting="Gradient Boosting",mlp="MLP",lstm="LSTM",
    gru="GRU",cnn="CNN",tcn="TCN",transformer="Time Series Transformer",tft="Temporal Fusion Transformer")
ANCHORS = dict(archived_src_corrected="Archived SRC (corrected)",archived_src_base="Archived SRC (base)",
    archived_src_local_corrected="Archived local SRC (corrected)",archived_src_earlier_corrected="Archived earlier SRC (corrected)",
    archived_src_earlier="Archived earlier SRC (base)")
PRIMARY = "primary_pi_src"


def verify_axes(reference, candidate):
    for key in ("target_indices","actual","rpm","time_s"):
        np.testing.assert_array_equal(reference[key],candidate[key])
    np.testing.assert_array_equal(reference["target_indices"],TARGETS)


def guards(acc):
    out = {}
    for i,idx in enumerate(np.array_split(np.arange(len(acc["rpm"])),6),1):
        out[f"time_block_{i}"] = idx
    for label, vector in [("rpm",acc["rpm"]),("energy",signal.lfilter(np.ones(128)/128,[1.],acc["actual"]**2))]:
        order = np.argsort(vector,kind="stable")
        out.update({f"{label}_quartile_{i}":inds for i,inds in enumerate(np.array_split(order,4),1)})
    speed = acc["rpm"]
    change = speed[64:]-speed[:-64]
    out.update(rpm_rising=np.flatnonzero(change>0)+64,rpm_nonrising=np.flatnonzero(change<=0)+64)
    return out


def blocked(delta, width, draws, seed):
    """Noncircular paired moving-block bootstrap with exact original target count."""
    n = len(delta)
    blocks,dangling = divmod(n,width)
    rng = np.random.default_rng(seed)
    cumulative = np.r_[0,np.cumsum(delta)]
    ix = rng.integers(0,n-width+1,(draws,blocks))
    iterations = (cumulative[ix+width]-cumulative[ix]).sum(axis=1)
    if dangling:
        ix2 = rng.integers(0,n-dangling+1,draws)
        iterations += cumulative[ix2+dangling]-cumulative[ix2]
    return np.quantile(iterations/n,[.025,.975])


def run(project):
    result = project/"results"
    expected = pd.read_csv(result/"ablations/src_ablations.csv").set_index("model")
    baselines = pd.read_csv(result/"metrics/baseline_comparison.csv").set_index("model")
    archived = pd.read_csv(result/"metrics/archived_src_metrics.csv").set_index("model")
    bundle_dir = result/"predictions"
    with np.load(bundle_dir/"src_sigvalid.npz",allow_pickle=False) as sf, \
         np.load(bundle_dir/"baselines_sigvalid.npz",allow_pickle=False) as bf, \
         np.load(bundle_dir/"archived_src_sigvalid.npz",allow_pickle=False) as af:
        verify_axes(sf,bf)
        verify_axes(sf,af)
        acc = {k:sf[k] for k in ("actual","rpm","time_s","target_indices")}
        arranged = [(PRIMARY,expected,sf,"new PI-SRC",expected.at[PRIMARY,"display"])]
        for key in BASE_NAMES:
            if key in baselines.index and key in bf:
                arranged.append((key,baselines,bf,"replayed comparator",BASE_NAMES[key]))
        for key in ANCHORS:
            if key in af and key in archived.index:
                arranged.append((key,archived,af,archived.at[key,"provenance"],ANCHORS[key]))
        arrays = {key:source[key] for key,_,source,_,_ in arranged}
    fresh_info = json.loads((result/"metrics/src_summary.json").read_text())
    rows = []
    for key,source,_,kind,display in arranged:
        frame = source.loc[key]
        p = arrays[key]
        got = scores(acc["actual"],p)
        if not np.isclose(got["mse"],frame.mse,rtol=0,atol=1e-12):
            raise ValueError(f"Saved versus recalculated MSE differs for {key}")
        if key==PRIMARY:
            costs = dict(train_mse=frame.train_mse,fit_s=frame.fit_s,warm_median_ms=frame.warm_median_ms,
                         warm_iqr_ms=frame.warm_iqr_ms,checkpoint_bytes=frame.checkpoint_bytes,parameters=frame.parameters)
        else:
            costs = dict(train_mse=frame.get("train_mse"),fit_s=frame.get("fit_s_prior"),
                         warm_median_ms=1000*frame.get("median_s",np.nan),warm_iqr_ms=1000*frame.get("iqr_s",np.nan),
                         checkpoint_bytes=frame.get("checkpoint_bytes"),parameters=frame.get("parameters"))
        row = got|costs|dict(model=key,display=display,scope=kind,mask_sha256=MASK_HASH,
                              target_start=START,target_stop=STOP,source_validation_sha256=fresh_info["development_sha256"])
        rows.append(row)
        atomic_npz(bundle_dir/key/"sigvalid.npz",target_indices=TARGETS,prediction=p,target=acc["actual"],
                   residual=acc["actual"]-p,rpm=acc["rpm"],time_s=acc["time_s"],target_mask_sha256=MASK_HASH)
    all_model = pd.DataFrame(rows).sort_values("mse").reset_index(drop=True)
    all_model.to_csv(result/"metrics/all_model_comparison.csv",index=False)
    (result/"metrics/all_model_comparison.json").write_text(all_model.to_json(orient="records",indent=2)+"\n")
    timing = all_model.loc[:,["model","display","scope","fit_s","warm_median_ms","warm_iqr_ms","parameters","checkpoint_bytes"]].copy()
    timing["batch_size"] = STOP
    timing["samples_per_second"] = STOP/(timing.warm_median_ms/1000)
    timing["batch_real_time_factor"] = timing.warm_median_ms/7500
    timing["device"] = "CPU"
    (result/"timings").mkdir(exist_ok=True)
    timing.to_csv(result/"timings/benchmark_table.csv",index=False)
    src_config = expected.reset_index().loc[:,["model","config_json","scope"]]
    base_config = pd.read_csv(result/"metrics/baseline_configs.csv")
    base_config["config_json"] = base_config["selected_config"].map(lambda value:json.dumps(json.loads(value),sort_keys=True))
    base_config["scope"] = "baseline checkpoint and explicit replay"
    pd.concat([src_config,base_config.loc[:,["model","config_json","scope"]]],ignore_index=True).to_csv(
        result/"metrics/all_model_configs.csv",index=False)

    squared = {k:(acc["actual"]-v)**2 for k,v in arrays.items()}
    regions = []
    for label,indices in guards(acc).items():
        for model,se in squared.items():
            if len(indices):
                regions.append(dict(model=model,group=label,n=len(indices),mse=float(se[indices].mean()),
                                    rpm_min=float(acc["rpm"][indices].min()),rpm_max=float(acc["rpm"][indices].max())))
    region_data = pd.DataFrame(regions)
    region_data.to_csv(result/"metrics/speed_and_time_strata.csv",index=False)
    boot = []
    cfg = settings("evaluation")
    for model,se in squared.items():
        if model == PRIMARY:
            continue
        delta = se-squared[PRIMARY]
        for block in cfg["bootstrap_blocks"]:
            lo,hi = blocked(delta,block,cfg["bootstrap_replicates"],cfg["random_seed"])
            boot.append(dict(model=model,reference=PRIMARY,delta_mse=float(delta.mean()),block_samples=block,
                             interval_low=float(lo),interval_high=float(hi),draws=cfg["bootstrap_replicates"]))
    pair = pd.DataFrame(boot)
    (result/"statistical_tests").mkdir(exist_ok=True)
    pair.to_csv(result/"statistical_tests/paired_moving_block.csv",index=False)
    m = all_model.set_index("model").mse
    anchor = "archived_src_corrected"
    prior = float(m[anchor])
    best_baseline = m.loc[m.index.intersection(BASE_NAMES)].idxmin()
    key_comparison = dict(required_baselines_checked=len(set(BASE_NAMES)&set(m.index)),missing_baselines=sorted(set(BASE_NAMES)-set(m.index)),
        primary_mse=float(m[PRIMARY]),anchored_src_mse=prior,
        primary_minus_anchor=float(m[PRIMARY]-prior),
        relative_improvement_over_archive_pct=100*(prior-float(m[PRIMARY]))/prior,
        best_required_baseline=best_baseline,best_required_baseline_mse=float(m[best_baseline]),
        relative_improvement_over_required_baseline_pct=100*float(m[best_baseline]-m[PRIMARY])/float(m[best_baseline]),
        target_indices_sha256=MASK_HASH,targets=len(TARGETS),validation_sse_recomputed=True)
    atomic_json(result/"metrics/comparison_summary.json",key_comparison)
    render(project,all_model,region_data,pair,acc,arrays)
    print(json.dumps(key_comparison,indent=2),flush=True)
    if key_comparison["missing_baselines"]:
        raise ValueError("Missing comparator coverage in authoritative ranking")


def render(project,summary,regions,pairs,axis,arrays):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                         "axes.titlesize":12,"legend.frameon":False,"savefig.facecolor":"white"})
    root = project/"figures/comparison"
    root.mkdir(parents=True,exist_ok=True)

    def save(fig,name):
        fig.savefig(root/f"{name}.png",dpi=320,bbox_inches="tight",pad_inches=.17)
        plt.close(fig)
    top = summary[summary.model.isin([PRIMARY,*BASE_NAMES,"archived_src_corrected"])].sort_values("mse",ascending=False)
    color = {PRIMARY:"#17675c","archived_src_corrected":"#945834","sparse_lpv_arma":"#5b71a0","linear":"#576474"}
    fig,ax = plt.subplots(figsize=(8,7.1),layout="constrained")
    ax.barh(top.display,top.mse,color=[color.get(k,"#8fa3ab") for k in top.model])
    ax.set(xlabel="Development MSE (standardized vibration squared)",xlim=(0,1.04*top.mse.max()))
    ax.grid(axis="x",alpha=.2)
    save(fig,"mse_ranking")

    leaders = [PRIMARY,"archived_src_corrected","linear","sparse_lpv_arma"]
    fig,(a,b) = plt.subplots(2,1,figsize=(8.3,4.25),sharex=True,layout="constrained")
    ix = slice(6300,6620)
    a.plot(axis["time_s"][ix],axis["actual"][ix],color="#30343c",linewidth=1.2,label="Measured")
    for key in (PRIMARY,"sparse_lpv_arma"):
        line = arrays[key][ix]
        a.plot(axis["time_s"][ix],line,color=color[key],label=top.loc[top.model==key,"display"].iloc[0],linewidth=.9)
        b.plot(axis["time_s"][ix],axis["actual"][ix]-line,color=color[key],linewidth=.9)
    a.set_ylabel("Vibration (standardized)")
    b.set(ylabel="Residual (standardized)",xlabel="Time (s)")
    a.legend(ncol=3,loc="lower left",bbox_to_anchor=(0,1.005))
    save(fig,"validation_trace")

    fig,ax=plt.subplots(figsize=(7.65,3.7),layout="constrained")
    for k in leaders:
        values = regions[(regions.model==k)&regions.group.str.startswith("rpm_quartile")].sort_values("group")
        ax.plot(range(1,5),values.mse,marker="o",color=color[k],label=top.loc[top.model==k,"display"].iloc[0])
    ax.set(xlabel="Measured RPM quartile (ascending)",ylabel="MSE (standardized vibration squared)",xticks=range(1,5))
    ax.grid(alpha=.24)
    ax.legend(ncol=2,loc="upper left")
    save(fig,"rpm_quartiles")

    named = ["archived_src_corrected","linear","ridge","sparse_lpv_arma","cnn","ar"]
    pp=pairs.query("block_samples == 640").set_index("model").loc[named]
    fig,ax=plt.subplots(figsize=(7.65,3.5),layout="constrained")
    ax.errorbar(pp.delta_mse,range(len(pp)),xerr=[pp.delta_mse-pp.interval_low,pp.interval_high-pp.delta_mse],
                fmt="o",capsize=3,color="#435d75")
    ax.set(yticks=range(len(pp)),yticklabels=[BASE_NAMES.get(k,ANCHORS.get(k)) for k in named],
           xlabel="Comparator MSE minus PI-SRC MSE (paired, 640 samples/block)")
    ax.axvline(0,linestyle="--",color="#53636f")
    ax.invert_yaxis()
    ax.grid(axis="x",alpha=.22)
    save(fig,"paired_intervals")

    timed = top.dropna(subset=["warm_median_ms"])
    fig,ax=plt.subplots(figsize=(7.65,4.1),layout="constrained")
    ax.scatter(timed.warm_median_ms,timed.mse,c=[color.get(k,"#8fa3ab") for k in timed.model])
    for item in timed.itertuples():
        positions = {PRIMARY:(-5,-15),"archived_src_corrected":(5,4),"linear":(-12,11),
                     "sparse_lpv_arma":(-18,-16),"ar":(4,6),"tft":(-8,-12)}
        if item.model in positions:
            ax.annotate(item.display,(item.warm_median_ms,item.mse),xytext=positions[item.model],
                        textcoords="offset points",fontsize=8)
    ax.set(xscale="log",xlabel="CPU batch median (ms, logarithmic)",ylabel="Development MSE")
    ax.grid(alpha=.21)
    save(fig,"accuracy_cost")

    fig,(a,b)=plt.subplots(1,2,figsize=(8.3,3.75),layout="constrained")
    for k in (PRIMARY,"archived_src_corrected","sparse_lpv_arma"):
        e=axis["actual"]-arrays[k]
        et=e-e.mean()
        corr=signal.correlate(et,et,method="fft")[len(e)-1:len(e)+128]
        a.plot(np.arange(1,len(corr)),corr[1:]/corr[0],label=top.loc[top.model==k,"display"].iloc[0],color=color[k])
        hz,psd=signal.welch(e,fs=3200,nperseg=1024)
        b.plot(hz,10*np.log10(psd),color=color[k])
    a.set(xlabel="Lag (samples)",ylabel="Residual autocorrelation")
    b.set(xlabel="Frequency (Hz)",ylabel="Residual PSD (dB re 1 standardized unit²/Hz)")
    a.legend(fontsize=8)
    save(fig,"residual_structure")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-dir",type=Path,default=PROJECT)
    a=p.parse_args()
    with threadpool_limits(limits=1):
        run(a.project_dir.resolve())


if __name__=="__main__":
    main()
