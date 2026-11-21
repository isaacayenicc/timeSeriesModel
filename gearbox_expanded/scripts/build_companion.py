"""Build the standalone model companion, comparative plots, PDF, and presentation addendum."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gearbox_expanded.study import ANCHORS, ANCESTOR, MODEL_NAMES
from gearbox_expanded.protocol import FAMILIES

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"statistical":"#2b5797","tabular":"#157f79","neural":"#8c4c8e","attention":"#be611c",
          "DHON reference":"#333b4c"}
FAMILY = {key: family for family, names in FAMILIES.items() for key in names}


def fmt(value, digits=4):
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def save(fig, out, name):
    for ext in ("png","pdf"):
        fig.savefig(out/f"{name}.{ext}", dpi=250, bbox_inches="tight")
    plt.close(fig)


def figures(out, table, paired, regimes, spectra):
    path = out/"figures"
    path.mkdir(exist_ok=True)
    palette = lambda k: COLORS[FAMILY.get(k, "DHON reference")]
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig, ax = plt.subplots(figsize=(9,8))
    ax.barh(table.display[::-1],table.mse[::-1],color=[palette(k) for k in table.model[::-1]])
    ax.set(xlabel="Development MSE (standardized vibration²)", xlim=(0,max(table.mse)*1.08))
    ax.grid(axis="x", alpha=.25)
    save(fig,path,"development_mse")

    subset = table.dropna(subset=["batch_median_s"])
    fig, ax = plt.subplots(figsize=(9,5))
    for _, row in subset.iterrows():
        ax.scatter(1e3*row.batch_median_s,row.mse,color=palette(row.model),s=58)
    for key in ("ar","linear","primary_dhon","archive_dhon_optimized_corrected","sparse_lpv_arma",
                "cnn","transformer","tft","random_forest","svr"):
        row = subset.set_index("model").loc[key]
        ax.annotate(row.display,(1000*row.batch_median_s,row.mse),xytext=(5,4),
                    textcoords="offset points",fontsize=8)
    ax.set(xscale="log",xlabel="Median full-record CPU time (ms, logarithmic scale)",
           ylabel="Development MSE (standardized vibration²)")
    ax.grid(alpha=.23)
    save(fig,path,"accuracy_cost")

    sub = paired[(paired.reference=="primary_dhon")&(paired.block_samples==640)]
    sub = sub.sort_values("delta_mse")
    y = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(9,8))
    for i, row in enumerate(sub.itertuples()):
        ax.plot([row.low95,row.high95],[i,i],color=palette(row.model),linewidth=2)
        ax.scatter(row.delta_mse,i,color=palette(row.model),s=27,zorder=3)
    ax.axvline(0,color="black",lw=1)
    ax.set(yticks=y,yticklabels=[MODEL_NAMES[k] for k in sub.model],
           xlabel="MSE minus primary DHON (95% paired 640-sample block interval)")
    ax.invert_yaxis()
    ax.grid(axis="x",alpha=.24)
    save(fig,path,"paired_intervals")

    selected = ["primary_dhon", "archive_dhon_optimized_corrected","sparse_lpv_arma","linear","cnn","tft"]
    speed = regimes.loc[regimes.group.str.startswith("rpm_quartile")].pivot(index="model",columns="group",values="mse").loc[selected]
    spec = spectra.assign(band=spectra.low_hz.astype(str)+"–"+spectra.high_hz.astype(str))
    spec = spec.pivot(index="model", columns="band",values="power").loc[selected]
    fig, ax = plt.subplots(1,2,figsize=(10,4),layout="constrained")
    for item in selected:
        ax[0].plot(range(1,5),speed.loc[item],marker="o",label=MODEL_NAMES[item])
        order = ["0–150","150–500","500–1000","1000–1200","1200–1600"]
        ax[1].plot(range(5),spec.loc[item,order],marker="o")
    ax[0].set(xlabel="RPM quartile",ylabel="MSE",xticks=range(1,5))
    ax[1].set(xlabel="Residual band (Hz)",ylabel="Welch band power",xticks=range(5),xticklabels=order)
    ax[1].tick_params(axis="x",labelrotation=30)
    ax[0].legend(fontsize=7)
    save(fig,path,"regions_and_bands")

    with np.load(out/"matched_validation.npz",allow_pickle=False) as z:
        ix = slice(6200,6550)
        fig,ax = plt.subplots(2,1,figsize=(9,5),sharex=True,layout="constrained")
        t = z["time_s"][ix]
        y = z["actual"][ix]
        ax[0].plot(t,y,label="Measured",color="#253342",lw=1.1)
        for k in ("primary_dhon","linear","cnn"):
            pred = z[k][ix]
            ax[0].plot(t,pred,label=MODEL_NAMES[k],lw=.8)
            ax[1].plot(t,y-pred,label=MODEL_NAMES[k],lw=.8)
        ax[0].set_ylabel("Standardized vibration")
        ax[1].set(xlabel="Time (s)",ylabel="Residual")
        ax[0].legend(ncol=2,fontsize=8)
        save(fig,path,"matched_trace")

    weights_path = out/"attention_tft_weights.npz"
    if weights_path.exists():
        with np.load(weights_path,allow_pickle=False) as z:
            lag = z["relative_lags"]
            attention = z["attention"].mean(axis=0)
            selection = z["variable_selection"].mean(axis=0)
        fig, ax = plt.subplots(2,1,figsize=(8,5),sharex=True,layout="constrained")
        ax[0].plot(lag,selection[:,0],label="Vibration")
        ax[0].plot(lag,selection[:,1],label="RPM")
        ax[0].legend()
        ax[0].set_ylabel("Mean variable weight")
        ax[1].plot(lag,attention,color="#be611c")
        ax[1].set(xlabel="Lag relative to forecast target",ylabel="Mean attention")
        save(fig,path,"tft_weights")


def table_md(frame, columns, labels, digits=4):
    def render(value):
        return fmt(value,digits) if isinstance(value,(float,np.floating)) else str(value).replace("|","\\|")
    return "\n".join(["| "+" | ".join(labels)+" |","| "+" | ".join("---" for _ in columns)+" |",
                      *["| "+" | ".join(render(v) for v in row)+" |" for row in
                        frame[columns].itertuples(index=False,name=None)]])


def local_plot(out, name, caption):
    relative = (out/"figures"/f"{name}.png").relative_to(ROOT).as_posix()
    return f"![{caption}](../{relative})"

def diagnostic_text(out):
    directory = out/"diagnostics"
    status_path = directory/"diagnostic_status.json"
    if not status_path.is_file():
        return "Diagnostics were not available for this build."
    status = json.loads(status_path.read_text())
    if status.get("status") != "complete":
        return "The diagnostic run had not completed when this view was built."
    parts = [
        "The fixed screen used one record every five in MATLAB order, plus the separately documented "
        "overlap audit record 63. Raw vibration and all 23 runnable predictors share the subset. "
        "A separate scan used all 200 records for the previous DHON, replayable archived DHON, "
        "sparse LPV-ARMA and lowest-guarded-MSE new model (Lasso), plus raw. "
        "The A/B index grouping is provisional. Acquisitions 61 and 63 are excluded from the summaries.",
    ]
    for scope in ("screen","full"):
        path = directory/f"fault_{scope}_group_summary.csv"
        if not path.exists():
            continue
        data = pd.read_csv(path)
        if "metric" in data:
            data = data[data.metric=="pmi_course"]
        wanted = ["model","median_A","median_B","median_gap_B_minus_A","conditional_auc_B_high"]
        if not all(col in data for col in wanted):
            continue
        data["model"] = data.model.map(lambda k:MODEL_NAMES.get(k,k))
        parts.append(f"### {scope.capitalize()} diagnostics\n\n" +
                     table_md(data,wanted,["Signal/model","Median A","Median B","B−A","Conditional AUC"]))
    parts.append("Compute angular position by trapezoidal RPM integration, then interpolate residuals onto "
                 "a 2,048-point/revolution grid. The Hilbert envelope is centered. For centered envelope "
                 r"$b$, the finite sample ratio is $r(k)=\sum_j b_jb_{j+k}/\sum_j b_j^2$ and "
                 "PMI is $r/(1-r)$. The primary mode uses the instructor's lag-2,047 indexing and "
                 "the mathematical correction uses lag 2,048. The unrelated 0.63-turn lag is a specificity check. "
                 "Scores are threshold-free because no confirmed label or calibration file was available.")
    parts.append("The stress comparison applies six fixed RPM perturbations separately to the predictor and the "
                 "diagnostic angle reconstruction, and jointly to both. Pulse trials compare paired residuals "
                 "with and without synthetic periodic and aperiodic injections. These controlled inputs "
                 "probe sensitivity and preservation, with limited evidence about actual crack severity.")
    stress = directory/"rpm_stress_summary.csv"
    if stress.exists():
        rows = pd.read_csv(stress)
        cols = ["model","mse_clean","mse_mean","mse_worst"]
        if all(col in rows for col in cols):
            subset = ["primary_dhon","archive_dhon_optimized_corrected","sparse_lpv_arma",
                      "lasso","linear","ridge","cnn","tft"]
            rows = rows.set_index("model").loc[subset].reset_index()
            rows["model"] = rows.model.map(lambda k:MODEL_NAMES.get(k,k))
            parts.append(table_md(rows,cols,["Model","Clean MSE","Mean stress MSE","Worst MSE"]))
            data = pd.read_csv(stress).set_index("model")
            parts.append(
                f"Linear Regression and Ridge selected the same physics library. Their clean errors "
                f"were {data.loc['linear','mse_clean']:.6f} and {data.loc['ridge','mse_clean']:.6f}, "
                f"while their worst errors were {data.loc['linear','mse_worst']:.6f} and "
                f"{data.loc['ridge','mse_worst']:.6f}. The matched library and large divergence under stress "
                "support regularization as a practical robustness control on this acquisition. It does not "
                "prove either model is robust to all out-of-distribution sensor failures."
            )
    pulses = directory/"pulse_preservation_summary.csv"
    if pulses.exists():
        rows = pd.read_csv(pulses)
        rows["model"] = rows.model.map(lambda k:MODEL_NAMES.get(k,k))
        cols = [c for c in ["model","pulse_kind","kind","cases","positive_pmi","median_correlation",
                             "median_pmi_delta","median_energy_retention"] if c in rows.columns]
        parts.append(table_md(rows,cols,[c.replace("_"," ") for c in cols]))
    return "\n\n".join(parts)


def build(out, pdf=True):
    table = pd.read_csv(out/"comparison_metrics.csv")
    paired = pd.read_csv(out/"paired_mse_intervals.csv")
    trials = pd.read_csv(out/"guarded_trials.csv")
    regimes = pd.read_csv(out/"by_operating_region.csv")
    spectra = pd.read_csv(out/"residual_spectral_bands.csv")
    figures(out,table,paired,regimes,spectra)
    values = table.set_index("model")
    dhon = values.loc["primary_dhon","mse"]
    sparse = values.loc["sparse_lpv_arma","mse"]
    ancestor = values.loc[ANCESTOR,"mse"]
    arch = values.loc["archive_dhon_optimized_corrected","mse"]
    new = table[table.fit_evidence=="New guarded selection and fit"]
    best_dev = new.iloc[0]
    best_fold = new.sort_values("fold_mean_mse").iloc[0]
    neural = table[table.model.isin(FAMILIES["neural"])].iloc[0]
    feasible = table.dropna(subset=["batch_median_s"]).sort_values(["batch_median_s","mse"])
    envelope = feasible["mse"].cummin().shift().fillna(np.inf)
    frontier = feasible.loc[feasible.mse < envelope,["model","display","mse","batch_median_s"]].copy()
    frontier["batch_ms"] = frontier["batch_median_s"] * 1000
    frontier.to_csv(out/"accuracy_latency_frontier.csv",index=False)
    frontier_table = table_md(frontier,["display","mse","batch_ms"],
                              ["Model","MSE","Median CPU ms"],5)
    regimes_new = regimes.loc[regimes.model.isin(set(new.model))]
    regime_best = (regimes_new.sort_values("mse").groupby("group",sort=True).first().reset_index())
    regime_best["model"] = regime_best.model.map(MODEL_NAMES)
    regime_best.to_csv(out/"new_model_regime_leaders.csv",index=False)
    regime_table = table_md(regime_best,["group","model","mse"],["Operating region","Lowest MSE among new fits","MSE"],5)
    result = table.copy()
    result["batch_ms"] = result.batch_median_s*1e3
    comparison = table_md(result,["display","fold_mean_mse","train_mse","mse","mae","batch_ms"],
                          ["Model","Guarded MSE","Train MSE","Dev MSE","Dev MAE","CPU ms"],5)
    diag = diagnostic_text(out)
    visible = paired[(paired.reference=="primary_dhon")&(paired.block_samples==640)].set_index("model")
    intervals = visible.loc[["linear","sparse_lpv_arma","cnn",ANCESTOR]]
    intervals = intervals.reset_index().assign(model=lambda x:x.model.map(MODEL_NAMES))
    interval_table = table_md(intervals,["model","delta_mse","low95","high95"],
                              ["Comparator","MSE minus DHON","95% low","95% high"],6)
    rows = []
    for fam, item in trials.groupby("family"):
        rows.append(f"{fam}: {len(item)} fold rows")
    summary = f"""# Results and interpretation

The selected DHON replay scored **{dhon:.6f}** on 22,208 official development targets. The strongest newly
tuned model on this record was **{best_dev.display}** ({best_dev.mse:.6f}).
**{best_fold.display}** had the lowest comparable guarded-training MSE among newly tuned families
({best_fold.fold_mean_mse:.6f}), then scored {best_fold.mse:.6f} on official development. This difference
between rankings is important because selecting across families with official validation would use the
development acquisition as tuning data. {neural.display} was the best of the matched patched neural group
on this development record ({neural.mse:.6f}).

All 21 requested families are in the table. Nineteen had a new guarded search and full-training fit.
The two sparse LPV methods had fresh coefficient refits on prior guarded supports. The three DHON
references are the previous primary checkpoint, the strongest runnable archived checkpoint, and the
strongest saved archived forecasts. The primary scored {100*(sparse-dhon)/sparse:.2f}% lower MSE
than sparse LPV-ARMA. The replayable archived checkpoint ({arch:.6f}) and forecast-only anchor
({ancestor:.6f}) remain lower than primary DHON. The external published LPV-ARMA value 0.2893
comes from a different segment and is not in this ranking.

{comparison}

Guarded rows are **{len(trials)}** across the new experiments ({'; '.join(rows)}), including candidates
retained as evidence but excluded by solver or inversion checks. Sparse LPV guarded comparisons refer
to the earlier experiment and are left blank in this column. CPU time is the median of seven calls on
a 7.5-second record from a loaded predictor. Fault diagnostic computation ran separately on the shared host during the benchmark. Train MSE is an in-sample diagnostic and is not a
generalization estimate.

## Paired error and operating conditions

{interval_table}

Intervals use the predeclared 640-sample moving-block design. The CSV also contains sensitivity to block
lengths of 320 and 1,280 samples. Confidence estimates are for this acquisition and comparisons are
exploratory across multiple families.

{local_plot(out,"regions_and_bands","Prediction error by RPM quartile and residual Welch band power.")}

{local_plot(out,"matched_trace","Observed and estimated vibration on fixed development targets, with residuals.")}

## What the model ranking suggests

- The selected physics-aware Linear and Ridge regressions are stronger than classical lag-only AR,
  conditional ARMA or additive ARX on official development. These differences combine representation
  and estimation. A time-only method can be cheaper and useful when RPM is absent.
- The guarded Lasso and Elastic Net performed strongly on the internal folds. The best official
  development score came from a different regularization choice and feature library. Fit on one
  acquisition and evaluate on another before attributing performance to shrinkage alone.
- The selected NARX adds a tanh residual learner to a physics-conditioned ridge predictor.
  Its development score does not establish a gain from nonlinearity over the independently tuned
  linear/ridge models. The exact selection metadata distinguish their input libraries and regularization.
- The matched five-neural-model suite preserves a raw-lag shortcut across heads. Transformer and TFT
  use distinct configurations and training row strides. Their errors show limited performance under these
  specific configurations and resource budgets. TFT performs poorly even in-sample, so the result
  does not establish an architecture-wide generalization penalty or a definitive upper performance limit.
- Prior DHON controlled comparisons isolated an adapter benefit (+0.014029 MSE when removed)
  and a much smaller learned-state benefit (0.000148 over fixed oscillator states). Those controls
  are stronger attribution evidence than rank-based comparisons with unrelated model families.

{local_plot(out,"accuracy_cost","Trade-off between development MSE and measured in-memory CPU batch cost.")}

{local_plot(out,"tft_weights","TFT average validation variable and historical attention weights.")}

## Inference trade-offs and operating variation

The table gives the empirical non-dominated rows for clean development error and loaded CPU batch time.
A row is dominated if another runnable model is at least as accurate and fast with a strict advantage
on either metric. Forecast-only archived arrays have no latency and are excluded.

{frontier_table}

The following regional winners are descriptive. It is invalid to pick a different predictor for each
validation stratum and then report the resulting composite as independent performance.

{regime_table}

## Fault diagnostics and robustness

{diag}

## Scope of verification

Retain all target predictions for independent numerical rescoring, and all numeric checkpoint arrays
for independent replay. The shared causal perturbation changes RPM and vibration together at scoring
index 12,375. Forecasts through that target must agree with the original record within numerical
tolerance. The frozen run records checkpoint hashes and preserves the target and input ordering.
Blind results are reserved for sigtest and sigfault2. A missing-input preflight records their availability.
No diagnosis threshold or blind score is inferred from the development results.
"""
    (ROOT/"docs/results.md").write_text(summary, encoding="utf-8")
    front = """---
title: "Gearbox Vibration Modelling"
subtitle: "MECG 7780 companion to DHON and 21 model families"
geometry: margin=0.8in
fontsize: 10pt
colorlinks: true
toc: true
---

# How to use this companion

Start with the results chapter to see the measured ranking and scope. Read the experimental protocol
before interpreting any score. The model chapters connect equations to actual source components and
candidate settings. The matched-analysis notebooks independently recalculate evidence from stored arrays.
See the bundled prior DHON report and textbook for the original source audit, course slides and raw-file
provenance. This follow-up adds the measured family comparisons and maintains the original physics model
as the principal research hypothesis.

"""
    pieces = [front, summary]
    for name in ("methods.md","models.md","neural.md","trees.md"):
        pieces.append((ROOT/"docs"/name).read_text(encoding="utf-8"))
    pieces.append(model_cards(table, trials, visible, out))
    if (ROOT/"README.md").exists():
        pieces.append("# Reproduction\n\n" + (ROOT/"README.md").read_text(encoding="utf-8").split("\n",1)[1])
    target = ROOT/"docs"/"Expanded_DHON_Companion.md"
    target.write_text("\n\n".join(pieces), encoding="utf-8")
    if pdf:
        subprocess.run(["pandoc",str(target),"--pdf-engine=xelatex","--from=markdown+tex_math_single_backslash",
                        f"--resource-path={ROOT/'docs'}","--output",str(out/"Expanded_DHON_Companion.pdf")],
                       check=True,cwd=ROOT)
    return table, best_dev, best_fold


def model_cards(table, trials, vis, out):
    diagnostic_paths = {"rpm": out/"diagnostics/rpm_stress_summary.csv",
                        "pmi": out/"diagnostics/fault_screen_group_summary.csv"}
    sources = {name: pd.read_csv(path) for name,path in diagnostic_paths.items() if path.exists()}
    rpm = sources.get("rpm", pd.DataFrame(columns=["model"])).set_index("model")
    pmi = sources.get("pmi", pd.DataFrame(columns=["model","metric"]))
    pmi = pmi.loc[pmi.metric=="pmi_course"].set_index("model")
    text = ["# Measured model profiles", "",
            "Each profile pairs the selected implementation with the measured errors, CPU timing and "
            "uncertainty relative to the primary DHON. The full selected configuration is portable JSON "
            "in the CSV. Candidate points are bounded search results."]
    for _, r in table.iterrows():
        slug = r.model
        text.extend(["",f"## {r.display}","",
                     f"{r.fit_evidence}. Train MSE {fmt(r.train_mse,6)}. Guarded MSE "
                     f"{fmt(r.fold_mean_mse,6)}. Development MSE {r.mse:.6f}, MAE {r.mae:.6f}, "
                     f"and signed bias {r.bias:.6f}."])
        if not pd.isna(r.batch_median_s):
            text.append(f"Batch time {1000*r.batch_median_s:.2f} ms, median of seven calls. "
                        f"RTF {r.batch_rtf:.4f}.")
        if slug in vis.index:
            s = vis.loc[slug]
            text.append(f"Paired model-minus-DHON MSE {s.delta_mse:+.6f}, 95% within-record interval "
                        f"[{s.low95:+.6f}, {s.high95:+.6f}].")
        if slug in rpm.index:
            sensor = rpm.loc[slug]
            text.append(f"Across six predictor RPM stresses, mean MSE {sensor.mse_mean:.6f}, "
                        f"worst {sensor.mse_worst:.6f}, {int(sensor.successful)} successful runs.")
        if slug in pmi.index:
            signature = pmi.loc[slug]
            text.append(f"On the common screening mask ({int(signature.comparison_record_count)} eligible "
                        f"records), the provisional B-minus-A median PMI gap was "
                        f"{signature.median_gap_B_minus_A:+.6f} and conditional AUC was "
                        f"{signature.conditional_auc_B_high:.4f}.")
        if isinstance(r.selection,str):
            selection = json.loads(r.selection)
            text.extend(["","Selected configuration", "", "```json",
                         json.dumps(selection,indent=2),"```"])
        subset = trials[trials.model==slug]
        if len(subset):
            text.append(f"Search record retained {len(subset)} held-block fits.")
    return "\n".join(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("outputs/expanded"))
    parser.add_argument("--no-pdf",action="store_true")
    args = parser.parse_args()
    build(args.out.resolve(),pdf=not args.no_pdf)
