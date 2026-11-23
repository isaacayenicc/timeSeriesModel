"""Assemble reviewed chapters, verified citations, and canonical numeric tables."""
from pathlib import Path
import json
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "report_generation"
EV = ROOT / "evidence"
DATA = ROOT / "assets/tables"

FIGURES = {
    "3_1": ("rig", "Equipment and input-shaft measurement schematic. The mechanical path and sensor placement follow the source experiment."),
    "3_2": ("workflow", "Information flow for guarded fitting, frozen development comparison and candidate diagnostics."),
    "3_3": ("records", "Supplied training and development streams over their full 7.5 s duration. Upper axes show standardized vibration and lower axes show measured RPM."),
    "4_1": ("spectrogram", "Training STFT with speed-derived mesh trajectories and the 1.0–1.2 kHz band."),
    "4_2": ("orders", "Training Welch spectra on time-frequency and angular-order axes. Order labels refer to the input shaft."),
    "5_1": ("architecture", "Causal PI-SRC prediction system. Preceding vibration and speed drive the structured state, regularized readout and separate bounded correction."),
    "6_1": ("selection", "Guarded search progress and model conditioning across the executed training-only candidates."),
    "8_1": ("ranking", "Development MSE for the 21 required comparators, selected PI-SRC and the executable corrected archive. Appendix A contains the five archive comparisons."),
    "8_2": ("trace", "Matched measured and predicted vibration and residuals at native target indices 8,092 to 8,411. Both vibration ordinates are standardized."),
    "8_3": ("residuals", "Development-residual autocorrelation and Welch power spectra for selected PI-SRC and its two physical reference models."),
    "8_4": ("rpm_quartiles", "Development MSE by ascending measured RPM quartile. Each quartile contains 5,552 common targets."),
    "8_5": ("cost", "Matched development MSE and warm CPU batch median. Forecast-only archive rows lack time measurements."),
    "9_1": ("fault_order", "Representative angular vibration and residual envelope orders of eligible candidate acquisitions. Examples illustrate signal structure within provisional index groups."),
    "9_2": ("fault_distribution", "Lecturer-compatible PMI distributions for the verified 198-record common cohort. Groups are defined by provisional file indices."),
    "9_3": ("pulse", "Periodic injection waveform and preservation measures. Results are paired on the supplied development recording."),
    "10_1": ("ablations", "Change in development MSE versus selected PI-SRC on the common mask. Negative changes are retained as exploratory outcomes."),
    "10_2": ("rpm_stress", "Development scores for the RPM intervention suite. Each model uses the same frozen weights under clean and altered speed."),
}

NAMES = {
    "primary_pi_src":"Selected PI-SRC", "archived_src_corrected":"Archived corrected SRC",
    "linear":"Linear Regression", "ridge":"Ridge", "sparse_lpv_arma":"Strict sparse LPV–ARMA",
    "cnn":"CNN", "ar":"AR", "mlp":"MLP", "raw":"Raw vibration",
    "pi_src_adapter_off":"Same weights, adapter off", "pi_src_fixed":"Fixed-physics SRC",
    "pi_src_pure_mse":"Clean-MSE training-only selection", "pi_src_compact":"Compact configuration",
    "pi_src_no_angle":"No angular lags", "pi_src_no_carrier":"No carrier bank",
    "pi_src_no_angle_no_carrier":"No angular lags or carrier bank", "pi_src_no_innovation":"No innovation history",
    "pi_src_fixed_hz_only":"Fixed-Hz carriers only", "pi_src_order12":"Post hoc 12-order bank",
}


def named(path):
    return path.read_text(encoding="utf-8").strip()


def numeric(name, index=None):
    return pd.read_csv(DATA / name, index_col=index)


def table(header, rows):
    return "\n".join([
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
        *["| " + " | ".join(str(x).replace("|", r"\|") for x in row) + " |" for row in rows],
    ])


def num(value, digits=6, signed=False):
    if pd.isna(value):
        return "—"
    return (f"{float(value):+,.{digits}f}" if signed else f"{float(value):,.{digits}f}").replace("-", "−")


def exchange(source, number, replacement):
    pattern = rf"(Table {re.escape(number)}\. [^\n]+\n\n)(?:\|[^\n]*\n)+"
    out, count = re.subn(pattern, lambda m: m[1] + replacement + "\n", source, count=1)
    if count != 1:
        raise ValueError(f"Missing table number {number}")
    return out


def prediction_tables(chapters):
    ranked = numeric("all_model_comparison.csv", "model")
    target = ranked.at["primary_pi_src", "mse"]
    paired = numeric("paired_moving_block.csv").query("block_samples == 640").set_index("model")
    models = ["archived_src_corrected", "primary_pi_src", "linear", "ridge", "sparse_lpv_arma", "cnn"]
    rows = []
    for key in models:
        r = ranked.loc[key]
        delta = r.mse - target
        rows.append([NAMES[key], num(r.mse), num(r.rmse), num(r.mae),
                     num(delta) if key != "primary_pi_src" else "Reference",
                     num(100 * delta / r.mse, 2) if key != "primary_pi_src" else "Reference"])
    chapters = exchange(chapters, "8.1", table(["Predictor","MSE","RMSE","MAE","Difference","Gain (%)"], rows))
    chapters = exchange(chapters, "8.2", table(["Comparator","Mean paired difference","Lower 95%","Upper 95%"],
        ([NAMES[k],num(r.delta_mse),num(r.interval_low),num(r.interval_high)]
         for k in ["linear","ridge","sparse_lpv_arma","cnn","archived_src_corrected"] for r in [paired.loc[k]])))
    strata = numeric("speed_and_time_strata.csv")
    rpm = strata[strata["group"].str.startswith("rpm_quartile_")]
    pivot = rpm.pivot(index="group", columns="model", values="mse")
    chapters = exchange(chapters,"8.3",table(["RPM quartile","Targets","PI-SRC","Sparse LPV–ARMA","Archived SRC"],
        ([i,num(rpm.loc[rpm["group"].eq(key),"n"].iloc[0],0),
          *[num(r[k]) for k in ("primary_pi_src","sparse_lpv_arma","archived_src_corrected")]]
         for i,(key,r) in enumerate(pivot.iterrows(),1))))
    chapters = exchange(chapters, "8.4", table(["Predictor","Fit (s)","Batch (ms)","IQR (ms)","Coeffs.","Bytes"],
        ([NAMES[k],num(r.fit_s,3),num(r.warm_median_ms,3),num(r.warm_iqr_ms,3),num(r.parameters,0),num(r.checkpoint_bytes,0)]
         for k in ("ar","mlp","sparse_lpv_arma","archived_src_corrected","primary_pi_src","linear") for r in [ranked.loc[k]])))
    ablations = numeric("src_ablations.csv","model")
    variants = ["primary_pi_src","pi_src_pure_mse","pi_src_compact","pi_src_adapter_off","pi_src_no_angle","pi_src_no_carrier",
                "pi_src_no_angle_no_carrier","pi_src_no_innovation","pi_src_fixed_hz_only","pi_src_order12"]
    chapters = exchange(chapters,"10.1",table(["Control","MSE","Change","Fitted coefficients"],
        ([NAMES[k],num(r.mse),num(r.mse-target,signed=True) if k!="primary_pi_src" else "—",num(r.parameters,0)]
         for k in variants for r in [ablations.loc[k]])))
    stress = numeric("rpm_stress.csv").query("model == 'primary_pi_src' and scope == 'predictor'").set_index("condition")
    noise = next(i for i in stress.index if "vibration" in i)
    settings = [("Clean measured input",["clean"]),("−2% / +2% RPM bias",["bias_minus_2pct","bias_plus_2pct"]),
        ("Correlated 0.6% / independent 0.3% relative RPM jitter",["correlated_jitter_0p6pct","independent_jitter_0p3pct"]),
        ("RPM delayed 40 ms",["delay_40ms"]),("RPM quantized to 1 RPM",["quantized_1rpm"]),
        ("75 ms hold-last-value intervals",["dropout_hold_75ms"]),("Smooth profile warp",["smooth_profile_warp"]),
        ("Vibration noise SD 0.05",[noise])]
    return exchange(chapters,"10.2",table(["Input intervention","MSE"],
        ([label," / ".join(num(stress.at[k,"mse"]) for k in keys)] for label,keys in settings)))


def fault_tables(fault):
    summary = numeric("fault_group_summary.csv").query("metric == 'lecturer_compatible'")
    group_table = table(["Model or signal","Median A","Median B","Gap","Margin","AUC"],
        ([NAMES[r.model],num(r.median_A,5),num(r.median_B,5),num(r.median_gap_B_minus_A,5),
          num(r.margin_min_B_minus_max_A,5),num(r.conditional_auc_B_high,3)] for r in summary.itertuples()))
    fault = fault.replace("TABLE_9_1", "Table 9.1. Lecturer-compatible PMI on 198 eligible candidates. AUC and group statistics are conditional on the provisional file index.\n\n" + group_table)
    pulses = numeric("pulse_summary.csv").query("kind == 'periodic'").set_index("model")
    pulse_table = table(["Model or signal","Positive","PMI increase","Distortion","Energy ratio","Correlation"],
        ([NAMES[k],f"{int(r.positive_delta_count)}/{int(r.cases)}",num(r.median_pmi_delta,6),num(r.median_distortion,3),
          num(r.median_energy_retention,3),num(r.median_pulse_correlation,3)]
         for k in ["primary_pi_src","pi_src_adapter_off","pi_src_fixed","archived_src_corrected","sparse_lpv_arma","ridge","cnn"]
         for r in [pulses.loc[k]]))
    return fault.replace("TABLE_9_2", "Table 9.2. Paired pulse outcomes on five periodic development interventions. Entries other than counts are medians.\n\n" + pulse_table)


def appendices():
    ranking = numeric("all_model_comparison.csv")
    def scope(row):
        if row.model == "primary_pi_src":
            return "Primary fit"
        if row.model.startswith("archived_"):
            return "Saved forecasts" if pd.isna(row.warm_median_ms) else "Archive replay"
        return "Required family"
    a = table(["Model","Evidence","MSE","RMSE","MAE","CPU (ms)"],
        ([row.display,scope(row),num(row.mse),num(row.rmse),num(row.mae),num(row.warm_median_ms,2)] for row in ranking.itertuples()))
    paper_names={"ari":"ARI","arima":"ARIMA","spa_lpv_ar":"Sparse LPV–AR","spa_lpv_arma":"Sparse LPV–ARMA"}
    b = table(["Model","Paper training MSE","Paper test MSE"],
        ([paper_names[r.model],num(r.train_mse,4),num(r.authors_test_segment_mse,4)] for r in numeric("published_paper_reference.csv").itertuples()))
    c = table(["Convention","Speed timing","Start","Count","MSE","Common MSE"],
        ([r.convention.replace("_"," "),r.scheduling,num(r.start,0),num(r.n,0),num(r.development_mse),num(r.common_mask_mse)]
         for r in numeric("local_statistical_reference.csv").query("model == 'spa_lpv_arma'").itertuples()))
    return named(GEN/"appendices.md").replace("TABLE_A_1",a).replace("TABLE_B_1",b).replace("TABLE_B_2",c)


def bibliography(source):
    course = """@misc{mecgbrief,
    author={{MECG 7780 Engineering Data Analytic}},
    title={{Course Project: Time Serial Dependency Modeling via Advanced Machine Learning Algorithms}},
    howpublished={Course handout, pp. 1--2, and supplied MATLAB fault-feature reference}
}"""
    bibs = ["lit_mechanics.bib","lit_reservoir.bib","lit_comparators.bib"]
    joined = "\n\n".join([course,*[named(EV/f) for f in bibs]])
    records = re.split(r"(?m)(?=^@\w+\s*\{)",joined)
    by_key = {}
    for rec in records:
        key = re.match(r"@\w+\s*\{([^,]+),",rec)
        if key:
            by_key.setdefault(key[1].strip(),rec.strip())
    aliases = {"compHoerl1970":"hoerl1970","fyfe1997":"fyfe1997order","chen2025":"chen2025sparse"}
    for old,new in aliases.items():
        source = re.sub(rf"@{old}(?![\w-])","@"+new,source)
    cites = set(re.findall(r"(?<!\w)@([A-Za-z][\w:-]*)",source))
    assert cites <= set(by_key),sorted(cites-set(by_key))
    use = {key:by_key[key] for key in sorted(cites)}
    dois = {}
    for key,record in use.items():
        m = re.search(r'(?i)\bdoi\s*=\s*[{"]([^}"]+)',record)
        if m:
            dois.setdefault(m[1].casefold(),[]).append(key)
    assert all(len(keys)==1 for keys in dois.values()),[k for k in dois.values() if len(k)>1]
    (EV/"references.bib").write_text("\n\n".join(use.values())+"\n",encoding="utf-8")
    (EV/"bibliography_index.json").write_text(json.dumps(dict(count=len(use),doi_count=len(dois),ids=sorted(use)),
                                                           indent=2)+"\n",encoding="utf-8")
    return source


def build():
    intro = named(GEN/"front_and_data.md")
    mechanics = named(EV/"lit_mechanics.md").split("<!-- BEGIN MECHANICS BACKGROUND -->",1)[1].split("<!-- END MECHANICS BACKGROUND -->",1)[0].strip()
    mechanics = re.sub(r" The training speed range of .*?(?= \[@fyfe1997order;)", "", mechanics, flags=re.S)
    mechanics = mechanics.replace("$s(t)$","$n(t)$").replace("s(u)/60","n(v)/60").replace("$q(t)=\\int_{0}^{t}","$u(t)=\\int_{0}^{t}").replace("n(v)/60\\,du","n(v)/60\\,dv").replace("m q(t)","m u(t)").replace("m s(t)","m n(t)").replace("s/60","n/60").replace("\\Delta q","\\Delta u")
    background = "# 2. Scientific background and related literature\n\n" + mechanics + "\n\n" + named(EV/"lit_reservoir.md")
    heading_no = iter(range(1, 8))
    background = re.sub(r"(?m)^#{2,3} (.+)$", lambda m: f"## 2.{next(heading_no)} {m[1]}", background)
    fault, conclusion = named(GEN/"fault_and_conclusion.md").split("<!-- CONCLUSION START -->\n\n")
    chapters = prediction_tables(named(EV/"results_insert.md")+"\n")
    leads = {
        "8_1":"Figure 8.1 puts the matched required-family scores and the archival anchor in the same numerical frame.",
        "8_2":"Figure 8.2 illustrates local amplitude tracking using one fixed interval.",
        "8_3":"Figure 8.3 shows the residual dependence left in both the time and frequency domains.",
        "8_4":"Figure 8.4 shows how the common-mask comparison changes with speed.",
        "8_5":"Figure 8.5 compares measured batch cost with development error.",
        "10_1":"Figure 10.1 distinguishes removal penalties from lower-error retrospective variants.",
        "10_2":"Figure 10.2 shows the effect of measured-speed perturbations under the frozen predictors.",
    }
    for key, lead in leads.items():
        chapters = chapters.replace("FIGURE_"+key,lead+"\n\nFIGURE_"+key)
    results = chapters.split("# 10. ",1)
    full = "\n\n".join([intro,background,named(GEN/"data_and_eda.md"),named(EV/"src_methods_insert.md"),
        named(EV/"lit_comparators.md").replace("Ljung–Box","Ljung‑Box"),results[0].strip(),fault_tables(fault.rstrip()),"# 10. "+results[1].strip(),
        conclusion,"# References\n\n::: {#refs}\n:::",appendices()])
    for key,(file,caption) in FIGURES.items():
        hit = "FIGURE_" + key
        assert full.count(hit)==1,hit
        full = full.replace(hit,f"![Figure {key.replace('_','.')}. {caption}](assets/figures/{file}.png){{width=6.35in}}")
    full = re.sub(r"(?m)^## (\d+\.\d+)\. ",r"## \1 ",full)
    full = re.sub(r"(?m)^### (?=[A-Z])","## ",full)
    assert not re.search(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]",full)
    full = bibliography(full)
    (ROOT/"report_source.md").write_text(full.rstrip()+"\n",encoding="utf-8")
    print(f"Manuscript contains {len(full.split()):,} whitespace-delimited words and {len(FIGURES)} figures.")


if __name__ == "__main__":
    build()
