"""Synchronize measured fault and research discussion with finalized CSV outputs."""
from pathlib import Path
import json
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pandas as pd


def update(root):
    root=Path(root)
    summary=json.loads((root/'reports/fault_study_summary.json').read_text())
    if summary.get('status')!='complete_descriptive_unlabeled':
        raise ValueError('Complete all fault records before documenting measured findings')
    scalar=pd.read_csv(root/'reports/fault_record_features.csv')
    angular=pd.read_csv(root/'reports/fault_angular_features.csv')
    audit=json.loads((root/'reports/data_audit.json').read_text())['sigfault1']
    best=summary['recommended_model_id']
    raw=scalar[scalar.model_id=='raw'].set_index('record_id')
    stats=[]
    for model,rows in scalar.groupby('model_id'):
        paired=rows.set_index('record_id').join(raw[['rms']],rsuffix='_raw')
        stats.append(dict(model_id=model,records=len(rows),median_rms=rows.rms.median(),
                          median_residual_raw_rms_ratio=(paired.rms/paired.rms_raw).median(),
                          median_crest_factor=rows.crest_factor.median(),median_kurtosis=rows['kurtosis'].median()))
    table=pd.DataFrame(stats).sort_values('median_rms')
    table.to_csv(root/'reports/fault_paired_summary.csv',index=False)
    posthoc=scalar[scalar.model_id.isin(['raw',best])].copy()
    posthoc['post_hoc_source_partition']=posthoc.record_id.gt(100).map({False:'source IDs 1–100',True:'source IDs 101–200'})
    posthoc=posthoc.groupby(['model_id','post_hoc_source_partition'])[
        ['rms','kurtosis','mean_speed_source_units','speed_std']].median().reset_index()
    posthoc.to_csv(root/'reports/fault_posthoc_source_partition.csv',index=False)
    units=angular[angular.model_id.isin(['raw',best])].groupby(['model_id','speed_unit_assumption'])[
        ['ck_order1','pmi_faithful_reference','pmi_scientifically_corrected','roundtrip_rmse_fraction_of_source_rms']].median().reset_index()
    measured=f'''## Measured full fault-candidate study

The completed study contains **{summary['n_records']} separate recordings**, **{summary['n_models']} frozen validation-selected causal predictors**, **{len(scalar):,} raw/residual time-feature rows**, and **{len(angular):,} angular-sensitivity rows**. Every scalar comparison uses indices 256–31,999, exactly 31,744 values per recording. Separate processing does not itself prove statistical independence of acquisitions; acquisition-order semantics are unknown.

{table.to_markdown(index=False,floatfmt='.6f')}

These are medians of record-level descriptors, not classification scores. The paired ratio divides each model residual RMS by that same record's raw RMS before taking the median. Zero prediction has the same descriptors as raw, an implementation sanity check. Candidate-record prediction errors were not used to reselect the primary model; the recommendation remains based on the original validation protocol.

### Post hoc source-order observation, without physical labels

The all-record plot shows an amplitude change near source ID101. The following partition was examined **after viewing the trend** and has no known class meaning; it is not a fault-label assignment, calibrated threshold or independent detection experiment.

{posthoc.to_markdown(index=False,floatfmt='.6f')}

Raw and residual RMS differ between these provenance groups while the coarse speed moments are similar. Similar moments do not establish identical operating histories or exclude load, acquisition, preprocessing or health changes. Source documentation is required to interpret the shift. Neither accuracy nor an onset claim can be inferred from this descriptive partition; it was not used to train or select models.

### Physical-unit and PMI-indexing sensitivity on all records

{units.to_markdown(index=False,floatfmt='.6g')}

Values above are medians across all 200 records under three **unverified** speed-unit interpretations and assumed modulation order one. They do not establish a known shaft or defect order. A different CK/PMI ranking from the MSE ranking shows that the objectives differ; it does not identify a physically superior detector. Interpolation error remains appreciable and is itself part of the sensitivity analysis.

The inspected normalized speed range is **{audit['normalized_speed_min']:.10f} to {audit['normalized_speed_max']:.10f}**. **{100*audit['aggregate_speed_outside_reference_fraction']:.6f}%** of all speed samples lie outside the supplied reference interval; no clipping or per-record normalization was introduced. There are no health-state, severity or onset fields in the inspected eight-field structs.

![All-record raw/residual trends](../reports/figures/fault_rms_trends.png)

*The horizontal coordinate preserves source order, not a validated progression of damage. Identical burn-in is applied to every raw/residual trace.*

![Record descriptor distributions](../reports/figures/fault_indicator_distributions.png)

*These distributions describe 200 recordings. Suppressed fliers in a box plot are not removed from the CSV or summary statistics.*

![Model objectives compared](../reports/figures/fault_prediction_diagnostic_tradeoff.png)

*Candidate-record prediction MSE and assumption-based PMI are distinct descriptors; larger PMI is not verified physical detection success.*

![All-record unit sensitivity](../reports/figures/fault_unit_sensitivity.png)

*The chosen unit changes the angular grid, period and resulting PMI. No unit is selected merely because its plot is more separated.*

![Fixed record waveform](../reports/figures/fault_neutral_record_overlay.png)

*Provenance record 1 is fixed by the neutral selection policy, not selected for dramatic appearance.*

The full numerical evidence is [fault_record_features.csv](../reports/fault_record_features.csv), [fault_angular_features.csv](../reports/fault_angular_features.csv), [paired summary](../reports/fault_paired_summary.csv), [speed/index associations](../reports/fault_feature_correlations.csv), and [study plan](../reports/fault_study_plan.json). Record IDs 1, 50, 100, 150 and 200 have retained traces; all 200 have restartable scalar-feature caches. No fault data were used for model fitting or hyperparameter tuning.
'''
    for target in (root/'docs/COMPREHENSIVE_STUDY_COMPANION.md',):
        text=target.read_text()
        start='<!-- MEASURED_FAULT_RESULTS_START -->';end='<!-- MEASURED_FAULT_RESULTS_END -->'
        block=start+'\n'+measured+'\n'+end+'\n\n'
        if start in text:text=re.sub(re.escape(start)+r'.*?'+re.escape(end)+r'\n*',lambda _:block,text,flags=re.S)
        else:
            marker='# Chapter 24:'
            pos=text.index(marker);text=text[:pos]+block+text[pos:]
        target.write_text(text)
    (root/'reports/fault_analysis_narrative.md').write_text(measured.replace('../reports/figures/','figures/').replace('../reports/',''))
    research=root/'research/research_synthesis.md'
    text=research.read_text()
    old=text[text.index('Evidence is classified'):text.index('\n\nThe runtime')]
    new='''Evidence is classified before drawing conclusions. The expanded [publication matrix](publication_matrix.csv) contains 19 publication records: 12 machinery-diagnostics sources (R08–R19), three architecture/conditioning sources (R02–R04), three software citations (R05–R07), and the course-cited but inaccessible anchor (R01). Nine official implementation sources and an unverified candidate are separate. The 15 method-relevant author/abstract records are not 15 full-text readings: R19 is limited citation/code context, most machinery material is an original author README abstract/method summary, and **no article full text was read**. Source metadata, year discrepancies, access dates and hashed bytes are retained. The expanded review improves direct domain coverage while leaving exact lecturer-method, historical order-tracking and MCKD/PMI full-method gaps explicit.'''
    text=text.replace(old,new)
    text=text.replace('and fault-candidate records/validated physical labels.','and validated physical fault labels. The subsequently supplied twenty-part fault dataset is now available and analyzed; the file-transfer obstacle is resolved.')
    heading='## 10. Expanded domain evidence and what it changes'
    if heading not in text:
        text+='''\n\n'''+heading+'''

### Continuously varying speed and transfer

R08's original dataset/author material distinguishes continuous acceleration and deceleration from separate fixed-speed domains. That distinction matters here: train and validation almost repeat a speed trajectory, while the 200 candidate records add descriptive record variation without verified health labels. R09's transfer-learning library and survey describe source/target adaptation protocols. They motivate future transfer studies, but adapting to the candidate set would change the present reference-only diagnostic question. No such transductive refit is hidden in this study.

### Sparse impulses, periodic structure and background suppression

R10 describes multiscale dictionary learning for impulsive structure amid harmonic/noise interference. R15 describes within-/across-period sparse group priors; R16's B-spline sparsity and R17's SALMA sparsity-assisted reconstruction describe other structured decompositions. R18 adds nonconvex low-rank plus weighted sparse decomposition. These sources share a concern with separating relevant structure from interference, but their objectives are not the same as minimizing next-sample MSE. The distinction helps explain why a predictor that removes more energy can also remove periodic content, and why neither larger residual kurtosis nor PMI alone proves better diagnosis. Several sources share an author group, which limits diversity; author-reported findings were not independently replicated. R19's intelligent time-frequency sparse method is only citation/code context and receives no unverified detailed algorithm claim.

### Interpretability, limited labels and timing

R12 embeds interpretable time-frequency kernels, an example of a physical inductive bias. Such architectural explanation does not establish causal indexing or identify units in an unrelated dataset. R13's few-shot setting has a few verified labeled examples, whereas the present candidate set has no verified health labels; its supervised scores cannot be imported. R11 explicitly treats acquisition duration and online processing as separate delay components. We consequently report lag duration, public inference time and feature extraction separately; fast offline batches do not establish streaming alarm delay. R14's benchmark makes split/augmentation and training protocol explicit, supporting our disclosure of validation reuse, limited training budget, capacity controls and sample-index fairness.

### Relation to the observed results

The original validation experiment favors the polynomial LPV-AR surrogate; FiLM does not improve mean lag-64 TCN MSE in three seeds. The full candidate study compares the same frozen family representatives on equal recording segments. These measurements adjudicate the local design hypotheses; the cited repository summaries do not guarantee them. Full article access and a directly comparable Spa LPV-ARMA reproduction remain necessary before any published-method superiority claim. Unknown units and labels still prevent physical calibration and classification-performance claims even though the input-transfer problem is now resolved.
'''
    research.write_text(text)
    (root/'docs/research_synthesis.md').write_text(text.replace('](literature_','](../research/literature_').replace('](publication_','](../research/publication_').replace('](references.bib)','](../research/references.bib)').replace('](source_access_log.md)','](../research/source_access_log.md)').replace('](evidence/','](../research/evidence/').replace('](../docs/METHOD_SELECTION_RATIONALE.md)','](METHOD_SELECTION_RATIONALE.md)'))


if __name__=='__main__':update(Path(__file__).resolve().parents[1])
