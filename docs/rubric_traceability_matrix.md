# Updated-rubric traceability matrix

The [updated course PDF requirements](course_requirements_and_open_questions.md) control these weights. Slide numbers refer to the generated main decks. Evidence distinguishes measured prediction and real candidate-record descriptors from unverified physical fault-detection accuracy.

| Assessment criterion | Printed weight | Research / tested implementation | Figure or table | Report section | Slide evidence |
|---|---:|---|---|---|---|
| Progress organization / presentation | 4% | Source inventory, declared RQs and pipeline | `data/manifest.json`; `reports/data_audit.json` | Scope; data/provenance | Progress 1–3, 10 |
| Progress clarity | 6% | Explicit units, past-only indexing, one-step example | `reports/figures/eda.png` | Causal formulation | Progress 2–5 |
| Progress algorithms understood | 8% | AR/ridge equations; GRU gates; TCN receptive field; MATLAB indexing tests | `reports/baseline_comparison.csv`; `reports/numerical_validation.md` | Models; diagnostics | Progress 6–9 |
| Progress novelty | 2% | Established FiLM adapted to past-speed TCN; falsifiable speed controls | `reports/model_ablation.csv` | Project-level novelty | Progress 8, 10 |
| Final organization / presentation | 16% | Reproducible CLI, source/model/metric manifests and scientific claim map | `RELEASE_MANIFEST.md`; `models/checkpoints_manifest.json` | Reproduction; conclusions | Final 1–3, 14–15 |
| Final clarity | 24% | Consistent dimensions; causal diagram; interpretation and limitations | `reports/figures/prediction.png`; `presentations/final_speaker_notes.md` | Formulation; discussion | Final 3–6, 10, 13–14 |
| Final algorithm performance | 30% | Common-index MSE; checkpoints reloaded; actual CPU timing; seed/context comparisons; independent real candidate-record raw/residual descriptors | `reports/results_summary.csv`; `reports/fault_record_features.csv`; `reports/fault_angular_features.csv`; `reports/figures/fault_rms_trends.png` | Prediction results; efficiency; robustness; candidate-record study | Final 7–12 |
| Final algorithm novelty | 10% | Equal-core speed-TCN comparison, late conditioning, larger unconditioned control, speed negative control | `reports/model_ablation.csv` | Speed-conditioning contribution and ablations | Final 6, 9, 14 |

## Scope of evidence and unmet criterion components

Prediction MSE and CPU time are measured. GPU time is unrun on this CPU-only installation. The split candidate-record dataset resolves the original transfer obstacle and enables actual independent raw/residual feature analysis, with explicit speed-unit assumptions and MATLAB-indexing sensitivity. The finalized [study summary](../reports/fault_study_summary.json) and feature CSVs provide measured completeness evidence. **Physical fault sensitivity/accuracy/AUC/F1 remains unidentifiable without verified labels.** The report and final deck distinguish those missing labels from available descriptor measurements; this matrix does not assert that supervised diagnostic performance has been established. The course's expected Spa LPV-ARMA comparison remains unreplicated.

For the actual numerical results, use the [final report](../reports/PROJECT_FINAL_REPORT.md), [research synthesis](../research/research_synthesis.md), and [numerical validation](../reports/numerical_validation.md). Progress and final deck text remains editable; chart images are reproducibly generated from stored outputs.
