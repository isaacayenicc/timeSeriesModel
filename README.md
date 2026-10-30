# MECG 7780 — Causal rotating-machinery time-series prediction

For the GitHub source checkout, see [publication contents and reproduction](docs/GITHUB_PUBLICATION.md).
Raw signals, prediction/trace arrays, logs and caches remain local. Run standard
reproduction with your original data before executing the walkthrough from a fresh checkout.

This is an executed Python research project built from the supplied training and validation recordings, the updated course brief and `fdt_featcal.m`. Start with [the final report](reports/PROJECT_FINAL_REPORT.md), [the educational walkthrough](notebooks/00_COMPLETE_PROJECT_WALKTHROUGH.ipynb) or [the comprehensive companion](docs/COMPREHENSIVE_STUDY_COMPANION.md).

**Status: prediction suite and full 200-record descriptive fault study executed.** Twenty uploaded MAT parts are verified against file/field digests and original record ranges; the 200 recordings remain separate. All13 validation-selected causal model-family representatives are frozen before candidate inference. Health labels and physical speed units remain unverified; no physical fault accuracy or independent-test performance is invented. Full lecturer-paper methods remain inaccessible, so the LPV-AR surrogate is not an exact Spa LPV-ARMA reproduction.

## Results and evidence

See [results_summary.csv](reports/results_summary.csv) for all 31 model/context/seed runs on exactly **23,744 validation targets, zero-based indices 256–23999**. Source x is dimensionless; the metric is one-step teacher-forced MSE. The best measured model in this bounded search is a ridge-regularized polynomial speed-dependent AR surrogate, not the cited sparse LPV-ARMA. Selection and early stopping reuse the validation recording, so these are provisional validation results, not independent-test estimates.

The primary model suite contains zero, empirical training mean, persistence, OLS AR, ridge AR, speed ARX, polynomial LPV-AR surrogate, MLP, GRU, causal dilated TCN and speed-FiLM TCN. Controls include late speed concatenation, a larger unconditioned TCN and an explicitly **noncausal offline rolled-speed control** excluded from best-model recommendation. Context lengths and repeated TCN seeds are in [the recorded plan](config/experiment_plan.json). CPU only; no GPU time is invented.

## Quick start

Python 3.12 was tested on Linux. Python 3.11+ is declared; Windows/macOS instructions are supplied but were not executed here. Use your own original data files; they are excluded from the release archive because redistribution rights are unknown.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install -r requirements.txt
python -m pip install -e .
# Place sigtrain.mat and sigvalid.mat in data/raw first.
python reproduce_minimal.py --data-dir data/raw
python -m pytest -q
```

`requirements.txt` pins the project packages observed here; `requirements-lock.txt` records the complete observed environment, including unrelated preinstalled packages. CPU PyTorch was installed from its official CPU index; do not disable TLS/signature verification if installation fails. Only load this release's trusted locally generated pickle artifacts; arbitrary pickle files can execute code.

## Reproduction levels

```bash
# Rescore packaged checkpoints, regenerate tables, figures, report and PPTX:
python reproduce_all.py --level standard --data-dir data/raw --device cpu
# Retrain the entire recorded bounded suite before regeneration:
python reproduce_all.py --level full --data-dir data/raw --device cpu
# Build and execute notebook (may require local kernel socket permissions):
python scripts/build_notebooks.py --execute
# Verify integrity and create archive excluding raw data:
python scripts/verify_release.py
python scripts/package_release.py
```

Standard reproduction does not claim retraining; minimal reproduction audits data and independently rescores the saved ridge256 checkpoint. The comprehensive notebook additionally refits ridge and checks future-data perturbation invariance. See [reproducibility evidence](reports/reproducibility_check.md) for exactly what ran and [timing methodology](reports/timing_and_hardware.md) for measured costs.

## Individual commands

All commands run from the repository root; `run_project.py` is a convenience wrapper for the same CLI without an editable installation.

```bash
python -m rotmach.cli audit --config config/default.yaml
python -m rotmach.cli train --model ar
python -m rotmach.cli train --model gru
python -m rotmach.cli train --model speed_tcn
python -m rotmach.cli evaluate
python -m rotmach.cli eda
python -m rotmach.cli summarize
python -m rotmach.cli report
python -m rotmach.cli slides
python -m rotmach.cli verify
# Run the complete frozen-model study from the verified twenty parts:
python -m rotmach.cli faults --input data/raw/sigfault1_parts
```

The directory-based fault command verifies all parts, evaluates thirteen causal family representatives plus raw x on all 200 complete recordings, and writes2,800 time-feature and8,400 angular-sensitivity rows. The common segment is indices256–31,999, with31,744 samples per record. The rpm/rad_s/rps branches are explicit assumptions; no unit, class or defect order is inferred. `--max-records 2` writes isolated pilot outputs; `--fresh` recomputes rather than reusing matching per-record caches. Model selection uses original validation only, never candidate features.

`standard` also regenerates fault tables/figures from matching input/model/diagnostic caches, whereas `full` retrains reference predictors and recomputes candidate inference. If you change model architecture or window source, force fresh fault inference; cache identity does not replace source review. Raw data and the original split source manifest must be placed locally as documented in [data/README.md](data/README.md).

## Load a model without training

```python
import json
from pathlib import Path
from rotmach.io_matlab import load_records
from rotmach.experiment import load_model_and_predict, metrics
root = Path('.')
run = json.loads((root/'experiments/best_model.json').read_text())['run_id']
record = load_records(root/'data/raw/sigvalid.mat')[0]
pred = load_model_and_predict(root, run, record)
print(metrics(record['x'][256:], pred[256:]))
```

Predictions before each model's lag are NaN; do not treat them as errors of zero. All reported comparisons use the common burn-in 256.

## Read and audit

- [Study companion](docs/COMPREHENSIVE_STUDY_COMPANION.md) and [implementation guide](docs/IMPLEMENTATION_GUIDE.md)
- [Updated rubric and dates](docs/course_requirements_and_open_questions.md) and [rubric traceability](docs/rubric_traceability_matrix.md)
- [Measured input audit](reports/data_audit.md) and [input manifest](data/manifest.json)
- [Research synthesis](research/research_synthesis.md), [literature matrix](research/literature_matrix.md), [source-access log](research/source_access_log.md)
- [Numerical validation](reports/numerical_validation.md), [scientific validity review](reports/scientific_validity_review.md), [label limitations](docs/fault_label_limitations.md)
- [Progress PPTX](presentations/Progress_Presentation_8min.pptx) and [final PPTX](presentations/Final_Presentation_12min.pptx)
- [Acceptance checklist](docs/FINAL_ACCEPTANCE_CHECKLIST.md), [release manifest](RELEASE_MANIFEST.md), [rights and attribution](RIGHTS_AND_ATTRIBUTION.md)

Course rubric is taken from the **uploaded updated PDF**, correcting the stale master-prompt percentages: progress 4/6/8/2%, final 16/24/30/10%. The printed 2026 progress and 2025 final dates conflict; final deadlines remain unconfirmed.


## Updated fault-study evidence

- [Verified source partitions](reports/fault_parts_verification.json) and [per-record audit](reports/fault_data_audit.csv)
- [Full study summary](reports/fault_study_summary.json), [time features](reports/fault_record_features.csv), [angular features](reports/fault_angular_features.csv), [paired summary](reports/fault_paired_summary.csv)
- [Measured fault-study narrative](reports/fault_analysis_narrative.md) and [source-qualified19-publication research matrix](research/publication_matrix.csv)

The earlier oversized-attachment limitation was resolved by individual uploads. Separate processing does not establish independent acquisitions or a known health progression. The source file's declared original byte fingerprint cannot be independently checked without its exact original bytes; all uploaded part and logical-data digests are verified.
