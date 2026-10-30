# Implementation guide

This repository separates data ingestion, causal prediction, diagnostic mathematics, communication and release verification. The educational derivations are in [the study companion](COMPREHENSIVE_STUDY_COMPANION.md). The CLI is the stable entry point; source modules make the same operations importable by notebooks and tests.

## Inputs and state

Run commands from the repository root. Place the original `sigtrain.mat` and `sigvalid.mat` in `data/raw/`. `--data-dir` can stage original files from another directory and refuses to overwrite differing existing raw files. The later verified fault package is preserved in `data/raw/sigfault1_parts/`; it contains all 200 records and replaces the earlier transfer blocker. The optional `sigtest.mat` and `sigfault2.mat` remain unavailable unless supplied locally. Keep raw files unchanged.

`config/default.yaml` states supplied normalization, CPU execution, past-only speed and evaluation burn-in. `config/experiment_plan.json` defines the actual model suite. A configuration is useful only if the executed resolved settings are saved beside results; inspect each run's metadata rather than assuming a subsequently edited global config describes earlier checkpoints.

## Executable workflow

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python run_project.py audit --config config/default.yaml
python reproduce_minimal.py --data-dir data/raw
python reproduce_all.py --level standard --data-dir data/raw --device cpu
```

The minimal path audits and rescores the packaged ridge-256 model in the validated environment. `standard` loads trained checkpoints, rescores and rebuilds figures/report/slides. `full` retrains the configured suite; do not confuse checkpoint replay with independent retraining.

For individual stages:

```bash
python -m rotmach.cli train --model ar --config config/default.yaml
python -m rotmach.cli train --model gru --config config/default.yaml
python -m rotmach.cli train --model tcn --config config/default.yaml
python -m rotmach.cli train --model speed_tcn --config config/default.yaml
python -m rotmach.cli evaluate
python -m rotmach.cli eda
python -m rotmach.cli summarize
python -m rotmach.cli report
python -m rotmach.cli slides
python -m pytest -q
python -m rotmach.cli verify
```

`--model` selects the kind used in the experiment plan, rather than an arbitrary architecture string. Consult `python -m rotmach.cli --help` and the plan for available kinds. `--root PATH` supports an explicit repository root. The actual parser accepts `--config`, `--data-dir`, `--model`, `--input` and `--root`; flags copied from earlier planning documents may not exist.

## Module responsibilities

- `io_matlab.py` loads structures, validates record shapes and preserves records.
- `audit.py` checks timestamps, finite arrays, normalization and input provenance.
- `windows.py` constructs past-only predictors with aligned target indices.
- `models.py` defines compact neural architectures and speed conditioning.
- `experiment.py` fits, persists and reloads model runs with prediction provenance.
- `evaluate.py` rescores saved artifacts on the common validation interval.
- `diagnostics.py`, `angle_resample.py` and `spectral.py` implement independently tested signal features and explicit-unit angle conversion.
- `plots.py` derives figures from actual inputs and saved predictions.
- Report/slide builders generate the narrative artifacts; `release.py` checks deliverables and hashes.

The basic inference interface is:

```python
from pathlib import Path
import json
from rotmach.io_matlab import load_records
from rotmach.experiment import load_model_and_predict

root = Path.cwd()
record = load_records(root / 'data/raw/sigvalid.mat')[0]
run_id = json.loads((root / 'experiments/best_model.json').read_text())['run_id']
prediction = load_model_and_predict(root, run_id, record)
```

Check the returned alignment and warm-up convention before computing a score; the official evaluation module handles the common target range. Do not accidentally score initial unavailable predictions.

## Real fault-candidate data

```bash
python -m rotmach.cli faults --input data/raw/sigfault1_parts
```

This command processes records individually and compares raw features with all thirteen frozen model-family representatives' residuals on common retained samples. It does not infer physical labels. The full directory-based study computes three explicitly assumption-labelled angular branches; none establishes the source unit. The diagnostic module supports explicit-unit angular sensitivity analysis for users who can justify the assumption. The full record-wise study now uses the verified twenty-part package, never concatenates records and does not retrain reference models on candidates. The source manifest, global record IDs and part hashes are preserved with actual feature tables. Explicit-unit angular sensitivity is a separate analysis branch; no assumed unit is promoted to a verified physical unit.

## Editing safely

Change a small documented configuration, give new experiments distinct run IDs and retain previous configurations. Run tests after changing alignment, preprocessing, architecture, resampling or feature conventions. Rebuild communication artifacts after results change; otherwise reports, decks and notebook tables can diverge. Do not edit a result CSV to improve a narrative. Fix the scientific procedure or acknowledge a negative result.

## Verification scope

The release reports exactly which tests, notebook execution and reproduction commands were executed in the current Linux environment. Windows and macOS command syntax is provided separately and was not executed here. CPU is the validated device. Live MATLAB equivalence, physical fault classification and independent test-set performance are not claimed.


Use `python -m rotmach.cli faults --input data/raw/sigfault1_parts --max-records 2` for an isolated pilot. Full outputs are not overwritten by this pilot. `--fresh` forces recomputation. The full command preserves global IDs1..200, uses common256:32000segments and computes all threeunitassumptions. `python scripts/sigfault1_splitter.py verify data/raw/sigfault1_parts` is the source verifier; the production `fault_io.verify_fault_parts` independently checks the same logical digest plus scientific vector/time validity.
