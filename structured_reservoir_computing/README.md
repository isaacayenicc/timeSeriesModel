# Physics-Informed Structured Reservoir Computing for gearbox vibration

This standalone MECG 7780 project models one-step gearbox vibration under measured variable speed, and uses causal prediction residuals for offline angular fault diagnostics. Data selection and scaling are explicit. The PI-SRC feature library combines vibration and meshing delays, finite damped quadratures at physical orders, speed and acceleration, energy, and proxy innovations. A group-weighted ridge trains the readout. A bounded adapter uses older base errors only.

The guarded utility selected a model with **training MSE 0.308688** and **held-out development MSE 0.329973**, in squared standardized vibration units across 22,208 targets. The 21 required comparators replayed on the same target mask. The strongest, Linear Regression, reached 0.367386. The strongest matched archived corrected SRC still has the lowest point estimate at 0.317788. Blind test data and authoritative fault-condition labels were unavailable. See the [model card](docs/model_card_src.md), [limitations](docs/limitations.md), and [experiment audit](docs/completion_audit.md).

## Package layout and setup

- `src/src_gearbox/`, `scripts/`, `configs/` and `tests/` contain standalone source and fit/replay/diagnostic entry points.
- `checkpoints/` holds numeric trained state. `results/` holds trial history, matched predictions, scope labels, reports, measured timings and data checks.
- `notebooks/` and `docs/companion_guide.md` explain the mathematics, analysis and conclusions. `report/` and `presentations/` have editable submission files and rendered review copies.
- `inputs/input_manifest.csv` maps original input names and SHA-256 identities. Raw inputs and licensed reference documents are supplied separately.

Use CPython 3.12. The study was tested on Linux x86-64, AMD EPYC 9V74, Python 3.12.14 and PyTorch 2.10.0+cpu, with single-thread comparators. The lock lists the tested direct versions. Use the version ranges in `pyproject.toml` when a platform has no exact wheel, and record the differences. Bash and PowerShell use the same Python commands after environment activation.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0
python -m pip install -e . --no-deps
```

On Windows PowerShell use ` .\.venv\Scripts\Activate.ps1` in place of the Bash activation line. Pandoc 3.1+ and LibreOffice Writer/Impress must be on PATH to rebuild and render Word and PowerPoint. For a numerical-only CPU smoke run, those programs are optional. A platform's supported CUDA wheel may be installed for experiments on a CUDA host, but the shipped timing and fit commands use CPU settings and the release has no measured GPU results.

Copy `sigtrain.mat`, `sigvalid.mat`, `fdt_featcal.m` and either `sigfault1.mat` or all 20 unique numbered parts into `data/raw`. Use `--data-dir` for a different location. A sibling `inputs/reference_archives/` can hold the course/lecture references if available. The manifest records both the original reference names and explicitly missing inputs. The assembled fault corpus includes records 61/63 that match the official two signals exactly across an initial 24,000 points and are excluded from provisional group metrics. Released numeric arrays include selected derived standardized target and prediction excerpts/results, so use the package within the course context.

## Quick verification and evaluation

```bash
python scripts/run_pipeline.py --mode smoke
python scripts/src_single_model.py --replay checkpoints/src/primary_pi_src.npz --record data/raw/sigvalid.mat
python scripts/07_compare_models.py
python scripts/15_verify_outputs.py
```

The smoke suite uses synthetic numeric fixtures and validates all 21 baseline fit/replay contracts. The single-model replay prints the MSE without modifying full comparison tables. The expected common mask is zero-based [1792, 24000), exactly 22,208 samples. Hash mismatches or incompatible sampling rates produce an error in the registry replay.

## Reproduce the shipped study

```bash
python scripts/run_pipeline.py --mode standard --data-dir data/raw --resume
python scripts/02_reproduce_lpv_reference.py --data-dir data/raw
python scripts/06_rpm_parity.py --data-dir data/raw
python scripts/06_baseline_seed_sensitivity.py --data-dir data/raw
python scripts/09_ablation_robustness.py --data-dir data/raw
python scripts/src_feature_benchmark.py --data-dir data/raw
python scripts/build_notebooks.py --data-dir data/raw
python scripts/build_companion.py
python scripts/build_report.py --pages
python scripts/build_presentations.py --render
python scripts/15_verify_outputs.py
python scripts/package_release.py
```

Standard replay reconstructs the PI-SRC and 21 baseline development vectors from frozen NPZ checkpoints, verifies their target mask and recalculates metrics and moving-block intervals. It also writes a new selected/representative fault replay under `results/fault/reproduction`. The shipped full report uses the matched fault tables traced in `results/fault/`. The same-cohort replay matrix and exact assembly are described in `docs/reproduction_protocol.md`. Notebook execution defaults to review mode, which scores saved evidence and uses small synthetic demonstrations rather than launching long searches. `--only 3 6` selects notebook numbers and `--build-only` regenerates cell source without executing.

## Research rerun and frozen blind protocol

```bash
python scripts/run_pipeline.py --mode full --data-dir data/raw --resume
python scripts/05_optimize_src.py --data-dir data/raw --profile standard --resume
python scripts/06_train_baselines.py --data-dir data/raw --tier all --resume
python scripts/audit_fresh_baseline_refit.py --data-dir data/raw
python scripts/10_frozen_final_test.py --data-dir data/raw
```

Full mode runs a larger explicitly logged SRC grid. The baseline refit command writes independently trained checkpoint groups under `results/experiments/baseline_refit`. Standard comparison intentionally references the original preregistered comparator registry. An independent full rerun's fits should be archived as a different experiment before promoting their registry into the ranking. `--only stage_name` can rerun one stage. The optimizer uses atomic fingerprint-keyed trial records and completed jobs survive interruption. It does not access `sigvalid`. Source-code and input hashes bind the preselection and fit to actual artifacts.

Place truly independent `sigtest.mat` and `sigfault2.mat` or numbered parts in the input path for the frozen command. It applies the selected coefficients and training scaling without refitting, records hashes and produces separate output. Fault-2 PMI is descriptive when no verified labels are available. An already scored blind input/model pair cannot be replaced in the same output record.

### Troubleshooting

Use quoted paths when there are spaces. If MATLAB shape or HDF5 compatibility fails, check the original struct, variable name, sample time and part ordering in the input manifest. Duplicated names with distinct bytes are ambiguous and must be separated; both merged and split representations cannot be mixed. If fit speed differs substantially, verify CPU affinity and BLAS pools in the recorded timing manifest. Keep a separate environment for alternate Torch/CUDA experiments. Missing plots/Office exports usually reflect missing LibreOffice, Pandoc or compatible fonts. Large original archives are inspected safely and separately rather than executed.
