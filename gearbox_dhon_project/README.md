# MECG 7780 DHON gearbox vibration project

This project fits and validates causal one-step gearbox vibration predictors, including a differentiable speed-conditioned harmonic oscillator, sparse LPV-ARMA, NG-RC, MLP and TCN controls. The submitted `development` run uses the actual course MAT data and the same 22,208 target indices for all matched development methods. The publication and legacy current-speed reference are identified separately. Angular PMI summarizes provisional A/B index groups.

## GitHub source edition

This folder adds the DHON study alongside the repository's original time-series project. The public source edition includes aggregate results and presentation files. Exact checkpoint replay, archived forecast controls, and the previously executed notebook outputs require the privately delivered project archive and original inputs. Source notebooks retain their code and Markdown with kernel outputs cleared. Paths to restricted run artifacts in the report and checklist refer to that private archive. A new fit can be run using the supplied Python source and original course inputs.

## Open these first

- [Research report](outputs/final/final_course_report.pdf) and [editable Markdown source](docs/final_course_report.md)
- [Numerical synthesis](docs/reproduction_summary.md), [JSON summary](outputs/final/RESULTS_SUMMARY.json) and [verification checklist](outputs/final/completion_checklist.md)
- [Study guide](docs/companion_guide.md) and [teaching notebooks](notebooks/)
- [8-minute progress slides](presentations/progress_presentation.pptx) and [12-minute final slides](presentations/final_presentation.pptx)
- [Artifact inventory](outputs/final/artifact_index.md) and SHA-256 manifest `CHECKSUMS.sha256`

## Setup

The tested environment was Linux x86-64, Python 3.12.14, and PyTorch 2.10.0 CPU on AMD EPYC. For CPU-only PyTorch, installing its official CPU wheel first avoids downloading CUDA libraries through the general PyPI build. The standard `torch==2.10.0` constraint accepts the tested local build `2.10.0+cpu`. Change the PyTorch installation method only when targeting a machine with a supported CUDA stack. Use Python 3.12 for the pinned reproduction.

Linux or macOS shell:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip install -e .
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

Windows PowerShell with the Python launcher:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip install -e .
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
```

The installed Python workflow is identical on both platforms. macOS uses the PyPI torch wheel if the official CPU index has no compatible wheel. PDF regeneration additionally needs Pandoc and XeLaTeX. Optional slide rendering uses LibreOffice and Poppler. MAT v7.3 support uses h5py. Raw files are external to this ZIP. Place the original MAT files under `data/raw` or supply a path with `--data-dir`. Keep original MAT filenames and numbered parts, and see the run's source inventory for the expected hashes. For a fresh full provenance audit, pass the directory containing supplied source PDFs, reference archives and their previously verified extraction tree with `--reference-dir` to `audit_data.py`.

## Verify and replay

```bash
python -m pytest -q
python -m gearbox_dhon.cli replay --run-id development --data-dir data/raw --device cpu
python scripts/execute_notebooks.py --run-id development --mode replay --data-dir data/raw
```

Checkpoint replay reproduces selected original predictions against the actual validation MAT. Notebook 05 independently recomputes the reported MSEs from the shipped, aligned numerical outputs. Notebook 06 rederives a PMI feature from an actual stored representative residual. Run Markdown and slides were generated from the recorded development files. The notebooks can also show saved evidence without the MAT files and report the reduced replay scope.

## Fit a new run

```bash
python scripts/run_all.py --data-dir data/raw --profile smoke --device cpu --run-id smoke_local
python scripts/run_all.py --data-dir data/raw --profile full --device cpu --run-id full_local
python scripts/run_all.py --data-dir data/raw --profile full --device cpu --run-id full_local --resume
```

The smoke profile limits fitting budgets and skips full diagnostic/robustness computation. All numerical smoke outputs are a separate run. The full profile performs the declared finite search, full 200-record diagnostics when available, and a frozen manifest. Run IDs name separate directories under `outputs/runs/`. The resume check verifies raw/config/code/library-archive fingerprints, device, profile and thread count before skipping completed stages. DHON additionally checkpoints completed fold evaluations and validates its own source and data digest. Other interrupted fit stages rerun the affected stage. The submitted development evidence came from individually executed stages and is replayable without running a new fit.

## Freeze and evaluate future blind inputs

```bash
python scripts/freeze_models.py --run-id development --data-dir data/raw
python scripts/evaluate_blind.py --data-dir data/raw --frozen-manifest outputs/runs/development/frozen_manifest.json --blind-output outputs/runs/blind_local
```

For the shipped freeze, verification also requires unchanged development inputs and shipped evidence. A fresh blind output path is required. The evaluator imports predictors for inference and checks for exact signal/RPM overlap with the development corpus. Scan coverage and unconfirmed acquisition identity are disclosed separately. No label thresholds are estimated during blind evaluation. With absent blind inputs the command writes status evidence and no new numeric scores. The designated independent datasets and authoritative labels were unavailable for this submission.

## Other entry points

```bash
python scripts/audit_data.py --data-dir data/raw --run-id audit_local
python scripts/benchmark.py --data-dir data/raw --run-id development --device cpu
python scripts/build_deliverables.py --run-id development
python scripts/build_slides.py --run-id development
```

Use a copied run directory with a new identifier when regenerating outputs against a frozen audit record. The benchmark writes hardware, per-repeat cost and timing scope. The archive's legacy strongest prediction vectors support metric recomputation only. Runnable expanded DHON/NG-RC checkpoint controls use the embedded numerical and configuration artifacts. This project does not require the original archived Python modules for inference.
