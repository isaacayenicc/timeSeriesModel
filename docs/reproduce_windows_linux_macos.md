# Reproduce on Windows, Linux and macOS

The project uses Python and CPU PyTorch. MATLAB is not required. The current execution was on Linux; the Windows PowerShell and macOS shell examples below are documented instructions, **not execution claims for those systems**. Consult `reports/reproducibility_check.md` for commands actually run and measured outcomes.

## 1. Unpack and supply data

Extract the release into a writable directory. Use its existing project checkout; cloud tasks already run in isolated environments and do not require a new Git worktree. Preserve the release's paths and enter its root, containing `pyproject.toml`.

Copy original `sigtrain.mat` and `sigvalid.mat` into `data/raw/`. These are user-supplied inputs; raw files may be excluded from a distributed source release because redistribution rights are not established. Additional fault/test files are optional and must not be fabricated. Compare actual hashes to `data/manifest.json`; a mismatch means a different data version, not a reason to overwrite the manifest without recording the change.

The earlier large-file transfer limitation is resolved by the subsequently supplied twenty-part MATLAB package. Place `sigfault1_part_001.mat` through `sigfault1_part_020.mat` and their `manifest.json` in `data/raw/sigfault1_parts/`. Preserve each complete record, filenames, ranges and hashes; do not concatenate records into a continuous waveform. The original unsplit file is not needed to stream this package. Its listed byte fingerprint is supplied provenance, whereas verification of received parts is performed against their actual bytes and logical field hashes.

## 2. Create a virtual environment

Python 3.11 or later is the project target; use a Python version supported by the pinned dependencies. The validated execution used the version recorded in the release's environment metadata.

Windows PowerShell:

```powershell
cd C:\path\to\timeSeriesModel
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

Invoking the virtual environment's Python directly avoids changing PowerShell's script-execution policy. If you prefer activation and your system permits it, use `.\.venv\Scripts\Activate.ps1`.

Linux or macOS:

```bash
cd /path/to/timeSeriesModel
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The full frozen environment is recorded in `requirements-lock.txt`; that file describes the validated environment and can include platform-specific versions. Start with the declared project requirements on a different operating system. Do not disable TLS or artifact verification to get an installation to pass. If a pinned wheel is unavailable for your Python/platform, use a supported interpreter and record any required compatibility changes.

## 3. Audit and quick CPU replay

Windows:

```powershell
.\.venv\Scripts\python.exe run_project.py audit --config config/default.yaml
.\.venv\Scripts\python.exe reproduce_minimal.py --data-dir data/raw
.\.venv\Scripts\python.exe -m pytest -q
```

Linux/macOS after activation:

```bash
python run_project.py audit --config config/default.yaml
python reproduce_minimal.py --data-dir data/raw
python -m pytest -q
```

The audit must fail loudly for invalid lengths, nonfinite arrays and broken timestamps. Unknown speed units and missing optional files are limitations, not permission to invent them. A successful quick replay validates the packaged ridge checkpoint and accessible data; it does not retrain the whole neural suite.

## 4. Replay the complete packaged study

Windows:

```powershell
.\.venv\Scripts\python.exe reproduce_all.py --level standard --data-dir data/raw --device cpu
.\.venv\Scripts\python.exe -m rotmach.cli verify
```

Linux/macOS:

```bash
python reproduce_all.py --level standard --data-dir data/raw --device cpu
python -m rotmach.cli verify
```

`standard` loads checkpoints, rescored validation and rebuilds the derived artifacts. Check `reports/results_summary.csv`, per-run predictions, figures and presentation output. It is the recommended first reproduction level. Back up output folders if you want to compare a new environment's outputs byte for byte with the distributed originals.

## 5. Retrain or run one model

```bash
python reproduce_all.py --level full --data-dir data/raw --device cpu
python -m rotmach.cli train --model gru --config config/default.yaml
python -m rotmach.cli train --model tcn --config config/default.yaml
python -m rotmach.cli train --model speed_tcn --config config/default.yaml
python -m rotmach.cli evaluate
```

On Windows, replace `python` with `.\.venv\Scripts\python.exe` unless your environment is activated. `full` uses the bounded experiment plan and can take substantially longer than checkpoint replay; measured current-machine timing is in the experiment registry. The suite's saved metadata, not an estimate for a different laptop, is the source of runtime claims.

CPU is the supported and validated path in this release. A documented `--device cpu` option is not a promise that `cuda` is implemented or benchmarked. Do not compare estimated GPU time to measured CPU time. Thread count and numerical precision are part of recorded execution settings.

## 6. Execute and edit the notebook

Install a notebook interface if it is not already available, select the virtual-environment kernel, and open `notebooks/00_COMPLETE_PROJECT_WALKTHROUGH.ipynb`. Its default walkthrough uses saved artifacts. Execute from a fresh kernel with “Run All”; avoid depending on manually prepared variables.

An automated execution can use the same `nbclient` path recorded by the release. Keep its working directory consistent with the notebook's root-discovery cell. The notebook should compare key results with official CSVs, not silently replace measured outputs with copied numbers.

## 7. Add real fault inputs

```bash
python -m rotmach.cli faults --input data/raw/sigfault1_parts
```

The CLI retains record boundaries and writes `reports/fault_record_features.csv`. It uses the selected prediction model and raw features. Unknown speed units cause angular features to be omitted from the default command. For explicit assumption-sensitive angular analysis, use `angle_resample` and the tested diagnostic functions, recording whether speed is assumed rpm, revolutions/s or radians/s. Never infer health labels from record order.

`sigtest.mat` and `sigfault2.mat` require inspection before claiming new results. Their placement alone does not establish that an evaluation protocol, label provenance or untouched-test policy was followed.

## 8. Rebuild individual communication artifacts

```bash
python -m rotmach.cli eda
python -m rotmach.cli summarize
python -m rotmach.cli report
python -m rotmach.cli slides
python -m rotmach.cli verify
```

The editable report source is Markdown. Both `.pptx` files contain editable text; measured plots may be embedded images. Speaker notes and claim maps are separate files. PDF exports, if present, do not replace source documents.

## Troubleshooting

**`ModuleNotFoundError: rotmach`.** Run `python -m pip install -e .` using the same interpreter as the command or notebook kernel.

**No Python command.** On Windows use `py -3`; on Unix use `python3`. After activation, verify the interpreter path before installing dependencies.

**MAT format error.** MATLAB v7.3 uses HDF5. Use the loader's explicit supported-format behavior; renaming a file does not convert it. Do not silently flatten unfamiliar struct arrays.

**Different raw-input hash.** Keep the new file and record a separate provenance version. Do not overwrite a known original or claim the old metrics apply automatically.

**Missing checkpoint.** Use the complete release with model artifacts, or run full training. A source-only directory cannot replay absent weights.

**CPU training is slow.** Run minimal or standard replay first, inspect the bounded plan and thread settings, and reduce only explicitly new exploratory experiments. Do not delete required checks from the historical study and still claim equivalent reproduction.

**Notebook kernel disagrees with terminal.** Install/select the virtual environment's kernel; compare `sys.executable` and package versions.

**PMI is NaN or infinite.** Inspect zero-energy envelope, insufficient lag length and denominator status. Such values can be mathematically appropriate; do not replace them with zero solely to make a chart attractive.

**PowerPoint export unavailable.** The actual editable `.pptx` artifacts remain the deliverable. Rendering and layout inspection status must be reported honestly.


## Verified complete-record partitions

Place manifest.json and all 20 `sigfault1_part_###.mat` files together in data/raw/sigfault1_parts. The complete frozen-model command is `python -m rotmach.cli faults --input data/raw/sigfault1_parts`. `--max-records 2` is an isolated pilot, `--fresh` forces full recomputation. Standard reproduction validates and reuses matching record caches, and full reproduction retrains reference predictors before candidate inference. None of these commands infers physical labels. Linux CPU execution is tested; Windows/macOS examples remain unexecuted.
