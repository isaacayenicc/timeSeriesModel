# Course inputs

Place the original course files in `data/raw/` or supply a separate directory using `--data-dir`. The standard dataset includes `sigtrain.mat`, `sigvalid.mat`, `sigfault1_part_001.mat` through `sigfault1_part_020.mat`, and `fdt_featcal.m`. A single `sigfault1.mat` may replace the twenty parts. File discovery accepts the instructor's alternate `(1)` suffix and case variations. Conflicting copies require resolving their provenance before use.

Store any downloaded reference PDFs and archives in `inputs/reference_archives/` or pass `--source-dir` for each directory. See `input_manifest.csv` for the exact original names and expected SHA-256 values. The paths under `source_1/`, `source_2/` and further prefixes refer to paths relative to each supplied reference root, in command order. They are provenance identifiers. The archive member inventory includes member hashes and checks byte-identical MAT copies.

The frozen inference entry point may use later supplied `sigtest.mat` and `sigfault2.mat`. The preparation, training and EDA stages leave their contents unopened. Place raw data and source archives outside shared Git history. The release includes manifests and output tables; the original datasets and archive bytes are obtained from the existing course inputs.

Run from the project directory after installing the package.

```bash
python scripts/00_audit_inputs.py --data-dir ./data/raw --source-dir ./inputs/reference_archives
python scripts/01_prepare_data.py --data-dir ./data/raw
python scripts/02_run_eda.py --data-dir ./data/raw
```

Each command accepts an alternate `--output` directory for a repeat run without replacing the reference results. The file and data audit performs deterministic SHA-256 checks, loads MATLAB structures and scans overlap candidates. EDA runs only on the designated training record.
