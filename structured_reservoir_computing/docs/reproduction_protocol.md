# Reproduction protocol

Run from the SRC project directory in its installed Python environment. Place the course MATLAB inputs and instructor script in `data/raw`, or supply the same directory explicitly with `--data-dir`. The project has two distinct numerical workflows. Training-fold selection reads `sigtrain.mat` alone. Replaying the frozen checkpoints scores the matched development target range without training.

## Development prediction and fault interventions

```text
python -m src_gearbox.study score --data-dir data/raw
python scripts/06_train_baselines.py --data-dir data/raw --tier standard
python scripts/07_compare_models.py
python scripts/09_ablation_robustness.py --data-dir data/raw
```

The common development prediction range is zero-based `[1792,24000)`. The comparison script reconciles arrays and metadata from all 21 requested comparator checkpoints, the newly fitted primary PI-SRC, and archived SRC rows. Some archived rows come from protected forecasts rather than executable checkpoints. Residual correction, perturbation and timing checks have distinct output tables.

For the complete released diagnostic comparison, run these one-line commands in either Bash or PowerShell. They write to separate directories so the release evidence is available for equivalent-score checks.

```text
python scripts/evaluate_faults.py --data-dir data/raw --out results/fault_replay/comparator_live --baseline sparse_lpv_arma --baseline archived_src_corrected --baseline cnn --baseline ridge --cohort sparse_lpv_arma archived_src_corrected --pulse sparse_lpv_arma archived_src_corrected cnn ridge
python scripts/evaluate_faults.py --data-dir data/raw --out results/fault_replay/src_live --src primary_pi_src=checkpoints/src/primary_pi_src.npz --src pi_src_adapter_off=checkpoints/src/pi_src_adapter_off.npz --src pi_src_optimized=checkpoints/src/pi_src_optimized.npz --src pi_src_fixed=checkpoints/src/pi_src_fixed.npz --src src_generic=checkpoints/src/src_generic.npz --src src_polynomial=checkpoints/src/src_polynomial.npz --cohort primary_pi_src pi_src_adapter_off pi_src_fixed --pulse primary_pi_src pi_src_adapter_off pi_src_fixed src_generic src_polynomial
python scripts/merge_fault_runs.py results/fault_replay/comparator_live results/fault_replay/src_live --out results/fault_replay/merged --figures figures/fault_replay
```

Each cohort acquisition resets its state. The live cohort uses zero-based `[1792,32000)` per recording. The merged result contains 1,200 live model-record rows across six residual or raw series, with 198 eligible candidate records in each series. Record 61 has a joint exact vibration/RPM overlap with `sigtrain`, and record 63 has the corresponding overlap with `sigvalid`. Group A consists of the remaining 98 record indices within 1–100. Group B contains 100 indices within 101–200. These groups are provisional. Any AUC or bootstrap interval describes the indexed split only, with source-acquisition dependence unverified.

The source reproduces the seven synthetic pulse cases and eight RPM scenarios on the official development signal. Diagnostics-only RPM interventions leave the vibration forecast unchanged. Predictor-only and combined interventions can change residual PMI materially and are reported independently. Reporter-facing tables are `fault_group_summary.csv`, `paired_pmi_bootstrap.csv`, `rpm_stress_summary.csv`, `rpm_stress.csv` and `pulse_summary.csv`. A live full-cohort run can take several minutes on one CPU thread. Exact elapsed times depend on hardware.

The supplemental 41-record screen in `results/fault/lineage_screen` contains all 21 comparator families through live results or reconciled, hash-identical historical evaluations. It also includes live raw and SRC reference rows. Its source provenance and exact record IDs are in `screen_lineage.json`. Rebuilding the historical join requires the earlier expanded project and is optional. The complete live comparison above needs only this SRC project, its checkpoints and the course inputs.

```text
python scripts/lineage_fault_screen.py /path/to/earlier/expanded/project --live results/fault_replay/merged --out results/fault_replay/lineage_screen
```

## Frozen external inputs

When `sigtest.mat` or `sigfault2.mat` is available, place it in the input directory and run `python scripts/10_frozen_final_test.py --data-dir data/raw`. The command locks the checkpoint and input hashes, logs missing inputs and writes a separate test record. Fault PMI for unverified labels remains descriptive. The supplied development experiment contained neither independent test file. Compare frozen run status with the data inventory before using performance values in the report.
