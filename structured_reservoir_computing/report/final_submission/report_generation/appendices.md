# Appendix A. Complete model comparison

Table A.1. All rows on the official development interval [1792, 24000). Warm medians are measured one-thread batch milliseconds. An em dash indicates an unavailable measurement.

TABLE_A_1

Each required family has a configuration, checkpoint and replay record. A fresh run reproduced all 21 configurations, saved checkpoint bytes, development forecasts and MSEs. Two sparse families refitted their protected supports. The executable archive rows use previously optimized weights. Saved-forecast rows support numeric alignment and error only.

# Appendix B. Reference-segment and protocol comparisons

The paper’s benchmark uses the authors’ own source segment. Table B.1 preserves their reported training and testing values. Table B.2 separates current-RPM replays on the supplied file from the strict lag-1 reference. Preprocessing, sample start and selection history matter for comparison. [@chen2025sparse]

Table B.1. External published values only. Unit and protocol attribution follow the original published experiment.

TABLE_B_1

Table B.2. Local Sparse LPV–ARMA sensitivity on the supplied development file.

TABLE_B_2

Rescoring on a common mask aligns evaluation positions. A valid causal comparison also requires the specified information cutoff.

# Appendix C. Reproducibility and data access

The report was assembled against project release cca9804101e and fitted SRC source 802e5b6238. Canonical evidence consists of the precisely aligned development bundle, original optimization journals and replay receipts. The course data are supplied separately. Report creation uses targeted arithmetic and inference checks on saved artifacts.

Place the original input files in data/raw in the installed project environment. The replay command should report 22,208 targets beginning at zero-based index 1,792. The standard invocation generates the matched predictions and comparison. The separate frozen command checks future independent inputs for known overlap and uses checkpoint and input hashes.

~~~bash
python scripts/src_single_model.py --replay checkpoints/src/primary_pi_src.npz --record data/raw/sigvalid.mat
python scripts/run_pipeline.py --mode standard --data-dir data/raw --resume
python scripts/10_frozen_final_test.py --data-dir data/raw
~~~

Table C.1. Project records for traceable reproduction.

| Evidence | Project location |
| --- | --- |
| Raw input identity and split policy | inputs/input_manifest.csv and data/splits/protocol.json |
| Code and serialized PI-SRC | src/src_gearbox/src_model.py and checkpoints/src/primary_pi_src.npz |
| Training-only selection | results/optimization/selected_configs.json and trials.csv |
| Full comparison and validation status | results/metrics/all_model_comparison.csv and results/frozen/availability.json |
| Controlled ablation | results/ablations/src_ablations.csv and identical_adapter_controls.csv |
| Fault diagnostics and scope | results/fault/fault_group_summary.csv and fault_status.json |
| Full comparator rerun | results/experiments/baseline_refit/fresh_refit_reconciliation.json |
| Implementation and timed talks | Project README, executed notebooks, companion guide and both editable decks |

The report archive contains the editable source, generation tools, citation sources, plots and staged evidence. It builds without loading the original recordings. The broader implementation archive contains fitted checkpoints, configurations and reuse instructions.
