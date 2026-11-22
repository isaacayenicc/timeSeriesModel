# Executable comparator suite

The suite evaluates 21 requested families with the exact 22,208 supplied development targets at zero-based indices [1792, 24000). Every checkpoint exposes a length-preserving one-step predictor. All runnable fits use measured vibration through $t-1$. Speed-aware fits may use historical RPM through $t-1$. AR, ARMA and ARIMA are intentionally vibration-only.

## Source lineage and training evidence

The statistical, tabular, neural and attention code originated in the project-authored expanded gearbox study. The SRC package includes adapted local copies, the original 21 model checkpoints, the selected configuration and SHA-256 for each, and their guarded trial tables. Core estimator logic retains the original forecast calculations. The local MAT reader and registry remove prior-project dependencies. A separate replay recalculates metrics from fresh local predictor calls. The results identify existing fitted checkpoints as such. They do not imply 21 new searches in the SRC run.

The original study fitted 19 families over 126 trial configurations and 378 guarded fold rows. Both sparse LPV checkpoints use the previous training-only supports, selected using 36 ADMM fold results and verified converged. The SRC reference stage refitted both final supports by QR on training data and checked equality with the frozen support coefficients. The two stages recovered coefficient arrays to an absolute difference no greater than $2.22\times10^{-16}$. The first four statistical families use fixed causal histories or a finite approximation of a stable innovation filter. Linear models use fold-fitted input transforms. Tree and kernel estimates retain their justified finite search and training-row budgets.

These are project study materials supplied or generated for this work. The third-party numerical dependencies remain under their published package licenses. The source archive's checkpoint bytes are preserved and the numerical payloads use renamed SRC identifiers to follow this project's naming convention. No separate open-source licensing grant is asserted for user data, archived parameters or private project-authored work.

## Model formulations and limits

| Family set | Implementation and evidence |
|---|---|
| AR, ARMA, ARIMA, ARX | AR uses lagged vibration and QR estimation. Conditional ARMA uses a high-order AR proxy at fit and its own causal innovation recursion at inference, with a stable inverse truncated to the 1,792-lag support. ARIMA restores first differences to the target scale. ARX adds positively lagged RPM with scaled ridge regression. |
| Sparse LPV-AR and LPV-ARMA | Previous-RPM scheduling with eight shifted Legendre terms. One stage fits vibration lags and the second incorporates older stage-one residuals. Fits on sealed supports retain 475 stage-one and 365 stage-two coefficients. |
| Linear, ridge, Lasso and Elastic Net | Causal delay and speed/physics libraries, with design-specific scaling fitted within training blocks. Ridge is L2 and Lasso is L1 penalized. Elastic Net combines both; prior diagnostics record iteration bounds and convergence. |
| SVR | Exact RBF fit on 4,500 deterministic time-stratified training rows in the selected version. Validation uses every common target at 3.2 kHz. |
| Random forest and gradient boosting | Portable threshold arrays implement forest averaging and histogram boosting. The parameter column represents stored nodes for these models. |
| NARX | Physics-feature ridge with a learned tanh nonlinear residual readout using a compact feature library. The signed observed vibration history supplies autoregressive feedback. |
| MLP, LSTM, GRU, CNN and TCN | Shared affine shortcut plus nonlinear encoding. The recurrent branches are unidirectional and reset per window. CNN padding lies inside complete past history. TCN convolutions use explicit left support. Inputs retain all time samples inside each eight-value patch. |
| Transformer and compact TFT | Causal historical input only. Transformer uses sinusoidal positions and strict masking. TFT combines variable selection, gated residual processing, bounded recurrent history and shared-value attention for one output. TFT has no static or known-future series in this dataset. |

The main table records feature histories and normalized MSE, RMSE, MAE, NRMSE with measured target SD as the denominator, residual bias, autocorrelation, runs and Ljung–Box diagnostics. Recomputed train MSE supplies a descriptive fit-to-development gap. These are conditional one-step forecasts. The guarded training-fold splits cover one autocorrelated acquisition and cannot substitute for independent test acquisitions.

The sensor-input control refits nine selected architectures with RPM variation withheld at training and inference. Filling the RPM slot with 145 RPM preserves shapes and parameterization. For NARX this fixes nominal frequencies and angular conversion in the inherited physics library. No measured RPM variation enters that control. The speed-aware model uses the preserved selected checkpoint, with identical fit configuration and training seed. Positive change in the reported measured-minus-withheld error means measured RPM increased observed MSE. This probes input utility at the selected configuration without retuning on development. The fresh seed-sensitivity script tests fixed configurations with two predeclared supplemental seeds and retains the original seed in summary statistics. It leaves the family selections unchanged.

## Statistical and archival references

The external paper reports the authors’ own validation segment. Its Table 6 gives 0.2893 for sparse LPV-ARMA. It uses a current-RPM scheduling convention, and the original segment corresponding to its Fig. 13(c) was unavailable. The authors’ §3.2 selection rule uses one sample standard deviation. The experimental design and simulations are distinct in the article. The local reference suite separates the strict development result from historical current-RPM runs and a controlled one-sample schedule shift on frozen historical coefficients. Five randomized training-row folds were used for the historical-convention diagnostics and are descriptive only due to shared histories and estimated proxy errors. The Legendre and printed monomial parameterizations produce different Lasso supports and convergence profiles. [Chen et al. (2025)](https://doi.org/10.1016/j.ymssp.2024.112161)

The original archive named `Gearbox_DHON_NGRC_Improved_Optimization.zip` supplied fixed-bank optimization and local-refinement checkpoints. `archived_src_corrected` is its optimized checkpoint with the archived causal correction. The checkpoint preserves the prior numerical model, including its historical RPM normalization, fixed harmonic decay bank, nonlinear design, readout and clipped correction. The older `archived_src_earlier_corrected` uses only archived forecasts because a runnable checkpoint was absent. The official course response and mask are hash-checked before rescoring those forecasts. The archival scores stay outside the 21 baseline count.

## Commands and outputs

From the SRC project root after creating the environment and installing the project in editable mode, supply the official MAT data directory explicitly.

```bash
python scripts/06_train_baselines.py --tier smoke
python scripts/06_train_baselines.py --tier standard --data-dir data/raw
python scripts/02_reproduce_lpv_reference.py --data-dir data/raw
python scripts/06_rpm_parity.py --data-dir data/raw
python scripts/06_baseline_seed_sensitivity.py --data-dir data/raw
python scripts/06_train_baselines.py --tier all --data-dir data/raw --out results/experiments/rebuilt_baselines
```

The standard run verifies immutable checkpoint and input hashes, then evaluates all 21 models and the archive references. The all tier repeats the complete guarded grids and refits the sealed sparse supports into a distinct run path. Resuming verifies the training data, development data, configuration and completed stage checkpoint hashes. The optional `--fit-paper` switch on the reference script repeats the historical five-row-fold diagnostic fits. It retains their separately labeled window and scheduling convention. The report's evidence separately identifies prior fits, fresh support refits, frozen replay and new input/seed controls.

| Artifact | Meaning |
|---|---|
| `configs/baselines/selection.json` | Protected dataset and checkpoint hashes, chosen configuration, earlier fit statistics and support provenance |
| `results/metrics/baseline_prior_trials.csv` | 378 prior training-only guarded-fold trials for 19 families |
| `results/metrics/baseline_comparison.csv` | Measured full comparator table with shared mask, error and latency |
| `results/metrics/model_coverage.csv` | Actual replay outcomes and limitations by family |
| `results/predictions/baselines_sigvalid.npz` | Ground truth, RPM, time, target indices and 21 predicted vectors |
| `results/predictions/baselines/<model>/sigvalid.npz` | Model-associated prediction, residual and common indices |
| `results/metrics/archived_src_metrics.csv` | Frozen optimized/local SRC replay and stored-forecast-only variants |
| `results/metrics/{local_statistical_reference,published_paper_reference}.csv` | Distinct local and published reference conventions |
| `results/metrics/vibration_rpm_parity.csv` | Nine fixed-configuration, independently refitted input-withholding controls |
| `results/metrics/baseline_seed_sensitivity_summary.csv` | Actual seed variability on declared representative selected architectures |

Timed results report CPU batches with one BLAS/PyTorch thread and elapsed full-record feature-to-output work from in-memory arrays. Model loading and cold call are separated. The per-repeat values and median/p95/IQR refer to warmed full-batch calls rather than per-target streaming deadlines. The cumulative process RSS value is a measurement of the running benchmark process and is not per-model allocation. CUDA was unavailable. Fit times are labeled prior fits, current support refits or new fixed-config controls as applicable.

## Fresh full comparator rerun

A separate execution of the complete guarded comparator workflow finished all 21 models. The main ranking continues to use the predeclared comparator registry. The fresh selected settings, model checkpoint bytes, reloaded predictions and matched MSE reproduced those registry outcomes exactly. The audit verified 57 selected guarded-fold rows across the directly selected families. The two sparse models refit the original guarded supports. Their 475 and 365 nonzero masks and frozen RPM calibration matched the retained source. The maximum change in sparse coefficients against the original source was 2.22e-16. See the model-by-model [reconciliation](../results/experiments/baseline_refit/fresh_refit_reconciliation.csv) and [summary](../results/experiments/baseline_refit/fresh_refit_reconciliation.json).
