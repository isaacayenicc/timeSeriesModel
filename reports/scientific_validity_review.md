# Independent scientific validity review

Reviewed actual source and accessible input audit on 2026-10-30. This review is
independent of selecting the best validation result; it looks for claims that
the experiment design cannot support. Final numerical rescore/test outputs are
linked separately rather than presumed from source inspection.

## 1. Target leakage and future input

**Eligible predictor code passes source inspection.** Windows end at x[t-1];
speed inputs end at s[t-1]. MLP, one-direction GRU and explicitly left-padded TCN
do not expose x[t]. FiLM uses speed only at each past window position. Signal
processing transforms are downstream diagnostics, not prediction inputs.
Future-perturbation tests exist for both vibration and speed, as well as a
TCN-block causality test. Their final executed status is recorded in the test
report, not inferred merely from their existence.

**Deliberate exception:** `shift_tcn` circularly shifts off-line speed and can see
future speed. It remains a noncausal control, excluded from model recommendation.
It must not be described as strictly causal merely because its convolution is
left-padded. Its initial train-window versus validation-sample roll mismatch was
identified during review; source was corrected to shift raw speed consistently
by N//3, and a replacement run with explicit noncausal status and source/config
fingerprints was saved after correction. This does not affect the causal
models because they never use the shifted channel.

## 2. Common indices and normalization

**Source protocol is fair for primary validation MSE.** All runs score indices
256:24000 with supplied normalized x. Smaller contexts still discard the common
burn-in. Linear arrays use float64 and neural arrays float32, disclosed in saved
configs. Saved per-run predictions and checkpoint rescoring enable independent
numeric checks. An independent execution of
`python scripts/review_experiment_integrity.py` checked **all 31 saved runs**:
every saved target index is exactly 256:24000, every y_true is exactly the
supplied validation slice, residuals equal y_true-y_pred, every saved MSE is
recomputed within 1e-14, both input hashes match, and the noncausal control is
excluded from winning-model selection. See
[integrity JSON](independent_integrity_review.json). This check rescored saved
arrays, not loaded models; checkpoint rescoring is recorded separately.
The same normalization is used for raw/residual diagnostic
comparisons; no per-record standardization removes amplitude changes.

The supplied scalers reconstruct the actual files exactly. Their original
upstream fitting provenance is not documented; no stronger train-only origin
claim is warranted. See [normalization audit](../docs/normalization_and_leakage_audit.md).

## 3. Tuning, seed selection and optimism

**Material limitation.** sigvalid is used for early-stopping/checkpoint selection
and comparing model/context/seed outcomes. Calling it validation is correct;
calling its final winning score an independent generalization estimate is not.
Multiple seeds are training repeats, not independent acquisition repeats.
The bounded declared model plan reduces uncontrolled search but does not remove
selection bias. The reference models were selected before the later fault-part uploads. The 200 candidate records were subsequently processed with those frozen artifacts and did not inform fitting or hyperparameter tuning. No final
test set was used or synthesized.

## 4. Missing test performance

sigtest and sigfault2 were not provided. There are no measured results from
either. Tables labeled validation must remain labeled validation in slides,
notebooks, README and conclusions. A paper's same-data statement does not turn
this split into an independently reproduced published comparison.

## 5. Fault classes, onset and severity

**Real descriptive fault analysis is complete.** The later twenty individually uploaded MAT parts were independently verified for byte/field digests, ranges, total count, ordering and scientific vector/time validity. All200 complete32000sample records were processed separately, using13 causal family representatives selected by original validation MSE. Every raw/residual comparison uses common indices256:32000. The 2,800 time-feature and8,400 angular-sensitivity rows contain source IDs and part hashes. No health, severity, onset or class field exists in the inspected eight-field structs, and no external label mapping was supplied.

All200-record computations completed before a plotting failure caused by pandas' `DataFrame.kurtosis` method/column name collision. Explicit column access fixed rendering; AST comparison verified that all non-rendering code was unchanged, so verified computed caches were retained rather than unnecessarily repeating inference. The repair and source fingerprints are recorded in [fault_plot_repair.json](fault_plot_repair.json). The final completion marker was written only after corrected plots and complete tables existed.

Residual RMS, kurtosis, crest factor, synthetic pulse response and unit-sensitivity
plots on accessible train/validation signals are useful diagnostics. They do not
establish fault detection accuracy, ROC/AUC, sensitivity, specificity, F1, crack
severity or onset. No threshold is optimized to pretend these are physical labels.

## 6. Angular units and MATLAB parity

**Physical interpretation is unverified.** Source speed units are not encoded in
the accessible data. Explicit rpm/rps/rad_s branches are sensitivity assumptions.
The assumed order of one cycle is not a verified tooth-fault order. The missing
custom MATLAB angular-resampling implementation prevents asserting its exact
units/output convention or numerical reproduction.

Python's two PMI variants isolate the actual source coarse-branch indexing
difference: MATLAB's one-based acf(P) selects zero-based lag P-1, whereas the
intended one-period lag is P. The source's returned timeindex also has an offset.
The analytic-envelope convention is explicitly named a candidate; live MATLAB
comparison remains unverified. The FFT odd-length final positive bin is corrected
in the scientific variant and exposed unchanged in the reference variant.

`python scripts/run_numerical_validation.py --run-tests` regenerated the actual
raw/reference and best-model residual comparison, and 28 diagnostic tests passed.
The best LPV residual RMS is 0.6010137001 versus supplied validation x RMS
1.0507508347 (ratio 0.5719849847). Its corrected PMI is 0.040146, 0.046954 and
0.041046 under rpm/rad_s/rps assumptions, respectively; raw x yields 0.084614,
0.068392 and 0.188517. Thus this prediction improvement does **not** coincide
with larger PMI on the available reference trace. Neither direction establishes
superior physical-fault performance without validated physical labels and unit calibration.
See [full precision sensitivity CSV](angular_unit_sensitivity.csv).

The measured angular round-trip error is appreciable. Interpolation can change
PMI and CK without any change in machine damage. Using enough samples/cycle
avoids nominal downsampling but does not eliminate linear-interpolation filtering.
These are central limitations, not minor numerical footnotes.

## 7. Uncertainty

Nonoverlapping blocks preserve local sequences for a within-trace bootstrap, but
only one validation recording exists. Blocks are not independently acquired
machines; there is no population-level confidence interval. Intervals are not
post-selection corrected and may remain optimistic. Different block lengths
256/512/1024 are an appropriate sensitivity check, not proof of full independence.
Bootstrap tables exclude incomplete tails and must not be confused with the
full-trace MSE when the tail changes their mean.

## 8. Published conventional benchmark

The implemented polynomial speed-interaction LPV-AR is a **surrogate**, lacking
the cited article's innovations/MA recursion, exact sparse estimation and verified
experimental alignment. Better scores than local AR or zero baselines cannot
prove that ML beats published Spa LPV-ARMA. No directly comparable published
score is invented, and no reproduction claim is justified by the citation alone.

## 9. Source verification

The literature access log must distinguish original full text, abstract,
bibliographic metadata and inaccessible sources. Knowledge of a DOI or title is
not full-text access. Publisher and official-document denials remain denials;
familiarity with a method cannot substitute for source verification. Paper-specific
technical details unavailable in accessible text are treated as unknown rather
than reconstructed by guesswork.

## 10. Reproducibility

The package saves model state, effective run config, data hashes, prediction
indices and metrics. Loading uses locally created trusted pickle/framework
artifacts; untrusted pickle files must not be executed. Rescoring compares full
predictions and MSE after reload. Timing includes loading/window construction
and full-record scoring, with one warmup and three repetitions; it is not
streaming device-only latency. Current-machine execution cannot prove that
Windows/macOS or GPU operation was tested when it was not.

The repeatable numerical diagnostic script regenerates raw/residual unit
sensitivity from actual files and the saved recommended checkpoint. Raw data
remain user supplied and outside the release's redistribution assumption.
Final release hashes, notebook execution, presentation validation and passed test
counts must be taken from actual final check outputs. This review does not
substitute for those runs.

## Verdict

The design can support a provisional causal one-step prediction comparison and
carefully qualified residual/numerical diagnostics on the supplied reference
traces. It cannot establish independent test generalization, unknown physical
units, fault-label performance, published Spa LPV-ARMA superiority or live MATLAB
identity. The uploaded split dataset now supports the complete requested descriptive200-record comparison. Physical detection accuracy remains unidentifiable without labels, and exact published-method reproduction still requires the full paper. No remaining data-transfer blocker is claimed.


## Actual full candidate-study evidence

The final summary reports200-records,13 predictors plus raw,2,800 time-feature rows and8,400 angular rows. For the validation-selected LPV-AR surrogate, median residual RMS is0.6239737962 versus raw1.0534097625; the median paired residual/raw RMS ratio is0.588109. Raw/residual kurtosis medians are3.391280/3.272660 and crest factors4.814118/4.908411. Corrected PMI medians decrease under all threeunit assumptions (rpm,rad_s,rps), which rules out casually equating energy reduction with a universal indicator increase. Larger PMI would not establish diagnosis either. No candidate outcomes were used to reselect the primary recommended model.

Separate processing establishes record isolation, not statistical independence of acquisitions or a known health progression. Original monolithic MAT byte provenance is declared by the source manifest; uploaded parts and logical data are independently verified. See [full study summary](fault_study_summary.json), [paired model summary](fault_paired_summary.csv) and [candidate angular features](fault_angular_features.csv).

The expanded research matrix has19 publication records with explicit evidence depth, not19 full-text readings. Twelve machineryauthor sources improve relevance; full lecturer-method, exact PMI/MCKD/order-tracking literature gaps remain. No article full text was read.
