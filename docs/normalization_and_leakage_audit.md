# Normalization, causality and leakage audit

This review inspected the actual loaders, audit, window builder, neural forward
paths, experiment selection and stored-source protocol on 2026-10-30. It
separates observed code behavior from source metadata whose original provenance
cannot be independently reconstructed.

## Verified amplitude and speed reconstruction

Both accessible MAT files contain 24,000 samples at 3,200 Hz. Using their first
stored scaling entries gives exactly, to the executed float64 reconstruction:

\[
x=(x^{raw}-0.0026221941677904226)/0.42517293168840065,
\]

\[
s^{norm}=(s-124.07455535625635)/(167.3376580579989-124.07455535625635).
\]

The measured maximum absolute reconstruction error is zero in both files for
both formulas. See [data audit JSON](../reports/data_audit.json) and
[manifest](../data/manifest.json) for exact observations and hashes.

The stored normalization is **not** equivalent to standardizing each accessible
record by its own empirical moments:

| Quantity | Train | Validation |
|---|---:|---:|
| Empirical xraw mean | 0.00131286051949 | -0.00241651362973 |
| Empirical xraw population std | 0.44222170846122 | 0.44527014699716 |
| Empirical supplied x mean | -0.00307953199914 | -0.01185096091962 |
| Empirical supplied x population std | 1.04009845289332 | 1.04726833203834 |

The project uses supplied x and spdnorm directly, without fitting a scaler on
validation or re-standardizing records. No speed clipping is applied. Original
fitting provenance of the provided metadata is unknown: correct reconstruction
does not prove the upstream scaler was originally estimated using training data
only. The other entries of three-element xmean/xstd arrays are retained without
inventing an interpretation.

Normalized vibration and residuals are dimensionless. If a raw-scale comparison
is explicitly desired, prediction errors multiply by sigma and MSE multiplies by
sigma²; the physical unit of xraw remains unverified. RMS/PMI sensitivity uses
supplied normalized x as the raw-series reference, not xraw and not re-scaled x.

## Window and prediction causality

[windows.py](../src/rotmach/windows.py) constructs the first L-sample window as
x[0:L] and its target as x[L]. The final predictor input is x[t-1], never x[t].
LPV interactions use s[t-1]; ARX uses s[t-1] and its square. Neural speed
conditioning uses observed speed at positions within the same past window.

The MLP uses only the vibration lag vector. GRU is unidirectional and starts a
fresh hidden state for each finite window. TCN convolutions use only left
padding. A block with two kernel-3 convolutions at dilation d adds 4d samples
to receptive field. For D blocks with dilations 1,2,...,2^(D-1), the field is
1+4(2^D-1), chosen to cover the declared window. FiLM applies each past position's
own speed and cannot access subsequent positions. There is no centered filter,
full-record normalization fit, FFT or angular resampling in model inputs.

Tests perturb vibration **and** speed from index 200 onward and compare windows
and neural predictions for targets <=200. These earlier predictions remain
unchanged; target 200 correctly sees only positions through 199. A separate
TCN-block test perturbs later positions and checks earlier block outputs.
See [test_core.py](../tests/test_core.py). These tests support the causal input
claim for eligible predictors, not the deliberately noncausal control below.

## Explicit off-line speed-shift control

`shift_tcn` intentionally circularly shifts the **complete** speed trace by
N//3 samples. Circular wrap means some early predictions see speed from later in
the record. This is a **noncausal off-line negative/control experiment**, not a
deployable predictor. No future vibration is exposed by that transformation.
It is excluded from the recommended-model selection in `collect` and must be
labeled wherever its metrics appear. The initial implementation shifted training
speed at the window level; it was corrected to use the same full-trace speed
shift as validation/loading, and a replacement run with source/config
fingerprints was saved after that correction. See final reproducibility evidence
for the executed result.

## Split and sample fairness

Training windows come only from sigtrain, and validation windows only from
sigvalid. Random training **batch order** is allowed: it does not create a random
holdout split or make overlapping windows independent. There is no asserted
internal blocked/embargoed holdout. Each complete official recording remains
distinct. Fault data and optional test data were not used for fitting or tuning.

All primary comparisons use the same targets, zero-based indices 256 through
23,999 inclusive: 23,744 teacher-forced one-step predictions. L ranges from 32 to
256, with shorter-context warm-up samples excluded equally from scoring.
Saved predictions carry target indices, time, y_true, y_pred, residual, run ID
and record ID. Evaluation reloads the checkpoint and checks predictions and MSE
against the saved result. Raw/residual diagnostic comparisons use the identical
segment after this burn-in.

## Selection optimism and limits of inference

The same validation trace is used for epoch checkpoint selection, comparison
across architectures/contexts and selection of a recommended seed. Its final
MSE is consequently a **provisional model-selection score**, not an unbiased
estimate from an untouched test set. No test file was supplied; no final test
performance is reported. A better selection policy with only these recordings
would reserve a blocked internal training segment for early stopping, then use
sigvalid once. That would be a new experiment, not a retrospective correction
to the current reported scores.

Measured train/validation speed Pearson correlation is 0.9999995705325337, and
all validation speeds lie within the observed training range. Vibration Pearson
correlation is only 0.04129076012774545; similar speed is not proof of duplicate
vibration or data leakage. It does limit any claim of transfer to a new operating
trajectory. Physical independence of machines, acquisition sessions or damage
states cannot be established from these arrays alone.

Blockwise bootstrap intervals summarize fluctuations along the single
validation trace. They do not create independent machine replicates, correct
model-selection optimism, or quantify uncertainty in a population of machines.
Independent initializations characterize training stochasticity only. These
different uncertainty sources must remain distinct in claims and plots.

## Remaining boundaries

- Source units of speed and physical vibration remain unresolved.
- The twenty complete-record parts are now verified: 200 records, 32,000 samples
  each, with matched file/logical digests and source order.
- Fault processing preserves boundaries and resets finite-window/recurrent state;
  the scalar-feature tables use 31,744 common samples after burn-in256.
- Whole-record Hilbert envelopes, spectra and angular interpolation are used
  only as post-prediction diagnostics; they are not causal feature extractors.
- None of these checks establishes a fair comparison with the cited Spa
  LPV-ARMA paper, which has not been independently reproduced.
