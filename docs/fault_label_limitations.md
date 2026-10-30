# Fault diagnostics: provenance and limits of inference

The later individual uploads resolve the original transfer obstacle. All twenty
MAT parts were independently checked against their supplied file/field digests,
record ranges and global logical digest. The resulting 200 separate records have
32,000 samples each, fs=3,200 Hz and exactly the eight documented source fields.
No health, severity, onset or class field is present. The monolithic original
file's byte fingerprint is declared by the source manifest, not independently
recomputed from unavailable original bytes. Logical split consistency is verified.

`sigtest.mat` and `sigfault2.mat` were not supplied. Performance on either is
unmeasured. The validation trace is a model-selection trace and is not an
independent final test after it has informed checkpoint selection.

## What the available data can establish

The train and validation vibration traces support one-step prediction MSE,
residual statistics, spectral/ACF structure and speed-conditioned comparisons on
their actual samples. These are descriptive predictive results on the supplied
reference recordings. The name `sigtrain` alone does not prove healthy machine
status. The lecturer's paper describing an experimental condition does not
establish a source label for a particular externally provided trace.

The implementation in [diagnostics.py](../src/rotmach/diagnostics.py) accepts
independent record dictionaries. It creates `fault_candidate_record_001` and
subsequent provenance IDs, processes each record independently, and compares raw
x and model residuals on the identical segment after a common burn-in. Prediction
callbacks are invoked separately per record and must initialize any recurrent
state within each call. It does not fit a model on the candidate records.

**Record identity is not class identity.** Even if a later plot separates records
1–100 from 101–200, this observation does not establish healthy/faulty labels,
fault onset or crack severity. Such a partition is post hoc descriptive analysis
unless its meaning is independently confirmed. Index correlates can also reflect
operating condition, acquisition order or preprocessing differences.

## Why better prediction does not prove better detection

A predictor can remove repeatable reference-state vibration, making unusual
impulses or modulation clearer in e=x-yhat. But a sufficiently flexible predictor
can also predict fault-related periodic content, especially during teacher-forced
one-step inference on a faulty trace. Low MSE can therefore coexist with reduced
fault-indicator contrast. RMS, crest factor, kurtosis, CK and PMI describe
different properties; there is no guaranteed monotonic relation to physical
damage or to one another.

Comparing residual statistics with raw x is useful only on equal samples and
with fixed preprocessing. Per-record renormalization could erase amplitude
changes and is not applied by the diagnostic code. The supplied normalized x is
dimensionless; physical acceleration units must not be attached without source
calibration information.

## Units and angular periodicity

Source speed units are not verified. Angular resampling requires explicit rpm,
rps or rad/s assumptions and reports them. The argument `fault_order=1` is an
assumed one-cycle modulation for a sensitivity exercise, not a verified gear
tooth crack characteristic order. Gear tooth count, shaft identity, transmission
ratio and defect frequency are not inferred from a plot. See the measured
assumption sensitivity in [numerical validation](../reports/numerical_validation.md).

The supplied MATLAB `toConstantSamplesPerCycle` dependency is unavailable. The
Python method is transparent trapezoidal integration plus linear interpolation,
with output in samples/cycle, endpoint loss and round-trip error recorded. Live
MATLAB/toolbox identity has not been verified. Large interpolation losses can
confound raw/residual indicator comparisons and must be assessed before making
physical conclusions.

## What is required for future detection claims

1. The actual candidate recordings are now retrieved and audited. Preserve the
   recorded hashes, source IDs, speed coverage and preprocessing in future use.
2. Obtain a source-verified mapping of record IDs to physical machine conditions,
   including what fault type/severity or absence of damage means.
3. Verify speed units, shaft coordinate and any proposed fault order from the
   experiment documentation rather than selecting the interpretation that gives
   the largest PMI.
4. Freeze prediction models and feature definitions before selecting thresholds.
   Any threshold tuning must use a designated training/calibration subset of
   independent records. Evaluate once on a separate labeled record set.
5. Report class counts, label provenance, threshold policy and record-level
   uncertainty. Only then compute sensitivity, specificity, ROC/AUC, F1 or
   accuracy, with a clear distinction between calibration and evaluation.

Synthetic periodic impulses and known-speed chirps verify code behavior. They
remain synthetic and cannot substitute for labeled physical damage experiments.
No physical-fault classification metrics are reported for the current inputs.
