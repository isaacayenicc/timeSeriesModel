# Fault diagnostics and model robustness

## 1. Compare raw vibration with forecast residuals

A local tooth crack can generate a transient when the affected tooth enters mesh. At approximately constant shaft speed, impacts arrive at nearly constant time spacing. Under variable speed, the time spacing changes while the impact angle remains approximately fixed. A shaft-relative analysis tests for this periodicity without requiring the same number of time samples per rotation. Chen et al. (2025), §3 and Eqs. (16–21), motivate PMI with an envelope correlation at the crack period. Chen (2020), §2.2.3, p. 45 and Eqs. (2.7–2.8), explicitly centers the envelope before computing the ratio. Positive PMI can also arise from healthy periodic modulation, so the signal statistic needs an independent calibration and verification record before it can support a fault declaration.

For each record, first predict vibration from measured samples and speed through time index $t-1$. The observed target gives $e_t=x_t-\widehat{x}_t$. Compute diagnostic features on the common slice $t\in[1792,N)$ for every forecast residual and for the raw signal. For the development fault collection this gives 30,208 time samples per record at 3,200 Hz. The order transform is offline. Its interpolation endpoints and full-segment FFT may depend on later samples within the diagnostic slice. Neither is an input to the predictor.

## 2. Trace the units from time to angle

Let $n_i$ be the measured tachometer speed in revolutions per minute. The trapezoidal integral

$$
u_i=u_{i-1}+\frac{n_{i-1}+n_i}{120}(t_i-t_{i-1})
$$

is in revolutions. Set $u_0=0$ at the first retained sample and use linear interpolation to obtain a vibration series $x_\theta$ on a uniform grid $m/S$, $m=0,1,\ldots,\lfloor S u_{N-1}\rfloor$. The grid ends inside the measured interval. At 150 rpm and 3,200 Hz, the time series contains 1,280 measured samples per revolution. At 167.4 rpm, it contains about 1,147. The 2,048-point grid keeps angular Nyquist above the acquisition's highest resolvable order across the observed speed range and preserves an interpolation-friendly interval. It adds no frequency information. A 1,024-point grid can fold measurable orders above 512, and a 512-point grid puts order 259 above its Nyquist. Compare the 1,024, 2,048 and 4,096-point grids on fixed representative raw records 1, 100, 101 and 200 in [angular_resolution_sensitivity.csv](../outputs/runs/development/metrics/angular_resolution_sensitivity.csv). Those cases illustrate sensitivity while keeping resolution fixed for every predictor.

A constant offset in grid angle changes the phase origin of a spectrum but preserves the lag for one measured-shaft turn. Using shaft angles also permits a residual amplitude spectrum in units of dimensionless vibration per order bin. The project uses a one-sided Hann-corrected amplitude spectrum. Its FFT magnitude is divided by the window sum and interior bins are doubled. Acceleration can broaden resonance features in order even after resampling. Finite-grid interpolation attenuates high orders, the Hann window spreads off-bin energy, and acquisition at 3,200 Hz has already limited the time-domain bandwidth to 1,600 Hz.

## 3. Derive periodic modulation intensity

Write $a_j=|x_\theta(j)+\mathrm i\mathcal Hx_\theta(j)|$ for the envelope and $b_j=a_j-\bar a$ for its centered version. Use

$$
R_b(k)=\sum_{j=0}^{M-1-k}b_jb_{j+k},\qquad
r(k)=\frac{R_b(k)}{R_b(0)},\qquad
\operatorname{PMI}(k)=\frac{r(k)}{1-r(k)}
                      =\frac{R_b(k)}{R_b(0)-R_b(k)}.
$$

There are $M-k$ valid products at lag $k$. Divide by the same total centered envelope energy $R_b(0)$ at each lag. Dividing by $M-k$ at one lag and by $M$ at the other would define a different finite-record ACF. A negative finite-record score means the selected lag had negative centered correlation. A constant envelope or an inadequate record span yields an explicitly invalid score. An unrelated 0.63-revolution lag is recorded as a diagnostic control.

For the envelope $[5,3,5,3]$, centering yields $[1,-1,1,-1]$. Then $R_b(0)=4$, $R_b(1)=-3$, and $R_b(2)=2$. The PMI values are $-3/7$ and $1$. Even the perfectly repeated period-two pattern gives $r(2)=1/2$ because the truncated sum contains two products. This example shows why comparison masks and episode duration affect PMI.

## 4. Instructor source and Python modes

The uploaded `fdt_featcal.m` has SHA-256 `6c1206b0cb7e1098b6b9b52cf437e2ba5097e05334e9c00cee51e5fc8fabbf06`; the copies in the EDA and DHON archives are byte-identical. Its top-level script computes RMS, `peak2rms`, default `kurtosis`, `toConstantSamplesPerCycle`, first-shift `CorrKurt`, and `PMIca(resp,fsp,1,false)`. The first-shift correlated kurtosis is $\sum_j y_j^2 y_{j-T}^2/(\sum_j y_j^2)^2$ with zero padding before the record. Time-domain kurtosis uses the biased Pearson form, for which a Gaussian population has kurtosis three.

In the default analytic `envelope(d)` algorithm, the mean is subtracted before the analytic envelope is built and then added to the upper envelope. The following call to `autocorr(yupper-mean(yupper),\dots)` removes that added offset. The autocorrelation uses a common variance denominator and `acf(1)` represents lag zero. In the coarse `PMIca` branch, `ran=0` so `acf(round(fs/f_fault))` selects lag $T-1$. At 2,048 points per revolution and one impact order it selects lag 2,047. The associated `timeindex` output is unused by the top-level call. [MATLAB envelope](https://www.mathworks.com/help/signal/ref/envelope.html) · [MATLAB autocorr](https://www.mathworks.com/help/econ/autocorr.html) · [MATLAB kurtosis](https://www.mathworks.com/help/stats/kurtosis.html)

| Output | Envelope input | Angular lag | Use |
|---|---|---:|---|
| `pmi_course` | Remove signal mean before Hilbert magnitude | $T-1$ | Instructor-literal source indexing |
| `pmi_lag_corrected` | Same preprocessing | $T$ | Isolate the indexing effect |
| `pmi_math` | Hilbert magnitude of delivered signal | $T$ | Thesis Eq. (2.7) input and one exact revolution |
| `pmi_unrelated_0p63_rev` | Same preprocessing as course mode | $\operatorname{round}(0.63T)$ | Check lag specificity |

Development parity status is **PARTIAL**. The source and analytic fixtures verify intended conventions. `toConstantSamplesPerCycle`, the MATLAB runtime and reference output arrays were unavailable. Python's trapezoidal integration and interpolation are explicit alternatives to the missing helper. Exact equivalence needs instructor output for the same residuals, the angular grid, the upper envelope and the returned coefficient arrays. The separate `pmi_lag_corrected` column makes it possible to distinguish one-index sensitivity from DC preprocessing and grid sensitivity.

## 5. Interpret and verify group separation

The supplied collection has 20 numbered parts and 200 records. Records 61 and 63 have full joint vibration and speed overlap with official training and validation at offset zero. Their time and diagnostic rows remain in the output, with eligibility disabled. Group A comprises eligible records with index at most 100 and group B comprises eligible records above 100. The paper's parameters describe its published collection. The supplied MATLAB structs and separate attachments do not carry authenticated health labels or acquisition genealogy. A/B separation and bootstrap intervals refer only to the index groups. Bootstrap resamples records within each group and treats available IDs as independent units, an assumption that unknown acquisition grouping can violate. No normalization, threshold, confusion matrix, or sensitivity/specificity is fitted from provisional labels.

For a fixed score $s$, the reported descriptive rank AUC is
$P(s_B>s_A)+\frac12P(s_B=s_A)$, always with B higher. For the raw measurement the direction is therefore allowed to be below one-half. Compare this statistic with the signed median gap and separation margin $\min_B s-\max_A s$, which shows whether all available records separate under this ordering. The margin can change even when rank AUC is one. It is sample-specific, in units of PMI, and is not an operating margin for a qualified detector. Crest factor, kurtosis and alternate PMI lags expose whether separation is more specific than simple impulsiveness.

## 6. Stress testing and impulse preservation

Use the frozen model on the actual supplied validation record. The speed battery repeats the archive's six fractional RPM biases ($\pm0.5\%$, $\pm1\%$, $\pm2\%$) and three normalized smooth jitter levels (0.1%, 0.3%, 0.6%), with seed 303. Add quantization to 1 rpm, a 40 ms sensor delay, 40 ms hold-last-value dropout, and 0.3% independent white jitter. The experiment distinguishes errors injected into the model's RPM history, errors confined to diagnostic angular resampling, and errors applied at both stages. Forecast MSE remains referenced to the original observed vibration. A separate model-input vibration noise condition uses standard deviation 0.05 on the standardized vibration scale. Computed speed derivative sign is used for descriptive acceleration-stratified errors. For startup sensitivity, reset at 512, 1,024 and 1,792 samples before fixed target anchors 8,000 and 16,000. Report how many of the next 800 predictions are finite. Compare prediction deltas only where estimates exist; a shorter available history cannot silently shrink the reported MSE mask.

In the paired injection battery, define $d_t$ as a damped sinusoid after crossing selected shaft phase. Its carrier, amplitude, phase and decay are varied, with the archived cases’ 18 ms window or the additional cases’ 25 ms window, optional per-turn amplitude modulation and two speed sensor perturbations. Aperiodic controls retain similar event count and change the angle from one turn to the next. Each injection case is paired with a no-injection prediction under the same RPM error. For a frozen predictor $f$, the residual increment is

$$
\Delta e=(x+d-f(x+d))-(x-f(x))=d-\Delta\widehat{x}.
$$

Report distortion $\|d-\Delta e\|_2^2/\|d\|_2^2$, energy retention $\|\Delta e\|_2^2/\|d\|_2^2$, correlation between $d$ and $\Delta e$, and residual PMI change. High correlation can coexist with a residual increment of the wrong magnitude. Positive PMI response can coexist with a weak waveform correlation. Compare the joint behavior and record the raw-signal PMI change to see if filtering amplifies or suppresses the diagnostic shift. The injection amplitudes are standardized illustrative perturbations, rather than calibrated tooth crack sizes.

## 7. Replay the diagnostic calculations

Use `python -m gearbox_dhon.cli faults --data-dir data/raw --run-id development --device cpu` from the installed project after model fitting. This recomputes group metrics from the source records and frozen development checkpoints. The complete per-record table carries eligibility, model, group, PMI variants, time indicators and MSE. [fault_per_record.csv](../outputs/runs/development/metrics/fault_per_record.csv) · [fault_group_summary.csv](../outputs/runs/development/metrics/fault_group_summary.csv)

The four representative acquisitions have full aligned arrays under NPZ keys `{id}:t`, `{id}:rpm`, `{id}:index`, and `{id}:{model}:{feature}`. Saved per-record forecasts and residuals in this file are float32 for display. All scored metrics are computed before that display serialization. Per-record diagnostics for other IDs are replayable from the model and source files. [fault_representative_traces.npz](../outputs/runs/development/predictions/fault_representative_traces.npz)

Test isolated fixtures with `python -m pytest -q tests/test_faults.py`. The variable-speed order fixture integrates a linear RPM ramp and checks a 37th-order peak. The amplitude-modulated carrier fixture tests MATLAB mean handling and analytic PMI, while the periodic versus random angle pulse fixture tests specificity. New blind inputs require a previously frozen manifest and run through the blind CLI with overlap checks. Verified labels and an independently defined calibrated threshold are needed before reporting fault classification accuracy.

## Self-check

1. Why can the same crack be periodic in angle while its interval in seconds changes? A fixed tooth angle corresponds to one shaft revolution, whose time duration decreases as RPM increases.
2. Why does the coarse reference select $T-1$? Its first array element is at MATLAB index one, but represents lag zero. Index $T$ therefore corresponds to lag $T-1$.
3. Why might improving forecast MSE reduce PMI discrimination? Better fit can predict both normal structure and part of a genuine impulse's continuation. Use the paired residual increment to measure this effect.
4. How should a rank AUC of one be stated for these uploads? It describes full rank separation of groups A and B in this checked collection. Metadata and independent calibration are necessary to validate the classes and operating threshold.

## 8. Development results

The completed run produced 800 diagnostic rows from 200 records and four signal types. The table uses the 98 and 100 eligible records in groups A and B, respectively. Model residual and raw tables share the same 30,208 target samples per record.

| Signal | Median PMI A | Median PMI B | B−A median gap | Sample separation margin | Conditional AUC |
|---|---:|---:|---:|---:|---:|
| Raw vibration | 0.09714 | 0.08855 | −0.00859 | −0.11882 | 0.36429 |
| Learned DHON residual | 0.02893 | 0.13558 | 0.10664 | 0.05855 | 1.00000 |
| Strict Spa LPV-ARMA residual | 0.04014 | 0.11270 | 0.07255 | 0.01940 | 1.00000 |
| Corrected archive DHON residual | 0.02805 | 0.13379 | 0.10574 | 0.05746 | 1.00000 |

All three residuals separate the provisional groups at the literal and mathematical lags. The paired record bootstrap estimates a median-gap advantage of 0.03409 for learned DHON over strict Spa LPV-ARMA on these data, 95% interval [0.02978, 0.03838]. The analogous gap difference against corrected archive DHON is only 0.00091, interval [−0.00191, 0.00403]. At the unrelated 0.63-revolution lag, the learned and archived DHON residuals have conditional AUC 0.5180 and 0.5089. The choice of rotational lag materially affects their group separation. [Group table](../outputs/runs/development/metrics/fault_group_summary.csv) · [PMI figure](../outputs/runs/development/figures/fault_pmi_groups.png) · [Paired audit](../outputs/runs/development/metrics/fault_verification.json)

Of the 13 predictor-RPM stress scenarios, the 2% high RPM bias causes the greatest MSE in each model. The worst MSE is 0.47282 for learned DHON, 6.55481 for strict Spa LPV-ARMA and 0.45255 for corrected archive DHON. This perturbation places about 14.7% of scheduled speeds above the frozen normalization maximum, at up to 170.67 rpm versus 167.34 rpm in the training schedule bounds. The sparse model uses an unconstrained polynomial scheduling basis, so extrapolation is a relevant limitation for that fitted model. Diagnostic-only RPM errors leave prediction MSE fixed by construction and change the measured period through the angular grid. [Stress table](../outputs/runs/development/metrics/rpm_stress.csv) · [Stress figure](../outputs/runs/development/figures/rpm_stress_pulses.png)

Learned DHON raises residual PMI in 13 of 15 periodic injection cases, compared with 9 for Spa LPV-ARMA and 11 for the corrected archive. The largest median waveform correlation with the injected pulse comes from Spa LPV-ARMA at 0.632, followed by learned DHON at 0.537 and the corrected archive at 0.527. These criteria measure distinct aspects of the residual. The archived predictor has the lower clean and stressed MSE of the two DHONs. The higher count of positive PMI changes for the learned model comes from one test record and a finite stress design. [Robustness summary](../outputs/runs/development/metrics/robustness_summary.csv) · [Injection table](../outputs/runs/development/metrics/pulse_preservation.csv)

The independent audit recomputed the 16 rank summaries, checked alignment across all 800 rows, replayed 16 PMI estimates from the stored representative traces and verified all three robustness summaries. The largest absolute PMI difference from diagnostic trace serialization was $1.21\times 10^{-9}$. Use `python scripts/audit_faults.py` from the installed project to rerun this audit. [Audit output](../outputs/runs/development/metrics/fault_verification.json)
