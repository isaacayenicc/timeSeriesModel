# Numerical validation of the Python diagnostic pipeline

Generated 2026-10-30. **Live MATLAB/toolbox parity is unverified.** The original
`fdt_featcal.m` was inspected, but MATLAB, its toolbox implementation of
`envelope`/`autocorr`, and the external `toConstantSamplesPerCycle` implementation
were not available. Passing the tests below establishes the declared Python
definitions, not undocumented toolbox equivalence or physical fault detection.

## Definitions and port decisions

`src/rotmach/diagnostics.py` computes RMS as the square root of mean squared
amplitude and crest factor as maximum absolute amplitude divided by RMS. Its
kurtosis is the biased Pearson fourth standardized central moment, using
`scipy.stats.kurtosis(fisher=False, bias=True)`. A Gaussian population has
Pearson kurtosis 3, whereas excess kurtosis subtracts 3. The selected convention
is explicit; live verification of the MATLAB default is not claimed. A constant
signal has undefined kurtosis, and a zero signal has undefined crest factor;
both return NaN rather than a physically misleading zero. Finite input is
required, and an entire matrix cannot accidentally become a flattened record.

For correlated kurtosis, the supplied MATLAB loop expands its initial N-by-M
array to N-by-(M+1). The Python formula is exactly the zero-padded expression

\[
\mathrm{CK}_{T,M}=\frac{\sum_{n=0}^{N-1}
 [\prod_{m=0}^{M}y_{n-mT}]^2}{(\sum_{n=0}^{N-1}y_n^2)^{M+1}},
\qquad y_j=0\ \text{for }j<0.
\]

Here T is an **integer lag in samples**, M is a nonnegative integer, and CK is
dimensionless. This is the reference's actual normalization, not a generic
fourth-moment formula. Its numerical value depends on record length, periodic
alignment and boundary padding. For `[1,2,3]`, T=1 and M=1, the numerator is
`0² + (2×1)² + (3×2)² = 40`; the denominator is `14² = 196`.
For T=1,M=2 it is `36/14³`. Zero energy returns NaN. Scaling the input before
calculation preserves the ratio while reducing overflow risk.

### Envelope and ACF

The base analytic envelope is `abs(scipy.signal.hilbert(y))`. The additional
`matlab_mean_restored_candidate` convention uses
`abs(hilbert(y-mean(y)))+mean(y)` as a **candidate** implementation of the
mean-removal/restoration described for MATLAB's analytic-envelope behavior.
The downstream ACF removes the envelope mean again, but the difference between
Hilbert transforms of centered and uncentered inputs can still matter.
This candidate has not been checked against a live MATLAB installation.
PMI defaults to this explicitly named candidate; the alternative is selectable.

For envelope u and centered z=u-mean(u), ACF is

\[
\rho_k=\frac{\sum_{n=k}^{N-1}z_n z_{n-k}}{\sum_{n=0}^{N-1}z_n^2}.
\]

This uses a common energy denominator, not the lag-wise N-k correction. It
therefore includes finite-record attenuation even for perfectly periodic data.
For example, a 4,000-sample, 100-sample-period envelope has rho at one period
0.975, not 1. Endpoint effects of the FFT Hilbert transform are retained.
An envelope whose relative range is <=1e-12 is numerically constant for this
implementation; normalized ACF and PMI are undefined. That policy prevents
amplification of floating-point roundoff from a constant-amplitude sinusoid.

### Two PMI indexing variants

Let P be positive MATLAB-style rounded `samples_per_unit/fault_order`. MATLAB
rounds a positive half-integer upward; Python's default `round` is not used.

| Variant | Python-selected ACF lag | Purpose |
|---|---:|---|
| `faithful_reference` | P-1 | Reproduce the supplied coarse branch's one-based `acf(P)` selection |
| `scientifically_corrected` | P | Select the intended one-period lag |

In both variants, PMI is rho/(1-rho). Negative rho and negative PMI are retained;
the coarse source branch does not take the absolute value. As rho approaches
one, the statistic is ill-conditioned. A denominator within 32 machine epsilons
of zero returns NaN and `undefined_near_unit_acf`; it is not silently capped.
The reference's coarse branch additionally returns `timeindex=P+1`, because its
single-element `max` index is one. That value is neither the selected zero-based
lag nor its proper one-based array index. The Python result exposes it as
`source_timeindex_matlab`, rather than treating it as a physical period.
The refined source branch's search around a supplied `timeindex` has not been
implemented; it is not presented as reproduced.

## Angular resampling and dimensional analysis

`src/rotmach/angle_resample.py` requires the caller to choose an explicit unit:

| Assumed source unit | Cycles/second used for integration |
|---|---|
| `rpm` | speed/60 |
| `rps` | speed |
| `rad_s` | speed/(2π) |

No supplied metadata establishes which interpretation is correct. Unknown units
raise an error in the angular routine; the higher-level record analyzer instead
omits angular results and records `blocked_unknown_speed_unit`.

Cycles are integrated by the trapezoidal rule from the first measured timestamp,
with cycles(t[0])=0. No unavailable initial sample or pre-record rotation is
invented. Strictly increasing timestamps and strictly positive finite speed are
required. Linear interpolation maps to a uniform cycles grid, ending within the
measured interval; no extrapolation is permitted. The second quantity is
**samples/cycle, not Hz**. This is a transparent Python method, not a verified
replacement for `toConstantSamplesPerCycle`, whose source was not supplied.

The implementation reports lost cycles at the last grid edge, a warning if
output samples/cycle can be below source samples/cycle, and round-trip RMSE.
Linear interpolation is not bandlimited and provides no anti-alias filter.
Choosing a dense enough output grid avoids nominal coordinate downsampling but
does not recover high-frequency content lost to fractional-sample interpolation.
Round-trip RMSE is a diagnostic of transformation loss, not a rigorous bound on
error relative to an unknown continuous physical signal.

## FFT reference discrepancy

`src/rotmach/spectral.py` returns rectangular-window one-sided **amplitude**,
not PSD. DC is not doubled; an even-length signal's Nyquist bin is not doubled.
For odd N the final positive-frequency bin does have a negative counterpart and
must be doubled. The MATLAB helper always doubles `2:end-1`, underestimating
that final bin by two for odd N. `faithful_reference` exposes that behavior;
`scientifically_corrected` doubles all positive bins when N is odd. Frequencies
are returned in Hz; division by 1000 is a plotting choice for kHz, not a change
to spectral sampling semantics.

## Executed synthetic checks

Executed from the repository root:

```text
PYTHONPATH=src python -m pytest tests/test_diagnostics.py -q
28 passed, 1 warning in 1.10s
```

The warning came from the environment's external anyio pytest plugin using a
deprecated pytest alias. It did not skip a test or invalidate an assertion.

| Check family | Concrete outcome established |
|---|---|
| RMS/crest/kurtosis | Exact known arrays, zero/constant handling and very large scale invariance |
| CK | Three hand-computed padding/order cases; scale invariance; zero energy; invalid lag rejection |
| ACF | `[0,1,2]` gives `[1,0,-0.5]` under the declared centered common-denominator convention |
| Hilbert envelope | Bin-aligned amplitude-modulated carrier recovers analytic envelope to 1e-12 |
| PMI indexing | P=100 selects lag 99 vs 100; historic `timeindex=101` exposed |
| PMI magnitude/sign | Known periodic envelope gives rho=.975 and PMI=39; half-period negative value retained |
| PMI stability | Zero/constant-envelope and lag-zero singular cases remain undefined |
| FFT | Odd/even unit-amplitude sinusoid, DC and even Nyquist; historic odd-bin factor of two demonstrated |
| Unit conversions | 10 rps, 600 rpm and 20π rad/s give equivalent cycle coordinates |
| Variable speed | Analytic linear-speed chirp recovers order-3 signal within the linear-interpolation curvature error bound |
| Periodic pulses | Synthetic periodic modulation gives stronger envelope ACF at the known period than half-period |
| Boundaries | Unknown units, nonpositive speed, duplicate time and matrix/NaN signals rejected |
| Multiple records | Record-specific persistence residuals reset; common burn-in and unknown-unit status preserved |

The periodic pulses are **synthetic numerical validation**, not simulated
ground-truth gearbox cracks, and do not establish fault sensitivity or AUC.

## Executed sensitivity on the supplied reference records

This section is regenerated by `python scripts/run_numerical_validation.py`.
Full precision and input hashes are in [angular_unit_sensitivity.csv](angular_unit_sensitivity.csv),
[signal_statistics.csv](signal_statistics.csv) and
[provenance JSON](angular_unit_sensitivity_provenance.json).

The validation residual uses the frozen recommended checkpoint `lpv_L256_s17_a1`.
All series use zero-based targets [256:24000] (23,744 samples).
The raw reference is the **supplied normalized x**, not xraw. Residuals
are x-yhat in the same dimensionless amplitude scale. No series or record
is re-standardized. RMS fractions refer to supplied x on the identical segment.

| Split | Series | RMS | Fraction of raw x RMS | Crest | Pearson kurtosis |
|---|---|---:|---:|---:|---:|
| train | supplied_x | 1.04277152 | 1.00000000 | 4.56872555 | 3.25184194 |
| valid | supplied_x | 1.05075083 | 1.00000000 | 4.19266380 | 3.18825424 |
| valid | residual:lpv_L256_s17_a1 | 0.60101370 | 0.57198498 | 4.01288417 | 3.14324425 |

Samples/cycle is the smallest power of two >=fs/minimum assumed cycles/second.
This prevents nominal density downsampling, but linear interpolation still filters
high-frequency content. The speed units and assumed one-cycle modulation order
are **unverified assumptions**, not confirmed physics or fault identity.

| Split | Series | Assumed unit | Samples/cycle | Angular N | Round-trip RMSE | CK (M=1) | Corrected PMI | Reference-index PMI |
|---|---|---|---:|---:|---:|---:|---:|---:|
| train | supplied_x | rpm | 2048 | 37405 | 0.27994809 | 2.8277676e-05 | 0.09511809 | 0.09114105 |
| train | supplied_x | rad_s | 256 | 44649 | 0.23325721 | 2.4764156e-05 | 0.09320605 | 0.08714675 |
| train | supplied_x | rps | 32 | 35068 | 0.29782372 | 3.1868939e-05 | 0.20157020 | 0.18853588 |
| valid | supplied_x | rpm | 2048 | 37406 | 0.28299452 | 2.7611772e-05 | 0.08461393 | 0.08380420 |
| valid | supplied_x | rad_s | 256 | 44650 | 0.23658127 | 2.4092836e-05 | 0.06839218 | 0.06957901 |
| valid | supplied_x | rps | 32 | 35068 | 0.30120877 | 3.2667006e-05 | 0.18851686 | 0.18991916 |
| valid | residual:lpv_L256_s17_a1 | rpm | 2048 | 37406 | 0.17048517 | 2.6933309e-05 | 0.04014647 | 0.04824608 |
| valid | residual:lpv_L256_s17_a1 | rad_s | 256 | 44650 | 0.14366516 | 2.3153214e-05 | 0.04695394 | 0.04723604 |
| valid | residual:lpv_L256_s17_a1 | rps | 32 | 35068 | 0.18201934 | 2.9242244e-05 | 0.04104650 | 0.04966707 |

Material variation between assumed units shows why undocumented unit selection
can change a diagnostic conclusion. Round-trip errors quantify transformation loss,
not error against an unknown continuous physical signal. CK is length-dependent:
raw/residual comparisons within one assumption have equal length, whereas comparisons
between unit assumptions have different angular N. Neither larger PMI nor kurtosis
establishes superior physical-fault detection without labels and verified shaft/order metadata.

## Unexecuted and blocked checks

- The initial monolithic fault upload was inaccessible; the subsequent twenty
  complete-record parts are verified and processed. Real-record numerical
  features and both PMI indexing variants under all three unit assumptions are
  in fault_angular_features.csv. Their mathematical validity is distinct from
  physical calibration or verified detection performance.
- Live numerical MATLAB comparison, especially envelope edge treatment and the
  custom angular-resampling dependency, remains unverified.
- Real physical-fault sensitivity, specificity, ROC/AUC, F1 and severity trends
  require validated labels and additional source information; they were not run.
