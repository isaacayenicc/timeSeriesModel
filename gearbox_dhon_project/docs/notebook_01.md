## Which clock best describes the waveform?

For the 37-tooth input gear and 20-tooth output pinion, the mesh events per input turn equal 37. The output speed is $37n/20$ rpm, so one pinion turn corresponds to order 1.85 in the input-shaft coordinates. At a measured input speed $n$, $f_m=37n/60$ Hz. A defect on the input gear has an expected modulation repetition of one input turn. The bearing housing sensor measures superposed gear excitation, transmission and structural response. Rotational speed and load varied together, so the design cannot isolate their amplitude effects.

The training speed range 124.323–167.334 rpm implies mesh frequency 76.666–103.189 Hz, roughly 42–31 samples per mesh period, and roughly 1544–1147 samples per input turn. At 3,200 Hz, Nyquist is 1,600 Hz. At highest observed speed the maximum observable order is 573.7 and the 15th mesh harmonic sits just under that ceiling. Model candidates should leave a margin for RPM bias and avoid near-Nyquist extrapolation. A 2048-point angular grid is an interpolation choice and cannot recover lost bandwidth. Left-end integration gives $u[t]=u[t-1]+n[t-1]/(60 f_s)$, with a record-local phase origin.

### The frequency-domain check

Welch estimates average spectral density in standardized vibration squared per Hz. The training raw record uses a Hann window of 2048 time samples and 75% overlap. STFT uses 1024 samples per frame, 896 overlap and zero-pads to 4096, which refines bin sampling while the true resolution is governed by the window. For each expected harmonic, local ridge search maximizes density inside a constrained neighborhood and uses adjacent log-density values for sub-bin refinement. Restricting the search by theory inflates its correlation with the proposed line, so ridge fit alone is insufficient.

The order calculation linearly resamples the vibration at uniformly spaced relative input turns. Angular Welch uses 8 turns per Hann window and 6 turns of overlap, giving 0.125-order bins. Order 148 has a training ridge median absolute discrepancy of 0.744 Hz and a roughly 160× angular PSD peak versus the local median. Order 259 also concentrates energy above its neighbors, while per-turn order amplitudes correlate differently with RPM across 18 complete turns. These data support testing distinct speed-dependent gains. A moving-block bootstrap over three-turn blocks describes uncertainty conditional on the one training record.

The 1.0–1.2 kHz band contains 31.3% of Welch power. The band combines changing-speed gear harmonics with any structural or broadband contribution. A constant 1,100 Hz oscillator is a physical hypothesis to test. Calling the observed region a modal frequency would require independent evidence such as a forced response measurement.

### The angular modulation and memory checks

Carrier sidebands at $37kpm1$ and $37kpm37/20$ can arise from modulation of the input or output shaft, respectively. Healthy rotating systems can generate sidebands through alignment and load variation. The offline envelope is $|x_a+imathcal H(x_a)|$ and exposes slower modulation in angle. The Hilbert transform and time and angle spectra use whole-record information for descriptive interpretation. The online model uses strictly past history.

A 0.10-second trailing RMS visualizes the evolving variance. Non-overlapping 0.25-second blocks give a training RPM/RMS correlation of 0.927. Inspect rise versus fall within common speed bins and their unequal block counts. ACF in time samples compares low- and high-RPM memory; ACF on the angle grid tests one mesh-period and full-turn recurrence. The mean amplitude response and short-lag dependence motivate RPM interactions and adaptive harmonic states. Neither ACF decay nor learned exponential memory estimates a mechanical damping coefficient directly.

## Guarded incremental tests

The fixed LPV-AR24 control uses 24 earlier vibration lags plus their products with affine-scaled speed and its square. Each feature family is added individually. Features fit using only the admissible training rows, fold scaling and fixed ridge penalty 10. The three blocks score the same target rows for every family, leaving a 1792-sample guard around each holdout. Angular delays are interpolated at a prior mesh or shaft revolution using only past samples. Order quadratures demodulate the last 128 observations at orders 148 and 259 and reconstruct the quadrature at the target angle.

| Guarded model | Mean MSE |
|---|---:|
| LPV-AR24 | 0.46153 |
| + order 148 and 259 quadratures | 0.43803 |
| + trailing speed derivative | 0.45930 |
| + shaft-angle delay | 0.46142 |
| + mesh-angle delay | 0.46242 |
| + fixed 1100 Hz quadrature | 0.46230 |

Order-locked quadratures improve this limited training screen. The fixed-Hz and simple angular-delay variants have little incremental effect here. Richer grouped features may behave differently and must be evaluated with matched controls. Shaft-order recurrence can also model away crack-periodic energy, which motivates a fault-pulse retention check. The registry links each causal construction, effective lookback, hypothesized benefit, and test range to the subsequent DHON search. [EDA archive audit](../docs/prior_work_eda.md)
