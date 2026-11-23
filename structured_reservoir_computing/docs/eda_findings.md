# Physics-informed SRC feature evidence

This EDA uses the 24,000-sample designated training record. `results/eda/summary.json` and `extension_summary.json` identify the deterministic results. Every exploratory predictor fits within the stated guarded training folds and shares the same held targets. Scope is engineering screening before final optimization: the three held training slices and closely repeated operating trajectory provide little evidence about broader operating conditions.

## RPM, finite sampling, and stationarity

The 37-tooth gear turns at 124.323–167.334 RPM. Input frequency is RPM/60, mesh frequency is 37 times the input frequency, and output shaft frequency is 37/20 times input. There is a single mesh carrier. The pinion has order 1.85 in input-turn coordinates. The training mesh spans 76.666–103.189 Hz with roughly 31.01–41.74 samples per mesh period. Nominal time duration is 7.5 s and integrated speed yields 18.430 turns. The 10-second records measure approximately 23.87 turns. This agrees with the thesis' RPM/60 axis convention.

At the fastest measured training speed, 3.2 kHz sampling implies a common temporal Nyquist order of 573.70. A first-twelve mesh bank reaches order 444, about 1,238 Hz at that speed, leaving margin under the 1,600 Hz temporal Nyquist cutoff. Harmonic 15 sits much closer to the limit and would require a larger calibration/view for order tracking. Higher angle resampling resolution cannot recover past the physical sample bandwidth.

The quarter-second training RMS–RPM correlation is 0.927. Higher speed generally accompanies higher vibration level, with overlapping rise and fall bins and small counts within some bins. This is an observational association of a shared speed/load trajectory. Speed derivative quantiles at 1, 10, 50, 90 and 99 percent are −14.26, −13.19, 2.95, 15.42 and 24.07 RPM/s. Exact extrema of vibration occur once apiece. The standardized signal excess kurtosis is 0.260 and skew −0.030. These checks provide no independent sensor precision, clipping threshold or noise-floor calibration. Use past-window RMS and speed differences as separate operating-context channels.

The figure `figures/eda/eda_speed.png` shows full training time history and past-only rolling moments. By-direction band power uses non-overlapping 800-point Hann periodograms and fixed RPM bins. `results/eda/tables/eda_band_power_by_speed_direction.csv` lists counts, means, sample standard deviations where enough blocks exist and spectral centroids. Unequal regime counts prevent a causal acceleration interpretation.

## Mesh-locked frequency, angle and fixed-Hz hypotheses

The training STFT uses a 1,024-point Hann window with 896 overlap and 4,096-point FFT. The fourth mesh harmonic, order 148, reaches median absolute ridge error 0.744 Hz to its expected path. Its constrained-track correlation is 0.987. The geometric window makes correlation conditional on the candidate line. Order 148's independent angular density peak is about 160 times the local off-order median, and order 259 also peaks above neighboring bands. Complete turn projections have a small sample of 18 turns. At orders 37, 74, 148 and 259, within-record amplitude correlation to mean RPM is −0.462, 0.800, 0.611 and 0.805. Conditional gains may differ across carriers.

In the 1.0–1.2 kHz region, Welch power is 31.3% of total measured power. Moving shaft orders can pass this frequency interval. The measurement supports testing fixed-Hz controls, with any modal identification requiring structural measurements. A fixed 1,100 Hz rectangular quadrature yields guarded MSE 0.46230 against 0.46153 for LPV-AR24. A pair at 1,080 and 1,125 Hz with finite damped states yields 0.46152, essentially equal to the control. These tested fixed-Hz settings are deprioritized.

Time PSD uses 2,048-point Hann Welch and 75% overlap. Angular spectra use strictly increasing integrated shaft phase, interior linear interpolation at 2,048 samples per revolution, and eight-turn Hann Welch with six-turn overlap. The effective alias-free angular reporting range is bounded by the highest-speed temporal Nyquist order. Hilbert-envelope order spectra are offline descriptive quantities. Synthetic order-148 signals recover 148.0040 under constant 144 RPM and 147.9764 under varying speed. Altering RPM after a cutoff leaves phase at and before that target unchanged; replacing vibration at the cutoff produces exactly zero change in the finite quadrature at and before the cutoff. Complex quadrature RMSE is near 0.0061 against the ideal analytic carrier, consistent with finite rejection of the negative-frequency component.

## Memory, nonlinear dependence and feature costs

One mesh-cycle angle autocorrelation is 0.128 and one input revolution is 0.075. Time autocorrelation at lag one is 0.076. The strongest additional partial autocorrelation within lags 2–96 is at four samples. These summaries motivate testing a grouped short lag bank alongside longer synchrony states. A single angular mesh lag has guarded MSE 0.46242 and a full-turn lag 0.46142 versus 0.46153 for the AR24 LPV control. Whole-turn memory may also predict away a fixed-angle fault impulse. Multinode coherent mesh histories are a separate candidate.

All damped-state screens use a finite 512-sample exponentially weighted demodulated state, delayed before remodulation at target phase. Regression features are standardized on permitted rows only. Both the temporal support and the 1,792-sample guards are recorded. Reported MSEs average the three equal-length held folds.

| Trial | Guarded mean MSE | Engineering decision |
|---|---:|---|
| LPV-AR24 | 0.46153 | Reference |
| Orders 148/259, decay 8 samples | 0.45228 | Compact short-memory control |
| Orders 148/259, decay 32 | 0.44560 | Compare to larger banks at fixed decay |
| Orders 148/259, decay 128 | 0.44049 | Focused bank candidate, longer memory |
| Six mesh orders 37, 74, 148, 222, 259, 444, decay 32 | 0.43537 | Broader geometry bank candidate |
| Shaft and sidebands 1, 147, 149 plus orders 148/259, decay 32 | 0.42903 | Requires explicit pulse-retention trial |
| Selected quadratic interactions on two-order bank | 0.43819 | Favor group-screened nonlinear degree two |
| Selected cubic interactions on two-order bank | 0.43803 | Incremental effect is small |
| Fixed Hz 1,080/1,125, decay 32 | 0.46152 | Deprioritize this package of fixed carriers |
| Fixed trailing RPM delay 32 samples | 0.44538 | Slight within-record change relative to synchronized .44560 |

A deliberately advanced 32-sample RPM diagnostic was also scored offline to probe misalignment. It achieved 0.44611 and is excluded from causal feature candidates. The close outcomes are unsurprising under the slow, almost overlapping RPM profiles and cannot establish acceptable timing error under rapidly changing speed. Residual lag-one autocorrelation and squared residual/backward RMS rank correlations are available for each screen, including held-fold cases with different baseline error. Correlation and PACF alone do not identify a mechanical decay constant.

The focused candidate uses 148/259 with 32–128 sample decays. The broader candidate uses measured mesh geometry plus a compact set of orders. A third candidate conditions sideband inclusion on synthetic pulse retention and eligible-record descriptive diagnostics. These are pre-specified screening alternatives. `results/eda_feature_hypotheses.csv` maps mechanisms, costs, history support, search ranges, failure modes and experiment IDs. Broad degree-three interaction expansion, unverified fixed-Hz resonance labeling, and unfiltered full-record envelope inputs are rejected as defaults.
