# 8. Matched prediction results and computational efficiency

## 8.1. Primary score and valid comparisons

The score covers the 22,208 targets with zero-based indices 1,792 to 23,999 of the official development recording. All 27 ranked rows use the same vibration values, RPM and time arrays and target order. The excluded prefix supplies model history. It contributes no squared errors to the score. PI-SRC forecasts one sample ahead from measured history through $t-1$. That task tests prediction under continued access to measurements and differs from free-running trajectory simulation.

For residual $e_t=x_t-\widehat{x}_t$ and retained indices $\mathcal{I}$, the evaluation uses

$$
\operatorname{MSE}=\frac{1}{|\mathcal{I}|}\sum_{t\in\mathcal{I}}e_t^2,\qquad
\operatorname{RMSE}=\sqrt{\operatorname{MSE}},\qquad
\operatorname{MAE}=\frac{1}{|\mathcal{I}|}\sum_{t\in\mathcal{I}}|e_t|.
$$

MSE is in squared standardized vibration units. RMSE, MAE and residual bias are in standardized vibration units. The selected PI-SRC achieved MSE 0.329973, RMSE 0.574433, MAE 0.456412 and mean residual +0.007643. Its resubstitution MSE on training was 0.308688. The difference of 0.021286 describes these two supplied segments and their evaluation scopes. The clean mean over the three guarded training folds was 0.355734. Guarded fold fits use less training data than the production fit and have distinct speed distributions, so the lower development value does not establish superior performance beyond the tested operating region.

Define the signed contrast and relative improvement against comparator $c$ by

$$
\Delta_c=\operatorname{MSE}_c-\operatorname{MSE}_{\mathrm{PI\text{-}SRC}},\qquad
I_c=100\,\frac{\Delta_c}{\operatorname{MSE}_c}.
$$

Under this sign convention, positive values favour PI-SRC. Table 8.1 gives primary contrasts and retains the historical anchor because it changes the scientific conclusion. Linear Regression was the strongest of the 21 required baselines on this segment. The strict sparse LPV–ARMA fit was the prescribed speed-varying statistical comparator and CNN was the strongest completed neural comparator. PI-SRC improved on these three comparators by 10.18%, 11.16% and 16.39%, respectively. The archived corrected SRC had lower MSE, RMSE and MAE than PI-SRC. Its MSE advantage of 0.012186 means that the primary score was 3.83% higher relative to the archived denominator.

Table 8.1. Representative registered results on the common development mask. The signed difference is comparator minus selected PI-SRC.

| Predictor | MSE | RMSE | MAE | Difference | Improvement (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Archived corrected SRC | 0.317788 | 0.563727 | 0.448410 | −0.012186 | −3.83 |
| Selected PI-SRC | 0.329973 | 0.574433 | 0.456412 | Reference | Reference |
| Linear Regression | 0.367386 | 0.606123 | 0.482352 | 0.037412 | 10.18 |
| Ridge | 0.368059 | 0.606678 | 0.482731 | 0.038085 | 10.35 |
| Strict sparse LPV–ARMA | 0.371406 | 0.609430 | 0.483517 | 0.041432 | 11.16 |
| CNN | 0.394638 | 0.628202 | 0.497356 | 0.064665 | 16.39 |

FIGURE_8_1

The complete appendix ranking accounts for every required comparator and distinguishes forecast-only historical rows from executable checkpoint replays. The separate fresh rerun reproduced the 21 protected comparator forecasts and saved checkpoints exactly in the recorded environment. For the two sparse families that rerun refitted the previously selected support, so it verifies the numerical procedure with that support held fixed. The archived SRC has a different selection history and serves as a performance anchor on the shared targets. Its replay supports the narrower finding that it scored better here.

The published sparse LPV–ARMA test MSE of 0.2893 belongs to the authors' test segment [@chen2025sparse]. The local archived implementation using current RPM and an earlier scoring start gave 0.336901 on 23,684 points. Rescoring that row on the common mask gave 0.345151. The strict previous-RPM sparse result of 0.371406 also uses a different fitted reference configuration. These three values cannot isolate the effect of a one-sample scheduling shift. A separate dense local LPV–ARMA diagnostic fit using previous RPM scored 0.330781 on the common mask. Its difference from selected PI-SRC is only 0.000808. This supplementary comparison puts the much larger sparse-reference improvement in context.

## 8.2. Paired uncertainty and residual structure

Let $d_t=e_{c,t}^{2}-e_{\mathrm{PI\text{-}SRC},t}^{2}$. The reported uncertainty preserves the pairing in $d_t$ so both candidates retain each target's operating conditions. A non-circular moving-block bootstrap samples valid starting locations for contiguous blocks of $d_t$, with a shorter terminal block to reproduce the original count. Each setting uses 2,000 draws, and the displayed interval takes the 2.5th and 97.5th percentiles [@compKunsch1989]. Block lengths 320, 640 and 1,280 samples span 0.10, 0.20 and 0.40 s. The directions in Table 8.2 were unchanged across all three block lengths. The intervals reflect within-record loss variation under these resampling conventions. The intervals are pointwise. Changing block length addresses some short-range dependence. Generalization across rigs, speed profiles or records requires repeated physical acquisitions.

Table 8.2. Paired moving-block intervals at 640 samples per block. All entries are in squared standardized vibration units.

| Comparator | Mean paired difference | Lower 95% | Upper 95% |
| --- | ---: | ---: | ---: |
| Linear Regression | 0.037412 | 0.032403 | 0.041436 |
| Ridge | 0.038085 | 0.033149 | 0.042579 |
| Strict sparse LPV–ARMA | 0.041432 | 0.036319 | 0.047242 |
| CNN | 0.064665 | 0.056022 | 0.073982 |
| Archived corrected SRC | −0.012186 | −0.014521 | −0.010361 |

FIGURE_8_2

The waveform display uses the same 320 consecutive targets for each model, array positions 6,300 through 6,619 of the cropped development bundle. These are native indices 8,092 through 8,411, approximately 2.529–2.629 s on the saved time axis. This fixed display rule makes it a local illustration of amplitude tracking and residual structure. Evidence for recovery of an impulse requires the paired injection in Chapter 9, where its location and waveform are known. The diagnosis of individual transient events in this measured trace would require ground-truth condition or event labels.

FIGURE_8_3

The primary residual autocorrelation is 0.000945 at lag one, yet its largest magnitude from lags 2 to 128 is 0.0524 at lag three. The Ljung–Box portmanteau diagnostic at 64 lags yields $p=3.39\times10^{-92}$ using its nominal chi-square reference [@compLjung1978]. Its signal here is that residual dependence extends beyond the first lag. The result is descriptive for this single changing-speed segment. The two-sided runs diagnostic, which counts transitions in the nonzero residual signs, gave $p=0.2358$. This is compatible with small sign imbalance or serial dependence at particular lags because the two diagnostics address different properties. The Welch estimate uses 1,024-sample segments at 3,200 Hz, a 3.125 Hz Fourier grid [@welch1967fft]. The surviving structure motivates inspecting the order-related envelope, rather than treating forecast loss or near-zero lag-one correlation as evidence of diagnostic sufficiency.

## 8.3. Operating-regime error

Measured RPM quartiles contain 5,552 retained targets each. The boundaries span approximately 124.36–167.32 RPM. Table 8.3 shows a rise from 0.259514 in the first quartile to 0.371653 in the fourth for selected PI-SRC. Its error is below strict sparse LPV–ARMA and above the archive in every quartile. Thus the two main contrasts do not rely on one RPM quartile. The full strata table also divides the retained record into six consecutive time blocks. Primary MSE ranges from 0.254959 in the first block to 0.378389 in the third. The primary and both reference methods consistently retain their ordering in those blocks.

Table 8.3. Development MSE by measured RPM quartile.

| RPM quartile | Targets | Selected PI-SRC | Strict sparse LPV–ARMA | Archived corrected SRC |
| --- | ---: | ---: | ---: | ---: |
| 1 | 5,552 | 0.259514 | 0.286705 | 0.248805 |
| 2 | 5,552 | 0.324015 | 0.368801 | 0.309555 |
| 3 | 5,552 | 0.364711 | 0.411718 | 0.352031 |
| 4 | 5,552 | 0.371653 | 0.418398 | 0.360760 |

FIGURE_8_4

The retrospective 128-sample vibration-energy strata show primary MSE from 0.276564 to 0.369472 between the lowest and highest quartiles. Rising RPM is defined by a positive difference across 64 samples. PI-SRC MSE was 0.322608 on 13,022 rising targets and 0.340879 on 9,122 nonrising targets. The first 64 cropped targets are omitted from this difference-based comparison. Speed, speed gradient, vibration energy and time are associated along the measured sweep. Estimating an independent effect of one operating variable would require replication across controlled conditions. Earlier-only training refits scored 1.413805, 1.110925 and 0.353668 on the three held regions, showing sensitivity to the length and operating coverage of the available past. Those refits had no role in model selection.

## 8.4. Runtime and efficiency

The timing host was an AMD EPYC 9V74 under Linux with one numerical thread and CPU-only Torch. The runtime manifest records Python 3.12.14, NumPy 2.5.5, SciPy 1.17.0 and Torch 2.10.0+cpu. The warm measurement times one prediction call on the whole 24,000-sample record, using arrays and a checkpoint already in memory. Feature production is part of the call. Checkpoint deserialization, acquisition and the separate angular diagnostic stage are outside it. Primary warm timing uses seven calls, with median 484.887 ms, IQR 31.806 ms and estimated 95th percentile 524.715 ms. The separate checkpoint load and first prediction took 2.302 and 523.467 ms, respectively. A cold-start distribution would require repeated loading and first-call measurements. The comparator and executable archive warm timings use seven saved repetitions per row. Primary fitting time covers its single final fit, while comparator fit times are the available recorded fit times for their respective training pipelines. Neither represents the full exploration cost.

Table 8.4. Measured CPU costs and stored model sizes. Missing historical values were not imputed.

| Predictor | Recorded fit (s) | Batch median (ms) | IQR (ms) | Parameters or coefficients | Checkpoint bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| AR | 0.247 | 0.333 | 0.018 | 96 | 2,855 |
| MLP | 0.849 | 19.862 | 17.283 | 7,050 | 29,004 |
| Strict sparse LPV–ARMA | 5.188 | 87.960 | 20.369 | 840 | 8,908 |
| Archived corrected SRC | — | 188.793 | 8.279 | 2,789 | — |
| Selected PI-SRC | 4.265 | 484.887 | 31.806 | 1,456 | 32,494 |
| Linear Regression | 3.142 | 951.884 | 58.010 | 530 | 12,738 |

FIGURE_8_5

The primary has 879 retained readout feature columns and 1,456 fitted coefficients, including the readout intercept and first-stage coefficients. The latter count differs in meaning from, for example, an attention network's trainable tensor count. Its compressed checkpoint is 32,494 bytes. A separate five-repeat timing probe measured the principal feature arrays at 247,939,328 bytes for explicit construction and 43,237,376 bytes for bounded batches of 4,096 rows. Median times were 350.768 and 359.231 ms. The maximum prediction difference was $8.79\times10^{-9}$. These values describe array storage and a bounded feature-evaluation probe. Full-runtime peak memory and warm inference are distinct quantities. The selected prediction verification recorded approximately 120.43 MiB of process RSS after the timed work, which is a different quantity again.

The primary processed a 7.5 s record at an amortized 49,496 input samples/s or a real-time factor of 0.0647. Worst-case 312.5 µs per-sample deadlines, buffering latency and device-side cost require streaming measurements. GPU and independent embedded timings were unavailable. The accuracy–time frontier illustrates different choices. Among the primary and 21 comparator fits, the primary gives the lowest MSE at the highest frontier latency. Strict sparse LPV–ARMA runs at approximately 18.1% of the primary's median latency with 0.041432 greater MSE. Including the executable archive yields a different frontier because archived corrected SRC has both lower MSE and lower warm median latency. The frontier uses executable replays with measured latency. Forecast-only archives lack such a measurement.

# 10. Ablation, robustness and limits of attribution

## 10.1. Controlled representation and adapter effects

The development ablation ladder provides evidence at several levels. SRC-Linear, SRC-Polynomial and SRC-Speed scored 0.447954, 0.447872 and 0.411939, respectively. The tiny reduction from adding the tested generic polynomial expansions to a time-only model is weak evidence of standalone benefit. The gain for SRC-Speed is consistent with operating-condition dependence. The separate fixed-architecture comparator parity test withheld measured speed by fitting with nominal 145 RPM. Seven of nine measured-RPM variants improved on their controls, including NARX by 0.068379 and CNN by 0.049183 in MSE. ARX and TFT were worse by 0.002418 and 0.013780. Useful speed conditioning therefore depends on model and feature construction.

The physics-state linear readout reached 0.351040, versus 0.412117 for the tested generic nonlinear SRC. PI-SRC without the adapter scored 0.340706. These ladder rows jointly change representation, budget and fitting conditions. More targeted refits show the strongest removal penalty for angular lags, with MSE rising to 0.359049. Removing both angle lags and carriers raised it to 0.377201. Removing innovations gave 0.347312, removing the carrier bank gave 0.334961, and using fixed-Hz carrier states in place of the selected order bank gave 0.334862. Such differences are conditional on the retained features, training sample and refitted representation. The removal rows change the number of fitted columns. A strictly dimension-matched replacement by time lags would help distinguish angular alignment from dimensionality and changed collinearity.

Table 10.1. Selected development ablations on an unchanged mask. Differences are signed against selected PI-SRC.

| Control | MSE | Change | Fitted coefficients |
| --- | ---: | ---: | ---: |
| Selected PI-SRC | 0.329973 | — | 1,456 |
| Clean-MSE training-only selection | 0.326401 | −0.003573 | 1,456 |
| Compact configuration | 0.358022 | +0.028048 | 834 |
| Same-weight adapter disabled | 0.340706 | +0.010733 | 1,456 |
| No angular lags | 0.359049 | +0.029076 | 1,072 |
| No carrier bank | 0.334961 | +0.004987 | 1,363 |
| No angular lags or carrier bank | 0.377201 | +0.047228 | 979 |
| No innovation history | 0.347312 | +0.017339 | 805 |
| Fixed-Hz carriers only | 0.334862 | +0.004888 | 1,414 |
| Post hoc 12-order bank | 0.324770 | −0.005204 | 1,600 |

FIGURE_10_1

The same-weight correction is the cleanest single intervention. With gain 0.5, time constant 512 samples, memory 768 samples and clip 1.75 in the saved residual units, the selected fit reduces MSE by 0.010733 from the uncorrected score, a 3.15% decrease relative to that score. The identical correction lowers strict sparse LPV–ARMA from 0.371406 to 0.367992 and the physics-free Polynomial SRC from 0.447872 to 0.444834. Uncorrected PI-SRC remains 0.027286 below the corrected strict sparse comparator. Thus physical and other fitting choices account for part of the overall contrast, and a correction effect exists even without the physical bank. This estimate applies to the saved architecture. Other architectures and joint interventions can yield different marginal effects.

The clean-MSE training-only selection takes a smaller ridge value and a longer adapter kernel. It slightly won the clean guarded objective (0.355511 versus 0.355734) and went on to score 0.326401 on development. The registered multi-objective candidate had a better composite training score (0.371689 versus 0.372125) and smaller guarded pulse distortion (1.050501 versus 1.132523). The development ranking consequently depends on the objective. The 12-order configuration, together with variants removing augmentation or acceleration, is reported as a retrospective sensitivity. The 12-order row is 0.005204 lower than the primary on this development segment. The no-augmentation and no-acceleration rows give 0.327177 and 0.328697. The quadratic-only control gives 0.330390, and the tested lower and higher nonlinear budgets have similarly small changes. Those observations discourage attributing the whole gain to cubic complexity or to every selected intervention. The compact configuration uses 42.7% fewer fitted coefficients and a three-call warm median of 282.268 ms. Its MSE increases by 0.028048. Those measurements illustrate an engineering trade-off rather than a precise runtime comparison with the seven-call primary.

## 10.2. Sensor stress and residual interpretation

Development stress changes RPM and vibration inputs on an otherwise fixed record and leaves model coefficients frozen. The predictor-RPM lane feeds altered speed to the predictor while retaining measured RPM for angular diagnostics. The diagnostics-only lane changes the angle map for the unchanged residual. The combined lane alters both. Training uses its own guarded stress and pulse designs. The specific development scenarios and their magnitudes characterize frozen-model sensitivity rather than adding observations of physical sensor faults.

Table 10.2. Primary development MSE for the predictor-input intervention lane. Vibration noise is separate from the eight RPM cases.

| Input intervention | MSE |
| --- | ---: |
| Clean measured input | 0.329973 |
| −2% / +2% RPM bias | 0.439358 / 0.457031 |
| Correlated 0.6% / independent 0.3% relative RPM jitter | 0.370028 / 0.339827 |
| RPM delayed 40 ms | 0.332180 |
| RPM quantized to 1 RPM | 0.333385 |
| 75 ms hold-last-value intervals | 0.329794 |
| Smooth profile warp | 0.360840 |
| Vibration noise SD 0.05 | 0.336254 |

FIGURE_10_2

The +2% bias raised primary MSE by 38.51%. Under the same eight-case suite, worst-case MSE was 0.454057 for archived corrected SRC, 0.433301 for Ridge, 0.412020 for CNN and 6.554814 for strict sparse LPV–ARMA. The sparse model showed pronounced degradation under this tested bias. The clean and stressed rankings differ under these interventions. The isolated hold-last-value outcome below the primary clean score is a result on one synthetic schedule. Generalizing this outcome would require independent dropout schedules and acquisitions. RPM quantization here is a rounding intervention on the measured-speed channel. The vibration grid remained at 3.2 kHz and the saved model weights were unchanged.

The +2% predictor-only case raised PMI by 0.098194, versus 0.004508 when the perturbation affected only angle tracking and 0.044913 when it affected both. The median PMI rise for the five periodic primary pulse injections was 0.000422. The speed intervention can create periodic residual structure far larger than this particular synthetic tooth-related perturbation. All five periodic cases had a positive increase and neither aperiodic control did. Median residual–pulse correlation across the five periodic cases was 0.690. An operating threshold and field false-alarm rate require controls with diagnostic ground truth. Similar conditional AUCs under the provisional candidate grouping leave this specificity question open.

The main limitation is the number and diversity of acquisitions rather than the number of time samples. The guard protects against overlapping local histories across fitting and validation within a single training record. Phase, speed and load transfer require tests beyond this shared run environment. The deterministic refit reproduced the shipped primary checkpoint and predictions on the recorded platform. A four-family comparator seed study measures algorithmic variability under repeated fitting, with CNN averaging 0.397804 and sample SD 0.003005 over three runs. The between-machine error distribution remains unknown. Further independent evaluation would need frozen acquisition-level testing, calibrated speed timing and known condition labels, with streaming deadlines and sensor perturbations measured under deployment conditions.
