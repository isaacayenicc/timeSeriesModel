---
lang: en-CA
---

# COURSE-TITLE-MARKER

MECG 7780 / ENGINEERING DATA ANALYTICS

Physics-Informed Structured Reservoir Computing for Variable-Speed Gearbox Vibration

Causal one-step prediction, matched benchmarks and residual diagnostics

Final course project research report

WORD_PAGE_BREAK

# Abstract

A change in shaft speed moves gear-mesh energy across frequency and alters the vibration history that is informative for prediction. This study evaluated Physics-Informed Structured Reservoir Computing (PI-SRC), a finite representation of earlier vibration and rotational speed with a regularized readout. The feature map combines time and angular delays, damped shaft-order quadratures, speed interactions and lagged proxy innovations. A bounded correction uses previous base residuals. Forty-one configurations were compared in three guarded blocks of the designated training recording. The chosen configuration was frozen before development scoring.

The selected PI-SRC obtained training and development mean squared errors of **0.308688** and **0.329973**, in squared standardized vibration units. On the 22,208 matched development targets, error was 11.16% below the strict local sparse linear parameter-varying autoregressive moving-average benchmark and 10.18% below Linear Regression, the strongest of 21 requested comparators. The replayable corrected SRC archive remained ahead at 0.317788. Median one-thread CPU time for PI-SRC was 484.89 ms per 7.5 s record. Matched ablations associated the largest tested structural loss with removing angular delays and showed a 0.010733 MSE reduction from the correction with weights fixed.

After excluding two overlapping records, residual periodic modulation intensity separated provisional groups across 198 candidates. PI-SRC’s median group gap was 0.08473. A controlled +2% speed bias in prediction alone increased the indicator by 0.09819, exceeding the median increment from the five periodic synthetic injections. These results support the physical feature design for causal prediction and constrain the diagnostic interpretation. Independent test inputs and verified condition labels were unavailable.

**Keywords** Structured reservoir computing, gear-mesh order, one-step prediction, sparse LPV–ARMA, periodic modulation intensity

WORD_PAGE_BREAK

# Contents

WORD_TOC_MARKER

## Abbreviations and notation

Table P.1. Terms and symbols used in the report.

| Abbreviation | Meaning |
| --- | --- |
| SRC, PI-SRC | Structured Reservoir Computing and its physics-informed variant |
| AR, MA, ARX | Autoregression, moving average and autoregression with exogenous input |
| LPV | Linear parameter-varying |
| ACF, PSD, STFT | Autocorrelation function, power spectral density and short-time Fourier transform |
| MSE, RMSE, MAE | Mean squared, root mean squared and mean absolute error |
| PMI | Periodic modulation intensity |
| RPM, $f_s$ | Revolutions per minute and sample rate in hertz |
| $x_t$, $n_t$ | Standardized vibration and measured input-shaft RPM at sample $t$ |
| $u_t$, $Z$ | Cumulative shaft revolutions and the number of teeth on the monitored gear |
| $q$ | A harmonic order expressed in cycles per input-shaft revolution |

# 1. Introduction and research objectives

Gear-mesh contacts excite a train of forces whose spacing in time changes with shaft speed. The same tooth mesh can therefore produce a moving frequency ridge even when its order relative to the input shaft stays fixed. The vibration measured at a bearing housing also reflects the transmission path, loading, shafts and noise. A tooth crack can change mesh stiffness periodically and alter the vibration envelope around a characteristic shaft angle. A prediction model that explains the regular component can make such modulation more visible in its residual, while any estimator that anticipates part of a periodic disturbance can also suppress the same structure. These mechanisms motivate evaluating short-horizon prediction and residual diagnostics as separate outcomes. [@mcfadden1986gears; @chen2021crack; @chen2025sparse]

The course asks for estimation of the next vibration sample from measured history, with rotational speed available as an additional channel. It evaluates mean squared error, computational efficiency and fault-detection performance. It also identifies a sparse linear parameter-varying autoregressive moving-average model, angular resampling and periodic modulation intensity as reference methods. [@mecgbrief] These objectives are addressed here as an engineering question about how to represent prior observations under a varying speed profile. The score uses the next sample of the supplied processed stream, and the deployment interface receives past measurements through the preceding sample. Multi-step free-running behaviour requires an independent experiment.

The candidate architecture places a physical transformation before a fitted statistical readout. Time delays preserve broadband short-memory dependence. Physical angle aligns events over distinct shaft speeds. Finite damped quadratures represent coherent orders, and a scheduled feature map can vary their utility with operating regime. The combination is investigated with both simple controls and a broad measured comparison. The novelty claim concerns the engineering design and its controlled measurement on the course data. Foundational nonlinear vector autoregression, speed-scheduled linear models, angular diagnostics and general physics-informed reservoir methods supply its intellectual antecedents. [@gauthier2021; @adler2026; @chen2025sparse]

Table 1.1. Testable questions and decision evidence.

| Question | Measured comparison | Scope of interpretation |
| --- | --- | --- |
| H1. Does the selected representation reduce causal prediction error? | Matched MSE against strict sparse LPV–ARMA, conventional and neural models, and an executable SRC archive | One official development record and identical processed target samples |
| H2. Which physical mechanisms contribute after fitting and correction are controlled? | Remove angular, oscillator or innovation features, compare generic and fixed controls, toggle the same correction with weights fixed | Single and bundled interventions have different causal interpretations |
| H3. Does residual prediction preserve periodic disturbance evidence? | Provisional PMI group separation, paired injected-pulse experiments, and isolated RPM measurement stress | Index groups are provisional. Synthetic injections assess defined waveform recovery |
| H4. What computational and robustness costs accompany the gain? | One-thread CPU repeats, coefficient and memory measures, lag support and measured RPM perturbations | Batch hardware results and stress scenarios on the same development record |

The practical contribution is a reproducible comparison with explicit failure modes. The analysis retains the corrected archive, the clean-error finalist and favourable post-selection ablations as distinct cases. It includes the complete comparator registry, validated input availability and a report-level evidence ledger. These records allow the reader to distinguish a design or modelling hypothesis from the observational evidence available to evaluate it.
