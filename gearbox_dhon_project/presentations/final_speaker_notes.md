# MECG 7780 Final presentation speaker notes
**Planned talk length** 12:00. Backup slides follow the timed talk.

| Slide | Planned window | Duration |
|---|---:|---:|
| 1. Predicting variable-speed gearbox vibration | 0:00–1:00 | 1:00 |
| 2. Audit the split before fitting | 1:00–1:45 | 0:45 |
| 3. Shaft angle gives an interpretable frequency prior | 1:45–2:30 | 0:45 |
| 4. A matched benchmark for sparse LPV–ARMA | 2:30–3:30 | 1:00 |
| 5. The prediction path is causal by construction | 3:30–4:30 | 1:00 |
| 6. Learn order and memory inside the state equation | 4:30–5:30 | 1:00 |
| 7. Search jointly before opening development results | 5:30–6:15 | 0:45 |
| 8. Read the selected configuration with its controls | 6:15–7:00 | 0:45 |
| 9. Development prediction on common targets | 7:00–8:30 | 1:30 |
| 10. Residual PMI measures periodicity by shaft angle | 8:30–9:15 | 0:45 |
| 11. Challenge the predictor and the diagnostic | 9:15–10:00 | 0:45 |
| 12. Account for correction and computation | 10:00–11:15 | 1:15 |
| 13. What this experiment establishes | 11:15–12:00 | 0:45 |
| 14. How an LPV filter differs from ordinary AR | Backup | — |
| 15. How phase and causality meet | Backup | — |
| 16. Why mean squared error and PMI can disagree | Backup | — |
| 17. Order spectra offer a second view of mesh energy | Backup | — |
| 18. How the guarded folds keep raw histories apart | Backup | — |
| 19. PMI reference lag and parity status | Backup | — |
| 20. Inspect a representative prediction trace | Backup | — |
| 21. Inspect the oscillator learning trajectory | Backup | — |
| 22. What the efficiency test measures | Backup | — |

## 1. Predicting variable-speed gearbox vibration

**0:00–1:00**

This project asks whether an oscillator whose phase follows shaft motion and whose order and memory can be trained supplies a practical one-step predictor for gearbox vibration. The assignment compares accuracy, computational cost and diagnostics. First, I will trace which measurements motivate the oscillator prior. Next, I will explain how the recurrence and optional bias adapter use only past observations. Finally, I will compare actual development prediction, efficiency and residual periodicity against strong reference models. The two kinds of performance answer different engineering questions, and the available labels and held-out data determine how far the conclusions apply.


## 2. Audit the split before fitting

**1:00–1:45**

The designated healthy training and development records each contain twenty-four thousand samples at 3,200 Hz. The development fault collection contains 200 recordings. Records 61 and 63 share exact training and validation prefixes respectively, so both are excluded from the eligible group comparison. The group labels denote index ranges until acquisition metadata establishes the meaning of each record.


## 3. Shaft angle gives an interpretable frequency prior

**1:45–2:30**

At the tracked shaft, a 37-tooth gear suggests first mesh order thirty-seven. RPM scales the expected time-domain frequency, while angular resampling expresses frequency as events per revolution. This makes exact mesh carriers a natural starting point. The observed spectrum also motivates a separate fixed-frequency sensitivity test. The model uses past RPM and relative phase to implement its order features. Whole-record descriptive time-frequency analysis stays in the exploratory workflow.


## 4. A matched benchmark for sparse LPV–ARMA

**2:30–3:30**

The conventional model is a speed-dependent autoregression with past proxy innovations. It schedules coefficients with RPM and restricts the active support. This implementation distinguishes the direct paper convention from strict prediction with speed through t minus one. Development scoring uses the same twenty-two thousand two hundred eight targets for the model comparisons. Chen and coauthors reported a test MSE of 0.2893 on their own validation segment. The supplied segment's different speed phase prevents a paired comparison with that external value. The independent rerun on supplied data is the benchmark for this project.


## 5. The prediction path is causal by construction

**3:30–4:30**

At each target position, previously observed RPM determines the phase increment. The encoder creates quadratures from finite histories, including delays expressed as shaft angle where the configured model uses them. The readout combines a speed-conditioned linear estimate with a bounded nonlinear component. Phase information can be carried by interpretable oscillator channels while amplitude can adapt with operating context. The optional residual adapter uses previous base-model errors and clipping. Even if that adapter improves MSE, its contribution is tested separately from learning the oscillator parameters.


## 6. Learn order and memory inside the state equation

**4:30–5:30**

The weighted oscillator sum brings each past sample into a shared rotating frame at the target. Higher damping shortens its statistical memory. Orders are bounded around physical initial values, and decay receives a positive parameterization. History is finite and state resets per recording, so the design has a calculable history requirement. The linear and nonlinear mixture can then learn from these states. Implementation checks compare direct summation with the optimized calculation and evaluate derivatives with respect to the oscillator parameters. These checks separate a mathematically differentiable design from evidence that the implemented optimizer actually updates oscillator states.


## 7. Search jointly before opening development results

**5:30–6:15**

A guarded training split assigns contiguous held-out blocks with protection on either side. This prevents lagged inputs in fitting rows from touching a held-out target. The finite search varies feature definitions, oscillator groups and learning controls under a recorded budget. The fold objective combines clean MSE, RPM stress MSE and synthetic pulse distortion. Selection uses training-fold scores only. The evaluation reports the resulting choices on the official development record.


## 8. Read the selected configuration with its controls

**6:15–7:00**

21 ranked configurations across 63 completed fold fits. Primary trial expanded short tau. Mean fold MSE 0.352. 12 mesh orders span 37.01–444.02 cycles/rev. Largest detuning 0.044 order. Decay spans 7.8–133.0 samples. The comparison isolates learning of the state parameters from adding capacity to the readout. An improvement on a small set of contiguous folds should be treated with care. Check the checkpoint and the trial ledger for exact settings and whether the individual changes are distinguishable at this sample size.


## 9. Development prediction on common targets

**7:00–8:30**

Selected DHON is 0.55% higher in MSE than the strongest archive checkpoint replay. It is 14.21% lower than the Spa LPV–ARMA benchmark. Chen et al. reported 0.2893 on the paper's external segment. The original archive prediction artifact re-scores at 0.316445. This row has no checkpoint timing. The second column is the mean squared one-step error on standardized vibration, using exactly the same development targets. The timing column, where recorded, uses complete prediction from input vibration and RPM on the same CPU benchmark. The separate published value is context for the original experiment. The common mask supports a paired comparison within this recording. A new acquisition would provide independent evidence. Consult the paired time-block results to check sensitivity to temporal dependence.


## 10. Residual PMI measures periodicity by shaft angle

**8:30–9:15**

Conditional B-high AUC 1.000 for each residual pipeline and 0.364 for raw vibration. DHON median gap B−A 0.107. The paired primary minus archive median-gap interval was -0.0019 to +0.0040. The diagnostic takes the residual after the same warm-up, resamples it to a uniform shaft angle, takes an analytic envelope and estimates its centered autocorrelation near one shaft turn. PMI increases with that lag correlation. We report the instructor-literal index and audit the mathematical one-turn indexing separately. Conditional AUC is the probability that a randomly chosen group B record ranks above a group A record, counting ties as half. Both groups have provisional labels.


## 11. Challenge the predictor and the diagnostic

**9:15–10:00**

For selected DHON, clean MSE 0.319 rose to 0.473 in the worst RPM condition. Median correlation with periodic injection 0.54. Residual PMI rose in 13 of 15 paired trials. Altering the speed input can change the predictor's phase estimate. Altering diagnostic speed can affect angular resampling even with the same residual. Keep these two errors distinct. Paired synthetic pulses let us measure residual retention under controlled injections and carry no acquisition label. Explain any MSE changes alongside pulse retention before selecting a practical model.


## 12. Account for correction and computation

**10:00–11:15**

Joint learning reduced MSE by 0.000148 versus matched fixed. Adding the adapter reduced MSE by 0.014029. DHON CPU median 0.278 s per 7.5 s record (IQR 0.008 s). CUDA unavailable. Paired 640-sample block bootstrap gives DHON − Spa LPV–ARMA MSE 95% interval -0.0593 to -0.0470. Inspect the fixed and learned versions under identical adapter settings. Then compare each version with its adapter disabled. A slow clipped adapter can reduce bias without changing oscillator parameters, but may also change a periodic pulse signature. The timing value includes its model's feature construction and inference under a batch protocol where measured. An individual streamed sample may have different latency, so batch throughput should be reported with that boundary. Multiple timing runs give variation rather than implying a fixed execution time.


## 13. What this experiment establishes

**11:15–12:00**

Selected DHON is 0.55% higher in MSE than the strongest archive checkpoint replay. It is 14.21% lower than the Spa LPV–ARMA benchmark. sigtest and sigfault2 are unavailable. Frozen independent evaluation is pending. The implementation directly tests whether constrained oscillator learning changes the forecast and keeps the effect of the optional adapter visible. Diagnosis is assessed through actual development residuals and synthetic retention tests with clear provenance. A truly independent conclusion requires executing the frozen protocol when the missing input records become available and verifying their acquisition labels and overlaps.


## 14. How an LPV filter differs from ordinary AR

**Backup**

Likely question. Where do the lagged innovations come from? The first-stage speed-conditioned AR model estimates innovations. The second stage adds lagged proxies as regressors. Read the protocol for when current versus previous speed is allowed. The paper result is supplied as external context.


## 15. How phase and causality meet

**Backup**

Likely question. Is previous vibration enough to predict phase? Shaft RPM advances the order phase by a causal extrapolation of one sample. Rotate past quadrature contributions into the estimated target phase. Since all inputs stop at t minus one, perturbing the future measurement cannot affect the current prediction.


## 16. Why mean squared error and PMI can disagree

**Backup**

Likely question. Can a lower MSE reduce crack evidence? Yes. MSE rewards prediction of predictable components including possible repetitive faults. PMI rewards angular periodicity in the residual envelope. A paired pulse test measures whether the predicted perturbation consumes the injected pulse. Verified classification requires authoritative condition labels.


## 17. Order spectra offer a second view of mesh energy

**Backup**

Likely question. Why examine the spectrum against angle as well as time? Shaft angle ties the frequency axis to the mechanical cycle. Components proportional to speed align to a fixed order. Fixed physical frequencies shift in the order coordinate as speed changes. Read the synthesis alongside the spectrogram and the matched feature ablations.


## 18. How the guarded folds keep raw histories apart

**Backup**

Likely question. Why guard a block on both sides? A row after a held-out block could draw raw samples from inside it. Finite model support means that the history can be bounded and its guard checked. A measured past observation can still serve as known input during one-step scoring.


## 19. PMI reference lag and parity status

**Backup**

Likely question. Why are there two PMI variants? MATLAB indexing begins at one, with ACF lag zero in its first position. Reading element T therefore accesses lag T minus one. Explicit zero-based lag T accesses one full turn. Exact end-to-end parity requires the instructor order-tracking helper and reference outputs.


## 20. Inspect a representative prediction trace

**Backup**

Likely question. What does the MSE average hide? Read the measured and predicted samples on a short interval. Aggregate statistics combine varying speed and changing vibration amplitude. The audit uses error by speed bin and paired blocks to expose that variation.


## 21. Inspect the oscillator learning trajectory

**Backup**

Likely question. How far do the carriers and memory scales move during learning? The run records order and decay updates by epoch. Small changes may matter locally and their contribution must be checked against the matched fixed-state model. Reproduction uses the exact parameters written to the checkpoint.


## 22. What the efficiency test measures

**Backup**

Likely question. Can the batch rate be used as a streaming guarantee? The benchmark measures prediction after warm-up on a full record. Online deployment would need its own latency benchmark, state initialization policy and budgets. CPU and GPU results are only claimed for hardware and execution observed in the run.
