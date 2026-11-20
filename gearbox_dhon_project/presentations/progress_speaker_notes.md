# MECG 7780 Progress presentation speaker notes
**Planned talk length** 8:00. Backup slides follow the timed talk.

| Slide | Planned window | Duration |
|---|---:|---:|
| 1. Predicting variable-speed gearbox vibration | 0:00–0:45 | 0:45 |
| 2. How the project will be judged | 0:45–1:30 | 0:45 |
| 3. The official split and overlap checks | 1:30–2:30 | 1:00 |
| 4. Mesh energy follows shaft speed | 2:30–4:00 | 1:30 |
| 5. Sparse LPV–ARMA establishes the reference | 4:00–5:00 | 1:00 |
| 6. DHON makes the oscillator trainable | 5:00–6:30 | 1:30 |
| 7. Tune within training and score once on development | 6:30–7:30 | 1:00 |
| 8. Progress and remaining evidence | 7:30–8:00 | 0:30 |
| 9. How an LPV filter differs from ordinary AR | Backup | — |
| 10. How phase and causality meet | Backup | — |
| 11. Why mean squared error and PMI can disagree | Backup | — |
| 12. Order spectra offer a second view of mesh energy | Backup | — |

## 1. Predicting variable-speed gearbox vibration

**0:00–0:45**

The gearbox speeds up and slows down, so a mesh-related component moves in frequency through a single acquisition. A fixed-lag model can still exploit short-time dependence, but it has to infer that changing phase relationship from samples. We want to estimate healthy vibration one sample ahead and test whether the remaining signal retains evidence of periodic tooth damage. This progress overview uses descriptive examples generated during preparation of the submitted project.


## 2. How the project will be judged

**0:45–1:30**

The course evaluates prediction accuracy, computational cost and fault detection. Our primary error metric is mean squared error on identical target indices. Efficiency means measuring inference with the deployed feature pipeline and recording the platform. For diagnostics, we first model normal vibration and then examine the envelope of the residual at one target-shaft revolution. A lower prediction error alone cannot establish a diagnostic benefit because a predictor that follows fault impulses can also suppress them.


## 3. The official split and overlap checks

**1:30–2:30**

The official training and development validation records each contain twenty-four thousand samples, corresponding to seven and a half seconds at 3,200 Hz. The split fault collection supports a separate diagnostic analysis. The exact-match audit verified duplicate prefixes in fault-development records 61 and 63, so we preserve those records and exclude them from the eligible group comparison. Fault labels require independent acquisition metadata. A split by record position can describe a group pattern, but a verified classification score needs confirmed labels and records withheld from tuning.


## 4. Mesh energy follows shaft speed

**2:30–4:00**

For an encoder on the monitored 37-tooth shaft, mesh frequency is thirty-seven times shaft speed divided by sixty. Multiples of that frequency trace families of curves on a time-frequency map. Integrating RPM gives shaft revolutions, so resampling by angle brings speed-locked components near stable orders. Broadband energy is an additional hypothesis for fixed-frequency features. A spectrogram alone cannot identify a natural mode. Each prior becomes a model change to test on guarded training blocks. Compare order-aligned lags with temporal lags, and compare fixed and learnable oscillator banks. The exploratory transforms use a whole record and the production predictor takes its inputs from measured history.


## 5. Sparse LPV–ARMA establishes the reference

**4:00–5:00**

An autoregression uses previous vibration samples. LPV means that the lag coefficients vary smoothly with an operating variable, here speed. The moving-average portion uses lagged innovations, approximated by first-stage prediction errors. The sparse fit restricts the number of active coefficients. The strict benchmark schedules coefficients using speed available at the preceding sample and reproduces performance on the supplied validation segment. The paper result supplies context because the segment described there differs in speed phase.


## 6. DHON makes the oscillator trainable

**5:00–6:30**

Here phase comes from the previous RPM and an integration of the measured speed history. For each carrier, the encoder combines earlier vibration samples after transporting them to the target phase. Damping gives a shorter or longer statistical memory. The archived model selected the oscillator settings externally and learned a mixing head. The proposed computation places bounded order and decay parameters inside the differentiable map. A fixed-state control uses the same architecture with those parameters held at their initial values. Separately, a slow residual adapter sees clipped errors only after each target arrives. The matched adapter control measures whether slow bias reduction explains any improvement in the final combined predictor.


## 7. Tune within training and score once on development

**6:30–7:30**

The fixed gap around each held-out training block protects the raw histories that feed fitting and scoring. Past samples before a held-out target are observed as they would be in one-step deployment. Every model uses the same development targets, from index 1,792 through index 23,999 inclusive. The key ablations compare the fixed versus trainable bank and apply the same optional adapter to both. RPM stress changes only the claimed measurement channel. Synthetic pulses test fault preservation in paired injected and unmodified runs.


## 8. Progress and remaining evidence

**7:30–8:00**

The descriptive evidence gives us interpretable feature candidates and a causal design. The next scientific check is whether the learning and correction controls change prediction error or the amount of periodic pulse retained. External test files, confirmed acquisition labels and reference MATLAB outputs govern the strength of the final claims.


## 9. How an LPV filter differs from ordinary AR

**Backup**

Likely question. Where do the lagged innovations come from? The first-stage speed-conditioned AR model estimates innovations. The second stage adds lagged proxies as regressors. Read the protocol for when current versus previous speed is allowed. The paper result is supplied as external context.


## 10. How phase and causality meet

**Backup**

Likely question. Is previous vibration enough to predict phase? Shaft RPM advances the order phase by a causal extrapolation of one sample. Rotate past quadrature contributions into the estimated target phase. Since all inputs stop at t minus one, perturbing the future measurement cannot affect the current prediction.


## 11. Why mean squared error and PMI can disagree

**Backup**

Likely question. Can a lower MSE reduce crack evidence? Yes. MSE rewards prediction of predictable components including possible repetitive faults. PMI rewards angular periodicity in the residual envelope. A paired pulse test measures whether the predicted perturbation consumes the injected pulse. Verified classification requires authoritative condition labels.


## 12. Order spectra offer a second view of mesh energy

**Backup**

Likely question. Why examine the spectrum against angle as well as time? Shaft angle ties the frequency axis to the mechanical cycle. Components proportional to speed align to a fixed order. Fixed physical frequencies shift in the order coordinate as speed changes. Read the synthesis alongside the spectrogram and the matched feature ablations.
