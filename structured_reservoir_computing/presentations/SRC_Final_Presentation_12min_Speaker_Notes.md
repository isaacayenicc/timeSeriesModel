# SRC Final speaker notes

Planned speaking duration is **12:00**. Backup slides are untimed.

## 1. Physics-informed prediction at variable speed

**0:00–0:45**

PI-SRC MSE 0.329973 is 11.16% lower than the strict sparse LPV–ARMA result. The contribution is a finite, causal reservoir built from samples, angle, phase-transported harmonic states and prior errors. A regularized readout predicts the next measured sample. The report also tests which features help, how much signal remains for diagnosis, and what the completed timing evidence can support.

## 2. A mesh order moves in frequency as speed changes

**0:45–1:30**

A fixed time lag represents different shaft rotations as operating speed changes. A finite oscillator bank can track the same order across those changes. The project tests whether this structure adds value beyond regularized statistical models and broad data-driven predictors. A second concern is whether prediction and residual correction preserve repetitive disturbances.

## 3. Hold model selection inside the training acquisition

**1:30–2:25**

Use vibration and RPM observed through time t minus one. Train with guarded contiguous blocks and score the chosen models on the official validation record. Candidate records are a separate diagnostic set. Exact overlapping records are excluded from group statistics. With absent blind test files, report validation and candidate diagnostics within their own scopes.

## 4. Use observed ridges to motivate reservoir features

**2:25–3:15**

The 37-tooth shaft defines the theoretical mesh-order family. The measured spectrogram tests whether spectral ridges track RPM. The angle-domain view tests whether peaks sharpen at fixed orders. Broad hertz bands and transient effects motivate separately controlled candidates. The model can use lagged phase through measured inputs, while this descriptive spectrum uses a complete training record.

## 5. Transport historical observations to target phase

**3:15–4:35**

The finite harmonic state sums past vibration after transporting each sample from its measured shaft phase to the predicted target phase. Decay weights make older samples matter less. The target phase advances using the previous RPM. Other groups include vibration delays, angular lags, speed context and lagged proxy innovations. The nonlinear map uses selected feature products. The model fits the readout inside each training split, while order and memory settings are selected externally. The adapter can be applied identically to eligible controls.

## 6. Separate physical structure from the error adapter

**4:35–5:27**

The innovation pipeline and harmonic states create a structured, finite-memory representation. The regularized readout builds on this representation. The clipped residual adapter adds a slow correction from already observed base errors. Match that adapter across eligible models before attributing improvement to the oscillator or nonlinear structure. Review the fit objective, exact input limits, and feature-group ablations together.

## 7. Choose physical and readout settings together

**5:27–6:25**

Mean guarded MSE 0.3557. Selection objective 0.3717. selected_correction. The search considers finite feature and regularization choices and tracks its actual budget. Score each configuration under clean, sensor-perturbed and pulse-injection training conditions. Selection is a prespecified compromise and the study reports fold and development outcomes separately.

## 8. Compare against strong measured controls

**6:25–7:31**

PI-SRC MSE 0.329973 is 11.16% lower than the strict sparse LPV–ARMA result. The strongest replayed archive SRC checkpoint scored 0.317788 on the same mask. All 21 required comparator checkpoints were independently replayed. The complete model table records their evaluation scopes. The paper’s 0.2893 test result belongs to its own segment.

## 9. Inspect local errors as well as the mean

**7:31–8:21**

A mean error can combine acceleration, speed peaks and different vibration amplitudes. A matched short trace shows when the model follows the waveform and what remains in the residual. Errors by RPM bin show whether an observed gain generalizes across operating regimes within this record. The paired block comparison respects dependence among adjoining targets.

## 10. Accuracy and inference cost share the decision

**8:21–9:15**

Batch runtime includes feature construction and model prediction on the same input under the recorded threading policy. Model size and fit time illustrate different practical constraints. A full batch timing describes throughput on this hardware. Deployment needs separate streaming and tail-latency measurements. GPU cost can only be reported for measured hardware.

## 11. Residual PMI measures one-turn modulation

**9:15–10:17**

Median PMI gap B−A was 0.085. Conditional B-high AUC was 1.000. The diagnostic resamples each residual onto a common shaft-angle grid and computes the centered Hilbert envelope. PMI transforms the lagged correlation near one shaft turn. Exact overlaps are removed. The remaining 198 records have index-group labels only, so AUC describes group ordering. Synthetic pulse injection answers a different question about residual preservation under a known perturbation.

## 12. Attribute the result across matched controls

**10:17–11:11**

Same weights: 0.340706 without correction and 0.329973 with correction. The comparison separates order-aware state families, interaction groups and the same residual correction. An improvement in clean MSE can change residual evidence and sensitivity to RPM error. Interpret the ablations and robustness metrics jointly before recommending a deployment variant.

## 13. What the released evidence establishes

**11:11–12:00**

PI-SRC MSE 0.329973 is 11.16% lower than the strict sparse LPV–ARMA result. The finite structured reservoir creates a reproducible physical hypothesis about speed-dependent vibration. The project documents performance and trade-offs on a shared benchmark. Independent labels, independent blind data and streaming latency remain needed to characterize deployment.

## 14. Order spectra align rotation-locked energy

**Backup**

Order is cycles per reference revolution. A spectral peak helps motivate a reservoir group. Guarded prediction and matched ablation test its incremental value.

## 15. Publication and project comparisons

**Backup**

The course paper and the supplied development record represent different evaluation segments. Report the published result as external context and compare methods on the frozen local scoring mask.

## 16. Prediction error by ascending RPM quartile

**Backup**

The quartile plot uses bins calculated once from measured development RPM under the common mask. Read these bins as operating-point descriptions within one acquisition.

## 17. Accuracy and residual preservation can diverge

**Backup**

Stress changes the predictor's speed history. Pulse tests inject known synthetic signals into the same record and measure changes in predicted and residual components.

## 18. Model coverage and settings evidence 1

**Backup**

The measured table uses common standardized targets and identifies unavailable values. Exact configurations and fitted weights are in the result registry.

## 19. Model coverage and settings evidence 2

**Backup**

The measured table uses common standardized targets and identifies unavailable values. Exact configurations and fitted weights are in the result registry.

## 20. Model coverage and settings evidence 3

**Backup**

The measured table uses common standardized targets and identifies unavailable values. Exact configurations and fitted weights are in the result registry.
