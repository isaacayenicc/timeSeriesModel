# Time Serial Dependency Modeling via Advanced Machine Learning Algorithms

MECG 7780 Engineering Data Analytic · Reproducible Python validation study  
Author / group: **to be supplied by the user**  
Generated: 2026-10-30 · Final submission/presentation dates: **unconfirmed**

> **Completion boundary.** The twenty split MAT uploads now enable analysis of **200 separate fault-candidate records**. Their structure, raw/residual features and explicit angular-unit sensitivities were measured without retraining predictors on those records. Physical fault-detection accuracy remains **unidentifiable without verified health labels**. `sigtest.mat` and `sigfault2.mat` were not supplied. All prediction scores below are validation scores, not held-out test scores.

## Abstract

Variable operating speed complicates vibration prediction because frequency content and dynamic response may vary over time. We implement strictly causal, one-step predictors of supplied dimensionless vibration using the accessible `sigtrain.mat` and `sigvalid.mat` recordings. Each contains 24,000 samples at 3,200 Hz. The comparison comprises naive predictors, ordinary and ridge autoregression, a causal speed-augmented ARX baseline, a polynomial speed-conditioned LPV-AR surrogate, a compact multilayer perceptron, a unidirectional GRU, a causal dilated temporal convolutional network (TCN), and FiLM-style speed-conditioned variants. All models are evaluated at the same 23,744 validation target indices. The best observed eligible configuration, `lpv_L256_s17_a1`, achieves MSE **0.361217**, versus **1.104077** for zero prediction and **0.420058** for ridge AR(256). This is a **67.28%** reduction relative to zero prediction on this trace. The recommendation is provisional because validation is also used for epoch/configuration selection, the speed histories are nearly identical, and independent test data are absent. CPU timing and checkpoint round trips are measured. The fault study processes 200 separate candidate recordings and compares raw/residual statistics and assumption-labelled angular indicators. These descriptive results do not establish physical fault classes, severity or detection accuracy. Physical angular calibration, live MATLAB parity and an exact Spa LPV-ARMA comparison remain unverified.

## 1. Engineering motivation and course scope

Rotating systems contain repeatable vibration components from excitation, meshing and structural dynamics. Under variable speed, a fixed-frequency representation may spread otherwise orderly periodic content across frequency. A next-sample model can quantify predictable structure and generate a residual for subsequent analysis. The course explicitly requests MSE, CPU/GPU efficiency and fault performance; it encourages speed conditioning and supplies MATLAB PMI code. Those obligations must remain separate from the smaller set of empirically completed tasks.

The authoritative updated PDF assigns progress weights of **4% organization/presentation, 6% clarity, 8% algorithm understanding and 2% novelty**, and final weights of **16%, 24%, 30% performance and 10% novelty**. These differ from the master prompt. The PDF prints Nov 4, 2026 for the 8-minute progress presentation but Dec 7, 2025 at 24:00 and Dec 9, 2025 for report and final presentation. We preserve the conflict rather than inventing dates. See [course requirements](../docs/course_requirements_and_open_questions.md) and [rubric mapping](../docs/rubric_traceability_matrix.md).

## 2. Research context and strength of evidence

The course cites Chen, Li, Jiang, Gong and Zhou (2025), *Sparse LPV-ARMA model for non-stationary vibration representation and its application on gearbox tooth crack detection under variable speed conditions*, MSSP 224, 112161, DOI 10.1016/j.ymssp.2024.112161. The PDF attributes the same data to that paper. Publisher/full-text access was restricted, so the exact sparse estimator, innovation recursion, data selection and reported MSE could not be independently reproduced. We therefore implement a related **polynomial LPV-AR surrogate** and make no numerical claim to outperform Spa LPV-ARMA.

Original TCN author material motivates causal dilated convolution as an efficient sequence-model alternative to recurrence. Original FiLM author material supports feature-wise conditioning by gains and offsets. These source tasks differ from gearbox diagnosis; their existence is architectural motivation, not evidence that either model wins here. Ridge is essential because a short high-rate recording supplies many highly dependent windows and a modest linear predictor may be accurate and inexpensive. The full [research synthesis](../research/research_synthesis.md), [literature matrix](../research/literature_matrix.csv) and [access log](../research/source_access_log.md) distinguish course citation, accessed implementation documentation, abstract evidence and inaccessible full methods. The requested deep domain-specific review is limited by those access constraints.

## 3. Research questions and predeclared hypotheses

**RQ1:** Which tested causal configuration minimizes common-index validation MSE? **RQ2:** What accuracy–CPU-cost–parameter trade-off is observed? **RQ3:** Does observed past-speed conditioning improve prediction beyond capacity and optimization changes? **RQ4:** Do residuals expose useful periodic/impulsive diagnostic structure relative to raw signals? **RQ5:** Are conclusions stable across contexts, seeds and feature conventions?

The falsifiable working hypotheses are that contextual predictors improve on naive prediction, speed conditioning may help under nonstationary operating conditions, and residual processing may clarify periodic/impulsive structure. None is presumed true. Physical fault-detection accuracy cannot be evaluated without verified labels. Model selection and descriptive diagnostic interpretation are separate decisions.

## 4. Data, provenance and forensic audit

The original accessible files are preserved in `data/raw/`. SHA256, byte counts, availability and source fields are stored in [the manifest](../data/manifest.json); machine-readable measured summaries are in [data_audit.json](data_audit.json).

| Property | Training | Validation |
|---|---:|---:|
| Samples | 24,000 | 24,000 |
| Sampling rate (Hz) | 3,200 | 3,200 |
| N/fs duration (s) | 7.5 | 7.5 |
| First timestamp (s) | 0.0003125 | 0.0003125 |
| Last timestamp (s) | 7.5 | 7.5 |
| Supplied x mean | -0.003079532 | -0.011850961 |
| Supplied x standard deviation, ddof=0 | 1.040098453 | 1.047268332 |
| Empirical xraw mean | 0.001312861 | -0.002416514 |
| Empirical xraw standard deviation, ddof=0 | 0.442221708 | 0.445270147 |

The first stored scaling entries give `x=(xraw−0.0026221941677904226)/0.42517293168840065`, and supplied speed normalization uses reference endpoints `124.07455535625635` and `167.3376580579989`. Reconstruction maximum error is zero at the inspected NumPy precision. The stored arrays have three entries; only the first entries' role in the accessible scalar vibration normalization is verified. Their other entries are not interpreted. The empirical training moments differ from the metadata and are not substituted into the primary experiment.

The speed-trace Pearson correlation is **0.999999570533**; vibration correlation is **0.041290760**. No validation speeds fall outside the observed training min/max in this pair of recordings. This is restricted coverage, not proof of data leakage. We cannot infer performance under a substantially different speed trajectory. The physical speed unit is absent from verified metadata, so figures use “source units.” Supplied normalized vibration and residuals are dimensionless; a historical MATLAB axis annotation does not establish acceleration calibration.

The original oversized fault upload was superseded by twenty accessible split MAT files in `data/raw/sigfault1_parts/`. The finalized loader/audit preserves **200 independent records**; all raw/residual descriptor rows use a common **31,744-sample** segment after **256** burn-in samples. No record-wise rescaling or concatenation is performed. The verified split files expose supplied x rather than xraw, and no health-state, fault-onset or severity labels were found. File-part names and SHA256 provenance accompany feature rows. The record index is an ordering identifier only, not a health label.

![Accessible vibration and speed traces](figures/eda.png)

*Figure 1. Actual accessible train/validation traces, sampled at 3,200 Hz. Speed and vibration must be interpreted with their separately stated units. Data hashes identify the exact files.*

## 5. Prediction formulation, alignment and leakage control

The primary target is `xhat[t] = f(x[t−L:t], spdnorm[t−L:t])`; the right slice endpoint is excluded. A lag-L window first predicts target `x[L]` from `x[0:L]`. For speed-only interactions, the latest operating value is `spdnorm[t−1]`. No contemporaneous or future vibration enters the model. The primary protocol is **teacher-forced one-step prediction**, using previously measured true vibration; it is not a recursive free-running forecast.

All reported MSEs use the identical zero-based validation indices **256 through 23,999 inclusive**, giving **23,744** targets. This maximum burn-in accommodates all tested lags. The loss is the arithmetic mean of squared dimensionless residuals, with no extra standardization or per-model target removal. RMSE, MAE and NMSE are secondary; NMSE divides by the target mean square, not target variance. Recording windows never cross file boundaries. Training minibatch permutation changes optimization ordering; it does not create a random train/test split.

The training file fits coefficients and weights. Validation is used for neural epoch selection and observed model/configuration ranking. Consequently, reported validation scores are useful comparison evidence on this trace but are not unbiased final generalization estimates. There is no claimed untouched test set. Any blockwise uncertainty describes temporal variation within this trace, not independent machine-to-machine uncertainty. An off-line phase-shifted-speed experiment is explicitly a negative control and excluded from causal model recommendations; it must not be deployed as a causal speed policy.

## 6. Models and mathematical objectives

### 6.1 Naive and linear models

Zero prediction uses `xhat[t]=0`; training mean uses one scalar fitted on the supplied training x; persistence uses `xhat[t]=x[t−1]`. These differ and are scored separately. AR(p) uses a row of the p previous vibration samples and an intercept. OLS minimizes `||y−Xb||²`; ridge minimizes `||y−Xb||² + α||b||²` with an unpenalized fitted intercept, `α=1` and scikit-learn's Cholesky solver. No matrix inverse is explicitly formed. Lags 32/64/128/256 correspond to 10/20/40/80 ms at 3,200 Hz.

ARX appends the last observed normalized speed and its square. The polynomial LPV-AR surrogate concatenates `X`, `X*s[t−1]` and `X*s[t−1]²`, so each AR coefficient can depend quadratically on speed. All supplied x/spdnorm fields retain their common scaling. These interaction columns are correlated, making ridge stabilization relevant. The surrogate does not include recursive innovation lags and does not reproduce the lecturer's sparse LPV-ARMA estimation. Parameter counts include the intercept.

### 6.2 Compact nonlinear models

The MLP uses a lag-vector input, one 32-unit ReLU hidden layer and a scalar output. The GRU is unidirectional with 16 hidden units and a scalar linear head; reset and update gates regulate the candidate state and retained history. Neither model is bidirectional. Both receive vibration-only inputs in the principal suite.

The TCN first projects the vibration channel to eight features. Each residual block contains two kernel-three convolutions at one dilation, with explicit **left padding only**, ReLU activations and a residual sum. Dilations double by block. Its receptive field is `R=1+4*Σdilation`, covering at least the declared lag; padding fills missing early context with zeros and never future data. A final linear head maps the last temporal feature to the next sample. Contexts 32/64/128 test a bounded range rather than an unrestricted architecture search.

### 6.3 Project-level speed-conditioned contribution

For each temporal position k in the observed window, FiLM-style conditioning maps its observed normalized speed to channel-wise gain and offset. The implementation uses a bounded residual gain `1+0.1*tanh(g(s[k]))` and additive `0.1*b(s[k])` after a TCN block. This is an adaptation of established conditioning, not a world-first method claim. Conditioning at past positions respects the availability policy.

Controls include the same core TCN without speed, late concatenation of speed, a nine-channel unconditioned TCN with a larger parameter budget, and an intentionally phase-misaligned speed negative control. The last is off-line and excluded from the eligible winner. Counts and timings are reported directly rather than described as exactly parameter matched. The controls improve interpretation but cannot eliminate all optimization, effective-capacity or trajectory-specific confounding.

## 7. Training budget, seeds and checkpoint selection

The bounded CPU suite uses neural seeds 17, 29 and 43 for principal lag-64 TCN/speed-TCN pairs; other neural configurations have seed 17. AdamW uses learning rate 0.003, weight decay 0.0001, batch size 512, gradient clipping at norm 1, a maximum of 10 epochs and patience 3. The checkpoint with lowest declared common-index validation MSE is restored. The complete effective configuration is saved beside each run; actual completed epochs and learning curves are retained. Short training is a deliberate feasibility budget, not proof that all architectures reached their best attainable accuracy.

Linear computation uses float64 and neural weights/windows float32. Two PyTorch CPU threads and deterministic algorithms are requested. Seeds cover initialization and shuffled minibatch order. CPU library versions and hardware are retained in the project metadata. Each model is persisted, reloaded through the public inference path and compared against pre-save predictions; round-trip errors appear in `results_summary.csv`. Trusted local pickle artifacts must not be loaded from untrusted distributions.

![Actual saved neural learning curves](figures/training_curves.png)

*Training evidence. Curves are read from saved per-run histories. Epoch selection uses validation and therefore does not constitute an independent test. Early plateaus or seed variation should be read in the context of the bounded training budget, not treated as a universal architecture limitation.*

## 8. Prediction results — every measured run

The table below is generated from [results_summary.csv](results_summary.csv). `train_wall_s` covers the implemented fit/training region including internal validation scoring and checkpoint generation; it is not pure optimizer-kernel time. `inference_record_median_s` is the median public full-record inference path, including checkpoint loading and window preparation. Results are sorted by observed MSE. All rows use validation and common target indices.

| run_id | mse | parameters | train_wall_s | inference_record_median_s |
|---|---|---|---|---|
| lpv_L256_s17_a1 | 0.361217 | 769 | 0.765804 | 0.122543 |
| lpv_L128_s17_a1 | 0.381163 | 385 | 0.450643 | 0.031192 |
| lpv_L64_s17_a1 | 0.394738 | 193 | 0.047654 | 0.012896 |
| ar_L256_s17_a1 | 0.420058 | 257 | 0.069029 | 0.006647 |
| ols_L256_s17 | 0.420059 | 257 | 0.263791 | 0.006362 |
| ar_L128_s17_a1 | 0.426825 | 129 | 0.027784 | 0.003237 |
| arx_L256_s17_a1 | 0.427821 | 259 | 0.063736 | 0.007752 |
| arx_L128_s17_a1 | 0.430216 | 131 | 0.026958 | 0.004128 |
| ar_L64_s17_a1 | 0.439050 | 65 | 0.012017 | 0.001776 |
| arx_L64_s17_a1 | 0.443191 | 67 | 0.014981 | 0.002280 |
| mlp_L64_s17 | 0.450469 | 2113 | 1.489657 | 0.007496 |
| lpv_L32_s17_a1 | 0.454310 | 97 | 0.021848 | 0.005997 |
| tcn_L128_s17 | 0.454490 | 2425 | 26.812453 | 0.535576 |
| speed_tcn_L128_s17 | 0.458226 | 2617 | 52.985214 | 1.259770 |
| speed_tcn_L64_s43 | 0.459519 | 2185 | 22.363303 | 0.548469 |
| tcn_L64_s17 | 0.461668 | 2025 | 11.971107 | 0.223912 |
| late_tcn_L64_s17 | 0.463274 | 2026 | 19.411123 | 0.413156 |
| shift_tcn_L64_s17 | 0.463683 | 2185 | 25.978204 | 0.586385 |
| tcn_L64_s29 | 0.464516 | 2025 | 12.359172 | 0.248665 |
| capacity_tcn_L64_s17 | 0.471419 | 2548 | 21.795842 | 0.459164 |
| speed_tcn_L64_s17 | 0.483214 | 2185 | 20.098585 | 0.567299 |
| tcn_L64_s43 | 0.484686 | 2025 | 11.811438 | 0.226302 |
| speed_tcn_L64_s29 | 0.494911 | 2185 | 17.724904 | 0.581169 |
| ar_L32_s17_a1 | 0.503324 | 33 | 0.010628 | 0.001210 |
| tcn_L32_s17 | 0.508317 | 1625 | 6.955784 | 0.112347 |
| speed_tcn_L32_s17 | 0.512236 | 1753 | 10.519432 | 0.252117 |
| arx_L32_s17_a1 | 0.517247 | 35 | 0.009464 | 0.001689 |
| gru_L64_s17 | 0.823606 | 929 | 12.339703 | 0.314197 |
| mean_L256_s17 | 1.104017 | 1 | 0.000256 | 0.000317 |
| zero_L256_s17 | 1.104077 | 0 | 0.000366 | 0.000331 |
| persistence_L256_s17 | 2.032667 | 0 | 0.000201 | 0.000224 |

![Common-index model comparison](figures/model_comparison.png)

*Figure 2. Measured common-index validation comparison. Multiple seeds are initialization repeats on the same recording; they are not independent machines.*

Ridge AR(256) reproduces an MSE of **0.420057590**, consistent with the exploratory value in the master prompt after independent execution. The polynomial LPV-AR(256) surrogate yields **0.361217468** under this configuration. The best eligible observed model is **lpv_L256_s17_a1**, with **769** parameters. Neither the baseline match nor an improvement within this suite establishes outperformance of the published Spa LPV-ARMA procedure.

The LPV-AR(256) surrogate reduces MSE by **14.008%** relative to ridge AR(256) on these common targets. No tested neural run beats ridge AR(256) under the declared short training budget. Appending speed as additive ARX terms does not match the benefit of polynomial speed-by-lag interactions here. That pattern is consistent with speed-dependent prediction coefficients being more useful than a speed-dependent output offset in this pair of traces; it is a conditional empirical observation, not proof of mechanism. The GRU's relatively high error indicates poor performance for this compact configuration and training budget, not that GRUs cannot model machinery signals.

![Observed/predicted validation vibration](figures/prediction.png)

*Figure 3. Saved validation predictions and residuals on a representative interval. The plotted values come from the aligned target arrays, not simulated performance. Residual energy is not physical fault accuracy.*

## 9. Robustness, context and speed-conditioning ablations

Seed summaries for principal lag-64 comparisons are:

| kind | mean | std | min | max | count |
|---|---|---|---|---|---|
| speed_tcn | 0.479215 | 0.018032 | 0.459519 | 0.494911 | 3 |
| tcn | 0.470290 | 0.012548 | 0.461668 | 0.484686 | 3 |

The standard deviation uses pandas' sample convention across seeds. With only three initializations, it is a coarse measure of training variability. It does not quantify independent operating-condition or fault-state uncertainty. Context comparisons are in [model_ablation.csv](model_ablation.csv), alongside late-conditioning, capacity and shifted-speed controls. A selected single-seed best result must not replace the multi-seed comparison. If a condition does not improve, the negative evidence remains part of the study.

At lag 64, speed-TCN has **1.898% higher** mean MSE than unconditioned TCN across the three matched seeds. The proposed FiLM speed adaptation is therefore **not supported as an average prediction improvement** in this experiment. A single favorable seed cannot reverse that conclusion. Longer tested context helps the principal TCN configurations relative to lag 32, but the short neural runs remain behind the stronger linear baselines. The project contribution is an implemented, falsifiable comparison with a negative neural-conditioning result, rather than a promised novel-model victory.

Attribution of a speed benefit requires caution: speed trajectories almost coincide between train and validation, LPV interactions may act as a convenient time-dependent coefficient basis, and the FiLM model has additional trainable parameters. The larger unconditioned control and late-concatenation baseline are informative but not a complete factorial decomposition. Claims should be restricted to this finite budget and trace. Speed-binned errors are descriptive stratification and do not mean the model observed target-time speed.

![Measured validation errors by operating-speed bin](figures/speed_errors.png)

*Speed-stratified evidence. Bin-wise MSE describes errors on the actual validation samples and has different sample counts across operating regions. Stratifying by target-time speed for interpretation does not add that value to any predictor's past-only inputs. This is not a test on an unseen speed trajectory.*

## 10. Computational efficiency and practical trade-offs

![Accuracy–cost comparison](figures/efficiency.png)

*Figure 4. Accuracy and measured CPU inference cost for actual saved models. Timing boundaries and precision differ from a hypothetical isolated GPU kernel benchmark and are stated explicitly.*

The public scoring path is warmed once and repeated three times per model; median, minimum and maximum wall times are saved. It loads the checkpoint, constructs windows and predicts the full validation recording, while metrics use the shared subset. This is useful local end-to-end cost but should not be labelled pure per-sample neural latency. Parameter counts and artifact bytes are measured. The CSV retains every final run. One speed-control trial was corrected and retrained; its superseded cost is retained in `experiments/superseded_trials.json`. The hardware JSON reports both final-artifact and all-attempt training-time sums. GPU measurements were **not run** on this CPU-only setup. Peak GPU memory and cross-device acceleration are therefore not claimed.

Within-trace uncertainty is recorded in [within_trace_uncertainty.csv](within_trace_uncertainty.csv): nonoverlapping blocks of 256, 512 and 1024 samples, 1000 block-mean resamples with a fixed seed, and explicit incomplete-tail exclusions. These intervals concern temporal variation within this validation trace, are not corrected for model selection, and do not estimate uncertainty across machines or independent recordings.

A compact AR model may be preferable in a latency-sensitive deployment even when a larger model yields a lower selected MSE. Conversely, a speed-conditioned surrogate may offer a good accuracy–cost compromise. Deployment decisions also require verified streaming overhead, independent speed transfer and diagnostic retention of anomalous components, none established by one validation trace.

## 11. Residual statistics and the diagnostic hypothesis

For a causal predictor, the residual is `e[t]=x[t]−xhat[t]`. RMS is `sqrt(mean(e²))`; crest factor is `max(abs(e))/RMS`; ordinary kurtosis is the fourth centered moment divided by squared variance. Ordinary Gaussian kurtosis is 3, whereas excess kurtosis subtracts 3. The implementation states the biased central-moment convention and handles zero/constant signal cases explicitly.

Predictable reference vibration may be suppressed by the model. However, a flexible teacher-forced predictor can also learn local evolution within a defect-containing signal and suppress fault-related components. Therefore lower MSE is not logically equivalent to better fault detection. Raw/residual comparisons on accessible data establish numerical signal behavior, not a physical defect label. No model is designated a proven best physical diagnostics model.

## 12. MATLAB-reference port, angular assumptions and numerical validation

The supplied `CorrKurt` forms delayed, zero-padded copies and divides the squared-product sum by signal energy raised to `M+1`. Python allocates all `M+1` columns explicitly rather than relying on dynamic growth. Its period is an integer **sample lag**, not an automatically known shaft frequency. The PMI workflow extracts an analytic-signal envelope and uses a centered energy-normalized autocorrelation. The exact treatment of envelope mean and finite boundaries is documented in [numerical validation](numerical_validation.md).

In MATLAB, autocorrelation lag zero is the first array element. The literal coarse reference access at one-based index `round(fs/f_fault)` therefore reads physical lag **T−1**. The scientifically corrected variant reads lag **T**; both are exposed rather than silently changing historical behavior. The script's returned `timeindex` contains a separate offset and must not be mistaken for the actual selected lag. `PMI=rho/(1−rho)` is ill-conditioned near rho=1, and undefined/constant-signal cases require visible handling. One-sided FFT scaling differs for odd/even lengths; the corrected odd-length last positive bin must be doubled.

Angular resampling integrates speed after explicit conversion to cycles/s: RPM divides by 60, rps is unchanged, and rad/s divides by 2π. Interpolation onto a uniform cycle grid is a diagnostic transformation, not primary predictive preprocessing. Unknown units prevent an unqualified physical order interpretation; a speed assumption must be named and sensitivity explored. The supplied external `toConstantSamplesPerCycle` dependency is absent, so live MATLAB/toolbox parity is unverified. Synthetic tests of known-speed signals and feature formulas support the Python implementation only; they do not prove tooth-crack detection.

The following **actual accessible validation-signal sensitivity** results are read from [angular_unit_sensitivity.csv](angular_unit_sensitivity.csv). They are not fault-record results and do not establish physical speed units or a true defect order. Samples per cycle vary to accommodate source sampling density under each assumption.

| series | speed_unit_assumption | samples_per_cycle | pmi_scientifically_corrected | ck_order1 | roundtrip_rmse_fraction_of_series_rms |
|---|---|---|---|---|---|
| supplied_x | rpm | 2048 | 0.084614 | 0.000028 | 0.269326 |
| supplied_x | rad_s | 256 | 0.068392 | 0.000024 | 0.225154 |
| supplied_x | rps | 32 | 0.188517 | 0.000033 | 0.286661 |
| residual:lpv_L256_s17_a1 | rpm | 2048 | 0.040146 | 0.000027 | 0.283663 |
| residual:lpv_L256_s17_a1 | rad_s | 256 | 0.046954 | 0.000023 | 0.239038 |
| residual:lpv_L256_s17_a1 | rps | 32 | 0.041046 | 0.000029 | 0.302854 |

The different PMI values show that an unverified unit assumption materially changes the evaluated lag/order interpretation. Raw and best-model residual values use equal source segments after burn-in. The residual PMI is not automatically larger; lower prediction MSE does not imply enhanced periodic diagnostic contrast. Round-trip interpolation RMSE is measured relative to the corresponding source-series RMS and reveals substantial reconstruction change for these broadband signals; denser angular interpolation cannot restore information absent from the time samples. These are sensitivity diagnostics, not a validated tooth-crack detector.

## 13. Real candidate-record study and diagnostic interpretation

### 13.1 Actual record-wise protocol and completeness

The supplied twenty-part dataset is now accessible and **200 distinct records** have been processed individually. This resolves the original single-upload transfer obstacle. Each predictor was trained only on the reference training recording, then frozen; candidate records were not used for retraining, epoch selection or the reported validation ranking. Recording state/context is reset at each boundary. The first 256 samples are excluded equally for raw and residual descriptors, leaving **31,744 samples per record**. No labels are inferred from part boundaries, record order or a striking figure.

The long-format [fault_record_features.csv](fault_record_features.csv) contains original part/hash provenance, model ID, burn-in/evaluated count, speed summaries, RMS, crest factor, ordinary kurtosis and measured computation times. [fault_angular_features.csv](fault_angular_features.csv) separately records speed-unit assumptions, angular sample density, interpolation change/edge information, CK, both PMI indexing variants and validity statuses. [fault_study_summary.json](fault_study_summary.json) summarizes actual completion. These are real candidate-record measurements, distinct from earlier synthetic numerical tests.

### 13.2 Raw versus residual statistics

| model_id | records | median_rms | median_raw_rms_ratio | median_crest_factor | median_kurtosis |
|---|---|---|---|---|---|
| ar_L256_s17_a1 | 200 | 0.665676 | 0.630237 | 4.818784 | 3.295679 |
| arx_L256_s17_a1 | 200 | 0.666277 | 0.631446 | 4.812602 | 3.298296 |
| capacity_tcn_L64_s17 | 200 | 0.695132 | 0.657809 | 4.858827 | 3.292343 |
| gru_L64_s17 | 200 | 0.913906 | 0.871571 | 5.011131 | 3.412047 |
| late_tcn_L64_s17 | 200 | 0.688605 | 0.653663 | 4.868368 | 3.304069 |
| lpv_L256_s17_a1 | 200 | 0.623974 | 0.588109 | 4.908411 | 3.272660 |
| mean_L256_s17 | 200 | 1.053415 | 0.999985 | 4.814454 | 3.391280 |
| mlp_L64_s17 | 200 | 0.681743 | 0.648656 | 4.795177 | 3.277209 |
| ols_L256_s17 | 200 | 0.665687 | 0.630243 | 4.818087 | 3.295630 |
| persistence_L256_s17 | 200 | 1.493909 | 1.426888 | 5.118118 | 3.562371 |
| raw | 200 | 1.053410 | 1.000000 | 4.814118 | 3.391280 |
| speed_tcn_L128_s17 | 200 | 0.680298 | 0.645790 | 4.826195 | 3.289983 |
| tcn_L128_s17 | 200 | 0.681158 | 0.647383 | 4.844180 | 3.322133 |
| zero_L256_s17 | 200 | 1.053410 | 1.000000 | 4.814118 | 3.391280 |

The recommended validation predictor's median per-record residual/raw RMS ratio is **0.588109**. This ratio measures energy suppression on equal segments, not accuracy of physical fault identification. Its median residual kurtosis is **3.272660** under the stated biased Pearson convention.

![Actual candidate-record RMS trends](figures/fault_rms_trends.png)

*Figure 5. Record-wise raw and residual RMS on equal source segments. All records are independent; connecting their indices visualizes ordering only and does not imply a continuous time trace or known fault onset.*

![Actual candidate-record indicator distributions](figures/fault_indicator_distributions.png)

*Figure 6. Raw/residual descriptor distributions across actual candidate records. These distributions have no known healthy/faulty class labels, so visual separation cannot be converted to fault sensitivity or classification accuracy.*

### Post hoc record-order observation

The all-record plot shows an amplitude change near source ID101. A post hoc descriptive comparison of source IDs1–100 and101–200 is saved in [fault_posthoc_source_partition.csv](fault_posthoc_source_partition.csv). The two groups have different median raw/residual RMS but similar coarse speed moments. This is not verified fault onset or healthy/faulty labeling: acquisition order, load, preprocessing and machine state are not documented, and similar speed moments do not prove identical trajectories. No thresholds or models were selected from this partition. Separate recording processing also does not establish statistical independence of acquisitions.

### 13.3 Angular-unit and MATLAB-indexing sensitivity

The following table gives per-condition **medians across independently processed records** for raw, the prediction-recommended model and ridge AR(256). The speed interpretations RPM, rad/s and rps remain assumptions. The assumed one-cycle modulation order is not a verified tooth-fault characteristic. Median feature changes are descriptive and can depend on speed, interpolation, energy, lag convention and record length.

| model_id | speed_unit_assumption | records | median_ck | median_pmi_literal | median_pmi_corrected |
|---|---|---|---|---|---|
| ar_L256_s17_a1 | rad_s | 200 | 0.000018 | 0.034216 | 0.038351 |
| ar_L256_s17_a1 | rpm | 200 | 0.000023 | 0.093846 | 0.089866 |
| ar_L256_s17_a1 | rps | 200 | 0.000023 | 0.044586 | 0.041620 |
| lpv_L256_s17_a1 | rad_s | 200 | 0.000018 | 0.045317 | 0.048818 |
| lpv_L256_s17_a1 | rpm | 200 | 0.000023 | 0.089413 | 0.087003 |
| lpv_L256_s17_a1 | rps | 200 | 0.000023 | 0.052396 | 0.052057 |
| raw | rad_s | 200 | 0.000019 | 0.101228 | 0.100360 |
| raw | rpm | 200 | 0.000023 | 0.100186 | 0.097157 |
| raw | rps | 200 | 0.000026 | 0.184264 | 0.188480 |

![Actual angular-unit sensitivity](figures/fault_unit_sensitivity.png)

*Figure 7. Candidate-record CK/PMI sensitivity under three explicitly named speed-unit interpretations. Neither choosing a convenient unit nor choosing the most separated distribution establishes physical calibration or fault labels.*

### 13.4 Prediction quality versus diagnostic utility

The validation MSE winner and a diagnostic recommendation answer different questions. A low-energy residual may remove predictable reference structure, but can also suppress periodic content that might matter diagnostically. Changes in kurtosis/PMI/CK are therefore inspected alongside raw descriptors, speed and implementation assumptions; a larger statistic is not universally a better detector. The safe practical recommendation is to retain **both raw descriptors and the recommended predictor's residual descriptors** for further calibrated investigation, rather than declare a proven physical diagnostic winner. Model-to-model plots are in [fault_prediction_diagnostic_tradeoff.png](figures/fault_prediction_diagnostic_tradeoff.png).

No physical accuracy, AUC, F1, sensitivity, fault onset, crack presence or severity is reported. Such metrics require documented labels and record-level threshold-selection/evaluation on independent data. Even if an apparent change aligns with a record index or part boundary, it remains a post hoc descriptive association until acquisition metadata establishes meaning. See [fault label limitations](../docs/fault_label_limitations.md).

## 14. Scientific validity, limitations and failure modes

The strongest verified prediction findings concern one-step prediction on the supplied validation trace. Sources of uncertainty include validation reuse for checkpoint/model selection, short duration, strongly dependent neighboring windows, nearly repeated speed histories, finite training budgets, no independently verified physical labels, unknown speed units and missing independent test inputs. Candidate-record descriptors add real signal evidence but do not supply missing health labels. Seed variability and temporal block variation cannot be treated as a population confidence interval. The normalization is inherited from supplied metadata; its historical fitting population is not reconstructed from these files. It is used consistently but cannot retrospectively establish a train-only provenance for the original scaling.

Architectural future leakage is prevented through past-only windows, unidirectional recurrence and left-only convolution padding. Offline diagnostic Hilbert transforms/interpolation must not be repurposed as causal online predictors without a new design. The shifted-speed negative control intentionally violates the primary information-availability policy and is excluded from eligible model selection. Exact Spa LPV-ARMA superiority, physical gearbox parameters, fault severity and GPU efficiency remain unsupported. These are limitations of the available evidence, not quantities that can be inferred from attractive plots.

## 15. Answers to the research questions

**RQ1:** The best eligible observed common-index validation run is `lpv_L256_s17_a1`, MSE **0.361217**; final test superiority is unknown. **RQ2:** Saved training times, full-record inference repeats, parameter counts and artifact bytes quantify the local CPU trade-off. **RQ3:** The speed-conditioning hypothesis is assessed by measured LPV/ARX and neural controls; effects are conditional on seed, context, capacity and this near-repeated trajectory. **RQ4:** real raw/residual and assumption-labelled angular descriptors were measured for 200 separate candidate records, while physical fault accuracy remains unidentifiable without labels. **RQ5:** Context, seed and feature-convention experiments bound some variability, while independent-record prediction generalization and physical fault sensitivity remain untested.

## 16. Reproduction and deliverable map

The executable project uses importable `src/rotmach` modules; the comprehensive notebook orchestrates them. Model weights, per-run configs, histories, common-index prediction arrays, metric JSON, registry CSV/JSON and SHA256 manifests are retained. Read [the implementation guide](../docs/IMPLEMENTATION_GUIDE.md), [cross-platform instructions](../docs/reproduce_windows_linux_macos.md), [study companion](../docs/COMPREHENSIVE_STUDY_COMPANION.md) and [reproduction status](reproducibility_check.md). Commands actually executed are distinguished from documented Windows/macOS equivalents. The editable progress/final PPTX files and speaker scripts are generated by `scripts/build_presentations.py` from the same CSV metrics used here.

## References and appendices

Verified records and evidence levels are maintained in [references.bib](../research/references.bib), [literature_matrix.csv](../research/literature_matrix.csv), and [research_synthesis.md](../research/research_synthesis.md). The lecturer citation is course-source verified; restricted publisher access is not described as full-text review. Author repositories and scientific-library documentation support implementation details within their stated scope. Exact model source: [models.py](../src/rotmach/models.py), [experiment.py](../src/rotmach/experiment.py), [windows.py](../src/rotmach/windows.py). Diagnostics and numerical conventions: [diagnostics.py](../src/rotmach/diagnostics.py), [angle_resample.py](../src/rotmach/angle_resample.py), [spectral.py](../src/rotmach/spectral.py).

The full run table above is the numerical appendix. Additional source-linked outputs include [baseline comparison](baseline_comparison.csv), [ablations](model_ablation.csv), [timing methodology](timing_and_hardware.md), [scientific validity review](scientific_validity_review.md), and [rubric traceability](../docs/rubric_traceability_matrix.md). Missing evidence in the acceptance checklist remains unchecked; physical fault-detection accuracy cannot be verified without health labels, and exact paper reproduction requires additional method access.
