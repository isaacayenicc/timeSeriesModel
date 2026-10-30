# Critical research synthesis: causal vibration prediction under varying speed

## 1. What this review establishes

The engineering question is narrow: predict the next supplied normalized vibration sample from past observations, compare errors on identical validation indices, measure computational cost, and study whether prediction residuals support interpretable diagnostic statistics. A method's success in language modeling, visual reasoning or another machine is a reason to test it, not evidence that it will succeed here.

Evidence is classified before drawing conclusions. The expanded [publication matrix](publication_matrix.csv) contains 19 publication records: 12 machinery-diagnostics sources (R08–R19), three architecture/conditioning sources (R02–R04), three software citations (R05–R07), and the course-cited but inaccessible anchor (R01). Nine official implementation sources and an unverified candidate are separate. The 15 method-relevant author/abstract records are not 15 full-text readings: R19 is limited citation/code context, most machinery material is an original author README abstract/method summary, and **no article full text was read**. Source metadata, year discrepancies, access dates and hashed bytes are retained. The expanded review improves direct domain coverage while leaving exact lecturer-method, historical order-tracking and MCKD/PMI full-method gaps explicit.

The runtime denied the Crossref, arXiv, MathWorks and ACL web destinations with HTTP 403 at the proxy. Original GitHub-hosted documentation and the ACL publisher's metadata repository were accessible. [The access log](source_access_log.md) records successes, denials, SHA-256 hashes and a corrected documentation URL. Draft allowlist additions have been saved by the coordinating agent, but a draft does not establish runtime access or that blocked sources have been read. This limitation bounds every literature-derived claim below.

## 2. The conventional anchor and the reproduction boundary

The course PDF names Chen, Li, Jiang, Gong and Zhou, “Sparse LPV-ARMA model for non-stationary vibration representation and its application on gearbox tooth crack detection under variable speed conditions,” *Mechanical Systems and Signal Processing* 224, 112161 (2025), DOI 10.1016/j.ymssp.2024.112161 [R01]. It also states that the course data are the same as those used in the article. These statements are attributed to the course brief, not independently confirmed by the article.

General LPV modeling lets coefficients depend on an observed scheduling variable. General ARMA modeling includes past output and innovation terms. Those definitions explain why a speed-dependent predictor is plausible, but do not establish the paper's actual scheduling basis, lag orders, sparsity penalty, coefficient selection, estimation procedure, innovation recursion, preprocessing, folds, fault labels or reported scores. All those article-specific details remain unknown on the material accessible here. They must not be reconstructed from the title or from an appealing mathematical guess.

A polynomial speed-conditioned AR model is thus a **related LPV-AR surrogate**, not Spa LPV-ARMA. One possible causal form is

\[
\widehat{x}_t=b+\sum_{j=1}^{L}\sum_{q=0}^{Q}a_{jq}\,s_{t-1}^{q}x_{t-j},
\]

where \(x\) is the supplied dimensionless vibration and \(s\) is supplied normalized speed. Ridge regularization stabilizes the expanded design. This model has no moving-average innovation recursion and no demonstrated equivalent sparse estimation scheme. Its comparison with ordinary AR asks whether simple scheduling interactions help these files; it does not reproduce the lecturer's algorithm. No “outperformed conventional Spa LPV-ARMA” conclusion is warranted until both procedures use the same files, normalization, prediction regime, warm-up, metric and target indices.

A later exact-reproduction task should obtain the full article and supplement, identify all parameter-selection steps, distinguish reference-condition training from fault-record inference, confirm speed units, and reproduce an article table before comparing new methods. The present study can still establish reliable local baselines and candidates without making that unavailable comparison.

## 3. Why linear prediction is indispensable

A regularized AR model is a difficult baseline to dismiss on only 24,000 training samples. A lag matrix already exposes short-time oscillatory structure. Linear least squares estimates a compact conditional expectation approximation and is fast to fit and inspect. In scikit-learn's implementation [D01], the objective is

\[
\|y-Xw\|_2^2+\alpha\|w\|_2^2,
\]

with an intercept handled separately. This sum-loss convention matters: an implementation using average MSE plus \(\lambda\|w\|^2\) has corresponding \(\alpha=n\lambda\), before any additional feature-scaling differences. Identical numerical regularization constants do not imply identical regularization across these conventions.

Zero, empirical training mean and persistence predictors answer distinct questions. Zero is meaningful on a supplied approximately centered signal, but does not equal the training empirical mean automatically. Persistence can perform poorly on rapidly oscillating signals despite appearing intuitively plausible. An OLS AR establishes unregularized fit behavior; ridge shows whether coefficient shrinkage helps conditioning and validation error. Adding causal speed history alone gives an ARX-like additive predictor; multiplying vibration lags by speed basis functions changes the coefficient dependence and should be labeled separately.

This ordering prevents a capacity argument from masquerading as a scientific contribution. A deep model that fails to improve ridge MSE may still offer another defensible property, but architecture names and parameter counts alone are not evidence of utility.

## 4. Advanced causal models: what transfers and what does not

Bai, Kolter and Koltun's author repository [R02] confirms the TCN comparison spans synthetic memory tasks, sequential/permuted digits, music and language. The inspected code contains residual blocks with two dilated convolutions, activations, dropout and right-tail cropping after padded convolution. These are concrete implementation facts. No gearbox benchmark score was extracted or transferred.

A causal convolution's availability must be interpreted relative to the predictor's input. If the window ends at \(x_{t-1}\), a convolution may include that last input while still predicting \(x_t\) causally. Symmetric padding without correct cropping can include later positions *within a sequence-to-sequence computation*; future-perturbation tests are therefore more persuasive than describing a network as “causal.” For a stack with kernel width \(k\), dilation \(d_l\) and one convolution per layer, the receptive field is \(1+\sum_l(k-1)d_l\). Two convolutions at each level double that sum. The implemented architecture, rather than the generic TCN name, determines the appropriate formula.

A GRU offers gated state updates with parameter sharing along time. Cho et al.'s publisher abstract [R03] concerns phrase representations for statistical machine translation and says an RNN encoder-decoder's conditional probabilities improve the translation system. This verifies historical context and the task, not machinery effectiveness. A local unidirectional GRU must be trained on the same causal windows, with hidden-state resets and output indexing defined. A compact MLP lag predictor controls for nonlinear function approximation without recurrence or temporal convolution.

No large attention or state-space model is necessary merely to satisfy “advanced machine learning.” Small-data overfitting, limited independent operating histories, hyperparameter-selection reuse and runtime are more material here than adopting the newest model family. Optional expansion should follow a concrete deficiency in the compact suite, not displace completed baseline, reproducibility or diagnostic work.

## 5. Speed conditioning: a falsifiable project-level adaptation

FiLM is established prior art. Perez et al.'s author repository [R04] identifies the AAAI 2018 visual-reasoning work and implements feature-wise modulation directly as

\[
\operatorname{FiLM}(h,s)=\gamma(s)\odot h+\beta(s).
\]

The original setting conditions visual features on questions. Applying the same mechanism to speed-conditioned temporal features is a project-level adaptation, not the invention of FiLM, LPV modeling, or speed-aware machinery monitoring. Literature access is insufficient to establish world-first novelty of any speed-TCN design.

The motivation is mechanical and statistical: changing rotational speed can shift repeatable excitation in time frequency and alter apparent lag dependence. A model might share a small temporal representation while adapting gains and offsets to the operating point. This is an inductive bias, not an assurance that measured prediction improves. Nearly repeated train/validation speed trajectories would also make a time-correlated operating-point input a weak test of transfer to genuinely new operating histories.

A meaningful comparison uses the same temporal core and comparable optimization budgets. Compare vibration-only TCN, late speed concatenation, and FiLM when feasible; report actual parameter counts. Include a nearby capacity control, since adding a conditioning network also adds parameters. An intentionally phase-misaligned or shuffled-speed control tests whether alignment contributes, but is an intervention with a distribution shift; a single degraded control result does not prove causal physical identification. Similarly, equal MSE with and without speed can reflect redundant vibration information, inadequate optimization, small-data uncertainty or an ineffective conditioner, rather than proving that physical speed is irrelevant.

At least three lag contexts and repeated seeds provide a modest robustness check. Seeds estimate optimization variability on these files, not machine-to-machine generalization. Parameter-matched comparisons are approximate unless counts are equal; report the mismatch openly rather than claiming perfect control.

## 6. Evaluation design is part of the model

The official scikit-learn time-split implementation [D02] preserves order and supports a gap between training and evaluation. The lagged-feature example [D04] illustrates why temporal evaluation matters. Neither source makes overlapping windows independent. For this project, the official training and validation files should remain separate. If internal tuning uses a split of training, explicitly validate raw-sample support around the embargo: a gap in target indices only protects context overlap if it is at least as large as the actual context requirement.

The preprocessing guidance [D03] explains how information unavailable at prediction time can enter through fitted transformations. Supplied normalized fields are used without per-file restandardization, but their provenance is still a limitation: numerical reconstruction from metadata verifies the transform, not which historical data were used to estimate the metadata. Angular resampling, FFT-derived features and centered smoothing must not be slipped into causal predictors as if they were past-only operations.

Every compared MSE should use the same target set, for example \(t\ge L_{\rm eval}\), where the declared common burn-in covers all contexts. The fact that a 32-lag model can predict earlier targets does not justify giving it additional easier samples in the main ranking. Save target indices, true samples, predictions and residuals so alignment is reviewable rather than inferred from a summary table.

Validation used for early stopping or model ranking is a selection set. Its resulting MSE is a provisional validation result and carries selection optimism. Missing `sigtest.mat` cannot be repaired by renaming validation to test. A block bootstrap quantifies sensitivity along one observed trace conditional on the method; it cannot create independent machines, runs or speed trajectories. Autocorrelation and block-length sensitivity should guide its interpretation, with no sample-wise independence fiction.

## 7. Residual diagnostics and the prediction–detection tension

The supplied MATLAB routine computes RMS, crest factor, time-domain kurtosis, correlated kurtosis and envelope-autocorrelation PMI. That code is an inspectable source for formulas and indexing; it is not evidence that source speed is RPM, that all records are healthy/faulty, or that a diagnostic threshold is validated.

A reference-trained predictor can reduce repeatable background, leaving transient or periodic residual structure. The opposite outcome is also plausible: a flexible teacher-forced predictor may predict some abnormal pattern and remove it from the residual. Lower MSE therefore does not logically imply better physical fault detection. Raw and residual statistics must be compared on equal burn-in and segment length, record by record, without retraining on candidate fault records for the principal comparison.

RMS measures signal energy scale, crest factor measures peak relative to RMS, and Pearson kurtosis measures a fourth-moment shape. SciPy's documented distinction [D05] makes the convention explicit: `fisher=False` requests ordinary rather than excess kurtosis, and `bias=True` uses the moment-ratio estimator. Constant-signal and zero-energy cases require deliberate validity flags, not silent zeros that resemble healthy evidence. None of these statistics is fault-specific by definition.

Hilbert-envelope computation uses an FFT analytic signal [D06]. Its finite-record periodic extension creates edge effects and does not isolate a physically meaningful modulation band automatically. The supplied PMI chooses an envelope ACF coefficient near an assumed period and transforms \(\rho\) to \(\rho/(1-\rho)\); it can be negative and becomes unstable near one. A large PMI without an independently justified period is a descriptive periodicity measure, not a confirmed crack label. The MATLAB one-based array index for lag selection must be compared with a physically corrected zero-based lag explicitly, rather than quietly merging versions.

Order tracking requires speed-to-cycles dimensional consistency. Treating speed in revolutions/second, RPM or radians/second yields different cumulative cycles and therefore different angular grids and presumed lag periods. Where unit metadata are absent, all such outputs are assumption/sensitivity studies. A synthetic known-speed signal validates numerical integration/interpolation and order recovery; it does not identify the real unit or the true fault order. The external `toConstantSamplesPerCycle` implementation is not supplied, so exact output-grid or `fsp` parity is unknown. Live MATLAB parity is also unverified.

The Antoni–Randall spectral-kurtosis paper [U01] is a future domain-reading candidate supplied in the master prompt, not accessed literature. Spectral kurtosis must not be conflated with the routine's ordinary time-domain kurtosis or used as an assumed definition of PMI. The inaccessible Chen article may explain the intended PMI context, but that information cannot be attributed before reading it.

## 8. Timing and reproducibility standards

Official PyTorch reproducibility guidance [D08] distinguishes fixed seeds from universal numerical reproducibility. Library versions, backends, threads, deterministic settings and hardware must accompany the experiment. Multiple seeds should be retained, not cherry-picked to report the most favorable member as a stable result.

Timing source [D09] supports warmup, repeated measurements and accelerator synchronization. A practical comparison still needs locally stated boundaries. Fitting time, lag-matrix construction, inference, and feature extraction should be separate quantities. Batched throughput does not equal single-step deployment latency. A small linear model may win on CPU inference even if a deep model has lower MSE. GPU timing cannot be invented when only CPU was used; if GPU is used, asynchronous dispatch without synchronization is not completed work.

The local source snapshots and their hashes make the research record inspectable even though mutable upstream URLs may change. Exact installed package versions belong in the environment lock; main-branch documentation does not override runtime tests. Checkpoints, effective configs, hashes and saved predictions permit re-evaluation without repeating optimization. A narrative conclusion should link to the run registry and common-index table rather than rely on a hand-entered presentation number.

## 9. What evidence would change the conclusion

The model-selection conclusion must come from the actual saved validation/timing tables; this review intentionally contains no invented local winner or external same-data score. The proposed suite is justified because it separates linear predictability, nonlinear capacity, recurrent versus convolutional representation and explicit operating-point conditioning. Each added model answers a controlled question.

Three unavailable inputs materially limit interpretation: the lecturer's full methods and benchmark, an independent test trace, and validated physical fault labels. The subsequently supplied twenty-part fault dataset is now available and analyzed; the file-transfer obstacle is resolved. Obtaining the full paper would permit exact methodology review and perhaps reproduction. New speed histories would probe operating-condition transfer. Physical labels with provenance would permit record-level threshold evaluation, sensitivity, specificity and ROC analysis after a declared training/calibration split. Neither a visually striking feature trajectory nor a synthetic impulse test supplies those labels.

A credible final discussion can report negative or inconclusive speed-conditioning effects, a strong ridge baseline, model-dependent residual statistics and bounded computational trade-offs. These are useful measured findings. Unsupported novelty, unverifiable benchmark victory or inferred fault accuracy would reduce the scientific value of an otherwise reproducible project.

## Source navigation

- [Evidence-qualified matrix](literature_matrix.md) and [CSV](literature_matrix.csv).
- [Verified bibliography records](references.bib); citation-only and unverified candidates are deliberately excluded.
- [Access log](source_access_log.md) and [machine-readable hashes](evidence/access.json).
- [Model-selection rationale](../docs/METHOD_SELECTION_RATIONALE.md).
- The final report and companion should cite source IDs above and the saved experiment tables independently; source verification is not experiment execution.


## 10. Expanded domain evidence and what it changes

### Continuously varying speed and transfer

R08's original dataset/author material distinguishes continuous acceleration and deceleration from separate fixed-speed domains. That distinction matters here: train and validation almost repeat a speed trajectory, while the 200 candidate records add descriptive record variation without verified health labels. R09's transfer-learning library and survey describe source/target adaptation protocols. They motivate future transfer studies, but adapting to the candidate set would change the present reference-only diagnostic question. No such transductive refit is hidden in this study.

### Sparse impulses, periodic structure and background suppression

R10 describes multiscale dictionary learning for impulsive structure amid harmonic/noise interference. R15 describes within-/across-period sparse group priors; R16's B-spline sparsity and R17's SALMA sparsity-assisted reconstruction describe other structured decompositions. R18 adds nonconvex low-rank plus weighted sparse decomposition. These sources share a concern with separating relevant structure from interference, but their objectives are not the same as minimizing next-sample MSE. The distinction helps explain why a predictor that removes more energy can also remove periodic content, and why neither larger residual kurtosis nor PMI alone proves better diagnosis. Several sources share an author group, which limits diversity; author-reported findings were not independently replicated. R19's intelligent time-frequency sparse method is only citation/code context and receives no unverified detailed algorithm claim.

### Interpretability, limited labels and timing

R12 embeds interpretable time-frequency kernels, an example of a physical inductive bias. Such architectural explanation does not establish causal indexing or identify units in an unrelated dataset. R13's few-shot setting has a few verified labeled examples, whereas the present candidate set has no verified health labels; its supervised scores cannot be imported. R11 explicitly treats acquisition duration and online processing as separate delay components. We consequently report lag duration, public inference time and feature extraction separately; fast offline batches do not establish streaming alarm delay. R14's benchmark makes split/augmentation and training protocol explicit, supporting our disclosure of validation reuse, limited training budget, capacity controls and sample-index fairness.

### Relation to the observed results

The original validation experiment favors the polynomial LPV-AR surrogate; FiLM does not improve mean lag-64 TCN MSE in three seeds. The full candidate study compares the same frozen family representatives on equal recording segments. These measurements adjudicate the local design hypotheses; the cited repository summaries do not guarantee them. Full article access and a directly comparable Spa LPV-ARMA reproduction remain necessary before any published-method superiority claim. Unknown units and labels still prevent physical calibration and classification-performance claims even though the input-transfer problem is now resolved.
