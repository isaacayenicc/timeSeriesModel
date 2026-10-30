# FAQ and viva preparation

Use this document with [the study companion](COMPREHENSIVE_STUDY_COMPANION.md) and the saved result tables. Explain the protocol before quoting a winning model. Model rankings are provisional validation findings, not independent test results.

## Explain the task in one minute

We predict the next supplied normalized vibration sample using only previously observed vibration and, for selected models, past normalized speed. We compare simple predictors, regularized linear autoregression and compact neural architectures on the same validation target indices. We measure MSE and CPU cost, preserve model artifacts and verify reloads. The verified 20-part package now supplies 200 real candidate records for descriptive residual analysis, while physical fault labels and speed units remain unverified. We therefore do not claim measured crack-detection accuracy or superiority over a reproduced Spa LPV-ARMA benchmark.

## Questions about the data

**What does one row represent?** One time sample, not an independent machine or fault case. A lagged supervised row aggregates past samples; neighboring rows overlap strongly.

**How long are the recordings?** Each accessible train/validation trace has 24,000 samples at 3,200 Hz, giving 7.5 seconds by sample count. First timestamp is 1/3,200 s and last is 7.5 s. Last-minus-first is therefore one sampling interval less than the sample-count duration.

**What are the units?** `x` is dimensionless supplied normalized vibration. `xraw` is in undocumented source amplitude coordinates. `spd` is in source speed units. We cannot label either as SI acceleration or rpm without additional evidence.

**Why are there three `xmean` values?** That is the stored array shape. We verified only that the first entry reconstructs the supplied `x` transform; we do not invent the other entries' semantic meaning.

**Is the training data healthy?** It is the supplied reference/training condition. Its healthy status is not independently established by accessible metadata.

**Did you inspect 200 fault records?** Yes. The later complete-record package resolves the original large-file transfer problem. Twenty parts contain ten records each; checksums, logical field hashes, ranges and ordering are checked against the supplied manifest. Each record remains separate throughout inference and feature extraction. This establishes data availability and descriptive results, not verified health labels.

## Questions about scientific design

**Why no random train/test split?** Overlapping windows from one trace share observations and operating conditions. We retain the supplied train/validation separation. Training minibatches may be shuffled without changing that split.

**Is validation an unbiased final test?** No. Validation is used for development and checkpoint selection. No `sigtest.mat` was supplied, so the study reports validation performance.

**What is the primary target set?** Zero-based target indices 256 through 23,999 when using common burn-in 256. That gives 23,744 targets. Check saved prediction indices before using this count for any separately configured experiment.

**Does strong train/validation speed correlation prove leakage?** No. It limits the challenge of operating-path transfer and warrants checking duplication, but correlation alone does not prove prohibited information sharing.

**Why not normalize every record independently?** That could erase amplitude differences relevant to diagnosis and changes the supplied scale. We preserve the provided `x` and speed scaling.

**Do three seeds provide a confidence interval for new machines?** No. Seeds measure training stochasticity on the same data. Independent machines or acquisitions require separate recordings.

## Questions about algorithms

**Why begin with ridge AR?** It is fast, interpretable, well matched to repeated oscillatory dependence and a strong benchmark. Deep learning must demonstrate a benefit against it rather than only against zero.

**How is ridge different from ordinary least squares?** Ridge penalizes coefficient norm. It stabilizes a collinear lag design at the cost of shrinkage bias. The numerical penalty depends on feature scaling and whether loss is summed or averaged.

**How is ARX different from LPV-AR?** An additive speed input changes a prediction offset; speed-by-lag interactions let lag coefficients change with speed. Neither automatically reproduces a full LPV-ARMA innovation model.

**Why GRU?** It learns a compact recurrent representation of ordered history. It offers a different temporal inductive bias from a fixed lag MLP or convolution. Recurrence can make CPU inference comparatively costly.

**What makes TCN causal?** Past-only input construction, correctly oriented convolution operations and adversarial future-perturbation tests. A model class name does not prove this property.

**What does FiLM do?** It maps known speed into gains and offsets on learned vibration channels. It changes the interpretation of features; it does not perform physical order tracking.

**Is the novelty globally new?** No such claim is made. The project contribution is a controlled application and evaluation of compact speed conditioning with transparent residual analysis. FiLM, TCN and LPV concepts already exist.

## Questions about results

**Which model should we deploy?** Use the saved accuracy–cost comparison and the intended constraints. Lowest validation MSE and lowest CPU latency may select different models. Broader deployment also needs independent operating conditions and diagnostic labels.

**What if the linear model wins?** That is a valid result. It may indicate adequate linear structure, a short training trace, easier optimization or neural overfitting. It does not establish that all vibration is globally linear.

**What if speed does not help?** Past vibration can already encode operating speed; the conditioning path may be weakly learned; the short dataset and similar trajectories may limit evaluation. These remain possible explanations, not proven causes.

**Why does lower MSE not guarantee better diagnosis?** A flexible teacher-forced predictor may explain predictable fault patterns and remove them from the residual. Diagnostic utility requires separate validation.

**Have you beaten the lecturer's method?** No matched reproduced Spa LPV-ARMA benchmark is available. Results against a related LPV-AR surrogate are labeled as such.

## Questions about diagnostic mathematics

**How does ordinary kurtosis differ from excess kurtosis?** Excess subtracts three. A Gaussian population has ordinary kurtosis three and excess zero. Finite-sample bias conventions matter.

**Why is PMI indexing delicate?** MATLAB's first ACF element is lag zero, so `acf(T)` refers to lag T−1. Literal and physically intended lag selections can differ by one sample. Both definitions must be labeled.

**What does a large PMI prove?** Strong selected envelope autocorrelation under the chosen convention and assumptions. It does not alone establish fault type, damage severity or detector sensitivity.

**Why not choose rpm as a reasonable default?** rpm, revolutions/s and radians/s imply different cycle counts and lag locations. Choosing silently can manufacture an apparently persuasive order-domain result.

**Why are angular outputs not sampled in Hz?** The new independent coordinate is cycles, so density is samples/cycle. A lag of one cycle corresponds to that density in output samples.

**Why no F1 or ROC?** Neither verified physical labels nor a valid independent threshold protocol is available. The 200 real records support descriptive paired comparisons, but record order is not ground truth.

**Why are parts not additional conditions?** The twenty parts are a storage/transfer partition of complete records. Their filenames preserve order; they do not define fault classes or independent acquisition protocols.

**Why compare equal burn-in?** All raw/residual features use indices 256–31,999, retaining 31,744 samples per record. Different retained lengths would change energy and periodicity statistics and make cross-model comparisons harder to interpret.

**Can feature trends identify a fault onset?** They can motivate a hypothesis, but without labels or an acquisition log they cannot verify an onset, crack severity or health transition. Speed and other unmeasured operating conditions can also alter features.

## Five-minute rehearsal exercise

1. Draw a five-sample signal and construct the first causal lag window by hand.
2. Explain the difference between teacher-forced one-step prediction and free-running forecasting.
3. Derive the ridge normal equations and state the intercept convention.
4. Calculate a TCN receptive field from its actual kernel sizes and dilations.
5. Show the saved result CSV, its prediction indices and the checkpoint that generated it.
6. Convert 600 rpm to cycles/s and explain why the supplied speed cannot yet be treated as 600 rpm.
7. Explain one negative or blocked finding without replacing it with an unsupported claim.

The most credible defense states what the evidence establishes, what it does not establish, and what additional measurement would resolve the uncertainty.
