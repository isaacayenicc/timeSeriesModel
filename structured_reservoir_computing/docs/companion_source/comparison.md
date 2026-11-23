## 17. Twenty-one comparator families

The comparator mechanisms below describe models implemented under `src/src_gearbox/models/baselines`. The 21 checkpoint identities are preserved from the earlier course study and replayed with the current target mask. Selected refitted controls remove speed variation. Claims about fitting and selection stay with the actual lineage in `results/metrics/model_coverage.csv`, `baseline_configs.csv`, and `baseline_prior_trials.csv`. New primary-model comparisons use PI-SRC fit in this work. Source policy, masks, target values, and causal verification are independent checks from the original fitting evidence.

### 17.1. AR

A vibration-only autoregression fits a linear map from $p$ observed delays to $x_t$. Its forecast is $b+\sum_{j=1}^p a_j x_{t-j}$. Such a model offers a useful speed and data-volume anchor, especially if coefficients are fitted by least squares or light ridge on an admissible linear design. If a sequence behaves approximately as a stationary damped process within a small operating region, a few AR coefficients can approximate a local linear recurrence. The same coefficients must cover the entire RPM range. Poor moving-harmonic agreement is one reason to consider conditioning. The stored result identifies selection and inference context, rather than assuming that a stationary model reconstructs the physical process.

### 17.2. ARMA

ARMA adds a moving-average term built from previously reconstructed innovations, as in Eq. (6). This lets previous surprising observations inform current error without requiring an arbitrary order increase in measured lags. Conditional fitting yields a computable causal predictor but requires startup and invertibility conventions. The comparator’s finite conditional implementation checks that truncation agrees with the expected recursion and that rejected configurations remain recorded. Excessive recursion memory would invalidate a previously chosen guard. Evaluating its bridge under the common mask checks that the expected warm-up covers the actual model.

### 17.3. ARIMA

ARIMA introduces integration order $d$ by fitting an ARMA model in differences. In this report the task is in vibration levels. In first differences, reconstruction adds predicted difference to $x_{t-1}$, a valid observed feedback. A model need not represent a random walk simply because differencing is available. A trend hypothesis should survive training-only folds, and numerically unstable earlier innovative recursions should be rejected. Its fit can cost more than AR even when it does not improve MSE. The released rows identify status and applied configuration.

### 17.4. ARX

The additive exogenous model augments measured lags by historical RPM lags $n_{t-d_1},\ldots,n_{t-d_k}$. It estimates which part of the target correlates with the exogenous context after accounting for vibrations. It does not automatically change the dynamics of the vibration-lag coefficients with RPM. The speed-withheld control fits an otherwise matched configuration with constant nominal RPM, so it measures the added value of observed variation. It is a stronger comparison than just replacing RPM at inference in a fitted speed-dependent learner, which can create out-of-distribution effects. Interpret the resulting difference alongside the specific design and actual training mask.

### 17.5. NARX

NARX replaces the additive mapping by $g_\theta(x_{t-1:t-p},n_{t-1:t-q})$ with a nonlinear part. In this project a physics-conditioned ridge backbone is supplemented by a tanh hidden block predicting its residual on compact features. The output is one-step and receives measured historical vibration as feedback. The residual head can add effects not expressible by an affine readout of the compact raw features, but it can also fit pattern coincidences in one recording. Compare training and development error. A lower training objective alone leaves the transfer of the extra capacity uncertain. Recursive feed of its own predictions would define a different forecasting experiment.

### 17.6. Sparse LPV-AR

A linear parameter-varying AR model uses coefficients $a_i(s)=\sum_k\gamma_{ik}P_k(s)$, with a previous normalized RPM scheduling coordinate $s$. Each lag-basis product forms one design column. With 240 lags and eight basis values, 1,920 candidate columns are possible. Sparse selection prunes many weak products and refitting supported columns reduces shrinkage bias. The comparator retains the guarded support lineage and masks of the established strict reproduction. Its coefficient counts reflect selected nonzero columns, not a claim that all basis terms represent independently verified physical mechanisms.

### 17.7. Sparse LPV-ARMA

The lecturer reference augments speed-varying AR with speed-varying proxy-innovation terms from a first-stage fit. Both coefficient blocks respond to the measured schedule. The innovation representation makes inference differ from an exact fitted state-space model. Count both stages in the deployed footprint. Verify that the first stage operates on the permitted training subset and that its predicted residuals are only used after the relevant observation. This is the principal strict conventional reference in the prediction comparison. The published original test values, earlier reproduction, and strict refit are traced separately in Chapter 21.

### 17.8. Linear Regression

The affine predictor $b+w^Tz_t$ solves least squares. A design with speed products and oscillator summaries can carry useful physical nonlinearity despite the linear readout. This comparator matters because PI-SRC also uses an affine readout of constructed features. Any advantage could come from how the basis, lag history, penalties or innovations are chosen. A stable solver handles similar columns, but unregularized coefficients can extrapolate sharply beyond observed states. Inspect the chosen design, effective rank and held-out performance.

### 17.9. Ridge Regression

Ridge adds squared coefficient penalties, Eq. (7), to the same score-fitting task. It reduces sensitivity to near-collinear directions. Larger $\lambda$ pulls weights toward zero and can underfit a strong narrow-band regularity. Tune on guarded training only and use a penalty convention consistent across candidates. The retained intercept often uses a separate, unpenalized direction. For a speed-conditioned ridge model, meaningful comparisons require checking whether angular lags and order features enter its library. Its deployment runtime may be dominated by that library.

### 17.10. Lasso Regression

Lasso adds an absolute-value penalty and can return exact zeros. If one important feature is represented by a correlated collection of lags, Lasso may select one representative. Its choice can be unstable across folds. Sparsity can reduce footprint under an inference backend that evaluates only retained coordinates. A checkpoint storing a sparse coefficient vector beside a dense feature constructor may have speed similar to a dense readout. Interpret stored weights, inference work and selected support separately.

### 17.11. Elastic Net

Elastic Net mixes L1 and L2 penalties. It can retain correlated features with distributed weight while removing directions receiving little predictive support. This matters for vibration designs with nearby lags and recurring phase history. Compare both mixing fraction and total penalty at common objective scale. If a solver reaches its stopping cap, look at recorded convergence, not only the apparent MSE. A sparse solution can be easier to inspect but does not automatically yield a better causal model.

### 17.12. Support Vector Regression

Epsilon-insensitive loss penalizes errors outside a tolerance tube. A radial basis kernel $K(z_i,z_t)=\exp(-\gamma\lVert z_i-z_t\rVert^2)$ defines nonlinear responses centered on support vectors. The prediction sums kernel values times signed fitted coefficients and includes an intercept. Large $\gamma$ makes the influence of each fit example local, often problematic when closely sampled windows from different speeds cover the design unevenly. The training support is capped deterministically for cost and the evaluation mask remains full. Report both reduction and runtime. A compact input dimension does not guarantee low inference cost when the support set is large.

### 17.13. Random Forest Regression

A forest averages regression-tree predictions across resampled fit examples and feature choices. Thresholds create local regions of delay/speed feature space. This can capture interactions without a prescribed hand-written polynomial product, at the cost of stepwise response and high sample demand. Nearby state vectors may cross a threshold and give discontinuous predictions. Guarded time splits are applied outside tree fitting, since randomly subsampling already overlapping row histories would underestimate dependence. Deep trees can fit the training record and still generalize poorly. Include tree nodes and serialized bytes so comparisons with weights are intelligible.

### 17.14. Gradient Boosting Regression

Boosting builds an additive sequence of trees to follow remaining loss gradients or residuals. At squared loss a new stage approximates $y-\hat y_m$ subject to its tree capacity. A learning rate scales additions, and the iteration budget determines the total number. Histogram splitting can reduce fit cost. It leaves the same limitation on representing a smoothly varying angular trajectory through many piecewise thresholds. Monitoring a fixed contiguous guarded block supports evaluation. An internal random validation split would not be a physically independent record.

### 17.15. Multilayer Perceptron

The MLP consumes a flattened vector of historical vibration and permitted RPM. Hidden affine maps with activation functions can approximate smooth nonlinear terms. A raw-history shortcut is useful because the strong AR-like relationship is already nearly linear in many coordinates. That shortcut and hidden branch solve a flexible but still fixed-context problem. Every window index identifies a specific raw relative lag. Reordering those features changes the function and must agree with checkpoint serialization. Short-record evidence should be combined with train-only regularization, a finite configuration count, and seed checks where available.

### 17.16. LSTM

An LSTM uses input, forget and output gates to control a memory cell. With past window $z_j$, the remembered state follows a rule like $c_j=f_j\odot c_{j-1}+i_j\odot\tilde c_j$. The bounded historical encoder reads measurements through $t-1$. Sequence resets or fixed-window setup prevent carrying a hidden state across acquisitions. More hidden units add capacity and cost. For one short recording it can be hard to distinguish a learned useful temporal memory from a flexible fit to the observed operating pattern. Its recurrent update also offers less straightforward batch parallelism than temporal convolutions.

### 17.17. GRU

A GRU uses an update and reset gate to combine a proposal and retained state without a separate LSTM cell. It can supply comparable seasonal or transient modeling with different capacity. The same bounded past window and reset policy apply. A comparison with LSTM is meaningful only in the context of chosen widths, sample exposure, optimization and seed variability. Neither architecture is given an advantage by a bidirectional pass through targets. The final scalar predictor is scored on the same measured indices.

### 17.18. CNN

A one-dimensional temporal CNN learns filters applied across historical samples. Locality and shared weights help recognize repeated short patterns and need fewer distinct coefficients than a dense lag map for the same number of positions. Its receptive field is set by kernel width and layer organization. Output positions used in causal inference must access historical samples only. A CNN can learn phase-local features without explicit mechanical frequencies, making it a useful representation comparator. Behavior may deteriorate when the operating speed changes substantially unless speed conditioning and learned filter diversity cover the phase changes.

### 17.19. TCN

A TCN uses finite causal convolutions, often with dilation. For layers with kernel lengths $k_i$ and dilations $d_i$, unit stride gives total support $1+\sum_i(k_i-1)d_i$ for a single convolution per layer. Additional serial stacks add their support. This enlarges context efficiently and can model nonlocal patterns without recurrent state. Verify the implementation's left padding and readout position. A longer context is still a capacity choice and can add cost or exacerbate scarcity of distinct contexts. The common matched evaluation does not infer robustness solely from architecture.

### 17.20. Time Series Transformer

Historical vibration and speed can be grouped into patches with position encoding. Masked self-attention combines token values with weights from scaled query/key similarities, $\mathrm{softmax}(QK^T/\sqrt d)V$. The scalar readout is trained for one-step prediction from past tokens. Patching reduces token count and attention computation, but reduces positional detail. Dense attention cost grows quadratically with token count. Its many expressive relations can be hard to identify from one short acquisition. Benchmark these actual selected settings without treating the result as a judgment about every Transformer.

### 17.21. Temporal Fusion Transformer

The compact TFT adaptation combines historical variable selection, gated residual blocks, a recurrent projection and attention. It is scoped to an available history plus a scalar next observation. Future RPM and future vibration are unavailable. A variable gate can be inspected as activity within this computation graph. Gate weights cannot establish a physical causal effect or assign confidence to a model. Poor performance may reflect data size, selected context, stride, hidden width and optimization together. Preserve the actual configuration and make this scope explicit when discussing the observed comparison.

### 17.22. Interpretation across families

The strict benchmark mask holds target units and positions equal, while architecture, requested covariates, fit subset and computational budget can vary. One matched speed-withheld experiment preserves input tensor shape and changes measured RPM to a constant nominal value during both fit and evaluation. The measured-speed version can benefit by exploiting variable context. It can also overfit a limited speed association. Observed benefits for several families do not imply monotonic benefits across all methods. The numerical appendix gives the measured deltas and exact artifact paths.

## 18. Evaluation metrics, timed scope, and uncertainty

### 18.1 Scales and definitions

For $T$ common target samples define

$$
\mathrm{MSE}=T^{-1}\sum_t e_t^2,\quad
\mathrm{RMSE}=\sqrt{\mathrm{MSE}},\quad
\mathrm{MAE}=T^{-1}\sum_t|e_t|,\quad
\mathrm{bias}=T^{-1}\sum_t e_t. \tag{15}
$$

Residual means depend on whether $e=x-\hat{x}$ or $\hat{x}-x$. State the sign convention. A normalized RMSE needs a stated scale, such as the standard deviation of the aligned observations. $R^2=1-\mathrm{MSE}/\mathrm{Var}(x)$ uses development variance under the matched mask. It may become negative. Both the standardized signal and an unverified physical-unit sensor scale should not be labeled as ordinary acceleration squared. Distinguish square units.

**Worked example.** For errors $[-2,0,1]$ the MSE is $5/3$, RMSE $\sqrt{5/3}\approx1.291$, MAE 1 and bias $-1/3$. If a comparator has MSE 2, relative reduction is $(2-5/3)/2=1/6$. A signed paired difference of $-1/3$ corresponds to model minus comparator. Report both the numerator and denominator interpretation. A small scalar reduction can still hide residual shape changes relevant to diagnosis.

### 18.2 Runs and serial structure

A sign runs test counts changes in sign sequence after a specified zero treatment. A Ljung–Box statistic aggregates autocorrelation over lags. The first reveals unusually clustered signs relative to a simple null, the second tests mean lag structure over a chosen range. Neither guarantees independent white noise in a heterogeneous or nonstationary series. Inspect ACF shape and residual power spectra. The expected useful feature may live in a band rather than a single lag, especially when speed changes.

If the report separates errors by RPM quartile, build bins from observed aligned targets and show counts. A grouping analysis may use measured target-time RPM to describe operating regime, although the model may not use it. Direction grouping should describe the index and slope window used. The full score is target-count-weighted across groups. A cold start or boundary contaminant can disproportionately affect the first chronological block, which motivates a prespecified common warm-up.

### 18.3 Paired block uncertainty

Build paired differences $d_t=e_{A,t}^2-e_{B,t}^2$. Draw moving blocks of that same sequence and compute the sample mean across resampled blocks of full length. A negative difference favors $A$. Multiple block sizes expose sensitivity to dependence and speed-cycle structure. The bootstrap percentile range reflects uncertainty under a resampling model for that recording. It does not simulate new gearboxes, acquisition geometries, speeds or sensor calibrations. A narrow interval on one record and transfer to another recording address different questions.

### 18.4 Compute and practical constraints

Record hardware, device, thread count, timing repeats, data and model sizes, and exact processing scope. Warm inference usually means a loaded model called on in-memory arrays after initialization. Cold/load timings should explain whether import and disk I/O are present. Runtime that includes feature construction is comparable to an end-to-end prediction step at record-level. Timer ratios need equal scope. Divide full-record median by 7.5 seconds for batch real-time factor in this course grid. Divide by the quantity of samples for amortized work, while strict online latency also depends on state updates, deadlines, system scheduling and buffering.

Check which memory quantity was recorded. Model weight file size is independent of process runtime footprint. Full process RSS can include many interpreter libraries and loaded candidate models. A snapshot differs from peak or model-attributable allocation. CPU-only measurements leave comparative GPU throughput unresolved. The completed result declares device evidence in timing metadata and gives practical resource interpretation in Notebook 09.
