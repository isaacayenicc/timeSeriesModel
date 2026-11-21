# Model equations, implementation and engineering interpretation

Let $z_t$ denote a vector built exclusively from the observed history needed for target $t$. Write $x_t$ for measured standardized vibration, $s_t$ for RPM and $e_t=x_t-\widehat{x}_t$ for a residual available after observing $x_t$. The same evaluation target does not impose the same inductive bias. An ARX model adds a separate input branch while an LPV model changes AR coefficients with speed. A physics-feature regression can remain linear in its fitted weights while using nonlinear measurements of the signal.

## 1. Autoregressive model

AR($p$) estimates $\widehat{x}_t=c+\sum_{i=1}^{p}\phi_i x_{t-i}$. The coefficient vector is constant across speed. The fit minimizes squared residuals by rank-aware least squares with candidate orders 16, 32, 64, 96, 160, 256 and 320. A 96-sample history spans 30 ms, long enough for a little over two mesh cycles around 150 RPM. A longer history resolves changing periodic correlations only if coefficients appropriate for those delays can be estimated. The model is particularly suitable as an inexpensive benchmark because its prediction uses a finite convolution and it receives measured previous vibration directly.

For homogeneous free-running analysis, roots of the associated AR recurrence matter. Robust closed-loop prediction and stability are separate from the one-step test. Inspect pole radii in the classical metrics and residual spectrum by speed. A movement in that spectrum across the speed profile can motivate additive speed history or a speed-scheduled pole structure.

## 2. Autoregressive moving average

ARMA($p,q$) adds $\sum_{j=1}^{q}\theta_j e_{t-j}$ to the autoregressive predictor. The MA component is a history of estimated innovations, not a moving average of the measured vibration. A simple proxy for innovation uses the residual from a higher-order AR fit. The conditional Hannan–Rissanen procedure fits this proxy in the allowed block, then regresses on lagged $x$ and proxy residuals. At inference, the implemented ARMA uses its own recursively defined innovations. Let $A(B)=1-\sum \phi_iB^i$ with delay operator $B$, and $C(B)=1+\sum\theta_jB^j$. Then $C(B)e_t=A(B)x_t-c$. The AR-only model is the $q=0$ limit. Near-unit MA inversion can produce a long tail even when $q$ is small.

The implementation screens candidate inversion radii and constructs a finite polynomial approximation with maximum 1,792 raw lags. It directly compares the predictions against the exact conditional recursion and checks the discarded impulse-tail mass. That provides a bounded support for guarded experiments. A larger model may improve innovation whitening on a stationary process and have little benefit when changing speed is the dominant misspecification.

## 3. Autoregressive integrated moving average

ARIMA($p,1,q$) fits an ARMA to $d_t=x_t-x_{t-1}$ and reconstructs $\widehat{x}_t=x_{t-1}+\widehat{d}_t$. In delay notation the innovation filter extends ARMA by multiplying its numerator by $(1-B)$. The candidate orders $p\in\{32,96,160,256\}$ and $q\in\{2,5,8\}$ apply also to ARMA. A time-varying local mean or low-frequency drift can make differencing attractive. In vibration, differencing emphasizes the upper band and can increase the effect of measurement noise. The model is compared on the undifferenced course scale. Conditional estimation, fixed finite histories and the fitted likelihood problem are distinct modelling choices. The independent external paper ARIMA score used a different segment.

## 4. Autoregression with exogenous inputs

The additive ARX design uses $\hat{x}_t=c+\sum_i\phi_i x_{t-i}+\sum_j\beta_j s_{t-d_j}$. The fitted exogenous coefficients depend on allowed prior RPM at offsets that include 1, 17, 65 and potentially 257 samples. Lagged covariates are highly collinear when RPM changes smoothly, so modest shrinkage is compared. The AR and speed terms have separate roles. It is possible to change speed influence through an additive intercept without changing how a vibration cycle maps to prior targets. This helps distinguish ARX from an LPV model, which interacts speed with vibration lag coefficients. If adding speed improves a method only in isolated operating regions, a useful check is whether its delay pattern tracks a physical phase difference or is mainly a proxy for the collection trajectory.

## 5. Nonlinear autoregression with exogenous inputs

NARX generalizes ARX into $\widehat{x}_t=g(x_{t-1:t-p},s_{t-d_1:t-d_k})$. The course implementation uses two terms $w^Tz_t+f_\theta(c_t)$ plus an intercept. The first is a physics-conditioned ridge fit using measured lag history, normalized speed terms and quadratures. The second is a small tanh network fitted to ridge residuals on the compact input library. Ridge alpha, head width and training epoch alternatives are selected on guarded folds. Each new input window is formed from measured prior vibration. Feeding predictions recursively at inference would alter the residual-learning task and would require separate validation.

Interpreting NARX needs the ridge comparison. If the observed improvement is small, the linear physics projection may be doing most of the work. If a neural residual term improves in the central speed band only, evidence could favor an interaction specific to the available operating condition. These contrasts motivate fixed-feature ablations and held-speed tests, instead of attributing the gain to nonlinear autoregression alone.

## 6. Sparse linear parameter-varying AR

Set $a_i(w)=\sum_{k=0}^{7}\beta_{ik}P_k(2w-1)$ where $P_k$ is a Legendre polynomial and $w$ derives from lag-one RPM scaled by the preserved input range. Regress $x_t$ on eight polynomial interactions for every vibration lag. With 240 lags there are 1,920 interaction columns. The sparse support comes from the prior guarded ADMM and AIC procedure. It minimizes scaled squared error plus a sum of absolute coefficient values and chooses a support before unpenalized refitting. This implementation recomputes the refit by QR on the frozen 475-column support and independently checks coefficient and forecast agreement. A retained high-order basis interaction is a modelling term, not independent physical evidence that a specific gear order is excited.

## 7. Sparse linear parameter-varying ARMA

A second speed-varying regression includes lagged proxy innovations from the first LPV stage. For orders $(135,75)$ and eight basis columns, the second design has $8(135+75)=1680$ candidate terms. The protected support retains 365. At inference establish the stage-one residual once each target arrives. The second stage can use older residuals only. Reusing QR on the frozen support isolates the benchmark change from a new sparse-selection search. Dense LPV, sparse LPV and exact-recursion ARMA can rank differently because they model speed, regularization and innovations differently. The external paper's current-speed and validation conventions should stay distinct from the course's lag-one protocol.

## 8. Linear Regression

Fit $c+\beta^Tz$ using an SVD-based least-squares solution on centered/scaled features. Candidate libraries range from time-only lags through speed interactions to physics terms. The time-only column pattern overlaps classical AR. The wider library exposes products such as lag times a speed basis value, angular delayed vibration and harmonic projections. The model is a linear map of this engineered representation. It is useful for testing how much structured feature design gains before discarding simple regressors. Collinear ridge-design columns can lead to high-variance individual coefficients although predictions remain comparatively accurate on the same profile.

## 9. Ridge Regression

Ridge solves $\min_{\beta}\Vert y-X\beta\Vert_2^2+\lambda\Vert\beta\Vert_2^2$. The fitted objective uses a centered intercept and candidate penalties 0.6, 15 and 80 with column scaling. A positive penalty stabilizes inversion in correlated feature directions and generally leaves dense coefficients. Comparing validation error with unregularized linear regression is meaningful only after examining selected feature designs. Different selected libraries vary the effect of features and shrinkage simultaneously. The train-to-validation gap and the selected fold penalty indicate useful scope without making a causal claim about regularization by itself.

## 10. Lasso Regression

Lasso solves $\min_{\beta}\Vert y-X\beta\Vert_2^2/(2N)+\lambda\Vert\beta\Vert_1$. Its soft-thresholding geometry permits exact zeros. Fold scaling matters because a distinct feature scale changes the penalty on its physical contribution. The implementation uses coordinate descent with a finite iteration budget, records the duality gap and prefers all-fold converged candidates. Highly correlated lags or quadratures may exchange places across fits without materially changing prediction. A nonzero fitted coefficient is evidence of useful representation conditional on the other retained features and the penalty.

## 11. Elastic Net Regression

Elastic Net combines $\lambda\big[\eta\Vert\beta\Vert_1+(1-\eta)\Vert\beta\Vert_2^2/2\big]$ with normalized squared loss. The compared ratios 0.25 and 0.75 trade a greater group-shrinkage effect against more sparsity. Correlated vibration lags, or a sine/cosine pair representing one physical carrier, can both matter. The L2 term can stabilize their joint use. Candidate combinations change the effective group penalty and feature library, so consult the fold table to learn which comparison isolates penalty versus design. Exact zero coefficients and normalized objective must be interpreted with the fitted scaler.

## 12. Support Vector Regression

The radial-basis SVR predicts $b+\sum_i(\alpha_i-\alpha_i^*)\exp[-\gamma\Vert z_i-z\Vert^2]$. It penalizes errors beyond an $\epsilon$-wide tube, and $C$ scales the tube violations. Standardize the features before computing distances. Large $\gamma$ isolates local training neighborhoods and can underpredict a shift in RPM if distances place the new window far away. The exact RBF solver was fitted on a deterministic time-stratified subset within each allowed fold fit set. Limited training rows control kernel cost, so its results are conditional on that budget. Support-vector count affects memory and inference and is more informative here than one neural-style dense parameter count.

## 13. Random Forest Regression

Each tree recursively partitions past-only features by threshold. A terminal node predicts an average over its training responses, and the forest averages the predictions over bootstrapped trees. Feature subsampling decorrelates errors across trees. It has strong capacity for interactions without prespecifying their algebra, but oscillation phase can change smoothly in a high-dimensional lag space where axis-aligned splits need many cells. Leaf size, depth and feature subset determine stability. Tree nodes, leaf values, branch metadata and feature arrays are serialized without pickle. A direct forecast equivalence test checks that the array evaluator reproduces fitted sklearn predictions.

## 14. Gradient Boosting Regression

Boosting builds $F_M(z)=F_0(z)+\nu\sum_{m=1}^{M}h_m(z)$ where each small tree fits the negative loss gradient. For squared error the gradient residual is proportional to $y-F_{m-1}$. Depth and shrinkage determine how rapidly the ensemble can fit difficult regions, and the finite ensemble-size budget controls overfitting. A deterministic subsample can reduce cost and variance. Boosting can distinguish interactions in RPM and energy without a predefined linear factorization. It can still struggle to extrapolate beyond its top-level splits. Verify the optimized portable evaluator by reconstructing ensemble offsets, weights and leaves.

## 15. Multilayer Perceptron

The neural comparison uses a raw-history affine shortcut plus a nonlinear encoder. For MLP the encoder flattens the whole window and passes it through two GELU layers before scalar readout. An MLP can combine efficiently across time positions and channels. Its weights are tied to absolute lag coordinates, so a shift of a local structure within the window generally changes its representation. The linear shortcut can learn the predominant autoregression. Thus a paired result against a plain linear lag estimator combines architecture, fitting and possible design cost differences.

## 16. Long Short-Term Memory

For a patch feature $u_j$, gates $i,f,o\in[0,1]$ act on a candidate $\tilde c_j$ and previous state $c_{j-1}$. The update $c_j=f\odot c_{j-1}+i\odot\tilde c_j$ yields $h_j=o\odot\tanh(c_j)$. The cell forms a selective information path across observed history. All recurrent state begins zero per window. Each patch concatenates eight raw RPM/vibration pairs without dropping raw samples. Sequence recurrence therefore operates over wider step intervals than the time grid, while the shortcut sees the full history. The small training pool offers limited evidence for the many gate choices. The comparison measures short-window one-step accuracy, and it provides no benchmark for long autoregressive rolled-out forecasting.

## 17. Gated Recurrent Unit

The GRU uses a reset gate to control previous-state influence when creating the candidate and an update gate to blend old and new hidden state. In one convention $h_j=(1-z_j)\odot\tilde h_j+z_j\odot h_{j-1}$. There is no separate output cell. In this experiment its patches, validation masks and schedule alternatives match the LSTM family. Relative work differs because GRU uses fewer gate transformations per step. Its hidden state can recognize acceleration-conditioned waveforms. The actual per-record CPU timings matter more than a generic assertion that a gated model is faster.

## 18. Convolutional Neural Network

A CNN applies spatially shared kernels along the historical patch sequence. Two kernel-three layers develop short local receptive fields that the readout combines across lag positions. Symmetric padding inside the fully observed input history is permissible because even the final patch ends at $t-1$. Symmetric convolution across future targets would violate the study's causal contract. Compared with MLP, local sharing decreases the number of independently fitted local pattern recognizers. FFT-style phase transforms are absent, so the learned filters must answer how vibration features vary with RPM from examples.

## 19. Temporal Convolutional Network

TCN uses residual causal convolutions with increasing dilation. For a two-tap filter and dilations $1,2,4,\ldots,2^{D-1}$ the effective patch receptive field is $2^D$. This reaches distant input history with relatively few operations. Left padding supplies zeros before the observed window, allowing a last-patch readout with predetermined support. All five neural families retain the shortcut, so this result addresses the utility of this TCN variant within the fixed search budget. A broader benchmark would vary the dilation schedule at fixed capacity, and include a shortcut-off control to distinguish inductive bias and fitting dynamics.

## 20. Time Series Transformer

The Transformer tokenizes historical vibration and RPM jointly and adds fixed sinusoidal positional context. Encoder attention computes $\operatorname{softmax}(QK^T/\sqrt{d})V$ with a left-support mask. A position can attend only to available current or earlier tokens, and the full history ends at $t-1$. The scalar output uses the final encoded token, with no separate raw-lag affine shortcut. Patching limits the number of token-to-token comparisons. Dense attention scales quadratically in token count, so a longer history trades expressiveness for time and memory in the tested implementation. Two guarded candidate configurations vary context, hidden width, heads and layers. This is a compact one-step encoder rather than a large pretrained time-series foundation model.

## 21. Temporal Fusion Transformer

The compact TFT adaptation preserves variable-selection networks, recurrent local processing, gated residual networks and interpretable shared-value attention. The variable mask assigns a normalized weight to vibration and RPM encodings at each history position. A recurrent stage supplies local context. Multi-head attention expresses where information is aggregated across historical positions with shared values. Gate-add-normalize modules preserve a direct path and let the model choose which transformations to use. The scalar output predicts one next sample. There are no static covariates, known-future exogenous channels, decoder forecasts or quantile heads here, so the label is a historical one-step TFT adaptation.

Recorded variable and attention weights are normalized and can describe model activity over the actual validation sequence. A weight assigns influence inside a particular learned computation graph. Such weights are not causal covariate effects or a confidence estimate. The training row stride reduces the number of overlapping windows passed to the estimator. Every historical position in a window is available to the recurrent branch. The architecture has internal gated residual paths and no raw-lag affine shortcut. Compare guarded error, test-record error and cost before treating added complexity as evidence of a stronger data fit.

## 22. Differentiable Harmonic Oscillator Neural Network

DHON aligns finite oscillator states with measured shaft phase. For order $o$ let $\phi_t = 2\pi o u_t$ where $u_t$ is the causal RPM integral. Let $q_t=x_te^{-i\phi_t}$ and $\rho=e^{-1/\tau}$. A fixed-memory exponentially weighted sum evolves via $E_{t+1}=\rho E_t+q_t-\rho^M q_{t-M}$ and remodulates to the next target grid. A readout can use real and imaginary states, lag history, angular history and speed. In the primary DHON both decay and order detuning are constrained learnable parameters. An innovation branch and causal residual adapter contribute predictive power via strictly older residuals. The old archive fixed the physical carriers and memories and fitted its readout. Both remain important reference points.

The observed fixed/learned and adapter-off contrasts in the prior run explain the primary result more directly than neural family rank. If a new baseline overtakes a physics model, inspect its conditioning, response to speed stress, per-acquisition diagnostics and clean inference costs. If DHON wins, check whether the advantage is robust to bootstrapping and compare the stronger available archival forecast anchor.

## Study checks and sources

- Derive the prediction of ARIMA(0,1,0) with zero mean innovation. It reduces to $\hat{x}_t=x_{t-1}$. Adding a nonzero drift enters through the fitted mean difference.
- Explain why OLS with order features resembles an LPV regression. A column $x_{t-j}P_k(w)$ creates a linear regression coefficient whose effective lag weight changes with speed.
- Compare Lasso and Elastic Net on two identical standardized columns. Lasso can trade one coefficient for the other. Elastic Net favors sharing more smoothly under a positive L2 weight.
- Explain why the same Tukey-like statistical evidence cannot follow from a vanilla independent-sample test across 22,208 overlapping windows. The source record and temporal dependence matter, even when summary MSE is arithmetically exact.
- How do the TFT branches differ from TCN? TFT uses per-variable representations, recurrent context and content-dependent weights. TCN propagates information through shared finite dilated filters.
- Which split would establish performance on new gear, load or sensor? A matched acquisition-level set with measured identity and condition held back until the checkpoint, preprocessing and diagnostic thresholds are frozen.

Selected method references are Chen et al. (2025) on sparse LPV-ARMA [DOI](https://doi.org/10.1016/j.ymssp.2024.112161), Hannan and Rissanen (1982) on ARMA identification [DOI](https://doi.org/10.1093/biomet/69.1.81), Tibshirani (1996) on the lasso [DOI](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x), Zou and Hastie (2005) on Elastic Net [DOI](https://doi.org/10.1111/j.1467-9868.2005.00503.x), Breiman (2001) on random forests [DOI](https://doi.org/10.1023/A:1010933404324), Friedman (2001) on boosting [DOI](https://doi.org/10.1214/aos/1013203451), Hochreiter and Schmidhuber (1997) on LSTM [DOI](https://doi.org/10.1162/neco.1997.9.8.1735), Vaswani et al. (2017) on attention [arXiv](https://arxiv.org/abs/1706.03762), and Lim et al. (2021) on TFT [DOI](https://doi.org/10.1016/j.ijforecast.2021.03.012).
