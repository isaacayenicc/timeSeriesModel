# 5. Physics-informed structured reservoir computing

## 5.1 Target, information set and forecasting scope

Let $x_t$ denote the supplied standardized vibration at time index $t$, $n_t$ the measured input-shaft speed in revolutions per minute, and $f_s=3200$ Hz the shared sampling rate. One tick lasts $312.5\,\mu\mathrm{s}$. The fitted readout uses a vector of deterministic, finite-history quantities constructed from $(x_j,n_j)$ through $j=t-1$. Write $\mathcal F_{t-1}=\sigma\{x_j,n_j:0\le j\le t-1\}$ for that measured history, and let $\widehat\Theta$ contain the configuration and fixed training estimates. The forecast and its scoring residual are

$$
\widehat x_t=F_{\widehat\Theta}(\mathcal F_{t-1}),\qquad e_t=x_t-\widehat x_t .
$$

The predictor has a strict lag-one cutoff for both channels. The instantaneous sample $n_t$ is reserved for later predictions, as is $x_t$. Once the vibration is observed, older errors may become inputs to subsequent predictions. This is teacher-forced, one-step forecasting on measured histories. Deployment as a free-running simulator would instead replace unavailable future measured histories with model outputs. The latter would require separate stability and accuracy evidence.

PI-SRC is a fixed map of causal histories after configuration selection. A regularized linear readout acts on its mixed linear and nonlinear features. Finite causal filters can update internal states recursively. The proxy and readout are trained by regularized least squares, while phase, decay, history length, candidate transformations and correction parameters are externally specified and selected over the guarded grid. This construction fits within fixed-feature reservoir and nonlinear autoregression practice [@gauthier2021; @chenbillings1989]. The physical constraint enters the representation and parameter search [@karniadakis2021]. Figure 5.1 links the measured history, finite phase-aligned feature groups and optional residual correction.

FIGURE_5_1

## 5.2 Sample delays and speed conditioning

The time-domain backbone records a prescribed set of delays $\mathcal L$:

$$
\mathbf{d}_t=\bigl[x_{t-\ell_1},\ldots,x_{t-\ell_L}\bigr]^{\mathsf T},
\qquad 1\le\ell_1<\cdots<\ell_L .
$$

The selected design keeps consecutive lags 1 through 96, equivalent to 30 ms of maximum time-domain displacement. The multiresolution candidate holds the first 16 lags consecutive and then fills longer spacings using a reproducible logarithmic grid. The candidate's maximum chosen lag governs its support. The time and angle embeddings can represent complementary structure. A fixed-time tap follows absolute time, while an angular tap follows the fraction of a mechanical period. Delay-coordinate ideas motivate the representation [@takens1981; @gauthier2021]. The study treats these finite vectors as predictive features in measured vibration with varying operating conditions.

Let $n_{\mathrm{mid}}$ and $n_{\mathrm{half}}$ be the midpoint and half-range of RPM in the admitted training fragments, with the half-range floored at 1 RPM. The operating variables include

$$
z_t=\operatorname{clip}\!\left(\frac{n_{t-1}-n_{\mathrm{mid}}}{n_{\mathrm{half}}},-1.2,1.2\right),
\qquad
a_t=\frac{n_{t-1}-n_{t-33}}{n_{\mathrm{half}}},\qquad
E_t=\frac{1}{64}\sum_{j=1}^{64}x_{t-j}^2.
$$

The acceleration proxy $a_t$ is the scaled 32-sample RPM difference. Division by the corresponding elapsed time would convert an unscaled RPM difference into RPM per second. The fitted features use $a_t$ as defined above, $z_t$, and $\sqrt{E_t}$. The fit recalculates the calibration for each admitted fold and stores the resulting full-training midpoint and half-range, 145.828179 and 21.505339 RPM, in the checkpoint. The deployed model applies that fixed calibration within each subsequent record.

Four linear feature blocks receive the Legendre factors $P_0(z)=1$, $P_1(z)=z$ and $P_2(z)=(3z^2-1)/2$. If $h$ is one of the delay, angle, innovation or quadrature quantities, its scheduled block is $\{h,P_1(z_t)h,P_2(z_t)h\}$. The architecture thereby gives the affine readout smoothly speed-varying weights over each original state, a design link to LPV coefficient scheduling [@chen2025sparse]. Separate use of $z_t$, $a_t$ and RMS preserves operating and amplitude information without multiplying every context variable into every group. Calibration, signal feature scaling and ridge grouping each serve different purposes.

## 5.3 Historical shaft phase and angular interpolation

The 37-tooth input gear defines mesh frequency $37n/60$ in hertz. For the 20-tooth mating pinion, the same mesh frequency corresponds to a pinion rotational rate $37/20$ times input-shaft rate. The feature orders in this study are referenced to the measured input shaft. Order tracking is appropriate when a component repeats at a given angular offset as speed varies [@fyfe1997order]. For each acquired record the code uses an arbitrary zero phase and trapezoidal integration of the observed speed. It then extrapolates the target phase one sample ahead with lag-one speed:

$$
u_0=0,\qquad
u_j=u_{j-1}+\frac{n_{j-1}+n_j}{120f_s},\qquad
\widehat u_t=u_{t-1}+\frac{n_{t-1}}{60f_s}.
$$

Here $u$ is measured in shaft turns. Multiplication by $2\pi$ gives radians. Trapezoidal integration can use $n_j$ for a historical location because $j\le t-1$ at the prediction boundary. The prospective phase $\widehat u_t$ uses approximated next-sample progression. Its forecast error can grow with RPM-change rate, sensor bias or speed dropout. The diagnostic resampling described later uses a completed trace available offline.

The selected angular grid has $Q=16$ points per mesh and eight mesh cycles, so it provides 128 features. For slot $i$ the requested angle is

$$
v_{i,t}=\min\!\left(\widehat u_t-\frac{i}{37Q},\ u_{t-1}\right),
\qquad i=1,\ldots,128.
$$

Find observed turns $u_j\le v_{i,t}\le u_{j+1}$ with $j+1<t$ and form $x_j+(x_{j+1}-x_j)(v_{i,t}-u_j)/(u_{j+1}-u_j)$. Positive RPM makes the observed turn sequence strictly increasing. The implementation checks allowed interpolation endpoints and uses 100 RPM as a resource bound for finite angular history. At 100 RPM the eight-cycle geometry plus two safety samples requires $\lceil 8f_s60/(37\times100)\rceil+2=418$ prior samples. This bound is shorter than the 512-sample oscillator memory. In the supplied training range there are about 31–42 samples per mesh, so the first one-sixteenth-mesh offset is larger than one sample advance. The cap at $u_{t-1}$ gives explicit protection for finer grids. Interpolation can alter spectral amplitudes and is sensitive to the speed channel. Maintaining the ordinary delay block allows evidence from the original time grid to contribute alongside the geometry-aligned representation [@fyfe1997order; @abboud2017envelope].

## 5.4 Finite speed-synchronous oscillator states

The complex oscillator coordinates act as damped synchronous filters with a target-frame phase. For shaft order $o$, define $\phi_j=2\pi o u_j$ and $\widehat\phi_t=2\pi o\widehat u_t$. With time constant $\tau$ in samples, take $\rho=\exp(-1/\tau)$. The code implements the normalized weighted finite history

$$
G_{\rho,M}=\sum_{j=0}^{M-1}\rho^j,\qquad
H_{o,\tau,t}=\frac{2}{G_{\rho,M}}
\sum_{j=1}^{M}\rho^{j-1}x_{t-j}
\exp\!\left\{i(\widehat\phi_t-\phi_{t-j})\right\}.
$$

The oldest allowed vibration sample is $x_{t-M}$. The lag weights start at exponent zero for $x_{t-1}$. The constant $G_{\rho,M}$ normalizes the finite geometric weight array. For a real oscillation with stationary amplitude in the carrier frame, multiplication by the negative carrier phase moves its positive-frequency component to near zero. The reverse component oscillates at twice the carrier phase. Finite smoothing attenuates that reverse image to a degree set by decay and phase progression. The factor of two compensates the positive-frequency half-amplitude in the ideal image-rejection limit. Real and imaginary parts then supply quadratures referenced to the candidate phase at $t$. The effective feature memory depends on $\rho$ and the cut-off $M$. It characterizes a predictive fading-memory kernel [@boyd1985; @abboud2017envelope].

If $c_j=x_j\exp(-i\phi_j)$ and $s_t=\sum_{j=1}^{M}\rho^{j-1}c_{t-j}$, reindexing yields

$$
s_{t+1}=\rho s_t+c_t-\rho^M c_{t-M},\qquad
H_{o,\tau,t}=\frac{2}{G_{\rho,M}}\exp(i\widehat\phi_t)s_t.
$$

Subtraction of the departing sample gives exact finite memory in arithmetic. The implementation applies a first-order filter to the difference signal $c_t-\rho^M c_{t-M}$, delays by one observation and remodulates. Evaluation of the normalization with cancellation-safe exponential expressions retains precision for $\rho$ close to one. For any bounded history, the triangle inequality gives $|H_{o,\tau,t}|\le 2\max_j|x_{t-j}|$. A report-side check directly evaluated 24 of these sums at several positions and order/decay pairs, including the boundary of the 512-sample memory. The largest absolute complex discrepancy with the implemented filter was $6.03\times10^{-13}$. The source also contains independent tests on synthetic signals. This finite-memory bound concerns the feature operator on measured histories. The closed-loop map associated with hypothetical free-running forecasts is a separate dynamical system.

The selected bank uses orders 37, 74, 148 and 259, equivalent to mesh harmonics 1, 2, 4 and 7. Each uses 8-, 32- and 128-sample decay constants (2.5, 10 and 40 ms), with 512-sample maximum history (160 ms). Different decays trade between temporal responsiveness and frequency selectivity. The four carriers and three decays provide 12 complex states and 24 real quadratures. The selected bank uses only shaft-dependent carriers. In fixed-Hz candidates the historical and target phases instead follow $2\pi f j/f_s$ and $2\pi f t/f_s$ for a specified frequency $f$. Candidate fixed-Hz states test for temporal-frequency structure that need not follow the gear order. Interpreting such structure as a specific structural resonance requires independent physical evidence [@abboud2017envelope]. At fit, the largest order frequency evaluated at 103% of the maximum allowed training-fragment RPM and any fixed-Hz frequency must remain below $0.49f_s=1568$ Hz. A new operating range would require its own frequency-validity check.

## 5.5 First-stage innovation proxy

The first stage computes a short-horizon prediction from a speed-scheduled finite AR model. Let $p=192$. For each acquired sample $j$ following adequate history, its proxy forecast and clipped error are

$$
\widetilde x_j=\sum_{d=0}^{2}z_j^d\sum_{\ell=1}^{p}b_{d\ell}x_{j-\ell},
\qquad
\widetilde e_j=\operatorname{clip}(x_j-\widetilde x_j,-3,3).
$$

The fit uses ridge 0.0003, no intercept, and the same outer-fragment RPM calibration as the main library. The scheduled first-stage coefficients multiply monomial powers of $z$, while the main library uses Legendre factors. The implementation filters each AR coefficient row then shifts it by one sample and combines the rows with their target-specific speed powers. The proxy residual uses the newly observed $x_j$, so the downstream readout at $t$ receives only the strictly historical vector $\bigl(\widetilde e_{t-1},\dots,\widetilde e_{t-24}\bigr)$ and scheduled or nonlinear versions of these values. The proxy uses $p+24=216$ raw past samples at maximum. Records reset without transients carried from preceding acquisitions. Within a record the code initializes proxy errors before their required lag order to zero, and scores only after complete context. A fitted AR proxy plus older estimated errors is related to staged innovation methods [@chenbillings1989; @chen2025sparse]. Proxy-error whiteness and exact recovery of physical innovations remain separate questions.

The primary fits the proxy and clean downstream readout on the same admitted outer fragments. A three-block inner cross-fit was tried in a different compound configuration. The upper-layer folds always exclude the held targets and prescribed guards from both fits. The cross-fit trial changes proxy and augmentation choices together, so its utility estimates a bundle.

## 5.6 Structured feature groups and final dimensions

The time, angular and proxy-lag states, oscillator quadratures, and the speed/RMS context form the design backbone. For a causal tap vector $q_{t,i}=x_{t-i}$ with $i\in\{1,2,4,8,16,32,64\}$, the structured group adds squares, consecutive products in that tap list and selected cubes. Energy interactions are $q_{t,i}E_t$ and $q_{t,i} E_t z_t$. There are four lag–innovation products plus a candidate averaged squared proxy-error term. With $h_{o,t}=H_{o,32,t}$, the central decay carrier state, the bounded energy is

$$
B_{o,t}=\frac{|h_{o,t}|^2}{1+|h_{o,t}|^2}.
$$

Candidates include $B_{o,t}$, $z_tB_{o,t}$, $B_{o,t}\operatorname{Re}h_{o,t}$, and $B_{o,t}\operatorname{Im}h_{o,t}$ for each order, along with seven products between the time taps and the mean bounded energy across the four carriers. The code can construct neighboring complex carrier cross-products under its coherent policy. The selected structured policy uses the listed within-carrier interactions. The generic polynomial comparison forms restricted all-pairs products and cubes from historical taps and scaled speed. Such polynomial and rational interactions increase representational capacity beyond an affine raw-delay predictor. The group structure provides a controlled choice of interactions grounded in lag, operating and oscillator quantities [@gauthier2021; @boyd1985].

**Table 5.1. Feature columns in the selected frozen PI-SRC checkpoint.**

| Design block | Unscheduled columns | Scheduled columns | Nonlinear columns |
|:---|---:|---:|---:|
| Vibration sample delays | 96 | 192 | 19 |
| Angular interpolation | 128 | 256 | 0 |
| Clipped proxy history | 24 | 48 | 4 |
| Complex oscillator quadratures | 24 | 48 | 23 |
| Operating context and amplitude | 3 | 0 | 14 |
| Total | 275 | 544 | 60 |

The combined library has 819 non-interaction features and 62 nonlinear candidates before selection. A column is eligible if its training standard deviation exceeds $10^{-7}$. All 881 candidates passed this threshold in the selected full fit. A nonlinear-only feature cap then ranks $\bigl|\sum_t(x_t-\overline x)v_{t,j}\bigr|/s_j$ on clean fitting rows, with $s_j$ the candidate standard deviation. The common target variance and row count are irrelevant to the ranking. Stable sorting resolves ties before original column order is restored within the selected nonlinear group. The two dropped candidates are $x_{t-32}^2$ and mean proxy-error energy. Exactly 879 feature columns remain. The code scales a retained quantity $v_{t,j}$ to $(v_{t,j}-\mu_j)/\max(s_j,0.005)$ using clean admitted rows only. The selected rank setting retains the actual columns. The reduced-rank alternative applies a fitted principal-direction projection to the screened nonlinear features. Column identities, group membership, moments, coefficients and any projection are serialized explicitly.

## 5.7 Group ridge and protection rows

Write $\boldsymbol\psi_t$ for the normalized retained vector, $b$ for an unpenalized intercept and $\boldsymbol w$ for readout weights. With $A$ denoting the actual design of clean and weighted auxiliary rows, $\boldsymbol y$ their targets and $N_A$ their total number, the optimized numerical objective is

$$
\min_{b,\boldsymbol w}\frac{1}{N_A}
\left\|A\begin{bmatrix}b\\\boldsymbol w\end{bmatrix}-\boldsymbol y\right\|_2^2
+\lambda\sum_g a_g\|\boldsymbol w_g\|_2^2.
$$

Diagonal group weighting retains the ridge objective [@hoerl1970]. Group-specific penalties are $a_g=6$ for nonlinear columns, $a_g=0.9$ for other oscillator columns (including their Legendre schedules), and $a_g=1$ elsewhere. The selected $\lambda$ is $0.022$, with squared feature coefficients penalized in the standardized basis. The group weights preserve all retained columns and adjust the degree of shrinkage. Group-sparse estimation with a different penalty can set an entire group to zero [@yuan2006].

Two auxiliary designs were activated in the selected fit. On every fifth clean design row, the RPM channel is multiplied by $1-0.0075$ or $1+0.0075$ and the feature vector is reconstructed with its original clean target. Both copies have loss weight $0.35$. For synthetic pulse preservation, two once-per-turn vibration bursts use 520/700 Hz and shaft phase offsets of 0.23/0.71 turns. The generator gives the bursts amplitude 0.9 in standardized vibration and an exponentially decaying envelope with a 3.5 ms time constant, truncated at 18 ms. The input row is the difference between the perturbed and pristine standardized feature vectors. Its intercept column and target are zero, so the penalized quantity is the change in the base forecast due to the pulse. Each of the two difference rows carries weight $0.2/2=0.1$. These weights are applied by scaling a row and its target by the square root of the relevant weight. The resulting extra 18,792 rows augment 23,488 clean full-training rows. These rows alter the objective while preserving the same acquisition count. The folded intervention vector in Section 6 is a separate stimulus from both fitting bursts.

The code forms the Gram matrix and right-hand side in blocks of 4,096 rows and accumulates both in float64. The primary normal-equation solver is Cholesky with a reciprocal condition estimate. When a factorization or reciprocal-condition check fails, the implementation solves an augmented weighted least-squares system by pivoted QR. The selected run completed one Cholesky solve with estimated Gram condition $2.34\times10^4$. The Huber option performs five reweighted solves, changing the robustness weights on clean rows. OLS is also exercised in the search. Runtime and near-collinearity are practical considerations because the angular, time and scheduled groups overlap. The reported total of 1,456 parameters consists of 880 readout scalars (879 plus intercept) and a $3\times192$ first-stage coefficient array. Moments and feature indices are stored separately from fit coefficients. The compactness definition counts projection weights in addition when a projection is present.

## 5.8 Historical correction and end-to-end support

Let $\widehat x^0_t=b+\boldsymbol\psi_t^{\mathsf T}\boldsymbol w$ and $e^0_t=x_t-\widehat x^0_t$. The selected adapter uses older, clipped base-model errors:

$$
\widehat x_t=\widehat x^0_t+
g_c\frac{\sum_{j=1}^{M_c}\exp[-(j-1)/\tau_c]\operatorname{clip}(e^0_{t-j},-C,C)}
{\sum_{j=1}^{M_c}\exp[-(j-1)/\tau_c]}.
$$

The selected gain is 0.5, decay is 512 samples, memory is 768 samples and clip is 1.75 standardized units. The clipped quantity is the base error, which becomes available after each earlier measurement. The finite filter uses overlap-add convolution and an explicit one-sample delay. Under valid recorded histories, changing vibration and RPM at and after the current target can affect later predictions. Four perturbation positions in the development record left the entire scored forecast prefix through each current target equal within $4.44\times10^{-16}$. An independently expanded correction sum agreed with the implementation to $1.11\times10^{-16}$. These discrepancies are on the scale of float64 arithmetic, including small convolution roundoff. The record-level software validates shape, finite inputs, sampling rate and minimum speed before applying the predictor, so numerical causality was checked on admissible histories.

The maximum base support is 512 samples. The oldest used base residual requires up to a further 768 samples, making the selected full support 1,280 samples (0.400 s). The 1,792-sample scoring mask is a comparison warm-up. It covers the primary and all included model variants. Each admitted fragment resets the feature construction. At inference no phase crosses acquisition boundaries, and the model builds features in 4,096-target batches with their required preceding context. Common-mask comparison with correction disabled on identical coefficients directly attributes the adapter effect. The development MSE decreased from 0.340706 to 0.329973 with the adapter. Applying the identical filter to strict Sparse LPV-ARMA and polynomial SRC lowered MSE by 0.003413 and 0.003038. This establishes an observed benefit of correction in multiple pipelines and motivates separate evaluation of PI-SRC's state design.

## 5.9 Complexity, interpretability and contribution

For a record of length $N$, $K$ order/frequency carriers and $D_\tau$ decay constants, finite oscillator state construction takes $O(NKD_\tau)$ operations with first-order filtering. Building the scheduled proxy takes finite-filter work associated with $p$ lags and its three degree rows. Angular interpolation locates two observed samples by binary search for each target and angular offset. The affine readout requires work proportional to the scored number of targets and $d$ retained feature columns. Dense Gram accumulation and factorization costs scale as $O(N_A d^2)$ and $O(d^3)$ respectively, conditional on materialized feature vectors. The delivered code performs blockwise accumulation and bounded-batch inference. The dedicated allocation measurement compares roughly 248 MB for full-matrix feature materialization with 43.2 MB for the two principal arrays in 4,096-target batches. The largest prediction difference between these implementations was $8.79\times10^{-9}$ on that benchmark. These figures quantify allocated bytes for two design buffers. Computational results later quantify batch timing of the full delivered pipeline.

Physical interpretability resides chiefly in input synchronization, finite memory, measured speed scheduling and group definitions. Coefficients of correlated groups can compensate one another and depend on feature scaling and the augmentation objective. Learned weights describe an empirical predictor, and orders/decays encode hypothesized structure. Claims about resonance or identified damping would require a mechanical experiment designed for that purpose.

The main contribution is a specified integration of measured-history order alignment, finite damped quadratures, lagged innovations and selected low-degree nonlinear interactions in a group-regularized predictor. Relative to the generic polynomial and Sparse LPV-ARMA comparators, it brings explicit shaft-phase states and interaction groups into the represented history, and tests a guarded joint selection criterion with RPM and pulse perturbations. Prior fixed-feature, physics-informed and regularized approaches establish distinct precedents for these ingredients [@gauthier2021; @karniadakis2021; @chen2025sparse]. This investigation establishes how their exact combination performed against aligned, executed comparators and control variants in the supplied experiment.

# 6. Guarded optimization and reproducibility

## 6.1 Folds, guard and candidate chronology

The optimization driver uses the supplied training file alone. Its three half-open target intervals are listed in Table 6.1. Given held targets $[l,h)$ and guard $g=1792$, admissible raw fit fragments are $[0,l-g)$ and $[h+g,24000)$. Each fold has 2,400 held targets and 18,016 raw training samples spread across the two fragments. The primary loses its 512-sample base context at each fragment origin, leaving 16,992 clean readout rows per fold. The largest full support among the 41 completed configurations was 1,536, so the guard covers all their dependency depths. Calibration, proxy weights, feature means, scaling, screens and readout are reconstructed inside each fold.

**Table 6.1. Zero-based training-series intervals for guarded model selection.**

| Fold | Held targets | First fitting fragment | Second fitting fragment |
|:---|:---|:---|:---|
| 1 | [4800, 7200) | [0, 3008) | [8992, 24000) |
| 2 | [11200, 13600) | [0, 9408) | [15392, 24000) |
| 3 | [18400, 20800) | [0, 16608) | [22592, 24000) |

The fold predictions use the original series with observed histories. The guard removes held targets and neighboring samples from coefficient estimation, while operational inference conditions on earlier measured samples. Fitting can use disjoint later sections of the training series. The reported fold score estimates guarded within-record comparison in a setting with speed and time dependence. A separate earlier-only refit evaluated the principal configuration using only the first admissible fragment of each fold and produced held-block MSEs of 1.413805, 1.110925 and 0.353668. Fragility in the early folds indicates limited coverage of subsequent conditions using those shorter fragments. The earlier-only figures were kept outside the selection objective. Both procedures sample blocks from the same machine and recording.

There were five stages, with 13, 13, 5, 6 and 4 configurations in stages A through E, respectively. The 41 successful configurations produced 123 guarded fits. Stage A evaluated controls and physical carrier alternatives, covering delay-only and speed-scheduled baselines, a linear physics readout, angular depth alternatives, no-carrier and fixed-Hz ablations, and banks of two, four, six and twelve mesh orders. It also tested selected shaft and near-mesh sideband orders. Stage B changed feature capacities, innovation history, time and angle geometry, acceleration, the polynomial family, the coefficient norm penalty, a reduced-rank alternative, a multiresolution/coherent configuration, Huber estimation and OLS. Stage C explored compound configurations with memory 256–720, decays from 6 to 190 samples across tested settings, angular grids from 8 to 16 points per mesh, and a mixed sideband/fixed-frequency bank. Stage D evaluated multiple adapter settings and targeted protection bundles including optional inner cross-fitting. Stage E refined ridge strength and correction around the best admissible physical configuration from stages A–D. This finite staged search evaluates trial bundles, so coupled changes in configurations limit attribution to any single varied variable.

The standard sweep used a fixed configuration sequence, deterministic operations for the SRC primary and a recorded protocol seed of 19. The trial log gives about 231 s in total recorded per-configuration runtime in that run. This sum includes fitting and per-fold clean, RPM-stress and pulse predictions. The development file entered only after the named choices were frozen.

## 6.2 What the scalar utility measured

Each fold drew clean and modified predictions for exactly the held indices. An RPM-stress intervention multiplies the RPM trace by $1+0.012\cos(2\pi(0.45)t/f_s)$, retaining the original vibration as the target. The folded pulse vector comprises eight 20-sample damped cosine bursts placed inside the scored interval, all with amplitude 0.6 and decay constant six samples. Their carrier has 0.19 cycles per sample (608 Hz). Define $D$ as the squared change in the model output, summed over the held interval and divided by the squared injected pulse norm. Both forecasts in the numerator use the same frozen fit. A small $D$ means comparatively little of this prescribed pulse was followed by the predictor. This surrogate concerns waveform retention under a defined intervention. Evaluation of PMI on real record groups addresses a separate question. The scalar objective averages fold quantities equally:

$$
J=\overline{\mathrm{MSE}}_{\mathrm{clean}}
+0.14\max\!\left[0,\overline{\mathrm{MSE}}_{\mathrm{stress}}
                    -\overline{\mathrm{MSE}}_{\mathrm{clean}}\right]
+0.009\overline D+0.000002p_{\max}.
$$

The final complexity term uses the maximum stored coefficient count across the three fitted folds. The first two terms use squared standardized vibration units. The $D$ term is dimensionless before weighting, and the coefficient-count term prices stored coefficients. The weights establish a specific trade-off on the utility scale. Timing is recorded separately. The code also writes a nondominated four-axis frontier using clean MSE, RPM-stress MSE, pulse distortion and coefficient count. Figure 6.1 shows the guarded selection trade-offs.

FIGURE_6_1

At the primary choice the mean clean fold MSE was 0.355734, the stress MSE 0.381362 and the mean pulse ratio 1.050501. The corresponding contributions to $J$ were about 0.355734, 0.003588, 0.009455 and 0.002912. These sum to 0.371689. A bounded report-side recomputation from saved targets and forecast vectors reproduced all 123 clean and stress fold MSEs to within $5.55\times10^{-17}$ of the archived tables, and all pulse ratios to within $2.22\times10^{-16}$. The archived scalar utilities reconciled within $5.55\times10^{-17}$. The primary's individual clean fold MSEs were 0.293951, 0.423995 and 0.349257. Variation across three dependent blocks informs sensitivity to the held interval. Acquisition-level uncertainty would require multiple independent recordings.

## 6.3 Selection, alternatives and artifact lineage

The driver restricts the primary selection to completed eligible physical PI-SRC variants, sorting by utility and a deterministic trial-ID tie rule. From the same admissible set it saves the minimum-clean-MSE fit, the best utility within the compactness constraint and the best utility with the adapter off. The compactness threshold is the larger of the minimum admissible coefficient count and 65% of the primary count. These rules produced the rows in Table 6.2. The trial named selected_correction, stage E, retains stage D's protected four-order design, lowers the memory from 1,024 to 768 correction samples and changes its decay from 640 to 512 samples. The 192-order proxy, 24 innovation lags, four order choices, eight angular cycles and group-weighted ridge are fixed in that final configuration. The minimum-clean-MSE row local_ridge_down uses ridge 0.0099 with the stage D correction geometry.

**Table 6.2. Training-selected PI-SRC choices and subsequent common-mask development results.**

| Locked choice | Trial identifier | Clean MSE | Stress MSE | Pulse ratio | Utility | Coef. | Dev. MSE |
|:---|:---|---:|---:|---:|---:|---:|---:|
| Utility | selected_correction | 0.355734 | 0.381362 | 1.050501 | 0.371689 | 1456 | 0.329973 |
| Clean MSE | local_ridge_down | 0.355511 | 0.380581 | 1.132523 | 0.372125 | 1456 | 0.326401 |
| Compact | lag_degree1 | 0.376050 | 0.402773 | 0.930360 | 0.389832 | 834 | 0.358022 |
| Adapter off | selected_uncorrected | 0.355942 | 0.381616 | 1.050319 | 0.371901 | 1456 | 0.340706 |

The utility winner improves $J$ by 0.000437 versus the pure-MSE choice despite a small clean-MSE disadvantage across its training folds. Its lower synthetic response drives that trade. The pure-MSE choice subsequently scores lower on development, and the primary retains the originally declared utility target. The compact design has less than 60% of the primary stored coefficient count and higher development MSE, a different accuracy and resource profile. Extending to a fixed 1,100 Hz carrier without order-state features took the stage-A clean fold MSE from 0.383623 (no carrier) to 0.383525. The improvement of about 0.00010 for this pairing gives limited evidence that a fixed frequency helped under those settings. Other changes to the angle grid, operator bandwidth or datasets could yield different outcomes. In the delivered experiment, the utility selected four explicitly specified shaft-synchronous orders.

The final configuration and forecast vectors are versioned with the experiment. Configuration files, the trial table, fold scores and saved fold forecast vectors record the evidence for the selected numerical checkpoint. The trial log records the model-source and optimization-config fingerprint and the training file hash. The recorded model-source commit prefix is 802e5b6. The tested environment used Python 3.12.14, NumPy 2.5.5 and SciPy 1.17.0 on Linux x86-64. The model constructs features in mixed float32/float64 arithmetic, with phase and oscillator calculations in double precision and Gram accumulation in float64. Measurements use one BLAS thread on an AMD EPYC 9V74. A previous deterministic refit produced the same checkpoint SHA-256 and exact development forecast vector. Loading the frozen selected checkpoint at report preparation reproduced all 22,208 development predictions exactly. These checks tie the reported values to the numerical pipeline under the supplied operating conditions.
