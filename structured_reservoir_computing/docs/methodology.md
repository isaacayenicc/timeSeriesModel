# Physics-informed structured reservoir computing

The model estimates the next sample of the standardized course vibration $x_t$ from measured observations through time index $t-1$. It uses shaft speed only as a measured lagged covariate. Once the target arrives, its prediction error becomes a candidate feature for later targets. This measured-history experiment is separate from recursive free-running forecasts. The reservoir is constructed by causal delay, filtering and a chosen nonlinear feature library. It does not require random recurrent weights or backpropagation. A numerical optimizer chooses physical and feature hyperparameters on guarded training folds, while each fitted readout is a deterministic regularized regression conditional on the configuration.

## Delay geometry and operating features

Let $n_t$ be input shaft RPM, $f_s$ the sampling rate and $\mathcal L$ the set of positive lag indices. Time-delay state is $z_t^{\mathrm{time}}=(x_{t-\ell})_{\ell\in\mathcal L}$. The consecutive design uses $\ell=1,\ldots,L$. The multiresolution alternative keeps the first 16 positions dense and selects the remaining lags on a deterministic logarithmic grid, deduplicating and filling the available indices. The largest lag, rather than the number of lag features, sets the raw support. Local energy is the mean of squared vibration over a finite past window. Its square root is an RMS feature. The acceleration proxy is the difference between the most recent RPM and the RPM a configured number of samples earlier. This implementation uses the speed-scaled difference, which is proportional to a causal windowed derivative at a fixed sample interval.

The observed shaft phase in turns is accumulated by trapezoidal integration,
$$
u_0=0,\qquad
u_j=u_{j-1}+\frac{n_{j-1}+n_j}{120 f_s}.
$$
At target $t$, $u_{t-1}$ is available. The strict target estimate is
$$
\widehat u_t=u_{t-1}+\frac{n_{t-1}}{60f_s}.
$$
The extrapolation uses only the previous RPM. A 37-tooth input gear gives mesh frequency $37n/60$ Hz and mesh order 37 relative to its shaft. With $Q$ samples per mesh cycle, the encoder queries $\widehat u_t-i/(37Q)$ for successively older values of $i$. It limits each query to $u_{t-1}$ and checks both interpolation endpoints precede $t$. In the normal RPM and resolution ranges the cap is inactive. Linear interpolation allows each feature to represent a physical rotation offset across changes in speed. It can attenuate high-frequency structure, so the time history remains available. Angle queries require finite prior time support because admissible RPM has a positive lower bound.

The nominal model speed is $s_t=(n_{t-1}-n_{\mathrm{mid}})/n_{\mathrm{half}}$, clipped to $[-1.2,1.2]$. Midpoint and half-range use only the permitted fitting blocks. Legendre terms $P_0(s)=1$, $P_1(s)=s$, $P_2(s)=(3s^2-1)/2$ and $P_3(s)=(5s^3-3s)/2$ provide optional scheduled products with each time, angle, innovation or oscillator state. The scaled speed and acceleration proxy also enter as context features. A documented vibration-only reference disables speed and angle inputs.

## Finite oscillator bank

For order $o$ use observed source phase $\phi_j=2\pi o u_j$ and prospective carrier phase $\widehat\phi_t=2\pi o\widehat u_t$. For a fixed frequency $f$ use $\phi_j=2\pi f j/f_s$, with the same formula at the target. A mesh-locked order changes its temporal frequency with RPM. A fixed frequency tests for absolute-Hz signal structure. Neither alone identifies a mechanical mode. A bank may include shaft order and near-mesh sidebands when training EDA supports testing them.

Let $\rho=\exp(-1/\tau)$ for memory constant $\tau$ in **samples**. The $M$-sample state is
$$
H_{o,\tau,t}=\frac{2}{Z}\sum_{j=1}^{M}\rho^{j-1}x_{t-j}
                  e^{i(\widehat\phi_t-\phi_{t-j})},
\qquad Z=\sum_{j=1}^{M}\rho^{j-1}.
$$
A real vibration sinusoid has positive- and negative-frequency components, each with half the sinusoid amplitude. Multiplication by $e^{-i\phi_j}$ translates the positive component toward zero frequency. The finite weighted average attenuates the opposite image with attenuation depending on carrier and memory. The factor two compensates the half-amplitude in the ideal image rejection limit. Real and imaginary parts then retain phase-sensitive information in the target frame. A memory specified in seconds would use $\rho=\exp(-1/(f_s\tau_{\mathrm s}))$. A fitted memory constant characterizes signal representation and is not a mechanical damping measurement.

The implementation uses stable envelope recurrence. Write $z_j=x_j e^{-i\phi_j}$ and $q_t=\sum_{j=1}^M\rho^{j-1}z_{t-j}$. Reindexing gives
$$
q_{t+1}=\rho q_t+z_t-\rho^M z_{t-M},
\qquad
H_{o,\tau,t}=(2/Z)e^{i\widehat\phi_t} q_t.
$$
The oldest term is subtracted at the same update that inserts the newest one. Subtraction makes this a finite-memory state despite the first-order recursive implementation. The envelope is delayed one position before remodulation. The geometric normalization uses cancellation-safe exponential arithmetic. Since normalized weights sum to two, $|H_{o,\tau,t}|\le 2\max_{1\le j\le M}|x_{t-j}|$ by the triangle inequality. Float64 checks compare both the rotating order state and fixed-Hz state with explicit finite sums. Fixtures cover harmonic input, a speed chirp, impulse truncation and zero vibration. Bounded feature states do not certify free-running predictor stability.

A fit rejects carrier banks whose maximum training frequency, including a 3% upward RPM allowance for orders, reaches $0.49f_s$. This margin tests training-observed speed coverage and the study's stress scale. An acquisition deployed far beyond that speed range would need a new Nyquist assessment. The feature library uses neither future speed interpolation nor any global evaluation-record vibration scaling.

## Proxy innovations and structured interactions

When enabled, a finite order-$p$ proxy AR with low-degree speed conditioning fits measured healthy vibration on allowed training blocks. It gives $\widetilde e_j=x_j-\widetilde x_j$ after target $j$ arrives. The SRC feature at target $t$ uses only the clipped history $\widetilde e_{t-1},\ldots,\widetilde e_{t-q}$. Hence raw vibration support extends to $p+q$ samples before $t$. The first-stage coefficients remain fixed at deployment. Optional inner cross-fitting creates downstream fit rows from projections of separate training chunks. Its first-stage fitting targets leave a proxy-history guard around each held innovation block. The final frozen proxy is fitted using the outer fit blocks and used at inference. For paired augmented rows both pulse or speed variants and their reference use this same full-fit proxy. The outer speed normalization uses only permitted outer fit data.

Each deterministic feature group has a recorded name and order. The structured library augments its anchors with a small set of selected interactions:

- Squares, neighboring products and selected cubes of recent and multiresolution vibration taps.
- Lag multiplied by past-window energy and, when supplied, by energy times normalized speed.
- Selected lag–innovation products and proxy error energy.
- Bounded oscillator energy $B=|H|^2/(1+|H|^2)$, optionally multiplied by speed or by its own quadrature. The last term modulates the corresponding carrier as its recent amplitude changes.
- Lag multiplied by mean bounded oscillator energy.
- In the coherent interaction configuration, $\operatorname{Re}(H_i\overline H_j)$ and $\operatorname{Im}(H_i\overline H_j)$ for neighboring order-bank entries. The product carries difference-phase behavior. It is a predictive modulation feature and does not prove that the measured signal contains a physical mixing interaction.

The generic polynomial control instead forms restricted all-pairs products of a small delay/speed library and optional cubes. A cap limits the nonlinear columns using fit-only absolute correlation with the target after accounting for the feature's standard deviation. This screening preserves the linear feature backbone. It is a marginal heuristic, and it does not prove that a selected feature adds conditional predictive value. An optional rank limit projects the retained nonlinear features through principal directions fitted on the fit matrix. Coefficients and group memberships are saved with feature identities, projection and scales. Features with near-constant training variation are dropped deterministically. Highly correlated features and overlapping schedules limit individual coefficient interpretability. Use retrained group-removal controls and pulse sensitivity to support interpretation.

## Scaling, fit and correction

The fit centers and scales feature columns with fit-only estimates and a positive scale floor. If $X$ is the scaled design with an explicit intercept and $w$ its coefficient vector, the group ridge fit uses
$$
\min_w\|Xw-y\|_2^2 + N_{\mathrm{aug}}\lambda
                   \sum_g a_g\|w_g\|_2^2.
$$
The intercept is not penalized. $N_{\mathrm{aug}}$ includes augmented rows when enabled. Selected physics and nonlinear families have their own group multipliers. Paired RPM augmentation supplies the original vibration target with the speed channel perturbed. The synthetic pulse penalty gives the difference between the feature matrices with and without an added pulse a target of zero. This term reduces response to designated synthetic bursts. It does not establish broader fault preservation. The optional Huber solver uses five weight updates on the clean rows while retaining weights for the auxiliary quadratic terms.

The numerical solve accumulates sufficient statistics in blocks using float64. A Cholesky solver is used for well-conditioned penalized normal equations, with a reciprocal condition estimate. The implementation falls back to pivoted QR on an augmented least-squares system when the normal equations are nearly singular or fail a positive-definiteness check. OLS is supported as an unregularized control. The complete feature matrix is still materialized at fit. Blockwise Gram assembly avoids forming one additional weighted full-height design for a well-conditioned solve.

For base error $e_t^0=x_t-\widehat x_t^0$, the optional slow adapter uses
$$
\widehat x_t=\widehat x_t^0+
    g_c\,\frac{\sum_{j=1}^{M_c}e^{-(j-1)/\tau_c}\operatorname{clip}(e^0_{t-j},-C,C)}
                  {\sum_{j=1}^{M_c}e^{-(j-1)/\tau_c}}.
$$
Only older errors from the base forecast enter this expression, and the correction parameters do not adapt to evaluation targets. The initial full-context requirement is the base history plus $M_c$ when active. With the studied settings this remains inside the 1,792-sample common mask. The adapter uses measured past vibration and therefore remains a conditional one-step mechanism. Reporting base and corrected forecasts from the same fitted coefficients separates the contribution of the adapter from feature improvements.

Oscillator feature generation costs $O(NKH)$ for $N$ samples, $K$ carriers and $H$ decay settings. The proxy finite filter costs $O(NpD_s)$ with $D_s$ speed-polynomial components. The angle feature path uses bounded interpolation queries, implemented by vectorized binary search within each batch. A linear readout of $d$ final features adds $O(Nd)$ time. A dense ridge factorization and Gram assembly are $O(d^3)$ and $O(Nd^2)$ respectively. Inference limits materialized features to one target batch and its required independently reset history. A production streaming implementation could maintain bounded buffers and envelope recurrences. Batch timing alone does not establish a hard-real-time worst-case latency at the 3,200 Hz target rate.
