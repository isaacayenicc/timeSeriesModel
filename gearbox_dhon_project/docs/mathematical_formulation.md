# Differentiable Harmonic Oscillator Neural Network

## Information and units

Let $x_t$ be the supplied standardized acceleration, $n_t$ the measured shaft speed in revolutions per minute, and $f_s=3200$ Hz. The predictor at index $t$ observes vibration and speed through $t-1$. The shaft angle in turns satisfies $u_0=0$ and

$$
u_t = u_{t-1} + \frac{n_{t-1}}{60f_s}.
$$

This left-step rule follows from predicting the next phase under constant last observed speed. The archived encoder instead uses a trapezoidal phase at each observed sample and extrapolates its next step using lag-1 RPM. The two conventions are close for smooth speed but yield different angular features and checkpoints. The course error is standardized sample MSE on indices 1792 through 23999 inclusive, evaluated with measured past vibration.

A 37-tooth gear meshing at $n/60$ shaft revolutions per second gives 37 contacts per revolution and $37n/60$ Hz. Harmonic $k$ gives $37k$ cycles per revolution. An order-locked carrier uses $\phi_{k,t}=2\pi o_k u_t$. A fixed-frequency carrier uses $\phi_{k,t}=2\pi f_k t/f_s$. The fixed centers 1080 and 1125 Hz probe a repeatable spectral region. Their trial performance can qualify that hypothesis. A fixed center does not identify a mechanical mode. Detuning is dimensionless order displacement for the first family and Hz for the second.

## Finite oscillator

The state takes a weighted average of each past measurement after rotating its phase into the target frame:

$$
h_{k,m,t}=\frac{2}{Z_{k,m}}\sum_{j=1}^{M}\rho_{k,m}^{j-1}x_{t-j}
     \exp\left(i(\phi_{k,t}-\phi_{k,t-j})\right),\quad
\rho_{k,m}=e^{-1/\tau_{k,m}},\quad
Z_{k,m}=\sum_{a=0}^{M-1}\rho_{k,m}^{a}.
$$

Here $m$ indexes memory scales, $M$ counts samples, and $\tau$ is a statistical memory constant in samples. To see the factor 2, write a real harmonic as the sum of positive and negative carriers: $x_s=A\cos(\phi_s+\gamma)=(A/2)(e^{i(\phi_s+\gamma)}+e^{-i(\phi_s+\gamma)})$. Demodulation by $e^{-i\phi_s}$ puts the first term at zero phase rate and moves the second to a negative double-carrier term. With averaging over enough periods at an adequately sampled carrier, the second term is attenuated. Reducing memory allows faster response at the cost of a broader filter and incomplete negative-carrier rejection. The phase origin cancels in $\phi_t-\phi_s$. The state bound is $|h_{k,m,t}|\le 2\max_{1\le j\le M}|x_{t-j}|$, even for adversarial bounded samples. The deployed teacher-forced one-step model uses measured past samples. Its learned readout does not have a free-running stability guarantee.

For numerical verification define $z_{k,t}=x_t e^{-i\phi_{k,t}}$ and $q_{k,m,t}=\sum_{j=1}^M\rho^{j-1}z_{k,t-j}$. Reindexing the sum yields

$$
q_{t+1}=\rho q_t+z_t-\rho^M z_{t-M},\qquad h_t=(2/Z)e^{i\phi_t}q_t,
$$

with $z_t=0$ before acquisition start. The implementation performs the same linear convolution using a zero-padded PyTorch FFT, shifts by one target index, and rotates back. The explicit recurrence and independent float64 direct sum agree on impulse, chirp, fixed-Hz and zero fixtures. FFT execution may introduce roundoff from the full batch, so future perturbation is tested within a numerical tolerance. The learned branches never convert physical coefficients or oscillator states to NumPy during updates.

Constrained carriers are $o_k=37k+0.75\tanh(a_k)$ and $f_k=f_{0,k}+24\tanh(a_k)$, with $a_k=0$ initially. Each memory parameter satisfies $\tau=4+380\sigma(b)$. Frequency bounds are checked against $0.475 f_s$ for the entire record at the allowed detuning. RPM is required to be at least 100, which also bounds raw history for angular interpolation. Optimized detuning represents model flexibility. It does not imply a tooth-count change, and the optimized memory is not a mechanical damping ratio.

## Feature construction and head

Let $s_t$ be lag-1 speed centered by fit-set range midpoint and divided by its half-range, then clipped to $[-1.2,1.2]$. The context $c_t$ concatenates $s_t$, a 48-sample vibration RMS, and optionally the lag-1 versus lag-33 RPM change in the same scale. For angular lags start at $u_{t-1}$ and query $u_{t-1}-j/(37P)$ for $j=1,\dots,PC$, with $P$ grid points per mesh and $C$ mesh cycles. Both interpolation endpoints precede the target. Proxy innovations use measured vibration minus a separately fitted LPV ridge AR model. The first stage uses only permitted fit blocks, and its innovation at $t-j$ depends on no later vibration than $x_{t-j}$.

A design row $z_t$ contains an intercept, the raw and angular lags, proxy innovation lags, context and flattened oscillator quadratures. For the linear branch the design expands into products with Legendre polynomials evaluated at $s_t$, from degree zero up to the configured degree. It is scaled by training-only column standard deviations with a lower floor. Let $L_t$ denote this expanded row and write $\hat{x}_t^0=L_t^T w + g H_t$. The neural branch computes normalized memory weights $\pi_{k,m,t}=\operatorname{softmax}_m(Ac_t+b)$ and gain $1+0.25\tanh(Bc_t+b')$. It combines $\sum_m\pi_{k,m,t}h_{k,m,t}$, applies gain per oscillator, and concatenates the resulting quadratures with $c_t$. A linear–SiLU–linear–tanh head returns $H_t$ in $(-1,1)$. The configured gate $g$ bounds the nonlinear correction. The linear branch can still assign distinct weights to each decay and quadrature.

The array of oscillator features has shape $[T,K,D,2]$ after the reset context, with $T$ targets, $K$ carriers and $D$ memory scales. The neural branch sees $2K+\dim(c_t)$ columns after mixing, while the linear branch sees all $2KD$ quadratures at each speed polynomial degree. All learned state settings are stored with the numeric weight arrays in NPZ and loaded with pickle disabled.

## Training and adaptation

1. Split raw training into separate fit blocks and derive speed scale within the fold. Fit an LPV ridge proxy AR, freeze its weights, and construct the deterministic finite-history arrays for each block.
2. Initialize the readout by ridge. In eligible variants add every fifth target from fit-only RPM variants at ±0.75% bias. Add paired feature differences for synthetic 520 and 700 Hz decaying transients with a zero target, which penalizes prediction of pulse content. Estimate both the feature scale and pulse penalty on permitted raw fit segments. The paired penalty applies to initialization.
3. Run AdamW updates. Hold physical settings fixed for the first two epochs, then differentiate a mean squared error plus prior penalty toward nominal carrier and decay settings. Optimize readout and head jointly with the enabled physical parameters at distinct learning rates. No target held out for the fold enters an optimizer step. Log physical parameters and gradient norms per epoch.
4. Compute the optional clipped, finite-memory correction from base residuals. For $e^0_t=x_t-\hat{x}^0_t$, let $c_t=\operatorname{clip}(e^0_t,-C,C)$. The corrected prediction is $\hat x^0_t+g_c\sum_{j=1}^{M_c} w_j c_{t-j}$, with $w_j=e^{-(j-1)/\tau_c}/\sum_a e^{-a/\tau_c}$. Default $C=1.75$, $\tau_c=640$, $M_c=1024$, $g_c=0.5$. The adapter never sees $e_t^0$ before predicting $x_t$. The adapter is measured separately from oscillator learning.

At 100 RPM, the longest configured eight-mesh-cycle interpolation spans at most $\lceil8(60 f_s)/(37\cdot100)\rceil+2 = 418$ past samples at 3200 Hz. The longest oscillator uses 512, the largest raw lag bank 192, and nested proxy innovations use up to $256+24=280$. All fit blocks reset with 768 discarded samples, so the 1024-sample adapter has full compatible past at index 1792. Every held-out interval gets a 1792-sample guard to either side. Training and evaluation raw support is disjoint, including prior innovations and later fit blocks. Full-record FFT inference exploits batch numerics, and its measured batch time does not demonstrate per-sample real-time performance.

## What changed from the archive

The archived head learns mixing weights over oscillator states precomputed using NumPy/SciPy, with externally searched carriers and decays. Here the state itself depends on continuous learned parameters in the PyTorch graph, and the archive's fixed-state concept is retained as a capacity-matched control. Other changes include left-step phase, a slightly older angular-grid endpoint, fold-specific bounded speed scaling, a full-batch AdamW refinement and a finite train-only paired ridge augmentation. Inference stores both base and adapted series to identify correction gains. The study compares complete model implementations and a controlled retrained fixed/learnable DHON contrast. Only the latter controls architecture and geometry at the state-learning comparison.
