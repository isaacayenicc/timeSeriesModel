# Matched learned comparison models

The comparison estimates one-step prediction error on the course development record. The target is the standardized vibration sample \(x_t\). Every newly fitted comparator uses measured vibration and RPM through \(t-1\), resets per acquisition and reports MSE on the same 22,208 zero-based indices \([1792,24000)\). The vibration- and speed-informed families quantify whether load/speed information or structured features help under the task's actual information set.

## From observation to regressor

Let \(v_t=(x_{t-1},\ldots,x_{t-L})\). Persistence sets \(\hat{x}_t=x_{t-1}\), a test of local smoothness at 3.2 kHz. Ridge AR models learn one intercept and \(L\) delay weights. For LPV ridge, let \(q_t=(n_{t-1}-145)/25\), where speed is in RPM. The numbers 145 and 25 define fixed basis coordinates in engineering units. The row contains \(v_t\), \(q_t\), the backward 32-sample RPM change divided by 25, and \(T_j(q_t)v_t\) for Chebyshev polynomials \(T_j\) up to the selected order. An interaction such as \(q_t x_{t-2}\) lets the second delay's weight change with speed. Each row is formed before seeing \(x_t\) or \(n_t\).

The linear physics comparator adds fractional angular vibration lags at mesh-cycle multiples \(\{1,2,3,4,6\}/37\) and at 0.25 and 0.375 shaft revolutions, with speed interactions. Integrating RPM through \(t-1\) yields the available target phase. Interpolation uses past vibration samples bracketing the desired phase. The input history is capped at 768 samples. When very low RPM would require a longer history, the wanted phase is clipped to the oldest admissible phase. An optional fixed short quadrature bank estimates complex order responses at 37, 74, 111, and 148 using 96 prior samples and exponential sample weights with a 28-sample scale. This is a causal FIR representation with specified frequency axes. The readout can select fixed order structure, while DHON adjusts oscillator parameters through its training objective.

Each feature column is centered and scaled using the permitted fold-fit rows. Define \(Z\) as the standardized design matrix and \(y\) as corresponding vibration targets. Centering the design yields the unpenalized intercept \(\bar y\). The coefficient vector solves
\[
\min_w \frac12\|Zw-(y-\bar y)\|_2^2+
\frac12\sum_j\lambda_jw_j^2 .
\]
Cholesky solves the positively regularized system without an explicit inverse. The folds and final fit use the same mathematics and refit their own scales.

## Explicit nonlinear delays

NG-RC learns a regularized linear readout of a nonlinear library. A short delay basis consists of lags 1, 3, 6, 11, 18, 25, 34 and 48. A quadratic library takes its 36 upper-triangular products including self-products. The cubic starting library multiplies the short delays by 32-sample past energy and, where selected, adds order quadrature × energy, two-component squared order magnitude and order energy × selected delays. The library avoids an unrestricted cubic expansion of all temporal lags. The search varies temporal span, polynomial degree, order and angle structure, ridge strength and a separate penalty multiplier for the polynomial block. No random recurrent reservoir is used. The archived NG-RC replay uses its own archived feature definitions and checkpoints and is reported separately.

## Generic neural controls

The MLP receives \([N,L]\) lag input for \(N\) targets and \(L\) lags, or \([N,L+2]\) with RPM and its backward change. The predictor is a ridge-initialized frozen linear skip plus \(w_2^\top\tanh(W_1 z+b_1)+b_2\). The hidden readout starts at zero and is trained by AdamW using MSE. This supplies a strong linear initialization to study what generic nonlinear capacity adds. The MLP forms no explicit order or angular features.

The temporal convolutional network (TCN) consumes \([1,C,N]\) channels of strictly prior-sample vibration and, for the speed-informed variant, RPM. A pointwise map projects the channels into a compact latent width. Four left-padded convolutions at dilations 1, 3, 9 and 27 each add a tanh-transformed causal residual to the latent time series. A pointwise readout predicts one sample per target. With kernel length \(k\), the raw history is \(1+(k-1)(1+3+9+27)\), giving 81 or 161 samples for the tested kernel sizes. During training each minibatch is a contiguous target chunk preceded by its full receptive field. The loss indexes only the target chunk. The TCN tests nonlinear temporal features with shared weights and bounded context. The comparator search covers one feed-forward and one convolutional neural family. GRU and LSTM adaptation were outside this finite comparator run.

## Validation under autocorrelation

Fold-selection ranges are \([4800,7200)\), \([11200,13600)\), and \([18400,20800)\) inside `sigtrain`, using a 1,792-sample guard on both sides of each test interval. The longest comparator raw support is 768 samples. Training rows after a fold cannot see that fold's held-out samples through their lag or network receptive fields. The TCN receives no acquisition-spanning recurrent state and therefore cannot carry hidden state across a held-out interval during training. During fold scoring all models can use measured observations earlier than each evaluation target.

For each family the lowest mean MSE across these three guarded folds selects the configuration. The selected model is subsequently refit on `sigtrain` using the common target start. The supplied `sigvalid` scores are development results on a separate acquisition segment used for course comparison. Comparing three overlapping-window folds provides a finite sensitivity check. It does not estimate generalization over an entire gearbox population.

## Reconstructing an individual feature and score

For \(t=1792\), a two-lag AR uses \(x_{1791}\) and \(x_{1790}\). Its speed-informed extension uses \(n_{1791}\). The angular extension estimates the usable phase at 1792 by accumulating speed increments through 1791 and locates earlier observations by fixed phase subtraction. In `gearbox_dhon.comparators.feature_matrix` the first columns are always the dense lag vector, so checking row order against the raw samples verifies target alignment before examining any fit error. For any prediction NPZ key, check its unscored prefix, subtract its scored suffix from the target record and average squared errors over the fixed mask. Per-record predictions contain the original acquisition indices so speed, time, and residual structure are aligned by construction.

The companion `metrics/comparators.csv` lists training and development MSE, mean guarded-fold MSE, secondary metrics, parameter total, model size, final fit time, and seven warmed inference timings summarized by median, IQR and p95. Per-fold configurations and scores are in `search/comparators_folds.csv`; the epoch losses are in `search/comparators_learning_curves.csv`. The `checkpoints/comparator_*.npz` files store coefficient/weight arrays and scaling parameters without serialization code objects. The single-load inference API supports alternate records with finite positive RPM and the same sample rate. The reference notebook recomputes MSE from the emitted predictions to check reported values.

Ridge and feature construction use CPU float64 arrays and SciPy linear algebra. Neural inference uses PyTorch float32 activations with NumPy preprocessing. Timings measure batched inference from an in-memory record and include feature generation, so they compare user-visible prediction cost. Timing on a single 7.5-second record does not establish closed-loop streaming deadline compliance.

## What the guarded comparisons support

The recorded search evaluated 69 candidate configurations and completed 207 fold scores with one CPU thread. Those scores chose nine deployable checkpoints with exact recomputed development predictions. The completed full comparator stage took 160.7 s, including final fits, checkpoint replay checks and warm prediction timing. The nine development MSEs in ascending order are fixed physics linear 0.376432, NG-RC 0.376930, RPM-conditioned ridge 0.383354, ridge AR 0.437474, vibration MLP 0.450408, RPM MLP 0.456610, RPM TCN 0.475057, vibration TCN 0.475430, and persistence 2.087465. Check definitions and timing summaries in the machine-readable outputs.

Adding angular lags and order features as a group lowered all 36 matched NG-RC fold scores, with a mean MSE change of −0.019278. Within the tested delay library, adding cubic terms gave a mean change of +0.000647, and the fold-selected NG-RC retained the quadratic library. The fixed FIR order bank added to the linear angular model had a mean change of −0.002637 across 12 matched scores, with changes of both signs. These outcomes support speed-aware physical representation for this record and offer limited evidence for widening the polynomial degree.

Adding prior RPM improved the mean matched training-fold MSE by 0.001323 for the MLP and 0.012266 for the TCN. On supplied validation, it changed selected-model MSE by +0.006202 for the MLP and −0.000373 for the TCN. The guarded-fold gains did not reliably transfer to the supplied development acquisition. The matched differences are descriptive across correlated folds and grid entries. The calculation is reproducible with `python scripts/compare_features.py --run-dir outputs/runs/development`.
