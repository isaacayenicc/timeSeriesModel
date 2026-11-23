# Selected PI-SRC model card

The primary checkpoint is `checkpoints/src/primary_pi_src.npz`. Its fitted configuration and both code and data identities appear in `checkpoints/src_registry.json` and `results/metrics/src_summary.json`. The numeric NPZ contains feature layout, model settings, normalization, first-stage coefficients, selected nonlinear columns and the fitted readout. The inference loader disables NumPy pickle. A fresh process verifies the checkpoint identity before scoring. Every acquisition starts from reset.

The training-only utility selected trial `E:selected_correction` from 41 completed candidates. Ninety-six vibration lags feed the readout alongside historical speed (quadratic), acceleration, 64-sample energy and 16 angle samples per mesh cycle across eight cycles. The oscillator bank uses shaft orders 37, 74, 148 and 259, with decay constants 8, 32 and 128 samples and a hard 512-sample memory. The proxy uses lag order 192 and 24 clipped historical innovations. The cubic-capable structured interaction family is screened to at most 60 nonlinear columns. Ridge multiplies normalized group penalties, at a global weight of 0.022. Training uses explicit speed-noise examples and a synthetic preservation term. The correction has gain 0.5, tau 512 samples, a 768-sample window and error clip 1.75 standardized units. Its physical time constants were externally tuned. The deterministic readout has 879 input features and the deployed checkpoint contains 1,456 learned coefficients including the proxy and intercept. The 1,280-sample raw information support includes correction.

| Measurement | Value |
|---|---:|
| Three guarded training-fold mean MSE | 0.355734 |
| Guarded robustness-aware utility | 0.371689 |
| In-sample MSE on the common training mask | 0.308688 |
| Held-out development MSE | 0.329973 |
| Held-out development MAE | 0.456412 |
| Exact same-mask archived corrected SRC MSE | 0.317788 |
| Same input and weights with correction disabled | 0.340706 |
| Strongest required comparator, Linear Regression | 0.367386 |
| Strict local Sparse LPV-ARMA MSE | 0.371406 |

Standardized units apply. The selected PI-SRC is 0.012186 MSE, or 3.83%, above the archived SRC and 10.18% below the strongest required comparator. Further development analyses include the clean-MSE-only choice, paired ablations, temporal bootstrap and stress diagnostics. Blind MSE, field fault classification and GPU performance remain unverified. Scope, timing and data limits are documented in `limitations.md`.

A separate refit generated a byte-identical primary checkpoint and reproduced every development forecast on the controlled platform. For feature evaluation, bounded 4,096-row batches reduced the two principal allocated feature arrays from 248 MB to 43.2 MB. Their largest difference from full-matrix prediction was 8.79 × 10⁻⁹. Detailed CPU timings are in `results/timings/feature_materialization.csv`. This is an allocation-count comparison for the two arrays. Neither row measures whole-process peak memory.
