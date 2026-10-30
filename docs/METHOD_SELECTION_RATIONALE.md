# Method selection and falsifiable novelty

## Selected suite

| Model family | Question answered | Necessary control |
|---|---|---|
| Zero, training mean, persistence | Does any learned model beat trivial scale/history assumptions? | Identical validation indices; mean fitted only on training |
| OLS AR and ridge AR | How much of next-sample vibration is linearly predictable? | Multiple declared contexts; intercept and ridge objective documented |
| Speed-additive ARX | Is a causal speed input useful without multiplicative scheduling? | Same vibration context and regularization policy |
| Polynomial LPV-AR surrogate | Does a speed-dependent lag response improve prediction? | Label surrogate clearly; do not call Spa LPV-ARMA reproduction |
| Compact MLP | Does generic nonlinearity help before temporal architectural claims? | Comparable causal input window and optimization budget |
| Unidirectional GRU | Does gated recurrent representation help under the available budget? | State reset per window/record; no bidirectional pass |
| Causal dilated TCN | Can a parallel finite-context convolutional model improve the accuracy/time trade-off? | Compute actual receptive field; future-perturbation test |
| Speed-conditioned TCN | Does operating-point modulation help the same core representation? | Vibration-only core, late-concatenation/capacity controls and misaligned-speed control when feasible |

These are project design decisions, not claims imported from unrelated literature. The author TCN repository verifies a general sequence-modeling architecture [R02]; its digit/music/language benchmarks are not same-data evidence. GRU's historical source concerns phrase representations [R03]. FiLM's original work concerns visual reasoning [R04]. See [research synthesis](../research/research_synthesis.md) for evidence status and transfer boundaries.

## Precise novelty claim

The proposed contribution is an experimentally controlled **project-level adaptation of established feature-wise modulation to a compact speed-conditioned causal vibration predictor**, evaluated alongside a related linear scheduling surrogate and residual diagnostics. FiLM itself, LPV scheduling, causal TCNs, and speed-aware monitoring are existing ideas. Available literature access cannot establish priority or world-first novelty. A failed speed benefit remains a valid outcome.

For temporal feature channels \(h\), conditioning applies \(\gamma(s)\odot h+\beta(s)\), where \(s\) contains only speed observations available under the declared protocol. If only \(s_{t-1}\) is used, no claim about access to \(s_t\) is needed. The temporal predictor must never see \(x_t\) or future vibration. Added conditioning parameters must be counted; a same-core model is not automatically a perfectly matched-parameter model.

## Predeclared interpretation rules

1. Rank models by MSE on a common target set, with training time, inference time and complexity reported separately. No missing test data are renamed.
2. Retain every seed and trial. Report variability; do not call a best-seed difference a robust family improvement.
3. Interpret speed benefit only with appropriate controls. Near-repeated train/validation speed trajectories cannot establish transfer to new operating schedules.
4. Consider ridge a deployable recommendation if its accuracy/time balance dominates. Deep models must justify their cost through observed benefit.
5. Lower prediction MSE is not evidence of better physical fault detection. Reference-trained residuals can remove abnormal structure as well as predictable background.
6. Raw/residual diagnostic comparisons use equal segments and valid feature conventions. Unknown speed units force assumption-labeled angle-domain results.
7. Physical fault accuracy, ROC, F1, onset or severity require independently sourced labels and a declared calibration/evaluation policy. Record IDs are not labels.
8. No direct superiority to the lecturer's method is claimed: its full specification and comparable numerical benchmark were inaccessible. The implemented LPV-AR is a surrogate.

## Resource proportionality

Only two short reference traces are presently accessible. A compact model suite and bounded context/seed experiments are more informative than a large architecture search that overuses one validation trace. GPU use is optional and must be measured honestly; CPU-only results are acceptable. Search cost includes all runs, not only the selected checkpoint. New model families should be added only after causal alignment, audit, baseline, artifact reload and reproduction checks pass.

## Evidence not yet available

The expanded source pool contains 19 publication records, including twelve machinery-diagnostics author sources and three architecture sources, with software citations and the inaccessible lecturer anchor counted separately. No article full text was read; source depth and metadata/year discrepancies remain explicit. Chen et al.'s citation is verified as printed in the course PDF, without publisher/full-method verification. The later twenty individually uploaded fault parts resolve the input-transfer limitation: all 200-records now support real descriptive diagnostics. Unknown health labels and physical speed units still limit detection claims independently of successful prediction and feature computation.
