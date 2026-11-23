# Course marking traceability

The two-page MECG 7780 handout grades next-sample prediction by MSE, computational efficiency by CPU/GPU time and fault-diagnostic performance. The 80% final component includes the report and separate final presentation. The progress presentation accounts for 20%.

| Criterion | Progress | Final report and presentation | Evidence integrated in the report |
| --- | ---: | ---: | --- |
| Report organization and presentation | 4% | 16% | Abstract and objectives in §1, verified input provenance in §3, results in §§8–10, and limitations and conclusion in §§10–11. Supplementary evidence identifies inputs, checkpoints, model registry and reproducible commands. |
| Presentation clarity | 4% | 16% | §4 presents spectrogram, order spectrum and EDA-to-feature decision evidence. §5 represents the causal state and correction as native equations and a diagram. §§8–10 present tables and plots with their scopes and units. |
| Algorithms understood | 8% | — | §§2 and 5 discuss established models and derive PI-SRC states, augmented fitting, intercept and group penalties. §7 covers actual comparator families and state assumptions. See src/src_gearbox/src_model.py and results/metrics/baseline_configs.csv. |
| Algorithm performance | — | 32% | §8 covers 27 common-target rows, all 21 required comparators, CPU runtime and dependent errors. §§9–10 cover matched fault masks, pulse preservation, RPM intervention and attribution. See results/metrics/all_model_comparison.csv, results/timings/benchmark_table.csv and results/fault/. |
| Algorithm novelty | 4% | 16% | §§4–6 demonstrate physical angular history, finite order states, grouped adjacent and cross-history interactions, guarded optimization and readout. §10 retains adverse comparisons. See results/eda/, results/optimization/trials.csv and results/ablations/. |
| Total | 20% | 80% | Both components comprise the full grading rubric. |

The separate earlier presentation artifacts support the same assessment and are outside this Word build. Section numbers here follow the requested final-report structure. The generated Word contents and captions provide final document navigation.
