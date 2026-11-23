# Scientific limitations

- The design has one designated training recording and one held-out development recording. Within-record folds and block resampling do not estimate between-machine or between-acquisition performance.
- Causal one-step predictors consume true observed history from the recording. Free-running forecasting and hard real-time streaming performance were outside the measured task.
- Fault-candidate records 61 and 63 duplicate the official training and development signals. The remaining diagnostic groups are index-defined and have no verified condition labels. Any AUC is conditional on that grouping. The project gives no externally validated crack classification or operating threshold.
- Neither blind MAT input was supplied. Frozen commands perform inference only, record hashes and preserve previously scored blind inputs.
- The software implements lecturer-compatible PMI and synthetic conformance checks. Exact MATLAB helper/runtime parity remains unverified where only the supplied partial reference is available.
- All reported compute timings use CPU. Complete-record batch speed includes prediction feature construction, subject to runtime conditions. CUDA and production GPU timing remain unmeasured.
- Pure clean MSE, robustness-aware utility and the archived SRC can favor different models. Physical bank selection, nonlinear readout, augmentation and residual correction are reported separately where the design isolates them.
- Archived checkpoint families and the independent journal-paper experiment are different kinds of evidence. Source training budgets and comparator input/architecture regimes vary even though development scoring targets are identical.
