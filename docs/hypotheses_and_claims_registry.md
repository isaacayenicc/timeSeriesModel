# Research questions and claims registry

This internal protocol was documented during implementation, before completion of the fixed experimental suite. It is not an externally timestamped preregistration. Architecture families and bounded search are in `config/experiment_plan.json`. Model selection and early stopping use the same supplied validation trace; performance is provisional and subject to selection optimism.

| Question | Testable hypothesis | Evidence and interpretation |
|---|---|---|
| RQ1 prediction | At least one learned predictor reduces common-index MSE relative to zero and persistence | `reports/results_summary.csv`; compare measured rows, not a published score |
| RQ2 efficiency | Compact linear predictors can offer a favorable accuracy/cost trade-off | measured training wall and end-to-end record inference, parameter count, artifact bytes |
| RQ3 speed | Polynomial speed interactions improve AR and FiLM improves the corresponding TCN | `reports/model_ablation.csv`; distinguish linear improvement from neural failure and seed variability |
| RQ4 diagnostics | Residuals reveal physical faults more clearly than raw signals | The later verified twenty-part input supports all 200-record raw/residual descriptors. Physical superiority remains inconclusive without labels/unit calibration; see full fault feature tables. Synthetic examples are code validation only |
| RQ5 robustness | Ranking persists across context, seed and speed bins | context32/64/128 (and linear256), TCN64 seeds17/29/43; `seed_variability.csv`, `speed_bin_errors.csv`. No independent-machine generalization claim |

The experiment includes an offline rolled-speed negative control `shift_tcn`. Rolling the complete speed trace uses future information at some indices. It is deliberately an inappropriate conditioning control, is labelled noncausal, and is excluded from causal-model recommendation. It has no future vibration inputs. All primary models observe vibration and optional speed strictly before the target.

No claims of world-first novelty, exact Spa LPV-ARMA reproduction, independent-test scores, healthy-state certainty, known speed physical units, physical-fault AUC/F1 or GPU timings are supported. The 200-record descriptive study is now executed and verified.
