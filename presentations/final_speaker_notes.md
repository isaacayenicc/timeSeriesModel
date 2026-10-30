# Final Presentation 12min speaker script

Total planned speaking time: 720 seconds (12.0 minutes). Slide times are estimates, not a guarantee of delivery pace.

## Slide 1. Causal prediction under variable speed — 29 seconds

Introduce the engineering aim: predict the next sample causally, compare accuracy and cost, then inspect raw and residual diagnostic descriptors. The complete split upload now enables independent analysis of all 200 fault-candidate records. These are descriptive results without verified physical fault labels, not a validated diagnostic classifier.

Evidence: `reports/results_summary.csv`; `experiments/runs/*/config.json`

## Slide 2. Data, provenance and limits — 44 seconds

The two reference/validation files each contain 24,000 samples at 3,200 Hz. The newly available twenty MAT parts preserve 200 separate candidate records, each 32,000 samples or 10 seconds. No records are flattened together. No second normalization is applied to supplied fault x. Speed physical units and health labels remain unverified.

Evidence: `docs/course_requirements_and_open_questions.md`; `reports/data_audit.json`; relevant report section

## Slide 3. Causality before model complexity — 49 seconds

Point to the ordering. For lag L, x[L] sees x[0:L], and speed inputs stop at L−1. Teacher forcing means previously measured vibration is available; this is not a free-running forecast. For causal models, changing future samples cannot change earlier predictions.

Evidence: `docs/course_requirements_and_open_questions.md`; `reports/data_audit.json`; relevant report section

## Slide 4. The conventional comparison — 49 seconds

Explain ridge as squared-error fitting with a coefficient-size penalty. The LPV-AR surrogate expands past vibration with speed and speed-squared interactions, but has no moving-average innovation recursion or lecturer-paper sparse estimation. It is not Spa LPV-ARMA. No published score is invented.

Evidence: `reports/results_summary.csv`; `experiments/runs/*/config.json`

## Slide 5. Three compact neural ideas — 54 seconds

MLP maps a finite history to a scalar; the GRU summarizes a unidirectional sequence with update/reset gates; the TCN uses left-only padded convolutions and increasing dilations. Two convolutions per block give receptive field 1+4 times the sum of dilations. Limited training budgets must be disclosed.

Evidence: `docs/course_requirements_and_open_questions.md`; `reports/data_audit.json`; relevant report section

## Slide 6. Speed-conditioning: contribution and controls — 54 seconds

FiLM is an established conditioning mechanism. Here each past temporal feature receives a bounded gain and offset from that position's observed speed. Compare to unconditioned same-core TCN, late speed concatenation, a larger unconditioned capacity control and an explicitly off-line phase-shifted-speed negative control. The shifted control is excluded from causal recommendations.

Evidence: `docs/course_requirements_and_open_questions.md`; `reports/data_audit.json`; relevant report section

## Slide 7. Measured validation performance — 69 seconds

Read the table as measured validation performance, not test performance. The top row is an observed best configuration and its selection used this validation trace. Inspect the full CSV for all seeds and contexts rather than interpreting a single winning row as a population estimate.

Evidence: `reports/results_summary.csv`; `experiments/runs/*/config.json`

## Slide 8. Accuracy has a computational cost — 54 seconds

Report CPU wall time and the actual timing boundary. The public inference path includes checkpoint deserialization, window preparation and full-record prediction; one warm-up and three repeats are used. These are not isolated kernel latencies. Linear fitting is float64 and neural fitting float32. GPU time was not measured.

Evidence: `reports/results_summary.csv`; `experiments/runs/*/config.json`

## Slide 9. Does conditioning help consistently? — 64 seconds

The seed summary reflects initialization variability, not machine-to-machine variability. Context effects are compared on common target indices. A speed gain can depend on capacity, optimization and nearly matched operating histories. If the speed model loses, retain the result; novelty is a falsifiable experimental proposal.

Evidence: `reports/results_summary.csv`; `experiments/runs/*/config.json`

## Slide 10. A residual does not prove a defect — 44 seconds

Every supplied fault-candidate record is scored using previously trained reference predictors with no fault-data retraining. All models and raw features use the same 256-sample burn-in and remaining 31,744 samples per record. The displayed RMS trajectories are descriptive: record-index trends are not fault-onset labels, and reduced RMS is not detection accuracy.

Evidence: `docs/course_requirements_and_open_questions.md`; `reports/data_audit.json`; relevant report section

## Slide 11. PMI and angular resampling need assumptions — 54 seconds

The angle-domain indicators are computed under explicitly named RPM, revolutions-per-second and radians-per-second assumptions; none establishes the true speed unit. The literal MATLAB coarse index selects lag T minus one while the corrected branch selects T. The sensitivity plot compares actual record descriptors across these choices, not physical fault probabilities.

Evidence: `docs/course_requirements_and_open_questions.md`; `reports/data_audit.json`; relevant report section

## Slide 12. Unlabeled records: measured descriptors — 39 seconds

All candidate records now have real raw/residual feature measurements. Compare the displayed distributions and note speed dependence, normalization and equal segment length. There are no verified healthy/faulty classes, onset dates or severity values. A model with small residual MSE can also suppress periodic content. Consequently no ROC, F1 or sensitivity is justified and diagnostic model choice must remain provisional.

Evidence: `docs/course_requirements_and_open_questions.md`; `reports/data_audit.json`; relevant report section

## Slide 13. Limits of inference — 44 seconds

The transfer blocker has been resolved using the twenty split MAT files. Remaining limitations are unknown health labels and physical speed unit, validation reuse for model selection, only one short validation trace and nearly matched speed histories, and incomplete publisher access for exact Spa LPV-ARMA replication. Descriptive fault-record patterns do not resolve these inferential limits.

Evidence: `docs/course_requirements_and_open_questions.md`; `reports/data_audit.json`; relevant report section

## Slide 14. What the evidence supports — 49 seconds

Answer RQ1 using the saved common-index validation table, and RQ2 using the measured CPU costs. Speed-FiLM does not improve mean validation MSE across the matched seeds. RQ4 now has real record-wise raw/residual and angular-assumption evidence, but physical fault accuracy remains unidentifiable without labels. Recommend inspecting both raw and model residual descriptors rather than declaring a physical fault winner.

Evidence: `reports/results_summary.csv`; `experiments/runs/*/config.json`

## Slide 15. Sources and reproducibility — 24 seconds

Direct the audience to the verified literature matrix and access log. The course citation motivates speed-dependent modeling; author repositories support TCN and FiLM implementation concepts. Original TCN and FiLM studies were not gearbox validation studies. All claims here map to stored tables, configurations and figures.

Evidence: `docs/course_requirements_and_open_questions.md`; `reports/data_audit.json`; relevant report section
