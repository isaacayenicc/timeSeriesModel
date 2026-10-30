# Progress Presentation 8min speaker script

Total planned speaking time: 480 seconds (8.0 minutes). Slide times are estimates, not a guarantee of delivery pace.

## Slide 1. Predicting variable-speed vibration — 35 seconds

Introduce the engineering aim: predict the next sample causally, compare accuracy and cost, then inspect raw and residual diagnostic descriptors. The complete split upload now enables independent analysis of all 200 fault-candidate records. These are descriptive results without verified physical fault labels, not a validated diagnostic classifier.

Evidence: `reports/results_summary.csv`; `experiments/runs/*/config.json`

## Slide 2. A causal prediction task — 45 seconds

Point to the ordering. For lag L, x[L] sees x[0:L], and speed inputs stop at L−1. Teacher forcing means previously measured vibration is available; this is not a free-running forecast. For causal models, changing future samples cannot change earlier predictions.

Evidence: `docs/course_requirements_and_open_questions.md`; `reports/data_audit.json`; relevant report section

## Slide 3. What the accessible data establish — 50 seconds

The two reference/validation files each contain 24,000 samples at 3,200 Hz. The newly available twenty MAT parts preserve 200 separate candidate records, each 32,000 samples or 10 seconds. No records are flattened together. No second normalization is applied to supplied fault x. Speed physical units and health labels remain unverified.

Evidence: `docs/course_requirements_and_open_questions.md`; `reports/data_audit.json`; relevant report section

## Slide 4. Operating speed changes the signal — 45 seconds

Describe the actual speed trajectory and amplitude/frequency variation without assigning unverified gear tooth counts or shaft units. The two speed traces are nearly identical. Correlation is not proof of leakage, but it sharply limits conclusions about unseen operating histories.

Evidence: `docs/course_requirements_and_open_questions.md`; `reports/data_audit.json`; relevant report section

## Slide 5. A fair evaluation contract — 50 seconds

Every score uses the same 23,744 validation targets after a 256-sample burn-in, so changing lag does not change the scored sample population. Validation also selects neural epoch and model configuration; it is therefore optimistic for final generalization. No test file exists here.

Evidence: `docs/course_requirements_and_open_questions.md`; `reports/data_audit.json`; relevant report section

## Slide 6. Strong baselines come first — 55 seconds

Explain ridge as squared-error fitting with a coefficient-size penalty. The LPV-AR surrogate expands past vibration with speed and speed-squared interactions, but has no moving-average innovation recursion or lecturer-paper sparse estimation. It is not Spa LPV-ARMA. No published score is invented.

Evidence: `reports/results_summary.csv`; `experiments/runs/*/config.json`

## Slide 7. Compact nonlinear alternatives — 55 seconds

MLP maps a finite history to a scalar; the GRU summarizes a unidirectional sequence with update/reset gates; the TCN uses left-only padded convolutions and increasing dilations. Two convolutions per block give receptive field 1+4 times the sum of dilations. Limited training budgets must be disclosed.

Evidence: `docs/course_requirements_and_open_questions.md`; `reports/data_audit.json`; relevant report section

## Slide 8. A falsifiable speed-conditioning proposal — 55 seconds

FiLM is an established conditioning mechanism. Here each past temporal feature receives a bounded gain and offset from that position's observed speed. Compare to unconditioned same-core TCN, late speed concatenation, a larger unconditioned capacity control and an explicitly off-line phase-shifted-speed negative control. The shifted control is excluded from causal recommendations.

Evidence: `docs/course_requirements_and_open_questions.md`; `reports/data_audit.json`; relevant report section

## Slide 9. Residuals are diagnostic candidates — 55 seconds

The angle-domain indicators are computed under explicitly named RPM, revolutions-per-second and radians-per-second assumptions; none establishes the true speed unit. The literal MATLAB coarse index selects lag T minus one while the corrected branch selects T. The sensitivity plot compares actual record descriptors across these choices, not physical fault probabilities.

Evidence: `docs/course_requirements_and_open_questions.md`; `reports/data_audit.json`; relevant report section

## Slide 10. Evidence, blockers and next steps — 35 seconds

Summarize completed audit, trained and saved causal predictors, all 200 candidate records processed independently, tested numerical features and reproducible plots. Remaining needs are calibrated speed units, physical health labels and independent test data, plus clarification of the course date conflict. No additional upload is needed for the completed 200-record study.

Evidence: `docs/course_requirements_and_open_questions.md`; `reports/data_audit.json`; relevant report section
