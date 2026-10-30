# Project gates and evidence — 2026-10-30

Status is for this execution, not publication or instructor acceptance.

| Gate | Status | Evidence |
|---|---|---|
| G0 scope, PDF, corrected rubric and conflicting dates | passed | course_requirements_and_open_questions.md |
| G1 training/validation structural audit and normalization | passed with warnings | ../reports/data_audit.json; ../data/manifest.json |
| G1 fault MAT forensic audit | passed | ../reports/fault_parts_verification.json; ../reports/fault_data_audit.csv |
| G2 causal alignment and common-index protocol | passed | ../tests/test_core.py; normalization_and_leakage_audit.md |
| G3 fitted baselines and persisted artifacts | passed | ../reports/baseline_comparison.csv |
| G4 advanced training | passed bounded CPU experiment | ../reports/results_summary.csv; ../models/checkpoints_manifest.json |
| G5 speed conditioning and context/seed controls | passed experiment; positive neural-benefit hypothesis unsupported | ../reports/model_ablation.csv |
| G6 diagnostic mathematical and synthetic verification | passed | ../tests/test_diagnostics.py; ../reports/numerical_validation.md |
| G6 actual fault-record analysis | passed descriptive scope | ../reports/fault_study_summary.json; 200separate records,13 predictors,2800/8400feature rows |
| G7 timing, within-trace uncertainty and model selection | passed within stated scope | ../reports/timing_and_hardware.md; ../reports/within_trace_uncertainty.csv |
| G8 report, notebook, companion and two slide decks | artifact generation and validation tracked in final checklist | FINAL_ACCEPTANCE_CHECKLIST.md |
| G9 release checks and reproduction | see recorded final outcomes | ../reports/reproducibility_check.md; ../reports/release_verification.json |
| Literature full domain review and article reproduction | blocked/incomplete | ../research/source_access_log.md |

No external test file, fault health labels, physical speed units, or exact conventional-model benchmark is invented. This is a substantial partial scientific delivery with a completed prediction and descriptive200-record experiments, not unconditional completion of every master-prompt acceptance item.
