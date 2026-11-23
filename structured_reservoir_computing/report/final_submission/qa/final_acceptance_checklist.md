# Final report acceptance checklist

| Requirement | Status | Evidence |
| --- | --- | --- |
| Course rubric and objectives reflected | PASS | `evidence/rubric_traceability.md`, Chapters 1 and 11 |
| Original SRC research outputs examined | PASS | `evidence/source_inventory.csv`, `evidence/model_registry.csv` |
| Appropriate terminology throughout | PASS | `report_source.md`, Table 5.1, `check_document.py` |
| Implemented physics and structured features derived | PASS | Chapter 5, `evidence/src_method_numeric_check.json` |
| Causal roles, guards and normalization explained | PASS | Chapters 3, 5 and 6 |
| EDA connected to chosen feature groups | PASS | Chapter 4, Figures 4.1–4.2 |
| Executed training-only optimization history | PASS | Chapter 6, `assets/tables/trials.csv`, `assets/tables/fold_scores.csv` |
| 21 comparator families accounted for | PASS | Chapter 7, Table 8.1, Appendix A |
| Common-mask scoring and efficiency checks | PASS | Chapter 8, `qa/numeric_consistency_audit.md`, `assets/tables/runtime.json` |
| GPU measurement | NOT AVAILABLE | Frozen CPU-only benchmark and the hardware scope in Chapter 8 |
| Published and local sparse LPV results scoped | PASS | Appendix B, `evidence/discrepancy_log.md` |
| Overlap and provisional fault evidence | PASS | Chapters 3 and 9, `evidence/dataset_provenance.md` |
| Independent prediction, fault labels and MATLAB angular parity | NOT AVAILABLE | Chapters 3, 9 and 11. Blind records, independently confirmed conditions, and the MATLAB angular helper were absent. |
| Ablation, robustness and adverse comparisons included | PASS | Chapters 8–10, `assets/tables/src_ablations.csv`, `assets/tables/rpm_stress.csv` |
| Claims and numbers traceable | PASS | `evidence/claim_to_evidence.md`, `evidence/results_ledger.csv`, `qa/numeric_consistency_audit.md` |
| Verified citation and DOI metadata | PASS | `evidence/references_audit.md`, `evidence/bibliography_index.json` |
| Editable Word report and faithful PDF | PASS | `qa/document_structure.json`, `qa/layout_and_accessibility_audit.md` |
| Rebuild instructions, documented dependencies and source | PASS | `FINAL_REPORT_README.md`, `report_generation/README.md`, `report_source.md` |
| All 53 pages rendered and reviewed | PASS | `qa/layout_and_accessibility_audit.md`, `qa/page_map.json` |
| Archive and checksums | PASS | `PI_SRC_Report_Submission_Package.zip`, `SHA256SUMS.txt`, `PI_SRC_Report_Submission_Package.zip.sha256` |
