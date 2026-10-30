# Authoritative course requirements and open questions

Source: user-supplied **MECG 7780 Course Project - Score Allocation Updated.pdf**, pages 1–2, independently extracted on 2026-10-30. The PDF takes precedence over inconsistent percentages in the master implementation prompt.

## Confirmed scope and obligations

The exact title is **Time Serial Dependency Modeling via Advanced Machine Learning Algorithms** for MECG 7780 Engineering Data Analytic. Predict vibration sample `x[t]` from past vibration samples; inclusion of simultaneously measured speed is encouraged, while vibration-only models are allowed. Accuracy is measured by **MSE**, computational efficiency by **CPU/GPU time**, and **fault-detection performance** is also requested. The course describes angular-resampled PMI with supplied MATLAB code. Deliver code with a README naming the execution platform and required software, a report, an **8-minute progress presentation**, and a **12-minute final presentation**.

The PDF says the data are the same as the lecturer's cited paper. This is course-source attribution, not an independent demonstration that our available split, sample indices, normalization and evaluation conventions reproduce that paper's experiments.

## Updated rubric — exact PDF percentages

| Assessment | Organization / presentation | Presentation clarity | Algorithms understood | Algorithm performance | Algorithm novelty | Total |
|---|---:|---:|---:|---:|---:|---:|
| Progress | 4% | 6% | 8% | — | 2% | 20% |
| Final | 16% | 24% | — | 30% | 10% | 80% |

The master prompt's progress 4/4/8/4 and final 16/16/32/16 tables conflict with this updated PDF. They are not used as the assessment rubric in this release. These are project-mark percentages as printed, not an inferred grade conversion.

## Date conflict — do not silently correct

Page 1 states: “The progress presentation is scheduled on **Nov 4, 2026**,” and “Course project report is due on **Dec 7, 2025, 24:00**.” Page 2 states: “The final presentation ... is scheduled for **Dec 9, 2025**.” The final report/presentation dates precede the progress date. The PDF also says `sigtest.mat` and `sigfault2.mat` would be posted on Dec 2, 2025; that does not establish present availability.

**Final submission and presentation dates are unconfirmed.** The presentation configuration and slide labels use `FINAL_DATE_UNCONFIRMED`; no alternative year is invented. Author/group names remain user-editable fields.

## External prerequisites and open scientific questions

1. The original oversized `sigfault1.mat` upload was superseded by **twenty smaller split MAT files**, which are now available in `data/raw/sigfault1_parts/`. The current [data audit](../reports/data_audit.json) and finalized [candidate-record study](../reports/fault_study_summary.json) establish the actual independently processed record count and structures. The previous transfer obstacle is resolved; no further upload is required for this dataset. Descriptive raw/residual and angular-unit-sensitivity evidence must still be distinguished from physical fault-detection accuracy.
2. `sigtest.mat` and `sigfault2.mat` were not supplied. All measured prediction scores are **validation scores**, including model/epoch selection on that validation trace; they are not unbiased held-out test estimates.
3. The physical speed unit and vibration calibration are not encoded in the verified fields. Plots use speed in **source units** and supplied normalized vibration as **dimensionless**. Angle-domain analysis must label RPM/rps/rad/s assumptions.
4. No supplied accessible source verifies physical health-state/fault-onset/severity labels. The training condition is a **reference condition**, not proven healthy.
5. Full methods and experimental details of Chen et al.'s Spa LPV-ARMA article were not accessible. Our polynomial speed-conditioned AR is a related **LPV-AR surrogate**, not an exact reproduction. No outperformance claim against Spa LPV-ARMA is made.
6. Native Windows/macOS execution and Microsoft PowerPoint are not available in this Linux session. Cross-platform commands are documented, but only Linux CPU execution is empirically verified. Both PPTX decks were successfully exported by LibreOffice, rasterized and visually inspected; that establishes tested LibreOffice rendering, not Microsoft PowerPoint/font-layout parity.

## Citation printed in the course source

Y. Chen, Z. Li, Y. Jiang, D. Gong, and K. Zhou, “Sparse LPV-ARMA model for non-stationary vibration representation and its application on gearbox tooth crack detection under variable speed conditions,” *Mechanical Systems and Signal Processing*, 224, 112161, Feb. 2025. DOI: [10.1016/j.ymssp.2024.112161](https://doi.org/10.1016/j.ymssp.2024.112161). Metadata here is verified against the course PDF, not a successful publisher/full-text fetch. See the [research access log](../research/source_access_log.md).
