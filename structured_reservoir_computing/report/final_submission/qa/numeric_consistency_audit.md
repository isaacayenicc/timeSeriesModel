# Numerical consistency audit

## Executed verification

- Verified 27 common-mask prediction rows, including all 21 required comparators, from three frozen NPZ releases. Recalculated MSE, RMSE, MAE and bias. Largest absolute tabulation difference 1.11e-16.
- Verified all 25 SRC variants, all local statistical-reference forecast windows, and matched adapter contrasts from saved arrays. The original training estimates remain sourced to the stored in-sample evidence.
- Reproduced all 78 common-window paired moving-block intervals with the recorded seed, block lengths and replicate counts. The intervals describe dependence within the same development record.
- Checked 41 executed optimization candidates and 123 guarded fold rows. Recalculated the mean fold components and objective for every candidate.
- Recalculated all 24 provisional group medians, differences, margins and ranking AUCs from the actual per-record CSV. Reproduced each provisional median-gap interval and its 20 paired gap contrasts from 1,000 subject-ID replicates. 198 of 200 subjects were eligible. Checked paired intervention ratios and synthetic pulse summary medians.
- Verified underlying file/record hashes and exact complete joint overlaps for all 202 source records. The only full matches were [('sigfault1:061', 'sigtrain', 0), ('sigfault1:063', 'sigvalid', 0)].
- Rehashed the frozen artifacts for 51 model registry rows and indexed 518 source files/manifest entries.
- Checked 879 stored feature columns by group, the intercept and stage-one coefficient inventory, exact base parameter equality for adapter on/off, and the primary timing median and IQR from seven repeats. The selected solver record is `cholesky` with Gram condition estimate 23404.489956.
- Checked 6850 exact/traceable scalar records in the generated results ledger.
- Frozen primary replay recovered training MSE and all 22,208 development predictions. This was one inference-only checkpoint check.
- Abstract and conclusion in `report_source.md` checked against source metrics.

## Preserved scoped findings

Primary development MSE 0.329973322560. RMSE 0.574433044454. MAE 0.456411558329. Training in-sample MSE 0.308687629550. Target indices [1792, 24000), mask `1522c0a1b184b971b59763df3d827515370162751cc61209f025765039a89fb1`.

| Comparator | MSE | Comparator minus selected | Improvement, comparator denominator |
| --- | ---: | ---: | ---: |
| linear | 0.367385521350 | +0.037412198790 | +10.183362% |
| sparse_lpv_arma | 0.371405516981 | +0.041432194421 | +11.155514% |
| cnn | 0.394638049132 | +0.064664726571 | +16.385832% |
| archived_src_corrected | 0.317787705841 | -0.012185616720 | -3.834515% |

Recovered train, development and archive timings retain their saved scope. No GPU, complete independently classified fault cohort or independent prediction result entered the comparison.

The paper values were checked against Tables 5 and 6 of the original PDF separately from local rows. The DOI and original scope remain attached to those rows.
