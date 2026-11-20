# Conventional one-step forecasting

The target is the standardized vibration sample at index $t$. Vibration history and scheduling speed end at $t-1$ for every native comparison. Published validation values describe the second 7.5 s segment of a healthy acquisition selected for the paper. The supplied development record starts near 124.6 RPM. The external plots show a different speed phase (Chen et al., 2025, §5.1 and Fig. 13(c)). The run uses the supplied development record once, with target indices 1,792 through 23,999 inclusive.

## Differencing and integration

For $d_t=x_t-x_{t-1}$, the integrated autoregression predicts $\hat x_t=x_{t-1}+\sum_i a_i d_{t-i}$. An ARI(275,1) uses 275 coefficients. The native conditional ARIMA uses 285 differenced lags and five lagged innovations from a separately fitted AR(310) on the differences. The first difference depends on one earlier raw vibration sample. The 315-lag innovation history therefore has a raw support of 316 vibration samples including $t-316$. The archived ARIMA additionally uses an invertible recursive innovations filter from the beginning of each record. Its published order split and exact initialization were unavailable. The native coefficients have a maximum characteristic root magnitude of 0.9620 for the moving-average part, and the native forecast uses explicit finite proxy innovations.

## Speed scheduling and lag design

Let $w_{t-1}=(n_{t-1}-n_\mathrm{lo})/(n_\mathrm{hi}-n_\mathrm{lo})$ with calibration extrema carried in the training MAT metadata and checkpoint. A row of the LPV-AR design is

$$X_t=[x_{t-1}(g_0,\ldots,g_{P-1})(w_{t-1}),\dots,x_{t-a}(g_0,\ldots,g_{P-1})(w_{t-1})].$$

The stage-II row appends analogous columns for $e^{(1)}_{t-j}$, where $e^{(1)}_s=x_s-\hat x^{(1)}_s$ is the completed stage-I forecast error. Replacing those regressors with errors from the final stage-II output produces a different estimator. With $(a_1,a_2,b)=(240,135,75)$ the strict sparse cascade accesses at most $\max(a_2,a_1+b)=315$ past samples. The 1,792-sample guard covers both the cascade and the differenced models. The training MAT speed scaling is verified and reused unchanged. The supplied vibration has already been standardized.

The dense and new sparse LPV fits use $g_j(w)=P_j(2w-1)$, with eight Legendre terms per lag including the constant term. The archive uses monomials for its sparse checkpoint. The two complete polynomial bases span the same space, while LASSO sparsity is coordinate-dependent. The intercept convention is zero intercept with a constant term in each coefficient curve. The code leaves columns at their design scale, so regularization reflects both the lag and Legendre coordinate.

## Identification

The candidate AR orders are 96, 160, 220, 240 and 260. Stage II considers AR orders 55, 80, 110 and 135 and MA orders 30, 50, 65 and 75. Within each split all candidates use identical rows in the AIC formula $N\log(\mathrm{MSE})+2d$. The scan uses shared Gram statistics in Legendre coordinates plus a factor-diagonal conditioning check. It falls back to a rank-aware least-squares solve when required. Final dense and sparse support refits use rank-revealing QR with no explicit matrix inverse.

Within each guarded training fold both the order scan and stage-I proxy coefficients are estimated from permitted training rows. A validation row can use the measured validation vibration through its previous target as teacher-forced one-step prediction. Training feature histories never enter the held-out target interval. The LASSO solves $\|X\beta-y\|_2^2/(2N)+\lambda\|\beta\|_1$. The matching unnormalized penalty is $N\lambda$. ADMM alternates a Cholesky least-squares update, soft thresholding and a scaled multiplier update. The convergence tests combine absolute/relative primal and dual residuals with a dual-feasible objective gap. Grid values are fractions of the fold-specific $\|X^\top y/N\|_\infty$ and span 0.1 to 0.002. The largest converged penalty fraction whose mean fold loss is within one sample standard deviation of the minimum mean is selected. QR refits the retained support afterward.

All 36 guarded ADMM paths at the evaluated penalty points converged. The selected full fits retain 475 of 1,920 stage-I coefficients and 365 of 1,680 stage-II coefficients. Deployment uses 840 active coefficients for the complete cascade. The stage-I and stage-II normalized objective gaps are $1.17\times10^{-5}$ and $4.44\times10^{-6}$. The stage-II AIC optimum lies on the upper edge of both declared ARMA grids. Wider ranges are a remaining sensitivity check. Dense-proxy stage II uses orders $(110,65)$, so the dense and sparse benchmarks differ in order, fitted coefficients and proxy rank.

## Historical random-row diagnostic

A separate current-RPM fit uses five shuffled folds at the verified archived orders 240 and (110, 65), a common start of 316, and eight shifted Legendre coordinates. It reuses the training-only proxy fitted to the whole training record for stage II. Row overlap and proxy estimation cross random fold boundaries. Its CV scores describe the historical tuning procedure and cannot support a generalization claim. The supplied validation remains external to that identification step. A one-SD, converged ADMM selection and QR refit retain 1,199 stage-I and 737 stage-II terms. This experiment uses the same supplied signals and a declared basis interpretation, with finite grids and frozen archived order candidates.

The exact same bounded search with printed monomials retains 120 stage-I and 52 stage-II terms and scores 0.424537 and 0.439783. Lower-penalty monomial candidates often run into the 800-iteration fold limit and are excluded. This controlled stress shows the joint effect of a basis-dependent penalty and numerical convergence. The selected monomial supports passed the stated criteria, and the result cannot establish that a wider monomial search or changed solver tolerance has the same performance.

## Development results and reconciliation

| Model | Published Table 6 | Archived replay, start 316 | New random-row fit, start 316 | Native strict, start 1,792 |
|---|---:|---:|---:|---:|
| ARI | 0.3658 | 0.415816 | — | 0.425069 |
| ARIMA | 0.3671 | 0.415719 | — | 0.425223 |
| Sparse LPV-AR | 0.2911 | 0.340886 | 0.341468 | 0.381496 |
| Sparse LPV-ARMA | 0.2893 | 0.336901 | 0.334422 | 0.371406 |
| Dense LPV-AR | — | — | — | 0.344585 |
| Dense LPV-ARMA | — | — | — | 0.330781 |

At start 1,792, the archived sparse LPV-ARMA scores 0.345151 with its original current-speed schedule. Recomputing the score with the same coefficients and speed delayed by one sample gives 0.345133. The maximum forecast change is 0.002285. On this record, the observed difference between archived and native sparse methods therefore comes chiefly from model identification, basis and penalty choices. The strict dense fit outperformed the strict sparse cascade. The native one-SD sparse support is smaller than the archived or published support, and a conservative penalty may reduce predictive flexibility. A guarded one-record CV and sparse-basis dependence are material differences from the historical random-row convention.

The published sparse LPV-ARMA runs-test $p=0.6233$ is close to 0.60410 from the new historical-convention fit and differs from 0.05460 in the current-speed archival replay, both starting at 316. The new fit still exceeds the published MSE by 0.0451 on the supplied segment. This test uses runs of residual signs and checks a limited aspect of serial dependence. Segment phase, speed conditions, residual sampling and coefficient differences all warrant scrutiny. The optional retrospective validation-window archive and original 20 s source acquisition were unavailable. The prior simulation discrepancy, including 0.52423 measured archived MSE versus 0.3570 in the paper and weak F1 discrimination, is historical evidence and was not rerun for this project.

The polynomial basis wording and displayed monomials appear on paper p. 3 around Eqs. (1)–(3). The two-stage identification is in §3.2, Eqs. (7), (9)–(10) on pp. 5–6 and the ADMM details in Appendix A, pp. 19–20. Experimental design and published values are in §§5.1–5.2 and Tables 5–6, pp. 14–17. Chen's 2020 thesis discusses PMI in §2.2.3, pp. 45–50 and sparse FP-AR selection in §§3.2.2–3.2.3, pp. 76–79.

Use the shared prediction and metrics artifacts to verify the reported numbers. The comparison table is in `tables/lpv_paper_comparison.csv` under the development run. CSV order scans, full solver diagnostics, unit tests and causal perturbation checks show the decisions from data to prediction. Batch timing covers the complete forecast on a 24,000-sample record including warm-up. It does not establish sample-by-sample streaming latency.

From the repository root, run the independent publication-convention diagnostic with:

```bash
python scripts/paper_row_diagnostic.py --data-dir data/raw --run-dir outputs/runs/paper_diagnostic --basis legendre
```

For the causal comparison, run `python scripts/reproduce_baselines.py --data-dir data/raw --run-dir outputs/runs/conventional`. Add `--reuse` to replay its existing conventional parameters and regenerate predictions and figures. The CPU measurements in `metrics/conventional_timing.csv` report one warm-up call followed by seven repeats at one BLAS thread. The batch reference preserves zero sparse coefficients in a dense matrix layout, so its measured cost does not directly express sparse implementation potential.
