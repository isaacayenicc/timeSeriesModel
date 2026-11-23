# 9. Residual periodicity and tooth-crack evidence

## 9.1 Why the residual and its envelope are distinct

A tooth that engages once per input-shaft revolution can excite a damped response whose waveform varies with transmission path, torque and speed. When pulse timing remains coherent in angle, energy can be dispersed in time frequency while envelope modulation remains concentrated near shaft order one. Residual prediction changes this mixture. Regular mesh content may be suppressed, a repeated fault-related component may be removed in part, and a phase error in the predictor may generate its own periodic residual. McFadden’s gear work on mesh demodulation and later variable-speed LPV studies motivate inspecting the residual envelope at a physically specified order. Cyclostationary analysis provides a wider explanation for why repeated carrier effects become apparent in the envelope. In the present experiment order-one periodicity is a feature of the signal, and attribution to a particular defect requires labelled measurements. [@mcfadden1986gears; @antoni2009cyclo; @abboud2017envelope; @chen2021crack; @chen2025sparse]

For each record, the frozen predictor is initialized independently. Retained outputs start at zero-based sample 1,792. The residual is measured vibration minus the causal one-step estimate. The diagnostic then integrates measured speed over the retained time grid and linearly interpolates onto $S=2048$ uniform samples per input-shaft revolution. This diagnostic calculation is offline. It is distinct from the forecasting-phase extrapolation in Equation (5.4) and it does not enter the fitted predictors. On the resulting angular trace $r_j$, the lecturer-compatible analytic magnitude and its centred version are

$$
A_j=\left|(r_j-\overline r)+\mathrm{i}\mathcal H[r-\overline r]_j\right|+\overline r,
\qquad a_j=A_j-\overline A.
$$

The centred envelope $a_j$ is used with finite-record correlations. If its length is $L_\theta$, then the lag product and indicator are

$$
C_\ell=\sum_{j=0}^{L_\theta-1-\ell} a_j a_{j+\ell}, \qquad
\rho_\ell=\frac{C_\ell}{C_0}, \qquad
\mathrm{PMI}_\ell=\frac{C_\ell}{C_0-C_\ell}
=\frac{\rho_\ell}{1-\rho_\ell}.
$$

The finite sum uses fewer pairs at nonzero lags than at zero. PMI is dimensionless, can be negative and grows rapidly as the normalized correlation approaches one. A zero-energy envelope or degenerate denominator is flagged. The direct physical one-turn setting is $\ell=S$. The supplied MATLAB main branch takes element $S$ of a one-based autocorrelation array whose first entry is lag zero. The lecturer-compatible result therefore uses $\ell=S-1=2047$. The project separately computes the exact-lag variant and the direct Hilbert-envelope variant. In a reference fixture with eight rotations, the exact one-turn score was 7 and the literal row gave 7.00958. The fixture verifies the arithmetic and sensitivity to the lag convention. Exact numerical parity with the original MATLAB angular-resampling helper remains unverified because the helper and numerical reference outputs were not supplied. No healthy-reference rescaling, polynomial detrending, bandpass, padding or extra end trimming is applied in the local primary branch. [@zhao2017pmi; @chen2020vibration; @chen2025sparse]

The source paper interprets PMI in terms of repeated pulse energy under assumptions about pulse shape, periodicity and disturbance correlation. In local finite records a number can reflect periodicity arising from either defect dynamics or known operating variations. An envelope spectrum locates that structure and a companion waveform shows its magnitude. Figure 9.1 illustrates the physical coordinate and visible model dependence of the residual under the published display rule. These display records are case illustrations with provisional file grouping and no asserted defect identities.

FIGURE_9_1

## 9.2 Real candidate acquisitions and conditional separation

The full diagnostic matrix evaluated six raw or residual series across all 200 supplied candidates. Each was passed through the same angular analysis and exclusion protocol. Only one-based records 61 and 63 were dropped for exact overlap with the two official modelling records, leaving 98 eligible members of A and 100 of B. The distinction between the groups is their original index range. None of the supplied acquisition metadata establishes a health-state mapping. The gap reported here is median PMI of B minus median PMI of A. The extremal margin is the smallest score in B minus the largest score in A. Both increase when the provisional groups separate more strongly in the hypothesized direction.

TABLE_9_1

The selected residual increased median PMI from 0.03019 in provisional A to 0.11492 in B. Its gap of 0.08473 exceeds the raw signal’s negative gap and the strict sparse LPV–ARMA gap of 0.07255, while the corrected archived SRC gap is larger at 0.10335. Figure 9.2 plots the observed distributions. For the lecturer-compatible score, the selected-minus-sparse increase in median gap was 0.01218. The paired within-group bootstrap interval was [0.00853, 0.01595] using matched candidate IDs. Relative to the archive, the corresponding contrast was −0.01862, with interval [−0.02123, −0.01606]. The same-weight adapter-off contrast was 0.00070 with an interval crossing zero. These intervals quantify resampling sensitivity over the supplied index cohorts. They cannot verify that each file represents an independent original acquisition.

FIGURE_9_2

All five residual models attained conditional AUC 1.000 on this partition, so AUC cannot rank these residual models there. The larger archive and fixed-physics gaps, and positive margins for the residual models, provide more discriminating descriptions of the supplied scores. On a different instrument channel, fault severity or operating cycle, both group central tendencies and thresholds may shift. Fitting a decision threshold on those provisional labels would conflate score separation with true-positive and false-positive rates on independent machines. The diagnostic package retains a supplemental 41-record comparative screen across the full family roster. Table 9.1 uses the live 198-record common cohort to avoid mixing counts or claiming all 21 baselines were run on every candidate acquisition.

## 9.3 Paired pulse response and calibrated stress

The pulse study perturbs a single measured development signal with known damped sine events. For a periodic scenario, event times satisfy a fixed phase in successive input-shaft revolutions, derived from measured speed. Aperiodic controls use seeded random phase within turns. The five periodic cases vary carrier frequency (390 through 1,260 Hz), amplitude (0.12 through 0.88 standardized units), decay (1.7 through 6.1 ms), phase and modulation. One periodic case applies a warped RPM profile to both its clean and injected predictions. Two aperiodic cases are waveform-matched controls. Separate, differently specified disturbance fixtures were used in guarded training and selection. The development injection patterns here probe a frozen model’s response on the measured development trace.

Let $d_t$ be the injected waveform, let $\Delta\widehat x_t=\widehat x_t(x+d)-\widehat x_t(x)$ be the difference in the forecast under the same speed path, and let $z_t$ be the change in residual. Then

$$
z_t=d_t-\Delta\widehat x_t,\qquad
D_{\mathrm{pulse}}=\frac{\lVert z-d\rVert_2^2}{\lVert d\rVert_2^2},
\qquad Q_{\mathrm{pulse}}=\frac{\lVert z\rVert_2^2}{\lVert d\rVert_2^2}.
$$

$D_{\mathrm{pulse}}$ measures relative distortion in the recovered residual waveform. $Q_{\mathrm{pulse}}$ measures its relative energy. For an unchanged recovered pulse, distortion is zero and energy ratio is one. The waveform can change even when its recovered energy appears close to one. The saved results also report correlation, signed projection and paired change in PMI. These complementary quantities avoid interpreting one component as a complete measure of preservation.

TABLE_9_2

PI-SRC yielded positive PMI changes in five of five periodic and zero of two aperiodic cases. Its median periodic increase was 0.000422, median correlation was 0.690, median distortion was 0.946 and median energy ratio was 1.794. The energy above unity is compatible with amplification and reshaping. The archive’s median PMI change was larger at 0.000713, while its pulse correlation was weaker at about 0.616. The fixed-physics variant recovered somewhat greater waveform correlation at 0.756. Figure 9.3 relates the paired diagnostics to an example of known injected structure. The test demonstrates response to the specified pulses on this trace. Field sensitivity requires independent physical tooth-crack data.

FIGURE_9_3

A particularly important negative control changes measured speed while holding physical vibration fixed. Applying +2% RPM bias only to the predictor raised PMI by 0.098194. Changing only diagnostic angle raised it by 0.004508, with the same forecast. Biasing both gave 0.044913. The predictor-only PMI increment is over two orders of magnitude greater than the median periodic pulse increment in this experiment. Phase-locked features can improve prediction while their residuals remain vulnerable to measurement error. Gear-fault condition monitoring needs sensor quality controls, known health labels and independent validation of decision thresholds. [@fyfe1997order; @cheng2014deformation; @abboud2017envelope]

<!-- CONCLUSION START -->

# 11. Conclusions and next experiments

Matched forecasting confirmed the predictive advantage of the selected PI-SRC over the 21 requested comparators on the official development segment. Training and development MSE were 0.308688 and 0.329973, respectively. Development MSE was 11.16% below the strict sparse LPV–ARMA and 10.18% below Linear Regression. The executable archived corrected SRC retained a lower MSE of 0.317788, a 3.83% disadvantage for the selected PI-SRC relative to that anchor. The clean-only finalist and the 12-order retrospective ablation also achieved lower development errors than the registered multi-objective choice. Those outcomes preserve the distinction between H1’s comparative gain over the required families and the larger problem of beating the strongest earlier model.

For H2, angle history was the strongest tested structural contributor. Removing it raised development MSE to 0.359049 and removing both angular lags and carriers raised it to 0.377201. The causal correction, evaluated with fitted weights fixed, lowered MSE from 0.340706 to 0.329973. The same correction also benefited a conventional fit and a physics-free control. Evidence supports an advantage for the tested combination of physically aligned history, lagged innovation and a structured readout. The removal effects and different-sized feature sets do not identify additive causal contributions for all feature groups. The selected frequencies and decay constants are engineering settings selected in a bounded search.

For H3, the residual median PMI gap was 0.08473 under the provisional A/B labels, versus 0.07255 for the strict sparse comparator and 0.10335 for the archive. Five periodic synthetic interventions increased PMI, and changes in waveform correlation and energy show that preservation was imperfect. The +2% predictor-only RPM bias raised PMI by 0.09819. Confirmed crack sensitivity, false-alarm rates and decision thresholds require independent fault labels and measured sensor tolerance.

For H4, the selected model required 484.89 ms median CPU time for the 7.5 s record, with feature construction in the timed call. The strict sparse fit had lower latency and the executable archive had both lower latency and lower clean MSE. The strongest requirements for a next experiment are a predeclared frozen checkpoint, independent acquisitions across speed and load regimes, verified condition labels, speed-calibration controls and measured streaming deadlines. The existing replay, guards and mask provide the repeatable starting point for collecting that evidence.
