# 3. Experimental system and data integrity

## 3.1 Measurement chain and source roles

The published experiment uses an electric drive motor and an alternator that applies a load correlated with rotational speed. Its test gearbox is a single-mesh helical stage. The input shaft is attached to a 37-tooth gear and the output shaft to a 20-tooth pinion. The authors report the $x$-direction vibration from the accelerometer on the input-shaft bearing housing and rotational information from an optical 88-pulse-per-revolution encoder. Figure 3.1 depicts the major equipment and measurements. Under this kinematic arrangement an input-shaft order of 37 and a pinion order of 20 describe the same mesh-event frequency on their respective references. The reported original vibration sampling rate was 25.6 kHz. The authors low-pass filtered at 1.6 kHz and downsampled to 3.2 kHz for their experiments. The supplied project files begin at 3.2 kHz. Their preprocessing metadata support use of the processed stream. Original filter taps and delay are unavailable, so raw-sensor latency remains unresolved. [@chen2025sparse; @chen2020vibration]

FIGURE_3_1

The paper describes a 20 s healthy acquisition selected for modelling. Two 7.5 s segments were extracted and assigned to training and testing. The corresponding paper test speed trajectory starts in a different part of the cycle from the supplied development sequence. The original 20 s sample origins and exact mapping to the supplied file are unavailable. The reported paper score is therefore a contextual research benchmark on the authors’ segment. Section 7 gives values and distinguishes local historical reproductions, published performance and the strict lag-1 comparison. The published healthy/damaged acquisition counts motivate investigation of the supplied diagnostic corpus. The release lacks the mapping needed to assert the health labels of those files. [@chen2025sparse]

Table 3.1. Supplied signals, scores and permitted uses.

| File or corpus | Verified size at 3,200 Hz | Role and equality controls |
| --- | --- | --- |
| sigtrain.mat | 24,000 samples, 7.5 s | EDA, guarded fit and selection, final refit, in-sample benchmark |
| sigvalid.mat | 24,000 samples, 7.5 s | Development replay only, 22,208 targets after the common guard |
| sigfault1 parts 001–020 | 200 matched vibration/RPM records, 32,000 samples or 10 s each | One-based indices. Exclude 61 and 63. Conditional groups A/B with 98 and 100 eligible records |
| sigtest.mat and sigfault2.mat | Unavailable at report generation | Frozen interfaces exist for separate evaluation when supplied |
| Acquisition condition and absolute tooth keyphase | Unavailable | Diagnostic labels and absolute damaged-tooth identity require additional independent records |

The intake loaded all 202 supplied arrays and verified finite vibration, positive rotational speed, lengths and the regular timestamp grid starting at one sample interval. It treated the supplied vibration field as dimensionless standardized vibration. In the short files, reconstructing the pre-standardization channel from the stored offset 0.00262219 and scale 0.425173 agrees with the supplied array within $2.22\times10^{-16}$. That agreement verifies an affine relationship. Traceable physical calibration of the supplied acceleration channel is unavailable. MSE throughout this study is therefore in squared standardized vibration units. The stored speed extrema reconstruct the supplied normalized channel, but the selected pipeline reads RPM and fits its own feature transforms on the authorized fit rows. No held-out record supplies new vibration standardization or scaling of the fitted feature library.

## 3.2 Overlap, fit boundaries and frequency coverage

All 8,001 possible contiguous positions of each 24,000-sample short record within each 32,000-sample diagnostic candidate were assessed for an exact joint vibration and speed match. Two pairs matched exactly from offset zero. One-based record 61 contains the designated training file and record 63 contains the designated development file. A second exact-marker scan using 64-point identities on a 512-point anchor grid confirmed the same pairings. The finite scan and tolerance checks address the audited overlap patterns, while original acquisition relationships and all possible approximate partial overlap remain unresolved. The two entire candidate records are excluded from group statistics. The experiment resets its predictors for each other candidate. Figure 3.2 specifies the role and direction of each split.

FIGURE_3_2

The primary measured prediction mask is the zero-based interval $[1792,24000)$, with 22,208 one-step targets. A reset and common warm-up give all approaches the declared comparison interval. The selected base representation has finite support of 512 samples. Its adapter increases the full support to 1,280, below the common 1,792-sample guard. The three held training blocks and their disjoint fit regions are specified in Section 6. Observed history in a scoring block is available at inference after the relevant sample has arrived. Its target values never update the fitted coefficients. This distinction is important for innovation histories and the optional adapter.

Figure 3.3 compares the two designated speed profiles and vibration channels under identical plotting scales. The observed speed intervals are 124.323–167.334 RPM in training and 124.335–167.323 RPM in development. Across the fault-candidate corpus the extrema are 123.655–167.416 RPM. These ranges are narrow and overlapping. An estimator can be valid at the target information horizon and still extrapolate poorly to a different operating regime. The late-to-early gaps in the chronological sensitivity control show why coverage and timed causal validity are distinct.

FIGURE_3_3

# 4. Physics-informed exploratory analysis

## 4.1 From rotational speed to features

Rotational speed determines the observable locations of several candidate components. With $n_t$ in RPM and $Z=37$, the shaft and gear-mesh frequencies in hertz satisfy

$$
f_{\mathrm{shaft}}(t)=\frac{n_t}{60},
\qquad f_{\mathrm{mesh}}(t)=Zf_{\mathrm{shaft}}(t),
\qquad f_q(t)=q f_{\mathrm{shaft}}(t).
$$

Here $q$ is order, measured in cycles per input revolution. The order of mesh harmonic number $k$ is $kZ$ cycles per input revolution. On the training profile the fundamental mesh frequency spans 76.666–103.189 Hz. The period in the original sampled stream spans about 31.0–41.7 samples. A single discrete-time delay therefore aligns to different fractions of a mesh period across the record. At 3.2 kHz the frequency Nyquist limit is 1.6 kHz, equivalent to about shaft order 573.7 at the highest training speed. This permits the first 15 mesh harmonics at that speed on the ideal grid. The study keeps a finite frequency margin under speed perturbation during candidate checking. Angular interpolation aligns speed-synchronous energy, and it cannot recover frequencies already removed by the original acquisition or preprocessing. [@fyfe1997order; @chen2025sparse]

A training-only STFT with a 1,024-sample Hann window, 896-sample overlap and 4,096-point Fourier grid tested the model’s physical premise. The measured local ridge near the fourth mesh harmonic tracked the speed-derived trajectory with correlation 0.987 and median absolute frequency discrepancy 0.744 Hz. The search was restricted to a neighbourhood of that predicted ridge, so these numbers characterize local agreement, rather than proof that the source is a single physical mode. Figure 4.1 shows the moving trajectories and frequency band. Welch estimates with 1,024-sample segments and angle-domain interpolation contrast frequency and shaft-order evidence in Figure 4.2. The order-148 peak was approximately 160 times its local order-spectrum median. The 1.0–1.2 kHz band contained 31.3% of the training Welch power. A broad band shared by moving mesh lines and potential structural effects cannot identify a fixed resonance. A dedicated fixed-frequency test is required to assess predictive benefit. [@welch1967fft; @fyfe1997order]

FIGURE_4_1

FIGURE_4_2

## 4.2 Time memory, operating regime and targeted screens

Short-time RMS and measured RPM correlated 0.927 across the descriptive training blocks. Load was correlated with speed in the rig, so this statistic associates the operating variables with vibration amplitude. It does not isolate a causal speed-to-amplitude mechanism. The lag-one time correlation was 0.0759, while a one-mesh-period angular correlation was 0.1278. A separate one-shaft-turn angular correlation was 0.0745. These are quantities on different coordinates and supports and direct comparisons are limited. They motivate keeping local time history as well as angular delays. The fourth time lag had the largest absolute partial autocorrelation among lags 2–96 in the stored screen. That observation supports modelling short histories and does not identify a unique order or justify setting memory to a fixed shaft period.

An LPV–AR24 screen within the designated training data separated these hypotheses before the joint PI-SRC sweep. Its mean guarded MSE was 0.46153. Adding order-148 and order-259 quadratures yielded 0.43803 in the initial feature screen. In the extension, the same two orders with decay constants 8, 32 and 128 samples yielded means 0.45228, 0.44560 and 0.44049. A six-order candidate at 32 samples yielded 0.43537. Adding selected shaft/sideband orders to the two-order bank gave 0.42903 in the corresponding highly constrained comparison. These values justify examining carrier family, damping and sideband combinations in the full search. Selection of the final four-order bank remains determined by the declared joint objective and fold scoring.

Two fixed-Hz states near 1.08 and 1.125 kHz produced 0.46152, effectively the initial linear control’s score, despite their plausible frequencies. The single 1.1 kHz feature scored 0.46230. Selective degree-two and degree-three extensions gave 0.43819 and 0.43803 in a narrower nonlinear comparison. This weak marginal difference did not justify a full cubic expansion. It instead supports testing a constrained, reproducible selection and reporting capacity as a separate factor. The screen’s artificial RPM lead is a descriptive timing check only. Forecasting requires previously arrived measurements. The synthetic constant-speed and chirp fixtures recovered order 148 to within 0.024 order, and their suffix perturbations changed the guarded quadrature outputs by zero in those fixtures.

Table 4.1. From exploratory physical evidence to a measured model decision. Reported screening values use training folds only.

| Physical observation | Controlled candidate implication | Verification in the final study |
| --- | --- | --- |
| Ridge near moving order 148 and a strong angular peak | Synchronous carriers at physically plausible orders and multiple decay scales | Joint bank and support search, finite-sum check, carrier and order-count ablations |
| Wide training power band near 1.1 kHz | Try a small fixed-Hz bank alongside angular states | Fixed-Hz screening matched the control, selected model has no fixed-Hz state |
| Mesh period changes by roughly 10.7 time samples over training speed | Mix time delays and angle-aligned history | Remove angle history while keeping model family and score mask |
| Local RMS follows the operating cycle | Include energy and measured speed context with bounded interactions | Screened energy/scheduling groups and model-record stratification |
| Constrained cubic increment is small in the preliminary screen | Restrict and regularize nonlinear interactions | Compare capacity and degree variants, and retain group-level evidence |
| Sensor reliability is distinct from the utility of a speed channel | Preserve bounded history and separate sensor perturbation pathways | Prediction-only, angle-only and combined stress tests |

The measurements distinguish a physical hypothesis from a fitting choice. Knowledge of the gear geometry supplies candidate orders and defines the interpretation of a quadrature. The guarded search resolves whether a combination is useful on this record. Later ablations provide additional attribution under a common target mask.
