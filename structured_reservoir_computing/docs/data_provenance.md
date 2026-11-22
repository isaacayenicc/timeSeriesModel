# Course data, source provenance, and information cutoff

## Source precedence and assessment

The supplied `MECG 7780 Course Project.pdf`, pp. 1–2, gives the assessment and deliverables. It specifies present-vibration prediction from past vibration, MSE, CPU/GPU time and fault detection. Speed input is encouraged. It points to angular PMI and provides `fdt_featcal.m`. Progress presentation is 8 minutes and final presentation is 12 minutes. The progress date is November 4, 2026, while the report and final presentation are dated December 7 and 9, 2025. The quoted blind release date is December 2, 2025. Schedule status is **INSTRUCTOR CONFIRMATION REQUIRED**.

| Criterion | Progress | Final |
|---|---:|---:|
| Organization/presentation | 4% | 16% |
| Presentation clarity | 4% | 16% |
| Algorithms understood | 8% | — |
| Algorithm performance | — | 32% |
| Algorithm novelty | 4% | 16% |
| Total | 20% | 80% |

Actual uploaded MAT arrays and fitted code govern the computed experiment. Chen et al. (2025), §§3–5 and Tables 5–6, govern paper equations and publication values. Chen (2020), §2.2.3 and §3.3.3, supplies corroborating PMI and rig context. Chen et al. (2025) is Mechanical Systems and Signal Processing 224, article 112161. The DOI is 10.1016/j.ymssp.2024.112161. The original thesis is *Vibration-based fault detection and severity assessment for fixed-axis gearboxes*, University of Alberta.

## File inventory and measured dimensions

| Input | Variables and shape | Rate/duration | Role |
|---|---|---|---|
| `sigtrain.mat` | Scalar `sigtrain`, 24,000 samples | 3,200 Hz / 7.5 s | Fit and training-fold selection |
| `sigvalid.mat` | Scalar `sigvalid`, 24,000 samples | 3,200 Hz / 7.5 s | Official held-out development scoring |
| `sigfault1_part_001.mat`–`020.mat` | Twenty 1×10 `sigfault1` struct arrays, each struct with 32,000 samples | 3,200 Hz / 10 s per record | Provisional index-group diagnostics |
| `sigtest.mat`, `sigfault2.mat` | Missing from supplied inputs | — | Await independent frozen inference |
| Acquisition condition labels and chronology | Missing from supplied inputs | — | A/B remains provisional |

The preparation audit loaded all 202 records and checked finite vibration, positive RPM and uniform time intervals. In each record `N` matches vector lengths and `fs` matches the grid. The time tags start at 1/3,200 s. Time is in seconds, speed `spd` is in revolutions per minute, and primary vibration `x` uses the supplied dimensionless scaling. The two short files also contain `xraw`, `xmean`, `xstd`. Reconstructing the stored pre-standardization channel from `xmean[0] + xstd[0] * x` agrees within 2.22×10⁻¹⁶. The first-channel offset is 0.0026221942 and scale is 0.4251729317. The calibration unit of the supplied `xraw` is undocumented. Using the stored `spdmin=124.0745554` and `spdmax=167.3376581` reproduces `spdnorm` exactly for all 202 records. The archive bounds differ from the observed extrema of the training segment.

The journal uses an optical zebra-strip encoder on the input shaft, axial bearing-house vibration, and a single mesh of 37 input teeth and 20 output teeth. The source study sampled at 25.6 kHz and low-pass filtered at 1.6 kHz before downsampling to 3.2 kHz. The MAT files match the latter rate. The original filter taps and phase response are unspecified in the supplied assets. The analysis consumes the processed vibration stream as supplied. Any calibrated raw-stream latency guarantee would require original acquisition information.

Train speed is 124.322840–167.333518 RPM, development 124.335179–167.322539 RPM, and fault-candidate speed across the corpus 123.655276–167.416491 RPM. The paper speed-axis values 124–167 are labelled Hz, yet the thesis explicitly uses Hz = RPM/60 and the measured mesh ridges support input RPM. No raw torque or absolute tooth keyphase is supplied. An alternator imposed load that covaried with speed.

## Duplicate evidence and masks

Ten consecutive IDs are mapped to each input part in ascending numeric suffix order. All 8,001 possible complete 24,000-sample starting offsets were assessed for each fault-record/official-record pair. This yields 3,200,400 candidate offsets in 400 comparisons. Direct equality of both vibration and RPM holds for record 61 against training at offset zero and record 63 against development at offset zero. A second scan of exact contiguous 64-sample markers from a 512-sample anchor grid across all records confirms only these same pairings. The grid guarantees an anchor check within any contiguous exact match spanning at least 575 samples. A finite tolerance check assessed the most vibration-correlated full alignment for each official pair. Arbitrary approximate or very short partial overlaps and original physical acquisition identity remain unresolved.

The whole diagnostic acquisitions numbered 61 and 63 are excluded from score summaries and retained in audit tables. Group A is the first 100 numbered records and Group B is the last 100. At full corpus scope 98 A and 100 B remain after exclusions. Their health conditions remain unknown. Confirmation requires the original condition-to-file map, not a favorable ROC result.

The official score mask is zero-based [1792, 24000), with 22,208 targets and one-sample forecast horizon. The primary predictor observes data through target-minus-one. Fit folds use [4800, 7200), [11200, 13600) and [18400, 20800), with a 1,792-sample guard on either side. Fit windows and state are initialized within each permitted segment. For [a,b), allowed raw training intervals are [0,a−1792) and [b+1792,24000). Evaluation may use prior observed samples within its own block. These folds share one autocorrelated acquisition. The protocol stores an index hash and field-level split manifest in `data/splits/`.

## Archive and publication discrepancies

| Evidence or discrepancy | Outcome |
|---|---|
| Prior EDA ZIP | The 36-cell notebook has stored outputs for 18 code cells. Its README, analysis code, result tables and JSON were inspected. The two MAT copies are byte-identical to supplied course inputs. Source cells implement STFT ridge search, angular PSD, guarded ridge, proxy residuals and Morlet controls. This project independently reran order, speed and selection screens. |
| Sparse LPV-ARMA reproduction ZIP | The 49-cell notebook has stored outputs for nine cells. Source and record metadata are present and reference input MAT files are external. The supplied core exposes ADMM and least-squares routines. The experimental checkpoint contains separate AR, MA, proxy and speed-basis coefficient arrays, distinct from the simulated system checkpoint. Historical coefficient replay uses original model and mask conventions and stays distinct from strict historical-speed fits. |
| Prior optimization ZIP | The 16-cell notebook has eight cells containing stored results. Both designated MAT copies are identical. Original numeric checkpoints and ablations are available as historical controls. The two legacy reservoir checkpoint files use three numeric weight arrays with lengths 768, 962 and 1,056. Their source separates phase interpolation, structured expansion, prediction and slow-error correction. Retain the original settings and prior search exposure in any comparison. Legacy reservoir checkpoint paths are recorded literally in the archive manifest. New figures use SRC naming. |
| Historical segment | Fig. 13(c) of the paper starts near 140 on the plotted speed axis and descends before rising. Course development starts at 124.551 RPM, closely mirroring training. The original 20-second file and complete retrospective window archive are absent. Publication test MSE belongs to a distinct segment. A circular wrap would create an artificial boundary. |
| Diagnoses from the supplied MATLAB code | `fdt_featcal.m` is present and byte-identical to archived copies. It calls external `toConstantSamplesPerCycle`, unavailable among supplied sources, and computes envelope autocorrelation PMI. The literal MATLAB array index and the physical one-revolution lag can differ by one sample. The diagnostic verifier distinguishes exact arithmetic from approximate angular-resampling compatibility. |
| Optional sources | The standalone precursor and retrospective window ZIPs were absent. Actual code and metrics found within the three supplied archives are described at their literal scope. |

The report's measured evidence links to `inputs/input_manifest.csv`, `inputs/archive_members.csv`, `results/data_inventory.json`, `results/record_overlap_audit.csv`, `results/exact_shifted_overlap.csv`, `results/normalization_audit.csv`, and `figures/eda/eda_overlaps.png`. Archive scanning bounded individual and total decompressed entry sizes, rejected unsafe paths/symlinks and hashed the validated member streams. No archive source code executed during intake.
