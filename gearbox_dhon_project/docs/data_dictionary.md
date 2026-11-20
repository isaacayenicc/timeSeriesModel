# Course data dictionary and split manifest

## Acquisitions

| Logical file | MATLAB layout | Record duration | Use |
|---|---|---:|---|
| `sigtrain.mat` | One `sigtrain` struct, 24,000 samples | 7.5 s | Fit, feature selection and guarded folds |
| `sigvalid.mat` | One `sigvalid` struct, 24,000 samples | 7.5 s | Official development comparison with a fixed [1792, 24000) mask |
| `sigfault1_part_001.mat`–`020.mat` | Each file contains a 1×10 `sigfault1` struct array, 32,000 samples per struct | 10 s | Development diagnostics on 200 acquisitions |
| `sigtest.mat` | Expected single blind prediction record | Unknown | Missing from supplied files |
| `sigfault2.mat` | Expected blind diagnostic collection | Unknown | Missing from supplied files |

File order follows the numeric part suffix. Record IDs preserve MATLAB linear, column-major order within each part. Python sample ranges are zero-based with exclusive stops. A record number in the original course collection is one-based. Every observed time grid is consistent with 3,200 Hz. The time tag at index 0 is 1/3200 seconds. The time tag at index 23,999 is 7.5 seconds. Duration in samples is distinct from elapsed time between the first and last observed samples.

| MAT field | Meaning | Verified convention |
|---|---|---|
| `x` | Supplied standardized vibration | Dimensionless archived sample. Primary MSE uses this scale. |
| `xraw` | Supplied pre-standardization vibration | Present for training and validation. Appears to be the already conditioned sample. Its acquisition unit is undocumented by the MAT fields. |
| `t` | Sample time in seconds | Strictly increasing, grid interval 0.0003125 s. |
| `fs`, `N` | Rate in Hz and record sample count | Scalar metadata checked against array dimensions and time differences. |
| `spd` | 37-tooth input gear speed | RPM. Divide by 60 for input shaft Hz. |
| `spdnorm` | Stored min-max scaled speed | Exactly agrees with `(spd-spdmin)/(spdmax-spdmin)` using the bounds stored in training, across all 202 loaded records. |
| `spdmin`, `spdmax` | Scaling bounds | 124.0745554 and 167.3376581 rpm, distinct from the extrema of the supplied 7.5-second segment. |
| `xmean`, `xstd` | Stored vibration affine transform | First channel entries 0.0026221942 and 0.4251729317 in both fit and validation. The reconstruction `xmean[0] + xstd[0]*x` matches `xraw` within 2.22×10⁻¹⁶. |

The course signals follow the paper's reduced 3.2 kHz sample rate. The original 25.6 kHz samples and FIR taps are unavailable. The x-axis bearing-house accelerometer is described in Chen et al. (2025) §§5.1–5.2 and Figs. 11–13. The presence of a calibrated accelerometer in the rig does not establish physical units on the uploaded `xraw` field. Multiply dimensionless MSE by `0.4251729317²` only to convert to squared supplied pre-standardization units. Archive arithmetic preserves the supplied affine transform instead of renormalizing each record. No absolute tooth keyphase is present, so integrated phase has a record-local origin.

## Eligibility and overlap

The audit scans all 8,001 contiguous 24,000-point offsets in each 32,000-point record against each official prediction record, using normalized vibration correlation and direct vibration/speed equality at any exact first-sample anchor. It records differences for the highest-correlation candidate, uses an explicit joint absolute tolerance of 10⁻⁸ for near-equality, and also fingerprints complete fault waveforms and speeds. Record 61 includes all official training samples at offset zero, and record 63 includes all official validation samples at offset zero. No other full-length matches were found. Both entire ten-second acquisitions are excluded from comparative diagnostics. An independent all-pairs search matches 64 consecutive vibration and RPM samples at 512-sample reference anchors. It confirms the same two exact coincidences and no others. Any exact shared run of at least 575 samples is detectable. Detection of arbitrary approximate partial overlap remains dependent on acquisition metadata.

The one-based index partition 1–100 / 101–200 is retained as provisional A/B for comparison. The paper's description of separate healthy and crack collections does not provide a mapping for the uploaded struct order. No per-record health or operating-condition labels, absolute tachometer/revolution index, device calibration certificate, or measured torque channel were supplied. The resulting exploratory comparisons describe index groups.

`outputs/runs/development/tables/record_inventory.csv` contains per-record fields, extrema, duration, affine reconstruction checks and array hashes. `split_manifest.csv` contains global IDs, source-based grouping, offsets and exclusions. `duplicate_scan.csv` contains the best offset and coincidence evidence. `exact_shifted_overlap.csv` contains all-pairs anchor matches. `source_inventory.csv` and `archive_members.csv` contain file and archive-member digests. `data_availability.json` records the blind and label gate. Original MAT inputs are placed locally in `data/raw/` and excluded from redistribution.
