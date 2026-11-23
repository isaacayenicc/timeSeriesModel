# Dataset provenance

The provided handout, Chen et al. (2025), Chen (2020), saved MATLAB files and release evidence delimit the experiment. The journal describes a step-up helical stage with a single gear pair driven through a motor and alternator rig, 37 teeth on the input gear and 20 on the output pinion. The measured accelerometer direction is the input bearing-housing x channel. The encoder uses zebra tape. The reported original rate is 25.6 kHz with a 1.6 kHz FIR low-pass before downsampling to 3.2 kHz. Exact FIR coefficients, raw-stream delay and independently verified field calibration are unavailable for the supplied files.

| Input identity | Shape at saved rate | Use |
| --- | --- | --- |
| sigtrain.mat, SHA-256 47d23efd20d86ed9e67181a96c9b73039a79a874e0d9ae91be5bcb8a421c19a6 | 24,000 synchronized x/RPM pairs, 3,200 Hz, 7.5 s | EDA, fitting and guarded selection |
| sigvalid.mat, SHA-256 07013d4b17d7ae6fc740365264def53882664a49fde278b0d839979652b99b73 | 24,000 synchronized x/RPM pairs, 3,200 Hz, 7.5 s | Held-out scoring and designed diagnostic interventions after freeze |
| sigfault1_part_001.mat through sigfault1_part_020.mat | Twenty files of ten ordered structs. Each has 32,000 x/RPM pairs, 3,200 Hz, 10 s. | Post-fit diagnostic candidate cohort |
| sigtest.mat and sigfault2.mat | Unavailable in the supplied release | No blind test score |
| fdt_featcal.m, SHA-256 6c1206b0cb7e1098b6b9b52cf437e2ba5097e05334e9c00cee51e5fc8fabbf06 | Lecturer feature-calculation reference script | Index and envelope semantics. Exact numerical parity depends on the omitted angular helper and reference outputs. |

All 22 loaded MAT files agree with the manifest file hashes. All 202 rows match individual vibration and RPM array hashes. The official files use archived dimensionless x. Their stored xraw field reconstructs as offset 0.0026221941677904226 plus 0.42517293168840065 times x, with error at machine precision. Calibration of xraw is undocumented. Speed spd is RPM with a separate archived normalized field. Predictor feature scaling draws only on permitted fit rows. Train speed spans 124.322840 to 167.333518 RPM, development 124.335179 to 167.322539 RPM and the full candidate corpus 123.655276 to 167.416491 RPM.

The audit searches each entire 24,000-sample official vibration-plus-RPM pair within each ten-second candidate. Candidate 61 reproduces training at offset zero and 63 reproduces development at offset zero. These are the only exact full matches. Record IDs follow the ascending numbered parts and are one-based. Approximate, very short or unidentifiable shared acquisition fragments remain possible. Removing two records leaves 198 candidates, comprising 98 index-A subjects and 100 index-B subjects. The supplied artifacts do not establish A and B as true health labels.

One-step comparison uses zero-based, half-open slice [1792, 24000), 22,208 entries and a 0.56 s warm-up. Target-vector SHA-256 is 1522c0a1b184b971b59763df3d827515370162751cc61209f025765039a89fb1 over little-endian 64-bit integer indices. Candidate forecasts start at the same 1,792 index and retain 30,208 targets per candidate. The prediction archives agree exactly on target index, vibration, RPM and timestamp. The reproducibility script independently reverified the official target arrays and the two overlaps.

The paper used two 7.5 s segments of one healthy 20 s record. Similar sampling and the source description establish equipment context. The supplied validation record has no sample-level mapping to the paper's test segment. A difference in the plotted speed trajectory and scoring treatment limits direct benchmarking.
