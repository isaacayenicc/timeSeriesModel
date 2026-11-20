# Reproduction summary

Run `development`. All matched scores use targets 1,792–23,999, 22,208 samples at 3.2 kHz. Fit selection used only guarded blocks of `sigtrain`.

| Model | Targets | MSE | MAE | Bias |
| --- | --- | --- | --- | --- |
| primary dhon | 22208 | 0.318626 | 0.449037 | -0.003755 |
| alternative dhon | 22208 | 0.338247 | 0.462787 | 0.005736 |
| dhon matched fixed | 22208 | 0.318773 | 0.449126 | -0.003745 |
| dhon matched linear | 22208 | 0.319213 | 0.449507 | -0.003564 |
| spa lpv arma | 22208 | 0.371406 | 0.483517 | -0.008890 |
| lpv arma dense | 22208 | 0.330781 | 0.458267 | -0.008440 |
| archived spa lpv arma | 22208 | 0.345151 | 0.466209 | -0.002634 |
| physics linear | 22208 | 0.376432 | 0.487596 | 0.004488 |
| ngrc | 22208 | 0.376930 | 0.487993 | 0.003916 |
| tcn rpm | 22208 | 0.475057 | 0.547844 | 0.024603 |
| archive dhon optimized corrected | 22208 | 0.316889 | 0.447911 | -0.002539 |
| archive ngrc optimized corrected | 22208 | 0.317788 | 0.448410 | -0.003392 |
| archive dhon tau context extended gated corrected | 22208 | 0.316445 | 0.448727 | -0.004849 |

**Interpretation.** Primary DHON improves on the strict sparse refit by 14.21% and trails the strongest forecast-only archive anchor by 0.69%. The learned-state improvement over the matched fixed oscillator is 0.000148. Removing correction worsens MSE by 0.014029.

**Selection.** `expanded_short_tau` is primary and `focused_joint` is alternative. Exact configs are in [the selection manifest](../outputs/runs/development/search/dhon_selection.json). There are 21 ranked DHON configs and 63 guarded fold rows. The current-speed archived comparison and the published Table 6 value 0.2893 refer to distinct conventions.

**Limits.** `sigtest` and `sigfault2` were absent. Condition labels and acquisition start IDs were absent. GPU hardware was unavailable. Instructor-compatible PMI indexing was reproduced algebraically, and exact executable MATLAB parity remains unverified.

**Replay.** From the project root run `python -m gearbox_dhon.cli replay --run-id development --data-dir data/raw` after placing the original MAT files. See [README](../README.md) for full run, smoke, notebooks, and frozen blind evaluation.
