## Measured full fault-candidate study

The completed study contains **200 separate recordings**, **13 frozen validation-selected causal predictors**, **2,800 raw/residual time-feature rows**, and **8,400 angular-sensitivity rows**. Every scalar comparison uses indices 256–31,999, exactly 31,744 values per recording. Separate processing does not itself prove statistical independence of acquisitions; acquisition-order semantics are unknown.

| model_id             |   records |   median_rms |   median_residual_raw_rms_ratio |   median_crest_factor |   median_kurtosis |
|:---------------------|----------:|-------------:|--------------------------------:|----------------------:|------------------:|
| lpv_L256_s17_a1      |       200 |     0.623974 |                        0.588109 |              4.908411 |          3.272660 |
| ar_L256_s17_a1       |       200 |     0.665676 |                        0.630237 |              4.818784 |          3.295679 |
| ols_L256_s17         |       200 |     0.665687 |                        0.630243 |              4.818087 |          3.295630 |
| arx_L256_s17_a1      |       200 |     0.666277 |                        0.631446 |              4.812602 |          3.298296 |
| speed_tcn_L128_s17   |       200 |     0.680298 |                        0.645790 |              4.826195 |          3.289983 |
| tcn_L128_s17         |       200 |     0.681158 |                        0.647383 |              4.844180 |          3.322133 |
| mlp_L64_s17          |       200 |     0.681743 |                        0.648656 |              4.795177 |          3.277209 |
| late_tcn_L64_s17     |       200 |     0.688605 |                        0.653663 |              4.868368 |          3.304069 |
| capacity_tcn_L64_s17 |       200 |     0.695132 |                        0.657809 |              4.858827 |          3.292343 |
| gru_L64_s17          |       200 |     0.913906 |                        0.871571 |              5.011131 |          3.412047 |
| zero_L256_s17        |       200 |     1.053410 |                        1.000000 |              4.814118 |          3.391280 |
| raw                  |       200 |     1.053410 |                        1.000000 |              4.814118 |          3.391280 |
| mean_L256_s17        |       200 |     1.053415 |                        0.999985 |              4.814454 |          3.391280 |
| persistence_L256_s17 |       200 |     1.493909 |                        1.426888 |              5.118118 |          3.562371 |

These are medians of record-level descriptors, not classification scores. The paired ratio divides each model residual RMS by that same record's raw RMS before taking the median. Zero prediction has the same descriptors as raw, an implementation sanity check. Candidate-record prediction errors were not used to reselect the primary model; the recommendation remains based on the original validation protocol.

### Post hoc source-order observation, without physical labels

The all-record plot shows an amplitude change near source ID101. The following partition was examined **after viewing the trend** and has no known class meaning; it is not a fault-label assignment, calibrated threshold or independent detection experiment.

| model_id        | post_hoc_source_partition   |      rms |   kurtosis |   mean_speed_source_units |   speed_std |
|:----------------|:----------------------------|---------:|-----------:|--------------------------:|------------:|
| lpv_L256_s17_a1 | source IDs 101–200          | 0.676885 |   3.340254 |                143.378533 |   13.999946 |
| lpv_L256_s17_a1 | source IDs 1–100            | 0.581834 |   3.202482 |                143.376490 |   14.000718 |
| raw             | source IDs 101–200          | 1.141549 |   3.467404 |                143.378533 |   13.999946 |
| raw             | source IDs 1–100            | 0.993399 |   3.310569 |                143.376490 |   14.000718 |

Raw and residual RMS differ between these provenance groups while the coarse speed moments are similar. Similar moments do not establish identical operating histories or exclude load, acquisition, preprocessing or health changes. Source documentation is required to interpret the shift. Neither accuracy nor an onset claim can be inferred from this descriptive partition; it was not used to train or select models.

### Physical-unit and PMI-indexing sensitivity on all records

| model_id        | speed_unit_assumption   |   ck_order1 |   pmi_faithful_reference |   pmi_scientifically_corrected |   roundtrip_rmse_fraction_of_source_rms |
|:----------------|:------------------------|------------:|-------------------------:|-------------------------------:|----------------------------------------:|
| lpv_L256_s17_a1 | rad_s                   | 1.82325e-05 |                0.0453168 |                      0.0488183 |                                0.251704 |
| lpv_L256_s17_a1 | rpm                     | 2.30612e-05 |                0.0894128 |                      0.0870028 |                                0.300441 |
| lpv_L256_s17_a1 | rps                     | 2.33043e-05 |                0.0523959 |                      0.0520566 |                                0.320434 |
| raw             | rad_s                   | 1.92645e-05 |                0.101228  |                      0.10036   |                                0.244736 |
| raw             | rpm                     | 2.27428e-05 |                0.100186  |                      0.0971572 |                                0.292669 |
| raw             | rps                     | 2.57445e-05 |                0.184264  |                      0.18848   |                                0.311116 |

Values above are medians across all 200 records under three **unverified** speed-unit interpretations and assumed modulation order one. They do not establish a known shaft or defect order. A different CK/PMI ranking from the MSE ranking shows that the objectives differ; it does not identify a physically superior detector. Interpolation error remains appreciable and is itself part of the sensitivity analysis.

The inspected normalized speed range is **-0.0096913894 to 1.0018221697**. **0.622453%** of all speed samples lie outside the supplied reference interval; no clipping or per-record normalization was introduced. There are no health-state, severity or onset fields in the inspected eight-field structs.

![All-record raw/residual trends](figures/fault_rms_trends.png)

*The horizontal coordinate preserves source order, not a validated progression of damage. Identical burn-in is applied to every raw/residual trace.*

![Record descriptor distributions](figures/fault_indicator_distributions.png)

*These distributions describe 200 recordings. Suppressed fliers in a box plot are not removed from the CSV or summary statistics.*

![Model objectives compared](figures/fault_prediction_diagnostic_tradeoff.png)

*Candidate-record prediction MSE and assumption-based PMI are distinct descriptors; larger PMI is not verified physical detection success.*

![All-record unit sensitivity](figures/fault_unit_sensitivity.png)

*The chosen unit changes the angular grid, period and resulting PMI. No unit is selected merely because its plot is more separated.*

![Fixed record waveform](figures/fault_neutral_record_overlay.png)

*Provenance record 1 is fixed by the neutral selection policy, not selected for dramatic appearance.*

The full numerical evidence is [fault_record_features.csv](fault_record_features.csv), [fault_angular_features.csv](fault_angular_features.csv), [paired summary](fault_paired_summary.csv), [speed/index associations](fault_feature_correlations.csv), and [study plan](fault_study_plan.json). Record IDs 1, 50, 100, 150 and 200 have retained traces; all 200 have restartable scalar-feature caches. No fault data were used for model fitting or hyperparameter tuning.
