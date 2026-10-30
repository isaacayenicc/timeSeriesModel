# Common-index validation results

| run_id               | causal_status                   |      mse |   parameters |   train_wall_s |   inference_record_median_s |
|:---------------------|:--------------------------------|---------:|-------------:|---------------:|----------------------------:|
| lpv_L256_s17_a1      | past-only causal                | 0.361217 |          769 |    0.765804    |                 0.122543    |
| lpv_L128_s17_a1      | past-only causal                | 0.381163 |          385 |    0.450643    |                 0.0311916   |
| lpv_L64_s17_a1       | past-only causal                | 0.394738 |          193 |    0.0476541   |                 0.0128957   |
| ar_L256_s17_a1       | past-only causal                | 0.420058 |          257 |    0.0690292   |                 0.00664733  |
| ols_L256_s17         | past-only causal                | 0.420059 |          257 |    0.263791    |                 0.00636214  |
| ar_L128_s17_a1       | past-only causal                | 0.426825 |          129 |    0.0277835   |                 0.00323707  |
| arx_L256_s17_a1      | past-only causal                | 0.427821 |          259 |    0.0637356   |                 0.00775241  |
| arx_L128_s17_a1      | past-only causal                | 0.430216 |          131 |    0.0269577   |                 0.00412757  |
| ar_L64_s17_a1        | past-only causal                | 0.43905  |           65 |    0.0120172   |                 0.00177642  |
| arx_L64_s17_a1       | past-only causal                | 0.443191 |           67 |    0.014981    |                 0.00228043  |
| mlp_L64_s17          | past-only causal                | 0.450469 |         2113 |    1.48966     |                 0.00749581  |
| lpv_L32_s17_a1       | past-only causal                | 0.45431  |           97 |    0.0218477   |                 0.00599732  |
| tcn_L128_s17         | past-only causal                | 0.45449  |         2425 |   26.8125      |                 0.535576    |
| speed_tcn_L128_s17   | past-only causal                | 0.458226 |         2617 |   52.9852      |                 1.25977     |
| speed_tcn_L64_s43    | past-only causal                | 0.459519 |         2185 |   22.3633      |                 0.548469    |
| tcn_L64_s17          | past-only causal                | 0.461668 |         2025 |   11.9711      |                 0.223912    |
| late_tcn_L64_s17     | past-only causal                | 0.463274 |         2026 |   19.4111      |                 0.413156    |
| shift_tcn_L64_s17    | NONCAUSAL offline speed control | 0.463683 |         2185 |   25.9782      |                 0.586385    |
| tcn_L64_s29          | past-only causal                | 0.464516 |         2025 |   12.3592      |                 0.248665    |
| capacity_tcn_L64_s17 | past-only causal                | 0.471419 |         2548 |   21.7958      |                 0.459164    |
| speed_tcn_L64_s17    | past-only causal                | 0.483214 |         2185 |   20.0986      |                 0.567299    |
| tcn_L64_s43          | past-only causal                | 0.484686 |         2025 |   11.8114      |                 0.226302    |
| speed_tcn_L64_s29    | past-only causal                | 0.494911 |         2185 |   17.7249      |                 0.581169    |
| ar_L32_s17_a1        | past-only causal                | 0.503324 |           33 |    0.0106278   |                 0.00121017  |
| tcn_L32_s17          | past-only causal                | 0.508317 |         1625 |    6.95578     |                 0.112347    |
| speed_tcn_L32_s17    | past-only causal                | 0.512236 |         1753 |   10.5194      |                 0.252117    |
| arx_L32_s17_a1       | past-only causal                | 0.517247 |           35 |    0.00946375  |                 0.00168887  |
| gru_L64_s17          | past-only causal                | 0.823606 |          929 |   12.3397      |                 0.314197    |
| mean_L256_s17        | past-only causal                | 1.10402  |            1 |    0.000256046 |                 0.000316878 |
| zero_L256_s17        | past-only causal                | 1.10408  |            0 |    0.000366303 |                 0.000331034 |
| persistence_L256_s17 | past-only causal                | 2.03267  |            0 |    0.000201495 |                 0.000224492 |

One validation trace used for selection. Inference includes loading and window preparation. Shift-speed control is noncausal and not eligible for recommendation.
