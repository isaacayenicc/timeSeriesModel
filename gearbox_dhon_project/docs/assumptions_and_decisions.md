# Experiment decisions

| Decision | Basis and implication |
|---|---|
| Freeze the strict cutoff and 1,792 common warm-up | The current-speed paper schedule is a reference variant. Strict estimators consume speed and vibration through t−1. The selected DHON's base bound plus correction reaches 1,792. |
| Use stored vibration values | The input reconstructs stored channel-one raw values from its own documented mean and scale. No per-record restandardization or new filtering was applied. |
| Interpret labels as index groups | No authoritative labels or acquisition start metadata were supplied. Eligible class-like summaries are conditional and no operational threshold is frozen. |
| Retain both PMI lags | The instructor script accesses the 1-based ACF at T. The literal result has lag T−1, and the mathematical convention has lag T. Python parity is scoped to the derivable convention. |
| Use 2,048 angular samples/revolution | This reproduces the nominal archive grid density and supports observed high orders. Interpolation cannot add bandwidth absent from time sampling. |
| Use paired blocks of 640 samples | Local serial dependence is preserved to a degree. The interval is a within-record sensitivity statistic and does not establish replication across acquisitions. |
| Treat archived strongest as reconstructed metric | Its predictions are available and align with the course data. A checkpoint for the older ancestor is absent. The best self-contained archive checkpoint is a separate reproducibility anchor. |
| Report native CPU timing | Direct measured CPU costs and CUDA detection are included. Streaming and deployed hard latency require separate evidence. |
| Preserve official dates as recorded | The brief mixes a 2026 progress date with 2025 final milestones. An instructor clarification would be needed to correct them. |
