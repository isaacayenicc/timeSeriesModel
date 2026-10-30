# mean_L256_s17

past-only causal. Teacher-forced one-step validation, indices 256–23999, provided dimensionless x. Lag 256; fs=3200 Hz.

```text
mean least-squares/ridge causal lag predictor or fixed baseline
```

```json
{
  "kind": "mean",
  "lag": 256,
  "seed": 17,
  "run_id": "mean_L256_s17",
  "burnin": 256,
  "device": "cpu",
  "precision": "float64 linear/float32 neural",
  "train_sha256": "47d23efd20d86ed9e67181a96c9b73039a79a874e0d9ae91be5bcb8a421c19a6",
  "valid_sha256": "07013d4b17d7ae6fc740365264def53882664a49fde278b0d839979652b99b73",
  "family": "linear"
}
```

MSE 1.104016698; parameters 1; measured training wall 0.000256046s; public record inference median 0.000316878s. Hardware and timing definitions: ../../reports/timing_and_hardware.md. Artifact SHA256 a5ebb51372a25d10100a96241a7d4d14900449fd1f66dd7014f8288a966a0578. Predictions in ../../experiments/runs/mean_L256_s17/predictions.npz.

Training/early stopping uses supplied train/validation traces; no independent test exists. No physical fault accuracy established. This model is not a reproduced Spa LPV-ARMA. All speed-aware primary inputs are past-only; shift_tcn is a deliberately noncausal offline speed control excluded from best-model selection. Linear polynomial design is not additionally standardized; regularization acts on its supplied basis. See config for exact optimizer and capacity.
