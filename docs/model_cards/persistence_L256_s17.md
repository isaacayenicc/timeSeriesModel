# persistence_L256_s17

past-only causal. Teacher-forced one-step validation, indices 256–23999, provided dimensionless x. Lag 256; fs=3200 Hz.

```text
persistence least-squares/ridge causal lag predictor or fixed baseline
```

```json
{
  "kind": "persistence",
  "lag": 256,
  "seed": 17,
  "run_id": "persistence_L256_s17",
  "burnin": 256,
  "device": "cpu",
  "precision": "float64 linear/float32 neural",
  "train_sha256": "47d23efd20d86ed9e67181a96c9b73039a79a874e0d9ae91be5bcb8a421c19a6",
  "valid_sha256": "07013d4b17d7ae6fc740365264def53882664a49fde278b0d839979652b99b73",
  "family": "linear"
}
```

MSE 2.032666961; parameters 0; measured training wall 0.000201495s; public record inference median 0.000224492s. Hardware and timing definitions: ../../reports/timing_and_hardware.md. Artifact SHA256 c5e7cf4e4d8e0cc28451fb1e93385858b641f366105ddb48d8260b461f91bfc8. Predictions in ../../experiments/runs/persistence_L256_s17/predictions.npz.

Training/early stopping uses supplied train/validation traces; no independent test exists. No physical fault accuracy established. This model is not a reproduced Spa LPV-ARMA. All speed-aware primary inputs are past-only; shift_tcn is a deliberately noncausal offline speed control excluded from best-model selection. Linear polynomial design is not additionally standardized; regularization acts on its supplied basis. See config for exact optimizer and capacity.
