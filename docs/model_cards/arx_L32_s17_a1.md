# arx_L32_s17_a1

past-only causal. Teacher-forced one-step validation, indices 256–23999, provided dimensionless x. Lag 32; fs=3200 Hz.

```text
arx least-squares/ridge causal lag predictor or fixed baseline
```

```json
{
  "kind": "arx",
  "lag": 32,
  "seed": 17,
  "alpha": 1.0,
  "run_id": "arx_L32_s17_a1",
  "burnin": 256,
  "device": "cpu",
  "precision": "float64 linear/float32 neural",
  "train_sha256": "47d23efd20d86ed9e67181a96c9b73039a79a874e0d9ae91be5bcb8a421c19a6",
  "valid_sha256": "07013d4b17d7ae6fc740365264def53882664a49fde278b0d839979652b99b73",
  "family": "linear"
}
```

MSE 0.517247448; parameters 35; measured training wall 0.00946375s; public record inference median 0.00168887s. Hardware and timing definitions: ../../reports/timing_and_hardware.md. Artifact SHA256 9776a05d38752225006668a39f35f32bfadc8df88d28a0120878d40590398b3f. Predictions in ../../experiments/runs/arx_L32_s17_a1/predictions.npz.

Training/early stopping uses supplied train/validation traces; no independent test exists. No physical fault accuracy established. This model is not a reproduced Spa LPV-ARMA. All speed-aware primary inputs are past-only; shift_tcn is a deliberately noncausal offline speed control excluded from best-model selection. Linear polynomial design is not additionally standardized; regularization acts on its supplied basis. See config for exact optimizer and capacity.
