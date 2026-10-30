# mlp_L64_s17

past-only causal. Teacher-forced one-step validation, indices 256–23999, provided dimensionless x. Lag 64; fs=3200 Hz.

```text
Predictor(
  (net): Sequential(
    (0): Linear(in_features=64, out_features=32, bias=True)
    (1): ReLU()
    (2): Linear(in_features=32, out_features=1, bias=True)
  )
)
```

```json
{
  "channels": 8,
  "learning_rate": 0.003,
  "weight_decay": 0.0001,
  "batch_size": 512,
  "epochs": 10,
  "patience": 3,
  "kind": "mlp",
  "lag": 64,
  "seed": 17,
  "run_id": "mlp_L64_s17",
  "burnin": 256,
  "device": "cpu",
  "precision": "float64 linear/float32 neural",
  "train_sha256": "47d23efd20d86ed9e67181a96c9b73039a79a874e0d9ae91be5bcb8a421c19a6",
  "valid_sha256": "07013d4b17d7ae6fc740365264def53882664a49fde278b0d839979652b99b73",
  "family": "neural"
}
```

MSE 0.4504687603; parameters 2113; measured training wall 1.48966s; public record inference median 0.00749581s. Hardware and timing definitions: ../../reports/timing_and_hardware.md. Artifact SHA256 664eecf87d4ee2679ea869a940df62abf0de907e15622198613e65914c9b0a60. Predictions in ../../experiments/runs/mlp_L64_s17/predictions.npz.

Training/early stopping uses supplied train/validation traces; no independent test exists. No physical fault accuracy established. This model is not a reproduced Spa LPV-ARMA. All speed-aware primary inputs are past-only; shift_tcn is a deliberately noncausal offline speed control excluded from best-model selection. Linear polynomial design is not additionally standardized; regularization acts on its supplied basis. See config for exact optimizer and capacity.
