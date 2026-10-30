# speed_tcn_L32_s17

past-only causal. Teacher-forced one-step validation, indices 256–23999, provided dimensionless x. Lag 32; fs=3200 Hz.

```text
Predictor(
  (stem): Conv1d(1, 8, kernel_size=(1,), stride=(1,))
  (blocks): ModuleList(
    (0): CausalBlock(
      (a): Conv1d(8, 8, kernel_size=(3,), stride=(1,))
      (b): Conv1d(8, 8, kernel_size=(3,), stride=(1,))
    )
    (1): CausalBlock(
      (a): Conv1d(8, 8, kernel_size=(3,), stride=(1,), dilation=(2,))
      (b): Conv1d(8, 8, kernel_size=(3,), stride=(1,), dilation=(2,))
    )
    (2): CausalBlock(
      (a): Conv1d(8, 8, kernel_size=(3,), stride=(1,), dilation=(4,))
      (b): Conv1d(8, 8, kernel_size=(3,), stride=(1,), dilation=(4,))
    )
    (3): CausalBlock(
      (a): Conv1d(8, 8, kernel_size=(3,), stride=(1,), dilation=(8,))
      (b): Conv1d(8, 8, kernel_size=(3,), stride=(1,), dilation=(8,))
    )
  )
  (film): ModuleList(
    (0-3): 4 x Linear(in_features=1, out_features=16, bias=True)
  )
  (head): Linear(in_features=8, out_features=1, bias=True)
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
  "kind": "speed_tcn",
  "lag": 32,
  "seed": 17,
  "run_id": "speed_tcn_L32_s17",
  "burnin": 256,
  "device": "cpu",
  "precision": "float64 linear/float32 neural",
  "train_sha256": "47d23efd20d86ed9e67181a96c9b73039a79a874e0d9ae91be5bcb8a421c19a6",
  "valid_sha256": "07013d4b17d7ae6fc740365264def53882664a49fde278b0d839979652b99b73",
  "family": "neural"
}
```

MSE 0.5122363885; parameters 1753; measured training wall 10.5194s; public record inference median 0.252117s. Hardware and timing definitions: ../../reports/timing_and_hardware.md. Artifact SHA256 c52683906e852fe16d73fe22fc76cbc558d322e9818bee39192ac5b6d4fb2305. Predictions in ../../experiments/runs/speed_tcn_L32_s17/predictions.npz.

Training/early stopping uses supplied train/validation traces; no independent test exists. No physical fault accuracy established. This model is not a reproduced Spa LPV-ARMA. All speed-aware primary inputs are past-only; shift_tcn is a deliberately noncausal offline speed control excluded from best-model selection. Linear polynomial design is not additionally standardized; regularization acts on its supplied basis. See config for exact optimizer and capacity.
