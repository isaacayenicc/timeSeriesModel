# capacity_tcn_L64_s17

past-only causal. Teacher-forced one-step validation, indices 256–23999, provided dimensionless x. Lag 64; fs=3200 Hz.

```text
Predictor(
  (stem): Conv1d(1, 9, kernel_size=(1,), stride=(1,))
  (blocks): ModuleList(
    (0): CausalBlock(
      (a): Conv1d(9, 9, kernel_size=(3,), stride=(1,))
      (b): Conv1d(9, 9, kernel_size=(3,), stride=(1,))
    )
    (1): CausalBlock(
      (a): Conv1d(9, 9, kernel_size=(3,), stride=(1,), dilation=(2,))
      (b): Conv1d(9, 9, kernel_size=(3,), stride=(1,), dilation=(2,))
    )
    (2): CausalBlock(
      (a): Conv1d(9, 9, kernel_size=(3,), stride=(1,), dilation=(4,))
      (b): Conv1d(9, 9, kernel_size=(3,), stride=(1,), dilation=(4,))
    )
    (3): CausalBlock(
      (a): Conv1d(9, 9, kernel_size=(3,), stride=(1,), dilation=(8,))
      (b): Conv1d(9, 9, kernel_size=(3,), stride=(1,), dilation=(8,))
    )
    (4): CausalBlock(
      (a): Conv1d(9, 9, kernel_size=(3,), stride=(1,), dilation=(16,))
      (b): Conv1d(9, 9, kernel_size=(3,), stride=(1,), dilation=(16,))
    )
  )
  (head): Linear(in_features=9, out_features=1, bias=True)
)
```

```json
{
  "channels": 9,
  "learning_rate": 0.003,
  "weight_decay": 0.0001,
  "batch_size": 512,
  "epochs": 10,
  "patience": 3,
  "kind": "capacity_tcn",
  "lag": 64,
  "seed": 17,
  "run_id": "capacity_tcn_L64_s17",
  "burnin": 256,
  "device": "cpu",
  "precision": "float64 linear/float32 neural",
  "train_sha256": "47d23efd20d86ed9e67181a96c9b73039a79a874e0d9ae91be5bcb8a421c19a6",
  "valid_sha256": "07013d4b17d7ae6fc740365264def53882664a49fde278b0d839979652b99b73",
  "family": "neural"
}
```

MSE 0.4714192725; parameters 2548; measured training wall 21.7958s; public record inference median 0.459164s. Hardware and timing definitions: ../../reports/timing_and_hardware.md. Artifact SHA256 cfcdc21b249141684d1c3da750ee333006401e968d36a7a302f02e6c8d249b75. Predictions in ../../experiments/runs/capacity_tcn_L64_s17/predictions.npz.

Training/early stopping uses supplied train/validation traces; no independent test exists. No physical fault accuracy established. This model is not a reproduced Spa LPV-ARMA. All speed-aware primary inputs are past-only; shift_tcn is a deliberately noncausal offline speed control excluded from best-model selection. Linear polynomial design is not additionally standardized; regularization acts on its supplied basis. See config for exact optimizer and capacity.
