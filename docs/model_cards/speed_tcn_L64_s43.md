# speed_tcn_L64_s43

past-only causal. Teacher-forced one-step validation, indices 256–23999, provided dimensionless x. Lag 64; fs=3200 Hz.

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
    (4): CausalBlock(
      (a): Conv1d(8, 8, kernel_size=(3,), stride=(1,), dilation=(16,))
      (b): Conv1d(8, 8, kernel_size=(3,), stride=(1,), dilation=(16,))
    )
  )
  (film): ModuleList(
    (0-4): 5 x Linear(in_features=1, out_features=16, bias=True)
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
  "lag": 64,
  "seed": 43,
  "run_id": "speed_tcn_L64_s43",
  "burnin": 256,
  "device": "cpu",
  "precision": "float64 linear/float32 neural",
  "train_sha256": "47d23efd20d86ed9e67181a96c9b73039a79a874e0d9ae91be5bcb8a421c19a6",
  "valid_sha256": "07013d4b17d7ae6fc740365264def53882664a49fde278b0d839979652b99b73",
  "family": "neural"
}
```

MSE 0.4595186528; parameters 2185; measured training wall 22.3633s; public record inference median 0.548469s. Hardware and timing definitions: ../../reports/timing_and_hardware.md. Artifact SHA256 5c2055584dd58e1d7af41eea2e05cd870b9ef4692d12b1b069a21afdffcd9629. Predictions in ../../experiments/runs/speed_tcn_L64_s43/predictions.npz.

Training/early stopping uses supplied train/validation traces; no independent test exists. No physical fault accuracy established. This model is not a reproduced Spa LPV-ARMA. All speed-aware primary inputs are past-only; shift_tcn is a deliberately noncausal offline speed control excluded from best-model selection. Linear polynomial design is not additionally standardized; regularization acts on its supplied basis. See config for exact optimizer and capacity.
