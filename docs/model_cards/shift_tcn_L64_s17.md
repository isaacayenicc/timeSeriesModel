# shift_tcn_L64_s17

NONCAUSAL offline speed control. Teacher-forced one-step validation, indices 256–23999, provided dimensionless x. Lag 64; fs=3200 Hz.

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
  "kind": "shift_tcn",
  "lag": 64,
  "seed": 17,
  "run_id": "shift_tcn_L64_s17",
  "burnin": 256,
  "device": "cpu",
  "precision": "float64 linear/float32 neural",
  "train_sha256": "47d23efd20d86ed9e67181a96c9b73039a79a874e0d9ae91be5bcb8a421c19a6",
  "valid_sha256": "07013d4b17d7ae6fc740365264def53882664a49fde278b0d839979652b99b73",
  "family": "neural",
  "source_sha256": {
    "spectral.py": "fe0aafebafd8efdc0dc439684017b8765e29005456ecf4f15211fb4188b56bad",
    "cli.py": "778519de9475c9771d0e474376945125247910f83ca0291a2ccd84a7db277a96",
    "__init__.py": "7ad775ec741401e95593c04c928281eade67ee8b077acd06e1aab091a125bdb8",
    "angle_resample.py": "427b78b15c06213344644c3ba7db2173fe049f898f0d51450ca6339f506cf35f",
    "release.py": "8a737bb4264ebc7473980c636514f5f7d2831403fdd8f683cec3d6e2c3235466",
    "experiment.py": "144017ee94b6db39b56e576337b0104749132eb2c3aa3c808a508d33b5891e28",
    "windows.py": "216f5fa8003172bf618481cf40c2c8ac6d82632b6551995c9642bc63c3093860",
    "report_build.py": "6bd7f61751e0777be3d8a2a49f8be65bba46b83932d46690b97030cd10a86746",
    "diagnostics.py": "1ecdee3fa492fb036fd79127e3991e3acd40be1d7f461be79100b6556c9b5133",
    "audit.py": "3f4f597446d56bde07a1e72cdc430aa530b5bafed188e1327091e3725f920714",
    "plots.py": "bce17c89503abe52c3f254630a97e7bf1cd6060215a024cfdfc623f463a0a6ff",
    "evaluate.py": "d0ab802d425c6ced590a563f9cde1d0f3d596531ecced9170103982729a7f353",
    "io_matlab.py": "ee9f20a269a548081929f9e9e483323467dc8a50c3d0c05e4c95ab544c550807",
    "models.py": "9310de0a0b5cb3f08a29b36337c7972fc793cb53190b5248d9dd3fefeb135b36",
    "slides_build.py": "a44aee5b0512d1d1727b267934219bbf3ce3928209bb895b8ab5ffdac4402045"
  },
  "config_fingerprint": "6793bae44244950a851ebf85df8503167244ddbbb0273161e9d90843fc27f5a9"
}
```

MSE 0.4636831899; parameters 2185; measured training wall 25.9782s; public record inference median 0.586385s. Hardware and timing definitions: ../../reports/timing_and_hardware.md. Artifact SHA256 613c837c34a21f694e669503add0da85964e3e553b25dfd938fa4bea6fe73549. Predictions in ../../experiments/runs/shift_tcn_L64_s17/predictions.npz.

Training/early stopping uses supplied train/validation traces; no independent test exists. No physical fault accuracy established. This model is not a reproduced Spa LPV-ARMA. All speed-aware primary inputs are past-only; shift_tcn is a deliberately noncausal offline speed control excluded from best-model selection. Linear polynomial design is not additionally standardized; regularization acts on its supplied basis. See config for exact optimizer and capacity.
