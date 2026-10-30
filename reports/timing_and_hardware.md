# Timing and hardware

```json
{
  "cpu": "INTEL(R) XEON(R) PLATINUM 8573C",
  "platform": "Linux-6.18.57-x86_64-with-glibc2.41",
  "python": "3.12.14",
  "numpy": "2.3.5",
  "scipy": "1.17.0",
  "sklearn": "1.8.0",
  "torch": "2.15.0+cpu",
  "cuda_available": false,
  "torch_threads": 2,
  "precision": "linear float64; neural float32",
  "training_wall_s_sum": 276.4010805279995,
  "superseded_control_training_wall_s": 33.496943697000006,
  "all_attempts_training_wall_s_sum": 309.8980242249995,
  "process_peak_rss_kib": 759844
}
```

CPU only; no GPU timing was measured. 2 PyTorch and 2 BLAS threads during original training. Preprocessing is separate. Training wall time includes fit/epochs, within-epoch validation, saving and a final forward pass; it excludes later reload/timing repetitions. Inference is the public load_model_and_predict full-record path: checkpoint loading + window preparation + scoring all N−L targets of the 24,000-sample validation record, one warmup then 3 repetitions, median/min/max. This is not device-only latency or sustained single-window streaming latency. Training counts differ with context. RSS is process-level Linux high-water memory for rescoring, not per-model incremental peak RAM. Timing varies with concurrent machine load and storage cache.
