"""Measured identical-record batch inference timing for frozen predictors."""
import json
import platform
import subprocess
from time import perf_counter

import numpy as np
import pandas as pd
import psutil
import scipy
from threadpoolctl import threadpool_info, threadpool_limits


def benchmark(predictors, records, run, device="cpu", threads=1, repeats=7):
    try:
        import torch
        gpu = torch.cuda.is_available()
        torch.set_num_threads(threads)
        sync = torch.cuda.synchronize if device == "cuda" else lambda: None
        torch_info = {"torch": torch.__version__, "cuda_built": torch.version.cuda,
                      "cuda_available": gpu, "device": torch.cuda.get_device_name(0) if gpu else None}
        if device == "cuda" and not gpu:
            raise RuntimeError("CUDA requested but unavailable")
    except ImportError:
        if device == "cuda":
            raise RuntimeError("CUDA requested but PyTorch is unavailable")
        torch_info = {"cuda_available": False, "torch": None}
        sync = lambda: None
    run.mkdir(parents=True, exist_ok=True)
    cpuinfo = subprocess.run(["lscpu"], capture_output=True, text=True, check=False).stdout if platform.system() == "Linux" else ""
    cpuline = next((line.split(":", 1)[1].strip() for line in cpuinfo.splitlines()
                    if line.startswith("Model name:")), platform.processor())
    process = psutil.Process()
    rows, all_samples = [], []
    with threadpool_limits(limits=threads):
        metadata = {"host": platform.platform(), "cpu": cpuline, "visible_cpus": psutil.cpu_count(),
                    "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                    "blas": threadpool_info(), "threads": threads, "repeat_count": repeats,
                    "scope": "batch inference from in-memory NumPy record including features, transfer, warm-up and output",
                    "device_requested": device, **torch_info}
        for model, predict in predictors.items():
            for label, record in records.items():
                def measure():
                    sync()
                    start = perf_counter()
                    predict(record)
                    sync()
                    return perf_counter()-start
                cold = measure()
                times = np.asarray([measure() for _ in range(repeats)])
                all_samples.extend({"model": model, "record": label, "repeat": i+1, "seconds": v}
                                   for i, v in enumerate(times))
                duration = len(record.x)/record.fs
                median = np.median(times)
                rows.append(dict(model=model, record=label, device=device, sample_count=len(record.x),
                                 median_s=median, iqr_s=np.ptp(np.quantile(times, [.25, .75])),
                                 p95_s=np.quantile(times, .95), cold_s=cold, rtf=median/duration,
                                 samples_per_s=len(record.x)/median, process_rss_bytes=process.memory_info().rss))
    pd.DataFrame(rows).to_csv(run / "timings.csv", index=False)
    pd.DataFrame(all_samples).to_csv(run / "timing_repeats.csv", index=False)
    (run / "timing_environment.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
