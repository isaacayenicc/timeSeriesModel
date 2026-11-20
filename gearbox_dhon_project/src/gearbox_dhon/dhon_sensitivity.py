"""Post-selection pulse sensitivity and supplemental seed replication."""

import argparse
import csv
import itertools
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from .dhon import Config, SCORE_START, fit, injected_pulses, load_model, slow_adapter
from .dhon_search import _read


def write_csv(path, rows):
    with path.open("w",newline="",encoding="utf-8") as output:
        writer=csv.DictWriter(output,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(data_dir: Path, run_dir: Path, extend_seeds: bool = False) -> None:
    valid = _read(data_dir, "sigvalid")
    y, rpm, fs = valid["x"], valid["spd"], float(valid["fs"])
    metrics = run_dir / "metrics"
    checkpoints = run_dir / "checkpoints"
    search = run_dir / "search"
    states = ("primary_dhon", "alternative_dhon", "dhon_matched_fixed")
    selection = json.loads((search / "dhon_selection.json").read_text())
    idx = np.arange(SCORE_START, len(y))
    if extend_seeds:
        training = _read(data_dir, "sigtrain")
        config = Config(**selection["configurations"]["primary_dhon"])
        results = []
        predictions = {"indices":idx, "actual":y, "rpm":rpm, "time":valid["t"]}
        for seed in (59, 71):
            start = perf_counter()
            model, trace = fit(config, [(training["x"],training["spd"])],fs,seed,"cpu")
            slug = f"primary_dhon_seed{seed}"
            path = checkpoints / f"{slug}.npz"
            model.save(path)
            (search / f"{slug}_learning.json").write_text(json.dumps(trace, indent=2)+"\n")
            prediction = load_model(path).predict(y,rpm,fs)
            predictions[slug] = prediction
            results.append({"model":slug,"seed":seed,"n":len(idx),
                            "mse":float(np.mean((prediction[idx]-y[idx])**2)),
                            "elapsed_s":perf_counter()-start})
        write_csv(metrics/"dhon_supplemental_seeds.csv",results)
        np.savez_compressed(run_dir/"predictions"/"dhon_supplemental_seeds.npz",**predictions)
    rows = []
    for slug in states:
        path = checkpoints / f"{slug}.npz"
        if not path.exists():
            continue
        model = load_model(path)
        c = model.config
        base = model.predict_base(y, rpm, fs)
        adapted = slow_adapter(y, base, c.adapter_gain, c.adapter_tau,c.adapter_memory,c.adapter_clip)
        for hz, amp, phase in itertools.product((430., 930., 1100.),(.45,1.2),(.19,.77)):
            pulse = injected_pulses(rpm, fs, hz, phase, amp)
            dressed = y + pulse
            pulse_base = model.predict_base(dressed,rpm,fs)
            pulse_adapted = slow_adapter(dressed,pulse_base,c.adapter_gain,c.adapter_tau,
                                         c.adapter_memory,c.adapter_clip)
            for adapter,pred,pulse_pred in (("off",base,pulse_base),("on",adapted,pulse_adapted)):
                delta = (pulse_pred-pred)[idx]
                truth = pulse[idx]
                carried = truth - delta
                power = np.dot(truth,truth)
                rows.append({"model":slug,"adapter":adapter,"carrier_hz":hz,
                             "amplitude_standardized":amp,"phase_turn":phase,"decay_s":.0035,
                             "n":len(idx),"pulse_distortion":float(np.dot(delta,delta)/power),
                             "residual_pulse_projection":float(np.dot(carried,truth)/power),
                             "residual_pulse_correlation":float(np.corrcoef(carried,truth)[0,1])})
    write_csv(metrics/"dhon_pulse_grid.csv",rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--extend-seeds",action="store_true")
    options=parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(1):
        run(options.data_dir,options.run_dir,options.extend_seeds)


if __name__=="__main__":
    main()
