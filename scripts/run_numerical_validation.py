"""Regenerate angular-unit sensitivity from actual inputs and a saved model.

Run from any working directory: python scripts/run_numerical_validation.py
The tables use supplied normalized x and same-scale residuals, never per-series
standardization. They are assumption sensitivity, not physical-fault evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rotmach.angle_resample import angle_resample  # noqa: E402
from rotmach.diagnostics import (  # noqa: E402
    correlated_kurtosis, pmi, signal_features,
)
from rotmach.experiment import load_model_and_predict  # noqa: E402
from rotmach.io_matlab import load_records, sha256  # noqa: E402


def regenerate(root: Path, best_run: str | None = None) -> pd.DataFrame:
    """Recompute both unit assumptions and PMI indexing variants, saving CSVs."""
    best_file = root / "experiments/best_model.json"
    if best_run is None:
        if not best_file.exists():
            raise FileNotFoundError("Train/collect the final suite before this script")
        best_run = json.loads(best_file.read_text())["run_id"]
    config = json.loads((root / "experiments/runs" / best_run / "config.json").read_text())
    if config["kind"] == "shift_tcn":
        raise ValueError("The noncausal off-line speed shift cannot be the recommended model")
    burnin = int(config["burnin"])
    stats, rows = [], []
    for split in ("train", "valid"):
        filename = f"sig{split}.mat"
        path = root / "data/raw" / filename
        record = load_records(path)[0]
        reference_rms = signal_features(record["x"][burnin:])["rms"]
        series = {"supplied_x": record["x"][burnin:]}
        if split == "valid":
            prediction = load_model_and_predict(root, best_run, record)
            series[f"residual:{best_run}"] = record["x"][burnin:] - prediction[burnin:]
        for name, y in series.items():
            features = signal_features(y)
            stats.append({
                "split": split, "series": name,
                "input_sha256": sha256(path), "n": len(y),
                "first_target_index": burnin, "last_target_index": record["N"] - 1,
                "amplitude_scale": "supplied_normalized_x_no_rescaling",
                "rms_fraction_of_supplied_x": features["rms"] / reference_rms,
                **features,
            })
            for unit, factor in (("rpm", 1 / 60), ("rad_s", 1 / (2*np.pi)), ("rps", 1)):
                # Choose enough output samples/cycle to avoid nominal angular
                # downsampling anywhere under each explicitly assumed speed unit.
                maximum_density = record["fs"] / np.min(record["spd"][burnin:] * factor)
                density = int(2 ** np.ceil(np.log2(max(2, maximum_density))))
                start = perf_counter()
                angular = angle_resample(
                    y, record["t"][burnin:], record["spd"][burnin:], unit,
                    samples_per_cycle=density,
                )
                output = {
                    "split": split, "series": name,
                    "input_sha256": sha256(path),
                    "source_n": len(y), "first_target_index": burnin,
                    "last_target_index": record["N"] - 1,
                    "amplitude_scale": "supplied_normalized_x_no_rescaling",
                    "speed_unit_assumption": unit,
                    "physical_speed_unit_verified": False,
                    "assumed_modulation_order": 1,
                    "fault_order_verified": False,
                    "maximum_source_samples_per_cycle": maximum_density,
                    "samples_per_cycle": density,
                    "angular_n": len(angular["signal"]),
                    "edge_loss_cycles": angular["edge_loss_cycles"],
                    "roundtrip_rmse": angular["roundtrip_rmse"],
                    "roundtrip_rmse_fraction_of_series_rms": angular["roundtrip_rmse"] / features["rms"],
                    "ck_order1": correlated_kurtosis(angular["signal"], density),
                    "rms_source": features["rms"],
                    "rms_fraction_of_supplied_x": features["rms"] / reference_rms,
                }
                for variant in ("faithful_reference", "scientifically_corrected"):
                    result = pmi(angular["signal"], density, variant=variant)
                    for key in ("pmi", "rho", "selected_lag_samples", "status"):
                        output[f"{key}_{variant}"] = result[key]
                output["feature_seconds"] = perf_counter() - start
                rows.append(output)
    frame = pd.DataFrame(rows)
    (root / "reports").mkdir(exist_ok=True)
    frame.to_csv(root / "reports/angular_unit_sensitivity.csv", index=False)
    pd.DataFrame(stats).to_csv(root / "reports/signal_statistics.csv", index=False)
    provenance = {
        "command": "python scripts/run_numerical_validation.py",
        "best_run": best_run, "burnin": burnin,
        "checkpoint_hash": json.loads((root / "experiments/runs" / best_run / "metrics.json").read_text())["artifact_sha256"],
        "meaning": "unit-assumption and interpolation sensitivity; not verified fault evidence",
        "pmi_envelope": "matlab_mean_restored_candidate; no live MATLAB parity",
        "raw_scaling": "supplied x retained; xraw not used; no per-record standardization",
        "residual_scaling": "x-yhat on same normalized scale and target indices",
    }
    (root / "reports/angular_unit_sensitivity_provenance.json").write_text(json.dumps(provenance, indent=2))
    report = root / "reports/numerical_validation.md"
    if report.exists():
        content = report.read_text()
        start = content.index("## Executed sensitivity on the supplied reference records")
        end = content.index("## Unexecuted and blocked checks", start)
        replacement = _markdown_summary(frame, pd.DataFrame(stats), provenance)
        report.write_text(content[:start] + replacement + content[end:])
    print(frame[["split", "series", "speed_unit_assumption", "samples_per_cycle", "pmi_scientifically_corrected"]].to_string(index=False))
    return frame


def _markdown_summary(frame: pd.DataFrame, stats: pd.DataFrame, provenance: dict) -> str:
    """Build the measured section using CSV values, never hand-entered results."""
    lines = [
        "## Executed sensitivity on the supplied reference records", "",
        "This section is regenerated by `python scripts/run_numerical_validation.py`.",
        "Full precision and input hashes are in [angular_unit_sensitivity.csv](angular_unit_sensitivity.csv),",
        "[signal_statistics.csv](signal_statistics.csv) and",
        "[provenance JSON](angular_unit_sensitivity_provenance.json).", "",
        f"The validation residual uses the frozen recommended checkpoint `{provenance['best_run']}`.",
        f"All series use zero-based targets [{provenance['burnin']}:24000] (23,744 samples).",
        "The raw reference is the **supplied normalized x**, not xraw. Residuals",
        "are x-yhat in the same dimensionless amplitude scale. No series or record",
        "is re-standardized. RMS fractions refer to supplied x on the identical segment.", "",
        "| Split | Series | RMS | Fraction of raw x RMS | Crest | Pearson kurtosis |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in stats.to_dict("records"):
        lines.append(f"| {row['split']} | {row['series']} | {row['rms']:.8f} | {row['rms_fraction_of_supplied_x']:.8f} | {row['crest_factor']:.8f} | {row['kurtosis']:.8f} |")
    lines += ["", "Samples/cycle is the smallest power of two >=fs/minimum assumed cycles/second.",
              "This prevents nominal density downsampling, but linear interpolation still filters",
              "high-frequency content. The speed units and assumed one-cycle modulation order",
              "are **unverified assumptions**, not confirmed physics or fault identity.", "",
              "| Split | Series | Assumed unit | Samples/cycle | Angular N | Round-trip RMSE | CK (M=1) | Corrected PMI | Reference-index PMI |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in frame.to_dict("records"):
        lines.append(f"| {row['split']} | {row['series']} | {row['speed_unit_assumption']} | {row['samples_per_cycle']} | {row['angular_n']} | {row['roundtrip_rmse']:.8f} | {row['ck_order1']:.8g} | {row['pmi_scientifically_corrected']:.8f} | {row['pmi_faithful_reference']:.8f} |")
    lines += ["", "Material variation between assumed units shows why undocumented unit selection",
              "can change a diagnostic conclusion. Round-trip errors quantify transformation loss,",
              "not error against an unknown continuous physical signal. CK is length-dependent:",
              "raw/residual comparisons within one assumption have equal length, whereas comparisons",
              "between unit assumptions have different angular N. Neither larger PMI nor kurtosis",
              "establishes superior physical-fault detection without labels and verified shaft/order metadata.", "", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--best-run", default=None)
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()
    if args.run_tests:
        subprocess.run([sys.executable, "-m", "pytest", "tests/test_diagnostics.py", "-q"], cwd=args.root, check=True)
    regenerate(args.root.resolve(), args.best_run)


if __name__ == "__main__":
    main()
