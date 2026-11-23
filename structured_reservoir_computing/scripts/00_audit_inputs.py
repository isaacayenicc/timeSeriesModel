"""Audit immutable course data, archive member hashes, and overlap candidates."""
import argparse
import json
from pathlib import Path

from src_gearbox.data.io_matlab import run_audit
from src_gearbox.data.provenance import build_input_manifest


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-dir", type=Path, default=root/"data/raw")
    cli.add_argument("--source-dir", type=Path, action="append", default=[])
    cli.add_argument("--output", type=Path, default=root)
    args = cli.parse_args()
    refs = args.source_dir or [root/"inputs/reference_archives"]
    inputs = build_input_manifest(args.data_dir, refs, args.output/"inputs")
    status = run_audit(args.data_dir, args.output/"results", args.output/"data/splits")
    if status["sigfault1_count"] and (status["sigtrain"] or status["sigvalid"]):
        from src_gearbox.data.audit_plots import plot_overlap
        plot_overlap(args.output/"results/record_overlap_audit.csv", args.output/"figures/eda/eda_overlaps.png")
    print(json.dumps({"source_audit": inputs, "data_audit": status}, indent=2))
