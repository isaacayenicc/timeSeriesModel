"""Rerun deterministic physics-informed EDA on the designated training acquisition."""
import argparse
import json
from pathlib import Path

from src_gearbox.data import find_input, load_record
from src_gearbox.eda import run_eda
from src_gearbox.eda_extensions import run_extensions


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-dir", type=Path, default=root/"data/raw")
    cli.add_argument("--output", type=Path, default=root)
    args = cli.parse_args()
    path = find_input(args.data_dir, "sigtrain")
    if path is None:
        raise FileNotFoundError(f"Designated training MAT input absent in {args.data_dir}")
    record = load_record(path)
    out, figures = args.output/"results", args.output/"figures/eda"
    print(json.dumps(run_eda(record, out, figures), indent=2))
    print(json.dumps(run_extensions(record, out, figures), indent=2))
