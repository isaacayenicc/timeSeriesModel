"""Reproduce the training-only EDA."""
import argparse
from pathlib import Path

from gearbox_dhon.data import find_input, load_record
from gearbox_dhon.eda import run_eda

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    source = find_input(args.data_dir, "sigtrain")
    if source is None:
        parser.error("sigtrain MAT file is missing")
    run_eda(load_record(source), args.run_dir)
