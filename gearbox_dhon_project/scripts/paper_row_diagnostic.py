"""Reproduce the historical random-row convention as a diagnostic experiment."""
import argparse
from pathlib import Path
from threadpoolctl import threadpool_limits

from gearbox_dhon.baselines import find_record, mat_record, paper_row_diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--basis", choices=("legendre", "monomial"), default="legendre")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    train, valid = (mat_record(find_record(args.data_dir, name), name) for name in ("sigtrain", "sigvalid"))
    with threadpool_limits(args.threads):
        rows = paper_row_diagnostic(train, valid, args.run_dir, args.basis)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
