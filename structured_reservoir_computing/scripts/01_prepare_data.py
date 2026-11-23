"""Record the common target mask and guarded fold protocol without transforming signals."""
import argparse
import json
from pathlib import Path

from src_gearbox.data.splits import prepare_from_inputs


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-dir", type=Path, default=root/"data/raw")
    cli.add_argument("--output", type=Path, default=root)
    args = cli.parse_args()
    print(json.dumps(prepare_from_inputs(args.data_dir, args.output/"data/splits"), indent=2))
