"""Recompute strict, archive and published LPV reference comparisons."""
import argparse
from pathlib import Path
from src_gearbox.models.baselines.reference_replay import run

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--fit-paper",action="store_true",help="Repeat both current-RPM five-fold fits.")
    args=parser.parse_args()
    run(args.root,args.data_dir,fit_paper=args.fit_paper)
