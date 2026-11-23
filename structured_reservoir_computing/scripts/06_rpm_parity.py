"""Retrain matched fixed-config vibration-only baseline variants."""
import argparse
from pathlib import Path

from src_gearbox.models.baselines.training import SPEED_CONTROLS, fit_speed_controls

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--models",nargs="+",choices=SPEED_CONTROLS,default=SPEED_CONTROLS)
    args=parser.parse_args()
    fit_speed_controls(args.root,args.data_dir,args.models)
