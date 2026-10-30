"""Run cached frozen-model fault-candidate analysis without inventing labels."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from rotmach.fault_study import run_fault_study

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--model-id",action="append",default=None)
    parser.add_argument("--max-records",type=int,default=None,help="Pilot only, never complete status")
    parser.add_argument("--recompute",action="store_true")
    args=parser.parse_args()
    print(json.dumps(run_fault_study(args.root.resolve(),args.model_id,args.max_records,
                                    resume=not args.recompute),indent=2,default=str))
