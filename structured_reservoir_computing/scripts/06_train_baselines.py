"""Replay frozen course baselines, or repeat guarded fits in a fresh run directory."""
import argparse
from pathlib import Path

from threadpoolctl import threadpool_limits
import yaml

from src_gearbox.models.baselines import replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-dir",type=Path,help="Raw MAT directory. Default: ROOT/data/raw.")
    parser.add_argument("--tier",choices=("smoke","standard","all"),default="standard")
    parser.add_argument("--out",type=Path,help="Separate directory for --tier all.")
    parser.add_argument("--resume",action="store_true",help="Reuse hash-checked completed fit stages.")
    parser.add_argument("--config",type=Path,help="Evaluation YAML. Default: ROOT/configs/evaluation.yaml.")
    args=parser.parse_args()
    cfg_path=args.config or args.root/"configs/evaluation.yaml"
    config=yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    actual=(config["sampling_hz"],config["target_start"],config["target_stop"],config["information_cutoff"])
    if actual!=(3200,1792,24000,"t-1"):
        parser.error("This fitted comparator registry is tied to the strict 3.2-kHz course mask.")
    data=args.data_dir or args.root/"data/raw"
    if args.tier=="smoke":
        from tempfile import TemporaryDirectory
        from src_gearbox.models.baselines.training import smoke, smoke_fit
        with threadpool_limits(1), TemporaryDirectory() as scratch:
            rows=smoke(args.root)
            fresh=smoke_fit(Path(scratch))
        print(f"{len(rows)} frozen and {len(fresh)} synthetic fit/save/replay contracts passed.")
    elif args.tier=="all":
        out=args.out or args.root/"results/experiments/baseline_refit"
        from src_gearbox.models.baselines.training import fit_all
        with threadpool_limits(1):
            fit_all(data,args.root,out,args.resume)
    else:
        rows=replay.run(args.root,data,repeats=int(config["warm_repeats"]))
        if len(rows)!=len(replay.BASELINE_SLUGS):
            raise SystemExit(f"Only {len(rows)} of {len(replay.BASELINE_SLUGS)} baselines reproduced")


if __name__=="__main__":
    main()
