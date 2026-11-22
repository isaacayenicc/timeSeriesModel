"""Run or replay the study in named stages with isolated logs."""
from collections import OrderedDict
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def commands(data,mode):
    raw=["--data-dir",str(data)]
    s=lambda name,*args: ["scripts/"+name,*args]
    stages=OrderedDict(
        preflight=s("00_audit_inputs.py",*raw,"--source-dir",str(ROOT/"inputs/reference_archives")),
        prepare=s("01_prepare_data.py",*raw),
        eda=s("02_run_eda.py",*raw),
        reference=s("02_reproduce_lpv_reference.py",*raw),
        optimize=s("05_optimize_src.py",*raw,"--profile","full" if mode=="full" else "standard"),
        fit=s("04_train_src.py",*raw),
        score=["-m","src_gearbox.study","score",*raw],
        baselines=s("06_train_baselines.py",*raw,"--tier","standard"),
        full_baseline_refit=s("06_train_baselines.py",*raw,"--tier","all"),
        comparison=s("07_compare_models.py"),
        ablations=s("09_ablation_robustness.py",*raw),
        fault=s("evaluate_faults.py",*raw,"--out",str(ROOT/"results/fault/reproduction"),"--figures",str(ROOT/"figures/fault/reproduction"),
                "--src","primary_pi_src=checkpoints/src/primary_pi_src.npz",
                "--src","pi_src_same_weights_no_adapter=checkpoints/src/primary_pi_src.npz",
                "--baseline","sparse_lpv_arma","--baseline","archived_src_corrected",
                "--baseline","ridge","--baseline","cnn","--cohort","primary_pi_src","pi_src_same_weights_no_adapter","sparse_lpv_arma","archived_src_corrected"),
        frozen=s("10_frozen_final_test.py",*raw),
        search_figures=s("11_build_figures.py","--only-search"),
        notebooks=s("build_notebooks.py",*raw),
        word=s("build_report.py","--pages"),
        presentations=s("build_presentations.py","--render"),
        tests=["-m","pytest","-q"],
        smoke=s("06_train_baselines.py","--tier","smoke"),
        verify=s("15_verify_outputs.py"),
    )
    return stages


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=["smoke","standard","full"],default="standard")
    p.add_argument("--data-dir",type=Path,default=ROOT/"data/raw")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--only",nargs="+",help="Run named stages in the order given")
    args=p.parse_args()
    table=commands(args.data_dir.resolve(),args.mode)
    if args.only:
        if not set(args.only)<=set(table):
            p.error("Available stages are "+", ".join(table))
        plan=args.only
    elif args.mode=="smoke":
        plan=["tests","smoke"]
    elif args.mode=="standard":
        plan=["score","baselines","comparison","fault","frozen","search_figures","tests","verify"]
    else:
        plan=[stage for stage in table if stage!="smoke"]
    env=os.environ.copy()
    env["PYTHONPATH"]=os.pathsep.join([str(ROOT/"src"),env.get("PYTHONPATH","")]).rstrip(os.pathsep)
    env.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    folder=ROOT/"logs/reproduction"
    folder.mkdir(parents=True,exist_ok=True)
    summary=[]
    for stage in plan:
        run=[sys.executable,*table[stage]]
        if args.resume and stage in {"optimize","full_baseline_refit"}:
            run.append("--resume")
        output=folder/(stage+".log")
        start=time.perf_counter()
        with output.open("w",encoding="utf-8") as log:
            code=subprocess.run(run,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        state=dict(stage=stage,returncode=code,wall_s=time.perf_counter()-start,log=output.relative_to(ROOT).as_posix())
        summary.append(state)
        print(f"{stage}: {'PASS' if code==0 else 'FAIL'} ({state['wall_s']:.2f} s)",flush=True)
        (folder/"stage_status.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
        if code:
            raise SystemExit(f"{stage} failed. See {output}")


if __name__=="__main__":
    main()
