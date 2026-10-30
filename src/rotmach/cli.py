"""Cross-platform research workflow command-line interface."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import yaml
from .io_matlab import load_records, sha256


def stage_data(root, source):
    """Copy user input into canonical raw folder, refusing conflicting overwrites."""
    if source is None:
        return
    source=Path(source).resolve(); dest=root/'data/raw'; dest.mkdir(parents=True,exist_ok=True)
    if source==dest.resolve():
        return
    for name in ('sigtrain.mat','sigvalid.mat','sigfault1.mat','sigtest.mat','sigfault2.mat'):
        p=source/name
        if p.exists():
            target=dest/name
            if target.exists() and sha256(target)!=sha256(p):
                raise ValueError(f'Refusing to replace different raw input: {target}')
            if not target.exists():
                shutil.copy2(p,target)
    parts=source/'sigfault1_parts'
    if parts.is_dir():
        target_parts=dest/'sigfault1_parts';target_parts.mkdir(exist_ok=True)
        for p in parts.iterdir():
            if not p.is_file():continue
            target=target_parts/p.name
            if target.exists() and sha256(target)!=sha256(p):
                raise ValueError(f'Refusing to replace different split source: {target}')
            if not target.exists():shutil.copy2(p,target)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['audit','train','evaluate','eda','faults','summarize','report','slides','verify'])
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--config',type=Path)
    parser.add_argument('--data-dir',type=Path)
    parser.add_argument('--model',default='all',help='kind from config/experiment_plan.json or all')
    parser.add_argument('--input',type=Path,help='verified fault-parts directory or single MAT; never inferred labels')
    parser.add_argument('--max-records',type=int,help='fault pilot limit, outputs isolated from full results')
    parser.add_argument('--fresh',action='store_true',help='recompute fault study rather than use matching per-record cache')
    args=parser.parse_args(argv); root=args.root.resolve()
    cfg=yaml.safe_load(args.config.read_text()) if args.config else {}
    stage_data(root,args.data_dir or cfg.get('data_dir'))
    if args.command=='audit':
        from .audit import audit
        print(json.dumps(audit(root),indent=2))
    elif args.command=='train':
        from .audit import audit
        from .experiment import run_suite
        audit(root); run_suite(root,selected=None if args.model=='all' else args.model)
    elif args.command=='evaluate':
        from .evaluate import evaluate_saved
        print(evaluate_saved(root)[['run_id','mse']].to_string(index=False))
    elif args.command=='eda':
        from .plots import render_figures
        render_figures(root)
    elif args.command=='summarize':
        from .experiment import collect
        print(collect(root)[['run_id','mse']].to_string(index=False))
    elif args.command=='faults':
        p=args.input or root/'data/raw/sigfault1_parts'
        if p.is_dir():
            from .fault_study import run_fault_study
            result=run_fault_study(root,parts_dir=p,max_records=args.max_records,resume=not args.fresh)
            print(json.dumps(result,indent=2,default=str))
            return
        from .diagnostics import analyze_records
        from .experiment import load_model_and_predict
        import pandas as pd
        records=load_records(p)
        best=json.loads((root/'experiments/best_model.json').read_text())['run_id']
        predictors={best:lambda record:load_model_and_predict(root,best,record)}
        rows=analyze_records(records,predictors=predictors,burnin=256)
        pd.DataFrame(rows).to_csv(root/'reports/fault_record_features.csv',index=False)
        (root/'reports/fault_analysis_status.json').write_text(json.dumps(dict(status='processed descriptive features',
            input_file=str(p.name),input_sha256=sha256(p),real_fault_records_processed=len(records),
            model=best,labels_verified=False,speed_unit_verified=False),indent=2))
        print(f'Processed {len(records)} separate records; no class labels inferred; angular features omitted without verified unit.')
    elif args.command=='slides':
        from .slides_build import build_presentations
        build_presentations(root)
    elif args.command=='report':
        from .report_build import build_report, render_report_pdf
        build_report(root)
        render_report_pdf(root)
    elif args.command=='verify':
        from .release import verify
        print(json.dumps(verify(root),indent=2))


if __name__=='__main__':
    main()
