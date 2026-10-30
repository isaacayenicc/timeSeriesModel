"""Standard checkpoint regeneration or full CPU retraining; never downloads data."""
import argparse
import os
import sys
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from rotmach.cli import stage_data

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path)
    p.add_argument('--level',choices=['standard','full'],default='standard')
    p.add_argument('--device',choices=['cpu'],default='cpu',help='Only CPU execution was validated')
    a=p.parse_args(); root=Path(__file__).resolve().parent
    stage_data(root,a.data_dir)
    from rotmach.audit import audit
    from rotmach.experiment import run_suite
    from rotmach.evaluate import evaluate_saved
    from rotmach.plots import render_figures
    from rotmach.report_build import build_report
    from rotmach.slides_build import build_presentations
    from rotmach.report_build import render_report_pdf
    from scripts.run_numerical_validation import regenerate
    from scripts.refresh_model_cards import refresh
    from rotmach.fault_study import run_fault_study
    audit(root)
    if a.level=='full':run_suite(root)
    evaluate_saved(root);regenerate(root);refresh(root);render_figures(root)
    if (root/'data/raw/sigfault1_parts/manifest.json').exists():
        run_fault_study(root,resume=a.level=='standard')
        from rotmach.plots import render_fault_workflow_figure
        render_fault_workflow_figure(root)
        render_figures(root)  # merge the fault figure provenance sidecar
    build_report(root);render_report_pdf(root);build_presentations(root)
    print('Recomputed checkpoint scores, descriptive tables, figures, report and editable slides. Verified fault parts trigger the full per-record study, reusing only matching per-record caches in standard mode. Notebooks: scripts/build_notebooks.py --execute.')
