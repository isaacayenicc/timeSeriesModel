"""Build clean-kernel notebooks that independently inspect the released study."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import nbformat as n
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
PRELUDE = """from pathlib import Path
import os
import numpy as np
import pandas as pd
from IPython.display import Image, display
root = Path.cwd()
if root.name == 'notebooks':
    root = root.parent
run = root / os.getenv('GEARBOX_EXPANDED_RUN', 'outputs/expanded')
table = pd.read_csv(run/'comparison_metrics.csv')
"""

VIEWS = (
    ("01_Compare_All_Models", "Compare all requested models on official validation",
     "Use one mask and the same teacher-forced one-step target convention. Re-score every array and compare "
     "guarded-fold, training and development metrics.",
     """from gearbox_expanded.protocol import mse
with np.load(run/'matched_validation.npz', allow_pickle=False) as vectors:
    target = vectors['actual']
    assert np.array_equal(vectors['target_indices'], np.arange(1792, 24000))
    reconstructed = np.array([mse(target, vectors[key]) for key in table.model])
np.testing.assert_allclose(table.mse, reconstructed, atol=1e-12, rtol=0)
display(table[['display','fold_mean_mse','train_mse','mse','mae','batch_median_s']])
display(Image(filename=str(run/'figures/development_mse.png'), width=720))
"""),
    ("02_Guarded_Folds_and_Uncertainty", "Inspect model selection and temporal uncertainty",
     "The held blocks are inside the training acquisition and include two-sided guards. Moving-block uncertainty "
     "pairs errors on identical targets and varies block length. It does not independently replicate the study.",
     """folds = pd.read_csv(run/'guarded_trials.csv')
display(folds.groupby(['family','model']).mse.agg(['size','min','median','max']))
pairs = pd.read_csv(run/'paired_mse_intervals.csv')
display(pairs.query("reference == 'primary_dhon' and model in ['linear','sparse_lpv_arma','tft']"))
display(Image(filename=str(run/'figures/paired_intervals.png'), width=720))
"""),
    ("03_Timing_Faults_and_Stress", "Distinguish accuracy, cost and residual diagnostics",
     "A/B refers to the record index rule. Screening compares all runnable methods on the same subset. "
     "Whole-corpus evaluation uses a smaller model list. Relative speed stress during inference is distinct from "
     "speed error during angular resampling. Future independent files are handled by the freeze command.",
     """display(table[['display','mse','batch_median_s','batch_rtf','batch_iqr_s']])
diag = run/'diagnostics'
if (diag/'fault_full_group_summary.csv').exists():
    rows = pd.read_csv(diag/'fault_full_group_summary.csv')
    display(rows.query("metric == 'pmi_course'"))
if (diag/'rpm_stress_summary.csv').exists():
    display(pd.read_csv(diag/'rpm_stress_summary.csv'))
display(Image(filename=str(run/'figures/accuracy_cost.png'), width=720))
"""),
)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("outputs/expanded"))
    args=parser.parse_args()
    out=args.out.resolve()
    os.environ["GEARBOX_EXPANDED_RUN"]=out.relative_to(ROOT).as_posix()
    status=[]
    for filename,title,explanation,code in VIEWS:
        book=n.v4.new_notebook(cells=[n.v4.new_markdown_cell(f"# {title}\n\n{explanation}"),
                                   n.v4.new_code_cell(PRELUDE),n.v4.new_code_cell(code),
                                   n.v4.new_markdown_cell(
                                       "See the [model companion](../docs/Expanded_DHON_Companion.md) "
                                       "for exact configurations and interpretation.")],
                             metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}})
        start=perf_counter()
        NotebookClient(book, timeout=180, resources={"metadata":{"path":str(ROOT)}}).execute()
        n.write(book, ROOT/"notebooks"/f"{filename}.ipynb")
        status.append(dict(notebook=f"{filename}.ipynb",execution="PASS",seconds=perf_counter()-start))
    (out/"notebook_execution.json").write_text(json.dumps(status,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
