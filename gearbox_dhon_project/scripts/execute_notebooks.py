"""Generate and execute the nine teaching notebooks in clean kernels."""
import argparse
import csv
import os
from pathlib import Path
from time import perf_counter

import nbformat as nbf
from nbclient import NotebookClient

NAMES = [
    "Project_Overview_and_Data_Audit", "Physics_Informed_EDA", "Spa_LPV_ARMA_Reproduction",
    "DHON_Mathematical_Development", "DHON_Joint_Physics_and_Model_Optimization",
    "Comparative_Model_Performance", "Fault_Detection_and_Robustness",
    "Ablation_and_Computational_Efficiency", "Independent_Test_and_Final_Synthesis",
]

CODE = [
"""audit = json.loads((run / 'data_availability.json').read_text())
print(json.dumps(audit, indent=2))
show_files(run, ('audit', 'inventory'), limit=12)
""",
"""from matplotlib import pyplot as plt
from scipy import signal
# Frequencies from a known order and two prior RPM observations.
known_rpm = np.array([125., 165.])
print('Mesh frequency at two speeds:', 37 * known_rpm / 60, 'Hz')
if data_dir is not None:
    rec = load_record(find_input(data_dir, 'sigtrain'))
    freq, power = signal.welch(rec.x, fs=rec.fs, nperseg=2048)
    print('Peak Welch bin (Hz):', freq[np.argmax(power)])
show_figures(('spectrogram', 'order', 'speed_bin', 'ridge'), limit=5)
display(read('tables/eda_guarded_feature_screen.csv').head(16))
show_files(run / 'eda', ('registry', 'summary'), limit=4)
""",
"""from gearbox_dhon.baselines import speed_basis
# Rows are time samples, columns are basis functions; lag-major tensor
# product multiplies each past vibration sample by all basis columns.
print(speed_basis(np.array([0., 0.5, 1.]), 3, 'legendre'))
read('metrics/summary.csv').query("model.str.contains('lpv|arima|ari')", engine='python').pipe(display)
display(read('metrics/published_external.csv'))
show_files(run, ('aic', 'cv', 'convergence'), limit=7)
show_figures(('aic', 'lambda'), limit=4)
""",
"""import torch
from gearbox_dhon.dhon import oscillator_states, oscillator_recurrence
# A single order-locked carrier at precisely 150 rpm.
u = np.arange(300) * 150.0 / (60 * 3200)
phase = 2 * np.pi * 37 * u
x = np.cos(phase)
tau = torch.tensor([[24.]], dtype=torch.float64, requires_grad=True)
phase_t = torch.tensor(phase[None, :], dtype=torch.float64)
z = oscillator_states(torch.tensor(x), phase_t, tau, 64)
objective = z.abs().square().mean()
objective.backward()
print('State shape:', tuple(z.shape), 'decay gradient:', tau.grad)
show_files(run, ('oscillator', 'gradient', 'learned'), limit=5)
""",
"""# Show the full finite search source path and preview the actual scores.
show_files(run / 'search', ('trial', 'fold', 'history', 'config', 'selection'), limit=10)
show_figures(('search', 'pareto', 'trial', 'learn'), limit=5)
show_files(run / 'checkpoints', ('config', 'score'), limit=6)
""",
"""from gearbox_dhon.results import read_predictions, scores
truth = np.load(run / 'predictions' / 'development_targets.npz')
target = truth['target']
start = int(truth['zero_based_indices'][0])
predictions = read_predictions(run, len(target) + start)
computed = pd.DataFrame([dict(model=key, **scores(target, pred[start:]))
                        for key, pred in predictions.items()])
verify = computed.merge(read('metrics/summary.csv'), on='model', suffixes=('_new','_saved'))
np.testing.assert_allclose(verify.mse_new, verify.mse_saved, atol=1e-12, rtol=0)
display(computed.sort_values('mse'))
display(read('metrics/paired_block_bootstrap.csv').head(12))
show_figures(('mseloss', 'trace', 'residual_structure'), limit=3)
""",
"""from gearbox_dhon.faults import diagnostics, pmi_from_envelope
# Constant envelopes carry no centered variance, hence PMI is undefined.
print('Constant-envelope fixture:', pmi_from_envelope(np.ones(1024), 64))
trace_file = run / 'predictions' / 'fault_representative_traces.npz'
if trace_file.exists():
    arrays = np.load(trace_file)
    measured, _ = diagnostics(arrays['100:raw:residual'], arrays['100:rpm'], arrays['100:t'])
    table = read('metrics/fault_per_record.csv')
    expected = table.query("record_id == 100 and model == 'raw'").pmi_course.iloc[0]
    np.testing.assert_allclose(measured['pmi_course'], expected, atol=2e-8, rtol=0)
    print('Record 100 recomputed PMI:', measured['pmi_course'])
show_files(run / 'metrics', ('fault', 'pmi', 'pulse', 'stress'), limit=9)
show_figures(('fault', 'pmi', 'pulse', 'stress'), limit=5)
""",
"""display(read('metrics/timings.csv'))
show_files(run, ('ablation', 'stress', 'timing_environment', 'pulse'), limit=9)
show_figures(('timing', 'pareto', 'ablation'), limit=3)
""",
"""manifest = json.loads((run / 'frozen_manifest.json').read_text())
print('Frozen models:', ', '.join(manifest['models']))
print('Prediction mask starts at:', manifest['scoring_start_zero_based'])
print('Threshold policy:', manifest['diagnostic_policy'])
for path in sorted(run.parent.glob('*/data_availability.json')):
    if path.parent != run:
        print(path.parent.name, json.loads(path.read_text()))
print('For future independent evaluation see the hash-checked command in README.md.')
""",
]

EXPLANATIONS = [
("Data and experimental design", "Identify the source of each result, interpret one-based record IDs versus zero-based sample offsets, and explain why selective record exclusion is documented before diagnostic evaluation.",
 "The designated training and development validation files provide a single short time series each. Fault records 61 and 63 duplicate the beginning of the two designated records. They are retained in audit tables and excluded from group summaries. Group A/B refers to record index only."),
("Gearbox physics and exploratory analysis", "Interpret mesh trajectories, speed-locked orders and fixed-Hz peaks. Connect each spectrum to a candidate model feature or a test.",
 "A tooth count of 37 gives mesh frequency $37 n/60$ Hz at shaft speed $n$ RPM. The frequency ridge moves with speed while the corresponding order stays at 37. Shaft phase is the integral of speed in revolutions. Whole-record STFT and order plots are descriptive. The predictor reconstructs angular history from known past samples only."),
("Sparse LPV-ARMA and paper reproduction", "Distinguish the published experiment, coefficient replay, strict-causal refits, and currently observed validation scores.",
 "Each AR lag is multiplied by an eight-function basis in normalized speed. Stage II adds lagged innovations from a stage-I proxy model. The innovation at a previous time uses the already observed target. Penalized fitting selects support, then a least-squares refit removes shrinkage on retained terms. The paper's validation acquisition phase differs from the course segment."),
("The differentiable harmonic oscillator", "Derive the finite history state and inspect gradients of the actual state computation.",
 r"For $Z=sum_{j=1}^{M}\rho^{j-1}$ and $\rho=e^{-1/\tau}$, the quadrature is $h_t = 2 Z^{-1}sum_{j=1}^{M}\rho^{j-1}x_{t-j}e^{i(phi_t-phi_{t-j})}$. Target phase uses RPM through $t-1$. A phase offset cancels; damping in this model is a statistical memory constant. Training differentiates through the exponential weights and carrier phase."),
("Physics and model optimization", "Explain the finite search budget, train-only guarded validation, and why configurations are compared on the same support.",
 "Three held-out blocks inside the training record test each configuration. Both boundaries are guarded by 1,792 samples, and independent segments reset their dynamics. The final development record never enters candidate selection. Inspect search tables to distinguish alternatives, controls, and the actual selected configuration."),
("Matched prediction comparison", "Recompute scores from portable arrays, interpret block-bootstrap comparisons, and separate the external publication.",
 "For all model vectors the target mask starts at index 1,792 and ends before 24,000. MSE is the average squared standardized residual. The paired bootstrap samples contiguous 640-point sequences from differences of squared errors. It describes stability within the supplied record and cannot estimate acquisition-to-acquisition uncertainty."),
("Angular residual diagnostics", "Explain the center-and-correlate PMI definition, MATLAB indexing, overlap exclusions and diagnostic status.",
 "The completed residual segment is interpolated on a 2,048-samples/revolution grid. The literal MATLAB index corresponds to lag 2,047, and the corrected lag is 2,048. Both use a centered envelope autocorrelation and the transformation $r/(1-r)$. Index-group separation is descriptive evidence until authoritative condition labels and acquisition identities are supplied."),
("Ablation, uncertainty and cost", "Read matched-adapter and oscillator controls and distinguish batch throughput from a verified streaming deadline.",
 "Fit separate fixed/learned physical variants when estimating the value of learning. A slow causal adapter uses only clipped earlier base-model residuals. Repeated CPU timings include feature construction and inference; RTF divides compute time by record duration. A low batch RTF leaves per-sample streaming latency untested."),
("Freeze and independent evaluation", "Verify which artifacts are frozen, describe the blind entry point, and distinguish missing files from failed evaluation.",
 "The freeze records data and code hashes, checkpoint hashes, score bounds and the diagnostic convention. The blind command loads only frozen predictors and checks exact numerical overlap. Actual acquisition identity requires separate metadata. A missing test file creates an explicit status record rather than an empty score presented as evidence."),
]

PRELUDE = """import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import Image, display
from gearbox_dhon.data import find_input, load_record

root = Path.cwd()
if root.name == 'notebooks':
    root = root.parent
run_id = os.getenv('GEARBOX_RUN_ID', 'development')
run = root / 'outputs' / 'runs' / run_id
candidate = Path(os.getenv('GEARBOX_DATA_DIR', str(root / 'data' / 'raw')))
data_dir = candidate if candidate.is_dir() and find_input(candidate, 'sigtrain') else None

def read(relative):
    path = run / relative
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

def show_files(directory, patterns, limit):
    candidates = [p for p in sorted(directory.rglob('*'))
                  if p.suffix in {'.csv', '.json'} and any(k in p.stem.lower() for k in patterns)]
    for path in candidates[:limit]:
        print(path.relative_to(root))
        if path.suffix == '.csv':
            display(pd.read_csv(path).head(8))
        else:
            value = json.loads(path.read_text())
            print(json.dumps(value, indent=2)[:3600])

def show_figures(patterns, limit):
    candidates = [p for p in sorted((run / 'figures').rglob('*.png'))
                  if any(k in p.stem.lower() for k in patterns)]
    for path in candidates[:limit]:
        print(path.relative_to(root))
        display(Image(filename=str(path), width=690))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="development")
    parser.add_argument("--mode", choices=["replay", "full"], default="replay")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.mode == "full":
        from gearbox_dhon.cli import run_pipeline
        run_pipeline(root, args.data_dir.resolve(), args.run_id, "full", "auto", resume=True)
    os.environ["GEARBOX_RUN_ID"] = args.run_id
    os.environ["GEARBOX_DATA_DIR"] = str(args.data_dir.resolve())
    status = []
    for i, (name, narrative, code) in enumerate(zip(NAMES, EXPLANATIONS, CODE)):
        title, objective, content = narrative
        chapters = {2: "conventional_baselines.md", 3: "mathematical_formulation.md",
                    4: "notebook_04_dhon_search.md", 5: "model_comparisons.md", 6: "fault_diagnostics.md",
                    7: "dhon_results_interpretation.md"}
        note_name = root / "docs" / chapters.get(i, f"notebook_{i:02d}.md")
        details = note_name.read_text(encoding="utf-8") if note_name.exists() else ""
        notebook = nbf.v4.new_notebook(
            cells=[nbf.v4.new_markdown_cell(
                f"# {i:02d} {title}\n\n## Learning outcome\n\n{objective}\n\n"
                f"## Method and notation\n\n{content}\n\n"
                "Prerequisites are listed in the project README. This notebook reads the configured run directory. "
                "The tables and measurements below come from the recorded run or are recomputed in Python. "
                "Use `scripts/execute_notebooks.py --mode full --run-id NEW --data-dir PATH` to regenerate the shared pipeline."),
                   nbf.v4.new_code_cell(PRELUDE),
                   nbf.v4.new_markdown_cell(details or "## Recorded experiment\n\nInspect the measurements below against the stated information cutoff and split."),
                   nbf.v4.new_code_cell(code),
                   nbf.v4.new_markdown_cell(
                       "## Interpretation\n\nReview data scope and limitations in the [course report](../docs/final_course_report.md) "
                       "and [companion guide](../docs/companion_guide.md). The executed values are attributable to the named run "
                       "and the exact input and checkpoint hashes in its manifest.")],
            metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
        name = f"{i:02d}_{name}.ipynb"
        tic = perf_counter()
        result, error = "NOT RUN", ""
        if not args.generate_only:
            try:
                NotebookClient(notebook, timeout=180, resources={"metadata": {"path": str(root)}}).execute()
                result = "PASS"
            except Exception as exc:
                result, error = "ERROR", str(exc)
                for cell in notebook.cells:
                    if cell.cell_type == "code":
                        cell.outputs = [o for o in cell.outputs if o.output_type != "error"]
        nbf.write(notebook, root / "notebooks" / name)
        status.append(dict(notebook=name, mode=args.mode, status=result, seconds=perf_counter()-tic, error=error))
    output = root / "outputs" / "runs" / args.run_id / "notebook_execution.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(status[0]))
        writer.writeheader()
        writer.writerows(status)
    if any(row["status"] == "ERROR" for row in status):
        raise SystemExit(f"Notebook execution failures: {output}")


if __name__ == "__main__":
    main()
