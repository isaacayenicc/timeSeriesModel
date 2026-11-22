# Clean-kernel notebook verification

The generation command executed all eleven notebooks in clean kernels against the completed guarded search and matched measured-data artifacts. It used Python 3.12.14, NumPy 2.5.5, nbclient 0.11.0, and one-thread BLAS. Run from the project checkout with the supplied data under the configured course input directory.

```bash
python scripts/build_notebooks.py --data-dir data/raw
```

The resulting notebooks contain 41 executed code cells and 19 embedded figures. The verifier read every notebook with nbformat, checked its saved SHA-256 against `results/notebook_execution.json`, and confirmed that each code cell has an execution count. No error cells, missing result-file messages, or absolute machine paths appear in notebook outputs. Kernel startup environment notices are recorded separately from notebook outcomes in `logs/notebook_execution.log`.

A separate clean-kernel run of notebooks 00, 05, 07 and 10 used a temporary empty data folder. All four completed from the saved evidence. Notebook 05 uses the measured input to repeat its waveform replay when available. Its synthetic fit and restore example also runs from the evidence-only configuration. See `logs/notebook_review_without_raw.log`.

| Notebook | Verification |
| --- | --- |
| 00 | Displays acquisition audit and shared prediction-mask identity. |
| 01 | Reads the training-only numerical EDA artifacts and retained/rejected feature hypotheses. |
| 02 | Verifies strict sparse-reference target count and a known lag-by-speed design. |
| 03 | Matches finite oscillator recurrence and direct sum, verifies future perturbation isolation, checks angular interpolation and compares a ridge fit with a dense oracle. |
| 04 | Independently recomputes the utility of every completed trial and displays selected settings, conditioning and Pareto evidence. |
| 05 | Verifies the primary checkpoint SHA-256 and measured forecast against archived arrays. A synthetic fit/save/reload provides a small executable teaching example. |
| 06 | Checks target, vibration, RPM and time equality across primary, baseline and archive bundles and independently recalculates all 27 reported MSEs. |
| 07 | Checks a known biased-envelope PMI of 7 and displays live provisional cohort and pulse results. |
| 08 | Recomputes matched correction MSE deltas and displays physics-off and sensor-stress evidence. |
| 09 | Recomputes dependent block-bootstrap intervals against the released table and displays measured training and inference evidence. |
| 10 | Rechecks primary checkpoint identity and explicitly records the absence of both independent inputs. |
