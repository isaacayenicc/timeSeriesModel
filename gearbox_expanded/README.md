# Expanded MECG 7780 gearbox comparison

## GitHub source edition

The sibling `../gearbox_dhon_project` provides the original study source. For exact replay, extract the privately delivered expanded project with its bundled prior directory, checkpoints and prediction arrays. The commands below describe that complete bundle. This public edition includes aggregate results and notebooks with cell outputs cleared. Install the sibling source with `python -m pip install -e ../gearbox_dhon_project -e .` to run the unit tests from this folder.

This is the second research run. The original DHON project remains in `prior/gearbox_dhon_project`. The expanded study compares all 21 requested model families with three DHON reference rows. The detailed model companion is `docs/Expanded_DHON_Companion.md`. A PDF, the complete comparison CSV, timing, paired intervals, score-aligned arrays, fitted model weights, and diagnostic results are in `outputs/expanded`. The original two course presentations are bundled with the prior study. The presentation addendum summarizes the expanded comparisons.

## Input and installation

Use Python 3.12 and the original course MAT files in `data/raw` or an explicit input directory. Raw files and source attachment archives are excluded from this bundle. The expected signal filenames and data hashes are in the bundled prior project and the expanded freeze. You can inspect the saved results without the input files. Fitting, checkpoint verification against the inputs, and diagnostics require the original records.

Linux or PowerShell inside a prepared virtual environment:

```sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip install -e prior/gearbox_dhon_project -e .
python -m pytest -q
```

The benchmark was executed on CPU. Use the standard CPU-compatible PyTorch installation instructions for your platform if the CPU wheel command has no matching wheel. Markdown generation uses Python and PDF regeneration uses Pandoc and XeLaTeX. The notebook generator also uses nbclient.

## Fit and verify

```sh
python -m gearbox_expanded.study fit --data-dir data/raw --out outputs/reproduction
python -m gearbox_expanded.study compare --data-dir data/raw --out outputs/reproduction
python -m gearbox_expanded.study diagnostics --data-dir data/raw --out outputs/reproduction
```

The `fit` stage can also be split into `statistical`, `tabular`, `neural`, and `attention` stages, using the same output path. The computation logs all held-block scores and preserves full-match checkpoint forecasts. The two sparse priors use stored support and a fresh coefficient fit. Reference archive models use bundled original-safe numeric weights. The final benchmark includes some relatively slow kernel, tree and attention predictors. Run one worker at a time when measuring an isolated CPU comparison.

To replay the shipped numeric results without refitting:

```sh
python scripts/verify_results.py --out outputs/expanded
python scripts/execute_notebooks.py --out outputs/expanded
python scripts/build_companion.py --out outputs/expanded
```

The verification script checks saved metric arithmetic and result coverage. The `compare` stage additionally predicts from the original input and checks each checkpoint across the actual development record. Use a copied evidence directory if regenerating measured timings while retaining the released artifacts.

## Frozen blind evaluation

```sh
python -m gearbox_expanded.blind --data-dir data/raw --out outputs/future_blind
```

The freeze verifies checkpoint and source identity before loading blind observations. Keep the original train and development data for alignment and overlap auditing. Blind fault diagnostics remain unlabelled in the absence of a confirmed label map. The existing release contains the missing-input preflight and no blind performance figures.
