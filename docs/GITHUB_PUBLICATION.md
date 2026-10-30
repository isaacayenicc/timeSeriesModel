# GitHub source publication

The connected `isaacayenicc/timeSeriesModel` GitHub repository was empty at review.
Environment publication had preserved the workspace but had not created Git
commits or published this project to GitHub. An empty `main` base enables review
of the project on a dedicated pull-request branch.

## Included

- Authored Python package, entry points, scripts and tests.
- Configurations, dependency declarations and observed environment versions.
- Executed educational notebook, its execution metadata and study companion.
- Documentation, scientific report, bibliography and source-qualified research.
- Model checkpoints, run configurations, histories, metrics and registries.
- Aggregate candidate-record features, audit metadata and the 17 final figures.
- Editable PowerPoint decks, speaker scripts and final PDF exports.
- Source/citation evidence metadata and concise author-source documentation.

## Retained locally and excluded from Git

- Raw MAT files, original supplied course documents and release ZIPs.
- Validation prediction NPZs and candidate trace NPZs, which contain supplied
  signal samples rather than just learned parameters or aggregate results.
- All per-record fault caches, including duplicate and pilot cache namespaces.
- Python/kernel/package caches, logs and rendered slide/contact-sheet images.
- Search HTML and full downloaded third-party implementation-source dumps.
- Credential-shaped local files such as `.env`, private keys and certificates.
- The old `checksums.sha256`, which describes a larger local release archive.

No excluded files are deleted. The raw-data boundary is especially relevant
because this repository is public and the supplied-data redistribution rights
are not established. The executed notebook retains its reviewed plots, tables
and educational outputs; complete source signals are not added as separate
data or prediction artifacts.

## Reproduce a fresh checkout

Install the declared dependencies and supply the original data as described in
[data placement](../data/README.md). Then run:

```bash
python reproduce_minimal.py --data-dir data/raw
python reproduce_all.py --level standard --data-dir data/raw --device cpu
python scripts/build_notebooks.py --execute
python -m pytest -q
python scripts/verify_release.py
```

Standard reproduction restores omitted validation prediction caches directly
from the frozen checkpoints, validating the original input hash and recorded
MSE first. It regenerates candidate traces/caches from the original fault parts
when these caches are absent; this requires the original full-study inference
cost but does not retrain reference models. Supply the raw training/validation
inputs before running data-dependent tests or executing notebook code.

The full local release contains additional derived arrays and execution evidence;
its archive manifest is distinct from this source-publication inventory.
`checksums-source.sha256` covers the published files except itself. The preserved
research access ledger records hashes/URLs of excluded source downloads, whose
snapshots remain in the local release. Historical reports describing local
archives and logs are execution evidence, not a claim that those excluded files
are present in a Git checkout.

## Publication validation — 2026-10-30

The reviewed source tree passes 68 tests, including a new missing-cache recovery
and wrong-input rejection check. The GRU causality assertion retains exact input
equality and uses an explicit float32 output tolerance to account for CPU-kernel
roundoff. This changes test portability, not model architecture or saved results.

A clean staged copy, using the locally supplied raw data only for validation,
also passed all 68 tests. It restored all 31 validation prediction caches from
frozen checkpoints, recovered ridge MSE 0.42005759033115847, passed artifact/link
verification and executed all 31 notebook code cells without errors. The public
executed notebook preserves the original study output; this additional replay
took approximately 9.68 seconds. Raw data, local logs and regenerated signal
caches were not added to the publication. Two credential scans found no exposed
tokens, private keys or credential-bearing URLs in the reviewed publication.
