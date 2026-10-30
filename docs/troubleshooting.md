# Troubleshooting

**No MAT files:** copy your original sigtrain.mat and sigvalid.mat into data/raw. The release omits raw data. Do not download unrelated data and call it the supplied experiment.

**Fault file unavailable:** attachment transfer is capped at 32 MiB. Split the archive or copy the original directly into your local project's data/raw. The loader preserves structs and recording boundaries; never concatenate all fault signals into one record.

**MATLAB v7.3:** the supplied train/valid files are v5 family. HDF5 fallback is explicitly unsupported, not silently flattened; convert with a trusted tool or add a separately verified loader.

**Torch CPU install:** use the official CPU wheel command in README, then requirements and editable package. Keep TLS verification on. GPU is not required or benchmarked. Requirement versions match this execution, not a guarantee on every future operating system.

**No module rotmach:** run python -m pip install -e . from the checkout, or use python run_project.py audit. Notebook locates src relative to the project root.

**Kernel permission error:** notebook execution opens local kernel sockets. In this managed environment this may require the supported execution permission path; it is not a scientific algorithm failure. Never bypass a blocked remote destination or disable TLS.

**NaN at start of predictions:** warm-up entries before lag are intentionally undefined. Use common indices 256:24000 for official comparison. Undefined zero-energy diagnostic features also return NaN with a documented reason.

**Unexpected poor GRU/TCN MSE:** inspect histories and context/learning-rate budget, compare simple baselines first. Ten maximum epochs are a bounded experiment, not proof these architecture families can never improve. Do not discard an unfavorable measured result.

**PMI changes with assumed units:** rpm, revolutions/s and radians/s imply different cumulative cycles. No physical speed unit is verified here. Values are sensitivity results, not physical-fault proof; interpolation error is reported.

**Report/deck mismatch:** regenerate from the same saved results with reproduce_all.py --level standard. Do not hand-type headline numbers into slides.

**Academic sources blocked:** domain additions were saved to the environment draft, not activated. Review/save those settings and publish only when you want them active; source_access_log.md records what was actually read.
