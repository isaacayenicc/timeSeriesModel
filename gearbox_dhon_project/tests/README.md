# Validation suite

Run `python -m pytest -q` from the project root. Fixtures cover target alignment and future perturbations, finite oscillator state parity, gradients and checkpoint round trips, fold support, LASSO solver parity, PMI indexing and angular diagnostics. Tests using synthetic inputs exercise known identities. Measured comparisons use the supplied MAT files. For that integration path also run the `replay` command and a separate `smoke` run as documented in the README.
