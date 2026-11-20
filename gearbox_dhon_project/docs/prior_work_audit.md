# Prior-work audit

The shipped research project preserves the source attachments outside the deliverable. Source and member inventories sit in `outputs/runs/development/tables`. Each reproduction entry identifies whether it is an executed independent reimplementation, a checkpoint replay, a score reconstructed from archived forecast vectors, or a historical result.

- The [primary source and EDA audit](prior_work_eda.md) records paper/thesis references, original source conditions, independent EDA, and paper validation-phase mismatch.
- The [conventional and archive audit](baseline_archive_audit.md) records training and replay contents, coefficient convention and numerical differences, corrected controls, and excluded legacy artefacts.
- The [conventional study chapter](conventional_baselines.md) documents the implemented fits, convergence protocol and same-mask versus external comparisons.
- The [diagnostic audit](fault_diagnostics.md) records the instructor MATLAB script and its missing executable/helper dependencies.
