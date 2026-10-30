# Updated reproducibility evidence

This execution resolves the initial fault-upload obstacle. Original train/validation inputs and learned model artifacts were retained and rescored; all 200candidate records were newly processed. Current evidence is distinguished from historical logs.

## Commands executed and outcomes

- Supplied splitter verifier plus independent fault_io verifier: PASS20 parts/200-records, matched file/logical hashes, ranges, fields and original record ordering. The original monolithic container bytes were unavailable; its declared fingerprint is not an independentbytecomparison. Evidence: fault_parts_verification.json and fault_parts_original_verification.log.
- Full frozen-model candidate computation:200-records,13 predictors plus raw,2,800 time-feature and8,400 angular rows. Totalrecord compute time809.301 s; median 4.028 s. Evidence: fault_full.log and fault_study_summary.json. A plottingcolumn/method collision was repaired after numerical computation; ASTcomparison confirms unchanged non-rendering code. Corrected rendering and finalization passed; numerical caches retained with documented provenance in fault_plot_repair.json.
- python -m pytest -q --junitxml=reports/pytest_results.xml:67 passed,0 failed,0 skipped, one external plugin deprecation warning. Tests include sourcepart corruption/order/missing-file rejection, exactFIR/linear-design prediction parity, recurrent record resets and angularPMIvariant checks.
- python reproduce_all.py --level standard --data-dir data/raw --device cpu:PASS. Rescored31 checkpoints, validatedsourceparts, reused matching200-record caches, regenerated tables/17figures/report/PDF/PPTX. This is cache/checkpoint regeneration, not a second independent full training. Evidence: standard_reproduction_updated.log.
- python scripts/build_notebooks.py --execute:PASS69 cells/31 code cells,31 executed,zeroerrors; measured10.23 s in finalrecordednotebookrun. It independentlyrefits ridge, verifies causality, checks all 200-record/tablecoverage and recomputes actual record 1modelresidual and all threeangularunit branches. Evidence: ../notebooks/execution_summary.json.
- Minimal CPU checkpoint command and extracted-release smoke are recorded after final archive verification below. Raw data are not included in the archive.

The67tests cover mathematical/causal/data invariants; they do not create physicalfault groundtruth. Prediction/context/seed results remain validation-only and selectionoptimistic. No sigtest/fault2 results, GPUtimes, Windows/macOS execution, liveMATLAB parity, verified health labels or published Spa LPV-ARMA reproduction is claimed.


Both updated PPTX decks exported successfully through LibreOffice/Poppler; all25slides were visually inspected and the finalreportPDF is searchable with13pages. Structural dimensions/notes/shape bounds passed. Filelinks/headinganchors/codefences passed automated checks. Microsoft PowerPoint and Windows/macOS execution remain unverified.


## Updated extracted-release verification — PASS

The updated ZIP was extracted into a new temporary directory. Only the original train/validation MAT files were copied into its raw folder; the packaged source and ridge checkpoint recovered validation MSE0.42005759033115847 with the documented minimal command. Data-independent release verification also passed against the retained complete200record tables,31run model artifacts,executed69cell notebook,17figure hashes,slide structures and local links. The large original faultparts were not duplicated into this smoke folder and the200record experiment was not falsely counted as independently recomputed there. Evidence: extracted_updated_release_smoke.json/.log. Final packaging verifies ZIP integrity and every retained file hash.

The release was refreshed with these verification logs and tightened notebook/figure checksum checks after the successful extracted smoke. No trained artifacts, numerical results or model/feature computations changed.
