# Building and verifying the PI-SRC report

## Environment

The reference build used Python 3.12.14, Pandoc 3.1.11.1 with citeproc, and LibreOffice Writer 26.8.0.0.alpha0 with `soffice` on `PATH`. Create a separate Python environment if those packages are missing. The Word styles request Times New Roman, Arial, and Cambria Math. The inspected Linux PDF embeds Liberation Serif, Liberation Sans, and the math glyph fonts available to LibreOffice. Font or office-suite changes can affect pagination. The build recomputes the page map from the rendered document.

```sh
python -m venv .venv-report
. .venv-report/bin/activate
python -m pip install -r report_generation/requirements-report.txt
pandoc --version
soffice --version
```

Pandoc and LibreOffice are external system prerequisites. After the dependencies are installed, compilation reads bundled files and can run offline. The IEEE CSL style is included with its upstream metadata and copyright.

## Compile and check the bundled manuscript

Run from the extracted report root. The `--pages` flag writes a high-resolution PNG for every rendered page and may be omitted for a routine rebuild. The document check uses the source bundle's manifest and the resolved PDF page map.

```sh
python report_generation/build_report.py --output-dir rebuilt --pages
python report_generation/check_document.py --output-dir rebuilt
```

After changing chapter sources or the bundled extraction tables, run `python report_generation/assemble_manuscript.py` before the build. The build uses Pandoc for equations and scholarly citations, Python to apply Word formatting and live fields, and LibreOffice to render and verify the contents' page numbers. PDF font substitutions may lead to different wrapping in another office suite. The delivered tables, equations, and navigation remain editable in Word.

## Reconcile with the original study

With an existing checkout and the original measurements, use explicit paths. `--destination` places the resulting machine-readable evidence and audit in a new directory. `--replay-primary` performs bounded inference with the frozen checkpoint and requires the original measurements. The study dependencies needed by the checkpoint replay are specified in the original project. For a report-only rebuild, skip this operation.

```sh
python report_generation/verify_report_numbers.py \
  --project-root /path/to/structured_reservoir_computing \
  --raw-dir /path/to/original/mat/files \
  --reference-dir /path/to/original/references \
  --markdown report_source.md --replay-primary --destination rerun-evidence
```

`prepare_assets.py` rebuilds figure PNGs from an original checkout and the original MATLAB traces using `--project-root` and `--raw-dir`. Run it only to regenerate report assets. It writes into the report's `assets/` directory. `package_report.py` creates the report-only ZIP and manifests from the canonical bundle. An installed `sha256sum` can verify the archive with `sha256sum -c PI_SRC_Report_Submission_Package.zip.sha256` and the extracted contents with `sha256sum -c SHA256SUMS.txt`.
