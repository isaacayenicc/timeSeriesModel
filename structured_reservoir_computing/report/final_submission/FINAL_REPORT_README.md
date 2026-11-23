# MECG 7780 Physics-Informed Structured Reservoir Computing

The study concerns causal one-step prediction of nonstationary gearbox vibration under varying input-shaft speed. It presents the physical feature design, grouped ridge readout, guarded training-only selection, comparison with 21 required model families, computational measurements, and residual tooth-crack evidence. The report uses 59 verified scholarly and course sources.

## Submission files

| File | Use |
| --- | --- |
| `MECG7780_PI_SRC_Final_Course_Report.docx` | Editable research report with native Word equations and tables, populated linked contents, and technical figures with alternative text |
| `MECG7780_PI_SRC_Final_Course_Report.pdf` | Fixed-layout inspection copy of the editable report |
| `report_source.md` | Complete citation- and LaTeX-bearing manuscript |
| `assets/` | Figures and evidence tables used to assemble the report |
| `evidence/` | Verified bibliography, literature search, provenance, rubric map, model registry, and results ledger |
| `qa/` | Numerical, scientific, layout, and delivery checks |
| `report_generation/` | Offline rebuild and verification scripts, instructions, and dependencies |
| `SHA256SUMS.txt` | File-level checksum manifest inside the ZIP |

The `outputs/MECG7780_PI_SRC_Graduate_Research_Report.docx` workspace copy contains the same document bytes as the canonical Word file. The document uses the verified course title and omits unverified student, instructor, and term details. Update the populated contents page numbers in Word after editing by selecting the document and updating fields. Rebuild the report to refresh the list after adding or renaming chapters.

## Rebuild from the report package

Extract the ZIP and run these commands from the resulting report root. Installation instructions and the optional audit against the original study are in [report_generation/README.md](report_generation/README.md).

```sh
python report_generation/build_report.py --output-dir rebuilt --pages
python report_generation/check_document.py --output-dir rebuilt
```

Files in the report package suffice to rebuild the editable Word and PDF files. The complete initial research distributions, raw MATLAB measurements, trained artifacts, and project environment remain in the original SRC project and its inputs. The companion package keeps the source terms and evidence boundaries documented. The original project README and Appendix C describe access to those materials. No fit is required to reproduce the report.
