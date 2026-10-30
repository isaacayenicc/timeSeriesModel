# Actual presentation and report rendering verification

Verified on the available Linux CPU environment on 2026-10-30.

## Commands executed successfully

```bash
python scripts/build_presentations.py
python scripts/render_presentations.py
```

The build source reads `reports/results_summary.csv`, `reports/data_audit.json`, available angular-sensitivity CSV and actual saved figure images. Missing required figures produce a failure rather than fabricated replacements. All text, tables, flow diagrams and speaker notes are editable PowerPoint objects; plots remain embedded, reproducibly generated images.

## Verified outcomes

- **Progress:** `Progress_Presentation_8min.pptx` opens with python-pptx and LibreOffice; 10 slides; 16:9; 480 seconds of allocated speaking time; speaker notes embedded and supplied in Markdown.
- **Final:** `Final_Presentation_12min.pptx` opens with python-pptx and LibreOffice; 15 slides; 16:9; 720 seconds of allocated speaking time; speaker notes embedded and supplied in Markdown.
- Automated checks verify expected slide counts, slide dimensions, nonempty notes and object boundaries inside the slide canvas. An initial overly tall body-text bounding box was corrected; the final check passes.
- LibreOffice successfully exported both decks to `exported_pdf/`; Poppler rasterized every page. The PDFs have exactly 10 and 15 pages, respectively, as independently opened by pypdf.
- Both contact sheets were visually inspected across **all 25 slides**. The final efficiency and ablation slides were additionally viewed at full raster resolution. No visible text clipping was found in those inspected exports. Figures, model comparisons, assumptions and blocker labels were readable; architecture alternatives are displayed as separate models rather than a spurious sequential pipeline.
- The final report's convenience PDF opens with pypdf, contains 10 pages after figure insertion, and exposes searchable text. The canonical, fully editable scientific report remains `reports/PROJECT_FINAL_REPORT.md`; PDF equations use readable plain-text conventions rather than complete LaTeX typesetting.

## Artifacts and reproducibility

`exported_pdf/Progress_Presentation_8min.pdf` and `exported_pdf/Final_Presentation_12min.pdf` are optional convenience exports. Their corresponding `*_contact_sheet.jpg` and `*_slides/slide-*.png` files document the rendered pages. Re-run the render script after changing plots, metrics or slides. LibreOffice printed Fontconfig cache warnings in this sandbox, but conversion completed successfully and raster pages were verified.

This establishes the tested LibreOffice rendering. Microsoft PowerPoint, native Windows/macOS execution, presenter delivery time and font-layout parity across other machines were not tested. Allocated speaker times are rehearsal targets, not measured human speaking durations. Fault-record analysis is explicitly labelled blocked in both decks; no physical fault classes or performance figures were invented.


## Updated full-fault delivery

The updated decks were regenerated from the complete200-record tables. Both exported successfully via LibreOffice and Poppler (10progress/15final pages); all25slides were visually inspected in contact sheets, with actual fault RMS/kurtosis, angular-assumption and record-distribution plots replacing the earlier blocked-data slides. All text/diagrams remain editable. The updated searchable report PDF has13pages. Microsoft PowerPoint/native Windows editing was not executed. See rendering_updated.log.
