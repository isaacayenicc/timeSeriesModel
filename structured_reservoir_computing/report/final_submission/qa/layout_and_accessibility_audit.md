# Layout and accessibility audit

## Final document

The Letter-format Word document has 1-inch side margins, 11-point serif body text at 1.17 spacing, styled headings through the third level, and numbered pages. The 53-page PDF comes from the delivered Word file via LibreOffice. The report was rendered to 53 PNGs at approximately 155 dpi. The PDF embeds its text and symbol fonts. Word may reflow pages where its fonts or print settings differ from those of the inspection environment.

| Review | Result and evidence |
| --- | --- |
| Front matter and literature, pp. 1–9 | Title, abstract, linked contents, terminology, and inline math readable. Table 1.1 repeats its header on page 5. |
| Data and EDA, pp. 10–15 | Checked rig and workflow label placement, plot axes, captions, overlaps table, and mathematical phase notation. Final workflow labels use actual line breaks. |
| Model and selection, pp. 16–25 | Checked architecture figure, inline and display subscripts, delimiters and numbering, ridge/constrained extensions, and table precision. Equation 5.11 fits the text block. All six decimals in Table 6.2 remain together. |
| Comparators and prediction, pp. 26–36 | Checked the model rubric summary, errors, residual graphics, ranked table, CPU table, and figure labels. Continuation tables repeat headers. The cost plot labels and leaders are separated. |
| Fault, robustness and synthesis, pp. 37–45 | Checked analytic-signal magnitude bars, PMI indices, group data, intervention and ablation plots, and limitations. Tables may continue onto the next page with a repeated header. |
| Bibliography and appendices, pp. 46–53 | Verified citation/DOI linking, complete editable comparison, measurement provenance, and local/published results tables. |
| Structural verification | `document_structure.json` records 226 native math objects, 21 numbered displays, 19 editable data tables, 27 complete comparison rows, and 16 live contents fields. `page_map.json` records the resolved page numbers. |
| Figures and accessible structure | All 17 figures have alternative text, nearby numbered captions, and at least 353 effective dpi. Ran `check_document.py` for headings, language, navigation, table headers, missing references, tracked changes, image and text bounds, and placeholders. |

The visual passes used rendered pages at reading scale and magnified crops for Greek symbols, custom delimiters, equation labels, closely spaced graph labels, and width-constrained table columns. Later render comparisons located changed pages for targeted reinspection. Corrections included proper OMML delimiter order, spacing in the rig and cost figures, the data-use diagram, and the optimization table grid. Heading and caption pagination uses document styles. Some multi-page tables and citation groups wrap at page or line boundaries.
