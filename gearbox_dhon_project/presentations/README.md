# MECG 7780 presentations

- **Progress**. Eight timed slides for 8:00, followed by four discussion slides.
- **Final**. Thirteen timed slides for 12:00, followed by nine discussion slides.

The PowerPoint files contain editable text, tables, and architecture diagrams. The plots embed high-resolution images generated from the recorded experiment. Timing appears on each slide and in its presenter notes. The separately supplied Markdown notes contain the same timing and text.

Build both decks from the project root.

```sh
python scripts/build_slides.py --run-id development
```

Add `--render-check` to render the finished decks with LibreOffice, verify page counts and review PDF text extraction. Add `--previews` to retain the generated PDFs. The build also reloads both PowerPoint files and checks embedded notes, plan lengths and shape bounds. The talk durations are planned speaking allocations and allow time to point to axes, plots and diagrams.
