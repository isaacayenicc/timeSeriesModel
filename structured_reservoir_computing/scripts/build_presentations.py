"""Build editable, timed SRC PowerPoint decks from frozen development evidence."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

from report_evidence import Evidence, ROOT, fmt, model_key, name, number

W, H = 13.33333, 7.5
PAPER, WHITE = "F8FAFB", "FFFFFF"
INK, NAVY, TEAL, MUTED = "21313E", "153654", "086660", "466171"
PALE, STROKE = "E7F2F3", "CBDCE2"
FONT = "Liberation Sans"


def clock(seconds):
    return f"{seconds // 60}:{seconds % 60:02d}"


def color(value):
    return RGBColor.from_string(value)


def box(slide, x, y, w, h, fill=WHITE, outline=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    style = shape.element.find("{http://schemas.openxmlformats.org/presentationml/2006/main}style")
    if style is not None:
        shape.element.remove(style)
    shape.fill.solid()
    shape.fill.fore_color.rgb = color(fill)
    if outline:
        shape.line.color.rgb = color(outline)
        shape.line.width = Pt(0.7)
    else:
        shape.line.fill.background()
    return shape


def text(slide, value, x, y, w, h, size=21, ink=INK, bold=False, align=PP_ALIGN.LEFT):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    f = shape.text_frame
    f.clear()
    f.word_wrap = True
    f.margin_left = f.margin_right = f.margin_top = f.margin_bottom = 0
    for i, line in enumerate(str(value).split("\n")):
        p = f.paragraphs[0] if i == 0 else f.add_paragraph()
        p.text = line
        p.font.name = FONT
        p.font.size = Pt(size)
        p.font.bold = bold
        p.font.color.rgb = color(ink)
        p.space_after = Pt(2)
        p.alignment = align
    return shape


def panel(slide, title, body, x, y, w, h, accent=TEAL):
    box(slide, x, y, w, h, WHITE, STROKE)
    box(slide, x, y, 0.065, h, accent)
    text(slide, title, x + 0.22, y + 0.25, w - 0.44, 0.8, 23, accent, True)
    text(slide, body, x + 0.22, y + 1.1, w - 0.44, h - 1.22, 20)


def panels(slide, items):
    width = (12.02 - (len(items) - 1) * 0.25) / len(items)
    for i, (title, body) in enumerate(items):
        panel(slide, title, body, 0.65 + i * (width + 0.25), 2.04, width, 3.92)


def takeaway(slide, label):
    box(slide, 0.66, 6.25, 12.01, 0.57, PALE)
    text(slide, label, 0.86, 6.37, 11.65, 0.34, 17, TEAL, True)


def fig(slide, path, x, y, w, h):
    if path is None:
        box(slide, x, y, w, h, PALE, STROKE)
        text(slide, "Plot unavailable in the saved run", x + 0.1, y + h / 2 - 0.18,
             w - 0.2, 0.45, 18, MUTED, align=PP_ALIGN.CENTER)
        return
    with Image.open(path) as im:
        ratio = im.width / im.height
    true_w = min(w, h * ratio)
    true_h = true_w / ratio
    slide.shapes.add_picture(str(path), Inches(x + (w - true_w) / 2), Inches(y + (h - true_h) / 2),
                             width=Inches(true_w), height=Inches(true_h))


def figure_panel(slide, path, heading, body, caption):
    fig(slide, path, 0.64, 1.7, 7.56, 4.86)
    panel(slide, heading, body, 8.39, 1.95, 4.29, 4.23)
    text(slide, caption, 0.65, 6.61, 12.0, 0.34, 16, MUTED)


def table(slide, headers, rows, widths, top=2.04, height=0.43):
    native = slide.shapes.add_table(len(rows) + 1, len(headers), Inches(0.65), Inches(top),
                                    Inches(sum(widths)), Inches(height * (len(rows) + 1))).table
    for column, w in zip(native.columns, widths, strict=True):
        column.width = Inches(w)
    for i, row in enumerate([headers, *rows]):
        native.rows[i].height = Inches(height)
        for j, value in enumerate(row):
            cell = native.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.10)
            cell.margin_top = cell.margin_bottom = Inches(0.025)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = color(PALE if i == 0 else WHITE if i % 2 else PAPER)
            cell.text = str(value)
            p = cell.text_frame.paragraphs[0]
            p.font.name = FONT
            p.font.size = Pt(18)
            p.font.bold = i == 0
            p.font.color.rgb = color(NAVY if i == 0 else INK)


def architecture(slide, include_equation=True):
    blocks = (
        ("Prior samples", "Vibration, RPM and causal speed changes"),
        ("Reservoir state", "Sample and angular lags, oscillator quadratures, innovations"),
        ("Structured library", "Selected interactions and group-scaled regularization"),
        ("Forecast", "Fitted readout and optional historical-error correction"),
    )
    for i, (heading, body) in enumerate(blocks):
        x = 0.65 + i * 3.07
        box(slide, x, 2.19, 2.80, 2.85, PALE if i == 1 else WHITE, STROKE)
        text(slide, heading, x + 0.15, 2.40, 2.50, 0.71, 21, TEAL, True)
        text(slide, body, x + 0.15, 3.31, 2.50, 1.51, 19)
        if i < 3:
            text(slide, "→", x + 2.80, 3.23, 0.27, 0.35, 16, TEAL, True, PP_ALIGN.CENTER)
    if include_equation:
        text(slide, "x̂(t) = b + βᵀ Φ(z(t)) + g m(t)\nhₖ(t) ∝ Σ wⱼ x(t−j) exp[i(φₖ(t)−φₖ(t−j))]",
             0.67, 5.28, 11.97, 0.79, 20, NAVY, True)


class Deck:
    def __init__(self, kind, budget, stem):
        self.kind, self.budget, self.stem = kind, budget, stem
        self.prs = Presentation()
        self.prs.slide_width = Inches(W)
        self.prs.slide_height = Inches(H)
        self.elapsed = 0
        self.plan = []

    def slide(self, section, heading, seconds, notes):
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = color(PAPER)
        window = f"{clock(self.elapsed)}–{clock(self.elapsed + seconds)}" if seconds else "Backup"
        self.elapsed += seconds
        self.plan.append((heading, window, seconds, notes))
        text(slide, section.upper(), 0.65, 0.31, 12, 0.28, 13, TEAL, True)
        text(slide, heading, 0.65, 0.74, 12.03, 0.75, 33, NAVY, True)
        box(slide, 0.65, 1.54, 12.03, 0.015, STROKE)
        box(slide, 0.65, 7.01, 12.03, 0.015, STROKE)
        text(slide, f"MECG 7780  /  {self.kind}", 0.65, 7.13, 5.5, 0.25, 11, MUTED)
        text(slide, window, 6.03, 7.13, 1.28, 0.25, 11, MUTED, align=PP_ALIGN.CENTER)
        text(slide, f"{len(self.plan):02d}", 12.05, 7.1, 0.63, 0.28, 13, TEAL,
             True, PP_ALIGN.RIGHT)
        slide.notes_slide.notes_text_frame.text = f"{window} • {clock(seconds)}\n\n{notes}"
        return slide

    def save(self, directory):
        if self.elapsed != self.budget:
            raise ValueError(f"{self.kind} requires {clock(self.budget)}")
        path = directory / f"{self.stem}.pptx"
        self.prs.core_properties.title = f"MECG 7780 Structured Reservoir Computing | {self.kind}"
        self.prs.core_properties.subject = "Causal gearbox vibration prediction and residual diagnostics"
        self.prs.core_properties.author = ""
        self.prs.save(path)
        reread = Presentation(path)
        if len(reread.slides) != len(self.plan):
            raise RuntimeError(f"Unexpected slide count in {path}")
        notes_md = [f"# SRC {self.kind} speaker notes", "",
                    f"Planned speaking duration is **{clock(self.budget)}**. Backup slides are untimed.", ""]
        for i, (slide, (heading, window, seconds, notes)) in enumerate(zip(reread.slides, self.plan, strict=True), 1):
            if window not in slide.notes_slide.notes_text_frame.text:
                raise RuntimeError(f"Missing notes on slide {i} of {path}")
            for shape in slide.shapes:
                if shape.left < 0 or shape.top < 0 or shape.left + shape.width > self.prs.slide_width + 8 or (
                        shape.top + shape.height > self.prs.slide_height + 8):
                    raise RuntimeError(f"Out-of-bounds shape on slide {i} of {path}")
            notes_md.extend((f"## {i}. {heading}", "", f"**{window}**", "", notes, ""))
        (directory / f"{self.stem}_Speaker_Notes.md").write_text("\n".join(notes_md), encoding="utf-8")
        return path


def speed_plot(e):
    return e.plot("eda/vibration_speed.png", "eda/time_speed.png", "eda/speed_vibration.png", "eda/eda_speed.png")


def spectrogram(e):
    return e.plot("eda/spectrogram.png", "eda/mesh_spectrogram.png", "eda/eda_spectrogram.png")


def order_plot(e):
    return e.plot("eda/order_spectrum.png", "eda/order_tracking.png", "eda/eda_orders.png")


def title_panel(slide, line):
    box(slide, 0.66, 1.87, 12.0, 3.73, PALE)
    text(slide, "Physics-Informed Structured\nReservoir Computing", 1.02, 2.23, 11.32, 1.47, 39, NAVY, True)
    text(slide, line, 1.05, 4.12, 10.99, 0.97, 24, TEAL)


def progress(e):
    d = Deck("Progress", 480, "SRC_Progress_Presentation_8min")
    s = d.slide("Retrospective progress review", "Predicting gearbox vibration at changing speed", 45,
        "This is a retrospective reconstruction of the progress stage from the completed project. Imagine following a gear through its speed cycle. A shaft-synchronous component changes frequency in time, even when it repeats consistently in shaft angle. The practical objective is to predict the next healthy-reference vibration sample. We then inspect the remaining signal for evidence that repeats with shaft angle. This deck explains the initial study design and the representative evidence available in the released run.")
    title_panel(s, "Causal one-step prediction and residual periodicity under variable operating speed")
    takeaway(s, "Retrospective progress deck • Course presentation plan")

    s = d.slide("Engineering objectives", "What should a better model improve?", 45,
        "We compare mean squared one-step error against the course reference on fixed target indices. We measure actual CPU batch time and report GPU availability separately. The diagnostic question concerns whether a repetitive disturbance remains visible after subtracting the predicted waveform. A lower MSE can coexist with lower pulse preservation. This distinction gives us a way to compare model structure and diagnostic utility.")
    panels(s, [("Prediction", "Compare standardized MSE on common causal targets."),
               ("Computation", "Measure complete inference on the same record and hardware."),
               ("Residual diagnosis", "Check angular periodicity and preservation of injected pulses.")])

    s = d.slide("Acquisition and integrity", "Use the official training and validation records", 60,
        "The supplied prediction records each have twenty-four thousand samples at 3,200 hertz. Keep the supplied standardization and use training-only transforms for new features. Treat record grouping as an acquisition question. Exact joint vibration and speed checks can reveal shared signals across the candidate and official files. Verify labels separately because an index group cannot establish a verified condition. The final blind test and fault files are unavailable in this release.")
    figure_panel(s, speed_plot(e), "Original course traces",
                 "24,000 samples per prediction record at 3,200 Hz. Preserve normalized vibration and retain measured RPM.",
                 "Actual training trace • Vibration and rotational speed")

    s = d.slide("Training exploratory analysis", "Follow mesh harmonics in a shaft-angle frame", 70,
        "The monitored shaft's 37-tooth gear provides a nominal mesh order of thirty-seven. RPM determines instantaneous hertz and makes mesh ridges drift in a time-frequency display. Angular resampling offers a complementary view in cycles per turn. Energy near the nominal harmonics motivates order-locked features. Broad fixed-hertz content can come from several mechanisms, so it needs separate controlled testing. Whole-record spectra are explanatory observations. Prediction features use only measurements available through the previous sample.")
    figure_panel(s, spectrogram(e), "Order × revolutions per second",
                 "The order-148 ridge differs by 0.744 Hz from its theoretical track. Test order-locked and fixed-Hz candidates separately.",
                 "Training spectrogram • Measured signal and physically derived mesh trajectories")

    s = d.slide("Model principle", "Build a structured reservoir of past observations", 65,
        "The reservoir is a deliberately structured expansion of measured history. It includes raw vibration lags, equal-angle histories, finite damped harmonic states and prior innovations. A learned readout maps these coordinates into the next sample. The optional clipped historical-error correction updates only after a target arrives. The profile can select or disable a feature family within the guarded training folds. The oscillator state summarizes previous measurements at their relative phase.")
    architecture(s)
    takeaway(s, "Every predictor input ends at the preceding sample")

    s = d.slide("Original contribution", "Constrain the feature library before fitting", 60,
        "Each group has an engineering interpretation. Sample lags model short material and structural responses. Angular lags compare points at similar tooth phase. Oscillator states retain coherent energy with a finite decay, and innovation lags can capture serial error structure. A restricted interaction library tests useful coupling with speed and amplitude. Group-scaled regularization limits the effect of weak, correlated inputs. The archive and direct regression models provide useful controls for this contribution.")
    panels(s, [("Physics groups", "Use phase-locked oscillators and bounded history."),
               ("Restricted interactions", "Expand selected connections among measured predictors."),
               ("Controls", "Compare linear structure, oscillator groups and matched correction.")])

    s = d.slide("Training-only selection", "Guard the held-out time blocks", 55,
        "Each held-out block is contiguous. A gap on both sides protects raw histories used for fit. Rank candidates only on training blocks using the predeclared combination of clean error, RPM stress error and synthetic pulse distortion. The supplied validation record is reserved for the selected models. Keep every benchmark on targets 1,792 through 23,999 and apply the same strict prior-input convention where eligible.")
    panels(s, [("Fit", "Guard both sides of each held-out training interval."),
               ("Select", "Compare clean error, RPM stress and pulse distortion."),
               ("Score", "Evaluate official development on the common 22,208 targets.")])

    s = d.slide("Measured checkpoint", "Use a verified comparison to frame the next test", 45,
        e.comparison_text() + " The complete report distinguishes newly fitted methods, checkpoint replays and the external publication’s independent segment. At the progress stage, this is representative evidence for the method and the validation rules.")
    shortlist = [r for r in (e.primary, e.sparse_reference) if r]
    if shortlist:
        table(s, ["Method", "Development MSE", "Scored targets"],
              [(name(r), fmt(r, "mse"), fmt(r, "n", digits=0)) for r in shortlist],
              [6.1, 3.3, 2.6], height=0.63)
    takeaway(s, "Development MSE is in squared standardized vibration units")

    s = d.slide("Open evidence", "Prepare for an independent test", 35,
        "Candidate-record scores describe index groups while condition labels remain unverified. The reported computational measurements concern available batch CPU inference. The frozen manifest preserves model and evaluation identity for a future independent test. Continue by confirming condition labels, acquisition independence and the intended streaming hardware.")
    panels(s, [("Record labels", "Verify condition and original acquisition identity."),
               ("Independent record", "Keep the final model frozen for later test files."),
               ("Timing scope", "Measure streaming latency on deployment hardware.")])
    technical_backup(d, e)
    return d


def technical_backup(d, e):
    s = d.slide("Backup / physical evidence", "Order spectra align rotation-locked energy", 0,
        "Order is cycles per reference revolution. A spectral peak helps motivate a reservoir group. Guarded prediction and matched ablation test its incremental value.")
    figure_panel(s, order_plot(e), "Angle-domain evidence", "Inspect mesh multiples and nearby bands against the time-frequency spectrum.",
                 "Training order spectrum • Analysis does not supply future observations to the model")
    s = d.slide("Backup / reference", "Publication and project comparisons", 0,
        "The course paper and the supplied development record represent different evaluation segments. Report the published result as external context and compare methods on the frozen local scoring mask.")
    panels(s, [("Published", "Chen et al., Table 6, reports 0.2893 on the paper segment."),
               ("Project", "Course development uses the same 22,208 targets across eligible methods."),
               ("Source", "Mechanical Systems and Signal Processing, volume 224, article 112161.")])
    takeaway(s, "doi.org/10.1016/j.ymssp.2024.112161")


def primary_artifact_row(rows, e):
    key = model_key(e.primary)
    return next((row for row in rows if model_key(row) == key and key), {})


def diagnostic_text(e):
    primary = next((row for row in e.fault if model_key(row) == model_key(e.primary)
                    and row.get("metric") == "lecturer_compatible"), {})
    auc = number(primary, "conditional_auc_B_high", "conditional_auc", "provisional_auc", "auc")
    gap = number(primary, "median_gap_B_minus_A", "median_gap", "pmi_median_gap")
    if auc is not None and gap is not None:
        return f"Median PMI gap B−A was {gap:.3f}. Conditional B-high AUC was {auc:.3f}."
    if e.fault:
        return "Read the measured residual PMI distributions as provisional index-group separation."
    return "The candidate-record group statistic is unavailable in the saved run."


def selection_text(e):
    selected = e.selection
    trial = str(selected.get("trial_id", ""))
    fold = number(selected, "fold_mse")
    utility = number(selected, "utility")
    if fold is None:
        return "Saved model selection metadata is unavailable."
    item = f"Mean guarded MSE {fold:.4f}"
    if utility is not None:
        item += f". Selection objective {utility:.4f}"
    if trial:
        item += f". {trial}"
    return item + "."


def ablation_text(e):
    if not e.ablations:
        return "No ablation rows are available."
    off = next((row for row in e.ablations if model_key(row) == "pi-src-adapter-off"), {})
    base, adjusted = number(off, "mse"), number(e.primary, "mse")
    if base is not None and adjusted is not None:
        return f"Same weights: {base:.6f} without correction and {adjusted:.6f} with correction."
    ranked = [(r, number(r, "mse", "validation_mse", "clean_mse")) for r in e.ablations]
    ranked = [(r, v) for r, v in ranked if v is not None]
    if not ranked:
        return "Measured configurations are listed in the ablation ledger."
    low, high = min(v for _, v in ranked), max(v for _, v in ranked)
    return f"Measured ablation MSE range {low:.4f}–{high:.4f}. Compare controls under the same correction."


def final(e):
    d = Deck("Final", 720, "SRC_Final_Presentation_12min")
    s = d.slide("MECG 7780 / Final research presentation", "Physics-informed prediction at variable speed", 45,
        e.comparison_text() + " The contribution is a finite, causal reservoir built from samples, angle, phase-transported harmonic states and prior errors. A regularized readout predicts the next measured sample. The report also tests which features help, how much signal remains for diagnosis, and what the completed timing evidence can support.")
    title_panel(s, "A finite speed-conditioned reservoir with a trained readout")
    result = fmt(e.primary, "mse")
    takeaway(s, f"PI-SRC development MSE {result} • 22,208 matched targets • Residual diagnosis")

    s = d.slide("Research gap", "A mesh order moves in frequency as speed changes", 45,
        "A fixed time lag represents different shaft rotations as operating speed changes. A finite oscillator bank can track the same order across those changes. The project tests whether this structure adds value beyond regularized statistical models and broad data-driven predictors. A second concern is whether prediction and residual correction preserve repetitive disturbances.")
    panels(s, [("Nonstationarity", "Frequency and amplitude change with operating point."),
               ("Physical structure", "Specify phase transport through measured angular history."),
               ("Trade-off", "Test prediction accuracy against cost and fault preservation.")])

    s = d.slide("Experimental design", "Hold model selection inside the training acquisition", 55,
        "Use vibration and RPM observed through time t minus one. Train with guarded contiguous blocks and score the chosen models on the official validation record. Candidate records are a separate diagnostic set. Exact overlapping records are excluded from group statistics. With absent blind test files, report validation and candidate diagnostics within their own scopes.")
    panels(s, [("Training", "Three contiguous held-out blocks with a 1,792-sample guard."),
               ("Development", "Known past only. Score [1792, 24000) on sigvalid."),
               ("Diagnostic corpus", "Exclude exact record overlaps. Condition labels remain unverified.")])
    takeaway(s, "Every scored development method uses the same 22,208 targets")

    s = d.slide("Training-only physical evidence", "Use observed ridges to motivate reservoir features", 50,
        "The 37-tooth shaft defines the theoretical mesh-order family. The measured spectrogram tests whether spectral ridges track RPM. The angle-domain view tests whether peaks sharpen at fixed orders. Broad hertz bands and transient effects motivate separately controlled candidates. The model can use lagged phase through measured inputs, while this descriptive spectrum uses a complete training record.")
    figure_panel(s, spectrogram(e), "A feature hypothesis",
                 "A six-order bank with 32-sample decay reduced mean guarded screen MSE from 0.462 to 0.435.",
                 "Training spectrogram • Measured trace, physical overlays")

    s = d.slide("PI-SRC implementation", "Transport historical observations to target phase", 80,
        "The finite harmonic state sums past vibration after transporting each sample from its measured shaft phase to the predicted target phase. Decay weights make older samples matter less. The target phase advances using the previous RPM. Other groups include vibration delays, angular lags, speed context and lagged proxy innovations. The nonlinear map uses selected feature products. The model fits the readout inside each training split, while order and memory settings are selected externally. The adapter can be applied identically to eligible controls.")
    architecture(s)
    takeaway(s, "Physical settings are selected in guarded folds and frozen in deployment")

    s = d.slide("Novelty and attribution", "Separate physical structure from the error adapter", 52,
        "The innovation pipeline and harmonic states create a structured, finite-memory representation. The regularized readout builds on this representation. The clipped residual adapter adds a slow correction from already observed base errors. Match that adapter across eligible models before attributing improvement to the oscillator or nonlinear structure. Review the fit objective, exact input limits, and feature-group ablations together.")
    panels(s, [("Reservoir map", "Angular and order-aware states with interpretable memory."),
               ("Readout", "Train group-scaled regularization on the structured library."),
               ("Adapter control", "Compare the same base model with correction disabled or matched.")])

    s = d.slide("Guarded joint optimization", "Choose physical and readout settings together", 58,
        selection_text(e) + " The search considers finite feature and regularization choices and tracks its actual budget. Score each configuration under clean, sensor-perturbed and pulse-injection training conditions. Selection is a prespecified compromise and the study reports fold and development outcomes separately.")
    trial_rows = e.csv("results/optimization/trials.csv")
    panels(s, [("Physical search", "Vibration and angle delays, order banks, memory and optional fixed-frequency components."),
               ("Objective", "Guarded clean MSE plus RPM stress and pulse-distortion penalties."),
               ("Selected evidence", selection_text(e) + (f" {len(trial_rows)} trial-ledger rows." if trial_rows else ""))])

    archive = fmt(e.archive_reference, "mse")
    s = d.slide("Matched development prediction", "Compare against strong measured controls", 66,
        e.comparison_text() + f" The strongest replayed archive SRC checkpoint scored {archive} on the same mask." + " All 21 required comparator checkpoints were independently replayed. The complete model table records their evaluation scopes. The paper’s 0.2893 test result belongs to its own segment.")
    entries = e.comparison_top
    if entries:
        table(s, ["Predictor", "MSE", "MAE", "CPU median (ms)"],
              [(name(r), fmt(r, "mse"), fmt(r, "mae"), fmt(r, "warm_median_ms", digits=1)) for r in entries],
              [6.12, 1.64, 1.66, 2.60], height=0.52)
    takeaway(s, e.comparison_text())

    s = d.slide("Operating regimes", "Inspect local errors as well as the mean", 50,
        "A mean error can combine acceleration, speed peaks and different vibration amplitudes. A matched short trace shows when the model follows the waveform and what remains in the residual. Errors by RPM bin show whether an observed gain generalizes across operating regimes within this record. The paired block comparison respects dependence among adjoining targets.")
    figure_panel(s, e.plot("comparison/validation_trace.png", "comparison/matched_waveforms.png", "comparison/matched_trace.png"),
                 "Use paired targets", "Inspect a common trace and speed-regime summaries. Keep inference at the acquisition level.",
                 "Official development • Measured and predicted vibration on aligned targets")

    s = d.slide("Computational efficiency", "Accuracy and inference cost share the decision", 54,
        "Batch runtime includes feature construction and model prediction on the same input under the recorded threading policy. Model size and fit time illustrate different practical constraints. A full batch timing describes throughput on this hardware. Deployment needs separate streaming and tail-latency measurements. GPU cost can only be reported for measured hardware.")
    timing_rows = list(r for r in (e.primary, e.sparse_reference, e.archive_reference) if r)
    if timing_rows:
        table(s, ["Predictor", "MSE", "CPU median (ms)", "Fitted coefficients"],
              [(name(r), fmt(r, "mse"), fmt(r, "warm_median_ms", digits=1), fmt(r, "parameters", digits=0))
               for r in timing_rows], [5.50, 1.52, 2.30, 2.70], top=2.10, height=0.62)
        text(s, "One 24,000-sample record per batch • Includes feature construction and prediction",
             0.65, 5.21, 12.0, 0.57, 20, NAVY)
    takeaway(s, "Report CPU throughput and retain online latency as a separate measure.")

    s = d.slide("Candidate-record diagnosis", "Residual PMI measures one-turn modulation", 62,
        diagnostic_text(e) + " The diagnostic resamples each residual onto a common shaft-angle grid and computes the centered Hilbert envelope. PMI transforms the lagged correlation near one shaft turn. Exact overlaps are removed. The remaining 198 records have index-group labels only, so AUC describes group ordering. Synthetic pulse injection answers a different question about residual preservation under a known perturbation.")
    saved_pulse = next((r for r in e.pulses if model_key(r) == model_key(e.primary) and r.get("kind") == "periodic"), {})
    increase = fmt(saved_pulse, "positive_delta_count", digits=0)
    cases = fmt(saved_pulse, "cases", digits=0)
    pulse_line = f" PMI rose in {increase} of {cases} synthetic periodic cases." if saved_pulse else ""
    figure_panel(s, e.plot("fault/fault_group_distribution.png"), "Provisional A/B separation",
                 diagnostic_text(e) + pulse_line,
                 "Course candidate records • Identical warm-up and angular diagnostic convention")

    s = d.slide("Controlled ablations", "Attribute the result across matched controls", 54,
        ablation_text(e) + " The comparison separates order-aware state families, interaction groups and the same residual correction. An improvement in clean MSE can change residual evidence and sensitivity to RPM error. Interpret the ablations and robustness metrics jointly before recommending a deployment variant.")
    panels(s, [("Feature families", "Compare fitted controls with oscillator and angular groups removed."),
               ("Residual correction", "Match the correction across eligible forecast baselines."),
               ("Measured attribution", ablation_text(e))])

    s = d.slide("Conclusions", "What the released evidence establishes", 49,
        e.comparison_text() + " The finite structured reservoir creates a reproducible physical hypothesis about speed-dependent vibration. The project documents performance and trade-offs on a shared benchmark. Independent labels, independent blind data and streaming latency remain needed to characterize deployment.")
    panels(s, [("Prediction", e.comparison_text() + f" Archive replay {archive}."),
               ("Fault evidence", "Use candidate PMI and paired synthetic pulses within their documented scope."),
               ("Independent test", "The supplied release lacks sigtest and sigfault2. Frozen evaluation remains available.")])

    technical_backup(d, e)
    s = d.slide("Backup / operating regimes", "Prediction error by ascending RPM quartile", 0,
        "The quartile plot uses bins calculated once from measured development RPM under the common mask. Read these bins as operating-point descriptions within one acquisition.")
    figure_panel(s, e.plot("comparison/rpm_quartiles.png"), "Within-record comparison",
                 "Match targets across methods. Stratified scores indicate where relative differences arise.",
                 "Official development • Measured RPM only for defining post-fit strata")
    stress = primary_artifact_row(e.stress, e)
    pulse = next((r for r in e.pulses if model_key(r) == model_key(e.primary) and r.get("kind") == "periodic"), {})
    s = d.slide("Backup / stress and signal preservation", "Accuracy and residual preservation can diverge", 0,
        "Stress changes the predictor's speed history. Pulse tests inject known synthetic signals into the same record and measure changes in predicted and residual components.")
    nuisance = next((r for r in e.csv("results/fault/rpm_stress.csv")
                     if model_key(r) == model_key(e.primary)
                     and r.get("condition") == "bias_plus_2pct" and r.get("scope") == "predictor"), {})
    panels(s, [("RPM stress", f"Primary clean {fmt(stress, 'clean_mse', digits=3)} and worst tested {fmt(stress, 'worst_stress_mse', digits=3)} MSE."),
               ("Synthetic pulses", f"Primary residual PMI rose in {fmt(pulse, 'positive_delta_count', digits=0)} of {fmt(pulse, 'cases', digits=0)} periodic injections."),
               ("Sensor effect", f"+2% predictor RPM bias changed PI-SRC validation PMI by {fmt(nuisance, 'delta_pmi', digits=3)}.")])
    all_rows = e.comparison
    for i in range(0, len(all_rows), 10):
        s = d.slide("Backup / complete comparison", f"Model coverage and settings evidence {i // 10 + 1}", 0,
            "The measured table uses common standardized targets and identifies unavailable values. Exact configurations and fitted weights are in the result registry.")
        chunk = all_rows[i:i + 10]
        table(s, ["Model", "MSE", "CPU ms", "Parameters"],
              [(name(r), fmt(r, "mse"), fmt(r, "warm_median_ms", digits=1),
                fmt(r, "parameters", digits=0)) for r in chunk],
              [6.58, 1.80, 1.78, 1.86], top=1.94, height=0.445)
    return d


def render(paths, directory):
    import fitz

    exe = shutil.which("soffice") or shutil.which("libreoffice")
    if not exe:
        raise RuntimeError("LibreOffice is required for presentation visual QA")
    review = directory / "rendered"
    review.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="src_slides_") as tmp:
        profile = (Path(tmp) / "profile").resolve().as_uri()
        subprocess.run([exe, f"-env:UserInstallation={profile}", "--headless", "--convert-to", "pdf",
                        "--outdir", str(review), *(str(path) for path in paths)], check=True,
                       text=True, capture_output=True, timeout=120)
    rows = []
    for path in paths:
        pdf_name = review / f"{path.stem}.pdf"
        preread = Presentation(path)
        with fitz.open(pdf_name) as document:
            if len(document) != len(preread.slides):
                raise RuntimeError(f"Render count differs for {path.name}")
            thumbs = []
            for i, page in enumerate(document, 1):
                pix = page.get_pixmap(matrix=fitz.Matrix(1.1, 1.1), alpha=False)
                pix.save(review / f"{path.stem}_slide_{i:02d}.png")
                title = preread.slides[i - 1].shapes[1].text
                printed = "".join(page.get_text().split())
                if "".join(title.split()) not in printed:
                    raise RuntimeError(f"Missing rendered title on {path.name} slide {i}")
                thumb = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                thumb.thumbnail((309, 174))
                thumbs.append(thumb)
            sheet = Image.new("RGB", (1280, ((len(thumbs) + 3) // 4) * 186), "#dbe5e8")
            for i, thumb in enumerate(thumbs):
                sheet.paste(thumb, (i % 4 * 320 + 5, i // 4 * 186 + 5))
            sheet.save(review / f"{path.stem}_contact.png")
            rows.append(f"- {path.name} rendered {len(document)} slides. All titles extracted and slide bounds verified.")
    (directory / "presentation_verification.md").write_text(
        "# Presentation verification\n\n" + "\n".join(rows) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "presentations")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    e = Evidence(args.project_dir.resolve())
    target = args.out.resolve()
    target.mkdir(parents=True, exist_ok=True)
    paths = [progress(e).save(target), final(e).save(target)]
    if args.render:
        render(paths, target)
    for path in paths:
        print(path.name)


if __name__ == "__main__":
    main()
