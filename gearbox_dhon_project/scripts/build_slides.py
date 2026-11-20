"""Regenerate both editable MECG 7780 presentations from run artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
W, H = 13.333333, 7.5
FONT = "Liberation Sans"
INK, MUTED, TEAL = "162731", "405760", "046962"
PAPER, WHITE, LINE = "F7FAF9", "FFFFFF", "CFDEDB"
PALE, WARN, AMBER = "E3F1EE", "FBEDE0", "925321"


def clock(seconds: int) -> str:
    return f"{seconds // 60}:{seconds % 60:02d}"


def rgb(hex_color: str) -> RGBColor:
    return RGBColor.from_string(hex_color)


def rect(slide, x, y, w, h, fill=WHITE, border=LINE):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    style = shape.element.find("{http://schemas.openxmlformats.org/presentationml/2006/main}style")
    if style is not None:
        shape.element.remove(style)
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    if border:
        shape.line.color.rgb = rgb(border)
        shape.line.width = Pt(0.75)
    else:
        shape.line.fill.background()
    return shape


def textbox(
    slide, value, x, y, w, h, size=20, color=INK, bold=False,
    align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP,
):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.vertical_anchor = valign
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0
    for index, line in enumerate(str(value).split("\n")):
        para = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        para.text = line
        para.alignment = align
        para.font.name = FONT
        para.font.size = Pt(size)
        para.font.color.rgb = rgb(color)
        para.font.bold = bold
        para.space_after = Pt(3)
    return box


def picture(slide, source, x, y, w, h):
    """Place a run plot at native aspect ratio in a bounded panel."""
    if not source or not source.is_file():
        rect(slide, x, y, w, h, PALE)
        textbox(slide, "Plot unavailable for this run", x + 0.28, y + h / 2 - 0.17,
                w - 0.56, 0.40, 18, MUTED, align=PP_ALIGN.CENTER)
        return
    with Image.open(source) as image:
        aspect = image.width / image.height
    actual_w = min(w, h * aspect)
    actual_h = actual_w / aspect
    slide.shapes.add_picture(str(source), Inches(x + (w - actual_w) / 2),
                             Inches(y + (h - actual_h) / 2),
                             width=Inches(actual_w), height=Inches(actual_h))


def tag(slide, value, y=1.68):
    textbox(slide, value, 0.66, y, 12.02, 0.36, 17, MUTED)


def cards(slide, items, y=2.24, height=3.55):
    w = (12.02 - 0.26 * (len(items) - 1)) / len(items)
    for i, (label, value) in enumerate(items):
        x = 0.66 + i * (w + 0.26)
        rect(slide, x, y, w, height)
        rect(slide, x, y, 0.065, height, TEAL, None)
        label_height = 0.88 if len(items) == 3 else 0.65
        textbox(slide, label, x + 0.25, y + 0.27, w - 0.50, label_height, 22, TEAL, True)
        textbox(slide, value, x + 0.25, y + label_height + 0.43,
                w - 0.50, height - label_height - 0.57, 19)


def strip(slide, text, y=6.19, warning=False):
    rect(slide, 0.66, y, 12.02, 0.59, WARN if warning else PALE, None)
    textbox(slide, text, 0.86, y + 0.075, 11.62, 0.44, 16,
            AMBER if warning else TEAL, True, valign=MSO_ANCHOR.MIDDLE)


def rows(slide, headers, values, widths, y=2.26, row_height=0.52):
    height = row_height * (len(values) + 1)
    shape = slide.shapes.add_table(len(values) + 1, len(headers), Inches(0.66), Inches(y),
                                   Inches(sum(widths)), Inches(height))
    table = shape.table
    for column, width in zip(table.columns, widths, strict=True):
        column.width = Inches(width)
    for r, entries in enumerate([headers, *values]):
        table.rows[r].height = Inches(row_height)
        for c, value in enumerate(entries):
            cell = table.cell(r, c)
            cell.margin_left = cell.margin_right = Inches(0.12)
            cell.margin_top = cell.margin_bottom = Inches(0.035)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(PALE if r == 0 else WHITE if r % 2 else PAPER)
            cell.text = str(value)
            cell.text_frame.word_wrap = True
            para = cell.text_frame.paragraphs[0]
            para.font.name = FONT
            para.font.size = Pt(16 if c == 0 else 17)
            para.font.bold = r == 0 or c == 0
            para.font.color.rgb = rgb(TEAL if r == 0 else INK)
            para.alignment = PP_ALIGN.LEFT
    return shape


def steps(slide, items, y=2.34, h=2.65):
    gap = 0.31
    w = (12.02 - gap * (len(items) - 1)) / len(items)
    for i, (label, desc) in enumerate(items):
        x = 0.66 + i * (w + gap)
        rect(slide, x, y, w, h, PALE if i == 1 else WHITE)
        textbox(slide, label, x + 0.17, y + 0.21, w - 0.34, 0.73, 19, TEAL, True)
        textbox(slide, desc, x + 0.17, y + 1.01, w - 0.34, h - 1.12, 18)
        if i < len(items) - 1:
            textbox(slide, "→", x + w, y + 0.95, gap, 0.42, 16, TEAL,
                    True, PP_ALIGN.CENTER)


def figure_and_text(slide, source, headline, body, caption):
    picture(slide, source, 0.66, 1.95, 7.49, 4.48)
    rect(slide, 8.38, 2.04, 4.30, 3.92, WHITE)
    textbox(slide, headline, 8.62, 2.27, 3.82, 0.83, 22, TEAL, True)
    textbox(slide, body, 8.62, 3.20, 3.82, 2.55, 19)
    textbox(slide, caption, 0.66, 6.52, 12.02, 0.27, 13, MUTED)


@dataclass
class RunData:
    path: Path
    overrides: dict

    def csv(self, *names):
        for name in names:
            path = self.path / name
            if path.is_file():
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    return list(csv.DictReader(stream))
        return []

    def json(self, *names):
        for name in names:
            path = self.path / name
            if path.is_file():
                return json.loads(path.read_text(encoding="utf-8"))
        return {}

    def setting(self, key, default):
        return self.overrides.get(key, default)

    def fig(self, *names):
        return next((self.path / "figures" / n for n in names
                     if (self.path / "figures" / n).is_file()), None)


class Deck:
    def __init__(self, kind: str):
        self.kind = kind
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(W), Inches(H)
        self.plan = []
        self.elapsed = 0

    def slide(self, title, chapter, seconds, notes):
        s = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        fill = s.background.fill
        fill.solid()
        fill.fore_color.rgb = rgb(PAPER)
        num = len(self.prs.slides)
        window = f"{clock(self.elapsed)}–{clock(self.elapsed + seconds)}" if seconds else "Backup"
        self.elapsed += seconds
        textbox(s, chapter.upper(), 0.66, 0.34, 10.5, 0.22, 12, TEAL, True)
        textbox(s, title, 0.66, 0.74, 12, 0.86, 33, INK, True)
        rect(s, 0.66, 1.55, 12.02, 0.015, LINE, None)
        rect(s, 0.66, 7.04, 12.02, 0.015, LINE, None)
        textbox(s, f"MECG 7780   /   {self.kind.title()}", 0.66, 7.13, 5.50, 0.22, 11, MUTED)
        textbox(s, window, 6.20, 7.13, 2.03, 0.22, 11, MUTED, align=PP_ALIGN.CENTER)
        textbox(s, f"{num:02d}", 12.00, 7.10, 0.68, 0.23, 13, TEAL,
                True, PP_ALIGN.RIGHT)
        s.notes_slide.notes_text_frame.text = f"{window} • {clock(seconds)}\n\n{notes}"
        self.plan.append((title, window, seconds, notes))
        return s

    def save(self, directory, expected):
        assert self.elapsed == expected, f"Planned length {clock(self.elapsed)}"
        stem = f"{self.kind}_presentation"
        self.prs.core_properties.title = f"MECG 7780 | {self.kind.title()} presentation"
        self.prs.core_properties.subject = "Variable-speed gearbox vibration and fault diagnostics"
        self.prs.core_properties.author = "MECG 7780 project"
        pptx = directory / f"{stem}.pptx"
        self.prs.save(pptx)
        markdown = [
            f"# MECG 7780 {self.kind.title()} presentation speaker notes",
            f"**Planned talk length** {clock(expected)}. Backup slides follow the timed talk.",
            "",
            "| Slide | Planned window | Duration |",
            "|---|---:|---:|",
        ]
        markdown.extend(f"| {i}. {title} | {window} | {clock(seconds) if seconds else '—'} |"
                        for i, (title, window, seconds, _) in enumerate(self.plan, 1))
        for i, (title, window, _, notes) in enumerate(self.plan, 1):
            markdown.extend(("", f"## {i}. {title}", "", f"**{window}**", "", notes, ""))
        (directory / f"{self.kind}_speaker_notes.md").write_text("\n".join(markdown), encoding="utf-8")
        self.validate(pptx)
        return pptx

    def validate(self, path):
        reread = Presentation(path)
        if len(reread.slides) != len(self.plan):
            raise ValueError(f"Slide count differs after reload of {path.name}")
        for num, (slide, (_, window, _, notes)) in enumerate(zip(reread.slides, self.plan, strict=True), 1):
            stored = slide.notes_slide.notes_text_frame.text
            if window not in stored or notes[:30] not in stored:
                raise ValueError(f"Missing notes on {path.name} slide {num}")
            for shape in slide.shapes:
                if (min(shape.left, shape.top) < 0
                        or shape.left + shape.width > reread.slide_width + 5
                        or shape.top + shape.height > reread.slide_height + 5):
                    raise ValueError(f"Out-of-bounds shape on {path.name} slide {num}")


def title_panel(slide, audience, question):
    rect(slide, 0.66, 1.95, 12.02, 3.84, PALE, None)
    textbox(slide, "Physics-informed differentiable\nharmonic oscillator neural network",
            1.04, 2.29, 11.23, 1.53, 37, INK, True)
    textbox(slide, question, 1.06, 4.22, 10.82, 0.93, 23, TEAL)
    textbox(slide, audience, 0.68, 6.25, 11.97, 0.39, 17, MUTED)


def protocol(slide, note):
    tag(slide, "Use prior vibration and RPM to predict the next observed sample")
    steps(slide, [
        ("Fit", "Use sigtrain only. Guard all folds against shared raw history."),
        ("Select", "Rank candidates on held-out training blocks."),
        ("Evaluate", "Use sigvalid on the same 22,208 targets."),
        ("Freeze", "Save model, calibration and protocol before any blind data."),
    ])
    strip(slide, note)


def causal_architecture(slide, note):
    tag(slide, "At target t, observed vibration and speed end at t−1")
    steps(slide, [
        ("Past samples", "Measured vibration, RPM, angle history and proxy innovations"),
        ("Oscillator bank", "Relative phase and finite-memory quadratures with trainable order and decay"),
        ("Prediction", "RPM-conditioned linear readout with a compact nonlinear head"),
        ("After arrival", "Measure residual. Update the optional adapter for the next target."),
    ], h=2.91)
    strip(slide, note)


def backup(slide, subtitle, items, foot):
    tag(slide, subtitle)
    cards(slide, items, y=2.30, height=3.59)
    strip(slide, foot)


def eda_story(data):
    eda = data.json("eda/summary.json")
    ridge = eda.get("ridge_148_median_abs_error_hz")
    mesh_min, mesh_max = eda.get("mesh_hz_min"), eda.get("mesh_hz_max")
    screen = eda.get("guarded_screen_mean_mse", {})
    base = screen.get("LPV-AR24")
    quadrature = screen.get("LPV-AR24 + 148 + 259 order quadrature")
    if any(v is None for v in (ridge, mesh_min, mesh_max)):
        return "For a tracked 37-tooth shaft, mesh frequency equals 37 × RPM / 60."
    story = f"Mesh spans {mesh_min:.1f}–{mesh_max:.1f} Hz. Order 148 tracks its theoretical ridge to {ridge:.2f} Hz median absolute error."
    if base is not None and quadrature is not None:
        story += f" Order quadratures lowered mean training-fold MSE from {base:.3f} to {quadrature:.3f} in the feature screen."
    return story


def progress(data: RunData) -> Deck:
    deck = Deck("progress")
    s = deck.slide("Predicting variable-speed gearbox vibration", "Research question", 45,
        "The gearbox speeds up and slows down, so a mesh-related component moves in frequency through a single acquisition. A fixed-lag model can still exploit short-time dependence, but it has to infer that changing phase relationship from samples. We want to estimate healthy vibration one sample ahead and test whether the remaining signal retains evidence of periodic tooth damage. This progress overview uses descriptive examples generated during preparation of the submitted project.")
    title_panel(s, "MECG 7780   /   Graduate course project   /   Progress overview",
                "Can shaft-angle dynamics improve one-step prediction while preserving crack-period evidence?")

    s = deck.slide("How the project will be judged", "Objectives", 45,
        "The course evaluates prediction accuracy, computational cost and fault detection. Our primary error metric is mean squared error on identical target indices. Efficiency means measuring inference with the deployed feature pipeline and recording the platform. For diagnostics, we first model normal vibration and then examine the envelope of the residual at one target-shaft revolution. A lower prediction error alone cannot establish a diagnostic benefit because a predictor that follows fault impulses can also suppress them.")
    cards(s, [("Prediction", "Compare one-step MSE with sparse LPV–ARMA and learned controls."),
              ("Efficiency", "Measure wall-clock inference on the available CPU and GPU."),
              ("Diagnostics", "Compare angular envelope periodicity in raw vibration and residuals.")])
    strip(s, "Evaluation uses common targets and a common residual-processing procedure.")

    s = deck.slide("The official split and overlap checks", "Data integrity", 60,
        "The official training and development validation records each contain twenty-four thousand samples, corresponding to seven and a half seconds at 3,200 Hz. The split fault collection supports a separate diagnostic analysis. The exact-match audit verified duplicate prefixes in fault-development records 61 and 63, so we preserve those records and exclude them from the eligible group comparison. Fault labels require independent acquisition metadata. A split by record position can describe a group pattern, but a verified classification score needs confirmed labels and records withheld from tuning.")
    figure_and_text(s, data.fig("eda_speed.png"), "One record per split",
                    "24,000 samples each at 3,200 Hz. The fault collection is audited separately for overlaps and label provenance.",
                    "Supplied training signal • RPM, standardized vibration and local statistics")

    s = deck.slide("Mesh energy follows shaft speed", "Training physics EDA", 90,
        "For an encoder on the monitored 37-tooth shaft, mesh frequency is thirty-seven times shaft speed divided by sixty. Multiples of that frequency trace families of curves on a time-frequency map. Integrating RPM gives shaft revolutions, so resampling by angle brings speed-locked components near stable orders. Broadband energy is an additional hypothesis for fixed-frequency features. A spectrogram alone cannot identify a natural mode. Each prior becomes a model change to test on guarded training blocks. Compare order-aligned lags with temporal lags, and compare fixed and learnable oscillator banks. The exploratory transforms use a whole record and the production predictor takes its inputs from measured history.")
    figure_and_text(s, data.fig("eda_spectrogram.png"), "Map RPM to mesh frequency",
                    data.setting("eda_summary", eda_story(data)),
                    "Descriptive training spectrogram • The overlays are speed-derived hypotheses")

    s = deck.slide("Sparse LPV–ARMA establishes the reference", "Conventional model", 60,
        "An autoregression uses previous vibration samples. LPV means that the lag coefficients vary smoothly with an operating variable, here speed. The moving-average portion uses lagged innovations, approximated by first-stage prediction errors. The sparse fit restricts the number of active coefficients. The strict benchmark schedules coefficients using speed available at the preceding sample and reproduces performance on the supplied validation segment. The paper result supplies context because the segment described there differs in speed phase.")
    tag(s, "A speed-conditioned filter uses vibration lags and lagged proxy innovations")
    steps(s, [("Stage I", "Fit a speed-dependent autoregression to estimate proxy innovations."),
              ("Stage II", "Fit a model with past vibration and past innovations."),
              ("Sparse fit", "Select support with an L1 penalty and refit active terms.")])
    strip(s, "Paper and course scores refer to different validation evidence.")

    s = deck.slide("DHON makes the oscillator trainable", "Proposed method", 90,
        "Here phase comes from the previous RPM and an integration of the measured speed history. For each carrier, the encoder combines earlier vibration samples after transporting them to the target phase. Damping gives a shorter or longer statistical memory. The archived model selected the oscillator settings externally and learned a mixing head. The proposed computation places bounded order and decay parameters inside the differentiable map. A fixed-state control uses the same architecture with those parameters held at their initial values. Separately, a slow residual adapter sees clipped errors only after each target arrives. The matched adapter control measures whether slow bias reduction explains any improvement in the final combined predictor.")
    causal_architecture(s, "Learn bounded detuning around mesh harmonics and positive decay from training data.")

    s = deck.slide("Tune within training and score once on development", "Experiment design", 60,
        "The fixed gap around each held-out training block protects the raw histories that feed fitting and scoring. Past samples before a held-out target are observed as they would be in one-step deployment. Every model uses the same development targets, from index 1,792 through index 23,999 inclusive. The key ablations compare the fixed versus trainable bank and apply the same optional adapter to both. RPM stress changes only the claimed measurement channel. Synthetic pulses test fault preservation in paired injected and unmodified runs.")
    protocol(s, "Development compares models. Blind evaluation requires separate independent data.")

    s = deck.slide("Progress and remaining evidence", "Checkpoint", 30,
        "The descriptive evidence gives us interpretable feature candidates and a causal design. The next scientific check is whether the learning and correction controls change prediction error or the amount of periodic pulse retained. External test files, confirmed acquisition labels and reference MATLAB outputs govern the strength of the final claims.")
    cards(s, [("Use now", "Training EDA supports testing mesh-locked phase features."),
              ("Measure next", "Compare fixed and learned states, with and without the same adapter."),
              ("Resolve externally", "Confirm labels, blind data, and instructor PMI reference outputs.")])
    append_backups(deck, data, is_final=False)
    return deck


def final(data: RunData) -> Deck:
    deck = Deck("final")
    s = deck.slide("Predicting variable-speed gearbox vibration", "Research question", 60,
        "This project asks whether an oscillator whose phase follows shaft motion and whose order and memory can be trained supplies a practical one-step predictor for gearbox vibration. The assignment compares accuracy, computational cost and diagnostics. First, I will trace which measurements motivate the oscillator prior. Next, I will explain how the recurrence and optional bias adapter use only past observations. Finally, I will compare actual development prediction, efficiency and residual periodicity against strong reference models. The two kinds of performance answer different engineering questions, and the available labels and held-out data determine how far the conclusions apply.")
    title_panel(s, "MECG 7780   /   Graduate course project   /   Final presentation",
                "One-step prediction and residual periodicity under changing speed")

    s = deck.slide("Audit the split before fitting", "Data integrity", 45,
        "The designated healthy training and development records each contain twenty-four thousand samples at 3,200 Hz. The development fault collection contains 200 recordings. Records 61 and 63 share exact training and validation prefixes respectively, so both are excluded from the eligible group comparison. The group labels denote index ranges until acquisition metadata establishes the meaning of each record.")
    figure_and_text(s, data.fig("eda_speed.png"), "Official prediction split",
                    "Fit on sigtrain and score on sigvalid. Exact matching excluded fault-development records 61 and 63 from group comparisons.",
                    "Supplied training signal • RPM, standardized vibration and local statistics")

    s = deck.slide("Shaft angle gives an interpretable frequency prior", "Training EDA", 45,
        "At the tracked shaft, a 37-tooth gear suggests first mesh order thirty-seven. RPM scales the expected time-domain frequency, while angular resampling expresses frequency as events per revolution. This makes exact mesh carriers a natural starting point. The observed spectrum also motivates a separate fixed-frequency sensitivity test. The model uses past RPM and relative phase to implement its order features. Whole-record descriptive time-frequency analysis stays in the exploratory workflow.")
    figure_and_text(s, data.fig("eda_spectrogram.png"), "An order is cycles per revolution",
                    data.setting("eda_summary", eda_story(data)),
                    "Training spectrogram • Speed-derived mesh overlays")

    s = deck.slide("A matched benchmark for sparse LPV–ARMA", "Reference and protocol", 60,
        "The conventional model is a speed-dependent autoregression with past proxy innovations. It schedules coefficients with RPM and restricts the active support. This implementation distinguishes the direct paper convention from strict prediction with speed through t minus one. Development scoring uses the same twenty-two thousand two hundred eight targets for the model comparisons. Chen and coauthors reported a test MSE of 0.2893 on their own validation segment. The supplied segment's different speed phase prevents a paired comparison with that external value. The independent rerun on supplied data is the benchmark for this project.")
    cards(s, [("Reference", "Two-stage speed-conditioned autoregression and lagged proxy innovations."),
              ("Common mask", "sigvalid indices [1792, 24000). Both speed and vibration inputs end at t−1."),
              ("External context", "Paper Table 6 gives 0.2893 on its own segment. Compare against the local rerun.")])
    strip(s, "Primary metric is MSE in standardized vibration squared.")

    s = deck.slide("The prediction path is causal by construction", "DHON architecture", 60,
        "At each target position, previously observed RPM determines the phase increment. The encoder creates quadratures from finite histories, including delays expressed as shaft angle where the configured model uses them. The readout combines a speed-conditioned linear estimate with a bounded nonlinear component. Phase information can be carried by interpretable oscillator channels while amplitude can adapt with operating context. The optional residual adapter uses previous base-model errors and clipping. Even if that adapter improves MSE, its contribution is tested separately from learning the oscillator parameters.")
    causal_architecture(s, "Keep the causal residual adapter separate from oscillator learning.")

    s = deck.slide("Learn order and memory inside the state equation", "Differentiability", 60,
        "The weighted oscillator sum brings each past sample into a shared rotating frame at the target. Higher damping shortens its statistical memory. Orders are bounded around physical initial values, and decay receives a positive parameterization. History is finite and state resets per recording, so the design has a calculable history requirement. The linear and nonlinear mixture can then learn from these states. Implementation checks compare direct summation with the optimized calculation and evaluate derivatives with respect to the oscillator parameters. These checks separate a mathematically differentiable design from evidence that the implemented optimizer actually updates oscillator states.")
    tag(s, "Each past sample contributes at its relative phase")
    rect(s, 0.66, 2.20, 12.02, 1.11, PALE, None)
    textbox(s, "hₖ,ₜ = (2 / Zₖ) Σⱼ ρₖʲ⁻¹ xₜ₋ⱼ exp[i(φ̂ₖ,ₜ − φₖ,ₜ₋ⱼ)]",
            0.97, 2.55, 11.40, 0.46, 27, TEAL, True)
    cards(s, [("Order", "Constrain detuning around physical mesh harmonics."),
              ("Decay", "Learn positive finite-memory weighting with ρ = exp(−1/τ)."),
              ("Control", "Freeze oscillator parameters while retaining the same readout.")],
          y=3.66, height=2.57)
    textbox(s, "Relative phase is computed from known RPM and a causal one-step phase estimate.",
            0.68, 6.50, 11.96, 0.27, 14, MUTED)

    s = deck.slide("Search jointly before opening development results", "Optimization protocol", 45,
        "A guarded training split assigns contiguous held-out blocks with protection on either side. This prevents lagged inputs in fitting rows from touching a held-out target. The finite search varies feature definitions, oscillator groups and learning controls under a recorded budget. The fold objective combines clean MSE, RPM stress MSE and synthetic pulse distortion. Selection uses training-fold scores only. The evaluation reports the resulting choices on the official development record.")
    protocol(s, "The search is bounded by the executed configuration and trial budget.")

    selection = data.json("search/dhon_selection.json")
    trial_rows = data.csv("search/dhon_trials.csv")
    completed = [row for row in trial_rows if row.get("status") == "complete"]
    trials = {row.get("trial") for row in completed} - {None}
    ranking = data.csv("search/dhon_ranking.csv")
    count = len(ranking) if ranking else len(trials)
    selected_trial = selection.get("primary_trial", "").replace("_", " ")
    fold_text = data.setting("fold_summary",
                            f"{count} ranked configurations across {len(completed)} completed fold fits."
                            if completed else "No completed search table available.")
    if selected_trial:
        fold_text += f" Primary trial {selected_trial}."
        top = next((row for row in ranking if row.get("trial") == selection.get("primary_trial")), {})
        fold_mse = numeric(top, "mse")
        if fold_mse is not None:
            fold_text += f" Mean fold MSE {fold_mse:.3f}."
    parameter_text = data.setting("parameter_summary", physical_parameters(selection, data))
    s = deck.slide("Read the selected configuration with its controls", "Optimization evidence", 45,
        f"{fold_text} {parameter_text} The comparison isolates learning of the state parameters from adding capacity to the readout. An improvement on a small set of contiguous folds should be treated with care. Check the checkpoint and the trial ledger for exact settings and whether the individual changes are distinguishable at this sample size.")
    cards(s, [("Search record", fold_text),
              ("Learned state", parameter_text),
              ("Reference controls", "Fixed oscillator, adapter off, and matched correction across eligible models.")])
    strip(s, "Optimization selects on training folds. The archive and paper provide separate references.")

    metrics = data.csv("metrics/summary.csv", "summary.csv")
    if not metrics:
        dhon_metrics = data.csv("metrics/dhon_metrics.csv")
        conventional = data.csv("metrics/conventional_metrics.csv")
        metrics = [*dhon_metrics, *data.csv("metrics/comparators.csv"),
                   *data.csv("metrics/archive_replay.csv"),
                   *(row for row in conventional if row.get("scheduling") == "strict")]
        primary_rows = [
            {"model": "primary_dhon_base", "mse": row.get("base_mse")}
            for row in dhon_metrics if row.get("model") == "primary_dhon"
        ]
        metrics.extend(primary_rows)
    timings = data.csv("metrics/timings.csv", "timings.csv", "metrics/timing.csv")
    timings = [row for row in timings if row.get("record", "sigvalid") in ("sigvalid", "development_7p5s")
               and row.get("device", "cpu") == "cpu"] or timings
    progress_rows, detail, outcome = model_results(metrics, timings, data)
    s = deck.slide("Development prediction on common targets", "Measured performance", 90,
        f"{outcome} {detail} The second column is the mean squared one-step error on standardized vibration, using exactly the same development targets. The timing column, where recorded, uses complete prediction from input vibration and RPM on the same CPU benchmark. The separate published value is context for the original experiment. The common mask supports a paired comparison within this recording. A new acquisition would provide independent evidence. Consult the paired time-block results to check sensitivity to temporal dependence.")
    tag(s, "sigvalid [1792, 24000) • standardized vibration² • 7.5-second CPU batch")
    if progress_rows:
        rows(s, ["Predictor", "MSE", "CPU median"], progress_rows[:7], [7.59, 2.16, 2.27], row_height=0.47)
    else:
        cards(s, [("Common target set", "22,208 samples are planned for fair comparison."),
                  ("Measured rows", "The comparative results table is unavailable for this run.")])
    strip(s, outcome.split(" It is ")[0], warning=not bool(progress_rows))

    fault_table = data.csv("metrics/fault_group_summary.csv")
    diagnostic = data.setting("diagnostic_summary", short_diagnostic(fault_table))
    fault_verify = data.json("metrics/fault_verification.json")
    diagnostic_pair = fault_verify.get("paired_provisional_index_group_gap_differences", {}).get(
        "archive_dhon_optimized_corrected", {})
    gap_ci = diagnostic_pair.get("CI_95_record_bootstrap")
    paired_pmi = (
        f"The paired primary minus archive median-gap interval was {gap_ci[0]:+.4f} to {gap_ci[1]:+.4f}."
        if gap_ci else "")
    s = deck.slide("Residual PMI measures periodicity by shaft angle", "Fault development", 45,
        f"{diagnostic} {paired_pmi} The diagnostic takes the residual after the same warm-up, resamples it to a uniform shaft angle, takes an analytic envelope and estimates its centered autocorrelation near one shaft turn. PMI increases with that lag correlation. We report the instructor-literal index and audit the mathematical one-turn indexing separately. Conditional AUC is the probability that a randomly chosen group B record ranks above a group A record, counting ties as half. Both groups have provisional labels.")
    figure_and_text(s, data.fig("fault_pmi_groups.png", "fault_pmi.png"),
                    "Provisional groups A and B", diagnostic,
                    "Development collection • Group A n=98 and B n=100 • Common warm-up")

    stress_rows = data.csv("metrics/rpm_stress.csv")
    pulse_rows = data.csv("metrics/pulse_preservation.csv")
    rpm_text, pulse_text = robustness(stress_rows, pulse_rows, data.csv("metrics/robustness_summary.csv"))
    s = deck.slide("Challenge the predictor and the diagnostic", "Robustness", 45,
        f"{rpm_text} {pulse_text} Altering the speed input can change the predictor's phase estimate. Altering diagnostic speed can affect angular resampling even with the same residual. Keep these two errors distinct. Paired synthetic pulses let us measure residual retention under controlled injections and carry no acquisition label. Explain any MSE changes alongside pulse retention before selecting a practical model.")
    cards(s, [("RPM robustness", data.setting("robustness_summary", rpm_text)),
              ("Synthetic pulse checks", data.setting("pulse_summary", pulse_text)),
              ("MATLAB parity", data.setting("matlab_summary",
                  "Python literal T−1 and corrected T indexing are available. Full reference parity requires the MATLAB helper and gold outputs."))])
    strip(s, "Keep synthetic robustness results separate from experimental index-group results.")

    ablation = data.setting("ablation_summary", short_ablation(metrics))
    paired = data.csv("metrics/paired_block_bootstrap.csv")
    pairing = next((row for row in paired if label(row) == "primary_dhon"
                    and "lpv" in row.get("reference", "").lower()), {})
    low, high = numeric(pairing, "low_95"), numeric(pairing, "high_95")
    paired_text = ("Paired 640-sample block bootstrap gives DHON − Spa LPV–ARMA MSE "
                   f"95% interval {low:+.4f} to {high:+.4f}.") if low is not None and high is not None else (
                   "Use paired time blocks for prediction differences. Targets share substantial history.")
    speed = data.setting("gpu_summary", short_timing(timings, data.json("timing_environment.json", "metrics/timing_environment.json")))
    s = deck.slide("Account for correction and computation", "Ablation and efficiency", 75,
        f"{ablation} {speed} {paired_text} Inspect the fixed and learned versions under identical adapter settings. Then compare each version with its adapter disabled. A slow clipped adapter can reduce bias without changing oscillator parameters, but may also change a periodic pulse signature. The timing value includes its model's feature construction and inference under a batch protocol where measured. An individual streamed sample may have different latency, so batch throughput should be reported with that boundary. Multiple timing runs give variation rather than implying a fixed execution time.")
    cards(s, [("Learning contribution", ablation),
              ("CPU measurement", speed),
              ("Uncertainty", paired_text)])
    strip(s, "Accuracy and timing are development measurements on the recorded platform.")

    blind = data.setting("blind_summary", "sigtest and sigfault2 are unavailable. Frozen independent evaluation is pending.")
    s = deck.slide("What this experiment establishes", "Conclusions and limitations", 45,
        f"{outcome} {blind} The implementation directly tests whether constrained oscillator learning changes the forecast and keeps the effect of the optional adapter visible. Diagnosis is assessed through actual development residuals and synthetic retention tests with clear provenance. A truly independent conclusion requires executing the frozen protocol when the missing input records become available and verifying their acquisition labels and overlaps.")
    cards(s, [("Prediction", outcome),
              ("Diagnostics", "Interpret PMI group separation with provisional labels and explicit exclusions."),
              ("Independent test", blind)])
    append_backups(deck, data, is_final=True)
    return deck


def numeric(row, *keys):
    lower = {str(k).lower(): v for k, v in row.items()}
    for key in keys:
        raw = lower.get(key.lower())
        if raw not in (None, ""):
            try:
                v = float(raw)
                if v == v and abs(v) != float("inf"):
                    return v
            except (TypeError, ValueError):
                continue
    return None


def label(row):
    return row.get("model") or row.get("name") or row.get("variant") or ""


def readable(name):
    known = {
        "primary_dhon": "Selected DHON", "alternative_dhon": "Alternative DHON",
        "spa_lpv_arma": "Spa LPV–ARMA", "spa_lpv_ar": "Spa LPV–AR",
        "ari": "ARI", "arima": "ARIMA", "ngrc": "NG-RC", "mlp": "MLP",
        "lstm": "LSTM", "tcn": "TCN", "gru": "GRU",
        "mlp_rpm": "MLP + RPM", "mlp_vibration": "Vibration-only MLP",
        "gru_rpm": "GRU + RPM", "lstm_rpm": "LSTM + RPM", "tcn_rpm": "TCN + RPM",
        "spa_lpv_arma_refit": "Spa LPV–ARMA refit",
        "refit_spa_lpv_arma": "Spa LPV–ARMA refit",
        "sparse_lpv_arma": "Spa LPV–ARMA",
        "lpv_arma_dense": "Dense LPV–ARMA",
        "spa_lpv_arma": "Spa LPV–ARMA refit",
        "primary_dhon_base": "Selected DHON without adapter",
        "dhon_matched_fixed": "Matched fixed-state DHON",
        "archive_dhon_optimized_corrected": "Archive expanded DHON + correction (replayed)",
        "archive_dhon_tau_context_extended_gated_corrected": "Archive original DHON + correction (artifact)",
    }
    if name.lower() in known:
        return known[name.lower()]
    return (name.replace("_", " ").replace("lpv", "LPV").replace("dhon", "DHON")
            .replace("ngrc", "NG-RC").replace("arma", "ARMA"))


def metric_entries(metrics):
    entries = []
    for row in metrics:
        mse = numeric(row, "mse", "mse_z", "development_mse", "validation_mse")
        name = label(row)
        if mse is not None and name:
            entries.append((name, mse))
    return sorted(entries, key=lambda pair: pair[1])


def model_results(metrics, timings, data):
    entries = metric_entries(metrics)
    index = dict(entries)
    archives = data.csv("metrics/archive_replay.csv")
    by_status = {label(row): row.get("status") for row in archives}
    primary = next((pair for pair in entries if pair[0] == "primary_dhon"), None)
    baseline = next(((name, index[name]) for name in (
                    "spa_lpv_arma_refit", "refit_spa_lpv_arma", "strict_sparse_lpv_arma",
                    "sparse_lpv_arma", "spa_lpv_arma", "archive_spa_lpv_arma",
                ) if name in index), None)
    baseline = baseline or next((pair for pair in entries if re.search(r"spa.*lpv.*arma", pair[0], re.I)), None)
    archived = next((pair for pair in entries if by_status.get(pair[0]) == "checkpoint_replay"), None)
    artifact = next((pair for pair in entries if by_status.get(pair[0]) == "prediction_artifact_only"), None)
    learned = next((pair for pair in entries if re.search(r"ngrc|ng-rc|gru|lstm|tcn|mlp", pair[0], re.I)
                    and "archive" not in pair[0]), None)
    alternative = next((pair for pair in entries if pair[0] == "alternative_dhon"), None)
    dense = next((pair for pair in entries if pair[0] == "lpv_arma_dense"), None)
    chosen = []
    for pair in (primary, artifact, archived, baseline, learned, alternative, dense, *entries):
        if pair and pair not in chosen:
            chosen.append(pair)
    chosen = sorted(chosen[:7], key=lambda pair: pair[1])
    timing_lookup = {label(row): numeric(row, "median_s", "median_seconds") for row in timings}
    table = []
    for name, mse in chosen:
        duration = timing_lookup.get(name)
        table.append((readable(name), f"{mse:.6f}", "—" if duration is None else f"{duration:.3f} s"))
    if primary and archived:
        gap = 100 * (archived[1] - primary[1]) / archived[1]
        outcome = f"Selected DHON is {abs(gap):.2f}% {'lower' if gap >= 0 else 'higher'} in MSE than the strongest archive checkpoint replay."
        if baseline:
            spa = 100 * (baseline[1] - primary[1]) / baseline[1]
            outcome += f" It is {abs(spa):.2f}% {'lower' if spa >= 0 else 'higher'} than the Spa LPV–ARMA benchmark."
    elif primary and baseline:
        gap = 100 * (baseline[1] - primary[1]) / baseline[1]
        outcome = f"Selected DHON MSE {primary[1]:.6f} is {abs(gap):.2f}% {'lower' if gap >= 0 else 'higher'} than the matched Spa LPV–ARMA result."
    elif entries:
        outcome = f"Lowest measured development MSE {entries[0][1]:.6f} ({readable(entries[0][0])})."
    else:
        outcome = "No comparative prediction metrics are available for this run."
    detail = "Chen et al. reported 0.2893 on the paper's external segment."
    if artifact:
        detail += f" The original archive prediction artifact re-scores at {artifact[1]:.6f}. This row has no checkpoint timing."
    return data.setting("prediction_rows", table), data.setting("results_detail", detail), data.setting("outcome", outcome)


def physical_parameters(selection, data):
    epochs = data.json("search/primary_dhon_learning.json")
    if isinstance(epochs, list) and epochs:
        last = epochs[-1]
        predicted = last.get("carriers", [])
        taus = last.get("taus", [])
        configured = selection.get("configurations", {}).get("primary_dhon", {})
        order_count = len(configured.get("harmonics", []))
        orders = predicted[:order_count]
        if len(orders) > 5:
            nominal = [37 * n for n in configured["harmonics"]]
            max_detuning = max(abs(estimate - start) for estimate, start in zip(orders, nominal))
            values = (f"{len(orders)} mesh orders span {orders[0]:.2f}–{orders[-1]:.2f} cycles/rev. "
                      f"Largest detuning {max_detuning:.3f} order. ")
        else:
            values = "Learned orders " + ", ".join(f"{v:.2f}" for v in orders) + ". " if orders else ""
        if order_count < len(predicted):
            hz = ", ".join(f"{v:.1f}" for v in predicted[order_count:])
            values += f"Frequency bank centers {hz} Hz. "
        flat_tau = [float(value) for bank in taus for value in bank]
        if flat_tau:
            values += f"Decay spans {min(flat_tau):.1f}–{max(flat_tau):.1f} samples."
        if values:
            return values
    return "Inspect the selected checkpoint and search manifest for the bounded order and decay values."


def short_diagnostic(table):
    if not table:
        return "Group A and group B use provisional index ranges. See the run ledger for available PMI results."
    clean = {label(row): row for row in table if row.get("metric") == "pmi_course"}
    primary = clean.get("primary_dhon", {})
    raw = clean.get("raw", {})
    primary_auc = numeric(primary, "conditional_auc_B_high", "conditional_auc")
    raw_auc = numeric(raw, "conditional_auc_B_high", "conditional_auc")
    gap = numeric(primary, "median_gap_B_minus_A")
    if primary_auc is not None and raw_auc is not None:
        residual_auc = [numeric(row, "conditional_auc_B_high", "conditional_auc")
                        for name, row in clean.items() if name != "raw"]
        if len(residual_auc) > 1 and all(v == primary_auc for v in residual_auc):
            text = (f"Conditional B-high AUC {primary_auc:.3f} for each residual pipeline "
                    f"and {raw_auc:.3f} for raw vibration.")
        else:
            text = f"Conditional B-high AUC {primary_auc:.3f} for DHON residuals and {raw_auc:.3f} for raw vibration."
        if gap is not None:
            text += f" DHON median gap B−A {gap:.3f}."
        return text
    return "Development PMI was measured for the two provisional index groups. Their health labels remain unverified."


def robustness(stress_rows, pulse_rows, summary):
    primary = next((row for row in summary if label(row) == "primary_dhon"), {})
    worst = numeric(primary, "rpm_mse_worst")
    clean = numeric(primary, "mse_clean")
    period_corr = numeric(primary, "median_periodic_pulse_corr")
    n = numeric(primary, "periodic_pmi_cases")
    positive = numeric(primary, "periodic_pmi_positive")
    scenarios = {row.get("scenario", row.get("condition")) for row in stress_rows} - {None, ""}
    if worst is not None and clean is not None:
        rpm = f"For selected DHON, clean MSE {clean:.3f} rose to {worst:.3f} in the worst RPM condition."
    elif scenarios:
        rpm = f"{len(scenarios)} measured conditions. Compare clean and perturbed one-step error."
    else:
        rpm = "No RPM stress table is available for this run."
    if period_corr is not None and n is not None and positive is not None:
        pulse = f"Median correlation with periodic injection {period_corr:.2f}. Residual PMI rose in {positive:.0f} of {n:.0f} paired trials."
    else:
        pulse = ("Paired pulse-injection results quantify residual pulse distortion and periodicity."
                 if pulse_rows else "No paired pulse-injection table is available for this run.")
    return rpm, pulse


def short_ablation(metrics):
    entries = dict(metric_entries(metrics))
    fixed = entries.get("dhon_matched_fixed")
    learned = entries.get("primary_dhon")
    no_adapter = entries.get("primary_dhon_base")
    if fixed is None or learned is None:
        return "Use fixed-state and adapter-off controls to isolate learning and slow correction."
    improvement = fixed - learned
    msg = f"Joint learning reduced MSE by {improvement:.6f} versus matched fixed." if improvement >= 0 else (
        f"Joint learning raised MSE by {-improvement:.6f} versus matched fixed.")
    if no_adapter is not None:
        adapter_gain = no_adapter - learned
        msg += f" Adding the adapter {'reduced' if adapter_gain >= 0 else 'raised'} MSE by {abs(adapter_gain):.6f}."
    return msg


def short_timing(timings, environment):
    primary = next((row for row in timings if label(row) == "primary_dhon"), {})
    duration = numeric(primary, "median_s", "median_seconds")
    iqr = numeric(primary, "iqr_s")
    if duration is not None:
        variability = f" (IQR {iqr:.3f} s)" if iqr is not None else ""
        gpu = " CUDA unavailable." if environment.get("cuda_available") is False else ""
        return f"DHON CPU median {duration:.3f} s per 7.5 s record{variability}.{gpu}"
    return "A matched CPU timing table is unavailable for this run."


def append_backups(deck, data, is_final):
    s = deck.slide("How an LPV filter differs from ordinary AR", "Backup / baseline", 0,
        "Likely question. Where do the lagged innovations come from? The first-stage speed-conditioned AR model estimates innovations. The second stage adds lagged proxies as regressors. Read the protocol for when current versus previous speed is allowed. The paper result is supplied as external context.")
    backup(s, "AR uses constant coefficients. LPV lets them depend on a measured operating variable.",
           [("AR term", "Previous vibration samples multiplied by lag coefficients."),
            ("MA term", "Earlier errors from the first-stage proxy innovation model."),
            ("Sparsity", "Fit candidate support with an L1 objective, then refit selected coefficients.")],
           "Chen et al. (2025), doi:10.1016/j.ymssp.2024.112161 • §§2–3 and 5")
    s = deck.slide("How phase and causality meet", "Backup / oscillator", 0,
        "Likely question. Is previous vibration enough to predict phase? Shaft RPM advances the order phase by a causal extrapolation of one sample. Rotate past quadrature contributions into the estimated target phase. Since all inputs stop at t minus one, perturbing the future measurement cannot affect the current prediction.")
    backup(s, "Integrate RPM in revolutions and predict one phase step with the previous speed.",
           [("Relative phase", "Order × shaft rotation × 2π gives a physical carrier phase."),
            ("Bounded history", "The finite oscillator horizon determines its raw sample support."),
            ("State update", "At target arrival, its measured innovation can inform later predictions.")],
           "A causal predictor can coexist with offline diagnostic resampling of its completed residual.")
    s = deck.slide("Why mean squared error and PMI can disagree", "Backup / diagnostics", 0,
        "Likely question. Can a lower MSE reduce crack evidence? Yes. MSE rewards prediction of predictable components including possible repetitive faults. PMI rewards angular periodicity in the residual envelope. A paired pulse test measures whether the predicted perturbation consumes the injected pulse. Verified classification requires authoritative condition labels.")
    backup(s, "Use healthy-prediction and diagnostic evidence together.",
           [("MSE", "Mean squared one-step error across fixed development targets."),
            ("PMI", "Lagged envelope correlation divided by one minus the correlation."),
            ("Pulse test", "Measure what remains after injecting a known disturbance into the same record.")],
           "PMI uses the instructor reference convention and documented lag alternatives.")
    s = deck.slide("Order spectra offer a second view of mesh energy", "Backup / training EDA", 0,
        "Likely question. Why examine the spectrum against angle as well as time? Shaft angle ties the frequency axis to the mechanical cycle. Components proportional to speed align to a fixed order. Fixed physical frequencies shift in the order coordinate as speed changes. Read the synthesis alongside the spectrogram and the matched feature ablations.")
    figure_and_text(s, data.fig("eda_orders.png"), "Cycles per revolution",
                    "Inspect mesh orders and neighboring peaks. Their presence motivates predictor features whose incremental value is measured by held-out prediction.",
                    "Training order spectrum • Angular interpolation and spectral settings are recorded with the run")
    if not is_final:
        return
    s = deck.slide("How the guarded folds keep raw histories apart", "Backup / validation", 0,
        "Likely question. Why guard a block on both sides? A row after a held-out block could draw raw samples from inside it. Finite model support means that the history can be bounded and its guard checked. A measured past observation can still serve as known input during one-step scoring.")
    backup(s, "Keep training fit rows and their histories clear of held-out target blocks.",
           [("Targets", "The training held-out blocks are contiguous in sample index."),
            ("Histories", "Nested innovations and correction add history to the base feature maximum."),
            ("Scoring", "Use previously observed measured samples as inputs inside the held-out block.")],
           "Keep the test records sealed until model parameters and evaluation policy are frozen.")
    s = deck.slide("PMI reference lag and parity status", "Backup / MATLAB reference", 0,
        "Likely question. Why are there two PMI variants? MATLAB indexing begins at one, with ACF lag zero in its first position. Reading element T therefore accesses lag T minus one. Explicit zero-based lag T accesses one full turn. Exact end-to-end parity requires the instructor order-tracking helper and reference outputs.")
    backup(s, "Inspect the instructor implementation and record its indexing convention.",
           [("Reference mode", "Read the array at the instructor's effective ACF lag T−1."),
            ("Corrected mode", "Evaluate the mathematical lag T for one rotation."),
            ("Parity", "Check analytic fixtures locally. Use instructor gold outputs for reference parity.")],
           "Feature source supplied as fdt_featcal.m.")
    s = deck.slide("Inspect a representative prediction trace", "Backup / development traces", 0,
        "Likely question. What does the MSE average hide? Read the measured and predicted samples on a short interval. Aggregate statistics combine varying speed and changing vibration amplitude. The audit uses error by speed bin and paired blocks to expose that variation.")
    figure_and_text(s, data.fig("trace.png"), "Read individual errors",
                    "The sample segment illustrates one-step prediction with measured history. Use the common-mask table for quantitative comparison.",
                    "Supplied validation record • Matched, causal one-step predictions")
    s = deck.slide("Inspect the oscillator learning trajectory", "Backup / oscillator parameters", 0,
        "Likely question. How far do the carriers and memory scales move during learning? The run records order and decay updates by epoch. Small changes may matter locally and their contribution must be checked against the matched fixed-state model. Reproduction uses the exact parameters written to the checkpoint.")
    figure_and_text(s, data.fig("dhon_parameter_trajectory.png"), "Shift relative to the prior",
                    physical_parameters(data.json("search/dhon_selection.json"), data),
                    "Selected DHON • Training fit parameter trajectory")
    s = deck.slide("What the efficiency test measures", "Backup / benchmarking", 0,
        "Likely question. Can the batch rate be used as a streaming guarantee? The benchmark measures prediction after warm-up on a full record. Online deployment would need its own latency benchmark, state initialization policy and budgets. CPU and GPU results are only claimed for hardware and execution observed in the run.")
    backup(s, "Use equivalent inputs, feature construction and prediction when timing each model.",
           [("Wall time", "Benchmark complete inference after warm-up with repeated trials."),
            ("Variation", "Report median and variation on the observed hardware."),
            ("Online bound", "Measure sample latency separately for a streaming deployment.")],
           "Benchmark metadata records hardware, device and thread configuration.")


def comparable_text(value):
    return "".join(value.replace("\u00ad", "").split())


def render_check(presentations, report_dir, keep_previews):
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise RuntimeError("LibreOffice is required for --render-check")
    import fitz

    report_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    with tempfile.TemporaryDirectory(prefix="dhon_slides_") as tmp:
        tmpdir = Path(tmp)
        subprocess.run([
            soffice, f"-env:UserInstallation={(tmpdir / 'profile').as_uri()}",
            "--headless", "--convert-to", "pdf", "--outdir", str(tmpdir),
            *(str(presentation) for presentation in presentations),
        ], check=True, capture_output=True, timeout=90)
        for presentation in presentations:
            pdf = tmpdir / f"{presentation.stem}.pdf"
            reread = Presentation(presentation)
            with fitz.open(pdf) as doc:
                if len(doc) != len(reread.slides):
                    raise ValueError(f"Rendered page count differs for {presentation.name}")
                samples = []
                missing = []
                for index, page in enumerate(doc):
                    slide = reread.slides[index]
                    title = slide.shapes[1].text
                    rendered = comparable_text(page.get_text())
                    for shape in slide.shapes:
                        if shape.has_table:
                            texts = [cell.text for row in shape.table.rows for cell in row.cells]
                        elif shape.has_text_frame:
                            texts = [shape.text]
                        else:
                            continue
                        for expected in texts:
                            if not expected.strip() or expected.startswith("hₖ,ₜ"):
                                # PDF extraction reorders combining marks in the editable equation.
                                continue
                            if comparable_text(expected) not in rendered:
                                missing.append(f"page {index + 1:02d} • {expected[:120]}")
                    samples.append({"page": index + 1, "title": title, "characters": len(rendered)})
                details = report_dir / f"{presentation.stem}_render_review.txt"
                if missing:
                    details.write_text("\n".join(missing) + "\n", encoding="utf-8")
                else:
                    details.unlink(missing_ok=True)
                result[presentation.name] = {"page_count": len(doc), "pages": samples,
                                             "text_extraction_review_flags": len(missing)}
            if keep_previews:
                shutil.copyfile(pdf, report_dir / pdf.name)
    (report_dir / "slide_render_check.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run-id", default="development")
    group.add_argument("--run-dir", type=Path)
    parser.add_argument("--slide-data", type=Path)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "presentations")
    parser.add_argument("--render-check", action="store_true")
    parser.add_argument("--previews", action="store_true")
    args = parser.parse_args()
    data = RunData(args.run_dir or ROOT / "outputs" / "runs" / args.run_id,
                   json.loads(args.slide_data.read_text(encoding="utf-8")) if args.slide_data else {})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        progress(data).save(args.out_dir, expected=480),
        final(data).save(args.out_dir, expected=720),
    ]
    if args.render_check or args.previews:
        render_check(paths, args.out_dir, args.previews)
    for path in paths:
        print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path.name)


if __name__ == "__main__":
    main()
