"""Regenerate an editable comparative addendum to the original two presentations."""
import argparse
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

ROOT=Path(__file__).resolve().parents[1]


def build(out):
    m=pd.read_csv(out/"comparison_metrics.csv")
    p=Presentation()
    p.slide_width=Inches(13.33)
    p.slide_height=Inches(7.5)
    for name, subtitle, rows, picture in [
        ("Gearbox vibration model comparison", "MECG 7780 • Expanded study companion",
         ["21 requested model families and three DHON references",
          "Causal, one-step evaluation on 22,208 shared development targets"], None),
        ("Comparison protocol", "Fit and select with the course training acquisition",
         ["Three guarded training folds and a uniform target mask",
          "RPM and vibration available through the preceding sample",
          "Development acquisition reserved for final scoring"], None),
        ("Measured accuracy", "Course development MSE on standardized vibration",
         [], "development_mse"),
        ("Accuracy and CPU cost", "One CPU thread, seven full-record calls",
         [], "accuracy_cost"),
        ("What changed with model capacity", "Prediction accuracy with one short training recording",
         [f"Selected DHON • {m.loc[m.model=='primary_dhon','mse'].iloc[0]:.6f}",
          f"Physics-aware Linear Regression • {m.loc[m.model=='linear','mse'].iloc[0]:.6f}",
          "A causal adapter contributed the largest measured gain within the DHON family",
          "The stronger archived forecasts remain ahead on this record"], None),
        ("Diagnostics and independent evaluation", "Residual periodicity, RPM stress and matched injection",
         ["All available candidate records audited and two overlapping acquisitions excluded",
          "A/B groups are provisional record indices",
          "Blind test files and authoritative labels were unavailable"], None),
    ]:
        slide=p.slides.add_slide(p.slide_layouts[6])
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb=RGBColor.from_string("F7F9FB")

        def put(text,x,y,w,h,size,color,bold=False):
            box=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
            paragraph=box.text_frame.paragraphs[0]
            paragraph.text=text
            paragraph.font.name="Aptos"
            paragraph.font.size=Pt(size)
            paragraph.font.bold=bold
            paragraph.font.color.rgb=RGBColor.from_string(color)

        put(name,.7,.45,12,1,33,"152E49",True)
        put(subtitle,.73,1.38,12,.5,19,"35645F")
        for line,text in enumerate(rows):
            put(text,.76,2.30+line*.74,11.7,.65,22,"243242")
        if picture:
            from PIL import Image
            path=out/"figures"/f"{picture}.png"
            with Image.open(path) as im:
                ratio=im.width/im.height
            height=min(5.0,11.8/ratio)
            width=ratio*height
            slide.shapes.add_picture(str(path),Inches((13.33-width)/2), Inches(2.02),
                                     width=Inches(width),height=Inches(height))
        slide.notes_slide.notes_text_frame.text=subtitle+". "+" ".join(rows)
    p.save(ROOT/"presentations/expanded_comparison_addendum.pptx")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("outputs/expanded"))
    build(parser.parse_args().out)
