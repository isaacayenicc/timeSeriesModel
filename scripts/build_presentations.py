"""Rebuild report and both editable decks from saved experiment outputs."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rotmach.slides_build import build_presentations, build_report, render_report_pdf

if __name__ == "__main__":
    print(build_report(ROOT))
    print(render_report_pdf(ROOT))
    for artifact in build_presentations(ROOT):
        print(artifact)
