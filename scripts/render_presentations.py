"""Optional LibreOffice/Poppler export and contact-sheet inspection aid.

Requires soffice and pdftoppm on PATH. Does not substitute graphical export for
the editable PPTX files. Run after scripts/build_presentations.py.
"""
from pathlib import Path
import subprocess
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    folder = ROOT / "presentations"
    output = folder / "exported_pdf"
    output.mkdir(exist_ok=True)
    for stem in ("Progress_Presentation_8min", "Final_Presentation_12min"):
        subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(output), str(folder / (stem + ".pptx"))], check=True)
        images = output / (stem + "_slides")
        images.mkdir(exist_ok=True)
        subprocess.run(["pdftoppm", "-scale-to", "1200", "-png", str(output / (stem + ".pdf")), str(images / "slide")], check=True)
        paths = sorted(images.glob("slide-*.png"))
        sheet = Image.new("RGB", (1800, 380 * ((len(paths)+2)//3)), "#cccccc")
        for i, p in enumerate(paths):
            im = Image.open(p).convert("RGB"); im.thumbnail((580, 345))
            x, y = (i % 3) * 600 + 10, (i // 3) * 380 + 25
            sheet.paste(im, (x, y)); ImageDraw.Draw(sheet).text((x, y-20), f"Slide {i+1}", fill="black")
        sheet.save(output / (stem + "_contact_sheet.jpg"))
        print(output / (stem + ".pdf"))
