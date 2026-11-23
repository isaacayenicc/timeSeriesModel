"""Verify native Word structures, numerical tables and the matching PDF."""
from argparse import ArgumentParser
import csv
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
from zipfile import ZipFile

import fitz
from lxml import etree
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = "MECG7780_PI_SRC_Final_Course_Report"
NS = {
    "w":"http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m":"http://schemas.openxmlformats.org/officeDocument/2006/math",
    "wp":"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a":"http://schemas.openxmlformats.org/drawingml/2006/main",
    "r":"http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
SENSITIVE = re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|(?:AKIA|ASIA)[A-Z0-9]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----")


def x(node,path):
    return node.xpath(path,namespaces=NS)


def inspect(folder):
    with (ROOT/"evidence/figure_manifest.csv").open(encoding="utf-8") as f:
        figures = list(csv.DictReader(f))
    for item in figures:
        original = ROOT/item["asset"]
        assert sha256(original.read_bytes()).hexdigest()==item["sha256"],item["asset"]

    expected = pd.read_csv(ROOT/"assets/tables/all_model_comparison.csv")
    bibliography = json.loads((ROOT/"evidence/bibliography_index.json").read_text())
    page_map = json.loads((folder/"qa/page_map.json").read_text())
    captions = []
    raster_dpi = []
    equation_labels = []
    data_tables = []
    with ZipFile(folder/(DOCUMENT+".docx")) as z:
        assert z.testzip() is None and not any(k.endswith("vbaProject.bin") for k in z.namelist())
        assert not any(SENSITIVE.search(z.read(k)) for k in z.namelist() if k.endswith((".xml",".rels")))
        d = etree.fromstring(z.read("word/document.xml"))
        words = " ".join(x(d,"//w:t/text()"))
        assert not re.search(r"\b(?:TODO|FIXME|TBD|PLACEHOLDER)\b|\[[@?]|Error! Reference source|WORD_TOC_MARKER|COURSE-TITLE-MARKER",words)
        assert not x(d,"//w:commentRangeStart | //w:ins | //w:del")
        math_count = len(x(d,"//m:oMath"))
        assert math_count>=100
        for tab in x(d,"//w:tbl"):
            rows = [["".join(x(c,".//w:t/text()")) for c in x(row,"./w:tc")] for row in x(tab,"./w:tr")]
            if x(tab,".//m:oMathPara"):
                assert len(rows)==1 and re.fullmatch(r"\(\d+\.\d+\)",rows[0][-1])
                equation_labels.append(rows[0][-1])
                continue
            assert len(x(tab,"./w:tr[1]/w:trPr/w:tblHeader"))==1
            data_tables.append(rows)
        assert len(equation_labels)>=18 and len(set(equation_labels))==len(equation_labels)

        ranking = next(t for t in data_tables if t[0]==["Model","Evidence","MSE","RMSE","MAE","CPU (ms)"])
        assert len(ranking)==28 and len(expected)==27
        for row,reference in zip(ranking[1:],expected.itertuples()):
            assert row[0]==reference.display and row[2:5]==[f"{reference.mse:.6f}",f"{reference.rmse:.6f}",f"{reference.mae:.6f}"]
        anchors=x(d,"//w:hyperlink[starts-with(@w:anchor,'report_')]/@w:anchor")
        toc=x(d,"//w:fldSimple[starts-with(@w:instr,' PAGEREF ')]")
        cached=[int("".join(x(p,".//w:t/text()"))) for p in toc]
        assert len(anchors)==len(cached)==len(page_map)
        assert cached==list(page_map.values())
        rels = {n.get("Id"):n.get("Target") for n in etree.fromstring(z.read("word/_rels/document.xml.rels"))}
        for node in x(d,"//wp:inline"):
            desc=x(node,"./wp:docPr/@descr")[0]
            found=re.match(r"Figure (\d+\.\d+)\.",desc)
            assert found,desc
            captions.append(found[1])
            link=x(node,".//a:blip/@r:embed")[0]
            with Image.open(BytesIO(z.read("word/"+rels[link]))) as im:
                assert min(im.size)>500
                inch=int(x(node,"./wp:extent/@cx")[0])/914400
                raster_dpi.append(round(im.width/inch))
        assert set(captions)=={f["figure"] for f in figures} and len(captions)==len(figures)
        assert min(raster_dpi)>=300
        assert "0.308688" in words and "0.329973" in words and "0.317788" in words
        # Multiple front-matter, body, and citation hyperlinks are expected.
        assert len(x(d,"//w:hyperlink"))>len(toc)
        refs = x(d,"//w:bookmarkStart[starts-with(@w:name,'ref-')]/@w:name")
        assert len(refs)==bibliography["count"],len(refs)

    image_pages = {}
    pages = []
    with fitz.open(folder/(DOCUMENT+".pdf")) as d:
        for idx,page in enumerate(d,1):
            body = []
            actual_images = page.get_image_info()
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines",[]):
                    spans = line.get("spans",[])
                    text = "".join(span["text"] for span in spans).strip()
                    if not text:
                        continue
                    a,b,c,e=line["bbox"]
                    assert a >= 35 and c <= page.rect.width-30,(idx,text,line["bbox"])
                    if 57 < b < page.rect.height-39:
                        body.append(text)
                    hit=re.match(r"Figure (\d+\.\d+)\.",text)
                    if hit and spans[0]["size"] < 10.4:
                        image_pages[hit[1]]=idx
            for image in actual_images:
                a,b,c,e=image["bbox"]
                assert a>=59 and c<=page.rect.width-58 and b>=57 and e<=page.rect.height-62,(idx,image["bbox"])
            pages.append(dict(page=idx,body_characters=len(" ".join(body)),images=len(actual_images)))
    assert len(image_pages)==len(figures)
    assert all(pages[page-1]["images"] for page in image_pages.values())
    assert sum(page["images"] for page in pages)==len(figures)
    assert all(x["body_characters"]>135 or x["images"] for x in pages)
    audit = {
        "pages":len(pages),"native_math_objects":math_count,"numbered_display_equations":len(equation_labels),
        "editable_data_tables":len(data_tables),"editable_comparison_rows":len(ranking)-1,
        "native_navigation_fields":len(toc),"figure_alt_text_count":len(captions),"minimum_effective_dpi":min(raster_dpi),
        "bibliography_entries":bibliography["count"],"verified_bibliography_dois":bibliography["doi_count"],
        "figure_pages":image_pages,"rendered_inventory":pages,
    }
    (folder/"qa/document_structure.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    print({key:value for key,value in audit.items() if key not in {"rendered_inventory","figure_pages"}})


if __name__ == "__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,default=ROOT)
    inspect(p.parse_args().output_dir.resolve())
