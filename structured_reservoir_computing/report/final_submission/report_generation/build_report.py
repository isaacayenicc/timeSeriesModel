"""Apply Word typography, navigable contents and equation numbering to the report."""
from copy import deepcopy
from pathlib import Path
import json
import re
import argparse
from shutil import copy2
import subprocess
from tempfile import TemporaryDirectory

import yaml
import fitz

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

TOOLS_DIR=Path(__file__).resolve().parent
ROOT=TOOLS_DIR.parent
CFG=yaml.safe_load((TOOLS_DIR/"report_config.yaml").read_text(encoding="utf-8"))
D=CFG["document"]
TOOLS=CFG["tools"]
NAVY=D["ink"]
TEAL=D["accent"]
SLATE="455666"
GRAY="D5E0E6"
PALE="EDF3F6"
CONTENT=D["page_width_in"] - 2*D["left_right_margin_in"]


def make(tag, **attributes):
    value = OxmlElement(tag)
    for k,v in attributes.items():
        value.set(qn(k.replace("_",":",1)), str(v))
    return value


def rgb(value):
    return RGBColor.from_string(value)


def style(doc, name, size, family, color=NAVY, bold=False, after=6, before=0, line=1.22):
    if name not in doc.styles:
        doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
    s=doc.styles[name]
    s.font.name=family
    s.font.size=Pt(size)
    s.font.color.rgb=rgb(color)
    s.font.bold=bold
    pf=s.paragraph_format
    pf.space_after=Pt(after)
    pf.space_before=Pt(before)
    pf.line_spacing=line
    pf.widow_control=True
    return s


def simple_field(node, code, value):
    fld=make("w:fldSimple", w_instr=code)
    r=make("w:r")
    t=make("w:t")
    t.text=str(value)
    r.append(t)
    fld.append(r)
    node.append(fld)


def bookmark(p,name,i):
    p._p.insert(1 if p._p.pPr is not None else 0,make("w:bookmarkStart",w_id=i,w_name=name))
    p._p.append(make("w:bookmarkEnd",w_id=i))


def toc_link(p,key,label):
    link=make("w:hyperlink",w_anchor=key,w_history=1)
    run=make("w:r")
    rpr=make("w:rPr")
    rpr.append(make("w:color",w_val=NAVY))
    run.append(rpr)
    content=make("w:t")
    content.text=label
    run.append(content)
    link.append(run)
    p._p.append(link)


def patch_table(t, idx):
    head = tuple(x.text.replace("\n"," ") for x in t.rows[0].cells)
    specs = {
        "Abbreviation":[.20,.80],
        "Question":[.27,.35,.38],
        "File or corpus":[.22,.30,.48],
        "Physical observation":[.31,.33,.36],
        "Method":[.26,.19,.18,.37],
        "Model or signal":[.32,.136,.136,.136,.136,.136],
        "Intervention":[.30,.18,.25,.27],
        "Question examined":[.26,.26,.48],
        "Locked choice":[.14,.22,.11,.115,.105,.115,.075,.12],
        "Input":[.19,.24,.57],
        "Feature family":[.17,.285,.30,.245],
        "Selected component":[.42,.58],
        "Guarded fold":[.16,.28,.28,.28],
        "Physical representation":[.29,.40,.31],
        "Learning and adaptation setting":[.33,.37,.30],
        "Evidence from guarded folds":[.39,.16,.45],
        "Design item":[.25,.75],
        "Prior conventional model":[.20,.21,.20,.20,.19],
        "Original supplemental control":[.33,.48,.19],
        "Control":[.37,.20,.20,.23],
        "Signal":[.265,.125,.125,.125,.20,.16],
        "Evidence":[.27,.73],
        "Research hypothesis":[.29,.27,.44],
        "Predictor":[.29,.15,.15,.15,.13,.13] if len(head)==6 else [.37,.21,.21,.21],
        "Final assessment criterion":[.25,.13,.62],
    }
    if head[0]=="Model":
        widths=( [.265]+[.1225]*6 if len(head)==7 else
                 [.36]+[.64/3]*3 if len(head)==4 else
                 [.30,.70] if len(head)==2 else [.29]+[.71/(len(head)-1)]*(len(head)-1))
    else:
        widths=specs.get(head[0],[1/len(head)]*len(head))
        if len(widths)!=len(head):
            widths=[.30]+[.70/(len(head)-1)]*(len(head)-1) if len(head)>1 else [1.]
    assert len(widths)==len(t.columns),(head,widths)
    t.alignment=WD_TABLE_ALIGNMENT.CENTER
    t.autofit=False
    for col,fraction in zip(t.columns,widths):
        col.width=Inches(CONTENT*fraction)
    pr=t._tbl.tblPr
    tw=pr.find(qn("w:tblW"))
    if tw is not None:
        tw.set(qn("w:type"),"dxa")
        tw.set(qn("w:w"),str(round(CONTENT*1440)))
    borders=make("w:tblBorders")
    for edge in ("top","left","bottom","right","insideH","insideV"):
        borders.append(make("w:"+edge,w_val="single",w_sz=4,w_color=GRAY))
    pr.append(borders)
    margins=make("w:tblCellMar")
    for edge in ("top","bottom","left","right"):
        margins.append(make("w:"+edge,w_w=79 if edge in {"left","right"} else 49,w_type="dxa"))
    pr.append(margins)
    for ri,row in enumerate(t.rows):
        rp=row._tr.get_or_add_trPr()
        for tag in ("w:cantSplit","w:tblHeader"):
            for node in rp.findall(qn(tag)):
                rp.remove(node)
        rp.append(make("w:cantSplit"))
        if ri==0:
            rp.append(make("w:tblHeader",w_val=1))
        for j, (c,frac) in enumerate(zip(row.cells,widths)):
            c.width=Inches(CONTENT*frac)
            c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cp=c._tc.get_or_add_tcPr()
            sh=make("w:shd",w_fill="DDE9EC" if ri==0 else (PALE if ri%2==0 else "FFFFFF"))
            cp.append(sh)
            for p in c.paragraphs:
                p.style="Table Header" if ri==0 else "Table Body"
                pf=p.paragraph_format
                pf.keep_with_next=ri==0
                pf.keep_together=True
                if j and re.fullmatch(r"[-+−–—0-9,.%/ ]+",p.text.strip()) and p.text.strip():
                    p.alignment=WD_ALIGN_PARAGRAPH.RIGHT
                else:
                    p.alignment=WD_ALIGN_PARAGRAPH.LEFT
                for r in p.runs:
                    r.font.color.rgb=rgb(NAVY)
                    r.bold=ri==0
                    if r.font.name in {"Courier New","Consolas"}:
                        r.font.size=Pt(8.7)


def numbered_equation(doc,p,label):
    math=p._p.find(qn("m:oMathPara"))
    if math is None:
        return
    math=deepcopy(math)
    table=doc.add_table(rows=1,cols=3)
    table.alignment=WD_TABLE_ALIGNMENT.CENTER
    table.autofit=False
    widths=(.57,CONTENT-1.14,.57)
    for col,c,width in zip(table.columns,table.rows[0].cells,widths):
        col.width=c.width=Inches(width)
        c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
    pr=table._tbl.tblPr
    margins=make("w:tblCellMar")
    for edge in ("top","bottom","left","right"):
        margins.append(make("w:"+edge,w_w=15 if edge in {"top","bottom"} else 0,w_type="dxa"))
    pr.append(margins)
    edges=make("w:tblBorders")
    for edge in ("top","left","bottom","right","insideH","insideV"):
        edges.append(make("w:"+edge,w_val="nil"))
    pr.append(edges)
    table.rows[0]._tr.get_or_add_trPr().append(make("w:cantSplit"))
    middle=table.cell(0,1).paragraphs[0]
    middle.style="Equation"
    middle.alignment=WD_ALIGN_PARAGRAPH.CENTER
    middle._p.append(math)
    ep=table.cell(0,2).paragraphs[0]
    ep.style="Report Equation Number"
    ep.add_run(f"({label})")
    table.cell(0,0).paragraphs[0].style="Report Equation Number"
    p._p.addnext(table._tbl)
    p._p.getparent().remove(p._p)


def build_document(intermediate, output, pages):
    doc=Document(intermediate)
    d=doc.sections[0]
    d.page_width,d.page_height=Inches(D["page_width_in"]),Inches(D["page_height_in"])
    d.left_margin=d.right_margin=Inches(D["left_right_margin_in"])
    d.top_margin,d.bottom_margin=Inches(D["top_margin_in"]),Inches(D["bottom_margin_in"])
    d.header_distance,d.footer_distance=Inches(.33),Inches(.36)
    d.different_first_page_header_footer=True
    mathpr=doc.settings._element.find(qn("m:mathPr"))
    if mathpr is not None:
        f=mathpr.find(qn("m:mathFont"))
        if f is not None:
            f.set(qn("m:val"),"Cambria Math")
    for key in ("Normal","Body Text","First Paragraph","Compact"):
        s=style(doc,key,D["body_size_pt"],D["body_font"],color="20262C",after=6.1,line=D["body_line_spacing"])
        s.paragraph_format.first_line_indent=Pt(0)
    for key,font,pre,after in [("Heading 1",21,0,12),("Heading 2",13.4,11,6),("Heading 3",12,10,4)]:
        s=style(doc,key,font,D["heading_font"],bold=True,before=pre,after=after,line=1.12)
        s.paragraph_format.keep_with_next=s.paragraph_format.keep_together=True
    style(doc,"Report Header",8.2,"Arial",color=SLATE,after=0,line=1.05)
    st=style(doc,"Caption",9.6,"Arial",color=SLATE,after=8,before=3,line=1.13)
    st.paragraph_format.keep_together=True
    style(doc,"Table Body",9.5,"Arial",color=NAVY,after=0,line=1.12)
    style(doc,"Table Header",9.5,"Arial",color=NAVY,bold=True,after=0,line=1.12)
    st=style(doc,"Equation",11.2,"Times New Roman",after=3,before=3,line=1.12)
    st.paragraph_format.keep_together=True
    st=style(doc,"Report Equation Number",9.4,"Arial",color=SLATE,after=0,line=1.12)
    st.paragraph_format.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    st=style(doc,"Report Contents",10.6,"Arial",color=NAVY,after=3,line=1.10)
    st.paragraph_format.tab_stops.add_tab_stop(Inches(CONTENT),WD_TAB_ALIGNMENT.RIGHT,WD_TAB_LEADER.DOTS)
    st=style(doc,"References",10.8,"Times New Roman",color="20262C",after=4.3,line=1.13)
    st.paragraph_format.left_indent=Inches(.28)
    st.paragraph_format.first_line_indent=Inches(-.28)
    st.paragraph_format.keep_together=True
    style(doc,"Source Code",9,"Courier New",color="23374A",after=6,before=4,line=1.08)

    hp=d.header.paragraphs[0]
    hp.style="Report Header"
    hp.paragraph_format.tab_stops.add_tab_stop(Inches(CONTENT),WD_TAB_ALIGNMENT.RIGHT)
    hp.add_run(D["course_header"]).bold=True
    hp.add_run("\t"+D["short_header"])
    border=make("w:pBdr")
    border.append(make("w:bottom",w_val="single",w_sz=7,w_space=5,w_color=GRAY))
    hp._p.get_or_add_pPr().append(border)
    fp=d.footer.paragraphs[0]
    fp.style="Report Header"
    fp.paragraph_format.tab_stops.add_tab_stop(Inches(CONTENT),WD_TAB_ALIGNMENT.RIGHT)
    fp.add_run("FINAL COURSE PROJECT REPORT\t")
    simple_field(fp._p," PAGE ",2)

    for shape in doc.inline_shapes:
        ratio=shape.height/shape.width
        # Use the configured text block rather than Pandoc's narrower default page grid.
        width=Inches(min(6.4,CONTENT))
        shape.width=width
        shape.height=round(width*ratio)
    for i,t in enumerate(doc.tables):
        patch_table(t,i)

    vals=list(doc.paragraphs)
    main=[]
    eq_seq={}
    section="0"
    bibliography=False
    front=True
    pending_break=False
    for p in vals:
        t=p.text
        if t=="WORD_PAGE_BREAK" or t=="COURSE-TITLE-MARKER":
            if t=="WORD_PAGE_BREAK":
                pending_break=True
                front=False
            p._p.getparent().remove(p._p)
            continue
        if front:
            if t.startswith("MECG 7780"):
                p.style=style(doc,"Cover Kicker",11,"Arial",color=TEAL,bold=True,before=22,after=18,line=1.0)
            elif t==D["title"]:
                p.style=style(doc,"Title",32,D["heading_font"],bold=True,after=16,line=1.14)
                p.paragraph_format.keep_with_next=True
            elif t.startswith("Causal one-step prediction"):
                p.style=style(doc,"Subtitle",15.5,D["heading_font"],color=SLATE,after=34,line=1.22)
            elif t.startswith("Final course project"):
                p.style=style(doc,"Cover Class",11,"Arial",color=TEAL,bold=True,after=11,line=1.12)
            elif t.startswith("Submission package"):
                p.style="Caption"
                p.paragraph_format.space_before=Pt(17)
            elif t.startswith("Project release"):
                p.style="Caption"
        if p.style.name=="Heading 1":
            if pending_break or t=="References" or re.match(r"(?:1\.|5\.|Appendix)",t):
                p.paragraph_format.page_break_before=True
                pending_break=False
            bibliography=t=="References"
            if t!="Contents":
                key=f"report_{len(main)+1}"
                bookmark(p,key,8000+len(main))
                main.append((t,key))
                prefix = re.match(r"^(\d+)\.",t)
                if prefix:
                    section=prefix[1]
        elif bibliography and t:
            p.style="References"
        elif p.style.name=="Image Caption":
            p.style="Caption"
            p.alignment=WD_ALIGN_PARAGRAPH.LEFT
        elif p._p.xpath(".//w:drawing"):
            p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            pf=p.paragraph_format
            pf.keep_with_next=True
            pf.keep_together=True
            pf.space_before=Pt(5)
            pf.space_after=Pt(0)
        elif re.match(r"Table (?:\d+|[A-Z])\.\d+\.",t):
            p.style="Caption"
            p.paragraph_format.keep_with_next=True
        if p._p.find(qn("m:oMathPara")) is not None:
            eq_seq[section]=eq_seq.get(section,0)+1
            numbered_equation(doc,p,f"{section}.{eq_seq[section]}")

    for p in doc.paragraphs:
        if p.text=="WORD_TOC_MARKER":
            for title,key in main:
                entry=p.insert_paragraph_before(style="Report Contents")
                toc_link(entry,key,title)
                entry.add_run("\t")
                simple_field(entry._p,f" PAGEREF {key} \\h ",pages.get(title,"00"))
            p._p.getparent().remove(p._p)
            break

    props=doc.core_properties
    props.title=D["title"]
    props.subject=D["subject"]
    props.author=""
    props.language="en-CA"
    props.keywords="SRC, gearbox vibration, causal forecasting, model comparison, residual monitoring"
    props.comments=""
    styles_node=doc.styles.element
    lang=styles_node.find(".//"+qn("w:lang"))
    if lang is not None:
        lang.set(qn("w:val"),"en-CA")
    # Open XML prescribes separator before closing character. LibreOffice
    # otherwise substitutes a default closing parenthesis for some symbols.
    order={name:i for i,name in enumerate(("begChr","sepChr","endChr","grow","shp","ctrlPr"))}
    for delimiter in doc.element.xpath(".//m:dPr"):
        delimiter[:]=sorted(delimiter,key=lambda child:order.get(child.tag.rsplit("}",1)[-1],99))
    doc.save(output)
    return main


def render(docx, profile, destination):
    output=destination/(docx.stem+".pdf")
    if output.exists():
        output.unlink()
    out=subprocess.run([TOOLS["libreoffice"],f"-env:UserInstallation={profile.as_uri()}",
                        "--headless","--convert-to","pdf","--outdir",str(destination),str(docx)],
                       cwd=ROOT,capture_output=True,text=True,check=True)
    if not output.is_file():
        raise RuntimeError("LibreOffice produced no inspection PDF.\n"+out.stdout+out.stderr)
    return output


def page_map(pdf, headings):
    expected={title for title,_ in headings}
    result={}
    with fitz.open(pdf) as d:
        for idx,p in enumerate(d,1):
            chunks=[]
            for b in p.get_text("dict")["blocks"]:
                for line in b.get("lines",[]):
                    block=[s["text"] for s in line.get("spans",[]) if s["size"]>=18 and s["bbox"][1]>45]
                    if block:
                        chunks.append("".join(block))
            line=re.sub(r"\s+"," "," ".join(chunks))
            for title in expected:
                if title in line and title not in result:
                    result[title]=idx
    if expected!=set(result):
        raise RuntimeError(f"Missing printed report headings: {sorted(expected-set(result))}")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=ROOT)
    parser.add_argument("--project-root",type=Path,help="Optional recorded source-study root for evidence audits")
    parser.add_argument("--evidence-mode",choices=["verified"],default="verified")
    parser.add_argument("--pages",action="store_true",help="Generate full-size PNG inspection pages.")
    args=parser.parse_args()
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    docx=out/(D["basename"]+".docx")
    with TemporaryDirectory(prefix="src_report_") as td:
        tmp=Path(td).resolve()
        base=tmp/"pandoc_base.docx"
        source=ROOT/D["source"]
        if not source.is_file():
            raise FileNotFoundError(source)
        result=subprocess.run([TOOLS["pandoc"],str(source),"--from=markdown+tex_math_dollars+tex_math_single_backslash",
                        "--resource-path",str(ROOT),"--citeproc","--bibliography",str(ROOT/"evidence/references.bib"),
                        "--csl",str(TOOLS_DIR/"ieee.csl"),"--metadata","link-citations=true",
                        "-o",str(base)],cwd=ROOT,capture_output=True,text=True,check=True)
        if result.stderr.strip():
            raise RuntimeError("Pandoc emitted warnings: "+result.stderr)
        pages={}
        for attempt in range(4):
            heads=build_document(base,docx,pages)
            pdf=render(docx,tmp/"office_profile",tmp)
            fresh=page_map(pdf,heads)
            if fresh==pages:
                break
            pages=fresh
        else:
            raise RuntimeError("Contents fields and rendered pagination did not converge")
        review=out/(D["basename"]+".pdf")
        copy2(pdf,review)
    qa=out/"qa"
    qa.mkdir(exist_ok=True)
    (qa/"page_map.json").write_text(json.dumps(pages,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with fitz.open(review) as pdf:
        print(f"Rendered {len(pdf)} pages; {len(pages)} verified contents entries.")
        if args.pages:
            renders=out/"renders"
            renders.mkdir(exist_ok=True)
            for prior in renders.glob("page-*.png"):
                prior.unlink()
            for i,page in enumerate(pdf,1):
                page.get_pixmap(matrix=fitz.Matrix(2.15,2.15),alpha=False).save(renders/f"page-{i:02}.png")


if __name__=="__main__":
    main()
