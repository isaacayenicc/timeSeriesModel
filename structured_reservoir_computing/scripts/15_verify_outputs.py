"""Check source identity, numeric arrays, editable documents and executed notebooks."""
import argparse
from pathlib import Path
from zipfile import ZipFile
import json
import re

import numpy as np
from lxml import etree
import pandas as pd

from src_gearbox.comparison import BASE_NAMES, verify_axes
from src_gearbox.study_protocol import PROJECT, MASK_HASH, fingerprint, atomic_json


def check_office(root):
    report=root/"report"
    word=report/"SRC_Final_Course_Report.docx"
    meta={}
    with ZipFile(word) as f:
        assert f.testzip() is None
        d=etree.fromstring(f.read("word/document.xml"))
        ns={k:d.nsmap[k] for k in ("m","w")}
        math_count=len(d.xpath(".//m:oMath",namespaces=ns))
        assert math_count>=10
        text=" ".join(d.xpath(".//w:t/text()",namespaces=ns))
        assert "0.329973" in text and "0.317788" in text
        assert not re.search(r"TODO|FIXME|Figure \?\?|Error! Reference source",text)
        meta["editable_word_equations"]=math_count
        meta["word_tables"]=len(d.xpath(".//w:tbl",namespaces=ns))
    import fitz
    with fitz.open(report/"SRC_Final_Course_Report.pdf") as f:
        meta["word_pdf_pages"]=len(f)
        assert len(f)>8 and all(p.get_text().strip() for p in f)
    from pptx import Presentation
    for timing, label in ((8,"Progress"),(12,"Final")):
        path=root/f"presentations/SRC_{label}_Presentation_{timing}min.pptx"
        deck=Presentation(path)
        assert len(deck.slides)>=7 and all(slide.has_notes_slide for slide in deck.slides)
        with fitz.open(root/"presentations/rendered"/(path.stem+".pdf")) as f:
            assert len(f)==len(deck.slides)
        meta[label.lower()+"_slides"]=len(deck.slides)
    return meta


def check_notebooks(root):
    notebooks=sorted((root/"notebooks").glob("*.ipynb"))
    assert len(notebooks)==11
    results={}
    for p in notebooks:
        doc=json.loads(p.read_text(encoding="utf-8"))
        code=[c for c in doc["cells"] if c["cell_type"]=="code"]
        assert code and all(c.get("execution_count") is not None for c in code),p.name
        errors=[o for c in code for o in c["outputs"] if o["output_type"]=="error"]
        assert not errors,p.name
        results[p.name]=len(code)
    return results


def verify(root, numbers_only):
    metric=root/"results/metrics"
    pred=root/"results/predictions"
    select=json.loads((root/"results/optimization/selected_configs.json").read_text())
    trials=pd.read_csv(root/"results/optimization/trials.csv")
    done=trials.loc[trials.status.eq("DONE")]
    folds=pd.read_csv(root/"results/optimization/fold_scores.csv")
    assert len(folds)==3*len(done)
    source=json.loads((metric/"src_summary.json").read_text())
    a=select["selection"][0]
    assert a["trial_id"]==source["trial_id"] and a["model"]=="primary_pi_src"
    assert source["mask_sha256"]==MASK_HASH
    registry=json.loads((root/"checkpoints/src_registry.json").read_text())
    assert registry["target_mask_sha256"]==MASK_HASH
    checkpoint_index={}
    for row in registry["models"]:
        assert fingerprint(root/row["checkpoint"])==row["sha256"], row["model"]
        checkpoint_index[row["model"]]=dict(scope="fresh SRC",
                                           files={row["checkpoint"]:row["sha256"]},config=row["config"])
    baseline_dir=root/"checkpoints/baselines"
    baseline_options=pd.read_csv(metric/"baseline_configs.csv").set_index("model")
    for slug in BASE_NAMES:
        path=baseline_dir/f"{slug}.npz"
        digest=fingerprint(path)
        assert digest==baseline_options.at[slug,"checkpoint_sha256"]
        checkpoint_index[slug]=dict(scope="protected comparator replay",
            files={path.relative_to(root).as_posix():digest},config=json.loads(baseline_options.at[slug,"selected_config"]))
    archive_files={method:{p.relative_to(root).as_posix():fingerprint(p) for p in sorted(baseline_dir.glob(stem+".*"))}
                   for method,stem in [("archived_src_corrected","archived_src_optimized"),
                                       ("archived_src_base","archived_src_optimized"),
                                       ("archived_src_local_corrected","archived_src_local")]}
    checkpoint_index.update({name:dict(scope="archive replay",files=files) for name,files in archive_files.items()})
    checkpoint_index.update({name:dict(scope="archive forecast only",files={})
                            for name in ("archived_src_earlier","archived_src_earlier_corrected")})
    atomic_json(root/"checkpoints/registry.json",dict(target_mask_sha256=MASK_HASH,models=checkpoint_index))
    with np.load(pred/"src_sigvalid.npz",allow_pickle=False) as new, \
         np.load(pred/"baselines_sigvalid.npz",allow_pickle=False) as base, \
         np.load(pred/"archived_src_sigvalid.npz",allow_pickle=False) as old:
        verify_axes(new,base)
        verify_axes(new,old)
        comparison=pd.read_csv(metric/"all_model_comparison.csv")
        assert set(BASE_NAMES)<=set(comparison.model)
        assert len(comparison.model)==comparison.model.nunique()
        targets=new["actual"]
        domains=[new,base,old]
        max_error=0
        for row in comparison.itertuples():
            forecast=next(data[row.model] for data in domains if row.model in data)
            error=abs(np.mean((targets-forecast)**2)-row.mse)
            assert error<=1e-12 and row.n==len(targets) and row.mask_sha256==MASK_HASH,row.model
            max_error=max(max_error,error)
    replay=json.loads((root/"results/manifests/src_prediction_verification.json").read_text())
    causal_max=max(d["max_through_target_change"] for d in replay["tests"])
    assert causal_max<1e-8
    detached=json.loads((root/"results/frozen/availability.json").read_text())
    diagnostics=json.loads((root/"results/frozen/fault2_availability.json").read_text())
    assert "status" in detached and "status" in diagnostics
    values=dict(optimization_trials=len(trials),completed_trials=len(done),guarded_rows=len(folds),
        scored_comparison_rows=len(comparison),required_comparators=len(BASE_NAMES),src_checkpoints=len(registry["models"]),
        recalculation_max_abs=max_error,causality_max_abs=causal_max,mask=MASK_HASH,
        independent_status=detached["status"],blind_fault_status=diagnostics["status"])
    if not numbers_only:
        values["office"]=check_office(root)
        values["notebooks"]=check_notebooks(root)
        companion=root/"docs/companion_guide.md"
        assert companion.is_file()
        values["companion_words"]=len(companion.read_text().split())
        assert (root/"results/fault/fault_group_summary.csv").is_file() or (root/"results/fault/fault_summary.csv").is_file()
    atomic_json(root/"results/manifests/output_verification.json",values)
    print(json.dumps(values,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-dir",type=Path,default=PROJECT)
    p.add_argument("--numbers-only",action="store_true")
    args=p.parse_args()
    verify(args.project_dir,args.numbers_only)
