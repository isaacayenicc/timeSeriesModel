"""Create a checked local reproduction ZIP without raw course inputs or caches."""
import argparse
import csv
from hashlib import sha256
from pathlib import Path
import re
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
CREDENTIALS=re.compile(rb"gh[pousr]_[a-zA-Z0-9]{20,}|github_pat_[a-zA-Z0-9_]{20,}|(?:AKIA|ASIA)[A-Z0-9]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----")


def include(item):
    names=item.parts
    if any(s in {".venv",".git","__pycache__",".pytest_cache",".ipynb_checkpoints","reproduction"} or s.endswith(".egg-info") for s in names):
        return False
    if names[0]=="artifacts" or names[-1]=="SHA256SUMS.txt" or (names[0]=="data" and names[1] in {"raw","processed"}):
        return False
    if names[:2]==("inputs","reference_archives") or names[-2:] == ("manifests","artifact_inventory.csv"):
        return False
    if item.suffix.lower() in {".mat",".zip",".pyc",".tmp"}:
        return False
    if (names[:2] in {("report","qa"),("presentations","rendered")} and item.suffix==".png"):
        return False
    return True


def package(root):
    known = root/"results/manifests/artifact_inventory.csv"
    all_paths = []
    for path in root.rglob("*"):
        rel=path.relative_to(root)
        if path.is_file() and include(rel):
            if path.is_symlink():
                raise ValueError(f"Package requires a distributable file, not a symlink: {rel}")
            all_paths.append(rel)
    all_paths.sort()
    manifest=[]
    for rel in all_paths:
        raw=(root/rel).read_bytes()
        if rel.suffix in {".py",".yaml",".md",".json",".csv",".txt",".toml",".cff",".log",".ipynb"}:
            assert not CREDENTIALS.search(raw),rel
        manifest.append((rel.as_posix(),len(raw),sha256(raw).hexdigest()))
    known.parent.mkdir(parents=True,exist_ok=True)
    with known.open("w",encoding="utf-8",newline="") as out:
        writer=csv.writer(out)
        writer.writerow(["file","bytes","sha256"])
        writer.writerows(manifest)
    raw=known.read_bytes()
    manifest.append((known.relative_to(root).as_posix(),len(raw),sha256(raw).hexdigest()))
    all_paths.append(known.relative_to(root))
    destination=root/"artifacts"
    destination.mkdir(exist_ok=True)
    output=destination/"SRC_Local_Reproduction_Package.zip"
    staging=output.with_suffix(".zip.tmp")
    prefix="structured_reservoir_computing"
    with ZipFile(staging,"w",ZIP_DEFLATED,compresslevel=7) as z:
        for path in all_paths:
            z.write(root/path,f"{prefix}/{path.as_posix()}")
        z.writestr(f"{prefix}/SHA256SUMS.txt","".join(f"{h}  {p}\n" for p,_,h in manifest))
    with ZipFile(staging) as z:
        assert z.testzip() is None
        for p,n,h in manifest:
            raw=z.read(f"{prefix}/{p}")
            assert len(raw)==n and sha256(raw).hexdigest()==h,p
    staging.replace(output)
    file_hash=sha256(output.read_bytes()).hexdigest()
    (destination/"SHA256SUMS.txt").write_text(f"{file_hash}  {output.name}\n",encoding="utf-8")
    print(f"{len(manifest)} files and checksums verified. {output.name}: {output.stat().st_size:,} bytes, SHA-256 {file_hash}")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=ROOT)
    a=p.parse_args()
    package(a.root)
