"""Write a verified expanded project archive without input files or caches."""
from argparse import ArgumentParser
from hashlib import file_digest
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT=Path(__file__).resolve().parents[1]
TOP=("README.md","AGENTS.md","pyproject.toml","requirements.txt","requirements-dev.txt",".gitignore")
PATHS=("src","scripts","tests","docs","notebooks","presentations","prior",
       "outputs/expanded","outputs/blind_missing")


def package(output):
    paths = [ROOT/p for p in TOP]
    for base in PATHS:
        paths.extend(p for p in (ROOT/base).rglob("*")
                     if p.is_file() and not p.is_symlink() and p.suffix.lower() not in {".mat",".pyc",".zip"}
                     and not {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}.intersection(p.parts)
                     and not any(part.endswith(".egg-info") for part in p.parts))
    if not (ROOT/"outputs/expanded/Expanded_DHON_Companion.pdf").exists():
        raise FileNotFoundError("Build companion PDF before packaging")
    output.parent.mkdir(parents=True,exist_ok=True)
    manifest={}
    with ZipFile(output,"w",ZIP_DEFLATED,compresslevel=5) as z:
        for p in sorted(paths):
            name=p.relative_to(ROOT).as_posix()
            with p.open("rb") as stream:
                manifest[name]=file_digest(stream,"sha256").hexdigest()
            z.write(p,"gearbox_expanded/"+name)
        z.writestr("gearbox_expanded/CHECKSUMS.sha256",
                   "".join(f"{sha}  {name}\n" for name,sha in manifest.items()))
    with ZipFile(output) as z:
        if z.testzip():
            raise RuntimeError("Package CRC failure")
        for name,sha in manifest.items():
            with z.open("gearbox_expanded/"+name) as stream:
                if file_digest(stream,"sha256").hexdigest()!=sha:
                    raise RuntimeError(f"Checksum failed for {name}")
    print(f"Verified {len(manifest)} files, {output.stat().st_size:,} bytes")


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    package(parser.parse_args().output)
