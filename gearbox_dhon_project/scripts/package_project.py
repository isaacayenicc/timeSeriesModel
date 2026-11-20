"""Stage a portable project and build a hash-verified archive without raw MAT files."""
import argparse
import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]


def deliverable(path):
    rel = path.relative_to(ROOT)
    parts = rel.parts
    if any(part in {"__pycache__", ".pytest_cache", ".venv", ".git", ".ipynb_checkpoints"}
           or part.endswith(".egg-info") for part in parts):
        return False
    if parts[:2] in {("data", "raw"), ("references", "prior_work")}:
        return False
    if parts[:3] == ("outputs", "runs", "smoke"):
        return False
    if parts[:3] == ("outputs", "runs", "smoke_verified"):
        return rel.name in {"pipeline_stages.json", "data_availability.json", "run_fingerprint.json"}
    return path.suffix.lower() != ".mat" and not path.is_symlink() and rel.name != "CHECKSUMS.sha256"


def package(destination):
    files = [p for p in sorted(ROOT.rglob("*")) if p.is_file() and deliverable(p)]
    for needed in (
        "README.md", "outputs/final/final_course_report.pdf", "outputs/final/completion_checklist.md",
        "outputs/runs/development/frozen_manifest.json", "outputs/runs/development/results_verification.json",
        "presentations/progress_presentation.pptx", "presentations/final_presentation.pptx",
    ):
        if ROOT / needed not in files:
            raise ValueError(f"Required artifact missing: {needed}")
    code = []
    for source in files:
        with source.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        code.append((digest, source, source.relative_to(ROOT).as_posix()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=5) as z:
        archive_root = ROOT.name
        for _, path, relpath in code:
            z.write(path, f"{archive_root}/{relpath}")
        z.writestr(f"{archive_root}/CHECKSUMS.sha256",
                   "".join(f"{digest}  {relpath}\n" for digest, _, relpath in code))
    with ZipFile(destination) as z:
        if z.testzip() is not None:
            raise RuntimeError("ZIP integrity failure")
        for digest, _, relative in code:
            with z.open(f"{archive_root}/{relative}") as f:
                if hashlib.file_digest(f, "sha256").hexdigest() != digest:
                    raise RuntimeError(f"Checksum mismatch: {relative}")
    print(f"{len(files)} files verified, {destination.stat().st_size:,} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    package(parser.parse_args().output)
