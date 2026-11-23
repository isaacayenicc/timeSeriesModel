"""Pack the editable report and the materials required to reproduce it."""
from hashlib import sha256
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
BASE = "MECG7780_PI_SRC_Final_Course_Report"
ARCHIVE = "PI_SRC_Report_Submission_Package.zip"
MAIN_FILES = (f"{BASE}.docx", f"{BASE}.pdf", "FINAL_REPORT_README.md", "report_source.md")
PACKAGE_TYPES = {".bib", ".csl", ".csv", ".json", ".md", ".png", ".py", ".txt", ".yaml"}


def manifest_entries():
    files = [ROOT / name for name in MAIN_FILES]
    for directory in ("assets", "evidence", "qa", "report_generation"):
        files.extend(path for path in (ROOT / directory).rglob("*")
                     if path.is_file() and path.suffix in PACKAGE_TYPES)
    for path in sorted(files):
        yield path.relative_to(ROOT).as_posix(), sha256(path.read_bytes()).hexdigest()


def main():
    entries = dict(manifest_entries())
    manifest = "".join(f"{digest}  {name}\n" for name, digest in entries.items())
    manifest_file = ROOT / "SHA256SUMS.txt"
    manifest_file.write_text(manifest, encoding="utf-8")
    archive = ROOT / ARCHIVE
    with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=6) as bundle:
        for name in entries:
            bundle.write(ROOT / name, name)
        bundle.write(manifest_file, manifest_file.name)
    with ZipFile(archive) as bundle:
        if bundle.testzip() or any(sha256(bundle.read(name)).hexdigest() != digest
                                   for name, digest in entries.items()):
            raise ValueError("Archive verification failed")
    (ROOT / f"{ARCHIVE}.sha256").write_text(
        f"{sha256(archive.read_bytes()).hexdigest()}  {ARCHIVE}\n", encoding="utf-8")
    print(f"Verified {len(entries)} files in {archive.name}")


if __name__ == "__main__":
    main()
