"""Inventory supplied files and safely inspect reference archives without execution."""
from __future__ import annotations

import json
import re
import stat
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import hashlib

from .io_matlab import _logical_stem, _write_csv, sha256


REFERENCES = {
    "MECG 7780 Course Project.pdf": "official course brief",
    "RPM_Confirmed_Physics_Informed_Gearbox_EDA_Complete.zip": "prior physics EDA",
    "Sparse_LPV_ARMA_complete_reproduction.zip": "published-model reproduction",
    "Gearbox_DHON_NGRC_Improved_Optimization.zip": "original checkpoint archive and legacy-model code",
    "2_Gearbox_DHON_NGRC_Adaptive_Compact.zip": "optional early archive",
    "Sparse_LPV_ARMA_validation_window_reproduction.zip": "optional segment-forensics archive",
    "Sparse LPV-ARMA model for non-stationary vibration.pdf": "Chen et al. (2025) journal paper",
    "Yuejian_Chen_202007_PhD.pdf": "Chen (2020) thesis",
    "fdt_featcal.m": "course MATLAB reference",
}
MAX_MEMBER_BYTES = 120_000_000
MAX_ARCHIVE_CONTENT_BYTES = 500_000_000


def _zip_members(archive: Path, raw_hash: dict, source_name: str):
    """Read verified ZIP member streams, keeping member bytes outside source code."""
    rows, names = [], set()
    with ZipFile(archive) as source:
        if sum(member.file_size for member in source.infolist()) > MAX_ARCHIVE_CONTENT_BYTES:
            raise ValueError(f"Archive exceeds uncompressed size budget: {archive.name}")
        for member in source.infolist():
            name = PurePosixPath(member.filename)
            folded = str(name).casefold()
            mode = (member.external_attr >> 16) & 0o170000
            if (name.is_absolute() or ".." in name.parts or "\\" in member.filename
                    or re.match(r"^[a-z]:", folded) or folded in names
                    or mode not in (0, stat.S_IFREG, stat.S_IFDIR)
                    or member.file_size > MAX_MEMBER_BYTES):
                raise ValueError(f"Unsafe archive member: {archive.name}::{member.filename}")
            names.add(folded)
            if member.is_dir():
                continue
            with source.open(member) as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            reference = raw_hash.get(name.name.casefold())
            relation = ("IDENTICAL" if digest == reference else "CONFLICT") if reference else ""
            rows.append({
                "source_archive": source_name,
                "archive_member": str(name),
                "bytes": member.file_size,
                "sha256": digest,
                "zip_timestamp_local_unknown_zone": datetime(*member.date_time).isoformat(),
                "relation_to_named_raw_input": relation,
                "provenance": "member stream only; not executed",
            })
    return rows


def build_input_manifest(data_dir: Path, source_dirs: list[Path], output: Path) -> dict:
    """Hash raw inputs, record unavailable roles, and verify safe archive layouts."""
    dirs = [Path(data_dir), *[Path(x) for x in source_dirs]]
    raw_files = [
        p for p in sorted(dirs[0].glob("*")) if p.is_file()
        and (p.suffix.casefold() == ".mat" or p.name.casefold() == "fdt_featcal.m")
    ]
    rows, archives = [], []
    raw_hash: dict[str, str] = {}

    def manifest_row(path, relative, role, preprocessing, relation="", freeze=False):
        st = path.stat()
        return {
            "filename": path.name,
            "original_path": relative,
            "role": role, "bytes": st.st_size,
            "sha256": "" if freeze else sha256(path),
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
            "access_status": "PRESENT_DEFER_UNTIL_FREEZE" if freeze else "READABLE",
            "preprocessing_status": preprocessing, "relation_to_other_sources": relation,
        }

    for p in raw_files:
        stem = _logical_stem(p)
        freeze = stem.startswith(("sigtest", "sigfault2"))
        role = "blind inference input" if freeze else "course MATLAB reference" if p.suffix.lower() == ".m" else "course measured input"
        preprocessing = ("uninspected blind pathname" if freeze else
                         "reference code, not executed" if p.suffix.lower() == ".m" else
                         "supplied processed 3.2 kHz; preserve stored x")
        row = manifest_row(p, f"data/raw/{p.name}", role, preprocessing, freeze=freeze)
        rows.append(row)
        if not freeze:
            raw_hash[p.name.casefold()] = row["sha256"]
            raw_hash[stem + p.suffix.lower()] = row["sha256"]

    missing = []
    for name, role in REFERENCES.items():
        if name.casefold() in raw_hash:
            continue
        expected = Path(name)
        candidates = [(i, path) for i, directory in enumerate(dirs[1:], start=1)
                      for path in sorted(directory.rglob("*")) if path.is_file()
                      and path.suffix.lower() == expected.suffix.lower()
                      and _logical_stem(path) == _logical_stem(expected)]
        if not candidates:
            missing.append(name)
            rows.append({
                "filename": name, "original_path": "", "role": role, "bytes": "", "sha256": "",
                "mtime_utc": "", "access_status": "MISSING", "preprocessing_status": "",
                "relation_to_other_sources": "",
            })
            continue
        fingerprints = {sha256(path) for _, path in candidates}
        if len(fingerprints) != 1:
            raise ValueError(f"Conflicting copies of supplied source {name}")
        for j, (i, path) in enumerate(candidates):
            path_id = f"source_{i}/{path.relative_to(dirs[i]).as_posix()}"
            rows.append(manifest_row(
                path, path_id, role, "historical source; keep bytes unchanged",
                "additional byte-identical source copy" if j else "first in declared source-root order",
            ))
            if j == 0 and path.suffix.lower() == ".zip":
                archives.extend(_zip_members(path, raw_hash, path.name))

    for stem in ("sigtrain", "sigvalid", "sigfault1", "sigtest", "sigfault2"):
        if not any(_logical_stem(path) == stem or _logical_stem(path).startswith(f"{stem}_part_")
                   for path in raw_files):
            name = f"{stem}.mat"
            missing.append(name)
            rows.append({
                "filename": name, "original_path": f"data/raw/{name}",
                "role": "expected course input", "bytes": "", "sha256": "", "mtime_utc": "",
                "access_status": "MISSING", "preprocessing_status": "",
                "relation_to_other_sources": "",
            })

    output = Path(output)
    _write_csv(output / "input_manifest.csv", rows)
    _write_csv(output / "archive_members.csv", archives)
    summary = {
        "files_catalogued": len(rows), "archive_members_read": len(archives),
        "missing": missing,
        "matching_archived_data": [
            row for row in archives if row["relation_to_named_raw_input"] == "IDENTICAL"
        ],
        "conflicting_named_archive_members": [
            row for row in archives if row["relation_to_named_raw_input"] == "CONFLICT"
        ],
        "precedence": "course PDF for requirements; course data and executable code for actual experiments; "
                      "journal for published equations/metrics; thesis for theory and rig corroboration",
    }
    (output / "input_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
