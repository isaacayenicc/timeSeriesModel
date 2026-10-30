"""Verified, record-preserving access to the supplied lossless MAT partitions.

File/field digests are checked against the user-supplied manifest. This verifies
internal consistency, not independent custody of the unavailable original MAT.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Iterator

import numpy as np
from scipy.io import loadmat

from .io_matlab import load_records, sha256

FIELDS = ('t', 'fs', 'N', 'spd', 'spdnorm', 'x', 'spdmin', 'spdmax')


def _new_digest():
    h = hashlib.sha256()
    h.update(f'sigfault1\x00{FIELDS}\x00'.encode('utf8'))
    return h


def _update_record(h, record):
    for field in FIELDS:
        array = np.asarray(record[field])
        h.update(field.encode('utf8') + b'\x00')
        h.update(str(array.dtype).encode('utf8') + b'\x00')
        h.update(json.dumps(array.shape).encode('utf8') + b'\x00')
        h.update(np.ascontiguousarray(array).tobytes(order='C'))


def verify_fault_parts(parts_dir: str | Path) -> dict:
    """Verify bytes, dtypes/shapes/values, order, coverage and data validity.

    A missing, duplicated, reordered, modified, unexpected or invalid part fails
    loudly. Only one part is held at a time. Returns safe measured metadata.
    """
    folder = Path(parts_dir)
    manifest_path = folder/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema_version') != 1 or manifest.get('variable_name') != 'sigfault1':
        raise ValueError('Unsupported fault split manifest schema/variable')
    if tuple(manifest.get('fields', [])) != FIELDS:
        raise ValueError('Manifest fault fields do not match the supplied schema')
    entries = manifest.get('parts', [])
    names = [entry['filename'] for entry in entries]
    if not entries or len(names) != len(set(names)) or len(entries) != manifest['part_total']:
        raise ValueError('Incorrect or duplicate part count')
    if set(p.name for p in folder.glob('*.mat')) != set(names):
        raise ValueError('Missing or unexpected MAT parts')
    global_hash = _new_digest()
    next_record = 1
    measured = []
    for part_index, entry in enumerate(entries, 1):
        name = entry['filename']
        if not re.fullmatch(r'sigfault1_part_\d{3}\.mat', name):
            raise ValueError('Unsafe or unsupported part filename')
        path = folder/name
        if entry['part_number'] != part_index or entry['record_start_1based'] != next_record:
            raise ValueError('Incorrect part/record ordering')
        if path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            raise ValueError(f'File size/SHA256 mismatch: {name}')
        raw = loadmat(path, struct_as_record=True, squeeze_me=False).get('sigfault1')
        if raw is None or raw.ndim != 2 or raw.shape[0] != 1 or raw.dtype.names != FIELDS:
            raise ValueError(f'Invalid original MATLAB struct shape/fields: {name}')
        if raw.shape[1] != entry['record_count'] or entry['record_end_1based_inclusive'] != next_record+raw.shape[1]-1:
            raise ValueError(f'Incorrect record range: {name}')
        digest = _new_digest()
        for record in raw.ravel(order='C'):
            _update_record(digest, record)
            _update_record(global_hash, record)
        if digest.hexdigest() != entry['logical_data_sha256']:
            raise ValueError(f'Logical field-data checksum mismatch: {name}')
        # Independent scientific gate, beyond the splitter's length-only checks.
        simplified = load_records(path)
        if len(simplified) != raw.shape[1]:
            raise ValueError('Record boundaries lost during simplification')
        measured.append(dict(filename=name,record_count=len(simplified),
                             first_record=next_record,last_record=next_record+len(simplified)-1,
                             bytes=path.stat().st_size,sha256=entry['sha256']))
        next_record += len(simplified)
    if next_record-1 != manifest['record_total'] or global_hash.hexdigest() != manifest['logical_data_sha256']:
        raise ValueError('Global logical digest or record total mismatch')
    return dict(status='PASS',part_total=len(entries),record_total=next_record-1,
                logical_data_sha256=global_hash.hexdigest(),manifest_sha256=sha256(manifest_path),
                parts=measured,source_fields=list(FIELDS),
                original_file_sha256_declared=manifest['original_file_sha256'],
                original_file_independently_available=False,
                provenance_limit='Original MAT fingerprint is declared by the supplied manifest; exact original-file bytes were not available for independent comparison.')


def iter_fault_records(parts_dir: str | Path, verify: bool = True) -> Iterator[dict]:
    """Yield complete validated records in original 1-based global order.

    x is already normalized; it is never rescaled. All metadata from the source
    is retained; source IDs indicate provenance, not fault state or severity.
    """
    folder = Path(parts_dir)
    if verify:
        verify_fault_parts(folder)
    manifest = json.loads((folder/'manifest.json').read_text())
    for entry in manifest['parts']:
        for local, record in enumerate(load_records(folder/entry['filename'])):
            global_id = entry['record_start_1based']+local
            record.update(record_id=f'fault_candidate_record_{global_id:03d}',
                          global_record_id=global_id,source_part=entry['filename'],
                          source_part_sha256=entry['sha256'],
                          source_logical_sha256=manifest['logical_data_sha256'])
            yield record
