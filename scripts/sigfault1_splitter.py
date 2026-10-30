#!/usr/bin/env python3
"""Record-aware splitting, verification, and reconstruction for sigfault1.mat.

Dependencies: numpy, scipy. Compression only changes the MAT container, never
numeric precision. SHA-256 of the original MAT file is NOT expected to match a
reconstructed MAT file, because MATLAB serialization can differ. The logical
field-level hash and record order, instead, must match.

Examples:
    python sigfault1_splitter.py split sigfault1.mat sigfault1_parts --records-per-part 10
    python sigfault1_splitter.py verify sigfault1_parts
    python sigfault1_splitter.py reconstruct sigfault1_parts sigfault1_reconstructed.mat
    python sigfault1_splitter.py compare sigfault1.mat sigfault1_reconstructed.mat
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat, savemat

VAR = 'sigfault1'
EXPECTED_FIELDS = ('t', 'fs', 'N', 'spd', 'spdnorm', 'x', 'spdmin', 'spdmax')
MANIFEST = 'manifest.json'


def file_hash(filename: Path) -> str:
    h = hashlib.sha256()
    with filename.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_records(filename: Path) -> np.ndarray:
    d = loadmat(filename, struct_as_record=True, squeeze_me=False)
    if VAR not in d:
        raise ValueError(f'{filename}: MAT variable {VAR!r} not found; found {list(d)}')
    a = d[VAR]
    if a.ndim != 2 or a.shape[0] != 1 or a.dtype.names != EXPECTED_FIELDS:
        raise ValueError(f'{filename}: expected (1,N) struct with fields {EXPECTED_FIELDS}, got {a.shape}, {a.dtype.names}')
    for i in range(a.shape[1]):
        r = a[0, i]
        n = int(r['N'].reshape(-1)[0])
        fs = float(r['fs'].reshape(-1)[0])
        if n <= 0 or fs <= 0:
            raise ValueError(f'{filename}: invalid metadata for record {i+1}: N={n}, fs={fs}')
        for key in ('x', 'spd', 'spdnorm', 't'):
            if np.asarray(r[key]).size != n:
                raise ValueError(f'{filename}: record {i+1}, field {key}: expected {n} samples')
    return a


def digest_array_record(h: 'hashlib._Hash', rec) -> None:
    for name in EXPECTED_FIELDS:
        a = np.asarray(rec[name])
        h.update(name.encode('utf8') + b'\x00')
        h.update(str(a.dtype).encode('utf8') + b'\x00')
        h.update(json.dumps(a.shape).encode('utf8') + b'\x00')
        h.update(np.ascontiguousarray(a).tobytes(order='C'))


def logical_hash(arrays: list[np.ndarray]) -> str:
    h = hashlib.sha256()
    h.update(f'{VAR}\x00{EXPECTED_FIELDS}\x00'.encode('utf8'))
    for a in arrays:
        for rec in a.ravel(order='C'):
            digest_array_record(h, rec)
    return h.hexdigest()


def split(input_mat: Path, output: Path, batch: int) -> None:
    if batch <= 0:
        raise ValueError('--records-per-part must be positive')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'{output} exists and is nonempty. Use a new output directory.')
    input_mat = input_mat.resolve()
    all_records = load_records(input_mat)
    total = all_records.shape[1]
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for part_no, start in enumerate(range(0, total, batch), 1):
        end = min(start + batch, total)
        dest = output / f'{VAR}_part_{part_no:03d}.mat'
        savemat(dest, {VAR: all_records[:, start:end]}, do_compression=True, oned_as='row')
        saved = load_records(dest)
        if saved.shape[1] != end - start or logical_hash([saved]) != logical_hash([all_records[:, start:end]]):
            raise RuntimeError(f'Failed verification: {dest}')
        entry = {
            'filename': dest.name,
            'part_number': part_no,
            'record_start_1based': start + 1,
            'record_end_1based_inclusive': end,
            'record_count': end - start,
            'bytes': dest.stat().st_size,
            'sha256': file_hash(dest),
            'logical_data_sha256': logical_hash([saved]),
        }
        files.append(entry)
        print(f'Wrote {dest.name}: records {start+1}–{end}, {entry["bytes"]/1024**2:.2f} MiB', flush=True)
    manifest = {
        'schema_version': 1,
        'description': 'Lossless MATLAB-struct partition of sigfault1.mat; complete records, no time-domain cutting',
        'variable_name': VAR,
        'original_filename': input_mat.name,
        'original_file_bytes': input_mat.stat().st_size,
        'original_file_sha256': file_hash(input_mat),
        'record_total': total,
        'part_total': len(files),
        'records_per_part': batch,
        'fields': list(EXPECTED_FIELDS),
        'logical_data_sha256': logical_hash([all_records]),
        'parts': files,
    }
    (output / MANIFEST).write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf8')
    print(f'Created manifest: {output / MANIFEST}')
    verify(output)


def load_manifest(folder: Path) -> dict:
    path = folder / MANIFEST
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}')
    return json.loads(path.read_text(encoding='utf8'))


def verify(folder: Path) -> None:
    manifest = load_manifest(folder)
    if manifest.get('schema_version') != 1:
        raise ValueError('Unknown manifest version')
    arrays = []
    next_index = 1
    for entry in manifest['parts']:
        name = entry['filename']
        if not re.fullmatch(r'sigfault1_part_\d{3}\.mat', name):
            raise ValueError(f'Unexpected chunk filename: {name}')
        p = folder / name
        if not p.exists():
            raise FileNotFoundError(f'Missing chunk: {p}')
        if p.stat().st_size != entry['bytes']:
            raise ValueError(f'Size mismatch: {p}')
        if file_hash(p) != entry['sha256']:
            raise ValueError(f'SHA-256 mismatch: {p}')
        a = load_records(p)
        if entry['record_start_1based'] != next_index or a.shape[1] != entry['record_count']:
            raise ValueError(f'Ordering / count mismatch in {name}')
        if entry['record_end_1based_inclusive'] != next_index + a.shape[1] - 1:
            raise ValueError(f'Incorrect record-end in {name}')
        if logical_hash([a]) != entry['logical_data_sha256']:
            raise ValueError(f'Logical field-data mismatch in {name}')
        arrays.append(a)
        next_index += a.shape[1]
    if next_index - 1 != manifest['record_total'] or len(arrays) != manifest['part_total']:
        raise ValueError('Incorrect overall number of chunks or records')
    if logical_hash(arrays) != manifest['logical_data_sha256']:
        raise ValueError('Overall logical SHA-256 mismatch')
    print(f'PASS: {len(arrays)} chunks / {next_index-1} records, correct order, all field-level hashes and file checksums')


def reconstruct(folder: Path, outfile: Path) -> None:
    if outfile.exists():
        raise FileExistsError(f'Output already exists: {outfile}')
    verify(folder)
    meta = load_manifest(folder)
    arrays = [load_records(folder / part['filename']) for part in meta['parts']]
    restored = np.concatenate(arrays, axis=1)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    savemat(outfile, {VAR: restored}, do_compression=True, oned_as='row')
    result = load_records(outfile)
    if logical_hash([result]) != meta['logical_data_sha256']:
        raise RuntimeError('Saved reconstruction did not match manifest')
    print(f'PASS: reconstructed {outfile} ({result.shape[1]} records, numerical data identical)')
    print('Note: reconstructed MAT bytes need not equal original MAT bytes due to serialization differences.')


def compare(original: Path, reconstructed: Path) -> None:
    a = load_records(original)
    b = load_records(reconstructed)
    if a.shape != b.shape or logical_hash([a]) != logical_hash([b]):
        raise ValueError('Mismatch in original and reconstructed logical data')
    for i in range(a.shape[1]):
        for field in EXPECTED_FIELDS:
            if not np.array_equal(a[0, i][field], b[0, i][field], equal_nan=True):
                raise ValueError(f'Original/reconstructed mismatch record={i+1} field={field}')
    print(f'PASS: identical data, shapes, field dtypes and record order for {a.shape[1]} records')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='mode', required=True)
    s = sub.add_parser('split'); s.add_argument('input_mat', type=Path); s.add_argument('output_dir', type=Path)
    s.add_argument('--records-per-part', type=int, default=10)
    v = sub.add_parser('verify'); v.add_argument('parts_dir', type=Path)
    r = sub.add_parser('reconstruct'); r.add_argument('parts_dir', type=Path); r.add_argument('output_mat', type=Path)
    c = sub.add_parser('compare'); c.add_argument('original', type=Path); c.add_argument('reconstructed', type=Path)
    args = parser.parse_args()
    if args.mode == 'split': split(args.input_mat, args.output_dir, args.records_per_part)
    if args.mode == 'verify': verify(args.parts_dir)
    if args.mode == 'reconstruct': reconstruct(args.parts_dir, args.output_mat)
    if args.mode == 'compare': compare(args.original, args.reconstructed)

if __name__ == '__main__':
    main()
