"""Mutating or reordering source parts must fail, without joining recordings."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from rotmach.fault_io import FIELDS, iter_fault_records, verify_fault_parts
from rotmach.io_matlab import sha256

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def parts(tmp_path):
    # Use the separately supplied splitter's digest as the reference fixture.
    spec=importlib.util.spec_from_file_location('supplied_splitter',ROOT/'scripts/sigfault1_splitter.py')
    splitter=importlib.util.module_from_spec(spec);spec.loader.exec_module(splitter)
    arrays=[];entries=[];start=1
    for i,count in enumerate((2,1),1):
        a=np.empty((1,count),dtype=[(field,object) for field in FIELDS])
        for j in range(count):
            n=16;data=dict(t=np.arange(1,n+1)[None,:]/16,fs=np.array([[16.]]),
                           N=np.array([[n]]),spd=np.full((1,n),3.),spdnorm=np.full((1,n),.5),
                           x=(np.arange(n)+100*(start+j))[None,:],spdmin=np.array([[2.]]),spdmax=np.array([[4.]]))
            for field in FIELDS:a[0,j][field]=data[field]
        name=f'sigfault1_part_{i:03d}.mat';file=tmp_path/name;savemat(file,{'sigfault1':a},do_compression=True)
        reread=splitter.load_records(file);arrays.append(reread)
        entries.append(dict(filename=name,part_number=i,record_start_1based=start,
                            record_end_1based_inclusive=start+count-1,record_count=count,
                            bytes=file.stat().st_size,sha256=sha256(file),logical_data_sha256=splitter.logical_hash([reread])))
        start+=count
    manifest=dict(schema_version=1,variable_name='sigfault1',fields=list(FIELDS),part_total=2,record_total=3,
                  original_file_sha256='not independently available fixture',logical_data_sha256=splitter.logical_hash(arrays),parts=entries)
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    return tmp_path


def test_splitter_parity_and_global_record_ids(parts):
    checked=verify_fault_parts(parts)
    assert checked['record_total']==3 and checked['part_total']==2
    records=list(iter_fault_records(parts))
    assert [r['global_record_id'] for r in records]==[1,2,3]
    assert [r['x'][0] for r in records]==[100,200,300]
    assert all(len(r['x'])==16 for r in records)


def test_missing_part_rejected(parts):
    (parts/'sigfault1_part_002.mat').unlink()
    with pytest.raises(ValueError,match='Missing'):verify_fault_parts(parts)


def test_corrupt_file_rejected(parts):
    p=parts/'sigfault1_part_001.mat';b=bytearray(p.read_bytes());b[-10]^=1;p.write_bytes(b)
    with pytest.raises(ValueError,match='SHA256'):verify_fault_parts(parts)


def test_wrong_logical_digest_rejected(parts):
    p=parts/'manifest.json';m=json.loads(p.read_text());m['parts'][0]['logical_data_sha256']='0'*64;p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='Logical'):verify_fault_parts(parts)


def test_wrong_record_order_rejected(parts):
    p=parts/'manifest.json';m=json.loads(p.read_text());m['parts'][1]['record_start_1based']=2;p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='ordering'):verify_fault_parts(parts)


def test_supplied_twenty_parts_when_present():
    folder=ROOT/'data/raw/sigfault1_parts'
    if not folder.exists():pytest.skip('Large user data not included in release')
    checked=verify_fault_parts(folder)
    assert checked['record_total']==200 and checked['part_total']==20
    first=next(iter_fault_records(folder,verify=False))
    assert first['N']==32000 and first['fs']==3200
