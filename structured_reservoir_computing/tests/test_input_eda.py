"""Data intake, safe archive and physical-coordinate regression checks."""
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from scipy.io import savemat

from src_gearbox.data import find_input, load_record
from src_gearbox.data.io_matlab import _partial_overlap_scan, iter_fault_records
from src_gearbox.data.provenance import _zip_members
from src_gearbox.eda_extensions import finite_quadrature, phase_fixtures


def values(x, rpm=None):
    t = (np.arange(len(x)) + 1) / 3200
    if rpm is None:
        rpm = 120 + 0.5*t
    return dict(x=np.asarray(x)[:, None], spd=rpm[:, None], t=t[:, None], fs=3200, N=len(x))


def part(path, vals):
    block = np.empty((1,len(vals)), dtype=[(key,"O") for key in vals[0]])
    for i, fields in enumerate(vals):
        for key, value in fields.items():
            block[0,i][key] = value
    savemat(path, {"sigfault1": block})


def test_mat_resolution_and_conflict(tmp_path):
    x = np.arange(800, dtype=float)
    layout = tmp_path/"SIGTRAIN (7).MAT"
    savemat(layout, {"sigtrain": values(x)}, appendmat=False)
    assert find_input(tmp_path,"sigtrain") == layout
    assert np.array_equal(load_record(layout).x,x)
    savemat(tmp_path/"sigtrain.mat", {"sigtrain": values(x+.1)})
    with pytest.raises(ValueError, match="Conflicting files"):
        find_input(tmp_path,"sigtrain")


def test_split_order_and_shifted_overlap(tmp_path):
    x = np.random.default_rng(12).normal(size=2400)
    rpm = 124+x*.1
    part(tmp_path/"sigfault1_part_002.mat",[values(x[1200:],rpm[1200:])])
    part(tmp_path/"sigfault1_part_001.mat",[values(x[:1800],rpm[:1800])])
    assert [v.index for v in iter_fault_records(tmp_path)] == [1,2]
    results = _partial_overlap_scan([],tmp_path)
    assert len(results)==1 and results[0]["second_minus_first_offset_samples"]==-1200


def test_archive_metadata_path_guard(tmp_path):
    bad=tmp_path/"bad.zip"
    with ZipFile(bad,"w") as z:
        z.writestr("../escape",b"must never extract")
    with pytest.raises(ValueError, match="Unsafe archive member"):
        _zip_members(bad,{},bad.name)


def test_synthetic_phase_fixtures_and_finite_quadrature():
    result = phase_fixtures()
    assert all(row["passed"] for row in result.values())
    rng=np.random.default_rng(350)
    x=rng.normal(size=2000)
    phase=np.cumsum(rng.uniform(.2,.3,size=len(x)))
    tau,length=72,256
    w=np.exp(-np.arange(length)/tau)
    result=finite_quadrature(x,phase,tau,length)
    ix=np.array([256,500,832,1000,1774])
    j=ix[:,None]-np.arange(1,length+1)
    expected=2*np.sum(w*x[j]*np.exp(2j*np.pi*(phase[ix,None]-phase[j])),axis=1)/w.sum()
    np.testing.assert_allclose(expected,result[ix],atol=1e-13,rtol=0)


def test_mat73_reference_struct(tmp_path):
    h5py=pytest.importorskip("h5py")
    path=tmp_path/"sigtrain.mat"
    x=np.arange(900,dtype=float)
    with h5py.File(path,"w",userblock_size=512) as file:
        refs=file.create_group("#refs#")
        struct=file.create_group("sigtrain")
        struct.attrs["MATLAB_class"]=np.bytes_("struct")
        for k,v in values(x).items():
            d=refs.create_dataset(k,data=np.atleast_2d(v).T)
            h=struct.create_dataset(k,(1,1),dtype=h5py.ref_dtype)
            h[0,0]=d.ref
    header=b"MATLAB 7.3 MAT-file, Platform: local regression fixture"
    with path.open("r+b") as f:
        f.write(header.ljust(116,b" ")+b"\0"*8+b"\x00\x02IM")
    np.testing.assert_array_equal(load_record(path).x,x)
