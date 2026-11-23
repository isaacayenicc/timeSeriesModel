from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from src_gearbox.models.baselines.replay import BASELINE_SLUGS, CHECKPOINT_DIR, load_predictor
from src_gearbox.models.baselines.training import smoke

ROOT=Path(__file__).resolve().parents[1]


def test_all_baselines_have_checkpoint_and_synthetic_replay():
    with threadpool_limits(1):
        rows=smoke(ROOT)
    assert len(rows)==21 and {r["model"] for r in rows}==set(BASELINE_SLUGS)
    assert all(r["status"]=="PASS" for r in rows)


def test_selected_arx_and_sparse_reference_use_only_previous_sample():
    samples=2300
    cut=2000
    rng=np.random.default_rng(9181)
    x=rng.normal(size=samples)
    rpm=143+7*np.sin(np.arange(samples)/210)
    altered_x=x.copy()
    altered_x[cut:]+=10
    altered_rpm=rpm.copy()
    altered_rpm[cut:]*=1.03
    with threadpool_limits(1):
        for slug in ("arx","sparse_lpv_arma","archived_src_corrected"):
            fitted=load_predictor(slug,ROOT/CHECKPOINT_DIR)
            before=fitted.predict(x,rpm,3200)
            after=fitted.predict(altered_x,altered_rpm,3200)
            np.testing.assert_allclose(before[1792:cut+1],after[1792:cut+1],rtol=0,atol=2e-5)


def test_all_baselines_fit_save_and_load_synthetic(tmp_path):
    from src_gearbox.models.baselines.training import smoke_fit

    with threadpool_limits(1):
        fitted = smoke_fit(tmp_path)
    assert tuple(fitted)==BASELINE_SLUGS
