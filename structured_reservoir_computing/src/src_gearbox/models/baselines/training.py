"""Separate comparator retraining, smoke checks and matched RPM withholding."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .protocol import START, SEED, FAMILIES, digest, load_data, mse, write_json, write_table
from .replay import BASELINE_SLUGS, CHECKPOINT_DIR, load_predictor

SPEED_CONTROLS = ("arx","narx","mlp","lstm","gru","cnn","tcn","transformer","tft")
NEURAL = ("mlp","lstm","gru","cnn","tcn")
NOMINAL_RPM = 145.


@dataclass(frozen=True)
class VibrationOnly:
    fitted: object

    def predict(self,x,rpm,fs):
        return self.fitted.predict(x,np.full(len(x),NOMINAL_RPM),fs)


def load_speed_control(slug, root):
    return VibrationOnly(load_predictor(slug,Path(root)/CHECKPOINT_DIR/"vibration_only"))


def smoke(root: Path):
    """Call every frozen predictor on finite non-course synthetic history."""
    import torch
    torch.set_num_threads(1)
    t = np.arange(2304) / 3200
    rpm = 145 + 10*np.sin(2*np.pi*3*t)
    phase = np.cumsum(rpm)/(60*3200)
    x = .6*np.cos(2*np.pi*37*phase) + .25*np.sin(2*np.pi*230*t)
    rows = []
    for slug in BASELINE_SLUGS:
        predictor=load_predictor(slug,Path(root)/CHECKPOINT_DIR)
        values=predictor.predict(x,rpm,3200)
        if values.shape!=x.shape or not np.isfinite(values[START:]).all():
            raise ArithmeticError(f"{slug} violated the synthetic output contract")
        rows.append(dict(model=slug,targets=len(x)-START,status="PASS"))
    return rows


def fit_all(data_dir, root, out, resume=False):
    """Re-execute the reported guarded grids. The sparse pair refits sealed supports."""
    import torch
    from . import statistical, tabular, neural, attention
    torch.set_num_threads(1)
    root, out = Path(root), Path(out)
    out = out.resolve()
    if out == root.resolve() or (out / "models").resolve() == (root/CHECKPOINT_DIR).resolve():
        raise ValueError("Use a separate fits directory")
    if out.is_dir() and any(out.iterdir()) and not resume:
        raise ValueError("Nonempty output needs --resume, or use a fresh path")
    out.mkdir(parents=True, exist_ok=True)
    train,valid=load_data(Path(data_dir))
    identity={
        "training_sha256":digest(Path(data_dir)/train.source),
        "development_sha256":digest(Path(data_dir)/valid.source),
        "frozen_selection_sha256":digest(root/"configs/baselines/selection.json"),
    }
    modes = (("statistical",statistical),("tabular",tabular),("neural",neural),("attention",attention))
    for name, module in modes:
        marker=out/f"{name}_stage_complete.json"
        paths={slug:out/"models"/f"{slug}.npz" for slug in FAMILIES[name]}
        if resume and marker.is_file():
            previous=json.loads(marker.read_text(encoding="utf-8"))
            if previous["identity"]!=identity or any(
                not path.is_file() or digest(path)!=previous["models"][slug] for slug,path in paths.items()
            ):
                raise ValueError(f"{name} resume marker does not match current datasets or checkpoints")
            continue
        module.run(Path(data_dir),out,root/"references/baselines")
        write_json(marker,{"stage":name,"status":"DONE","identity":identity,
                           "models":{slug:digest(path) for slug,path in paths.items()}})
    return out


def _fit_one(slug, model, train):
    from . import attention, neural, statistical, tabular
    fake = np.full(len(train.x),NOMINAL_RPM)
    if slug in NEURAL:
        fitted,trace,count=neural._fit([(train.x,fake)],train.fs,model.config,seed=SEED)
        return fitted,count,"AdamW fixed guarded config"
    if slug in ("transformer","tft"):
        fitted=attention.fit(asdict(model.config),[(train.x,fake)],train.fs)
        count=(len(train.x)-model.config.history+model.config.stride-1)//model.config.stride
        return fitted,count,"AdamW fixed guarded config"
    if slug=="arx":
        fitted=statistical.fit_model([(train.x,fake)],model.spec)
        return fitted,len(train.x)-statistical.FIT_WARMUP,"identical ridge design"
    config=model.config
    y, target=tabular.training_matrix(train.x,fake,train.fs,config.feature_set)
    extra,_=tabular.training_matrix(train.x,fake,train.fs,"physics_compact")
    fitted, info=tabular._fit(config,train.fs,y,target,extra)
    return fitted,info["n_fit"],info["state"]


def fit_speed_controls(root: Path, data_dir: Path, selected=SPEED_CONTROLS):
    """Retrain selected configurations with measured RPM withheld at fit and inference."""
    import torch
    from threadpoolctl import threadpool_limits
    from .replay import _verify

    root=Path(root)
    torch.set_num_threads(1)
    train,test=load_data(data_dir)
    checkpoint=root/CHECKPOINT_DIR/"vibration_only"
    checkpoint.mkdir(parents=True,exist_ok=True)
    rows, predictions=[],{}
    with threadpool_limits(1):
        for slug in selected:
            base=load_predictor(slug,root/CHECKPOINT_DIR)
            if slug not in SPEED_CONTROLS:
                raise ValueError(f"No RPM withholding protocol for {slug}")
            tick=perf_counter()
            fitted,count,state=_fit_one(slug,base,train)
            fit_s=perf_counter()-tick
            dest=checkpoint/f"{slug}.npz"
            fitted.save(dest)
            control=load_speed_control(slug,root)
            control_prediction=control.predict(test.x,test.rpm,test.fs)
            base_prediction=base.predict(test.x,test.rpm,test.fs)
            delta,repeat=_verify(control,test,control_prediction)
            measured=mse(test.x[START:],base_prediction[START:])
            withheld=mse(test.x[START:],control_prediction[START:])
            rows.append(dict(model=slug,measured_mse=measured,vibration_only_mse=withheld,
                             delta_measured_minus_vibration_only=measured-withheld,n=len(test.x)-START,
                             fixed_nominal_rpm=NOMINAL_RPM,fit_rows=count,fit_s=fit_s,fit_state=state,
                             causal_max_abs=delta,repeat_max_abs=repeat))
            predictions[f"{slug}__vibration_only"]=control_prediction[START:]
            print(f"{slug}: speed on {measured:.7f}, withheld {withheld:.7f}",flush=True)
            write_table(root/"results/metrics/vibration_rpm_parity.csv",rows)
            np.savez_compressed(root/"results/predictions/vibration_rpm_parity.npz",
                                target_indices=np.arange(START,len(test.x)),**predictions)
    return rows


def smoke_fit(output: Path):
    """Exercise all estimator fit/save/load routes with synthetic measurements."""
    from dataclasses import replace
    from scipy import signal
    import torch
    from . import attention, neural, statistical, tabular, trees

    torch.set_num_threads(1)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    fs=3200
    n=2176
    rng=np.random.default_rng(9054)
    rpm=145+9*np.sin(np.arange(n)/150)
    noise=rng.normal(size=n)
    x=signal.lfilter([1.,.3],[1.,-.45,.15],noise)*.35
    x+=.13*np.sin(2*np.pi*37*np.cumsum(rpm)/(60*fs))
    slices=[(x,rpm)]
    result=[]

    for family,members in FAMILIES.items():
        for slug in members:
            cfg={}
            if slug.startswith("sparse_lpv"):
                lo,hi=124.,168.
                speed_range=np.array([lo,hi])
                w=(rpm-lo)/(hi-lo)
                a1,a2,b2,terms=4,3,2,3
                beta1=statistical._lpv_fixed_support_fit(x,w,a1,0,terms,np.ones(a1*terms))
                if slug=="sparse_lpv_arma":
                    stage1=statistical._lpv_predict(x,w,a1,terms,beta1)
                    e=np.nan_to_num(x-stage1)
                    beta2=statistical._lpv_fixed_support_fit(x,w,a2,b2,terms,np.ones((a2+b2)*terms),e)
                else:
                    a2,b2=0,0
                    beta2=np.empty(0)
                model=statistical.SparseLPV(speed_range,terms,beta1,beta2,a1,a2,b2)
            elif family=="statistical":
                spec=statistical.FitSpec(slug,12,1 if slug in ("arma","arima") else 0,
                                          (1,7) if slug=="arx" else (),.002 if slug=="arx" else 0.)
                model=statistical.fit_model(slices,spec)
            elif slug in ("random_forest","gradient_boosting"):
                feature,cfg=trees.candidates(smoke=True)[slug][0]
                matrix,y=tabular.training_matrix(x,rpm,fs,feature)
                model=trees.export_estimator(slug,feature,trees.fit_tree(slug,cfg,matrix,y))
            elif family=="tabular":
                cfg=tabular.candidate_grid()[slug][0]
                if slug=="narx":
                    cfg=replace(cfg,epochs=2)
                matrix,y=tabular.training_matrix(x,rpm,fs,cfg.feature_set)
                extra=tabular.training_matrix(x,rpm,fs,"physics_compact")[0] if slug=="narx" else None
                model,_=tabular._fit(cfg,fs,matrix,y,extra)
            elif family=="neural":
                cfg=replace(neural.candidates(slug)[0],epochs=1)
                model,*_=neural._fit(slices,fs,cfg,seed=SEED)
            else:
                cfg=attention.candidates(slug)[0]
                model=attention.fit(dict(cfg,epochs=1),slices,fs)
            path=output/f"{slug}.npz"
            if slug in ("random_forest","gradient_boosting"):
                model.save(path,cfg)
            else:
                model.save(path)
            estimated=model.predict(x,rpm,fs)
            restored=load_predictor(slug,output).predict(x,rpm,fs)
            if not np.isfinite(restored[START:]).all() or estimated.shape!=x.shape:
                raise ArithmeticError(f"{slug} failed synthetic train/replay")
            np.testing.assert_allclose(estimated[START:],restored[START:],rtol=0,atol=3e-6)
            result.append(slug)
    return result
