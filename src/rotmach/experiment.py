"""Training, artifact round trips and measured common-index evaluation."""
import copy
import hashlib
import json
import pickle
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge, LinearRegression
from .io_matlab import load_records, sha256
from .models import build_network
from .windows import make_windows

EVAL_BURNIN = 256


def metrics(y, p):
    """Float64 MSE, RMSE, MAE and NMSE using target mean-square denominator."""
    e = np.asarray(y, dtype=float)-np.asarray(p, dtype=float)
    mse = float(np.mean(e**2))
    return dict(mse=mse, rmse=mse**0.5, mae=float(np.mean(np.abs(e))),
                nmse=mse/float(np.mean(np.asarray(y)**2)))


def predict_network(model, x, batch=512):
    """Predict arrays of windows on CPU, float32, in bounded batches."""
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(np.ascontiguousarray(x[i:i+batch]))).numpy()
                               for i in range(0, len(x), batch)])


def load_model_and_predict(root, run_id, record):
    """Load a locally generated trusted checkpoint; return full-length yhat.

    Initial lag entries are NaN and must not be included in metrics.
    Pickle artifacts must never be loaded from an untrusted source.
    """
    root = Path(root)
    cfg = json.loads((root/"experiments/runs"/run_id/"config.json").read_text())
    lag, kind = cfg["lag"], cfg["kind"]
    x, _, indices = make_windows(record, lag, kind)
    if kind in ("zero", "mean", "persistence"):
        checkpoint = np.load(root/"models"/(run_id+".npz"))
        pred = (record["x"][indices-1] if kind == "persistence"
                else np.full(len(indices), float(checkpoint["value"])))
    elif cfg["family"] == "linear":
        with open(root/"models"/(run_id+".pkl"), "rb") as f:
            model = pickle.load(f)
        pred = model.predict(x)
    else:
        if kind == "shift_tcn":
            x[:, 1] = np.roll(record["spdnorm"], len(record["x"])//3)[np.arange(lag)[None, :]+indices[:, None]-lag]
        model = build_network(kind, lag, cfg["channels"])
        model.load_state_dict(torch.load(root/"models"/(run_id+".pt"), map_location="cpu", weights_only=True))
        pred = predict_network(model, x)
    full = np.full(record["N"], np.nan)
    full[indices] = pred
    return full


def train_run(root, cfg, train, valid):
    root = Path(root)
    kind, lag, seed = cfg["kind"], cfg["lag"], cfg.get("seed", 17)
    run_id = f"{kind}_L{lag}_s{seed}" + (f"_a{cfg['alpha']:g}" if "alpha" in cfg else "")
    folder = root/"experiments/runs"/run_id
    folder.mkdir(parents=True, exist_ok=True)
    cfg = dict(cfg, run_id=run_id, burnin=EVAL_BURNIN, device="cpu", precision="float64 linear/float32 neural",
               train_sha256=sha256(root/"data/raw/sigtrain.mat"),valid_sha256=sha256(root/"data/raw/sigvalid.mat"))
    cfg["family"] = "linear" if kind in ("zero", "mean", "persistence", "ar", "arx", "lpv", "ols") else "neural"
    cfg['source_sha256']={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}
    cfg['config_fingerprint']=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    (folder/"config.json").write_text(json.dumps(cfg, indent=2))
    begin = time.perf_counter()
    x, y, _ = make_windows(train, lag, kind)
    xv, yv, iv = make_windows(valid, lag, kind)
    prep_s = time.perf_counter()-begin
    history = []
    start = time.perf_counter()
    if kind in ("zero", "mean", "persistence"):
        value = float(train["x"].mean()) if kind == "mean" else 0.
        np.savez(root/"models"/(run_id+".npz"),value=value)
        pred = valid["x"][iv-1] if kind == "persistence" else np.full(len(iv),value)
        params = 0 if kind != "mean" else 1
    elif cfg["family"] == "linear":
        model = LinearRegression() if kind == "ols" else Ridge(alpha=cfg.get("alpha", 1.0), fit_intercept=True, solver="cholesky")
        model.fit(x, y)
        params = int(model.coef_.size+1)
        with open(root/"models"/(run_id+".pkl"), "wb") as f:
            pickle.dump(model, f)
        pred = model.predict(xv)
    else:
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.use_deterministic_algorithms(True)
        model = build_network(kind, lag, cfg["channels"])
        if kind == "shift_tcn":
            # Explicitly noncausal negative control: off-line speed phase roll only.
            # It is excluded from causal model recommendations and main best selection.
            x[:, 1] = np.roll(train["spdnorm"], len(train["x"])//3)[np.arange(lag)[None, :]+np.arange(lag,len(train["x"]))[:, None]-lag]
            xv[:, 1] = np.roll(valid["spdnorm"], len(valid["x"])//3)[np.arange(lag)[None, :]+iv[:, None]-lag]
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        params = sum(p.numel() for p in model.parameters())
        xt = torch.from_numpy(np.ascontiguousarray(x))
        yt = torch.from_numpy(y.astype(np.float32))
        generator = torch.Generator().manual_seed(seed)
        best, stale = float("inf"), 0
        for epoch in range(cfg["epochs"]):
            model.train()
            order = torch.randperm(len(yt),generator=generator)
            total = 0.
            for inds in order.split(cfg["batch_size"]):
                opt.zero_grad()
                loss = torch.mean((model(xt[inds])-yt[inds])**2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                opt.step()
                total += loss.item()*len(inds)
            vp = predict_network(model,xv)
            vm = metrics(yv[iv>=EVAL_BURNIN], vp[iv>=EVAL_BURNIN])["mse"]
            history.append(dict(epoch=epoch+1,train_mse=total/len(yt),valid_mse=vm))
            if vm < best:
                best, state, stale = vm, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
            if stale >= cfg["patience"]:
                break
        model.load_state_dict(state)
        torch.save(state,root/"models"/(run_id+".pt"))
        pred = predict_network(model,xv)
    training_s = time.perf_counter()-start
    pd.DataFrame(history,columns=["epoch","train_mse","valid_mse"]).to_csv(folder/"history.csv",index=False)
    full = load_model_and_predict(root,run_id,valid)
    error = float(np.max(np.abs(full[iv]-pred)))
    if error > 1e-5:
        raise AssertionError(f"Checkpoint roundtrip differs: {error}")
    # Timings include checkpoint load, window preparation, and full-record scoring,
    # the same public inference path for all models; one warmup plus 3 repeats.
    load_model_and_predict(root,run_id,valid)
    times=[]
    for _ in range(3):
        start=time.perf_counter(); load_model_and_predict(root,run_id,valid)
        times.append(time.perf_counter()-start)
    keep=iv>=EVAL_BURNIN
    idx=iv[keep]
    np.savez_compressed(folder/"predictions.npz",target_indices=idx,time=valid["t"][idx],
                        y_true=yv[keep],y_pred=pred[keep],residual=yv[keep]-pred[keep],
                        record_id=valid["record_id"],run_id=run_id)
    row=dict(run_id=run_id,kind=kind,lag=lag,seed=seed,split="validation",causal_status="NONCAUSAL offline speed control" if kind=="shift_tcn" else "past-only causal",n_targets=len(idx),
             start_index=int(idx[0]),end_index=int(idx[-1]),parameters=params,
             preprocessing_s=prep_s,train_wall_s=training_s,inference_record_median_s=float(np.median(times)),
             inference_min_s=min(times),inference_max_s=max(times),
             epochs_completed=len(history),roundtrip_max_error=error,**metrics(yv[keep],pred[keep]))
    artifacts=list((root/"models").glob(run_id+".*"))
    row["artifact_bytes"]=artifacts[0].stat().st_size
    row["artifact_sha256"]=sha256(artifacts[0])
    (folder/"metrics.json").write_text(json.dumps(row,indent=2))
    print(json.dumps({k:row[k] for k in ("run_id","mse","train_wall_s")}),flush=True)
    return row


def experiment_plan():
    """Bounded predeclared suite: no fault/test data used in training/selection."""
    base=[dict(kind=k,lag=256,seed=17) for k in ("zero","mean","persistence","ols")]
    for lag in (32,64,128,256):
        for kind in ("ar","arx","lpv"):
            base.append(dict(kind=kind,lag=lag,seed=17,alpha=1.0))
    common=dict(channels=8,learning_rate=.003,weight_decay=.0001,batch_size=512,epochs=10,patience=3)
    neural=[dict(common,kind=k,lag=64,seed=s) for k in ("mlp","gru","tcn","speed_tcn") for s in ((17,29,43) if k in ("tcn","speed_tcn") else (17,))]
    neural += [dict(common,kind=k,lag=l,seed=17) for k in ("tcn","speed_tcn") for l in (32,128)]
    neural += [dict(common,kind=k,lag=64,seed=17) for k in ("late_tcn","shift_tcn")]
    # 9-channel unconditioned TCN is a capacity control (> FiLM 8-channel params).
    neural.append(dict(common,kind="capacity_tcn",lag=64,seed=17,channels=9))
    return base+neural


def collect(root):
    root=Path(root)
    rows=[json.loads(p.read_text()) for p in sorted((root/"experiments/runs").glob("*/metrics.json"))]
    for row in rows:
        row['causal_status']='NONCAUSAL offline speed control' if row['kind']=='shift_tcn' else 'past-only causal'
    df=pd.DataFrame(rows).sort_values("mse")
    df.to_csv(root/"reports/results_summary.csv",index=False)
    df.to_json(root/"experiments/registry.json",orient="records",indent=2)
    df.to_csv(root/"experiments/registry.csv",index=False)
    df[df.kind.isin(["ar","arx","lpv","ols","mean","zero","persistence"])].to_csv(root/"reports/baseline_comparison.csv",index=False)
    df[~df.kind.isin(["ar","arx","lpv","ols","mean","zero","persistence"])].to_csv(root/"reports/model_ablation.csv",index=False)
    eligible=df[df.kind!="shift_tcn"]
    winner=eligible.iloc[0].to_dict()
    (root/"experiments/best_model.json").write_text(json.dumps(winner,indent=2))
    (root/"models/checkpoints_manifest.json").write_text(json.dumps([{k:r[k] for k in ("run_id","artifact_bytes","artifact_sha256","roundtrip_max_error")} for r in rows],indent=2))
    return df


def run_suite(root, minimal=False, selected=None):
    root=Path(root)
    torch.set_num_threads(2)
    train=load_records(root/"data/raw/sigtrain.mat")[0]
    valid=load_records(root/"data/raw/sigvalid.mat")[0]
    plan_path=root/"config/experiment_plan.json"
    plan=json.loads(plan_path.read_text()) if plan_path.exists() else experiment_plan()
    if not plan_path.exists():
        plan_path.write_text(json.dumps(plan,indent=2))
    if minimal:
        plan=[dict(kind="ar",lag=256,seed=17,alpha=1.0)]
    if selected:
        plan=[c for c in plan if c["kind"]==selected]
    for cfg in plan:
        train_run(root,cfg,train,valid)
    return collect(root)
