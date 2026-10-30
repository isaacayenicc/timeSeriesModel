"""Render per-artifact model documentation from recorded configurations."""
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pandas as pd
from rotmach.models import build_network

def refresh(root):
    root=Path(root); dest=root/'docs/model_cards';dest.mkdir(exist_ok=True)
    df=pd.read_csv(root/'reports/results_summary.csv')
    for row in df.to_dict('records'):
        cfg=json.loads((root/'experiments/runs'/row['run_id']/'config.json').read_text())
        arch=str(build_network(cfg['kind'],cfg['lag'],cfg['channels'])) if cfg['family']=='neural' else cfg['kind']+' least-squares/ridge causal lag predictor or fixed baseline'
        text=f"# {row['run_id']}\n\n{row['causal_status']}. Teacher-forced one-step validation, indices 256–23999, provided dimensionless x. Lag {cfg['lag']}; fs=3200 Hz.\n\n```text\n{arch}\n```\n\n```json\n{json.dumps(cfg,indent=2)}\n```\n\nMSE {row['mse']:.10g}; parameters {row['parameters']}; measured training wall {row['train_wall_s']:.6g}s; public record inference median {row['inference_record_median_s']:.6g}s. Hardware and timing definitions: ../../reports/timing_and_hardware.md. Artifact SHA256 {row['artifact_sha256']}. Predictions in ../../experiments/runs/{row['run_id']}/predictions.npz.\n\nTraining/early stopping uses supplied train/validation traces; no independent test exists. No physical fault accuracy established. This model is not a reproduced Spa LPV-ARMA. All speed-aware primary inputs are past-only; shift_tcn is a deliberately noncausal offline speed control excluded from best-model selection. Linear polynomial design is not additionally standardized; regularization acts on its supplied basis. See config for exact optimizer and capacity.\n"
        (dest/(row['run_id']+'.md')).write_text(text)
    (root/'reports/results_summary.md').write_text('# Common-index validation results\n\n'+df[['run_id','causal_status','mse','parameters','train_wall_s','inference_record_median_s']].to_markdown(index=False)+'\n\nOne validation trace used for selection. Inference includes loading and window preparation. Shift-speed control is noncausal and not eligible for recommendation.\n')
if __name__=='__main__':refresh(Path(__file__).resolve().parents[1])
