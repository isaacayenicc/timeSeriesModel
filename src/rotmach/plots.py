"""Auditable figures from immutable input records and saved predictions.

All predictor comparisons use validation indices 256..23999. Frequency-domain
and Hilbert/ACF transforms are retrospective diagnostics, never model inputs.
"""
from pathlib import Path
import json
import hashlib
import os
import tempfile
import numpy as np
import pandas as pd
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'rotmach-mpl-cache'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir())/'rotmach-font-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch, spectrogram
from .io_matlab import load_records, sha256
from .diagnostics import signal_features, normalized_acf


PALETTE = ['#087e8b', '#264653', '#e76f51', '#6c4b8c', '#e9a23b']


def render_figures(root):
    """Save figures, underlying derived tables, and an input-hashed manifest."""
    root = Path(root).resolve()
    out = root / 'reports/figures'
    out.mkdir(parents=True, exist_ok=True)
    train = load_records(root / 'data/raw/sigtrain.mat')[0]
    valid = load_records(root / 'data/raw/sigvalid.mat')[0]
    table = pd.read_csv(root / 'reports/results_summary.csv')
    best = json.loads((root / 'experiments/best_model.json').read_text())
    best_id = best['run_id']
    prediction_path = root / 'experiments/runs' / best_id / 'predictions.npz'
    pred = np.load(prediction_path)
    idx, y, p, residual = (pred[k] for k in ['target_indices','y_true','y_pred','residual'])
    manifest = []
    plt.rcParams.update({'font.size':11, 'axes.spines.top':False,
                         'axes.spines.right':False, 'figure.dpi':120,
                         'axes.prop_cycle':plt.cycler(color=PALETTE)})

    def save(fig, filename, title, inputs):
        fig.tight_layout()
        path = out / filename
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        manifest.append(dict(figure_id=path.stem, path=str(path.relative_to(root)),
                             title=title, script='src/rotmach/plots.py:render_figures',
                             runs=best_id if 'prediction' in inputs else 'all saved runs' if 'results' in inputs else '',
                             input_hashes=';'.join(f'{name}:{digest}' for name,digest in inputs.items()),
                             sha256=sha256(path), destination='report/notebook/slides as referenced'))
    raw_hashes = {f'data/raw/{n}.mat':sha256(root/f'data/raw/{n}.mat') for n in ['sigtrain','sigvalid']}
    result_hash = {'results':sha256(root/'reports/results_summary.csv')}
    pred_hash = {'prediction':sha256(prediction_path)}

    fig, ax = plt.subplots(2,2, figsize=(12,6), sharex='col')
    for j, (r,name) in enumerate([(train,'Training'),(valid,'Validation')]):
        ax[0,j].plot(r['t'],r['x'],lw=.35)
        ax[0,j].set(title=f'{name}: supplied normalized vibration',ylabel='x (dimensionless)')
        ax[1,j].plot(r['t'],r['spd'],lw=1.4)
        ax[1,j].set(xlabel='Time (s)',ylabel='Speed (source units)')
    save(fig,'eda.png','Separate 7.5-second records; speed unit unverified',raw_hashes)

    fig,ax = plt.subplots(2,2,figsize=(12,7))
    for r,name in [(train,'Train'),(valid,'Validation')]:
        f,psd=welch(r['x'],r['fs'],nperseg=1024)
        ax[0,0].semilogy(f,psd,label=name)
        # Fixed 400-sample nonoverlapping windows; final partial block omitted.
        z=r['x'][:len(r['x'])//400*400].reshape(-1,400)
        tm=r['t'][:len(z)*400].reshape(-1,400).mean(axis=1)
        ax[0,1].plot(tm,np.sqrt(np.mean(z*z,axis=1)),label=name)
    ax[0,0].set(xlabel='Frequency (Hz)',ylabel='PSD (dimensionless²/Hz)',title='Welch PSD: 1024-sample Hann segments')
    ax[0,0].legend(); ax[0,1].legend()
    ax[0,1].set(xlabel='Time (s)',ylabel='Window RMS',title='Nonoverlapping 400-sample (0.125 s) windows')
    f,t,sxx=spectrogram(valid['x'],valid['fs'],nperseg=256,noverlap=192)
    im=ax[1,0].pcolormesh(t,f,10*np.log10(sxx+1e-15),shading='auto',cmap='viridis')
    ax[1,0].set(xlabel='Time (s)',ylabel='Frequency (Hz)',title='Validation spectrogram (retrospective EDA)')
    fig.colorbar(im,ax=ax[1,0],label='PSD (dB re dimensionless²/Hz)')
    # Both arrays correspond to aligned time positions, no fitted transformation.
    ax[1,1].scatter(train['spd'][::20],valid['spd'][::20],s=4,alpha=.4)
    ax[1,1].set(xlabel='Training speed (source units)',ylabel='Validation speed (source units)',title='Similar speed trajectories do not prove leakage')
    save(fig,'spectral_eda.png','Frequency/time dependence and operating-history similarity',raw_hashes)

    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for r,name in [(train,'Train'),(valid,'Validation')]:
        rec=(r['xraw']-r['xmean'][0])/r['xstd'][0]
        ax[0].plot(r['t'],rec-r['x'],lw=.6,label=name)
        ax[1].hist(r['spdnorm'],bins=35,histtype='step',density=True,label=name)
    ax[0].set(xlabel='Time (s)',ylabel='Reconstruction difference',title='Stored normalization is reproduced numerically')
    ax[1].set(xlabel='Provided spdnorm (dimensionless)',ylabel='Density',title='Speed support; no clipping or per-record refit')
    for a in ax:a.legend()
    save(fig,'normalization.png','Normalization reconstruction and speed support',raw_hashes)

    # Best measured run per method is selected using validation; not an independent test.
    chosen=table[table.kind!='shift_tcn'].sort_values('mse').groupby('kind',sort=False).first().reset_index().sort_values('mse')
    fig,ax=plt.subplots(figsize=(11,5))
    ax.barh(chosen.kind,chosen.mse,color=PALETTE[0]); ax.invert_yaxis()
    ax.set(xlabel='Validation MSE (same targets t=256..23999)',title='Best observed run per method — selection on validation')
    for i,v in enumerate(chosen.mse):ax.text(v+.008,i,f'{v:.4f}',va='center',fontsize=9)
    ax.set_xlim(0,float(chosen.mse.max())*1.16)
    save(fig,'model_comparison.png','Measured common-index validation error, no independent test',result_hash)

    fig,ax=plt.subplots(2,1,figsize=(12,6),sharex=True)
    keep=(pred['time']>=2.)&(pred['time']<2.08)
    ax[0].plot(pred['time'][keep],y[keep],lw=1.3,label='Observed validation x')
    ax[0].plot(pred['time'][keep],p[keep],lw=1.1,label=f'Prediction: {best_id}')
    ax[0].set(ylabel='Normalized vibration',title='Neutral fixed display interval 2.00–2.08 s; teacher-forced one-step')
    ax[0].legend(fontsize=9)
    ax[1].plot(pred['time'][keep],residual[keep],color=PALETTE[2],lw=1)
    ax[1].set(xlabel='Time (s)',ylabel='Residual x − prediction')
    save(fig,'prediction.png','Fixed-interval validation prediction and residual',pred_hash)

    fig,ax=plt.subplots(figsize=(10,6))
    offsets = {'ar':(-24,-28), 'ols':(-18,14), 'arx':(16,0),
               'mlp':(5,25), 'lpv':(5,-20), 'zero':(8,-20),
               'mean':(8,18), 'tcn':(0,-25), 'speed_tcn':(3,-25),
               'late_tcn':(-50,15), 'capacity_tcn':(12,35)}
    for i,row in chosen.iterrows():
        ax.scatter(row.inference_record_median_s,row.mse,s=45+8*np.log1p(row.parameters),color=PALETTE[i%len(PALETTE)],alpha=.85)
        ax.annotate(row.kind,(row.inference_record_median_s,row.mse),
                    xytext=offsets.get(row.kind,(4,5)),textcoords='offset points',
                    fontsize=9,arrowprops=dict(arrowstyle='-',lw=.5,color='#777777'))
    ax.set(xscale='log',xlabel='Full-record public inference path, median of 3 (s)',ylabel='Validation MSE',title='Accuracy–cost comparison; includes checkpoint load and window creation')
    ax.set_ylim(0.20, float(chosen.mse.max())*1.05)
    ax.grid(alpha=.2)
    save(fig,'efficiency.png','Measured CPU inference vs validation accuracy',result_hash)

    fig,ax=plt.subplots(1,2,figsize=(12,4.5))
    for kind in ['mlp','gru','tcn','speed_tcn']:
        candidates=table[(table.kind==kind)&(table.lag==64)&(table.seed==17)]
        if len(candidates):
            run=candidates.iloc[0].run_id
            h=pd.read_csv(root/'experiments/runs'/run/'history.csv')
            if len(h):
                ax[0].plot(h.epoch,h.train_mse,label=kind)
                ax[1].plot(h.epoch,h.valid_mse,label=kind)
    ax[0].set(title='Training loss (seed 17, lag 64)',xlabel='Epoch',ylabel='MSE')
    ax[1].set(title='Validation loss used for checkpoint selection',xlabel='Epoch',ylabel='MSE')
    for a in ax:a.legend(fontsize=9);a.grid(alpha=.2)
    save(fig,'training_curves.png','Neural training and validation selection curves',result_hash)

    speed=valid['spd'][idx]
    bounds=np.linspace(min(train['spd']),max(train['spd']),7)
    features=[]
    fig,ax=plt.subplots(figsize=(10,4.5))
    for kind in ['ar','lpv','tcn','speed_tcn']:
        r=chosen[chosen.kind==kind]
        if r.empty:continue
        run=r.iloc[0].run_id
        q=np.load(root/'experiments/runs'/run/'predictions.npz')
        errors=(q['y_true']-q['y_pred'])**2
        centers=[];values=[]
        for b,(lo,hi) in enumerate(zip(bounds[:-1],bounds[1:])):
            mask=(speed>=lo)&(speed<hi if b<5 else speed<=hi)
            if np.any(mask):
                centers.append((lo+hi)/2);values.append(float(errors[mask].mean()))
                features.append(dict(run_id=run,kind=kind,speed_lo=lo,speed_hi=hi,n=int(mask.sum()),mse=values[-1]))
        ax.plot(centers,values,'o-',label=kind)
    ax.set(xlabel='Validation target speed, analysis strata (source units)',ylabel='MSE',title='Speed-stratified errors; bins fixed from training speed range')
    ax.legend();ax.grid(alpha=.2)
    pd.DataFrame(features).to_csv(root/'reports/speed_binned_errors.csv',index=False)
    save(fig,'speed_errors.png','Errors by speed regime, target speed used for stratification only',result_hash)

    selected=['raw']+list(chosen[chosen.kind.isin(['ar','lpv','mlp','gru','tcn','speed_tcn'])].run_id)
    rows=[]
    for run in selected:
        if run=='raw':z=y
        else:z=np.load(root/'experiments/runs'/run/'predictions.npz')['residual']
        rows.append(dict(series=run,**signal_features(z)))
    stats=pd.DataFrame(rows)
    stats.to_csv(root/'reports/plots_validation_residual_features.csv',index=False)
    fig,ax=plt.subplots(1,3,figsize=(13,4.5))
    for a,feature in zip(ax,['rms','crest_factor','kurtosis']):
        a.bar(np.arange(len(stats)),stats[feature],color=PALETTE[0])
        a.set_xticks(np.arange(len(stats)),[v.replace('_L','\nL') for v in stats.series],rotation=65,ha='right',fontsize=8)
        a.set(title=feature.replace('_',' ').title(),ylabel='Dimensionless')
    fig.suptitle('Validation signal and residuals only — not real fault-detection performance',fontsize=12)
    save(fig,'residual_features.png','Descriptive reference-condition statistics, equal 23744-sample segments',result_hash)

    fig,ax=plt.subplots(1,3,figsize=(13,4))
    for z,label in [(y,'Observed x'),(residual,'Best-model residual')]:
        ax[0].hist(z,bins=80,density=True,histtype='step',label=label)
        acf=normalized_acf(z,max_lag=250)
        ax[1].plot(np.arange(len(acf)),acf,lw=1,label=label)
        f,psd=welch(z,valid['fs'],nperseg=1024)
        ax[2].semilogy(f,psd,label=label)
    ax[0].set(xlabel='Dimensionless amplitude',ylabel='Density',title='Distribution')
    ax[1].set(xlabel='Lag (samples)',ylabel='Centered normalized ACF',title='Serial dependence')
    ax[2].set(xlabel='Frequency (Hz)',ylabel='PSD (dimensionless²/Hz)',title='Retrospective Welch PSD')
    for a in ax:a.legend(fontsize=8)
    save(fig,'residual_structure.png','Validation residual distribution, autocorrelation and PSD',pred_hash)

    # Matched-L64 seeds separate training stochasticity from trace-level uncertainty.
    seeds=table[(table.kind.isin(['tcn','speed_tcn']))&(table.lag==64)]
    fig,ax=plt.subplots(1,2,figsize=(11,4.5))
    for kind in ['tcn','speed_tcn']:
        r=table[(table.kind==kind)&(table.seed==17)].sort_values('lag')
        ax[0].plot(r.lag,r.mse,'o-',label=kind)
        s=seeds[seeds.kind==kind]
        ax[1].scatter(np.full(len(s),0 if kind=='tcn' else 1),s.mse,s=55,label=kind)
    ax[0].set(xlabel='Past context L (samples)',ylabel='Validation MSE',title='Context ablation: one seed per context');ax[0].legend()
    ax[1].set(xticks=[0,1],xticklabels=['TCN','Speed FiLM TCN'],ylabel='Validation MSE',title='Independent initializations at L=64')
    save(fig,'ablations.png','Context and initialization sensitivity; not independent machine trials',result_hash)
    # Fault analysis is an independent, more expensive workflow. Preserve its
    # manifest when rebuilding validation figures; a sidecar lets its producer
    # update provenance without racing this routine's output.
    combined = pd.DataFrame(manifest)
    manifest_path = root/'reports/figure_manifest.csv'
    retained = []
    if manifest_path.exists():
        old = pd.read_csv(manifest_path)
        retained.append(old[~old.figure_id.isin(combined.figure_id)])
    fault_manifest = root/'reports/fault_figure_manifest.csv'
    if fault_manifest.exists():
        retained.append(pd.read_csv(fault_manifest))
    if retained:
        combined = pd.concat([combined]+retained,ignore_index=True).drop_duplicates('figure_id',keep='last')
    combined.to_csv(manifest_path,index=False)
    return combined


def render_fault_workflow_figure(root, unit='rad_s'):
    """Plot a fixed real recording with explicit unverified angular units.

    The selected record is global ID 1, fixed independently of diagnostic values.
    The speed interpretation is an assumption. No physical order or state is
    inferred. The same density rule/envelope convention as fault_study is used.
    """
    from .fault_io import iter_fault_records
    from .angle_resample import angle_resample
    from .diagnostics import analytic_envelope
    from .experiment import load_model_and_predict

    root=Path(root).resolve()
    plt.rcParams.update({'font.size':11, 'axes.spines.top':False,
                         'axes.spines.right':False, 'figure.dpi':120,
                         'axes.prop_cycle':plt.cycler(color=PALETTE)})
    record=next(iter_fault_records(root/'data/raw/sigfault1_parts',verify=False))
    best_id=json.loads((root/'experiments/best_model.json').read_text())['run_id']
    y=record['x'][256:]-load_model_and_predict(root,best_id,record)[256:]
    t=record['t'][256:]
    speed=record['spd'][256:]
    factors={'rpm':1/60,'rad_s':1/(2*np.pi),'rps':1.}
    if unit not in factors:
        raise ValueError('Explicit unit assumption must be rpm/rad_s/rps')
    density=int(2**np.ceil(np.log2(max(2,record['fs']/np.min(speed*factors[unit])))))
    angle=angle_resample(y,t,speed,unit,density)
    z=angle['signal']
    envelope=analytic_envelope(z,'matlab_mean_restored_candidate')
    acf=normalized_acf(envelope,min(3*density,len(z)-1))
    fig,ax=plt.subplots(2,2,figsize=(12,7))
    ax[0,0].plot(t,y,lw=.6)
    ax[0,0].set(xlim=(t[0],t[0]+.1),xlabel='Time (s)',ylabel='Residual (dimensionless)',
                title='Fixed record 1, first 0.1 s after common burn-in')
    ax[0,1].plot(angle['cycles'],z,lw=.6)
    ax[0,1].set(xlim=(0,3),xlabel='Cycles under declared assumption',ylabel='Resampled residual',
                title=f'Uniform angular grid: {density} samples/cycle')
    ax[1,0].plot(angle['cycles'],envelope,lw=.8)
    ax[1,0].set(xlim=(0,3),xlabel='Cycles under declared assumption',ylabel='Analytic envelope',
                title='Whole-record Hilbert convention; edge effects remain')
    ax[1,1].plot(np.arange(len(acf)),acf,lw=1)
    ax[1,1].axvline(density-1,color=PALETTE[2],ls='--',label=f'Reference lag {density-1}')
    ax[1,1].axvline(density,color=PALETTE[0],ls=':',label=f'Corrected lag {density}')
    ax[1,1].set(xlabel='Angular-domain lag (samples)',ylabel='Centered normalized ACF',
                title='Assumed one-cycle modulation; no verified defect order')
    ax[1,1].legend(fontsize=8)
    fig.suptitle(f'Real candidate record 1 / {best_id} — {unit} speed ASSUMPTION',fontsize=13)
    fig.tight_layout()
    path=root/'reports/figures/fault_angular_workflow.png'
    fig.savefig(path,dpi=200,bbox_inches='tight');plt.close(fig)
    manifest_path=root/'reports/figure_manifest.csv'
    new=pd.DataFrame([dict(figure_id='fault_angular_workflow',path=str(path.relative_to(root)),
        title=f'Neutral record1 residual/angular/envelope/ACF; {unit} assumption only',
        script='src/rotmach/plots.py:render_fault_workflow_figure',runs=best_id,
        input_hashes='manifest:'+sha256(root/'data/raw/sigfault1_parts/manifest.json')+';source_part:'+record['source_part_sha256'],
        sha256=sha256(path),destination='full walkthrough / fault methods')])
    if manifest_path.exists():
        old=pd.read_csv(manifest_path)
        new=pd.concat([old[old.figure_id!='fault_angular_workflow'],new],ignore_index=True)
    new.to_csv(manifest_path,index=False)
    return path
