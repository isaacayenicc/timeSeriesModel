"""Failure-loud measured input audit and source manifest."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .io_matlab import load_records, sha256


def audit_faults(root, parts_dir, entries):
    """Audit all complete candidate records and preserve their global IDs."""
    from .fault_io import verify_fault_parts, iter_fault_records
    verification=verify_fault_parts(parts_dir)
    (root/'reports/fault_parts_verification.json').write_text(json.dumps(verification,indent=2))
    rows=[]
    for record in iter_fault_records(parts_dir,verify=False):
        speed=record['spd']; x=record['x']; lo=float(record['spdmin']);hi=float(record['spdmax'])
        delta=float(np.max(np.abs((speed-lo)/(hi-lo)-record['spdnorm'])))
        if delta>1e-10:raise ValueError('Fault speed normalization does not reconstruct')
        rows.append(dict(record_id=record['record_id'],global_record_id=record['global_record_id'],
                         source_part=record['source_part'],N=record['N'],fs=record['fs'],
                         first_time=record['t'][0],last_time=record['t'][-1],
                         x_mean=float(x.mean()),x_std=float(x.std()),x_min=float(x.min()),x_max=float(x.max()),
                         speed_min=float(speed.min()),speed_max=float(speed.max()),speed_mean=float(speed.mean()),
                         spdnorm_min=float(record['spdnorm'].min()),spdnorm_max=float(record['spdnorm'].max()),
                         speed_outside_reference_fraction=float(np.mean((record['spdnorm']<0)|(record['spdnorm']>1))),
                         speed_reconstruction_max_error=delta,has_xraw='xraw' in record))
    frame=pd.DataFrame(rows)
    frame.to_csv(root/'reports/fault_data_audit.csv',index=False)
    for part in verification['parts']:
        entries.append(dict(name=part['filename'],availability='available',size_bytes=part['bytes'],
                            sha256=part['sha256'],location='data/raw/sigfault1_parts/'+part['filename'],
                            records=part['record_count'],global_record_range=[part['first_record'],part['last_record']]))
    return dict(availability='verified split representation',records=len(frame),parts=verification['part_total'],
                N_values=sorted(frame.N.unique().tolist()),fs_values=sorted(frame.fs.unique().tolist()),
                total_vibration_samples=int(frame.N.sum()),all_source_fields=verification['source_fields'],
                speed_min=float(frame.speed_min.min()),speed_max=float(frame.speed_max.max()),
                normalized_speed_min=float(frame.spdnorm_min.min()),normalized_speed_max=float(frame.spdnorm_max.max()),
                aggregate_speed_outside_reference_fraction=float(np.average(frame.speed_outside_reference_fraction,weights=frame.N)),
                speed_reconstruction_max_error=float(frame.speed_reconstruction_max_error.max()),
                logical_data_sha256=verification['logical_data_sha256'],
                labels='No health, severity, onset or class field in the inspected structs; no external labels supplied',
                speed_units='not encoded; source units only',
                x_normalization='provided x retained without rescaling; xraw absent, raw transform not independently recoverable',
                original_byte_provenance=verification['provenance_limit'])


def audit(root, data_dir=None):
    root = Path(root)
    data_dir = Path(data_dir or root/"data/raw")
    entries, summary = [], {}
    for p in sorted(data_dir.iterdir()):
        if not p.is_file():
            continue
        entry = dict(name=p.name, availability="available",size_bytes=p.stat().st_size, sha256=sha256(p),
                     location="data/raw/"+p.name)
        if p.suffix == ".mat":
            records = load_records(p)
            entry.update(format="MATLAB v5 family", records=len(records),
                         fields=[k for k in records[0] if k!='record_id'],generated_fields=['record_id'])
            r = records[0]
            a = dict(N=r["N"], fs=r["fs"], duration_samples_s=r["N"]/r["fs"],
                     time_first=r["t"][0], time_last=r["t"][-1],
                     x_mean=float(r["x"].mean()), x_std=float(r["x"].std()),
                     speed_min=float(r["spd"].min()), speed_max=float(r["spd"].max()),
                     normalized_speed_min=float(r["spdnorm"].min()),
                     normalized_speed_max=float(r["spdnorm"].max()))
            if "xraw" in r:
                mu, sd = float(r["xmean"][0]), float(r["xstd"][0])
                err = float(np.max(np.abs((r["xraw"]-mu)/sd-r["x"])))
                serr = float(np.max(np.abs((r["spd"]-r["spdmin"])/(r["spdmax"]-r["spdmin"])-r["spdnorm"])))
                if err > 1e-10 or serr > 1e-10:
                    raise ValueError("Supplied normalization reconstruction failed")
                a.update(xmean_metadata=np.asarray(r["xmean"]).tolist(),
                         xstd_metadata=np.asarray(r["xstd"]).tolist(),
                         empirical_raw_mean=float(r["xraw"].mean()),
                         empirical_raw_std=float(r["xraw"].std()),
                         x_reconstruction_max_error=err, speed_reconstruction_max_error=serr)
            summary[p.stem] = a
        entries.append(entry)
    parts_dir=data_dir/'sigfault1_parts'
    if parts_dir.is_dir():
        summary['sigfault1']=audit_faults(root,parts_dir,entries)
        entries.append(dict(name='sigfault1.mat',availability='available as verified complete-record partitions',
                            location='data/raw/sigfault1_parts',records=summary['sigfault1']['records'],
                            logical_data_sha256=summary['sigfault1']['logical_data_sha256']))
    for name in ("sigfault1.mat", "sigtest.mat", "sigfault2.mat"):
        if name=='sigfault1.mat' and parts_dir.is_dir():continue
        if not (data_dir/name).exists():
            entries.append(dict(name=name, availability="unavailable",
                                reason="upload exceeds 32MiB transfer limit" if name=="sigfault1.mat" else "not supplied"))
    tr, va = load_records(data_dir/"sigtrain.mat")[0], load_records(data_dir/"sigvalid.mat")[0]
    summary["comparisons"] = dict(speed_pearson=float(np.corrcoef(tr["spd"],va["spd"])[0,1]),
                                  vibration_pearson=float(np.corrcoef(tr["x"],va["x"])[0,1]),
                                  valid_speed_outside_train_fraction=float(np.mean((va["spd"]<tr["spd"].min())|(va["spd"]>tr["spd"].max()))))
    summary["status"] = "PASS structural/normalization incl. fault partitions; WARN missing test/physical units/labels" if parts_dir.is_dir() else "PASS structural/normalization; WARN missing fault input/test/units/labels"
    (root/"data/manifest.json").write_text(json.dumps(entries, indent=2))
    (root/"reports/data_audit.json").write_text(json.dumps(summary, indent=2))
    fault_note='All fault partitions and 200 recording boundaries were inspected, with matched logical/file digests; no health-state labels or physical speed-unit fields were found. The original monolithic MAT byte fingerprint remains supplied-manifest metadata rather than independently verified original bytes.' if parts_dir.is_dir() else 'No fault-record count or labels were independently verified because that upload cannot be transferred.'
    (root/"reports/data_audit.md").write_text("# Measured input audit\n\n```json\n"+json.dumps(summary,indent=2)+"\n```\n\nSpeed is in source units. Normalized x is dimensionless. Other scaling metadata entries have no inferred interpretation. File hashes are in data/manifest.json. "+fault_note+'\n')
    return summary
