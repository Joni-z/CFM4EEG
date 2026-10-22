"""Versioned, deterministic CHB-MIT/TUAB rebuild. Never overwrites old datasets.

All signal work runs inside a CPU allocation. Output is usable only after READY.json.
One-recording shards permit retries; row provenance and source checksums are retained.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import numpy as np
import yaml

SPLITS = ('train', 'val', 'test')
EXPECTED_CHB = {'train': (316205, 2883), 'val': (21094, 60), 'test': (17224, 114)}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def canonical_patient(case):
    return 'chb01' if case == 'chb21' else case


def assert_disjoint(groups):
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            overlap = set(groups[a]) & set(groups[b])
            if overlap:
                raise ValueError(f'patient overlap {a}/{b}: {sorted(overlap)}')


def global_tuab_split(train_items, test_items, ratio=0.8):
    """Assign a patient exactly once, across both diagnosis directories."""
    patients = sorted({x['patient'] for x in train_items})
    n = int(len(patients) * ratio)
    if not 0 < n < len(patients):
        raise ValueError('empty train/validation patient split')
    train = set(patients[:n])
    result = {'train': [], 'val': [], 'test': list(test_items)}
    for item in train_items:
        result['train' if item['patient'] in train else 'val'].append(item)
    assert_disjoint({s: [x['patient'] for x in result[s]] for s in SPLITS})
    return result


def window_plan(n_samples, win, stride, seizures=(), augmentation=None):
    """Coordinates in resampled samples. Labels follow half-open overlap."""
    natural = np.arange(0, max(0, n_samples - win + 1), stride, dtype=np.int64)
    starts = natural.tolist()
    augmented = [False] * len(starts)
    if augmentation:
        for lo, hi in seizures:
            a = max(0, lo - augmentation['pre_samples'])
            b = min(n_samples, hi + augmentation['post_samples'])
            extra = list(range(a, b - win + 1, augmentation['stride_samples']))
            starts.extend(extra)
            augmented.extend([True] * len(extra))
    y = [int(any(min(st + win, hi) > max(st, lo) for lo, hi in seizures))
         for st in starts]
    return np.asarray(starts, dtype=np.int64), np.asarray(y, dtype=np.int64), np.asarray(augmented, dtype=np.bool_)


def parse_summary_strict(path):
    text = Path(path).read_text()
    records = {}
    for block in re.split(r'(?=File Name:)', text):
        m = re.search(r'^File Name:\s*(\S+)', block, re.M)
        if not m:
            continue
        name = m.group(1)
        if name in records:
            raise ValueError(f'duplicate annotation {name}')
        starts = [int(x) for x in re.findall(r'Seizure(?: \d+)? Start Time:\s*(\d+)\s*seconds', block)]
        ends = [int(x) for x in re.findall(r'Seizure(?: \d+)? End Time:\s*(\d+)\s*seconds', block)]
        n = re.search(r'Number of Seizures in File:\s*(\d+)', block)
        if len(starts) != len(ends) or (n and int(n.group(1)) != len(starts)):
            raise ValueError(f'annotation mismatch {path}:{name}')
        if any(a >= b for a, b in zip(starts, ends)):
            raise ValueError(f'invalid seizure interval {path}:{name}')
        records[name] = list(zip(starts, ends))
    return records


def collect(cfg, raw_root):
    raw_root = Path(raw_root)
    groups = {s: [] for s in SPLITS}
    annotations = {}
    if cfg['dataset'] == 'chbmit':
        official = {x.strip() for x in (raw_root / 'RECORDS').read_text().splitlines() if x.strip()}
        positive_records = {x.strip() for x in (raw_root / 'RECORDS-WITH-SEIZURES').read_text().splitlines() if x.strip()}
        annotations['RECORDS'] = sha256(raw_root / 'RECORDS')
        annotations['RECORDS-WITH-SEIZURES'] = sha256(raw_root / 'RECORDS-WITH-SEIZURES')
        assigned_cases = sum((cfg['split'][s] for s in SPLITS), [])
        if len(assigned_cases) != len(set(assigned_cases)):
            raise ValueError('case assigned twice')
        if set(assigned_cases) != {f'chb{i:02d}' for i in range(1, 25)}:
            raise ValueError('CHB requires all 24 cases')
        for split in SPLITS:
            for case in sorted(cfg['split'][split]):
                summary = raw_root / case / f'{case}-summary.txt'
                ann = parse_summary_strict(summary)
                annotations[str(summary.relative_to(raw_root))] = sha256(summary)
                files = sorted((raw_root / case).glob('*.edf'))
                if not files:
                    raise FileNotFoundError(case)
                for path in files:
                    rel = str(path.relative_to(raw_root))
                    if rel not in official:
                        raise ValueError(f'EDF not in official RECORDS: {rel}')
                    seizures = ann.get(path.name, [])
                    if bool(seizures) != (rel in positive_records):
                        # Official RECORDS-WITH-SEIZURES misnames chb07_19 as
                        # chb07_18. Detailed summary and chb07_19 EDF annotation
                        # agree on 13688--13831s. Permit ONLY this checked typo.
                        checked = {'chb07/chb07_18.edf': ([], True),
                                   'chb07/chb07_19.edf': ([(13688, 13831)], False)}
                        if checked.get(rel) != (seizures, rel in positive_records):
                            raise ValueError(f'summary/RECORDS-WITH-SEIZURES disagree: {rel}')
                        annotations['known_seizure_index_disagreement:'+rel] = {
                            'index_has_seizure':rel in positive_records,
                            'summary_intervals':seizures, 'decision':'use detailed summary, confirmed by EDF seizure annotation'}
                        if seizures:
                            event_file=Path(str(path)+'.seizures')
                            event_digest=sha256(event_file)
                            if event_digest != 'f70b7cf41c1384e181e22873cb01096be28b4ac6eb67b4bd73220e6535949b5e':
                                raise ValueError('chb07_19 seizure annotation changed; review index exception again')
                            annotations[str(event_file.relative_to(raw_root))]=event_digest
                    # chb24 summary enumerates seizure EDFs only. Missing entries
                    # are accepted ONLY when official seizure list says negative.
                    if path.name not in ann and case != 'chb24':
                        raise ValueError(f'EDF missing from summary: {rel}')
                    groups[split].append({'recording': rel, 'case': case,
                        'patient': canonical_patient(case), 'seizures_sec': seizures})
        found = {r['recording'] for rows in groups.values() for r in rows}
        if found != official:
            raise ValueError(f'official EDF list mismatch: missing={sorted(official-found)} extra={sorted(found-official)}')
    else:
        official_groups = {}
        for split in ('train', 'eval'):
            rows = []
            for name, label in [('normal', 0), ('abnormal', 1)]:
                directory = raw_root / split / name / cfg['channel_std']
                if not directory.is_dir():
                    raise FileNotFoundError(directory)
                for path in sorted(directory.glob('*.edf')):
                    rows.append({'recording': str(path.relative_to(raw_root)),
                                 'patient': path.name.split('_')[0], 'label': label})
            official_groups[split] = rows
        groups = global_tuab_split(official_groups['train'], official_groups['eval'], cfg['split']['ratio'])
    assert_disjoint({s: [x['patient'] for x in groups[s]] for s in SPLITS})
    for split, rows in groups.items():
        rows.sort(key=lambda r: r['recording'])
        for r in rows:
            r['split'] = split
    return groups, annotations


def process_record(task):
    """All imports and raw-signal computation are inside allocated workers."""
    item, cfg, raw_root, out, expected_sha, allowed_exclusions = task
    import mne
    from .common import resample_to, norm_q95, preprocess_signal
    from .chbmit import pick_channels
    from .tuh_common import build_bipolar, MissingChannels
    raw_path = Path(raw_root) / item['recording']
    key = hashlib.sha256(item['recording'].encode()).hexdigest()[:24]
    shard = Path(out) / '.records' / key
    shard.mkdir(parents=True, exist_ok=True)
    receipt = shard / 'record.json'
    actual_sha = sha256(raw_path)
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(f'raw checksum changed: {raw_path}')
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved['source_sha256'] != actual_sha:
            raise ValueError(f'raw changed since cached shard: {raw_path}')
        for name, digest in saved.get('files_sha256', {}).items():
            if sha256(shard / name) != digest:
                raise ValueError(f'corrupt cache {shard/name}')
        return saved
    result = dict(item, source_sha256=actual_sha, source_bytes=raw_path.stat().st_size,
                  shard=key)
    raw = mne.io.read_raw_edf(str(raw_path), preload=False, verbose='ERROR')
    fs_in = float(raw.info['sfreq'])
    if cfg['dataset'] == 'chbmit' and fs_in != 256:
        raise ValueError(f'unexpected CHB sampling rate: {fs_in}')
    try:
        if cfg['dataset'] == 'chbmit':
            sig = pick_channels(raw, cfg['channels']) * 1e6
        else:
            sig = build_bipolar(raw) * 1e6
    except (KeyError, MissingChannels) as exc:
        if Path(item['recording']).name not in allowed_exclusions:
            raise RuntimeError(f'new exclusion, review required: {item["recording"]}') from exc
        result['excluded'] = str(exc)
        dump(receipt, result)
        return result
    del raw
    fs = int(cfg['sample_rate'])
    if cfg['dataset'] == 'chbmit':
        sig = np.ascontiguousarray(resample_to(sig.astype(np.float64), fs_in, fs), dtype=np.float32)
    else:
        sig = preprocess_signal(sig, fs_in, fs_out=fs, band=tuple(cfg['band']), notch_freq=cfg['notch'])
    win, stride = int(cfg['window_sec'] * fs), int(cfg['stride_sec'] * fs)
    seizures = [(int(s*fs), int(e*fs)) for s,e in item.get('seizures_sec', [])]
    if any(s < 0 or e > sig.shape[-1] for s,e in seizures):
        raise ValueError(f'annotation outside recording {item["recording"]}')
    augmentation = None
    if cfg['dataset'] == 'chbmit' and item['split'] == 'train' and cfg['augment_seizure']['enabled']:
        a = cfg['augment_seizure']
        augmentation = {'pre_samples': int(a['pre_sec']*fs), 'post_samples': int(a['post_sec']*fs), 'stride_samples': int(a['stride_sec']*fs)}
    starts, labels, augmented = window_plan(sig.shape[-1], win, stride, seizures, augmentation)
    if not len(starts):
        raise ValueError(f'unexpected short EDF: {item["recording"]}')
    if cfg['dataset'] == 'tuab':
        labels[:] = item['label']
    X = np.lib.format.open_memmap(shard/'signals.npy', mode='w+', dtype=np.float32,
                                 shape=(len(starts), sig.shape[0], win))
    scales = np.lib.format.open_memmap(shard/'q95_denominators.npy', mode='w+', dtype=np.float64,
        shape=(len(starts), sig.shape[0], 1)) if cfg['dataset']=='chbmit' else None
    for k in range(0, len(starts), 128):
        block = np.stack([sig[:, t:t+win] for t in starts[k:k+128]])
        if scales is not None:
            denominator = np.quantile(np.abs(block), 0.95, axis=-1, keepdims=True) + 1e-8
            scales[k:k+len(block)] = denominator
            block = block / denominator
        else:
            block = block / 100.0
        if not np.isfinite(block).all():
            raise ValueError(f'nonfinite signal {item["recording"]}')
        X[k:k+len(block)] = block
    X.flush(); del X, sig
    if scales is not None: scales.flush(); del scales
    for name, arr in [('labels',labels), ('start_samples',starts), ('augmented',augmented)]:
        np.save(shard/f'{name}.npy', arr)
    result.update(n_windows=len(starts), class_counts=dict(Counter(map(str,labels.tolist()))),
                  n_augmented=int(augmented.sum()), sample_rate=fs, source_sample_rate=fs_in,
                  window_samples=win, stride_samples=stride,
                  files_sha256={p.name:sha256(p) for p in sorted(shard.glob('*.npy'))})
    dump(receipt, result)
    return result


def assemble(out, cfg, groups, results, annotations, identity):
    out = Path(out)
    result_by_record = {r['recording']:r for r in results}
    manifest = {'dataset':cfg['dataset'], 'version':'rebuild-20261021-v1', 'protocol':cfg,
                'build_identity':identity, 'source_annotations_sha256':annotations,
                'created_utc':datetime.now(timezone.utc).isoformat(), 'splits':{},
                'raw_sha256':{r['recording']:r['source_sha256'] for r in results},
                'excluded':[r for r in results if 'excluded' in r], 'artifacts':{}}
    win = int(cfg['window_sec'] * cfg['sample_rate'])
    for split in SPLITS:
        records = [result_by_record[r['recording']] for r in groups[split]]
        records = [r for r in records if 'excluded' not in r]
        n = sum(r['n_windows'] for r in records)
        X = np.lib.format.open_memmap(out/f'{split}_signals.npy', mode='w+', dtype=np.float32,
                                     shape=(n,len(cfg['channels']),win))
        y = np.lib.format.open_memmap(out/f'{split}_labels.npy', mode='w+', dtype=np.int64,shape=(n,))
        scales = np.lib.format.open_memmap(out/f'{split}_q95_denominators.npy',mode='w+',dtype=np.float64,
            shape=(n,len(cfg['channels']),1)) if cfg['dataset']=='chbmit' else None
        offset = 0
        with open(out/f'{split}_provenance.jsonl','w') as f:
            for record_index,r in enumerate(records):
                shard = out/'.records'/r['shard']
                x = np.load(shard/'signals.npy',mmap_mode='r')
                lab = np.load(shard/'labels.npy')
                starts = np.load(shard/'start_samples.npy')
                aug = np.load(shard/'augmented.npy')
                count = len(lab)
                if split != 'train' and (aug.any() or not np.array_equal(starts,np.arange(count)*win)):
                    raise ValueError(f'evaluation grid is not label independent: {r["recording"]}')
                for k in range(0,count,128):
                    X[offset+k:offset+min(k+128,count)] = x[k:k+128]
                if scales is not None:
                    scales[offset:offset+count] = np.load(shard/'q95_denominators.npy',mmap_mode='r')
                y[offset:offset+count] = lab
                for k,st in enumerate(starts):
                    f.write(json.dumps({'row':offset+k,'record_index':record_index,
                        'patient':r['patient'],'recording':r['recording'],
                        'start_sample':int(st),'stop_sample':int(st+win),
                        'sample_rate':int(cfg['sample_rate']), 'label':int(lab[k]),
                        'augmented':bool(aug[k])},separators=(',',':'))+'\n')
                offset += count
        X.flush(); y.flush()
        if scales is not None: scales.flush(); del scales
        counts = np.bincount(y,minlength=2).tolist()
        if cfg['dataset']=='chbmit' and (n,counts[1]) != EXPECTED_CHB[split]:
            raise ValueError(f'CHB independent count reconstruction mismatch {split}: {(n,counts[1])}')
        subjects=sorted({r['patient'] for r in records})
        manifest['splits'][split]={'subjects':subjects,'n_subjects':len(subjects),
            'n_windows':n, 'class_counts':{str(i):v for i,v in enumerate(counts)},
            'n_recordings':len(records),'shape':list(X.shape[1:]),
            'n_augmented_positive':sum(r['n_augmented'] for r in records)}
        del X,y
        artifacts = [f'{split}_signals.npy',f'{split}_labels.npy',f'{split}_provenance.jsonl']
        if cfg['dataset']=='chbmit': artifacts.append(f'{split}_q95_denominators.npy')
        for name in artifacts:
            manifest['artifacts'][name]={'sha256':sha256(out/name),'bytes':(out/name).stat().st_size}
    assert_disjoint({s:manifest['splits'][s]['subjects'] for s in SPLITS})
    manifest['qc']={'patient_disjoint':True,'eval_label_independent_grid':True,
                    'normalization':cfg['normalization'], 'finite_signal_checked':True,
                    'row_provenance':'split_provenance.jsonl; samples at processed sample_rate',
                    'n_excluded':len(manifest['excluded'])}
    dump(out/'manifest.json',manifest)
    dump(out/'READY.json', {'manifest_sha256':sha256(out/'manifest.json'),
                           'build_identity':identity})
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['chbmit','tuab'],required=True)
    parser.add_argument('--raw-root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--reference-manifest',required=True)
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args()
    if not args.plan_only and not os.environ.get('SLURM_JOB_ID') and not os.environ.get('ALLOW_LOCAL_PREPROCESSING'):
        parser.error('signal processing requires a Slurm allocation')
    cfg=yaml.safe_load(Path(f'configs/datasets/{args.dataset}.yaml').read_text())
    cfg['raw_root']=str(Path(args.raw_root).resolve()); cfg['out_dir']=str(Path(args.output).resolve())
    cfg['protocol_version']='rebuild-20261021-v1'
    if args.dataset=='chbmit':
        cfg['q95_denominators_sidecar'] = True
        cfg['stored_units']='q95-normalized microvolts; recover microvolts with q95_denominators.npy'
        cfg['augment_seizure']['splits']=['train']
        cfg['augment_seizure']['force_positive']=False
        cfg['split']['patient_aliases']={'chb21':'chb01'}
    else:
        cfg['split']['mode']='global_patient_sort_ratio';cfg['split']['per_class']=False
    groups,annotations=collect(cfg,args.raw_root)
    ref=json.loads(Path(args.reference_manifest).read_text())
    raw_names = [Path(r['recording']).name for rows in groups.values() for r in rows]
    if len(raw_names) != len(set(raw_names)):
        raise ValueError('duplicate EDF basenames; legacy source hashes are ambiguous')
    expected_names = {Path(k).name for k in ref['raw_sha256']} | {Path(r['item']).name for r in ref['excluded']}
    if set(raw_names) != expected_names:
        raise ValueError(f'raw cohort changed: missing={sorted(expected_names-set(raw_names))}, extra={sorted(set(raw_names)-expected_names)}')
    # Bind code and libraries so a retry cannot silently mix incompatible shards.
    import importlib.metadata
    versions={k:importlib.metadata.version(k) for k in ('numpy','scipy','mne','PyYAML')}
    files=['preprocessing/rebuild_clinical.py','preprocessing/common.py','preprocessing/chbmit.py','preprocessing/tuh_common.py']
    payload={'protocol':cfg,'records':groups,'annotations':annotations,'source_code_sha256':{p:sha256(p) for p in files},
             'package_versions':versions,'reference_manifest_sha256':sha256(args.reference_manifest)}
    identity=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    out=Path(args.output)
    if out.exists() and any(out.iterdir()):
        old=out/'BUILD.json'
        if not old.exists() or json.loads(old.read_text())['identity']!=identity:
            raise RuntimeError(f'nonmatching/nonempty output; refusing overwrite: {out}')
        if (out/'READY.json').exists():
            raise RuntimeError(f'already complete: {out}')
    out.mkdir(parents=True,exist_ok=True)
    dump(out/'BUILD.json',{'identity':identity,**payload})
    if args.plan_only:
        print(json.dumps({s:len(v) for s,v in groups.items()}));return
    exclusions={r['item'] for r in ref['excluded']}
    tasks=[]
    for split in SPLITS:
        for item in groups[split]:
            expected=ref['raw_sha256'].get(item['recording'],ref['raw_sha256'].get(Path(item['recording']).name))
            tasks.append((item,cfg,args.raw_root,args.output,expected,exclusions))
    results=[]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i,r in enumerate(pool.map(process_record,tasks,chunksize=1)):
            results.append(r)
            print(f'{i+1}/{len(tasks)} {r["recording"]} n={r.get("n_windows",0)} excluded={r.get("excluded",False)}',flush=True)
    actual_exclusions={Path(r['recording']).name for r in results if 'excluded'in r}
    if actual_exclusions!=exclusions:
        raise ValueError(f'exclusion set changed: {actual_exclusions.symmetric_difference(exclusions)}')
    manifest=assemble(out,cfg,groups,results,annotations,identity)
    print(json.dumps({'ready':str(out/'READY.json'),'splits':manifest['splits']},indent=2))

if __name__=='__main__':
    main()
