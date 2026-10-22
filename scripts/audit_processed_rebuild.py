"""Full-array CPU acceptance audit; no mutation of existing datasets.

Run inside Slurm. Multiset hashes permit comparison of imap_unordered builds.
This verifies retained arrays/manifests, not original EDF-to-row provenance.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np


def hash_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def audit(root):
    root=Path(root); mp=root/'manifest.json'
    manifest=json.loads(mp.read_text())
    result={'root':str(root.resolve()),'dataset':manifest['dataset'],
        'manifest_sha256':hash_file(mp),'protocol':manifest['protocol'],
        'provenance_limit':'Legacy arrays generally lack row-to-recording provenance. This audit does not reconstruct it.',
        'splits':{},'patient_overlap':[],'errors':[]}
    sets={s:set(map(str,v['subjects'])) for s,v in manifest['splits'].items()}
    if manifest['dataset']=='chbmit':
        sets={s:{'chb01'if x=='chb21'else x for x in v} for s,v in sets.items()}
    names=list(sets)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            if sets[a]&sets[b]:result['patient_overlap'].append({'splits':[a,b],'patients':sorted(sets[a]&sets[b])})
    for split in ('train','val','test'):
        p=root/f'{split}_signals.npy';lp=root/f'{split}_labels.npy'
        x=np.load(p,mmap_mode='r'); y=np.load(lp,mmap_mode='r')
        if len(x)!=len(y):raise ValueError(f'rowcount mismatch {split}')
        if p.stat().st_size != x.offset+x.nbytes:raise ValueError(f'signal length mismatch {split}')
        if lp.stat().st_size != y.offset+y.nbytes:raise ValueError(f'label length mismatch {split}')
        if x.dtype != np.float32 or y.dtype.kind not in 'iu':raise ValueError(f'unexpected dtype {split}')
        counts=dict(Counter(map(str,np.asarray(y).reshape(-1).tolist())))
        expected={str(k):v for k,v in manifest['splits'][split]['class_counts'].items()}
        if counts!=expected:raise ValueError(f'classcount mismatch {split}: {counts} vs{expected}')
        if len(x)!=manifest['splits'][split]['n_windows']:raise ValueError(f'manifest length mismatch {split}')
        row_hashes=[];sum2=0.; maximum=0.
        for start in range(0,len(x),256):
            block=np.array(x[start:start+256],copy=True)
            if not np.isfinite(block).all():raise ValueError(f'nonfinite signals {split}:{start}')
            sum2+=float(np.square(block,dtype=np.float64).sum())
            maximum=max(maximum,float(np.abs(block).max()))
            for k,row in enumerate(block):
                h=hashlib.sha256();h.update(np.ascontiguousarray(row).tobytes())
                h.update(np.asarray(y[start+k],dtype='<i8').tobytes());row_hashes.append(h.digest())
        row_hashes.sort();h=hashlib.sha256()
        for digest in row_hashes:h.update(digest)
        result['splits'][split]={'shape':list(x.shape),'labels_shape':list(y.shape),
            'dtype':str(x.dtype),'n_rows':len(x),'class_counts':counts,
            'signals_sha256':hash_file(p),'labels_sha256':hash_file(lp),
            'row_multiset_sha256':h.hexdigest(),'duplicate_row_label_pairs':len(row_hashes)-len(set(row_hashes)),
            'rms':float(np.sqrt(sum2/x.size)),'max_abs':maximum,
            'patients':sorted(sets[split])}
    if result['patient_overlap']:result['errors'].append('patient split overlap')
    result['retained_array_integrity_pass']=not result['errors']
    result['raw_source_reconstruction_verified']=False
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('datasets',nargs='+');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') and not os.environ.get('ALLOW_LOCAL_PREPROCESSING'):p.error('CPU allocation required')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    for ds in a.datasets:
        print(f'auditing {ds}',flush=True)
        r=audit(Path(a.root)/ds);target=out/f'{ds}.json';tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2)+'\n');os.replace(tmp,target)
        print(f'{ds}: integrity={r["retained_array_integrity_pass"]}',flush=True)
        if not r['retained_array_integrity_pass']:raise RuntimeError(f'{ds} failed acceptance')
if __name__=='__main__':main()
