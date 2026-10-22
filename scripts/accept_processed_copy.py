"""Accept an exact immutable copy of CPU-audited arrays; run within Slurm."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(16<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--reference-qa',required=True)
    p.add_argument('--compute-node', action='store_true', help='Explicitly authorize CPU checks on a standalone compute server, never a cluster login node.')
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') and not a.compute_node:p.error('full-array checksum needs CPU allocation or an authorized standalone compute server')
    root=Path(a.root);ref=json.loads(Path(a.reference_qa).read_text())
    if not ref['retained_array_integrity_pass']:raise ValueError('reference arrays failed integrity')
    manifest=root/'manifest.json'
    if digest(manifest)!=ref['manifest_sha256']:raise ValueError('canonical manifest mismatch')
    names={'manifest.json':ref['manifest_sha256']}
    for split in ('train','val','test'):
        for kind in ('signals','labels'):names[f'{split}_{kind}.npy']=ref['splits'][split][f'{kind}_sha256']
    receipt={'status':'accepted','manifest_sha256':ref['manifest_sha256'],
        'files':{},'canonical_reference_qa_sha256':digest(a.reference_qa),
        'acceptance_scope':'Exact content match to audited canonical arrays; no claim of newly reconstructed legacy EDF provenance.',
        'dataset':ref['dataset']}
    for name,expected in names.items():
        path=root/name;got=digest(path)
        if got!=expected:raise ValueError(f'checksum mismatch: {path}')
        path.chmod(0o444)
        st=path.stat();receipt['files'][name]={'size':st.st_size,'mtime_ns':st.st_mtime_ns,'sha256':got}
        print(f'verified {name}',flush=True)
    tmp=root/'ACCEPTED.json.tmp';tmp.write_text(json.dumps(receipt,indent=2)+'\n');os.replace(tmp,root/'ACCEPTED.json')
    print(f'ACCEPTED {root}',flush=True)
if __name__=='__main__':main()
