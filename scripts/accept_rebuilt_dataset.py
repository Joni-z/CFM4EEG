"""Independent structural audit of newly rebuilt clinical arrays and provenance.

Run on a compute node. Every saved byte is checked against the build manifest;
every row label and coordinate is independently checked against source intervals.
"""
import argparse
from collections import Counter,defaultdict
import hashlib,json,os
from pathlib import Path
import numpy as np

# Independently reconstructed from every original EDF header (review/tuab-header-independent.json).
EXPECTED_TUAB_ROWS={'train':(299774,155663),'val':(72736,33138),'test':(36945,17038)}
EXPECTED_TUAB_GROUPS={'train':(2244,1660),'val':(473,416),'test':(276,253)}


def digest(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(16<<20),b''):h.update(b)
 return h.hexdigest()


def validate(root):
 root=Path(root);m=json.loads((root/'manifest.json').read_text());b=json.loads((root/'BUILD.json').read_text());ready=json.loads((root/'READY.json').read_text())
 mh=digest(root/'manifest.json');assert ready['manifest_sha256']==mh
 assert m['build_identity']==b['identity']==ready['build_identity']
 identity_payload=dict(b);identity_payload.pop('identity')
 assert hashlib.sha256(json.dumps(identity_payload,sort_keys=True).encode()).hexdigest()==b['identity']
 required={f'{s}_{kind}' for s in ('train','val','test') for kind in ('signals.npy','labels.npy','provenance.jsonl')}
 if m['dataset']=='chbmit':required.update(f'{s}_q95_denominators.npy' for s in ('train','val','test'))
 assert required.issubset(m['artifacts']),required-set(m['artifacts'])
 receipt={'status':'accepted','dataset':m['dataset'],'manifest_sha256':mh,
          'quality_level':'raw-rebuilt; exact-source-hashes; all-row-provenance-verified',
          'files':{},'row_checks':{}}
 sources={r['recording']:r for rows in b['records'].values() for r in rows}
 subjects={}
 for name,entry in m['artifacts'].items():
  p=root/name;assert p.stat().st_size==entry['bytes'],name
  assert digest(p)==entry['sha256'],name
  p.chmod(0o444);st=p.stat();receipt['files'][name]={'size':st.st_size,'mtime_ns':st.st_mtime_ns,'sha256':entry['sha256']}
 for split in ('train','val','test'):
  x=np.load(root/f'{split}_signals.npy',mmap_mode='r');y=np.load(root/f'{split}_labels.npy',mmap_mode='r');N=len(y)
  assert N==m['splits'][split]['n_windows']
  assert x.shape==(N,16,2000) and x.dtype==np.float32 and y.dtype==np.int64
  assert (root/f'{split}_signals.npy').stat().st_size==x.offset+x.nbytes
  assert (root/f'{split}_labels.npy').stat().st_size==y.offset+y.nbytes
  denominators=None
  if m['dataset']=='chbmit':
   p=root/f'{split}_q95_denominators.npy';denominators=np.load(p,mmap_mode='r')
   assert denominators.shape==(N,16,1) and denominators.dtype==np.float64
   assert p.stat().st_size==denominators.offset+denominators.nbytes
   assert np.isfinite(denominators).all() and (denominators>0).all()
  subjects[split]=set(); seen_eval=set();perrecord=defaultdict(list);n=0
  with open(root/f'{split}_provenance.jsonl')as f:
   for i,line in enumerate(f):
    row=json.loads(line);n+=1;assert row['row']==i and i<N
    source=sources[row['recording']];assert source['split']==split
    if m['dataset']=='chbmit':
     expected_patient='chb01' if source['case']=='chb21' else source['case']
    else:expected_patient=Path(row['recording']).name.split('_')[0]
    assert source['patient']==row['patient']==expected_patient;subjects[split].add(expected_patient)
    assert row['sample_rate']==200 and row['stop_sample']-row['start_sample']==2000
    assert row['start_sample']>=0 and row['label']==int(y[i])
    start,end=row['start_sample'],row['stop_sample']
    if m['dataset']=='chbmit':
     expected=int(any(max(start,s*200)<min(end,e*200) for s,e in source['seizures_sec']))
     assert expected==row['label'],(split,i,'label mismatch')
    else:assert row['label']==source['label']
    if split!='train':
     assert not row['augmented'] and start%2000==0
     key=(row['recording'],start);assert key not in seen_eval;seen_eval.add(key)
     perrecord[row['recording']].append(start)
  assert n==N
  for starts in perrecord.values():assert starts==list(range(0,len(starts)*2000,2000))
  for start in range(0,N,256):assert np.isfinite(x[start:start+256]).all()
  actual={str(k):int(v)for k,v in Counter(y.tolist()).items()}
  assert actual==m['splits'][split]['class_counts']
  if m['dataset']=='chbmit':assert (N,int(y.sum()))=={'train':(316205,2883),'val':(21094,60),'test':(17224,114)}[split]
  if m['dataset']=='tuab':
   assert (N,int(y.sum()))==EXPECTED_TUAB_ROWS[split]
   assert (m['splits'][split]['n_recordings'],len(subjects[split]))==EXPECTED_TUAB_GROUPS[split]
  assert subjects[split]==set(m['splits'][split]['subjects'])
  receipt['row_checks'][split]={'n_rows':N,'patient_count':len(subjects[split]),'class_counts':actual,'natural_eval_grid':split!='train'}
 for a,c in [('train','val'),('train','test'),('val','test')]:assert not subjects[a]&subjects[c]
 for name in ('manifest.json','BUILD.json','READY.json'):
  p=root/name;p.chmod(0o444);st=p.stat();receipt['files'][name]={'size':st.st_size,'mtime_ns':st.st_mtime_ns,'sha256':digest(p)}
 temp=root/'ACCEPTED.json.tmp';temp.write_text(json.dumps(receipt,indent=2)+'\n');os.replace(temp,root/'ACCEPTED.json')
 print(json.dumps(receipt['row_checks'],indent=2),flush=True)
 return receipt

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');p.add_argument('--compute-node',action='store_true');a=p.parse_args()
 if not a.compute_node and not os.environ.get('SLURM_JOB_ID'):p.error('run within allocation or on authorized standalone compute node')
 validate(a.root)
