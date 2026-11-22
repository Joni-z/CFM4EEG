import json, collections, re, statistics as st
recs=[json.loads(l) for l in open('raw-results-deduplicated.jsonl')]
pl=[r for r in recs if r['result'].get('model')=='paclock']

FIELDS=['tokenizer_mode','pac_token_mode','interaction_mode','n_bands','freq_mixer','space_over_bands',
        'patch_len','pac_patch_len','d_model','depth','waveform_modulation','waveform_content','coupling_self',
        'coupling_strength','filter_range','dropout','head','spatial_pe','raw_stem','arch','hybrid_gate','fusion_mode',
        'local_residual','learned_montage','content_source','content_scale','n_heads','block_variant','coupling_angle_pool']
CFG=['weight_decay','lr','batch_size','epochs','loss','optimizer','label_smoothing','patience','accum_steps']
DEF={'tokenizer_mode':'raw','pac_token_mode':'measured','interaction_mode':'product','n_bands':8,'freq_mixer':'attention',
     'space_over_bands':False,'pac_patch_len':None,'d_model':128,'depth':6,'waveform_modulation':'none','waveform_content':True,
     'coupling_self':False,'coupling_strength':False,'filter_range':None,'dropout':0.2,'head':'mean','spatial_pe':'index',
     'raw_stem':'linear','arch':'triaxial','hybrid_gate':'none','fusion_mode':'blend','local_residual':False,'learned_montage':False,
     'content_source':None,'content_scale':None,'n_heads':4,'block_variant':None,'coupling_angle_pool':'none','patch_len':200}

def sig(r):
    c=r['result']['config']; mk=c.get('model_kwargs',{}) or {}
    s={}
    for f in FIELDS:
        v=mk.get(f, DEF.get(f))
        if f=='pac_patch_len' and v is None: v=mk.get('patch_len',200)
        if isinstance(v,list): v=json.dumps(v)
        s[f]=v
    s['aug']=bool(mk.get('augmentations'))
    for f in CFG:
        v=c.get(f)
        if isinstance(v,list): v=json.dumps(v)
        s[f]=v
    s['focal']=(c.get('focal_alpha'),c.get('focal_gamma'))
    return s

def canonical(r):
    c=r['result']['config']
    ps=[s['path'] for s in r['sources']]
    if not all(re.search(r'/runs/',p) and 'archive/' not in p and 'quarantine' not in p for p in ps): return False
    if 'processed/' not in c.get('data_root','')+'/' : return False
    if c.get('checkpoint') or c.get('train_subsample') or c.get('freeze_backbone'): return False
    if r['result']['dataset'] in ('tusz_type','apava','eegmat','mumtaz'): return False
    if (r['result'].get('verdict') or {}).get('status')=='mis-configured': return False
    return True

rows=[]
for r in pl:
    if not canonical(r): continue
    R=r['result']
    t=R.get('test') or {}
    pm=R.get('primary_metric')
    if pm is None or pm not in t: continue
    s=sig(r)
    rows.append(dict(ds=R['dataset'],name=R['name'],seed=R['seed'],val=t[pm],metric=pm,sig=s,
                     peak=(R.get('verdict') or {}).get('peak_index'),nev=(R.get('verdict') or {}).get('n_evals'),
                     stop=R.get('stopped_by'),bal=t.get('balanced_acc'),params=R.get('n_params_M')))
print("canonical own-model rows:", len(rows))
# group by (ds, signature)
KEYS=list(rows[0]['sig'].keys())
def skey(s): return tuple((k, s[k] if not isinstance(s[k],tuple) else str(s[k])) for k in KEYS)
groups=collections.defaultdict(list)
for x in rows: groups[(x['ds'],skey(x['sig']))].append(x)
print("groups:", len(groups))
# resolve duplicate seeds within group: keep all (average) but mark
G={}
for k,xs in groups.items():
    byseed=collections.defaultdict(list)
    for x in xs: byseed[x['seed']].append(x['val'])
    seeds={s:st.mean(v) for s,v in byseed.items()}
    vals=list(seeds.values())
    G[k]=dict(ds=k[0],sig=dict(k[1]),seeds=seeds,n=len(vals),mean=st.mean(vals),sd=(st.stdev(vals) if len(vals)>1 else None),
              names=sorted(set(x['name'] for x in xs)),conflict=any(len(v)>1 for v in byseed.values()),
              peaks=[x['peak'] for x in xs], metric=xs[0]['metric'])
json.dump([dict(key=list(k[1]),**{kk:vv for kk,vv in v.items() if kk!='sig'}) for k,v in G.items()], open('groups.json','w'), default=str)

# single-field contrasts
CONTRAST_FIELDS=['tokenizer_mode','pac_token_mode','interaction_mode','n_bands','freq_mixer','space_over_bands','patch_len','pac_patch_len','d_model','depth','waveform_modulation','waveform_content','coupling_self','coupling_strength','filter_range','dropout','head','spatial_pe','raw_stem','arch','aug','weight_decay','n_heads','local_residual','hybrid_gate','learned_montage']
by_ds=collections.defaultdict(list)
for k,v in G.items(): by_ds[v['ds']].append((k,v))
contrasts=[]
for ds,items in by_ds.items():
    for i in range(len(items)):
        for j in range(i+1,len(items)):
            ka,va=items[i]; kb,vb=items[j]
            diff=[f for f in KEYS if va['sig'][f]!=vb['sig'][f]]
            if len(diff)!=1 or diff[0] not in CONTRAST_FIELDS: continue
            f=diff[0]
            a,b=va,vb
            # order: put 'treatment' consistently: sort by value string
            xa,xb=str(a['sig'][f]),str(b['sig'][f])
            if xa>xb: a,b,xa,xb=b,a,xb,xa
            shared=sorted(set(a['seeds'])&set(b['seeds']))
            paired=[b['seeds'][s]-a['seeds'][s] for s in shared]
            contrasts.append(dict(ds=ds,field=f,a=xa,b=xb,mean_a=a['mean'],mean_b=b['mean'],na=a['n'],nb=b['n'],
                                  delta=b['mean']-a['mean'],paired=paired,names_a=a['names'],names_b=b['names'],
                                  ctx={k:a['sig'][k] for k in ['tokenizer_mode','pac_token_mode','n_bands','freq_mixer','space_over_bands','patch_len','d_model','waveform_modulation','dropout','aug'] if k!=f}))
json.dump(contrasts, open('contrasts.json','w'), default=str, indent=0)
print("single-field contrasts:", len(contrasts))
# summary per field/value-pair
summ=collections.defaultdict(list)
for c in contrasts: summ[(c['field'],c['a'],c['b'])].append(c)
out=[]
for (f,a,b),cs in sorted(summ.items(), key=lambda kv:-len(kv[1])):
    ds_deltas=collections.defaultdict(list)
    for c in cs: ds_deltas[c['ds']].append(c['delta'])
    pairedall=[d for c in cs for d in c['paired']]
    line=f"{f}: {a} -> {b} | contrasts={len(cs)} datasets={len(ds_deltas)} | per-ds mean delta: " + \
         ", ".join(f"{d}:{st.mean(v):+.3f}(n={len(v)})" for d,v in sorted(ds_deltas.items())) + \
         (f" | paired-seed deltas n={len(pairedall)} mean={st.mean(pairedall):+.4f} pos={sum(x>0 for x in pairedall)}" if pairedall else "")
    out.append(line)
open('contrast_summary.txt','w').write("\n".join(out))
print("\n".join(out[:80]))
