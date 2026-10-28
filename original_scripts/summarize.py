"""Recompute paired recovery and calibrated record-level detection results."""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--release',action='store_true');args=parser.parse_args()
def read(tag):return json.loads((root/'artifacts'/tag/'records.json').read_text())
def wilson(hits,n):
 p=hits/n;z=1.959963984540054;den=1+z*z/n
 mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return [mid-half,mid+half]
def detection(calibration,final):
 null_names=('aac_only','independent')
 null_cal=[r['predictive_score'] for r in calibration if r['condition'] in null_names]
 index=min(len(null_cal)-1,math.ceil((len(null_cal)+1)*.95)-1)
 threshold=float(np.sort(null_cal)[index])
 null=[r for r in final if r['condition'] in null_names]
 positive=[r for r in final if r['d1']+r['d2']>0]
 fp=sum(r['predictive_score']>threshold for r in null)
 tp=sum(r['predictive_score']>threshold for r in positive)
 located=sum(r['pair_predictive_score']>threshold for r in positive)
 return dict(calibration_records=len(null_cal),threshold=threshold,null_records=len(null),
  false_positives=fp,false_positive_rate=fp/len(null),false_positive_95_interval=wilson(fp,len(null)),
  positive_records=len(positive),true_positive_rate=tp/len(positive),
  true_pair_detection_rate=located/len(positive),
  detection_auroc=float(roc_auc_score([0]*len(null)+[1]*len(positive),[r['predictive_score'] for r in null+positive])),
  by_condition={c:dict(n=len(rr),record_detection=sum(r['predictive_score']>threshold for r in rr)/len(rr),true_pair_detection=sum(r['pair_predictive_score']>threshold for r in rr)/len(rr))
   for c in sorted({r['condition'] for r in positive}) if (rr:=[r for r in positive if r['condition']==c])},
  stress_above_threshold={c:sum(r['predictive_score']>threshold for r in final if r['condition']==c)/sum(r['condition']==c for r in final)
   for c in ('transient','nonsinusoidal')})

base=read('042-v1-release' if args.release else '031-baseline-final');new=read('043-v2-release' if args.release else '032-winner-final')
assert [(r['condition'],r['seed']) for r in base]==[(r['condition'],r['seed']) for r in new]
conditions=sorted({r['condition'] for r in base if r['d1']+r['d2']>0})
deltas=[];per={}
for c in conditions:
 a=[r for r in base if r['condition']==c];b=[r for r in new if r['condition']==c]
 ea=np.array([r['curve_rmse'] if r['valid'] else 1 for r in a]);eb=np.array([r['curve_rmse'] if r['valid'] else 1 for r in b]);deltas.append(ea-eb)
 per[c]=dict(n=len(a),baseline_rmse=float(ea.mean()),new_rmse=float(eb.mean()),relative_reduction=float(1-eb.mean()/ea.mean()),
  new_valid_fraction=float(np.mean([r['valid'] for r in b])),
  new_depth1_mae=float(np.mean([r['depth1_error'] for r in b])),
  new_depth2_mae=float(np.mean([r['depth2_error'] for r in b])),
  new_phase1_mae_deg=float(np.mean([r['phase1_error_deg'] for r in b])) if b[0]['d1']>0 else None)
rng=np.random.default_rng(99101);bootstrap=[]
for _ in range(2000):bootstrap.append(float(np.mean([rng.choice(d,len(d),replace=True).mean() for d in deltas])))
summary=dict(source_commit='b2e9e80c48eee5d93e0d15d22fc665772276f2e1' if args.release else '62fc842',evaluation_unit='independently generated complete record; paired and condition-stratified',
 seed_selection='development only; calibration and final were evaluated after freeze',
 final_records_per_arm=len(new),positive_records_per_arm=sum(len(d) for d in deltas),
 baseline_mean_curve_rmse=float(np.mean([v['baseline_rmse'] for v in per.values()])),
 new_mean_curve_rmse=float(np.mean([v['new_rmse'] for v in per.values()])),
 paired_rmse_reduction_95_interval=np.quantile(bootstrap,[.025,.975]).tolist(),per_condition=per,
 baseline_detection=detection(read('040-v1-release-calibration' if args.release else '029-baseline-calibration'),base),
 new_detection=detection(read('041-v2-release-calibration' if args.release else '030-winner-calibration'),new),
 scoring_scope='All generated frequencies off grid; true pair selected by predeclared nearest physical source anchor. offgrid is an independent 20 dB repeat, not a new distribution.',
 untested='No real EEG, clinical classification, or tokenizer utility validation in this operator-only run.')
summary['relative_rmse_reduction']=1-summary['new_mean_curve_rmse']/summary['baseline_mean_curve_rmse']
audit=root/'artifacts'/'independent-numpy-audit.json'
summary['independent_numpy_audit']=json.loads(audit.read_text())
summary['independent_numpy_audit'].pop('cases_checked')
(root/'artifacts'/('release-audit-summary.json' if args.release else 'final-audit-summary.json')).write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('per_condition',)},indent=2))
