import json,gzip,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paclock_bench.models.paclock.frontend.pac_calibration import calibration_rank
from scripts.experiments.cfc_operator_capabilities_report import rate_interval

root=Path(__file__).parent

def records(name):
 with gzip.open(root/name/'records.json.gz','rt') as f:return json.load(f)
a=[r for r in records('baseline-confirm') if r['variant']=='curve']
b=[r for r in records('uncertain-confirm') if r['variant']=='curve']
assert all(x['seed']==y['seed'] and x['raw_sha256']==y['raw_sha256'] for x,y in zip(a,b))
assert len(a)==len(b)==2688
rng=np.random.default_rng(20261029)
groups={};boots=[]
for group in sorted({r['group'] for r in a}):
 aa=np.array([r['curve_rmse'] for r in a if r['group']==group]);bb=np.array([r['curve_rmse'] for r in b if r['group']==group]);diff=aa-bb
 boots.append(diff[rng.integers(len(diff),size=(2000,len(diff)))].mean(1))
 groups[group]=dict(raw=float(aa.mean()),stable=float(bb.mean()),n=len(aa),paired_reduction=float(diff.mean()),
  paired_ci95=np.quantile(boots[-1],[.025,.975]).tolist())
macro_raw=float(np.mean([g['raw'] for g in groups.values()]));macro_stable=float(np.mean([g['stable'] for g in groups.values()]))
weak=[groups['depth/'+str(d)] for d in (0.,.02,.05,.1)]
cal=json.loads((root/'calibration-confirm-10s/summary.json').read_text())
for result in cal['detections'].values():result['ci95']=rate_interval(result['false_positives'],cal['n'])
carrier=json.loads((root/'carrier-confirm/result.json').read_text())
local=json.loads((root/'local-confirm-fresh/summary.json').read_text())['local']
summary=dict(scope='PAC measurement component; ready before tokenizer integration/training',
 default_point='raw harmonic regression; normalized raw and explicitly stabilized feature views both available',
 groups=groups,macro=dict(raw=macro_raw,stable=macro_stable,relative_reduction=1-macro_stable/macro_raw,
 paired_ci95=np.quantile(np.mean(boots,axis=0),[.025,.975]).tolist()),
 weak=dict(raw=float(np.mean([g['raw'] for g in weak])),stable=float(np.mean([g['stable'] for g in weak])),
 relative_reduction=float(1-np.mean([g['stable'] for g in weak])/np.mean([g['raw'] for g in weak]))),
 calibration=cal,stochastic_carrier=carrier['summary'],local_companion=local,
 cpu=json.loads((root/'engineering-cpu.json').read_text()),batch_cpu=json.loads((root/'batch-shape-cpu.json').read_text()),
 cuda=json.loads((root/'gpu-check.json').read_text()),
 counts=dict(recovery_confirmation=2688,frequency_context_confirmation=1440,stochastic_carrier_confirmation=768,
 null_calibration=2000,null_confirmation=4000,local_companion_confirmation=64,oracle_diagnostic=960),
 caveats=['No claim to beat established multiphasic PAC estimators.',
 'Weak PAC detection and finite-record physical support remain limited.',
 'Stabilization is a bias-variance tradeoff; stochastic mixed strong PAC is worse.',
 'Cycle covariance is a diagnostic, not a calibrated confidence interval.',
 'Null-score calibration applies only to the declared exchangeable reference distribution.',
 'Fixed6x5 measurement coordinates are not16 learned content-band indices.',
 'Global coefficients repeated over patches are not independent local measurements.'])
(root/'ready-summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(9,2.55),layout='constrained')
depths=[0.,.02,.05,.1,.25,.5]
for key,color,label in [('raw','#3b6d9a','Raw relation'),('stable','#c47743','Stabilized relation')]:
 values=[groups['depth/'+str(d)][key] for d in depths]
 axes[0].plot(depths,values,'o-',ms=3,lw=1.3,color=color,label=label)
axes[0].set(xlabel='Modulation depth',ylabel='Curve RMSE',ylim=(0,None),title='Coherent carrier, 10 dB')
axes[0].legend(frameon=False,fontsize=7)
labels=['Single','Double','Mixed'];xx=np.arange(3)
for i,(key,col) in enumerate([('raw','#3b6d9a'),('stable','#c47743')]):
 axes[1].bar(xx+(i-.5)*.26,[carrier['summary'][s+'/0.4'][key] for s in ['single','double','mixed']],width=.24,color=col)
axes[1].set(xticks=xx,xticklabels=labels,ylabel='Curve RMSE',ylim=(0,None),title='Stochastic carrier, depth 0.4')
values=[cal['detections'][k]['observed_fpr']*100 for k in ('marginal','tolerance95')]
interval=np.array([cal['detections'][k]['ci95'] for k in ('marginal','tolerance95')])*100
axes[2].bar([0,1],values,color=['#3b6d9a','#c47743'],width=.45)
axes[2].errorbar([0,1],values,yerr=[np.array(values)-interval[:,0],interval[:,1]-np.array(values)],fmt='none',ecolor='#333333',capsize=3,lw=.9)
axes[2].axhline(5,color='#555555',linestyle='--',linewidth=.9)
axes[2].set(xticks=[0,1],xticklabels=['Rank','Tolerance (95%)'],ylabel='False positives (%)',ylim=(0,6.5),title='4,000 independent null records')
for ax in axes:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
for ext in ('pdf','png'):fig.savefig(root/('ready-validation.'+ext),dpi=240)
plt.close(fig)
print(json.dumps({k:v for k,v in summary.items() if k in ('macro','weak','counts')},indent=2))
