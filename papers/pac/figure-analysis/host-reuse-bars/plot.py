from pathlib import Path
import re,csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter,MaxNLocator
r=Path(__file__).parent
s=(r/'source.tex').read_text()
b=next(b for b in re.findall(r'\\begin\{table\}.*?\\end\{table\}',s,re.S) if '\\label{tab:host-reuse}' in b)
rows={};host=None
for line in b.splitlines():
 parts=line.split('&')
 if len(parts)!=11:continue
 h=parts[0].strip();arm=parts[1].strip()
 if h in ['BIOT','CBraMod']:host=h
 if arm not in ['Native','Identity','TFM-Tokenizer','CroFreMo']:continue
 values=[]
 for cell in parts[2:]:
  nums=re.findall(r'(?<![A-Za-z])\d+\.\d+',cell)
  assert len(nums) in [0,2],cell
  values.append(tuple(map(float,nums)) if nums else (np.nan,np.nan))
 rows[host,arm]=np.array(values)
assert len(rows)==6
colors={'Native':'#737D8B','Identity':'#C3C7CD','TFM-Tokenizer':'#465363','CroFreMo':'#F0545C'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.labelsize':7,'xtick.labelsize':6.3,'ytick.labelsize':6.5,'axes.linewidth':.45,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(3,3,figsize=(5.5,3.65))
fig.subplots_adjust(left=.085,right=.99,bottom=.075,top=.88,wspace=.40,hspace=.65)
metrics=[['Balanced accuracy','Cohen’s kappa','Weighted F1'],['Balanced accuracy','Cohen’s kappa','Weighted F1'],['Balanced accuracy','AUC-PR','AUROC']]
datasets=['TUEV','IIIC','TUSZ']
# Explicitly truncated ranges, all reported whiskers included.
lims=[(.35,.65),(.35,.68),(.65,.85),(.33,.52),(.22,.43),(.35,.53),(.45,.70),(.28,.59),(.77,.95)]
for k,ax in enumerate(axes.flat):
 dataset=datasets[k//3];metric=metrics[k//3][k%3];lo,hi=lims[k]
 biot_arms=[arm for arm in colors if np.isfinite(rows['BIOT',arm][k,0])]
 biot_xs=(np.arange(len(biot_arms))-(len(biot_arms)-1)/2)*.22
 for h,arms,xs in [('BIOT',biot_arms,biot_xs),('CBraMod',['Native','CroFreMo'],[.83,1.05])]:
  for arm,x in zip(arms,xs):
   mean,sd=rows[h,arm][k]
   if np.isnan(mean):continue
   assert mean-sd>=lo and mean+sd<=hi,(h,arm,k,mean,sd)
   ax.bar(x,mean-lo,bottom=lo,width=.18,color=colors[arm],edgecolor='none',zorder=2)
   ax.errorbar(x,mean,yerr=sd,fmt='none',color='#545454',lw=.55,capsize=1.4,capthick=.55,zorder=3)
 ax.set_ylim(lo,hi);ax.set_xlim(-.52,1.40)
 ax.set_yticks(np.linspace(lo,hi,4));ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
 ax.set_xticks([0,.94],['BIOT','CBraMod']);ax.tick_params(axis='x',length=0,pad=2)
 ax.set_ylabel(metric,labelpad=2);ax.set_title(dataset,fontsize=7,fontweight='normal',pad=3)
 ax.grid(axis='y',color='#D4D4D4',lw=.45,ls=':',zorder=0)
 ax.tick_params(axis='y',length=2,width=.45,pad=1)
 for sp in ax.spines.values():sp.set_edgecolor('#727272');sp.set_linewidth(.45)
legend=fig.legend([Patch(facecolor=c) for c in colors.values()],list(colors),ncol=4,loc='upper center',bbox_to_anchor=(.54,.985),frameon=True,fontsize=6.7,handlelength=1.7,handletextpad=.5,columnspacing=1.15,borderpad=.35)
legend.get_frame().set_edgecolor('#DADADA');legend.get_frame().set_linewidth(.45);legend.get_frame().set_facecolor('white')
for ext in ['pdf','png','svg']:fig.savefig(r/f'host_reuse_bars.{ext}',dpi=300,facecolor='white')
with (r/'plotted_data.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['host','frontend','dataset','metric','mean','reported_sd'])
 for (host,arm),vals in rows.items():
  for k,(m,d) in enumerate(vals):w.writerow([host,arm,datasets[k//3],metrics[k//3][k%3],m if np.isfinite(m) else '',d if np.isfinite(d) else ''])
(r/'provenance.json').write_text(json.dumps({'source':'Current Overleaf host-reuse table; exact snapshot in source.tex','note':'Table aggregates reproduced without raw-run verification; no seed counts inferred. Missing TFM/TUSZ bars omitted. All axes truncated and all whiskers within limits.','values_plotted':sum(np.isfinite(v[:,0]).sum().item() for v in rows.values())},indent=2))
print('51 means and SDs plotted from the source table; 3 missing cells retained as missing.')
