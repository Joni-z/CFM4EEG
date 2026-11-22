"""Regenerate two-column result figures from their manuscript lookup tables.

Data and SD are reproduced as displayed, not recalculated. Missing entries
remain absent. Error bars are lines without point markers. All y axes are
explicitly truncated, as disclosed in the manuscript captions.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import pubfig as pf
ROOT=Path(__file__).resolve().parent
D=json.loads((ROOT/'cfft_twocol_data.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':8,
 'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
 'axes.linewidth':.55,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
CORAL='#F0545C'
C={'Measured':CORAL,'Own phase':'#C3C7CD','Weights only':'#9CA3AD',
   'Envelope only':'#737D8B','Identity':'#465363',
   'Native':'#737D8B','TFM-Tokenizer':'#465363','CfFT':CORAL}


def style(ax,lo,hi):
 ax.set_ylim(lo,hi)
 ax.spines[['top','right']].set_visible(False)
 for side in ['left','bottom']: ax.spines[side].set_color('#9DA5AE')
 ax.tick_params(axis='both',length=2.5,width=.55,pad=2)
 ax.grid(axis='y',lw=.45,color='#E4E7EB',zorder=0)
 ax.set_axisbelow(True)
 ax.yaxis.set_major_locator(MaxNLocator(nbins=3,min_n_ticks=3))
 ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

def save(fig,axs,name,h,adjust):
 axs.flat[0]._pubfig_post_layout=lambda:fig.subplots_adjust(**adjust)
 fig.subplots_adjust(**adjust)
 for ext in ['pdf','png']:
  pf.save_figure(fig,ROOT.parent/(name+'.'+ext),width=82.55,height_mm=h*25.4,
                 raster_dpi=220,trim=False)
 plt.close(fig)

# Dataset columns, metric rows: retain all six comparisons at column width.
fig,axs=plt.subplots(3,2,figsize=(3.25,3.6))
limits=[[(.45,.72),(.53,.76),(.73,.89)],[(.54,.73),(.46,.74),(.85,.96)]]
for c,ds in enumerate(['TUEV','CHB-MIT']):
 rows=D['controls'][ds]
 metrics=['Bal. acc.',r"Cohen's $\kappa$",'Weighted F1'] if ds=='TUEV' else ['Bal. acc.','AUC-PR','AUROC']
 for r,metric in enumerate(metrics):
  ax=axs[r,c]
  for i,row in enumerate(rows):
   y,sd=row['metrics'][r]
   ax.bar(i,y,width=.60,color=C[row['name']],zorder=3)
   ax.errorbar(i,y,yerr=sd,fmt='none',ecolor='#545454',elinewidth=.65,capsize=1.6,capthick=.65,zorder=4)
  style(ax,*limits[c][r]);ax.set_xlim(-.65,4.65);ax.set_xticks([])
  ax.set_title(metric,pad=3,fontweight='normal',fontsize=7)
 axs[0,c].text(.5,1.55,ds,transform=axs[0,c].transAxes,ha='center',va='bottom',fontsize=8)
labels=['Measured','Own phase','Weights only','Envelope only','Identity']
fig.legend([Patch(facecolor=C[x]) for x in labels],labels,
 loc='upper center',bbox_to_anchor=(.5,1.01),ncol=3,frameon=False,
 handlelength=.95,handletextpad=.35,columnspacing=.75,fontsize=6.5)
save(fig,axs,'table3_controls',3.6,dict(left=.12,right=.985,bottom=.025,top=.73,wspace=.42,hspace=.65))

# Dataset rows, host columns. Each panel groups metrics horizontally;
# keep the same score scale across hosts and datasets for direct reading.
fig,axs=plt.subplots(3,2,figsize=(3.25,3.6))
for r,ds in enumerate(['TUEV','IIIC','TUSZ']):
 for c,host in enumerate(['BIOT','CBraMod']):
  ax=axs[r,c]
  rows=[row for row in D['hosts'][ds] if row['host']==host]
  labels=['Bal. acc.',r'$\kappa$','W-F1'] if ds!='TUSZ' else ['Bal. acc.','PR','ROC']
  width=.18
  for j,row in enumerate(rows):
   for k,pair in enumerate(row['metrics']):
    x=k+(j-(len(rows)-1)/2)*width
    if pair is None:
     ax.text(x,.315,'–',ha='center',va='bottom',color='#666F7C',fontsize=7)
     continue
    y,sd=pair
    color='#C3C7CD' if row['name']=='Identity' else C[row['name']]
    ax.bar(x,y,width=width*.92,color=color,zorder=3)
    ax.errorbar(x,y,yerr=sd,fmt='none',ecolor='#545454',elinewidth=.55,capsize=1,capthick=.55,zorder=4)
  style(ax,.22,1.0);ax.set_yticks([.3,.6,.9]);ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
  ax.set_xlim(-.55,2.55)
  ax.set_xticks([0,1,2],labels);ax.tick_params(axis='x',length=0,pad=3,labelsize=6.5)
  if c==1: ax.tick_params(axis='y',labelleft=False)
  if r==0: ax.set_title(host,pad=5,fontweight='normal',fontsize=8)
 axs[r,0].set_ylabel(ds,labelpad=4,fontweight='normal',fontsize=7.5)
legend=['Native','Identity','TFM-Tokenizer','CfFT']
colors=['#737D8B','#C3C7CD','#465363',CORAL]
fig.legend([Patch(facecolor=x) for x in colors],legend,loc='upper center',
 bbox_to_anchor=(.5,1.005),ncol=2,frameon=False,handlelength=1,
 handletextpad=.4,columnspacing=1.2,fontsize=7)
save(fig,axs,'cfft_host_reuse_bars',3.6,dict(left=.15,right=.99,bottom=.07,top=.79,wspace=.16,hspace=.48))
print('Saved original-palette, single-column figures.')
